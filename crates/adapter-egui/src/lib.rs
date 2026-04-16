//! egui frontend for Ox.
//!
//! The GUI owns only display state. All session work — LLM streaming, tool
//! execution, session persistence — happens in a separate `ox-agent` process
//! spawned by `bin-gui` and talked to through an [`AgentClient`].
//!
//! Per-session state lives in [`AgentSplit`] so the GUI can host N agents
//! concurrently. The tiling UI renders all active sessions as equal-width
//! vertical splits; `/new` adds a split, `/close` closes one, and `/quit`
//! closes the app.

mod agent_client;
mod workspace_layout;

pub use agent_client::{AgentClient, AgentSpawnConfig};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use domain::{ContentBlock, Message, Role, SessionId};
use eframe::egui;
use egui_file_dialog::FileDialog;
use protocol::{AgentCommand, AgentEvent};
use tokio::sync::mpsc;

use app::StreamAccumulator;
use workspace_layout::{RestoreLayout, WorkspaceLayouts, normalize_split_fracs};

/// Shared, thread-safe mirror of the per-split session IDs.
///
/// `eframe::run_native` consumes the `OxApp`, so after the window closes the
/// app's state is gone. This mirror keeps the latest session ID shape
/// externally observable without exposing split internals.
///
/// A `Mutex<Vec<...>>` is overkill for single-threaded frame updates but it's
/// the smallest thread-safe shape that keeps the binary decoupled from the
/// egui runtime's threading model, which is a problem we don't want to audit
/// every time we touch this code.
pub type SessionIdMirror = Arc<Mutex<Vec<Option<SessionId>>>>;

/// Per-agent state. One split per running agent subprocess.
///
/// Holds the `AgentClient` handle, the committed-message history, and a
/// transient `StreamAccumulator` that mirrors the in-flight assistant turn
/// so we can render live tokens without waiting for the final commit.
///
/// Invariant: after `AgentEvent::Ready`, `waiting == false` and
/// `streaming == None`. Replayed-history `MessageAppended` frames append to
/// `messages` without flipping `waiting` or touching `streaming`, so they
/// look to the renderer identical to live-appended messages. This means a
/// user may type while history is still replaying; the agent's read loop
/// only dequeues the next command between `run_turn` calls, so the queued
/// `SendMessage` sits in the pipe until replay finishes. No GUI-side guard
/// is needed.
pub struct AgentSplit {
    client: AgentClient,
    messages: Vec<Message>,
    /// A live mirror of the in-flight assistant turn. Lazily created on the
    /// first `StreamDelta` so the "waiting for any output" placeholder UI
    /// branch stays distinguishable from "tokens are arriving."
    streaming: Option<StreamAccumulator>,
    waiting: bool,
    error: Option<String>,
    session_id: Option<SessionId>,
    /// Set when the most recent turn was cancelled by the user. Cleared
    /// on the next `SendMessage`. The renderer shows a red "Cancelled"
    /// label when this is true.
    cancelled: bool,
}

impl AgentSplit {
    pub fn new(client: AgentClient) -> Self {
        Self {
            client,
            messages: Vec::new(),
            streaming: None,
            waiting: false,
            error: None,
            session_id: None,
            cancelled: false,
        }
    }

    /// Session ID reported by the agent via `Ready`. `None` until the
    /// handshake completes.
    pub fn session_id(&self) -> Option<SessionId> {
        self.session_id
    }

    /// Drain all events the client has made available without blocking.
    fn poll_events(&mut self) {
        loop {
            match self.client.try_recv() {
                Ok(AgentEvent::Ready { session_id, .. }) => {
                    // Record the ID but do NOT touch `messages`, `waiting`,
                    // or `streaming`. History arrives as a sequence of
                    // `MessageAppended` frames handled below.
                    self.session_id = Some(session_id);
                }
                Ok(AgentEvent::StreamDelta { event }) => {
                    self.streaming
                        .get_or_insert_with(StreamAccumulator::new)
                        .push(event);
                }
                Ok(AgentEvent::MessageAppended { message }) => {
                    // Single code path for committed messages — used for
                    // live turn messages and for historical replay on
                    // resume. Drop the accumulator on each commit so the
                    // live-view branch stops rendering the message we just
                    // appended.
                    self.messages.push(message);
                    self.streaming = None;
                    self.error = None;
                }
                Ok(AgentEvent::TurnComplete) => {
                    self.waiting = false;
                    self.streaming = None;
                    self.error = None;
                }
                Ok(AgentEvent::TurnCancelled) => {
                    self.waiting = false;
                    // Commit any partial streaming content as a message
                    // before discarding the accumulator.
                    if let Some(acc) = self.streaming.take() {
                        let msg = acc.into_message();
                        if !msg.content.is_empty() {
                            self.messages.push(msg);
                        }
                    }
                    self.cancelled = true;
                    self.error = None;
                }
                Ok(AgentEvent::Error { message }) => {
                    self.streaming = None;
                    self.error = Some(message);
                    self.waiting = false;
                }
                // AgentEvent is `#[non_exhaustive]`; any future variant the
                // agent learns to emit we haven't taught the GUI about yet
                // should just be skipped, not crash.
                Ok(_other) => {}
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.waiting = false;
                    self.streaming = None;
                    self.error = Some("agent disconnected".into());
                    break;
                }
            }
        }
    }
}

/// Top-level GUI state. Owns a vector of agent splits displayed as resizable
/// vertical splits. `focused` indexes the split whose input bar receives
/// keystrokes. Each split has its own input string in `inputs` (parallel to
/// `splits`) so rendering can borrow `&mut inputs[i]` and `&splits[i]`
/// simultaneously — Rust can split-borrow distinct struct fields but not
/// `&mut split.input` and `&split.messages` on the same struct.
///
/// Split widths are stored as fractions (`split_fracs`) summing to 1.0.
/// Dragging a separator transfers width between adjacent splits. Using
/// fractions rather than absolute pixels means splits scale naturally
/// when the window is resized.
pub struct OxApp {
    splits: Vec<AgentSplit>,
    /// Index of the focused split. Keystrokes go to `inputs[focused]`.
    focused: usize,
    /// Per-split input strings, parallel to `splits`.
    inputs: Vec<String>,
    /// Fractional width of each split, summing to 1.0. Updated by
    /// dragging the separator between adjacent splits.
    split_fracs: Vec<f32>,
    /// Template config for spawning new agents via `/new`. Cloned with
    /// `resume: None` each time a new split is created.
    spawn_config: AgentSpawnConfig,
    /// GUI-owned workspace layout state. This remembers which session IDs
    /// belong to each workspace and how their split widths were arranged.
    layout_state_path: Option<PathBuf>,
    /// Slot the app writes session IDs into each frame for observers that
    /// need the latest split/session shape without inspecting split state.
    session_id_mirror: SessionIdMirror,
    /// When set, the next frame will request egui focus on this split's
    /// input TextEdit, then clear the flag.
    pending_focus: Option<usize>,
    file_dialog: FileDialog,
    pending_workspace: Option<PathBuf>,
    confirm_replace_workspace: bool,
    confirm_quit: bool,
    quit_confirmed: bool,
    about_open: bool,
    app_version: String,
}

impl OxApp {
    pub fn new(
        splits: Vec<AgentSplit>,
        spawn_config: AgentSpawnConfig,
        app_version: impl Into<String>,
    ) -> (Self, SessionIdMirror) {
        Self::with_layout(splits, spawn_config, app_version, None, None, 0)
    }

    fn with_layout(
        splits: Vec<AgentSplit>,
        spawn_config: AgentSpawnConfig,
        app_version: impl Into<String>,
        layout_state_path: Option<PathBuf>,
        split_fracs: Option<Vec<f32>>,
        focused: usize,
    ) -> (Self, SessionIdMirror) {
        assert!(!splits.is_empty(), "OxApp requires at least one split");
        let n = splits.len();
        let mirror: SessionIdMirror = Arc::new(Mutex::new(vec![None; n]));
        let inputs = vec![String::new(); n];
        let split_fracs = split_fracs
            .map(|fracs| normalize_split_fracs(&fracs, n))
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        let focused = focused.min(n - 1);
        let app = Self {
            splits,
            focused,
            inputs,
            split_fracs,
            spawn_config,
            layout_state_path,
            session_id_mirror: mirror.clone(),
            pending_focus: Some(focused),
            file_dialog: FileDialog::new(),
            pending_workspace: None,
            confirm_replace_workspace: false,
            confirm_quit: false,
            quit_confirmed: false,
            about_open: false,
            app_version: app_version.into(),
        };
        (app, mirror)
    }

    pub fn restore(
        mut spawn_config: AgentSpawnConfig,
        layout_state_path: PathBuf,
        app_version: impl Into<String>,
    ) -> Result<(Self, SessionIdMirror)> {
        let layout = if spawn_config.resume.is_some() {
            None
        } else {
            load_workspace_layout(&layout_state_path)
                .restore_existing_for(&spawn_config.workspace_root, &spawn_config.sessions_dir)
        };
        let explicit_resume = spawn_config.resume;
        let restore = layout.as_ref();
        let spawn_configs = startup_spawn_configs(&spawn_config, restore);

        let mut splits = Vec::with_capacity(spawn_configs.len());
        let mut restored_spawn_failed = false;
        for config in &spawn_configs {
            match AgentClient::spawn(config.clone()) {
                Ok(client) => splits.push(AgentSplit::new(client)),
                Err(_err) if restore.is_some() && explicit_resume.is_none() => {
                    restored_spawn_failed = true;
                    break;
                }
                Err(err) => return Err(err).context("spawning initial agent"),
            }
        }

        let (split_fracs, focused) = if restored_spawn_failed {
            let mut fresh = spawn_config.clone();
            fresh.resume = None;
            let client = AgentClient::spawn(fresh)
                .context("spawning fresh agent after restored workspace layout failed")?;
            splits = vec![AgentSplit::new(client)];
            (None, 0)
        } else {
            (
                restore.map(|layout| layout.split_fracs.clone()),
                restore.map(|layout| layout.focused).unwrap_or(0),
            )
        };

        spawn_config.resume = None;
        Ok(Self::with_layout(
            splits,
            spawn_config,
            app_version,
            Some(layout_state_path),
            split_fracs,
            focused,
        ))
    }

    pub fn run(self) -> Result<()> {
        let options = eframe::NativeOptions::default();
        eframe::run_native("Ox", options, Box::new(|_cc| Ok(Box::new(self))))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn publish_session_ids(&self) {
        let ids: Vec<Option<SessionId>> = self.splits.iter().map(|s| s.session_id()).collect();
        if let Ok(mut slot) = self.session_id_mirror.lock() {
            *slot = ids;
        }
    }

    fn save_current_workspace_layout(&self) -> Result<bool> {
        let Some(path) = &self.layout_state_path else {
            return Ok(false);
        };
        let mut layouts = load_workspace_layout(path);
        let saved = layouts.save_current(
            &self.spawn_config.workspace_root,
            self.splits.iter().map(|split| split.session_id()),
            &self.split_fracs,
            self.focused,
        );
        if saved {
            layouts.save(path)?;
        }
        Ok(saved)
    }

    fn any_turn_in_progress(&self) -> bool {
        self.splits
            .iter()
            .any(|split| split.waiting || split.streaming.is_some())
    }

    fn request_quit(&mut self, ctx: &egui::Context) {
        if self.any_turn_in_progress() {
            self.confirm_quit = true;
        } else {
            self.quit_confirmed = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn should_cancel_viewport_close_request(&mut self) -> bool {
        if self.quit_confirmed || !self.any_turn_in_progress() {
            return false;
        }
        self.confirm_quit = true;
        true
    }

    /// Spawn a new agent and append a split to the right. On failure,
    /// the error is shown on the currently focused split.
    fn handle_new(&mut self, _ctx: &egui::Context) {
        let mut config = self.spawn_config.clone();
        config.resume = None;
        match AgentClient::spawn(config) {
            Ok(client) => {
                let split = AgentSplit::new(client);
                self.splits.push(split);
                self.inputs.push(String::new());
                self.focused = self.splits.len() - 1;
                self.pending_focus = Some(self.focused);
                // Redistribute widths equally.
                let n = self.splits.len();
                self.split_fracs = vec![1.0 / n as f32; n];
            }
            Err(e) => {
                self.splits[self.focused].error = Some(format!("failed to spawn agent: {e:#}"));
            }
        }
    }

    /// Close the split at `split_idx`. If it's the last split, follow the
    /// same app-quit path as File > Quit and `/quit`.
    fn handle_close_split(&mut self, split_idx: usize, ctx: &egui::Context) {
        if self.splits.len() == 1 {
            self.request_quit(ctx);
            return;
        }
        self.remove_split_state(split_idx);
        self.pending_focus = Some(self.focused);
    }

    /// Add a pre-built split. Used by tests that can't spawn a real agent
    /// but need to exercise split add/remove logic.
    #[cfg(test)]
    fn add_split(&mut self, split: AgentSplit) {
        self.splits.push(split);
        self.inputs.push(String::new());
        self.focused = self.splits.len() - 1;
        let n = self.splits.len();
        self.split_fracs = vec![1.0 / n as f32; n];
    }

    /// Remove the split at `split_idx` without viewport commands. Used by
    /// tests that don't have an egui context.
    #[cfg(test)]
    fn remove_split(&mut self, split_idx: usize) {
        self.remove_split_state(split_idx);
    }

    fn remove_split_state(&mut self, split_idx: usize) {
        let reclaimed = self.split_fracs.remove(split_idx);
        let neighbor = if split_idx < self.split_fracs.len() {
            split_idx
        } else {
            split_idx - 1
        };
        self.split_fracs[neighbor] += reclaimed;

        self.splits.remove(split_idx);
        self.inputs.remove(split_idx);
        adjust_focus_after_remove(&mut self.focused, split_idx, self.splits.len());
    }

    /// Send the input for the given split as a user message to that split's
    /// agent.
    ///
    /// We DON'T optimistically push the user message into `messages`: the
    /// agent round-trips it back as a `MessageAppended` event, and having a
    /// single render path for every message (user input, tool results,
    /// assistant replies) keeps the code paths small.
    fn send_message(&mut self, split_idx: usize) {
        let text = self.inputs[split_idx].trim().to_owned();
        if text.is_empty() {
            return;
        }
        self.inputs[split_idx].clear();

        let split = &mut self.splits[split_idx];
        if split.waiting {
            return;
        }

        // Best-effort send. If the writer task is gone (agent dead), the
        // next `poll_events` will surface the disconnect as an Error.
        let _ = split.client.send(AgentCommand::SendMessage { input: text });
        split.waiting = true;
        split.error = None;
        split.cancelled = false;
    }

    fn replace_workspace(&mut self, new_root: PathBuf) {
        self.replace_workspace_with(new_root, |_, config| {
            AgentClient::spawn(config.clone()).map(AgentSplit::new)
        });
    }

    fn replace_workspace_with(
        &mut self,
        new_root: PathBuf,
        mut factory: impl FnMut(usize, &AgentSpawnConfig) -> Result<AgentSplit>,
    ) {
        if let Err(e) = self.save_current_workspace_layout() {
            self.splits[self.focused].error =
                Some(format!("failed to save workspace layout: {e:#}"));
        }

        let mut next_config = self.spawn_config.clone();
        next_config.workspace_root = new_root.clone();
        next_config.resume = None;
        let layouts = self
            .layout_state_path
            .as_deref()
            .map(load_workspace_layout)
            .unwrap_or_default();
        let restore = layouts.restore_existing_for(&new_root, &next_config.sessions_dir);
        let spawn_configs = restore_spawn_configs(&next_config, restore.as_ref());
        let n = spawn_configs.len();

        let mut next_splits = Vec::with_capacity(n);
        for (idx, config) in spawn_configs.iter().enumerate() {
            match factory(idx, config) {
                Ok(split) => next_splits.push(split),
                Err(e) => {
                    self.splits[self.focused].error = Some(format!("failed to spawn agent: {e:#}"));
                    self.confirm_replace_workspace = false;
                    self.pending_workspace = None;
                    return;
                }
            }
        }

        self.spawn_config = next_config;
        self.splits = next_splits;
        self.inputs = vec![String::new(); n];
        self.split_fracs = restore
            .as_ref()
            .map(|layout| normalize_split_fracs(&layout.split_fracs, n))
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        self.focused = restore
            .as_ref()
            .map(|layout| layout.focused.min(n - 1))
            .unwrap_or(0);
        self.pending_focus = Some(self.focused);
        if let Ok(mut slot) = self.session_id_mirror.lock() {
            *slot = vec![None; n];
        }
        self.confirm_replace_workspace = false;
        self.pending_workspace = None;
    }
}

impl eframe::App for OxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for split in &mut self.splits {
            split.poll_events();
        }

        if ctx.input(|input| input.viewport().close_requested())
            && self.should_cancel_viewport_close_request()
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }

        // Publish the latest session IDs for observers that need the
        // current split/session shape without borrowing the splits.
        self.publish_session_ids();

        let mut menu_actions: Vec<MenuAction> = Vec::new();
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open...").clicked() {
                        menu_actions.push(MenuAction::OpenWorkspacePicker);
                        ui.close_menu();
                    }
                    if ui.button("Quit").clicked() {
                        menu_actions.push(MenuAction::QuitRequested);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        menu_actions.push(MenuAction::OpenAbout);
                        ui.close_menu();
                    }
                });
            });
        });

        self.file_dialog.update(ctx);
        if let Some(path) = self.file_dialog.take_picked() {
            menu_actions.push(MenuAction::ConfirmReplaceWorkspace(path));
        }

        if self.confirm_replace_workspace {
            egui::Modal::new(egui::Id::new("confirm_replace_workspace")).show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.heading("Open Workspace");
                    ui.label("A turn is in progress.");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            menu_actions.push(MenuAction::CancelReplaceWorkspace);
                        }
                        if ui.button("Open").clicked() {
                            menu_actions.push(MenuAction::ReplaceWorkspaceConfirmed);
                        }
                    });
                });
            });
        }

        if self.confirm_quit {
            egui::Modal::new(egui::Id::new("confirm_quit")).show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.heading("Quit Ox");
                    ui.label("A turn is in progress. Are you sure you want to quit?");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            menu_actions.push(MenuAction::CancelQuit);
                        }
                        if ui.button("Quit").clicked() {
                            menu_actions.push(MenuAction::ConfirmQuit);
                        }
                    });
                });
            });
        }

        if self.about_open {
            egui::Modal::new(egui::Id::new("about_ox")).show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.heading("About");
                    ui.label(about_text(&self.app_version));
                    ui.add_space(8.0);
                    if ui.button("Close").clicked() {
                        menu_actions.push(MenuAction::CloseAbout);
                    }
                });
            });
        }

        // Tiling layout: a single CentralPanel divided into N vertical
        // splits with draggable separators between them. Split widths are
        // stored as fractions of the available width (`split_fracs`),
        // updated by dragging a separator.
        //
        // Borrow-checker constraint: we can't call `&mut self` methods
        // while borrowing splits/inputs, so we collect deferred "actions"
        // and execute them after all splits are rendered.
        let mut actions: Vec<SplitAction> = Vec::new();
        let mut new_focus: Option<usize> = None;
        let any_waiting = self.splits.iter().any(|t| t.waiting);
        let n = self.splits.len();

        let panel_frame = egui::Frame::central_panel(ctx.style().as_ref()).inner_margin(0.0);
        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                let total_rect = ui.available_rect_before_wrap();
                let sep_width = 6.0;
                let total_sep = sep_width * (n.saturating_sub(1)) as f32;
                let content_width = total_rect.width() - total_sep;
                // Minimum fraction a split can shrink to (60px equivalent).
                let min_frac = (60.0 / content_width).min(1.0 / n as f32);

                let mut x = total_rect.left();

                for i in 0..n {
                    // -- Split content area --
                    let w = content_width * self.split_fracs[i];
                    let split_rect = egui::Rect::from_min_max(
                        egui::pos2(x, total_rect.top()),
                        egui::pos2(x + w, total_rect.bottom()),
                    );

                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(split_rect));
                    let split = &self.splits[i];
                    let input = &mut self.inputs[i];
                    let grab_focus = self.pending_focus == Some(i);
                    render_split(
                        &mut child,
                        split,
                        input,
                        i,
                        grab_focus,
                        &mut actions,
                        &mut new_focus,
                    );

                    x += w;

                    // -- Draggable separator --
                    if i < n - 1 {
                        let sep_rect = egui::Rect::from_min_max(
                            egui::pos2(x, total_rect.top()),
                            egui::pos2(x + sep_width, total_rect.bottom()),
                        );
                        let sep_id = egui::Id::new("split_sep").with(i);
                        let sep_response = ui.interact(sep_rect, sep_id, egui::Sense::drag());

                        if sep_response.dragged() {
                            let delta = sep_response.drag_delta().x / content_width;
                            let left = (self.split_fracs[i] + delta).max(min_frac);
                            let right = (self.split_fracs[i + 1] - delta).max(min_frac);
                            self.split_fracs[i] = left;
                            self.split_fracs[i + 1] = right;
                        }

                        // Visual: thin line, highlighted on hover/drag.
                        let color = if sep_response.hovered() || sep_response.dragged() {
                            ui.visuals().widgets.active.bg_fill
                        } else {
                            ui.visuals().widgets.noninteractive.bg_stroke.color
                        };
                        ui.painter().rect_filled(
                            sep_rect.shrink2(egui::vec2(2.0, 0.0)),
                            0.0,
                            color,
                        );

                        if sep_response.hovered() || sep_response.dragged() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                        }

                        x += sep_width;
                    }
                }
            });

        self.pending_focus = None;

        // Apply focus changes detected inside the render closure.
        if let Some(idx) = new_focus {
            self.focused = idx;
        }

        for action in menu_actions {
            match action {
                MenuAction::OpenWorkspacePicker => self.file_dialog.pick_directory(),
                MenuAction::QuitRequested => self.request_quit(ctx),
                MenuAction::OpenAbout => self.about_open = true,
                MenuAction::ConfirmReplaceWorkspace(path) => {
                    if self.any_turn_in_progress() {
                        self.pending_workspace = Some(path);
                        self.confirm_replace_workspace = true;
                    } else {
                        self.replace_workspace(path);
                    }
                }
                MenuAction::CancelReplaceWorkspace => {
                    self.pending_workspace = None;
                    self.confirm_replace_workspace = false;
                }
                MenuAction::ReplaceWorkspaceConfirmed => {
                    if let Some(path) = self.pending_workspace.clone() {
                        self.replace_workspace(path);
                    } else {
                        self.confirm_replace_workspace = false;
                    }
                }
                MenuAction::ConfirmQuit => {
                    self.quit_confirmed = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                MenuAction::CancelQuit => self.confirm_quit = false,
                MenuAction::CloseAbout => self.about_open = false,
            }
        }

        // Execute deferred actions outside the borrow of `self.splits` /
        // `self.inputs`.
        for action in actions {
            match action {
                SplitAction::Send(idx) => self.send_message(idx),
                SplitAction::New => self.handle_new(ctx),
                SplitAction::QuitApp => self.request_quit(ctx),
                SplitAction::CloseSplit(idx) => self.handle_close_split(idx, ctx),
                SplitAction::Cancel(idx) => {
                    let _ = self.splits[idx].client.send(AgentCommand::Cancel);
                }
            }
        }

        // Keep polling while any split is waiting so we pick up streaming
        // tokens and the final response promptly.
        if any_waiting {
            ctx.request_repaint();
        }
    }
}

impl Drop for OxApp {
    fn drop(&mut self) {
        if let Err(e) = self.save_current_workspace_layout() {
            eprintln!("failed to save workspace layout: {e:#}");
        }
    }
}

fn load_workspace_layout(path: &Path) -> WorkspaceLayouts {
    WorkspaceLayouts::load(path).unwrap_or_else(|e| {
        eprintln!("ignoring workspace layout file {}: {e:#}", path.display());
        WorkspaceLayouts::default()
    })
}

fn restore_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    match restore {
        Some(layout) => layout
            .sessions
            .iter()
            .map(|id| {
                let mut config = base_config.clone();
                config.resume = Some(*id);
                config
            })
            .collect(),
        None => {
            let mut config = base_config.clone();
            config.resume = None;
            vec![config]
        }
    }
}

fn startup_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    if base_config.resume.is_some() {
        vec![base_config.clone()]
    } else {
        restore_spawn_configs(base_config, restore)
    }
}

/// Adjust `focused` after removing the split at `removed_idx`.
///
/// If the removed split was before the focused one, decrement to keep
/// pointing at the same logical split. If the focused split itself was
/// removed, clamp to the last valid index (left neighbor or 0).
fn adjust_focus_after_remove(focused: &mut usize, removed_idx: usize, new_len: usize) {
    if removed_idx < *focused {
        *focused -= 1;
    } else if *focused >= new_len {
        *focused = new_len.saturating_sub(1);
    }
}

/// Classify user input as a command (`/new`, `/quit`, `/close`) or a regular
/// message.
///
/// Extracted from the render closure so it can be unit-tested without an
/// egui context.
fn classify_input(text: &str, split_idx: usize) -> SplitAction {
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("/new") {
        SplitAction::New
    } else if trimmed.eq_ignore_ascii_case("/quit") {
        SplitAction::QuitApp
    } else if trimmed.eq_ignore_ascii_case("/close") {
        SplitAction::CloseSplit(split_idx)
    } else {
        SplitAction::Send(split_idx)
    }
}

fn about_text(version: &str) -> String {
    format!("Ox v{version}")
}

/// Actions that a split's render code can request. Collected inside the
/// `columns()` closure (where we can't call `&mut self` methods) and
/// executed afterward.
#[derive(Debug)]
enum SplitAction {
    /// Send the current input as a user message to the agent at this index.
    Send(usize),
    /// Spawn a new agent and append a split to the right.
    New,
    /// Quit the whole app.
    QuitApp,
    /// Close the split at this index. Drops the agent (`kill_on_drop`).
    CloseSplit(usize),
    /// Cancel the in-progress turn on this split's agent.
    Cancel(usize),
}

#[derive(Debug)]
enum MenuAction {
    OpenWorkspacePicker,
    QuitRequested,
    OpenAbout,
    ConfirmReplaceWorkspace(PathBuf),
    CancelReplaceWorkspace,
    ReplaceWorkspaceConfirmed,
    ConfirmQuit,
    CancelQuit,
    CloseAbout,
}

/// Render one vertical split: scroll area with message history, streaming
/// view, error display, and an input bar pinned at the bottom.
///
/// Any actions the user triggers (send, `/new`, `/quit`, `/close`) are pushed
/// into `actions` for deferred execution. Focus changes are recorded in
/// `new_focus`.
fn render_split(
    ui: &mut egui::Ui,
    split: &AgentSplit,
    input: &mut String,
    split_idx: usize,
    grab_focus: bool,
    actions: &mut Vec<SplitAction>,
    new_focus: &mut Option<usize>,
) {
    // The split fills its column top to bottom. The input bar is pinned at
    // the bottom using `bottom_up` layout; the scroll area fills the rest.
    // We use `with_layout` to render the input bar first (so it claims its
    // space), then render the scroll area in the remaining space above.
    // Reserve a stable area at the bottom for the input bar. We render the
    // scroll area first (top-down), then the input bar in the remaining
    // space at the bottom, by using nested allocations.
    let padding = 8.0;
    let total_rect = ui.available_rect_before_wrap().shrink(padding);

    let input_height = ui.spacing().interact_size.y;
    let scroll_rect = egui::Rect::from_min_max(
        total_rect.min,
        egui::pos2(total_rect.max.x, total_rect.max.y - input_height),
    );
    let input_rect = egui::Rect::from_min_max(
        egui::pos2(total_rect.min.x, total_rect.max.y - input_height),
        total_rect.max,
    );

    // Build a tool-result index so tool-call blocks can inline their results.
    // Tool results arrive as separate Role::Tool messages containing ToolResult
    // blocks. We collect them here and skip those messages in the render loop.
    let mut tool_results: HashMap<&str, (&str, bool)> = HashMap::new();
    for msg in &split.messages {
        if msg.role == Role::Tool {
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                } = block
                {
                    tool_results.insert(tool_call_id.as_str(), (content.as_str(), *is_error));
                }
            }
        }
    }

    // -- Scroll area with messages --
    let mut scroll_ui = ui.new_child(egui::UiBuilder::new().max_rect(scroll_rect));
    egui::ScrollArea::vertical()
        .id_salt(egui::Id::new("split_scroll").with(split_idx))
        .auto_shrink(false)
        .stick_to_bottom(true)
        .show(&mut scroll_ui, |ui| {
            for msg in &split.messages {
                // Tool-role messages are consumed via the tool-result index
                // and rendered inline under their paired tool-call block.
                if msg.role == Role::Tool {
                    continue;
                }
                render_blocks(ui, &msg.content, &msg.role, &tool_results);
                ui.add_space(8.0);
            }

            // Live view of the in-flight turn. No tool results exist yet
            // during streaming — they arrive after the turn completes.
            if let Some(acc) = &split.streaming {
                let snapshot = acc.snapshot();
                let empty = HashMap::new();
                render_blocks(ui, snapshot.content, &Role::Assistant, &empty);
                ui.add_space(8.0);
            } else if split.waiting {
                ui.label("...");
            }

            if split.cancelled {
                ui.colored_label(egui::Color32::RED, "Cancelled");
            }

            if let Some(err) = &split.error {
                ui.colored_label(egui::Color32::RED, err);
            }
        });

    // -- Input bar --
    let mut input_ui = ui.new_child(egui::UiBuilder::new().max_rect(input_rect));
    let input_response =
        input_ui.add(egui::TextEdit::singleline(input).desired_width(f32::INFINITY));

    if grab_focus {
        input_response.request_focus();
    }

    // Track focus: clicking a split's input bar makes it focused.
    if input_response.gained_focus() {
        *new_focus = Some(split_idx);
    }

    let enter_pressed =
        input_response.lost_focus() && input_ui.input(|i| i.key_pressed(egui::Key::Enter));

    if enter_pressed {
        let action = classify_input(input, split_idx);
        // Commands are consumed here; regular messages are consumed
        // by `send_message` after the closure returns.
        if matches!(
            action,
            SplitAction::New | SplitAction::QuitApp | SplitAction::CloseSplit(_)
        ) {
            input.clear();
        }
        actions.push(action);
        input_response.request_focus();
    }

    // Escape cancels the in-progress turn. Checked globally (not just on
    // the input widget) so it works even if the input doesn't have focus.
    if split.waiting && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        actions.push(SplitAction::Cancel(split_idx));
    }
}

/// Render a sequence of `ContentBlock`s with role-aware styling.
///
/// One renderer for both history and live streaming: each call takes a
/// borrowed slice so it doesn't care whether the blocks came from a
/// persisted `Message` or a fresh `StreamAccumulator::snapshot`. That
/// symmetry is the whole point — the live view and the final view can't
/// drift apart because they run through the same function.
///
/// - **User text**: cornflower blue.
/// - **Assistant text**: white.
/// - **Thinking**: gray, rendered inline (no collapsing).
/// - **Tool calls**: clickable toggle header with white triangle + gray
///   monospace `name(args)`. Body shows the paired tool result indented and
///   in gray monospace. Open/closed state stored in egui's per-frame
///   `Memory` keyed by the tool call ID.
/// - **Tool results**: not rendered standalone — consumed via the tool call
///   index and shown inside their paired tool call's body.
fn render_blocks(
    ui: &mut egui::Ui,
    blocks: &[ContentBlock],
    role: &Role,
    tool_results: &HashMap<&str, (&str, bool)>,
) {
    let mut rendered_any = false;
    for block in blocks {
        // Spacing between consecutive visible blocks — same 8px used
        // between messages so the rhythm is consistent throughout.
        let needs_space = match block {
            ContentBlock::Reasoning { content, .. } => !content.is_empty(),
            ContentBlock::ToolResult { .. } => false,
            _ => true,
        };
        if needs_space && rendered_any {
            ui.add_space(8.0);
        }

        match block {
            ContentBlock::Text { text } => {
                let trimmed = text.trim_end_matches(['\n', ' ']);
                if trimmed.is_empty() {
                    continue;
                }
                let color = match role {
                    Role::User => egui::Color32::from_rgb(100, 149, 237),
                    _ => egui::Color32::WHITE,
                };
                ui.label(egui::RichText::new(trimmed).color(color));
                rendered_any = true;
            }
            ContentBlock::Reasoning { content, .. } => {
                if !content.is_empty() {
                    ui.label(egui::RichText::new(content).color(egui::Color32::WHITE));
                    rendered_any = true;
                }
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                // Manual toggle: clickable header with a triangle prefix.
                // State stored in egui's temp data keyed by the tool call ID,
                // defaulting to open when a result exists, closed otherwise.
                let toggle_id = egui::Id::new("tool_toggle").with(id);
                let default_open = tool_results.contains_key(id.as_str());
                let mut open = ui
                    .ctx()
                    .data_mut(|d| *d.get_temp_mut_or(toggle_id, default_open));

                let tool_color = egui::Color32::from_rgb(160, 160, 160);
                let header_text = format!("{name}({arguments})");
                let item_spacing = 4.0_f32;
                let font_height = ui.text_style_height(&egui::TextStyle::Body);
                let tri_size = font_height * 0.5;
                // Total left offset from the triangle widget + gap so
                // the output text aligns with the tool name.
                let output_indent = tri_size + item_spacing;

                let clicked = ui
                    .horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = item_spacing;
                        let (tri_rect, tri_response) = ui.allocate_exact_size(
                            egui::vec2(tri_size, font_height),
                            egui::Sense::click(),
                        );
                        let center = tri_rect.center();
                        let half = tri_size * 0.5;
                        let points = if open {
                            // Down-pointing triangle
                            vec![
                                egui::pos2(center.x - half, center.y - half * 0.5),
                                egui::pos2(center.x + half, center.y - half * 0.5),
                                egui::pos2(center.x, center.y + half * 0.5),
                            ]
                        } else {
                            // Right-pointing triangle
                            vec![
                                egui::pos2(center.x - half * 0.5, center.y - half),
                                egui::pos2(center.x + half * 0.5, center.y),
                                egui::pos2(center.x - half * 0.5, center.y + half),
                            ]
                        };
                        ui.painter().add(egui::Shape::convex_polygon(
                            points,
                            egui::Color32::WHITE,
                            egui::Stroke::NONE,
                        ));
                        let r2 = ui.add(
                            egui::Label::new(
                                egui::RichText::new(&header_text)
                                    .color(tool_color)
                                    .monospace(),
                            )
                            .sense(egui::Sense::click()),
                        );
                        tri_response.clicked() || r2.clicked()
                    })
                    .inner;
                if clicked {
                    open = !open;
                    ui.ctx().data_mut(|d| d.insert_temp(toggle_id, open));
                }

                if open && let Some(&(content, is_error)) = tool_results.get(id.as_str()) {
                    let output_color = if is_error {
                        egui::Color32::RED
                    } else {
                        egui::Color32::from_rgb(140, 140, 140)
                    };
                    let trimmed = content.trim_end_matches(['\n', ' ']);
                    ui.horizontal(|ui| {
                        ui.add_space(output_indent);
                        ui.label(egui::RichText::new(trimmed).color(output_color).monospace());
                    });
                }
                rendered_any = true;
            }
            // Tool results are rendered inline under their paired tool-call
            // block via the tool_results index. Nothing to do here.
            ContentBlock::ToolResult { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! `AgentSplit` state-machine tests.
    //!
    //! `OxApp::update` is driven by egui and exercised by running the app;
    //! we don't try to simulate egui here. Instead we pin down the invariants
    //! the plan calls out for `AgentSplit`:
    //!
    //! - `Ready` records `session_id` and *only* that — `messages`, `waiting`,
    //!   `streaming`, `error` stay untouched.
    //! - `MessageAppended` is the single entry point for both historical
    //!   replay and live turn messages; it appends unconditionally.
    //! - `TurnComplete` clears `waiting` and `streaming` and `error`.
    //! - `Error` clears `waiting` and `streaming` and records the message.
    //!
    //! Each test drives a real `AgentSplit` through its `poll_events` method
    //! by feeding frames into an in-memory duplex pipe.

    use std::path::PathBuf;
    use std::time::Duration;

    use domain::{Message, StreamEvent, Usage};
    use protocol::{AgentEvent, write_frame};
    use tokio::io::{BufReader, duplex};

    use super::*;

    /// Build a split whose `AgentClient` reads from `agent_writer`'s peer.
    /// The caller writes `AgentEvent` frames into `agent_writer` and then
    /// awaits `recv_until`, which pumps `split.poll_events()` until a given
    /// condition holds.
    fn make_split() -> (AgentSplit, tokio::io::DuplexStream) {
        let (agent_writer, client_reader) = duplex(4096);
        // Throwaway command pipe — these tests don't inspect commands.
        let (client_writer, _agent_reader) = duplex(4096);
        std::mem::forget(_agent_reader); // keep the pipe's read-end alive
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);
        (AgentSplit::new(client), agent_writer)
    }

    /// Pump `split.poll_events()` until `check` returns true or the timeout
    /// elapses. Returns Ok iff `check` ended true. Uses short sleeps (5ms)
    /// to let the reader task pick up the frames we wrote.
    async fn wait_until(
        split: &mut AgentSplit,
        timeout: Duration,
        mut check: impl FnMut(&AgentSplit) -> bool,
    ) -> Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            split.poll_events();
            if check(split) {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!("wait_until timed out");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn ready_records_session_id_and_leaves_other_fields_untouched() {
        // Pin down the documented invariant: after `Ready`, `waiting == false`
        // and `streaming == None`, and `messages` is unaffected.
        let (mut split, mut writer) = make_split();
        let id = SessionId::new_v4();
        write_frame(
            &mut writer,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: PathBuf::from("/w"),
            },
        )
        .await
        .unwrap();

        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.session_id() == Some(id)
        })
        .await
        .unwrap();

        assert!(split.messages.is_empty(), "Ready must not touch messages");
        assert!(!split.waiting, "Ready must not flip waiting");
        assert!(split.streaming.is_none(), "Ready must not touch streaming");
        assert!(split.error.is_none(), "Ready must not set an error");
    }

    #[tokio::test]
    async fn message_appended_extends_history_without_touching_streaming() {
        let (mut split, mut writer) = make_split();
        write_frame(
            &mut writer,
            &AgentEvent::MessageAppended {
                message: Message::user("hi"),
            },
        )
        .await
        .unwrap();

        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.messages.len() == 1
        })
        .await
        .unwrap();
        assert_eq!(split.messages[0].text(), "hi");
        assert!(!split.waiting, "history replay should not set waiting");
        assert!(split.streaming.is_none());
    }

    #[tokio::test]
    async fn message_appended_drops_any_inflight_stream_accumulator() {
        // If a StreamDelta has already arrived, the split is building an
        // accumulator mirror. When the committed message lands, the
        // accumulator must be dropped so the renderer doesn't render the
        // same tokens twice.
        let (mut split, mut writer) = make_split();
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.streaming.is_some()
        })
        .await
        .unwrap();

        write_frame(
            &mut writer,
            &AgentEvent::MessageAppended {
                message: Message::user("final"),
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.messages.len() == 1
        })
        .await
        .unwrap();
        assert!(
            split.streaming.is_none(),
            "streaming mirror must be dropped on commit"
        );
    }

    #[tokio::test]
    async fn turn_complete_clears_waiting_and_streaming_and_error() {
        let (mut split, mut writer) = make_split();
        // Seed state the terminator has to clear.
        split.waiting = true;
        split.error = Some("old".into());
        split.streaming = Some(app::StreamAccumulator::new());

        write_frame(&mut writer, &AgentEvent::TurnComplete)
            .await
            .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| !t.waiting)
            .await
            .unwrap();
        assert!(split.streaming.is_none());
        assert!(split.error.is_none());
    }

    #[tokio::test]
    async fn error_frame_records_message_and_clears_streaming() {
        let (mut split, mut writer) = make_split();
        split.waiting = true;
        split.streaming = Some(app::StreamAccumulator::new());

        write_frame(
            &mut writer,
            &AgentEvent::Error {
                message: "model overloaded".into(),
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| t.error.is_some())
            .await
            .unwrap();
        assert_eq!(split.error.as_deref(), Some("model overloaded"));
        assert!(!split.waiting);
        assert!(split.streaming.is_none());
    }

    #[tokio::test]
    async fn stream_delta_seeds_accumulator_on_first_event() {
        let (mut split, mut writer) = make_split();
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "hello".into(),
                },
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.streaming.is_some()
        })
        .await
        .unwrap();

        // A subsequent delta should accumulate into the same snapshot, and
        // a `Finished` event with usage should produce a full snapshot with
        // the expected token count — we stop short of asserting that the
        // accumulator's internals are correct (that's StreamAccumulator's
        // job, not this file's).
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: " world".into(),
                },
            },
        )
        .await
        .unwrap();
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::Finished {
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 2,
                        reasoning_tokens: 0,
                    },
                },
            },
        )
        .await
        .unwrap();

        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.streaming
                .as_ref()
                .map(|a| a.snapshot().token_count == 2)
                .unwrap_or(false)
        })
        .await
        .unwrap();
        let snap = split.streaming.as_ref().unwrap().snapshot();
        match &snap.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello world"),
            _ => panic!("expected Text block"),
        }
    }

    /// Build a dummy `AgentSpawnConfig` for tests. The binary path is
    /// garbage — these tests never actually spawn a process.
    fn dummy_spawn_config() -> AgentSpawnConfig {
        AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: PathBuf::from("/tmp"),
            model: "test-model".into(),
            sessions_dir: PathBuf::from("/tmp/sessions"),
            resume: None,
            env: vec![],
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ox-egui-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Helper to build an `OxApp` with N splits over duplex pipes. Returns
    /// the app, its session-id mirror, and a vec of the agent-side writer
    /// streams (one per split, for feeding events).
    fn make_app(n: usize) -> (OxApp, SessionIdMirror, Vec<tokio::io::DuplexStream>) {
        assert!(n >= 1);
        let mut splits = Vec::with_capacity(n);
        let mut writers = Vec::with_capacity(n);
        for _ in 0..n {
            let (split, writer) = make_split();
            splits.push(split);
            writers.push(writer);
        }
        let (app, mirror) = OxApp::new(splits, dummy_spawn_config(), "test-version");
        (app, mirror, writers)
    }

    // -- Split lifecycle tests --

    #[test]
    fn add_split_grows_splits_and_inputs() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        assert_eq!(app.splits.len(), 1);
        assert_eq!(app.inputs.len(), 1);
        assert_eq!(app.focused, 0);

        let (split2, _writer2) = make_split();
        app.add_split(split2);
        assert_eq!(app.splits.len(), 2);
        assert_eq!(app.inputs.len(), 2);
        // Focus moves to the new split.
        assert_eq!(app.focused, 1);
    }

    #[test]
    fn remove_split_shrinks_splits_and_inputs() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(3);
        assert_eq!(app.splits.len(), 3);

        // Remove the middle split.
        app.remove_split(1);
        assert_eq!(app.splits.len(), 2);
        assert_eq!(app.inputs.len(), 2);
    }

    #[test]
    fn remove_split_clamps_focus_when_removing_last_split() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(2);
        app.focused = 1; // focus on the last split
        app.remove_split(1); // remove it
        assert_eq!(app.focused, 0, "focus must clamp to valid range");
    }

    #[test]
    fn remove_split_preserves_focus_when_removing_before_focused() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(3);
        app.focused = 2; // focus on the last split
        app.remove_split(0); // remove the first split
        // Focus was at index 2, now there are only 2 splits (indices 0, 1).
        // Clamp to 1.
        assert_eq!(app.focused, 1);
    }

    #[test]
    fn session_id_mirror_reflects_dynamic_split_count() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, mirror, _writers) = make_app(1);
        let (split2, _writer2) = make_split();
        app.add_split(split2);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().len(), 2);

        app.remove_split(0);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().len(), 1);
    }

    #[test]
    fn any_turn_in_progress_is_true_when_a_split_is_waiting() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(2);
        app.splits[1].waiting = true;

        assert!(app.any_turn_in_progress());
    }

    #[test]
    fn any_turn_in_progress_is_true_when_a_split_is_streaming_only() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(2);
        app.splits[0].streaming = Some(app::StreamAccumulator::new());

        assert!(app.any_turn_in_progress());
    }

    #[test]
    fn any_turn_in_progress_is_false_when_all_splits_idle() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (app, _, _writers) = make_app(2);

        assert!(!app.any_turn_in_progress());
    }

    #[test]
    fn replace_workspace_updates_template_and_replaces_with_one_split() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(3);
        let new_root = PathBuf::from("/new/workspace");
        let mut seen = Vec::new();

        app.replace_workspace_with(new_root.clone(), |idx, config| {
            seen.push((idx, config.workspace_root.clone(), config.resume));
            Ok(make_split().0)
        });

        assert_eq!(app.spawn_config.workspace_root, new_root);
        assert_eq!(app.spawn_config.resume, None);
        assert_eq!(app.splits.len(), 1);
        assert_eq!(seen.len(), 1);
        assert!(seen.iter().all(|(_, root, resume)| {
            root == &PathBuf::from("/new/workspace") && resume.is_none()
        }));
    }

    #[test]
    fn replace_workspace_resets_inputs_focus_and_split_fracs() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(2);
        app.inputs[0] = "left".into();
        app.inputs[1] = "right".into();
        app.focused = 1;
        app.pending_focus = None;
        app.split_fracs = vec![0.25, 0.75];

        app.replace_workspace_with(PathBuf::from("/new/workspace"), |_, _| Ok(make_split().0));

        assert_eq!(app.inputs, vec![String::new()]);
        assert_eq!(app.focused, 0);
        assert_eq!(app.pending_focus, Some(0));
        assert_eq!(app.split_fracs, vec![1.0]);
    }

    #[test]
    fn replace_workspace_rebuilt_splits_are_published_to_session_id_mirror() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, mirror, _writers) = make_app(1);
        let (split2, _writer2) = make_split();
        app.add_split(split2);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().len(), 2);

        app.replace_workspace_with(PathBuf::from("/new/workspace"), |_, _| Ok(make_split().0));

        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);
    }

    #[test]
    fn replace_workspace_spawn_failure_preserves_existing_splits_and_template() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, mirror, _writers) = make_app(2);
        let old_root = app.spawn_config.workspace_root.clone();
        app.inputs[0] = "keep me".into();
        app.focused = 1;
        app.split_fracs = vec![0.4, 0.6];
        app.publish_session_ids();
        let old_mirror = mirror.lock().unwrap().clone();

        app.replace_workspace_with(PathBuf::from("/bad/workspace"), |_, _| {
            anyhow::bail!("boom");
        });

        assert_eq!(app.spawn_config.workspace_root, old_root);
        assert_eq!(app.splits.len(), 2);
        assert_eq!(app.inputs, vec!["keep me".to_owned(), String::new()]);
        assert_eq!(app.focused, 1);
        assert_eq!(app.split_fracs, vec![0.4, 0.6]);
        assert_eq!(mirror.lock().unwrap().clone(), old_mirror);
        assert!(
            app.splits[1]
                .error
                .as_deref()
                .is_some_and(|err| err.contains("failed to spawn agent") && err.contains("boom"))
        );
    }

    #[test]
    fn replace_workspace_restores_saved_sessions_fractions_and_focus() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let dir = temp_dir("replace-restore");
        let layout_path = dir.join("workspaces.json");
        let target_root = dir.join("target");
        std::fs::create_dir_all(&target_root).unwrap();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        let mut layouts = WorkspaceLayouts::default();
        layouts.save_current(&target_root, [Some(id1), Some(id2)], &[0.3, 0.7], 1);
        layouts.save(&layout_path).unwrap();

        let (mut app, _, _writers) = make_app(1);
        app.spawn_config.sessions_dir = dir.join("sessions");
        std::fs::create_dir_all(&app.spawn_config.sessions_dir).unwrap();
        std::fs::write(
            app.spawn_config.sessions_dir.join(format!("{id1}.json")),
            "{}",
        )
        .unwrap();
        std::fs::write(
            app.spawn_config.sessions_dir.join(format!("{id2}.json")),
            "{}",
        )
        .unwrap();
        app.layout_state_path = Some(layout_path);
        let mut seen = Vec::new();

        app.replace_workspace_with(target_root.clone(), |idx, config| {
            seen.push((idx, config.workspace_root.clone(), config.resume));
            Ok(make_split().0)
        });

        assert_eq!(
            seen,
            vec![
                (0, target_root.clone(), Some(id1)),
                (1, target_root.clone(), Some(id2))
            ]
        );
        assert_eq!(app.splits.len(), 2);
        assert_eq!(app.split_fracs, vec![0.3, 0.7]);
        assert_eq!(app.focused, 1);
        assert_eq!(app.spawn_config.workspace_root, target_root);
        assert_eq!(app.spawn_config.resume, None);
    }

    #[test]
    fn replace_workspace_saves_current_layout_before_switching() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let dir = temp_dir("replace-save-current");
        let layout_path = dir.join("workspaces.json");
        let old_root = dir.join("old");
        let new_root = dir.join("new");
        std::fs::create_dir_all(&old_root).unwrap();
        std::fs::create_dir_all(&new_root).unwrap();

        let (mut app, _, _writers) = make_app(2);
        app.layout_state_path = Some(layout_path.clone());
        app.spawn_config.workspace_root = old_root.clone();
        app.splits[0].session_id = Some(SessionId::new_v4());
        app.splits[1].session_id = Some(SessionId::new_v4());
        app.split_fracs = vec![0.4, 0.6];
        app.focused = 1;

        let old_ids: Vec<_> = app.splits.iter().map(|split| split.session_id()).collect();
        app.replace_workspace_with(new_root, |_, _| Ok(make_split().0));

        let layouts = WorkspaceLayouts::load(&layout_path).unwrap();
        let restored = layouts.restore_for(&old_root).unwrap();
        assert_eq!(
            restored.sessions,
            old_ids.into_iter().flatten().collect::<Vec<_>>()
        );
        assert_eq!(restored.split_fracs, vec![0.4, 0.6]);
        assert_eq!(restored.focused, 1);
    }

    #[test]
    fn restore_spawn_configs_use_saved_sessions_in_order() {
        let base = dummy_spawn_config();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        let layout = RestoreLayout {
            sessions: vec![id1, id2],
            split_fracs: vec![0.25, 0.75],
            focused: 1,
        };

        let configs = restore_spawn_configs(&base, Some(&layout));

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].resume, Some(id1));
        assert_eq!(configs[1].resume, Some(id2));
        assert!(
            configs
                .iter()
                .all(|config| config.workspace_root == base.workspace_root)
        );
    }

    #[test]
    fn restore_spawn_configs_falls_back_to_fresh_without_layout() {
        let mut base = dummy_spawn_config();
        base.resume = Some(SessionId::new_v4());

        let configs = restore_spawn_configs(&base, None);

        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].resume, None);
    }

    #[test]
    fn startup_spawn_configs_preserve_explicit_resume_override() {
        let mut base = dummy_spawn_config();
        let explicit = SessionId::new_v4();
        let saved = SessionId::new_v4();
        base.resume = Some(explicit);
        let layout = RestoreLayout {
            sessions: vec![saved],
            split_fracs: vec![1.0],
            focused: 0,
        };

        let configs = startup_spawn_configs(&base, Some(&layout));

        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].resume, Some(explicit));
    }

    // -- Command interception tests --

    #[tokio::test]
    async fn send_message_routes_to_correct_split() {
        // Build an app with two splits and verify that sending on split 1
        // doesn't touch split 0's agent.
        let (agent_writer0, client_reader0) = duplex(4096);
        let (client_writer0, agent_reader0) = duplex(4096);
        let client0 = AgentClient::new(BufReader::new(client_reader0), client_writer0);

        let (agent_writer1, client_reader1) = duplex(4096);
        let (client_writer1, agent_reader1) = duplex(4096);
        let client1 = AgentClient::new(BufReader::new(client_reader1), client_writer1);

        let (mut app, _mirror) = OxApp::new(
            vec![AgentSplit::new(client0), AgentSplit::new(client1)],
            dummy_spawn_config(),
            "test-version",
        );

        app.inputs[1] = "hello from split 1".into();
        app.send_message(1);

        // Split 1's agent should receive the command.
        let mut reader1 = BufReader::new(agent_reader1);
        let frame: Option<AgentCommand> = protocol::read_frame(&mut reader1).await.unwrap();
        match frame.unwrap() {
            AgentCommand::SendMessage { input } => assert_eq!(input, "hello from split 1"),
            other => panic!("unexpected {other:?}"),
        }

        // Split 0's input is untouched and no command was sent.
        assert!(app.inputs[0].is_empty());

        // Keep pipes alive to avoid spurious disconnect errors.
        drop(agent_writer0);
        drop(agent_writer1);
        drop(agent_reader0);
    }

    #[test]
    fn send_message_clears_input_and_sets_waiting() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.inputs[0] = "test message".into();
        app.send_message(0);
        assert!(
            app.inputs[0].is_empty(),
            "input should be cleared after send"
        );
        assert!(app.splits[0].waiting, "split should be in waiting state");
    }

    #[test]
    fn send_message_ignores_empty_input() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.inputs[0] = "   ".into(); // whitespace-only
        app.send_message(0);
        assert!(
            !app.splits[0].waiting,
            "empty input should not trigger send"
        );
    }

    #[test]
    fn send_message_skips_when_waiting() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.splits[0].waiting = true;
        app.inputs[0] = "should not send".into();
        app.send_message(0);
        // Input IS cleared (the text is consumed), but the split stays in
        // the same waiting state — no command is sent to the agent.
        assert!(app.inputs[0].is_empty());
        assert!(app.splits[0].waiting);
    }

    #[test]
    fn remove_split_tracks_focus_when_removing_before_non_last_focused() {
        // Regression: removing a split before the focused one should
        // decrement focus so it stays on the same logical split, not just
        // clamp when it overflows.
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(5);
        app.focused = 3;
        app.remove_split(1); // remove a split before focus
        // Was focused on logical split at old-index 3; it's now at index 2.
        assert_eq!(app.focused, 2);
        assert_eq!(app.splits.len(), 4);
    }

    #[test]
    fn close_split_on_last_split_uses_quit_confirmation_when_busy() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.splits[0].waiting = true;
        let ctx = egui::Context::default();

        app.handle_close_split(0, &ctx);

        assert_eq!(app.splits.len(), 1);
        assert!(app.confirm_quit);
    }

    #[test]
    fn native_close_request_is_cancelled_when_turn_is_running() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.splits[0].waiting = true;

        assert!(app.should_cancel_viewport_close_request());
        assert!(app.confirm_quit);
    }

    #[test]
    fn native_close_request_is_not_cancelled_after_quit_confirmed() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.splits[0].waiting = true;
        app.quit_confirmed = true;

        assert!(!app.should_cancel_viewport_close_request());
        assert!(!app.confirm_quit);
    }

    // -- Command classification tests --

    #[test]
    fn classify_input_recognizes_new() {
        assert!(matches!(classify_input("/new", 0), SplitAction::New));
        assert!(matches!(classify_input("  /new  ", 0), SplitAction::New));
        assert!(matches!(classify_input("/NEW", 0), SplitAction::New));
    }

    #[test]
    fn classify_input_recognizes_quit() {
        match classify_input("/quit", 2) {
            SplitAction::QuitApp => {}
            other => panic!("expected Quit, got {other:?}"),
        }
    }

    #[test]
    fn classify_input_recognizes_close() {
        match classify_input("/close", 2) {
            SplitAction::CloseSplit(idx) => assert_eq!(idx, 2),
            other => panic!("expected CloseSplit, got {other:?}"),
        }
    }

    #[test]
    fn classify_input_passes_unknown_commands_as_regular_messages() {
        // `/help`, `/foo`, etc. are NOT intercepted — they should be sent
        // to the agent as regular messages.
        match classify_input("/help", 0) {
            SplitAction::Send(idx) => assert_eq!(idx, 0),
            other => panic!("expected Send for /help, got {other:?}"),
        }
        match classify_input("hello world", 1) {
            SplitAction::Send(idx) => assert_eq!(idx, 1),
            other => panic!("expected Send for plain text, got {other:?}"),
        }
    }

    #[test]
    fn about_text_uses_supplied_app_version() {
        assert_eq!(about_text("9.8.7"), "Ox v9.8.7");
    }

    #[tokio::test]
    async fn unknown_slash_command_is_sent_to_agent() {
        // `/help` is not a known command — it should go to the agent.
        let (_agent_writer, client_reader) = duplex(4096);
        let (client_writer, agent_reader) = duplex(4096);
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);

        let (mut app, _mirror) = OxApp::new(
            vec![AgentSplit::new(client)],
            dummy_spawn_config(),
            "test-version",
        );

        app.inputs[0] = "/help".into();
        let action = classify_input(&app.inputs[0], 0);
        assert!(matches!(action, SplitAction::Send(0)));
        app.send_message(0);

        let mut reader = BufReader::new(agent_reader);
        let frame: Option<AgentCommand> = protocol::read_frame(&mut reader).await.unwrap();
        match frame.unwrap() {
            AgentCommand::SendMessage { input } => assert_eq!(input, "/help"),
            other => panic!("expected SendMessage, got {other:?}"),
        }
    }

    #[test]
    fn oxapp_publishes_session_ids_after_update() {
        // Independent test of `publish_session_ids` — does not need a tokio
        // runtime because it doesn't touch the reader/writer tasks. We check
        // that calling it reflects whatever `session_id` the splits have set.
        //
        // We have to pair construction with a client that won't panic in a
        // non-async context, so we create one with stub pipes and never
        // poll it (the reader/writer tasks sit idle).
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (split, _writer) = make_split();
        let (app, mirror) = OxApp::new(vec![split], dummy_spawn_config(), "test-version");
        // Before publish, the mirror is initialized to `vec![None]`.
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);

        // The field-level assertion — publishing the ID a split stores.
        let (mut split2, _writer2) = make_split();
        split2.session_id = Some(SessionId::new_v4());
        let expected = split2.session_id;
        let (app2, mirror2) = OxApp::new(vec![split2], dummy_spawn_config(), "test-version");
        app2.publish_session_ids();
        assert_eq!(mirror2.lock().unwrap().clone(), vec![expected]);
    }

    // -- TurnCancelled state transitions --------------------------------------

    #[tokio::test]
    async fn turn_cancelled_sets_cancelled_and_clears_waiting_and_streaming() {
        let (mut split, mut writer) = make_split();
        split.waiting = true;
        split.streaming = Some(app::StreamAccumulator::new());

        write_frame(&mut writer, &AgentEvent::TurnCancelled)
            .await
            .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| t.cancelled)
            .await
            .unwrap();

        assert!(!split.waiting, "waiting must be cleared");
        assert!(split.streaming.is_none(), "streaming must be cleared");
        assert!(split.error.is_none(), "error must be cleared");
        assert!(split.cancelled, "cancelled must be set");
    }

    #[tokio::test]
    async fn turn_cancelled_commits_streaming_content_as_message() {
        let (mut split, mut writer) = make_split();
        split.waiting = true;

        // Seed the streaming accumulator with content.
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial output".into(),
                },
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.streaming.is_some()
        })
        .await
        .unwrap();

        write_frame(&mut writer, &AgentEvent::TurnCancelled)
            .await
            .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| t.cancelled)
            .await
            .unwrap();

        // The streaming content should be committed as a message.
        assert_eq!(split.messages.len(), 1);
        assert_eq!(split.messages[0].role, Role::Assistant);
        assert_eq!(split.messages[0].text(), "partial output");
        assert!(split.streaming.is_none());
    }

    #[tokio::test]
    async fn turn_cancelled_with_no_streaming_appends_no_message() {
        let (mut split, mut writer) = make_split();
        split.waiting = true;
        // No streaming accumulator — cancel arrived before any deltas.

        write_frame(&mut writer, &AgentEvent::TurnCancelled)
            .await
            .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| t.cancelled)
            .await
            .unwrap();

        assert!(split.messages.is_empty(), "no message should be appended");
        assert!(split.cancelled);
    }

    #[tokio::test]
    async fn turn_cancelled_after_message_appended_does_not_double_commit() {
        // Realistic wire sequence: the agent commits the partial assistant
        // message via MessageAppended *before* sending TurnCancelled.
        // The GUI's MessageAppended handler clears the streaming accumulator,
        // so the TurnCancelled handler should find no accumulator and not
        // append a duplicate message.
        let (mut split, mut writer) = make_split();
        split.waiting = true;

        // 1. StreamDelta seeds the accumulator.
        write_frame(
            &mut writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.streaming.is_some()
        })
        .await
        .unwrap();

        // 2. Agent commits the partial message.
        write_frame(
            &mut writer,
            &AgentEvent::MessageAppended {
                message: Message::assistant(vec![domain::ContentBlock::Text {
                    text: "partial".into(),
                }]),
            },
        )
        .await
        .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| {
            t.messages.len() == 1
        })
        .await
        .unwrap();
        assert!(
            split.streaming.is_none(),
            "MessageAppended should clear the accumulator"
        );

        // 3. TurnCancelled arrives — no accumulator to commit.
        write_frame(&mut writer, &AgentEvent::TurnCancelled)
            .await
            .unwrap();
        wait_until(&mut split, Duration::from_secs(1), |t| t.cancelled)
            .await
            .unwrap();

        // Only one message — no double-commit.
        assert_eq!(
            split.messages.len(),
            1,
            "message must not be double-committed"
        );
        assert_eq!(split.messages[0].text(), "partial");
        assert!(split.cancelled);
        assert!(!split.waiting);
    }

    #[test]
    fn send_message_clears_cancelled_flag() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.splits[0].cancelled = true;
        app.inputs[0] = "next message".into();
        app.send_message(0);

        assert!(
            !app.splits[0].cancelled,
            "cancelled must be cleared on send"
        );
        assert!(app.splits[0].waiting, "split should be in waiting state");
    }
}
