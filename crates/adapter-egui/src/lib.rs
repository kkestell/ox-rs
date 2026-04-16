//! egui frontend for Ox.
//!
//! The GUI owns only display state. All session work — LLM streaming, tool
//! execution, session persistence — happens in a separate `ox-agent` process
//! spawned by `bin-gui` and talked to through an [`AgentClient`].
//!
//! Per-session state lives in [`AgentTab`] so the GUI can host N agents
//! concurrently. The tiling UI renders all active sessions as equal-width
//! vertical splits; `/new` adds a split, `/quit` closes one.

mod agent_client;

pub use agent_client::{AgentClient, AgentSpawnConfig};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use domain::{ContentBlock, Message, Role, SessionId};
use eframe::egui;
use protocol::{AgentCommand, AgentEvent};
use tokio::sync::mpsc;

use app::StreamAccumulator;

/// Shared, thread-safe mirror of the per-tab session IDs.
///
/// `eframe::run_native` consumes the `OxApp`, so after the window closes the
/// app's state is gone. To let the composition root print resume commands on
/// shutdown, the app writes the latest session IDs into this slot every
/// frame; the binary reads it after `run` returns.
///
/// A `Mutex<Vec<...>>` is overkill for single-threaded frame updates but it's
/// the smallest thread-safe shape that keeps the binary decoupled from the
/// egui runtime's threading model, which is a problem we don't want to audit
/// every time we touch this code.
pub type SessionIdMirror = Arc<Mutex<Vec<Option<SessionId>>>>;

/// Per-agent state. One tab per running agent subprocess.
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
pub struct AgentTab {
    client: AgentClient,
    messages: Vec<Message>,
    /// A live mirror of the in-flight assistant turn. Lazily created on the
    /// first `StreamDelta` so the "waiting for any output" placeholder UI
    /// branch stays distinguishable from "tokens are arriving."
    streaming: Option<StreamAccumulator>,
    waiting: bool,
    error: Option<String>,
    session_id: Option<SessionId>,
}

impl AgentTab {
    pub fn new(client: AgentClient) -> Self {
        Self {
            client,
            messages: Vec::new(),
            streaming: None,
            waiting: false,
            error: None,
            session_id: None,
        }
    }

    /// Session ID reported by the agent via `Ready`. `None` until the
    /// handshake completes. The GUI reads this on shutdown so it can print
    /// the user's resume command.
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

/// Top-level GUI state. Owns a vector of agent tabs displayed as resizable
/// vertical splits. `focused` indexes the split whose input bar receives
/// keystrokes. Each split has its own input string in `inputs` (parallel to
/// `tabs`) so rendering can borrow `&mut inputs[i]` and `&tabs[i]`
/// simultaneously — Rust can split-borrow distinct struct fields but not
/// `&mut tab.input` and `&tab.messages` on the same struct.
///
/// Split widths are stored as fractions (`split_fracs`) summing to 1.0.
/// Dragging a separator transfers width between adjacent splits. Using
/// fractions rather than absolute pixels means splits scale naturally
/// when the window is resized.
pub struct OxApp {
    tabs: Vec<AgentTab>,
    /// Index of the focused split. Keystrokes go to `inputs[focused]`.
    focused: usize,
    /// Per-split input strings, parallel to `tabs`.
    inputs: Vec<String>,
    /// Fractional width of each split, summing to 1.0. Updated by
    /// dragging the separator between adjacent splits.
    split_fracs: Vec<f32>,
    /// Template config for spawning new agents via `/new`. Cloned with
    /// `resume: None` each time a new split is created.
    spawn_config: AgentSpawnConfig,
    /// Slot the app writes session IDs into each frame so the composition
    /// root can read them after `run_native` consumes the app.
    session_id_mirror: SessionIdMirror,
    /// When set, the next frame will request egui focus on this split's
    /// input TextEdit, then clear the flag.
    pending_focus: Option<usize>,
}

impl OxApp {
    pub fn new(tabs: Vec<AgentTab>, spawn_config: AgentSpawnConfig) -> (Self, SessionIdMirror) {
        assert!(!tabs.is_empty(), "OxApp requires at least one tab");
        let n = tabs.len();
        let mirror: SessionIdMirror = Arc::new(Mutex::new(vec![None; n]));
        let inputs = vec![String::new(); n];
        let split_fracs = vec![1.0 / n as f32; n];
        let app = Self {
            tabs,
            focused: 0,
            inputs,
            split_fracs,
            spawn_config,
            session_id_mirror: mirror.clone(),
            pending_focus: None,
        };
        (app, mirror)
    }

    pub fn run(self) -> Result<()> {
        let options = eframe::NativeOptions::default();
        eframe::run_native("Ox", options, Box::new(|_cc| Ok(Box::new(self))))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn publish_session_ids(&self) {
        let ids: Vec<Option<SessionId>> = self.tabs.iter().map(|t| t.session_id()).collect();
        if let Ok(mut slot) = self.session_id_mirror.lock() {
            *slot = ids;
        }
    }

    /// Spawn a new agent and append a split to the right. On failure,
    /// the error is shown on the currently focused split.
    fn handle_new(&mut self, _ctx: &egui::Context) {
        let mut config = self.spawn_config.clone();
        config.resume = None;
        match AgentClient::spawn(config) {
            Ok(client) => {
                let tab = AgentTab::new(client);
                self.tabs.push(tab);
                self.inputs.push(String::new());
                self.focused = self.tabs.len() - 1;
                self.pending_focus = Some(self.focused);
                // Redistribute widths equally.
                let n = self.tabs.len();
                self.split_fracs = vec![1.0 / n as f32; n];
            }
            Err(e) => {
                self.tabs[self.focused].error = Some(format!("failed to spawn agent: {e:#}"));
            }
        }
    }

    /// Close the split at `split_idx`. If it's the last split, close the
    /// app. Otherwise remove the tab and adjust focus so it stays on the
    /// same logical split (or its left neighbor if the focused split was
    /// removed).
    fn handle_quit(&mut self, split_idx: usize, ctx: &egui::Context) {
        if self.tabs.len() == 1 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        // Give the removed split's width to its neighbor.
        let reclaimed = self.split_fracs.remove(split_idx);
        let neighbor = if split_idx < self.split_fracs.len() {
            split_idx
        } else {
            split_idx - 1
        };
        self.split_fracs[neighbor] += reclaimed;

        self.tabs.remove(split_idx);
        self.inputs.remove(split_idx);
        adjust_focus_after_remove(&mut self.focused, split_idx, self.tabs.len());
        self.pending_focus = Some(self.focused);
    }

    /// Add a pre-built tab. Used by tests that can't spawn a real agent
    /// but need to exercise split add/remove logic.
    #[cfg(test)]
    fn add_tab(&mut self, tab: AgentTab) {
        self.tabs.push(tab);
        self.inputs.push(String::new());
        self.focused = self.tabs.len() - 1;
        let n = self.tabs.len();
        self.split_fracs = vec![1.0 / n as f32; n];
    }

    /// Remove the tab at `split_idx` without viewport commands. Used by
    /// tests that don't have an egui context.
    #[cfg(test)]
    fn remove_tab(&mut self, split_idx: usize) {
        let reclaimed = self.split_fracs.remove(split_idx);
        let neighbor = if split_idx < self.split_fracs.len() {
            split_idx
        } else {
            split_idx - 1
        };
        self.split_fracs[neighbor] += reclaimed;

        self.tabs.remove(split_idx);
        self.inputs.remove(split_idx);
        adjust_focus_after_remove(&mut self.focused, split_idx, self.tabs.len());
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

        let tab = &mut self.tabs[split_idx];
        if tab.waiting {
            return;
        }

        // Best-effort send. If the writer task is gone (agent dead), the
        // next `poll_events` will surface the disconnect as an Error.
        let _ = tab.client.send(AgentCommand::SendMessage { input: text });
        tab.waiting = true;
        tab.error = None;
    }
}

impl eframe::App for OxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for tab in &mut self.tabs {
            tab.poll_events();
        }

        // Publish the latest session IDs so the composition root sees them
        // on shutdown — `eframe::run_native` consumes the app and returns
        // no post-exit state.
        self.publish_session_ids();

        // Tiling layout: a single CentralPanel divided into N vertical
        // splits with draggable separators between them. Split widths are
        // stored as fractions of the available width (`split_fracs`),
        // updated by dragging a separator.
        //
        // Borrow-checker constraint: we can't call `&mut self` methods
        // while borrowing tabs/inputs, so we collect deferred "actions"
        // and execute them after all splits are rendered.
        let mut actions: Vec<SplitAction> = Vec::new();
        let mut new_focus: Option<usize> = None;
        let any_waiting = self.tabs.iter().any(|t| t.waiting);
        let n = self.tabs.len();

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
                    let tab = &self.tabs[i];
                    let input = &mut self.inputs[i];
                    let grab_focus = self.pending_focus == Some(i);
                    render_split(
                        &mut child,
                        tab,
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

        // Execute deferred actions outside the borrow of `self.tabs` /
        // `self.inputs`.
        for action in actions {
            match action {
                SplitAction::Send(idx) => self.send_message(idx),
                SplitAction::New => self.handle_new(ctx),
                SplitAction::Quit(idx) => self.handle_quit(idx, ctx),
            }
        }

        // Keep polling while any split is waiting so we pick up streaming
        // tokens and the final response promptly.
        if any_waiting {
            ctx.request_repaint();
        }
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

/// Classify user input as a command (`/new`, `/quit`) or a regular message.
///
/// Extracted from the render closure so it can be unit-tested without an
/// egui context.
fn classify_input(text: &str, split_idx: usize) -> SplitAction {
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("/new") {
        SplitAction::New
    } else if trimmed.eq_ignore_ascii_case("/quit") {
        SplitAction::Quit(split_idx)
    } else {
        SplitAction::Send(split_idx)
    }
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
    /// Close the split at this index. Drops the agent (SIGKILL via
    /// `kill_on_drop`).
    Quit(usize),
}

/// Render one vertical split: scroll area with message history, streaming
/// view, error display, and an input bar pinned at the bottom.
///
/// Any actions the user triggers (send, `/new`, `/quit`) are pushed into
/// `actions` for deferred execution. Focus changes are recorded in
/// `new_focus`.
fn render_split(
    ui: &mut egui::Ui,
    tab: &AgentTab,
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
    for msg in &tab.messages {
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
            for msg in &tab.messages {
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
            if let Some(acc) = &tab.streaming {
                let snapshot = acc.snapshot();
                let empty = HashMap::new();
                render_blocks(ui, snapshot.content, &Role::Assistant, &empty);
                ui.add_space(8.0);
            } else if tab.waiting {
                ui.label("...");
            }

            if let Some(err) = &tab.error {
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
        if matches!(action, SplitAction::New | SplitAction::Quit(_)) {
            input.clear();
        }
        actions.push(action);
        input_response.request_focus();
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
/// - **Tool calls**: clickable toggle header `▼ name(args)` / `▶ name(args)`.
///   Body shows the paired tool result. Open/closed state stored in egui's
///   per-frame `Memory` keyed by the tool call ID.
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
                let color = match role {
                    Role::User => egui::Color32::from_rgb(100, 149, 237),
                    _ => egui::Color32::WHITE,
                };
                ui.label(egui::RichText::new(text).color(color));
                rendered_any = true;
            }
            ContentBlock::Reasoning { content, .. } => {
                if !content.is_empty() {
                    ui.label(egui::RichText::new(content).color(egui::Color32::GRAY));
                    rendered_any = true;
                }
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                // Manual toggle: clickable label with a triangle prefix.
                // State stored in egui's temp data keyed by the tool call ID,
                // defaulting to open when a result exists, closed otherwise.
                let toggle_id = egui::Id::new("tool_toggle").with(id);
                let default_open = tool_results.contains_key(id.as_str());
                let mut open = ui
                    .ctx()
                    .data_mut(|d| *d.get_temp_mut_or(toggle_id, default_open));

                let arrow = if open { "▼" } else { "▶" };
                let header_text = format!("{arrow} {name}({arguments})");
                let response = ui.add(
                    egui::Label::new(egui::RichText::new(&header_text).color(egui::Color32::GRAY))
                        .sense(egui::Sense::click()),
                );
                if response.clicked() {
                    open = !open;
                    ui.ctx().data_mut(|d| d.insert_temp(toggle_id, open));
                }

                if open
                    && let Some(&(content, is_error)) = tool_results.get(id.as_str())
                {
                    let color = if is_error {
                        egui::Color32::RED
                    } else {
                        egui::Color32::GRAY
                    };
                    ui.label(egui::RichText::new(content).color(color));
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
    //! `AgentTab` state-machine tests.
    //!
    //! `OxApp::update` is driven by egui and exercised by running the app;
    //! we don't try to simulate egui here. Instead we pin down the invariants
    //! the plan calls out for `AgentTab`:
    //!
    //! - `Ready` records `session_id` and *only* that — `messages`, `waiting`,
    //!   `streaming`, `error` stay untouched.
    //! - `MessageAppended` is the single entry point for both historical
    //!   replay and live turn messages; it appends unconditionally.
    //! - `TurnComplete` clears `waiting` and `streaming` and `error`.
    //! - `Error` clears `waiting` and `streaming` and records the message.
    //!
    //! Each test drives a real `AgentTab` through its `poll_events` method
    //! by feeding frames into an in-memory duplex pipe.

    use std::path::PathBuf;
    use std::time::Duration;

    use domain::{Message, StreamEvent, Usage};
    use protocol::{AgentEvent, write_frame};
    use tokio::io::{BufReader, duplex};

    use super::*;

    /// Build a tab whose `AgentClient` reads from `agent_writer`'s peer.
    /// The caller writes `AgentEvent` frames into `agent_writer` and then
    /// awaits `recv_until`, which pumps `tab.poll_events()` until a given
    /// condition holds.
    fn make_tab() -> (AgentTab, tokio::io::DuplexStream) {
        let (agent_writer, client_reader) = duplex(4096);
        // Throwaway command pipe — these tests don't inspect commands.
        let (client_writer, _agent_reader) = duplex(4096);
        std::mem::forget(_agent_reader); // keep the pipe's read-end alive
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);
        (AgentTab::new(client), agent_writer)
    }

    /// Pump `tab.poll_events()` until `check` returns true or the timeout
    /// elapses. Returns Ok iff `check` ended true. Uses short sleeps (5ms)
    /// to let the reader task pick up the frames we wrote.
    async fn wait_until(
        tab: &mut AgentTab,
        timeout: Duration,
        mut check: impl FnMut(&AgentTab) -> bool,
    ) -> Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            tab.poll_events();
            if check(tab) {
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
        let (mut tab, mut writer) = make_tab();
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

        wait_until(&mut tab, Duration::from_secs(1), |t| {
            t.session_id() == Some(id)
        })
        .await
        .unwrap();

        assert!(tab.messages.is_empty(), "Ready must not touch messages");
        assert!(!tab.waiting, "Ready must not flip waiting");
        assert!(tab.streaming.is_none(), "Ready must not touch streaming");
        assert!(tab.error.is_none(), "Ready must not set an error");
    }

    #[tokio::test]
    async fn message_appended_extends_history_without_touching_streaming() {
        let (mut tab, mut writer) = make_tab();
        write_frame(
            &mut writer,
            &AgentEvent::MessageAppended {
                message: Message::user("hi"),
            },
        )
        .await
        .unwrap();

        wait_until(&mut tab, Duration::from_secs(1), |t| t.messages.len() == 1)
            .await
            .unwrap();
        assert_eq!(tab.messages[0].text(), "hi");
        assert!(!tab.waiting, "history replay should not set waiting");
        assert!(tab.streaming.is_none());
    }

    #[tokio::test]
    async fn message_appended_drops_any_inflight_stream_accumulator() {
        // If a StreamDelta has already arrived, the tab is building an
        // accumulator mirror. When the committed message lands, the
        // accumulator must be dropped so the renderer doesn't render the
        // same tokens twice.
        let (mut tab, mut writer) = make_tab();
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
        wait_until(&mut tab, Duration::from_secs(1), |t| t.streaming.is_some())
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
        wait_until(&mut tab, Duration::from_secs(1), |t| t.messages.len() == 1)
            .await
            .unwrap();
        assert!(
            tab.streaming.is_none(),
            "streaming mirror must be dropped on commit"
        );
    }

    #[tokio::test]
    async fn turn_complete_clears_waiting_and_streaming_and_error() {
        let (mut tab, mut writer) = make_tab();
        // Seed state the terminator has to clear.
        tab.waiting = true;
        tab.error = Some("old".into());
        tab.streaming = Some(app::StreamAccumulator::new());

        write_frame(&mut writer, &AgentEvent::TurnComplete)
            .await
            .unwrap();
        wait_until(&mut tab, Duration::from_secs(1), |t| !t.waiting)
            .await
            .unwrap();
        assert!(tab.streaming.is_none());
        assert!(tab.error.is_none());
    }

    #[tokio::test]
    async fn error_frame_records_message_and_clears_streaming() {
        let (mut tab, mut writer) = make_tab();
        tab.waiting = true;
        tab.streaming = Some(app::StreamAccumulator::new());

        write_frame(
            &mut writer,
            &AgentEvent::Error {
                message: "model overloaded".into(),
            },
        )
        .await
        .unwrap();
        wait_until(&mut tab, Duration::from_secs(1), |t| t.error.is_some())
            .await
            .unwrap();
        assert_eq!(tab.error.as_deref(), Some("model overloaded"));
        assert!(!tab.waiting);
        assert!(tab.streaming.is_none());
    }

    #[tokio::test]
    async fn stream_delta_seeds_accumulator_on_first_event() {
        let (mut tab, mut writer) = make_tab();
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
        wait_until(&mut tab, Duration::from_secs(1), |t| t.streaming.is_some())
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

        wait_until(&mut tab, Duration::from_secs(1), |t| {
            t.streaming
                .as_ref()
                .map(|a| a.snapshot().token_count == 2)
                .unwrap_or(false)
        })
        .await
        .unwrap();
        let snap = tab.streaming.as_ref().unwrap().snapshot();
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

    /// Helper to build an `OxApp` with N tabs over duplex pipes. Returns
    /// the app, its session-id mirror, and a vec of the agent-side writer
    /// streams (one per tab, for feeding events).
    fn make_app(n: usize) -> (OxApp, SessionIdMirror, Vec<tokio::io::DuplexStream>) {
        assert!(n >= 1);
        let mut tabs = Vec::with_capacity(n);
        let mut writers = Vec::with_capacity(n);
        for _ in 0..n {
            let (tab, writer) = make_tab();
            tabs.push(tab);
            writers.push(writer);
        }
        let (app, mirror) = OxApp::new(tabs, dummy_spawn_config());
        (app, mirror, writers)
    }

    // -- Split lifecycle tests --

    #[test]
    fn add_tab_grows_tabs_and_inputs() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        assert_eq!(app.tabs.len(), 1);
        assert_eq!(app.inputs.len(), 1);
        assert_eq!(app.focused, 0);

        let (tab2, _w2) = make_tab();
        app.add_tab(tab2);
        assert_eq!(app.tabs.len(), 2);
        assert_eq!(app.inputs.len(), 2);
        // Focus moves to the new split.
        assert_eq!(app.focused, 1);
    }

    #[test]
    fn remove_tab_shrinks_tabs_and_inputs() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(3);
        assert_eq!(app.tabs.len(), 3);

        // Remove the middle split.
        app.remove_tab(1);
        assert_eq!(app.tabs.len(), 2);
        assert_eq!(app.inputs.len(), 2);
    }

    #[test]
    fn remove_tab_clamps_focus_when_removing_last_split() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(2);
        app.focused = 1; // focus on the last split
        app.remove_tab(1); // remove it
        assert_eq!(app.focused, 0, "focus must clamp to valid range");
    }

    #[test]
    fn remove_tab_preserves_focus_when_removing_before_focused() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(3);
        app.focused = 2; // focus on the last split
        app.remove_tab(0); // remove the first split
        // Focus was at index 2, now there are only 2 tabs (indices 0, 1).
        // Clamp to 1.
        assert_eq!(app.focused, 1);
    }

    #[test]
    fn session_id_mirror_reflects_dynamic_tab_count() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, mirror, _writers) = make_app(1);
        let (tab2, _w2) = make_tab();
        app.add_tab(tab2);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().len(), 2);

        app.remove_tab(0);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().len(), 1);
    }

    // -- Command interception tests --

    #[tokio::test]
    async fn send_message_routes_to_correct_split() {
        // Build an app with two tabs and verify that sending on split 1
        // doesn't touch split 0's agent.
        let (agent_writer0, client_reader0) = duplex(4096);
        let (client_writer0, agent_reader0) = duplex(4096);
        let client0 = AgentClient::new(BufReader::new(client_reader0), client_writer0);

        let (agent_writer1, client_reader1) = duplex(4096);
        let (client_writer1, agent_reader1) = duplex(4096);
        let client1 = AgentClient::new(BufReader::new(client_reader1), client_writer1);

        let (mut app, _mirror) = OxApp::new(
            vec![AgentTab::new(client0), AgentTab::new(client1)],
            dummy_spawn_config(),
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
        assert!(app.tabs[0].waiting, "tab should be in waiting state");
    }

    #[test]
    fn send_message_ignores_empty_input() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.inputs[0] = "   ".into(); // whitespace-only
        app.send_message(0);
        assert!(!app.tabs[0].waiting, "empty input should not trigger send");
    }

    #[test]
    fn send_message_skips_when_waiting() {
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(1);
        app.tabs[0].waiting = true;
        app.inputs[0] = "should not send".into();
        app.send_message(0);
        // Input IS cleared (the text is consumed), but the tab stays in
        // the same waiting state — no command is sent to the agent.
        assert!(app.inputs[0].is_empty());
        assert!(app.tabs[0].waiting);
    }

    #[test]
    fn remove_tab_tracks_focus_when_removing_before_non_last_focused() {
        // Regression: removing a split before the focused one should
        // decrement focus so it stays on the same logical tab, not just
        // clamp when it overflows.
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (mut app, _, _writers) = make_app(5);
        app.focused = 3;
        app.remove_tab(1); // remove a tab before focus
        // Was focused on logical tab at old-index 3; it's now at index 2.
        assert_eq!(app.focused, 2);
        assert_eq!(app.tabs.len(), 4);
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
            SplitAction::Quit(idx) => assert_eq!(idx, 2),
            other => panic!("expected Quit, got {other:?}"),
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

    #[tokio::test]
    async fn new_command_is_not_sent_to_agent() {
        // `/new` must be intercepted — the agent should never see it.
        // We verify by checking that the agent's reader has no frames.
        let (agent_writer, client_reader) = duplex(4096);
        let (client_writer, agent_reader) = duplex(4096);
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);

        let (mut app, _mirror) = OxApp::new(vec![AgentTab::new(client)], dummy_spawn_config());

        // Simulate what the render closure does: classify, clear, send.
        app.inputs[0] = "/new".into();
        let action = classify_input(&app.inputs[0], 0);
        assert!(matches!(action, SplitAction::New));
        app.inputs[0].clear();
        // We don't call handle_new (it would fail — no real binary), but
        // the key assertion is that no command reached the agent.

        // Drop the writer so the agent reader sees EOF, not a hang.
        drop(app);
        drop(agent_writer);

        // Confirm no command was sent.
        let mut reader = BufReader::new(agent_reader);
        let frame: Option<AgentCommand> = protocol::read_frame(&mut reader).await.unwrap();
        assert!(
            frame.is_none(),
            "agent should not receive /new as a message"
        );
    }

    #[tokio::test]
    async fn unknown_slash_command_is_sent_to_agent() {
        // `/help` is not a known command — it should go to the agent.
        let (_agent_writer, client_reader) = duplex(4096);
        let (client_writer, agent_reader) = duplex(4096);
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);

        let (mut app, _mirror) = OxApp::new(vec![AgentTab::new(client)], dummy_spawn_config());

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
        // that calling it reflects whatever `session_id` the tabs have set.
        //
        // We have to pair construction with a client that won't panic in a
        // non-async context, so we create one with stub pipes and never
        // poll it (the reader/writer tasks sit idle).
        use tokio::runtime::Builder;
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();

        let (tab, _writer) = make_tab();
        let (app, mirror) = OxApp::new(vec![tab], dummy_spawn_config());
        // Before publish, the mirror is initialized to `vec![None]`.
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);

        // The field-level assertion — publishing the ID a tab stores.
        let (mut tab2, _w2) = make_tab();
        tab2.session_id = Some(SessionId::new_v4());
        let expected = tab2.session_id;
        let (app2, mirror2) = OxApp::new(vec![tab2], dummy_spawn_config());
        app2.publish_session_ids();
        assert_eq!(mirror2.lock().unwrap().clone(), vec![expected]);
    }
}
