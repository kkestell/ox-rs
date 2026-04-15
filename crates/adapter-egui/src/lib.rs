//! egui frontend for Ox.
//!
//! The GUI owns only display state. All session work — LLM streaming, tool
//! execution, session persistence — happens in a separate `ox-agent` process
//! spawned by `bin-gui` and talked to through an [`AgentClient`].
//!
//! Per-session state lives in [`AgentTab`] so the GUI can host N agents
//! concurrently. The current iteration always uses `tabs[0]`; tiling UI and
//! tab switching land in a follow-up without another restructure.

mod agent_client;

pub use agent_client::{AgentClient, AgentSpawnConfig};

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

/// Top-level GUI state. Owns a vector of agent tabs; `current` indexes the
/// tab whose messages are rendered. For this iteration `current` is always
/// 0 and no UI exists to switch tabs — the plumbing sits here so a future
/// tiling UI can grow into it without another restructure.
pub struct OxApp {
    tabs: Vec<AgentTab>,
    current: usize,
    input: String,
    /// Slot the app writes session IDs into each frame so the composition
    /// root can read them after `run_native` consumes the app.
    session_id_mirror: SessionIdMirror,
}

impl OxApp {
    pub fn new(tabs: Vec<AgentTab>) -> (Self, SessionIdMirror) {
        assert!(!tabs.is_empty(), "OxApp requires at least one tab");
        let mirror: SessionIdMirror = Arc::new(Mutex::new(vec![None; tabs.len()]));
        let app = Self {
            tabs,
            current: 0,
            input: String::new(),
            session_id_mirror: mirror.clone(),
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

    /// Send the current input as a user message to the current tab's agent.
    ///
    /// We DON'T optimistically push the user message into `messages`: the
    /// agent round-trips it back as a `MessageAppended` event, and having a
    /// single render path for every message (user input, tool results,
    /// assistant replies) keeps the code paths small.
    fn send_message(&mut self) {
        let text = self.input.trim().to_owned();
        if text.is_empty() {
            return;
        }
        self.input.clear();

        let tab = &mut self.tabs[self.current];
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

        let tab = &self.tabs[self.current];
        let waiting = tab.waiting;

        // Input bar at the bottom. Must be registered BEFORE the central
        // panel — egui allocates panels in declaration order, and the
        // central panel consumes whatever space remains. If the bottom
        // panel is added after the central panel, the central panel has
        // already claimed the full viewport and message content ends up
        // hidden behind the input bar.
        egui::TopBottomPanel::bottom("input_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let input_response = ui.add_sized(
                    [ui.available_width() - 60.0, 24.0],
                    egui::TextEdit::singleline(&mut self.input).hint_text("Type a message..."),
                );

                let enter_pressed =
                    input_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                let send_clicked = ui
                    .add_enabled(!waiting, egui::Button::new("Send"))
                    .clicked();

                if enter_pressed || send_clicked {
                    self.send_message();
                    input_response.request_focus();
                }
            });
            ui.add_space(4.0);
        });

        // Message history fills whatever space the bottom panel leaves.
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let tab = &self.tabs[self.current];
                    for msg in &tab.messages {
                        let label = match msg.role {
                            Role::User => "You",
                            Role::Assistant => "Ox",
                            Role::Tool => "Tool",
                        };
                        ui.label(egui::RichText::new(label).strong());
                        render_blocks(ui, &msg.content);
                        ui.add_space(8.0);
                    }

                    // Live view of the in-flight turn. `streaming` is `None`
                    // until the first delta arrives, so the "..." placeholder
                    // shows while the request is in flight but hasn't
                    // produced anything yet.
                    if let Some(acc) = &tab.streaming {
                        let snapshot = acc.snapshot();
                        ui.label(egui::RichText::new("Ox").strong());
                        render_blocks(ui, &snapshot.content);
                        ui.add_space(8.0);
                    } else if tab.waiting {
                        ui.label("...");
                    }

                    if let Some(err) = &tab.error {
                        ui.colored_label(egui::Color32::RED, err);
                    }
                });
        });

        // Keep polling while waiting so we pick up streaming tokens and
        // the final response promptly.
        if waiting {
            ctx.request_repaint();
        }
    }
}

/// Render a sequence of `ContentBlock`s as a flat column of labels.
///
/// One renderer for both history and live streaming: each call takes a
/// borrowed slice so it doesn't care whether the blocks came from a
/// persisted `Message` or a fresh `StreamAccumulator::snapshot`. That
/// symmetry is the whole point — the live view and the final view can't
/// drift apart because they run through the same function.
///
/// Visual treatment is deliberately plain: a `kind:` prefix per non-text
/// block, no collapsibles, no JSON prettifying. If reasoning or tool-call
/// volume becomes noisy we can revisit, but flat text is the right default
/// for now.
fn render_blocks(ui: &mut egui::Ui, blocks: &[ContentBlock]) {
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                ui.label(text);
            }
            ContentBlock::Reasoning { content, .. } => {
                // An encrypted-only reasoning block carries empty `content`
                // — the opaque blob is persisted so the provider can
                // re-verify it, but there's nothing to show the user. Skip
                // rather than emit a dangling "thinking:" header.
                //
                // Red visually separates the model's internal monologue
                // from its final answer. Works against both light and dark
                // egui themes.
                if !content.is_empty() {
                    ui.label(
                        egui::RichText::new(format!("thinking: {content}"))
                            .color(egui::Color32::RED),
                    );
                }
            }
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                ui.label(format!("tool call: {name}({arguments})"));
            }
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                // Plain-text "(error)" suffix — no color, no styling,
                // matches the no-fancy-formatting decision for the rest of
                // the UI.
                if *is_error {
                    ui.label(format!("tool result: {content} (error)"));
                } else {
                    ui.label(format!("tool result: {content}"));
                }
            }
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
        assert_eq!(snap.text(), "hello world");
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
        let (app, mirror) = OxApp::new(vec![tab]);
        // Before publish, the mirror is initialized to `vec![None]`.
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);
        app.publish_session_ids();
        assert_eq!(mirror.lock().unwrap().clone(), vec![None]);

        // The field-level assertion — publishing the ID a tab stores.
        let (mut tab2, _w2) = make_tab();
        tab2.session_id = Some(SessionId::new_v4());
        let expected = tab2.session_id;
        let (app2, mirror2) = OxApp::new(vec![tab2]);
        app2.publish_session_ids();
        assert_eq!(mirror2.lock().unwrap().clone(), vec![expected]);
    }
}
