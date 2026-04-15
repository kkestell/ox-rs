pub mod backend;

use anyhow::Result;
use domain::{ContentBlock, Message, Role};
use eframe::egui;
use tokio::sync::mpsc;

use app::StreamAccumulator;
use backend::{BackendCommand, BackendEvent};

pub struct OxApp {
    cmd_tx: mpsc::UnboundedSender<BackendCommand>,
    evt_rx: mpsc::UnboundedReceiver<BackendEvent>,
    messages: Vec<Message>,
    input: String,
    /// A live mirror of the in-flight assistant turn. We push every stream
    /// event into the accumulator and render a fresh snapshot each frame,
    /// so ordering and block shape match the final `Message` exactly —
    /// preventing a visual flicker when the accumulator is dropped and the
    /// final `AssistantMessage` takes its place in history.
    ///
    /// `None` when no turn is in flight (initial state and between turns).
    streaming: Option<StreamAccumulator>,
    /// True while waiting for the backend to respond — prevents double-sends
    /// and shows a visual indicator.
    waiting: bool,
    error: Option<String>,
}

impl OxApp {
    pub fn new(
        cmd_tx: mpsc::UnboundedSender<BackendCommand>,
        evt_rx: mpsc::UnboundedReceiver<BackendEvent>,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            cmd_tx,
            evt_rx,
            messages,
            input: String::new(),
            streaming: None,
            waiting: false,
            error: None,
        }
    }

    pub fn run(self) -> Result<()> {
        let options = eframe::NativeOptions::default();
        eframe::run_native("Ox", options, Box::new(|_cc| Ok(Box::new(self))))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    /// Drain all pending events from the backend without blocking.
    fn poll_events(&mut self) {
        loop {
            match self.evt_rx.try_recv() {
                Ok(BackendEvent::StreamDelta(event)) => {
                    // Lazily create the accumulator on the first delta.
                    // This keeps the "just submitted, no tokens yet" state
                    // distinguishable from the "tokens streaming" state for
                    // the UI's waiting placeholder.
                    self.streaming
                        .get_or_insert_with(StreamAccumulator::new)
                        .push(event);
                }
                Ok(BackendEvent::AssistantMessage(msg)) => {
                    self.messages.push(msg);
                    // The final message supersedes the accumulator — drop it
                    // so the live-view branch stops rendering.
                    self.streaming = None;
                    self.waiting = false;
                    self.error = None;
                }
                Ok(BackendEvent::Error(e)) => {
                    // Abandon the partial accumulator; whatever streamed is
                    // not a valid assistant turn.
                    self.streaming = None;
                    self.error = Some(e);
                    self.waiting = false;
                }
                Err(_) => break,
            }
        }
    }

    /// Send the current input as a user message.
    fn send_message(&mut self) {
        let text = self.input.trim().to_owned();
        if text.is_empty() || self.waiting {
            return;
        }

        // Show the user message immediately in the chat history.
        self.messages.push(Message::user(&text));

        let _ = self
            .cmd_tx
            .send(BackendCommand::SendMessage { input: text });
        self.input.clear();
        self.waiting = true;
        self.error = None;
    }
}

impl eframe::App for OxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();

        // Message history fills the central area.
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.messages {
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
                    // until the first delta arrives, so we fall back to the
                    // "..." placeholder while the request is in flight but
                    // hasn't produced anything yet.
                    if let Some(acc) = &self.streaming {
                        let snapshot = acc.snapshot();
                        ui.label(egui::RichText::new("Ox").strong());
                        render_blocks(ui, &snapshot.content);
                        ui.add_space(8.0);
                    } else if self.waiting {
                        ui.label("...");
                    }

                    if let Some(err) = &self.error {
                        ui.colored_label(egui::Color32::RED, err);
                    }
                });
        });

        // Input bar at the bottom.
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
                    .add_enabled(!self.waiting, egui::Button::new("Send"))
                    .clicked();

                if enter_pressed || send_clicked {
                    self.send_message();
                    // Return focus to the text input after sending.
                    input_response.request_focus();
                }
            });
            ui.add_space(4.0);
        });

        // Keep polling while waiting so we pick up streaming tokens and
        // the final response promptly.
        if self.waiting {
            ctx.request_repaint();
        }
    }
}

/// Render a sequence of `ContentBlock`s as a flat column of labels.
///
/// One renderer for both history and live streaming: each call takes a borrowed
/// slice so it doesn't care whether the blocks came from a persisted `Message`
/// or a fresh `StreamAccumulator::snapshot`. That symmetry is the whole point —
/// the live view and the final view can't drift apart because they run through
/// the same function.
///
/// Visual treatment is deliberately plain: a `kind:` prefix per non-text block,
/// no collapsibles, no JSON prettifying. If reasoning or tool-call volume
/// becomes noisy we can revisit, but flat text is the right default for now.
fn render_blocks(ui: &mut egui::Ui, blocks: &[ContentBlock]) {
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                ui.label(text);
            }
            ContentBlock::Reasoning { content, .. } => {
                // An encrypted-only reasoning block carries an empty `content`
                // — the opaque blob is persisted so we can re-send it, but
                // there's nothing to show the user. Skip rather than emit a
                // dangling "thinking:" header.
                if !content.is_empty() {
                    ui.label(format!("thinking: {content}"));
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
                // Plain-text "(error)" suffix — no color, no styling, matches
                // the no-fancy-formatting decision for the rest of the UI.
                if *is_error {
                    ui.label(format!("tool result: {content} (error)"));
                } else {
                    ui.label(format!("tool result: {content}"));
                }
            }
        }
    }
}
