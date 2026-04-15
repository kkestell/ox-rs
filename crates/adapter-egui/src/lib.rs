pub mod backend;

use anyhow::Result;
use domain::{Message, Role};
use eframe::egui;
use tokio::sync::mpsc;

use app::StreamEvent;
use backend::{BackendCommand, BackendEvent};

pub struct OxApp {
    cmd_tx: mpsc::UnboundedSender<BackendCommand>,
    evt_rx: mpsc::UnboundedReceiver<BackendEvent>,
    messages: Vec<Message>,
    input: String,
    /// Text accumulated from streaming text deltas. Rendered in place of the
    /// "..." indicator once the first token arrives, then cleared when the
    /// final AssistantMessage lands.
    streaming_text: String,
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
            streaming_text: String::new(),
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
                Ok(BackendEvent::StreamDelta(StreamEvent::TextDelta(s))) => {
                    self.streaming_text.push_str(&s);
                }
                Ok(BackendEvent::StreamDelta(_)) => {
                    // Non-text stream events (reasoning, tool calls, finished)
                    // are accumulated by SessionRunner but not rendered yet.
                }
                Ok(BackendEvent::AssistantMessage(msg)) => {
                    self.messages.push(msg);
                    self.streaming_text.clear();
                    self.waiting = false;
                    self.error = None;
                }
                Ok(BackendEvent::Error(e)) => {
                    self.streaming_text.clear();
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
                        ui.label(msg.text());
                        ui.add_space(8.0);
                    }

                    if !self.streaming_text.is_empty() {
                        ui.label(egui::RichText::new("Ox").strong());
                        ui.label(&self.streaming_text);
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
