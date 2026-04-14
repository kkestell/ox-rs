use std::path::PathBuf;

use app::{LlmProvider, SessionRunner, SessionStore};
use domain::{Message, SessionId};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Channel protocol — the contract between the GUI and the async backend
// ---------------------------------------------------------------------------

/// Commands the GUI sends to the backend controller.
pub enum BackendCommand {
    /// User submitted a chat message.
    SendMessage { input: String },
}

/// Events the backend sends back to the GUI.
pub enum BackendEvent {
    /// A completed assistant message (non-streaming for now).
    AssistantMessage(Message),
    /// An error from the LLM or session layer.
    Error(String),
}

// ---------------------------------------------------------------------------
// Backend controller — owns the SessionRunner, receives commands, sends events
// ---------------------------------------------------------------------------

/// Async loop that bridges the GUI (via channels) to the SessionRunner.
///
/// Maintains an `Option<SessionId>` — `None` on startup means the first
/// `SendMessage` creates a new session via `start()`, subsequent messages
/// resume the same session via `resume()`.
///
/// The loop exits when the command channel closes (GUI shutdown).
pub async fn run_backend<L, S>(
    runner: SessionRunner<L, S>,
    mut cmd_rx: mpsc::UnboundedReceiver<BackendCommand>,
    evt_tx: mpsc::UnboundedSender<BackendEvent>,
    workspace_root: PathBuf,
) where
    L: LlmProvider + Send + Sync + 'static,
    S: SessionStore + Send + Sync + 'static,
{
    let mut session_id: Option<SessionId> = None;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCommand::SendMessage { input } => {
                let result = match session_id {
                    None => {
                        // First message — create a fresh session.
                        let id = SessionId::new_v4();
                        match runner.start(id, workspace_root.clone(), &input).await {
                            Ok((id, msg)) => {
                                session_id = Some(id);
                                Ok(msg)
                            }
                            // start() failed — leave session_id as None so the
                            // next attempt retries with a new session. No orphan
                            // is created because start() only persists after a
                            // successful turn.
                            Err(e) => Err(e),
                        }
                    }
                    Some(id) => {
                        // Subsequent message — resume the existing session.
                        runner.resume(id, &input).await
                    }
                };

                let event = match result {
                    Ok(msg) => BackendEvent::AssistantMessage(msg),
                    Err(e) => BackendEvent::Error(format!("{e:#}")),
                };

                // If the event channel is closed the GUI is gone — exit.
                if evt_tx.send(event).is_err() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use app::fake::{FakeLlmProvider, FakeSessionStore};
    use domain::Role;

    use super::*;

    /// Helper: wire up a backend with fakes and return the channel endpoints
    /// plus a join handle. The caller drives the test by sending commands and
    /// receiving events through the channels.
    fn start_backend(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
    ) -> (
        mpsc::UnboundedSender<BackendCommand>,
        mpsc::UnboundedReceiver<BackendEvent>,
        tokio::task::JoinHandle<()>,
    ) {
        let runner = SessionRunner::new(llm, store);
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (evt_tx, evt_rx) = mpsc::unbounded_channel();

        let handle = tokio::spawn(run_backend(runner, cmd_rx, evt_tx, "/test/project".into()));

        (cmd_tx, evt_rx, handle)
    }

    #[tokio::test]
    async fn first_message_creates_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "Hello".into(),
            })
            .unwrap();

        let event = evt_rx.recv().await.unwrap();
        match event {
            BackendEvent::AssistantMessage(msg) => {
                assert_eq!(msg.role, Role::Assistant);
                assert_eq!(msg.text(), "Hello back!");
            }
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn second_message_resumes_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("first reply");
        llm.push_text("second reply");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store);

        // First message — triggers start()
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let event1 = evt_rx.recv().await.unwrap();
        match &event1 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "first reply"),
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
        }

        // Second message — triggers resume()
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "followup".into(),
            })
            .unwrap();
        let event2 = evt_rx.recv().await.unwrap();
        match &event2 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "second reply"),
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn llm_error_becomes_error_event() {
        let llm = FakeLlmProvider::new();
        llm.push_error("model overloaded");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();

        let event = evt_rx.recv().await.unwrap();
        match event {
            BackendEvent::Error(msg) => {
                assert!(msg.contains("model overloaded"), "error was: {msg}");
            }
            BackendEvent::AssistantMessage(_) => {
                panic!("expected Error, got AssistantMessage")
            }
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn error_on_first_message_allows_retry() {
        // Verifies that a failed start() leaves session_id as None so the
        // next SendMessage retries with a fresh session instead of calling
        // resume() with an ID that was never persisted.
        let llm = FakeLlmProvider::new();
        llm.push_error("connection refused");
        llm.push_text("recovered!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store);

        // First message fails.
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let event1 = evt_rx.recv().await.unwrap();
        assert!(
            matches!(&event1, BackendEvent::Error(_)),
            "expected Error on first attempt"
        );

        // Second message should retry via start(), not resume().
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello again".into(),
            })
            .unwrap();
        let event2 = evt_rx.recv().await.unwrap();
        match &event2 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "recovered!"),
            BackendEvent::Error(e) => panic!("expected recovery, got Error: {e}"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn channel_close_exits_cleanly() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store);

        // Drop the command sender immediately — backend should exit
        // without panicking or hanging.
        drop(cmd_tx);
        handle.await.unwrap();
    }
}
