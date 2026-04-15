use std::path::PathBuf;

use app::{LlmProvider, SessionRunner, SessionStore, StreamEvent};
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
    /// An incremental stream event from the LLM (e.g. a text token). Sent
    /// for each event as it arrives, before the final message is assembled.
    StreamDelta(StreamEvent),
    /// The completed assistant message, sent after the stream finishes.
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
    initial_session_id: Option<SessionId>,
) -> Option<SessionId>
where
    L: LlmProvider + Send + Sync + 'static,
    S: SessionStore + Send + Sync + 'static,
{
    let mut session_id: Option<SessionId> = initial_session_id;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCommand::SendMessage { input } => {
                // Closure that forwards each stream event to the GUI as a
                // StreamDelta. Ignores send failures — if the GUI is gone,
                // the final send below will detect it and exit the loop.
                let stream_callback = |event: &StreamEvent| {
                    let _ = evt_tx.send(BackendEvent::StreamDelta(event.clone()));
                };

                let result = match session_id {
                    None => {
                        // First message — create a fresh session.
                        let id = SessionId::new_v4();
                        match runner
                            .start(id, workspace_root.clone(), &input, stream_callback)
                            .await
                        {
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
                        runner.resume(id, &input, stream_callback).await
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

    session_id
}

#[cfg(test)]
mod tests {
    use app::fake::{FakeLlmProvider, FakeSessionStore};
    use domain::{Role, Session};

    use super::*;

    /// Helper: wire up a backend with fakes and return the channel endpoints
    /// plus a join handle. The caller drives the test by sending commands and
    /// receiving events through the channels.
    fn start_backend(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
        initial_session_id: Option<SessionId>,
    ) -> (
        mpsc::UnboundedSender<BackendCommand>,
        mpsc::UnboundedReceiver<BackendEvent>,
        tokio::task::JoinHandle<Option<SessionId>>,
    ) {
        let runner = SessionRunner::new(llm, store);
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (evt_tx, evt_rx) = mpsc::unbounded_channel();

        let handle = tokio::spawn(run_backend(
            runner,
            cmd_rx,
            evt_tx,
            "/test/project".into(),
            initial_session_id,
        ));

        (cmd_tx, evt_rx, handle)
    }

    /// Receive events until we hit a non-StreamDelta event, returning both the
    /// collected deltas and the terminal event (AssistantMessage or Error).
    async fn recv_past_deltas(
        evt_rx: &mut mpsc::UnboundedReceiver<BackendEvent>,
    ) -> (Vec<StreamEvent>, BackendEvent) {
        let mut deltas = Vec::new();
        loop {
            let event = evt_rx.recv().await.expect("channel closed unexpectedly");
            match event {
                BackendEvent::StreamDelta(se) => deltas.push(se),
                other => return (deltas, other),
            }
        }
    }

    #[tokio::test]
    async fn first_message_creates_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "Hello".into(),
            })
            .unwrap();

        let (_deltas, event) = recv_past_deltas(&mut evt_rx).await;
        match event {
            BackendEvent::AssistantMessage(msg) => {
                assert_eq!(msg.role, Role::Assistant);
                assert_eq!(msg.text(), "Hello back!");
            }
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
            _ => panic!("unexpected event"),
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
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        // First message — triggers start()
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let (_deltas, event1) = recv_past_deltas(&mut evt_rx).await;
        match &event1 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "first reply"),
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
            _ => panic!("unexpected event"),
        }

        // Second message — triggers resume()
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "followup".into(),
            })
            .unwrap();
        let (_deltas, event2) = recv_past_deltas(&mut evt_rx).await;
        match &event2 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "second reply"),
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
            _ => panic!("unexpected event"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn initial_session_id_resumes_on_first_message() {
        // Seed the store with a session so resume() can find it.
        let id = SessionId::new_v4();
        let store = FakeSessionStore::new();
        store.insert(Session::new(id, "/test/project".into()));

        let llm = FakeLlmProvider::new();
        llm.push_text("resumed!");

        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, Some(id));

        // First message should call resume(), not start().
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "continue".into(),
            })
            .unwrap();

        let (_deltas, event) = recv_past_deltas(&mut evt_rx).await;
        match event {
            BackendEvent::AssistantMessage(msg) => {
                assert_eq!(msg.text(), "resumed!");
            }
            BackendEvent::Error(e) => panic!("expected AssistantMessage, got Error: {e}"),
            _ => panic!("unexpected event"),
        }

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, Some(id));
    }

    #[tokio::test]
    async fn returns_none_when_no_message_sent() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, None);

        // Close immediately without sending any message.
        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, None);
    }

    #[tokio::test]
    async fn returns_session_id_after_successful_message() {
        let llm = FakeLlmProvider::new();
        llm.push_text("hello!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();
        let _ = recv_past_deltas(&mut evt_rx).await;

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert!(final_id.is_some());
    }

    #[tokio::test]
    async fn resumed_session_returns_id_even_without_message() {
        // When resuming an existing session and the user closes without
        // sending a message, the session ID should still be returned since
        // the session exists on disk.
        let id = SessionId::new_v4();
        let store = FakeSessionStore::new();
        store.insert(Session::new(id, "/test/project".into()));

        let llm = FakeLlmProvider::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, Some(id));

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, Some(id));
    }

    #[tokio::test]
    async fn resume_nonexistent_session_returns_error_and_keeps_id() {
        // When the backend is started with a session ID that doesn't exist
        // in the store, resume() fails. The session ID should stay — we
        // don't fall back to creating a new session for an explicit resume.
        let id = SessionId::new_v4();
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new(); // deliberately empty

        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, Some(id));

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        let (_deltas, event) = recv_past_deltas(&mut evt_rx).await;
        assert!(
            matches!(&event, BackendEvent::Error(_)),
            "expected Error for non-existent session"
        );

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(
            final_id,
            Some(id),
            "session ID should be preserved after resume failure"
        );
    }

    #[tokio::test]
    async fn llm_error_becomes_error_event() {
        let llm = FakeLlmProvider::new();
        llm.push_error("model overloaded");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();

        // Error occurs at stream creation time — no StreamDelta events
        // should precede the Error event.
        let (deltas, event) = recv_past_deltas(&mut evt_rx).await;
        assert!(deltas.is_empty(), "expected no StreamDelta before error");
        match event {
            BackendEvent::Error(msg) => {
                assert!(msg.contains("model overloaded"), "error was: {msg}");
            }
            _ => panic!("expected Error event"),
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
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        // First message fails.
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let (_deltas, event1) = recv_past_deltas(&mut evt_rx).await;
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
        let (_deltas, event2) = recv_past_deltas(&mut evt_rx).await;
        match &event2 {
            BackendEvent::AssistantMessage(msg) => assert_eq!(msg.text(), "recovered!"),
            BackendEvent::Error(e) => panic!("expected recovery, got Error: {e}"),
            _ => panic!("unexpected event"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn streaming_events_arrive_before_final_message() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::TextDelta("Hello, ".into()),
            StreamEvent::TextDelta("world!".into()),
            StreamEvent::Finished {
                usage: app::Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        let (deltas, final_event) = recv_past_deltas(&mut evt_rx).await;

        // Three StreamDelta events: TextDelta, TextDelta, Finished.
        assert_eq!(deltas.len(), 3);
        assert!(matches!(&deltas[0], StreamEvent::TextDelta(s) if s == "Hello, "));
        assert!(matches!(&deltas[1], StreamEvent::TextDelta(s) if s == "world!"));
        assert!(matches!(&deltas[2], StreamEvent::Finished { .. }));

        // Followed by the final assembled AssistantMessage.
        match final_event {
            BackendEvent::AssistantMessage(msg) => {
                assert_eq!(msg.text(), "Hello, world!");
            }
            _ => panic!("expected AssistantMessage after deltas"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mid_stream_error_sends_deltas_then_error() {
        let llm = FakeLlmProvider::new();
        llm.push_error_after(
            vec![StreamEvent::TextDelta("partial".into())],
            "stream interrupted",
        );
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        let (deltas, final_event) = recv_past_deltas(&mut evt_rx).await;

        // The TextDelta arrived before the error.
        assert_eq!(deltas.len(), 1);
        assert!(matches!(&deltas[0], StreamEvent::TextDelta(s) if s == "partial"));

        // Terminal event is an Error, not an AssistantMessage.
        match final_event {
            BackendEvent::Error(msg) => {
                assert!(msg.contains("stream interrupted"), "error was: {msg}");
            }
            _ => panic!("expected Error after mid-stream failure"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn channel_close_exits_cleanly() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, None);

        // Drop the command sender immediately — backend should exit
        // without panicking or hanging.
        drop(cmd_tx);
        handle.await.unwrap();
    }
}
