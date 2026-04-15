use std::path::PathBuf;

use app::{LlmProvider, SessionRunner, SessionStore, StreamEvent, TurnEvent};
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
///
/// The protocol separates *streaming* from *committing*:
/// - `StreamDelta` carries in-flight events (tokens, reasoning, tool-call
///   argument chunks) and drives the live-updating UI.
/// - `MessageAppended` fires once per message committed to the session —
///   the user's input, any intermediate assistant/tool messages produced
///   inside the tool-call loop, and the final assistant reply.
/// - `TurnComplete` signals the turn's end, so the GUI can re-enable input.
/// - `Error` ends the turn abnormally; no `TurnComplete` follows.
pub enum BackendEvent {
    /// An incremental stream event from the LLM (e.g. a text token).
    StreamDelta(StreamEvent),
    /// A message was just appended to the session (user, assistant, or tool).
    /// The GUI treats this as "commit this to history" — drop any live-
    /// streaming accumulator mirror for the same message.
    MessageAppended(Message),
    /// The turn has ended successfully — input can be re-enabled.
    TurnComplete,
    /// An error from the LLM or session layer. No `TurnComplete` follows.
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
                // Closure that forwards each TurnEvent to the GUI. A send
                // failure means the GUI channel is gone — that's handled
                // below by the terminal `evt_tx.send`, so ignore it here.
                let callback = |evt: TurnEvent<'_>| match evt {
                    TurnEvent::StreamDelta(e) => {
                        let _ = evt_tx.send(BackendEvent::StreamDelta(e.clone()));
                    }
                    TurnEvent::MessageAppended(m) => {
                        let _ = evt_tx.send(BackendEvent::MessageAppended(m.clone()));
                    }
                };

                let result = match session_id {
                    None => {
                        // First message — create a fresh session.
                        let id = SessionId::new_v4();
                        match runner
                            .start(id, workspace_root.clone(), &input, callback)
                            .await
                        {
                            Ok(id) => {
                                session_id = Some(id);
                                Ok(())
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
                        runner.resume(id, &input, callback).await
                    }
                };

                let terminal_event = match result {
                    Ok(()) => BackendEvent::TurnComplete,
                    Err(e) => BackendEvent::Error(format!("{e:#}")),
                };

                // If the event channel is closed the GUI is gone — exit.
                if evt_tx.send(terminal_event).is_err() {
                    break;
                }
            }
        }
    }

    session_id
}

#[cfg(test)]
mod tests {
    use app::fake::{FakeLlmProvider, FakeSessionStore, FakeTool, tool_registry_with};
    use app::{Tool, ToolRegistry};
    use domain::{ContentBlock, Role, Session};
    use std::sync::Arc;

    use super::*;

    /// Helper: wire up a backend with fakes and return the channel endpoints
    /// plus a join handle. The caller drives the test by sending commands and
    /// receiving events through the channels.
    fn start_backend(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
        tools: ToolRegistry,
        initial_session_id: Option<SessionId>,
    ) -> (
        mpsc::UnboundedSender<BackendCommand>,
        mpsc::UnboundedReceiver<BackendEvent>,
        tokio::task::JoinHandle<Option<SessionId>>,
    ) {
        let runner = SessionRunner::new(llm, store, tools);
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

    /// Pump events until we see `TurnComplete` or `Error`. Returns the
    /// stream deltas, the appended messages, and the terminal event.
    async fn drain_turn(
        evt_rx: &mut mpsc::UnboundedReceiver<BackendEvent>,
    ) -> (Vec<StreamEvent>, Vec<Message>, BackendEvent) {
        let mut deltas = Vec::new();
        let mut appended = Vec::new();
        loop {
            let event = evt_rx.recv().await.expect("channel closed unexpectedly");
            match event {
                BackendEvent::StreamDelta(se) => deltas.push(se),
                BackendEvent::MessageAppended(m) => appended.push(m),
                terminal @ (BackendEvent::TurnComplete | BackendEvent::Error(_)) => {
                    return (deltas, appended, terminal);
                }
            }
        }
    }

    #[tokio::test]
    async fn first_message_creates_session_and_appends_user_then_assistant() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "Hello".into(),
            })
            .unwrap();

        let (_deltas, appended, terminal) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminal, BackendEvent::TurnComplete));
        assert_eq!(appended.len(), 2);
        assert_eq!(appended[0].role, Role::User);
        assert_eq!(appended[0].text(), "Hello");
        assert_eq!(appended[1].role, Role::Assistant);
        assert_eq!(appended[1].text(), "Hello back!");

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn second_message_resumes_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("first reply");
        llm.push_text("second reply");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        // First turn.
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let (_, appended1, term1) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term1, BackendEvent::TurnComplete));
        assert_eq!(appended1[1].text(), "first reply");

        // Second turn resumes.
        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "followup".into(),
            })
            .unwrap();
        let (_, appended2, term2) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term2, BackendEvent::TurnComplete));
        assert_eq!(appended2[1].text(), "second reply");

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn initial_session_id_resumes_on_first_message() {
        let id = SessionId::new_v4();
        let store = FakeSessionStore::new();
        store.insert(Session::new(id, "/test/project".into()));

        let llm = FakeLlmProvider::new();
        llm.push_text("resumed!");

        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), Some(id));

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "continue".into(),
            })
            .unwrap();

        let (_, appended, term) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term, BackendEvent::TurnComplete));
        assert_eq!(appended[1].text(), "resumed!");

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, Some(id));
    }

    #[tokio::test]
    async fn returns_none_when_no_message_sent() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, None);
    }

    #[tokio::test]
    async fn returns_session_id_after_successful_message() {
        let llm = FakeLlmProvider::new();
        llm.push_text("hello!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();
        let _ = drain_turn(&mut evt_rx).await;

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert!(final_id.is_some());
    }

    #[tokio::test]
    async fn resumed_session_returns_id_even_without_message() {
        let id = SessionId::new_v4();
        let store = FakeSessionStore::new();
        store.insert(Session::new(id, "/test/project".into()));

        let llm = FakeLlmProvider::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), Some(id));

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, Some(id));
    }

    #[tokio::test]
    async fn resume_nonexistent_session_returns_error_and_keeps_id() {
        let id = SessionId::new_v4();
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();

        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), Some(id));

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        let (_, _, term) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term, BackendEvent::Error(_)));

        drop(cmd_tx);
        let final_id = handle.await.unwrap();
        assert_eq!(final_id, Some(id));
    }

    #[tokio::test]
    async fn llm_error_becomes_error_event_with_no_stream_deltas() {
        let llm = FakeLlmProvider::new();
        llm.push_error("model overloaded");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();

        let (deltas, appended, term) = drain_turn(&mut evt_rx).await;
        assert!(deltas.is_empty());
        // The user message IS appended before the LLM call — that matches
        // `run_turn`'s "commit user input first, then stream" ordering.
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].role, Role::User);
        match term {
            BackendEvent::Error(msg) => assert!(msg.contains("model overloaded"), "{msg}"),
            _ => panic!("expected Error"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn error_on_first_message_allows_retry() {
        let llm = FakeLlmProvider::new();
        llm.push_error("connection refused");
        llm.push_text("recovered!");
        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();
        let (_, _, term1) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term1, BackendEvent::Error(_)));

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello again".into(),
            })
            .unwrap();
        let (_, appended, term2) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term2, BackendEvent::TurnComplete));
        assert_eq!(appended[1].text(), "recovered!");

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn streaming_events_arrive_before_assistant_message_appended() {
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
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        // Order of events on the wire: user MessageAppended, N StreamDeltas,
        // assistant MessageAppended, TurnComplete.
        let mut seen_assistant_msg_after_deltas = false;
        let mut delta_count = 0;
        let mut saw_user_msg = false;
        loop {
            let evt = evt_rx.recv().await.unwrap();
            match evt {
                BackendEvent::StreamDelta(_) => {
                    assert!(
                        saw_user_msg,
                        "StreamDelta arrived before user MessageAppended"
                    );
                    delta_count += 1;
                }
                BackendEvent::MessageAppended(m) => match m.role {
                    Role::User => {
                        saw_user_msg = true;
                        assert_eq!(delta_count, 0, "user msg arrived after deltas");
                    }
                    Role::Assistant => {
                        assert_eq!(delta_count, 3, "assistant msg arrived before all deltas");
                        seen_assistant_msg_after_deltas = true;
                    }
                    Role::Tool => panic!("no tool messages expected"),
                },
                BackendEvent::TurnComplete => break,
                BackendEvent::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(seen_assistant_msg_after_deltas);

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn tool_call_round_trip_surfaces_messages_in_order() {
        // End-to-end-ish: model calls a tool, we execute it, model emits a
        // final reply. The backend should forward every message to the GUI
        // as a `MessageAppended` in chronological order.
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("call_1", "echo", r#"{"x":1}"#);
        llm.push_text("done");

        let echo = Arc::new(FakeTool::new("echo"));
        echo.push_ok("tool-output");
        let tools = tool_registry_with(vec![echo as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, tools, None);

        cmd_tx
            .send(BackendCommand::SendMessage { input: "hi".into() })
            .unwrap();

        let (_deltas, appended, term) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term, BackendEvent::TurnComplete));

        // user → asst(tool-call) → tool result → asst("done")
        assert_eq!(appended.len(), 4);
        assert_eq!(appended[0].role, Role::User);
        assert_eq!(appended[1].role, Role::Assistant);
        assert_eq!(appended[2].role, Role::Tool);
        assert_eq!(appended[3].role, Role::Assistant);
        assert_eq!(appended[3].text(), "done");

        // Tool-result message carries the tool's output.
        match &appended[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert_eq!(content, "tool-output");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
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
        let (cmd_tx, mut evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        cmd_tx
            .send(BackendCommand::SendMessage {
                input: "hello".into(),
            })
            .unwrap();

        let (deltas, _appended, term) = drain_turn(&mut evt_rx).await;
        assert_eq!(deltas.len(), 1);
        assert!(matches!(&deltas[0], StreamEvent::TextDelta(s) if s == "partial"));
        match term {
            BackendEvent::Error(msg) => assert!(msg.contains("stream interrupted"), "{msg}"),
            _ => panic!("expected Error"),
        }

        drop(cmd_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn channel_close_exits_cleanly() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let (cmd_tx, _evt_rx, handle) = start_backend(llm, store, ToolRegistry::new(), None);

        drop(cmd_tx);
        handle.await.unwrap();
    }
}
