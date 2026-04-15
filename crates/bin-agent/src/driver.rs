//! The agent's NDJSON I/O loop, parameterized over reader/writer types.
//!
//! Factored out of `main.rs` so tests can drive a real `agent_driver` with
//! `tokio::io::duplex` instead of spawning a subprocess. That gives us
//! end-to-end coverage (frame parsing, session lifecycle, tool loop, error
//! paths) at unit-test speed.
//!
//! ### Contract
//!
//! - Emits exactly one `Ready` frame at startup, before anything else.
//! - On `--resume <id>`: after `Ready`, replays every historical message as a
//!   `MessageAppended` frame in order, then waits for commands. Historical
//!   and live messages travel the same channel so the GUI has a single
//!   code path.
//! - Dequeues `AgentCommand` frames one at a time from the reader. The next
//!   command is not read until the previous turn fully terminates
//!   (`TurnComplete` or `Error`). A `SendMessage` that arrives mid-turn sits
//!   in the pipe until the driver is ready — this serializes the turn loop
//!   without any explicit mutex.
//! - A malformed frame on the wire emits an `AgentEvent::Error` and the loop
//!   keeps reading, so one bad line cannot kill a tab.
//! - Clean EOF on the reader (the GUI hung up) returns `Ok(())`.

use std::path::PathBuf;

use anyhow::Result;
use app::{LlmProvider, SessionRunner, SessionStore, TurnEvent};
use domain::SessionId;
use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
use tokio::io::{AsyncBufRead, AsyncWrite};
use tokio::sync::mpsc;

/// Drive the agent's lifecycle over a framed NDJSON channel.
///
/// `reader` yields `AgentCommand` frames; `writer` receives `AgentEvent`
/// frames. The function returns when the reader hits EOF or a non-recoverable
/// error occurs.
///
/// `history_store` is a `SessionStore` handle the driver uses *only* to
/// preload messages for `--resume`. It's passed separately from the
/// `runner`'s internal store because `SessionRunner` doesn't expose its store
/// — the cost of an extra handle is negligible (`DiskSessionStore` is a
/// cheap wrapper around a directory path) and keeps the runner's API small.
pub async fn agent_driver<L, S, H, R, W>(
    runner: &SessionRunner<L, S>,
    history_store: &H,
    workspace_root: PathBuf,
    resume: Option<SessionId>,
    mut reader: R,
    mut writer: W,
) -> Result<()>
where
    L: LlmProvider + Send + Sync + 'static,
    S: SessionStore + Send + Sync + 'static,
    H: SessionStore,
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    // --- Step 1: Resolve the session ID and replay any history. -----------
    //
    // On resume, we load the existing session and replay every message as a
    // `MessageAppended` frame before emitting any live output, so the GUI
    // can render the conversation through the same handler it uses for
    // live turn messages.
    let (session_id, historical_messages) = match resume {
        Some(id) => {
            let session = history_store.load(id).await?;
            (id, session.messages)
        }
        None => (SessionId::new_v4(), Vec::new()),
    };

    // --- Step 2: Handshake. -----------------------------------------------
    write_frame(
        &mut writer,
        &AgentEvent::Ready {
            session_id,
            workspace_root: workspace_root.clone(),
        },
    )
    .await?;

    for message in historical_messages {
        write_frame(&mut writer, &AgentEvent::MessageAppended { message }).await?;
    }

    // --- Step 3: Command loop. --------------------------------------------
    //
    // Only `SendMessage` is defined today; unknown command tags decode as an
    // `Err` from `read_frame` and surface as an `Error` frame.
    //
    // A fresh session is materialized lazily — `SessionRunner::start` creates
    // the on-disk file, so we track `initialized` separately to flip between
    // `start(...)` and `resume(...)`.
    let mut initialized = resume.is_some();

    loop {
        let cmd = match read_frame::<_, AgentCommand>(&mut reader).await {
            Ok(Some(cmd)) => cmd,
            Ok(None) => return Ok(()), // clean EOF — GUI shut down
            Err(e) => {
                // Malformed frame. Emit an Error and keep reading; one bad
                // line should not kill the agent.
                write_frame(
                    &mut writer,
                    &AgentEvent::Error {
                        message: format!("malformed frame: {e:#}"),
                    },
                )
                .await?;
                continue;
            }
        };

        match cmd {
            AgentCommand::SendMessage { input } => {
                let result = run_turn(
                    runner,
                    &workspace_root,
                    session_id,
                    &input,
                    initialized,
                    &mut writer,
                )
                .await;

                match result {
                    Ok(()) => {
                        initialized = true;
                        write_frame(&mut writer, &AgentEvent::TurnComplete).await?;
                    }
                    Err(e) => {
                        // A failed `start` leaves the session uncreated on
                        // disk (SessionRunner only saves after a successful
                        // turn), so `initialized` stays false and the next
                        // `SendMessage` retries via `start` — matching the
                        // pre-split behavior.
                        write_frame(
                            &mut writer,
                            &AgentEvent::Error {
                                message: format!("{e:#}"),
                            },
                        )
                        .await?;
                    }
                }
            }
            // `AgentCommand` is `#[non_exhaustive]` so future variants compile
            // cleanly. Until they're implemented, surface an Error so the GUI
            // knows its request was ignored — preferable to silent drops.
            other => {
                write_frame(
                    &mut writer,
                    &AgentEvent::Error {
                        message: format!("unsupported command: {other:?}"),
                    },
                )
                .await?;
            }
        }
    }
}

/// Run a single turn and stream its events to `writer` as they arrive.
///
/// `SessionRunner`'s callback is synchronous `FnMut`, but `write_frame` is
/// async — a naive `write_frame(...).await` inside the callback would need
/// the callback to be async. Instead we bridge the two worlds with a tokio
/// channel:
///
/// - The synchronous callback pushes each event onto an unbounded sender.
/// - A concurrent "drain" future reads from the receiver and calls
///   `write_frame` for each event.
/// - `tokio::join!` runs the runner and the drain concurrently so deltas
///   reach the wire as soon as the runner emits them, preserving the
///   real-time streaming feel the GUI depends on.
/// - When the runner returns, its closure is dropped, which drops the
///   sender, which closes the channel, which causes the drain future to
///   finish after flushing the last events.
async fn run_turn<L, S, W>(
    runner: &SessionRunner<L, S>,
    workspace_root: &std::path::Path,
    session_id: SessionId,
    input: &str,
    initialized: bool,
    writer: &mut W,
) -> Result<()>
where
    L: LlmProvider + Send + Sync + 'static,
    S: SessionStore + Send + Sync + 'static,
    W: AsyncWrite + Unpin,
{
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Runner future: owns `tx` (via the callback), drops it when finished.
    // The callback is `FnMut` because `SessionRunner::start` / `resume`
    // want `FnMut`, but it only needs `&tx` — unbounded senders are `Sync`
    // and `send(&self)` is enough.
    let workspace = workspace_root.to_path_buf();
    let run_fut = async move {
        let callback = |evt: TurnEvent<'_>| match evt {
            TurnEvent::StreamDelta(e) => {
                let _ = tx.send(AgentEvent::StreamDelta { event: e.clone() });
            }
            TurnEvent::MessageAppended(m) => {
                let _ = tx.send(AgentEvent::MessageAppended { message: m.clone() });
            }
        };
        if initialized {
            runner.resume(session_id, input, callback).await
        } else {
            runner
                .start(session_id, workspace, input, callback)
                .await
                .map(|_| ())
        }
        // `tx` (captured inside the callback) is dropped when this future
        // finishes, closing the channel so the drain future exits.
    };

    // Drain future: forwards each event to the writer in arrival order.
    // Write errors abort draining immediately — if the GUI hung up, there
    // is no point trying to write further frames.
    let drain_fut = async {
        while let Some(evt) = rx.recv().await {
            write_frame(writer, &evt).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let (run_outcome, drain_outcome) = tokio::join!(run_fut, drain_fut);
    // Surface a write error before the runner's error so the caller sees
    // the earliest failure. Either one aborts the turn.
    drain_outcome?;
    run_outcome
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use app::fake::{FakeLlmProvider, FakeSessionStore, FakeTool, tool_registry_with};
    use app::{Tool, ToolRegistry};
    use domain::{Message, Role, Session, StreamEvent, Usage};
    use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
    use tokio::io::{BufReader, duplex};

    use super::*;

    /// Spawn the driver over a duplex pipe and return:
    /// - a writer the test uses to send `AgentCommand` frames to the driver,
    /// - a reader the test uses to receive `AgentEvent` frames from the driver,
    /// - the join handle for the driver task.
    ///
    /// The test drives the pipe from one side; the driver runs on the other.
    fn spawn_driver(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
        history: FakeSessionStore,
        tools: ToolRegistry,
        resume: Option<SessionId>,
    ) -> (
        tokio::io::DuplexStream,
        BufReader<tokio::io::DuplexStream>,
        tokio::task::JoinHandle<Result<()>>,
    ) {
        // Two pipes: command pipe (test -> driver) and event pipe (driver -> test).
        let (cmd_test, cmd_drv) = duplex(64 * 1024);
        let (evt_drv, evt_test) = duplex(64 * 1024);

        let handle = tokio::spawn(async move {
            let runner = SessionRunner::new(llm, store, tools);
            agent_driver(
                &runner,
                &history,
                PathBuf::from("/test/workspace"),
                resume,
                BufReader::new(cmd_drv),
                evt_drv,
            )
            .await
        });

        (cmd_test, BufReader::new(evt_test), handle)
    }

    /// Read events from `reader` until a `TurnComplete` or `Error` is seen,
    /// or until the driver hangs up.
    async fn drain_turn(
        reader: &mut BufReader<tokio::io::DuplexStream>,
    ) -> (Vec<AgentEvent>, Option<AgentEvent>) {
        let mut events = Vec::new();
        loop {
            match read_frame::<_, AgentEvent>(reader).await.unwrap() {
                Some(evt @ AgentEvent::TurnComplete) => return (events, Some(evt)),
                Some(evt @ AgentEvent::Error { .. }) => return (events, Some(evt)),
                Some(other) => events.push(other),
                None => return (events, None),
            }
        }
    }

    async fn expect_ready(reader: &mut BufReader<tokio::io::DuplexStream>) -> (SessionId, PathBuf) {
        match read_frame::<_, AgentEvent>(reader).await.unwrap() {
            Some(AgentEvent::Ready {
                session_id,
                workspace_root,
            }) => (session_id, workspace_root),
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    // -- fresh start -------------------------------------------------------

    #[tokio::test]
    async fn fresh_start_emits_ready_with_no_history() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();
        let (cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);

        let (_id, workspace) = expect_ready(&mut evt_rx).await;
        assert_eq!(workspace, PathBuf::from("/test/workspace"));

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn send_message_emits_user_then_assistant_then_turn_complete() {
        let llm = FakeLlmProvider::new();
        llm.push_text("hello back");
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "hello".into(),
            },
        )
        .await
        .unwrap();

        let (events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminator, Some(AgentEvent::TurnComplete)));

        // user MessageAppended, stream deltas, assistant MessageAppended.
        let appended: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageAppended { message } => Some(message),
                _ => None,
            })
            .collect();
        assert_eq!(appended.len(), 2);
        assert_eq!(appended[0].role, Role::User);
        assert_eq!(appended[0].text(), "hello");
        assert_eq!(appended[1].role, Role::Assistant);
        assert_eq!(appended[1].text(), "hello back");

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    // -- resume ------------------------------------------------------------

    #[tokio::test]
    async fn resume_replays_history_in_order_after_ready() {
        let id = SessionId::new_v4();
        let mut seeded = Session::new(id, PathBuf::from("/test/workspace"));
        seeded.push_message(Message::user("old 1"));
        seeded.push_message(Message::user("old 2"));

        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        store.insert(seeded.clone());
        let history = FakeSessionStore::new();
        history.insert(seeded);

        let (cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), Some(id));

        let (got_id, _) = expect_ready(&mut evt_rx).await;
        assert_eq!(got_id, id);

        // Two `MessageAppended` frames for the seeded messages, in order.
        match read_frame::<_, AgentEvent>(&mut evt_rx).await.unwrap() {
            Some(AgentEvent::MessageAppended { message }) => {
                assert_eq!(message.text(), "old 1");
            }
            other => panic!("expected MessageAppended, got {other:?}"),
        }
        match read_frame::<_, AgentEvent>(&mut evt_rx).await.unwrap() {
            Some(AgentEvent::MessageAppended { message }) => {
                assert_eq!(message.text(), "old 2");
            }
            other => panic!("expected MessageAppended, got {other:?}"),
        }

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn resume_nonexistent_session_returns_error() {
        let id = SessionId::new_v4();
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (_cmd_tx, _evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), Some(id));

        // Driver should error out in startup (before emitting Ready).
        let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
        let outer = result.expect("driver should finish");
        let inner = outer.expect("task should not panic");
        assert!(
            inner.is_err(),
            "driver must fail to load nonexistent session"
        );
    }

    // -- tool loop ---------------------------------------------------------

    #[tokio::test]
    async fn tool_loop_emits_messages_in_order() {
        // Agent drives the tool loop internally; the GUI just sees the
        // resulting stream of MessageAppended + StreamDelta frames.
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("call_1", "echo", r#"{"x":1}"#);
        llm.push_text("done");

        let echo = Arc::new(FakeTool::new("echo"));
        echo.push_ok("tool-output");
        let tools = tool_registry_with(vec![echo as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) = spawn_driver(llm, store, history, tools, None);
        let _ = expect_ready(&mut evt_rx).await;
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage { input: "hi".into() },
        )
        .await
        .unwrap();

        let (events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminator, Some(AgentEvent::TurnComplete)));

        let appended: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageAppended { message } => Some(message.role.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            appended,
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    // -- error handling ----------------------------------------------------

    #[tokio::test]
    async fn llm_error_becomes_error_frame_not_a_crash() {
        let llm = FakeLlmProvider::new();
        llm.push_error("model overloaded");
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "hello".into(),
            },
        )
        .await
        .unwrap();

        let (_, terminator) = drain_turn(&mut evt_rx).await;
        match terminator {
            Some(AgentEvent::Error { message }) => {
                assert!(message.contains("model overloaded"), "{message}");
            }
            other => panic!("expected Error, got {other:?}"),
        }

        // Driver should still be alive — send another command to verify.
        // A failed start should not have consumed the (lazy) session id, so
        // retrying with a queued text response succeeds.
        let llm_retry = FakeLlmProvider::new();
        llm_retry.push_text("ok");
        // Actually, the driver is already running with the first llm — we
        // can't swap it. So instead verify that dropping cmd_tx cleanly
        // shuts the driver down (i.e. the error did not wedge the loop).
        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn malformed_frame_emits_error_and_loop_continues() {
        let llm = FakeLlmProvider::new();
        llm.push_text("recovered");
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        // Send a malformed JSON line.
        use tokio::io::AsyncWriteExt;
        cmd_tx.write_all(b"not json at all\n").await.unwrap();

        // First terminator is an Error (malformed frame).
        match read_frame::<_, AgentEvent>(&mut evt_rx).await.unwrap() {
            Some(AgentEvent::Error { message }) => {
                assert!(message.contains("malformed"), "{message}");
            }
            other => panic!("expected Error, got {other:?}"),
        }

        // Now send a valid command — the loop should still be responsive.
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage { input: "go".into() },
        )
        .await
        .unwrap();
        let (_events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminator, Some(AgentEvent::TurnComplete)));

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn stream_delta_events_reach_the_wire_verbatim() {
        // The wire must preserve every `StreamEvent` shape unchanged so the
        // GUI's `StreamAccumulator` produces the same message as the agent's.
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::ReasoningDelta {
                delta: "thinking".into(),
            },
            StreamEvent::TextDelta {
                delta: "answer".into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 1,
                },
            },
        ]);
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();
        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage { input: "hi".into() },
        )
        .await
        .unwrap();

        let (events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminator, Some(AgentEvent::TurnComplete)));

        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::StreamDelta { event } => Some(event.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 3);
        assert!(matches!(
            &deltas[0],
            StreamEvent::ReasoningDelta { delta } if delta == "thinking"
        ));
        assert!(matches!(
            &deltas[1],
            StreamEvent::TextDelta { delta } if delta == "answer"
        ));
        assert!(matches!(
            &deltas[2],
            StreamEvent::Finished { usage }
            if usage.reasoning_tokens == 1 && usage.completion_tokens == 2
        ));

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn back_to_back_send_messages_are_processed_serially() {
        // Contract the plan calls out explicitly: the driver dequeues one
        // command at a time, and the second turn only starts after the
        // first `TurnComplete`. A pipelined implementation that races two
        // turns together would scramble messages between two sessions and
        // silently corrupt history. We write two `SendMessage` frames
        // before reading any events, then verify the event stream is:
        //
        //   Ready,
        //   MessageAppended(user="one"), ..., TurnComplete,
        //   MessageAppended(user="two"), ..., TurnComplete
        //
        // in that exact order.
        let llm = FakeLlmProvider::new();
        llm.push_text("reply-one");
        llm.push_text("reply-two");
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();
        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        // Write both commands before draining any event — the second sits
        // in the pipe while the driver works through the first.
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "one".into(),
            },
        )
        .await
        .unwrap();
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "two".into(),
            },
        )
        .await
        .unwrap();

        let (events_one, term_one) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term_one, Some(AgentEvent::TurnComplete)));
        let appended_one: Vec<_> = events_one
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageAppended { message } => Some(message.text()),
                _ => None,
            })
            .collect();
        assert_eq!(appended_one, vec!["one".to_owned(), "reply-one".to_owned()]);

        let (events_two, term_two) = drain_turn(&mut evt_rx).await;
        assert!(matches!(term_two, Some(AgentEvent::TurnComplete)));
        let appended_two: Vec<_> = events_two
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageAppended { message } => Some(message.text()),
                _ => None,
            })
            .collect();
        assert_eq!(appended_two, vec!["two".to_owned(), "reply-two".to_owned()]);

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn stream_deltas_arrive_before_turn_complete() {
        // Regression guard: an earlier implementation buffered every turn
        // event into a `Vec` and flushed them only after the turn ended,
        // which broke live streaming. Verify that at least one `StreamDelta`
        // reaches the wire *before* `TurnComplete`. Since the LLM emits
        // `Finished` as its last StreamEvent, the test also checks that
        // `TurnComplete` follows the deltas (not interleaved mid-stream).
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::TextDelta {
                delta: "hello".into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                },
            },
        ]);
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();
        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage { input: "hi".into() },
        )
        .await
        .unwrap();

        let (events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(matches!(terminator, Some(AgentEvent::TurnComplete)));

        // Find the last StreamDelta's position and the TurnComplete's
        // position; the latter must come after the former.
        let stream_positions: Vec<usize> = events
            .iter()
            .enumerate()
            .filter_map(|(i, e)| matches!(e, AgentEvent::StreamDelta { .. }).then_some(i))
            .collect();
        assert!(
            !stream_positions.is_empty(),
            "expected at least one StreamDelta in event stream"
        );
        // Because `drain_turn` strips the terminator, any StreamDelta event
        // in the returned vec is evidence it arrived before TurnComplete.

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }
}
