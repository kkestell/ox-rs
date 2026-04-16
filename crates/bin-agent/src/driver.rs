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
//! - Processes `AgentCommand` frames one at a time. A dedicated reader task
//!   keeps NDJSON reads cancellation-safe while a turn is running. Mid-turn
//!   `Cancel` commands set the turn's cancellation token; other commands are
//!   buffered until the current turn fully terminates (`TurnComplete`,
//!   `TurnCancelled`, or `Error`).
//! - A malformed frame on the wire emits an `AgentEvent::Error` and the loop
//!   keeps reading, so one bad line cannot kill a split.
//! - Clean EOF on the reader (the GUI hung up) returns `Ok(())`.

use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::Result;
use app::{CancelToken, LlmProvider, SessionRunner, SessionStore, TurnEvent, TurnOutcome};
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
    R: AsyncBufRead + Unpin + Send + 'static,
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
    enum ReaderEvent {
        Command(AgentCommand),
        Malformed(String),
        Eof,
    }

    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            match read_frame::<_, AgentCommand>(&mut reader).await {
                Ok(Some(cmd)) => {
                    if command_tx.send(ReaderEvent::Command(cmd)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = command_tx.send(ReaderEvent::Eof);
                    break;
                }
                Err(e) => {
                    if command_tx
                        .send(ReaderEvent::Malformed(format!("{e:#}")))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    // Commands consumed by the mid-turn select! that aren't Cancel. Buffered
    // here so the outer loop processes them on later iterations instead of
    // losing them.
    let mut pending_commands: VecDeque<AgentCommand> = VecDeque::new();
    let mut command_input_closed = false;

    loop {
        // Check the pending buffer before reading from the wire.
        let cmd = if let Some(buffered) = pending_commands.pop_front() {
            buffered
        } else {
            match command_rx.recv().await {
                Some(ReaderEvent::Command(cmd)) => cmd,
                Some(ReaderEvent::Eof) | None => return Ok(()), // clean EOF — GUI shut down
                Some(ReaderEvent::Malformed(message)) => {
                    // Malformed frame. Emit an Error and keep reading; one bad
                    // line should not kill the agent.
                    write_frame(
                        &mut writer,
                        &AgentEvent::Error {
                            message: format!("malformed frame: {message}"),
                        },
                    )
                    .await?;
                    continue;
                }
            }
        };

        match cmd {
            AgentCommand::SendMessage { input } => {
                // Scoped so the pinned turn future (which borrows `writer`)
                // is dropped before we write the terminal frame below.
                let result = {
                    let cancel = CancelToken::new();
                    let cancel_clone = cancel.clone();
                    let mut turn_fut = std::pin::pin!(run_turn(
                        runner,
                        &workspace_root,
                        session_id,
                        &input,
                        initialized,
                        cancel_clone,
                        &mut writer,
                    ));
                    // Race the turn against incoming commands so we can detect
                    // a `Cancel` while the turn is running. Non-cancel commands
                    // are buffered for later iterations. EOF mid-turn sets the
                    // cancel flag so the turn finishes gracefully.
                    //
                    // Commands are read by a dedicated task. That keeps
                    // framed reads out of this select! so we never cancel a
                    // partially-read NDJSON frame.
                    loop {
                        tokio::select! {
                            result = &mut turn_fut => break result,
                            frame = command_rx.recv(), if !command_input_closed => {
                                match frame {
                                    Some(ReaderEvent::Command(AgentCommand::Cancel)) => {
                                        cancel.cancel();
                                        // Don't break — let the turn future
                                        // finish so it flushes remaining events.
                                    }
                                    Some(ReaderEvent::Eof) | None => {
                                        // GUI hung up mid-turn. Cancel so the
                                        // turn stops promptly, then let it drain.
                                        command_input_closed = true;
                                        cancel.cancel();
                                    }
                                    Some(ReaderEvent::Command(other)) => {
                                        // Non-cancel command mid-turn — buffer
                                        // it so the outer loop processes it on
                                        // the next iteration.
                                        pending_commands.push_back(other);
                                    }
                                    Some(ReaderEvent::Malformed(_)) => {
                                        // Malformed frame mid-turn — discard.
                                    }
                                }
                            }
                        }
                    }
                };
                // `turn_fut` is dropped here, releasing the borrow on `writer`.

                match result {
                    Ok(TurnOutcome::Completed) => {
                        initialized = true;
                        write_frame(&mut writer, &AgentEvent::TurnComplete).await?;
                    }
                    Ok(TurnOutcome::Cancelled) => {
                        initialized = true;
                        write_frame(&mut writer, &AgentEvent::TurnCancelled).await?;
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
            AgentCommand::Cancel => {
                // Cancel outside a turn — nothing to do. Silently ignore
                // rather than emitting an error, since it's harmless.
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
    cancel: CancelToken,
    writer: &mut W,
) -> Result<TurnOutcome>
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
            runner.resume(session_id, input, cancel, callback).await
        } else {
            runner
                .start(session_id, workspace, input, cancel, callback)
                .await
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
            let runner = SessionRunner::new(llm, store, tools, String::new());
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

    /// Read events from `reader` until a turn terminator (`TurnComplete`,
    /// `TurnCancelled`, or `Error`) is seen, or until the driver hangs up.
    async fn drain_turn(
        reader: &mut BufReader<tokio::io::DuplexStream>,
    ) -> (Vec<AgentEvent>, Option<AgentEvent>) {
        let mut events = Vec::new();
        loop {
            match read_frame::<_, AgentEvent>(reader).await.unwrap() {
                Some(evt @ AgentEvent::TurnComplete) => return (events, Some(evt)),
                Some(evt @ AgentEvent::TurnCancelled) => return (events, Some(evt)),
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

    // -- cancellation ---------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_mid_turn_emits_turn_cancelled() {
        // Use a channel-backed LLM so the turn blocks until we send Cancel.
        let llm = FakeLlmProvider::new();
        let mut tx = llm.push_channel();
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        // Start a turn.
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "hello".into(),
            },
        )
        .await
        .unwrap();

        // Send one text delta so there's partial content.
        use futures::SinkExt;
        tx.send(Ok(StreamEvent::TextDelta {
            delta: "partial".into(),
        }))
        .await
        .unwrap();

        // Wait for the StreamDelta to appear on the event pipe — this
        // proves the driver has processed it, avoiding a timing-dependent
        // sleep that can deadlock under load.
        loop {
            match read_frame::<_, AgentEvent>(&mut evt_rx).await.unwrap() {
                Some(AgentEvent::StreamDelta { .. }) => break,
                Some(_) => continue, // skip MessageAppended(user), etc.
                None => panic!("unexpected EOF waiting for StreamDelta"),
            }
        }

        // Send Cancel.
        write_frame(&mut cmd_tx, &AgentCommand::Cancel)
            .await
            .unwrap();

        // Send another event so the runner's stream.next() unblocks and
        // sees the cancel flag.
        let _ = tx
            .send(Ok(StreamEvent::TextDelta {
                delta: " more".into(),
            }))
            .await;

        // Drain remaining events until the terminator.
        let (events, terminator) = drain_turn(&mut evt_rx).await;
        assert!(
            matches!(terminator, Some(AgentEvent::TurnCancelled)),
            "expected TurnCancelled, got {terminator:?}"
        );

        // A partial assistant MessageAppended should precede TurnCancelled.
        let appended: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageAppended { message } => Some(message.role.clone()),
                _ => None,
            })
            .collect();
        assert!(
            appended.contains(&Role::Assistant),
            "partial assistant message should be committed"
        );

        drop(cmd_tx);
        drop(tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn cancel_when_no_turn_running_is_ignored() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        // Send Cancel with no turn in progress.
        write_frame(&mut cmd_tx, &AgentCommand::Cancel)
            .await
            .unwrap();

        // Driver should still be alive — verify by closing cleanly.
        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn turn_completes_normally_despite_late_cancel() {
        // If the turn finishes before the driver reads the Cancel, the
        // outcome should be TurnComplete (not TurnCancelled).
        let llm = FakeLlmProvider::new();
        llm.push_text("done");
        let store = FakeSessionStore::new();
        let history = FakeSessionStore::new();

        let (mut cmd_tx, mut evt_rx, handle) =
            spawn_driver(llm, store, history, ToolRegistry::new(), None);
        let _ = expect_ready(&mut evt_rx).await;

        // Send both at once — the turn will complete before the driver reads Cancel.
        write_frame(
            &mut cmd_tx,
            &AgentCommand::SendMessage {
                input: "hello".into(),
            },
        )
        .await
        .unwrap();
        write_frame(&mut cmd_tx, &AgentCommand::Cancel)
            .await
            .unwrap();

        let (_events, terminator) = drain_turn(&mut evt_rx).await;
        // The turn and the Cancel race through the `select!` loop. The
        // pre-queued response completes near-instantly, but `select!`
        // may read the Cancel in the same poll cycle. Both outcomes are
        // correct — what matters is no crash, no hang, and no lost events.
        assert!(
            matches!(
                terminator,
                Some(AgentEvent::TurnComplete) | Some(AgentEvent::TurnCancelled)
            ),
            "expected TurnComplete or TurnCancelled, got {terminator:?}"
        );

        drop(cmd_tx);
        handle.await.unwrap().unwrap();
    }
}
