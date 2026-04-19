//! Per-session runtime state shared by HTTP handlers and the pump task.
//!
//! An `ActiveSession` bundles the bits a single session needs:
//!
//! - `agent`: a `Mutex<AgentClient>` holding the subprocess's stdin
//!   pipe. The client owns a `kill_on_drop` handle on the child, so
//!   dropping the `ActiveSession` (or swapping the client via
//!   [`ActiveSession::replace_agent`]) kills the old agent.
//! - `history`: every `AgentEvent` the agent has emitted, for replay on
//!   SSE subscribe.
//! - `tx`: broadcast channel for live fan-out to SSE subscribers.
//! - `runtime`: the pure [`SessionRuntime`] state machine — used for
//!   the `begin_send` double-send guard.
//! - `alive`: flipped to false when the event stream closes, so
//!   handlers can return 410 without probing the agent.
//! - `fresh`: `true` for sessions created with no resume target. The
//!   slug-rename hook reads this to fire exactly once per session, and
//!   flips it to `false` right before respawning the agent so the new
//!   pump does not re-fire.
//! - `close_sink`: shared handle the pump calls when an
//!   `AgentEvent::RequestClose` frame arrives. Production wires the
//!   lifecycle coordinator through this; tests can substitute a fake.
//!
//! The **pump task** spawned by [`ActiveSession::start`] (or a later
//! [`ActiveSession::replace_agent`]) owns the `AgentEventStream` and
//! the per-field `Arc`s. It does not hold the `ActiveSession` itself —
//! so when the registry drops its Arc, the session drops, the agent
//! dies, the stream returns `None`, and the pump exits on its own.
//!
//! Locking discipline (important — the plan calls this out explicitly):
//!
//! 1. `history` serializes publishers against subscribers. The pump
//!    acquires it to append + broadcast as one atomic step; subscribe
//!    acquires it to clone history + create a receiver as one atomic
//!    step. This handshake is what guarantees the SSE replay→follow
//!    transition has no gaps and no duplicates.
//! 2. `runtime` serializes the double-send guard against the pump's
//!    state-machine updates. Only held while applying one event or
//!    running `begin_send`. Never held across an `.await`.
//! 3. `agent` and `pump` are held briefly — only to swap or reach the
//!    underlying value. Never across an `.await`.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use agent_host::{AgentClient, CloseRequestSink, FirstTurnSink, SessionRuntime};
use domain::SessionId;
use protocol::AgentEvent;
use tokio::sync::broadcast;

mod methods;
mod pump;

/// How many `AgentEvent`s the broadcast channel buffers per subscriber
/// before a slow consumer is "lagged" and drops intermediate events.
/// High enough that a momentary client stall during a big tool-output
/// frame doesn't drop events, but bounded so a disconnected subscriber
/// can't retain arbitrarily many events.
const BROADCAST_CAPACITY: usize = 1024;

/// Outcome of [`ActiveSession::send_message`]. Handlers map these to
/// status codes — `Ok` → 204, `AlreadyTurning` → 409, `Dead` → 410.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendOutcome {
    Ok,
    Dead,
    AlreadyTurning,
    Closing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseStart {
    Closing,
    TurnInProgress,
    AlreadyClosing,
}

pub struct ActiveSession {
    /// The session ID as reported by the agent's `Ready` frame. Set
    /// exactly once by the pump the first time it sees a `Ready`.
    /// Handlers and `SessionRegistry::create` poll this before routing.
    session_id: OnceLock<SessionId>,
    /// Send half of the IPC to the subprocess. Held behind a mutex so
    /// the slug-rename flow can swap in a new client without tearing
    /// down the session. Dropping the inner `AgentClient` drops its
    /// `_child`, which triggers `kill_on_drop` on the old subprocess.
    agent: Mutex<AgentClient>,
    history: Arc<Mutex<Vec<AgentEvent>>>,
    tx: broadcast::Sender<AgentEvent>,
    runtime: Arc<Mutex<SessionRuntime>>,
    alive: Arc<AtomicBool>,
    /// True for sessions created without a resume target. Cleared by
    /// the slug-rename hook after the first `TurnComplete` so the
    /// hook fires at most once per session. Shared with the pump so
    /// it can read the flag when it sees terminal frames.
    fresh: Arc<AtomicBool>,
    /// Signals `await_ready` that `session_id` has been populated. Used
    /// during session creation so the handler can block on the first
    /// `Ready` frame before returning the HTTP response.
    ready_notify: Arc<tokio::sync::Notify>,
    /// Handle to the pump task, wrapped in a mutex so `replace_agent`
    /// can abort the current pump and install a replacement without
    /// having to tear down the whole `ActiveSession`. Drop explicitly
    /// aborts the current pump so that `DELETE`-like code paths close
    /// every live SSE subscriber deterministically — without this, we
    /// would have to wait for the agent's stdout to close before the
    /// pump exited and dropped its broadcast sender clone.
    pump: Mutex<tokio::task::AbortHandle>,
    /// Routes `AgentEvent::RequestClose` frames out of the pump and
    /// into the lifecycle coordinator. Retained on the session so
    /// `replace_agent` can pass it into the replacement pump alongside
    /// the other per-session handles.
    close_sink: Arc<dyn CloseRequestSink>,
    /// Routes the first `TurnComplete` of a fresh session into the
    /// lifecycle coordinator so it can slug-rename the worktree and
    /// branch. Retained on the session so `replace_agent` can pass it
    /// into the replacement pump alongside the other per-session
    /// handles.
    first_turn_sink: Arc<dyn FirstTurnSink>,
}

impl Drop for ActiveSession {
    fn drop(&mut self) {
        // Kick the pump so the broadcast channel's last sender drops,
        // which closes every `BroadcastStream` subscribed to this
        // session. The abort also drops the `AgentEventStream` the
        // pump owned, but the `AgentClient` we still hold has already
        // been freed by this point — its `kill_on_drop` child (if any)
        // will SIGKILL the subprocess as the whole struct unwinds.
        self.pump
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .abort();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use agent_host::{
        AgentClient,
        fake::{FakeFirstTurnSink, NoopCloseRequestSink, NoopFirstTurnSink},
    };
    use domain::{ContentBlock, Message};
    use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
    use tokio::io::{BufReader, duplex};
    use tokio::time::timeout;

    use super::*;

    /// Pair of sides for a duplex-backed `ActiveSession`: the
    /// `AgentClient`/`AgentEventStream` go into the session; the writer
    /// and reader are what the test "plays" as the agent.
    fn session_pair(
        fresh: bool,
    ) -> (
        Arc<ActiveSession>,
        tokio::io::DuplexStream,
        BufReader<tokio::io::DuplexStream>,
    ) {
        let (agent_writer, client_reader) = duplex(64 * 1024);
        let (client_writer, agent_reader) = duplex(64 * 1024);
        let (client, stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        let session = ActiveSession::start(
            client,
            stream,
            Arc::new(NoopCloseRequestSink) as Arc<dyn CloseRequestSink>,
            Arc::new(NoopFirstTurnSink) as Arc<dyn FirstTurnSink>,
            fresh,
        );
        (session, agent_writer, BufReader::new(agent_reader))
    }

    /// Build a session wired to a `FakeFirstTurnSink` so the test can
    /// inspect first-turn-complete calls. Otherwise identical to
    /// `session_pair`.
    fn session_pair_with_first_turn_sink(
        fresh: bool,
    ) -> (
        Arc<ActiveSession>,
        tokio::io::DuplexStream,
        BufReader<tokio::io::DuplexStream>,
        Arc<FakeFirstTurnSink>,
    ) {
        let (agent_writer, client_reader) = duplex(64 * 1024);
        let (client_writer, agent_reader) = duplex(64 * 1024);
        let (client, stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        let sink = Arc::new(FakeFirstTurnSink::new());
        let session = ActiveSession::start(
            client,
            stream,
            Arc::new(NoopCloseRequestSink) as Arc<dyn CloseRequestSink>,
            sink.clone() as Arc<dyn FirstTurnSink>,
            fresh,
        );
        (session, agent_writer, BufReader::new(agent_reader), sink)
    }

    /// Ship one event frame to the session's pump.
    async fn ship(writer: &mut tokio::io::DuplexStream, event: &AgentEvent) {
        write_frame(writer, event).await.expect("write event");
    }

    #[tokio::test]
    async fn fresh_flag_is_readable_and_clearable() {
        let (session, _w, _r) = session_pair(true);
        assert!(session.is_fresh(), "expected is_fresh=true");
        session.mark_not_fresh();
        assert!(!session.is_fresh(), "mark_not_fresh should flip the flag");
    }

    #[tokio::test]
    async fn resumed_session_starts_not_fresh() {
        let (session, _w, _r) = session_pair(false);
        assert!(!session.is_fresh());
    }

    #[tokio::test]
    async fn drop_does_not_panic_when_pump_mutex_is_poisoned() {
        let (session, _w, _r) = session_pair(false);
        let poison_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let session = session.clone();
            move || {
                let _guard = session.pump.lock().unwrap();
                panic!("poison session pump mutex");
            }
        }));
        assert!(poison_result.is_err());

        let drop_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(session)));
        assert!(drop_result.is_ok());
    }

    #[tokio::test]
    async fn begin_close_blocks_later_send_without_writing_frame() {
        let (session, _agent_w, mut agent_r) = session_pair(true);

        assert_eq!(session.begin_close(), CloseStart::Closing);
        assert_eq!(
            session.send_message("after close".into()),
            SendOutcome::Closing
        );

        let frame = timeout(
            Duration::from_millis(100),
            read_frame::<_, AgentCommand>(&mut agent_r),
        )
        .await;
        assert!(frame.is_err(), "closing send unexpectedly wrote a frame");

        session.clear_closing();
        assert_eq!(session.send_message("after clear".into()), SendOutcome::Ok);
        let cmd = timeout(
            Duration::from_secs(2),
            read_frame::<_, AgentCommand>(&mut agent_r),
        )
        .await
        .expect("send after clear timed out")
        .expect("read_frame error")
        .expect("agent pipe closed");
        match cmd {
            AgentCommand::SendMessage { input } => assert_eq!(input, "after clear"),
            other => panic!("expected SendMessage after clear, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn replace_agent_preserves_history_and_keeps_subscribers_alive() {
        // Plan R8's acceptance test: a subscriber attached before
        // `replace_agent` must keep receiving events published by the
        // replacement pump, and the history snapshot must include the
        // pre-replacement events.
        let (session, mut agent_w, _agent_r) = session_pair(true);

        // Ship two frames and let the pump drain them.
        let id = SessionId::new_v4();
        ship(
            &mut agent_w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w".into(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::Error {
                message: "pre-1".into(),
            },
        )
        .await;
        // Wait until both frames have been appended to history.
        for _ in 0..50 {
            let (snapshot, _) = session.subscribe();
            if snapshot.len() >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Subscribe — this is the subscriber that must survive the
        // replacement.
        let (snapshot, mut rx) = session.subscribe();
        assert_eq!(snapshot.len(), 2, "expected 2 frames in history");

        // Build the replacement pair and swap it in.
        let (new_agent_writer, new_client_reader) = duplex(64 * 1024);
        let (new_client_writer, new_agent_reader) = duplex(64 * 1024);
        let mut new_agent_reader = BufReader::new(new_agent_reader);
        let (new_client, new_stream) =
            AgentClient::new(BufReader::new(new_client_reader), new_client_writer);
        // Replace, preserving history and tx.
        session.replace_agent(new_client, new_stream);

        // The old agent writer is now orphaned (its reader was the old
        // stream, which the aborted pump owned) — drop it.
        drop(agent_w);

        // Ship a post-replacement event on the new pipe.
        let mut w = new_agent_writer;
        ship(
            &mut w,
            &AgentEvent::Error {
                message: "post-1".into(),
            },
        )
        .await;

        // The pre-existing subscriber must see the new frame.
        let evt = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("no frame on pre-existing subscriber after replace")
            .expect("broadcast closed");
        match evt {
            AgentEvent::Error { message } => assert_eq!(message, "post-1"),
            other => panic!("expected post-1 Error frame, got {other:?}"),
        }

        // History now has three events: Ready + pre-1 + post-1.
        let (snapshot_after, _) = session.subscribe();
        assert_eq!(snapshot_after.len(), 3);

        // `send_message` still goes through the new client — read back
        // one frame to prove the stdin pipe is the new one.
        assert_eq!(session.send_message("hello".into()), SendOutcome::Ok);
        let cmd = timeout(
            Duration::from_secs(2),
            read_frame::<_, AgentCommand>(&mut new_agent_reader),
        )
        .await
        .expect("no command on new agent pipe")
        .expect("read_frame error")
        .expect("EOF on new agent pipe");
        match cmd {
            AgentCommand::SendMessage { input } => assert_eq!(input, "hello"),
            other => panic!("expected SendMessage on new pipe, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn replace_agent_suppresses_resumed_history_replay() {
        let (session, mut agent_w, _agent_r) = session_pair(true);

        let id = SessionId::new_v4();
        let user = Message::user("Hello!!!");
        let assistant = Message::assistant(vec![ContentBlock::Text {
            text: "Hello! I'm ready to help you with your code.".into(),
        }]);

        ship(
            &mut agent_w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w".into(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::MessageAppended {
                message: user.clone(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::MessageAppended {
                message: assistant.clone(),
            },
        )
        .await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;

        for _ in 0..50 {
            let (snapshot, _) = session.subscribe();
            if snapshot.len() >= 4 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let (_snapshot, mut rx) = session.subscribe();
        let (new_agent_writer, new_client_reader) = duplex(64 * 1024);
        let (new_client_writer, _new_agent_reader) = duplex(64 * 1024);
        let (new_client, new_stream) =
            AgentClient::new(BufReader::new(new_client_reader), new_client_writer);
        session.replace_agent(new_client, new_stream);
        drop(agent_w);

        let mut w = new_agent_writer;
        ship(
            &mut w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w/renamed".into(),
            },
        )
        .await;
        ship(
            &mut w,
            &AgentEvent::MessageAppended {
                message: user.clone(),
            },
        )
        .await;
        ship(&mut w, &AgentEvent::MessageAppended { message: assistant }).await;

        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "replacement startup replay should not broadcast duplicate history"
        );

        let (snapshot_after_replay, _) = session.subscribe();
        assert_eq!(
            snapshot_after_replay.len(),
            4,
            "replayed startup frames must not be appended to history"
        );

        ship(
            &mut w,
            &AgentEvent::MessageAppended {
                message: Message::user("next turn"),
            },
        )
        .await;
        let evt = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("live post-replay message was not broadcast")
            .expect("broadcast closed");
        match evt {
            AgentEvent::MessageAppended { message } => assert_eq!(message.text(), "next turn"),
            other => panic!("expected live MessageAppended after replay, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pump_fires_first_turn_sink_on_fresh_session_turn_complete() {
        // The slug-rename hook's plumbing: ship `Ready` + a user
        // `MessageAppended` + `TurnComplete` on a fresh session, and
        // the pump must call `on_first_turn_complete` exactly once with
        // the user message's text. The hook is fire-and-forget, so we
        // poll briefly for the record to appear.
        let (_session, mut agent_w, _agent_r, sink) = session_pair_with_first_turn_sink(true);

        let id = SessionId::new_v4();
        ship(
            &mut agent_w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w".into(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::MessageAppended {
                message: Message::user("make the login form work"),
            },
        )
        .await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;

        // Poll — the hook runs on a `tokio::spawn`, so by the time we
        // return from `ship`, the sink may not yet have observed it.
        for _ in 0..50 {
            if !sink.calls().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let calls = sink.calls();
        assert_eq!(
            calls.len(),
            1,
            "expected exactly one sink call, got {calls:?}"
        );
        assert_eq!(calls[0].0, id);
        assert_eq!(calls[0].1, "make the login form work");
    }

    #[tokio::test]
    async fn pump_does_not_fire_first_turn_sink_on_resumed_session() {
        // A resumed session starts with `fresh=false`. The CAS in the
        // pump must skip the hook every time, no matter how many
        // `TurnComplete` frames arrive.
        let (_session, mut agent_w, _agent_r, sink) = session_pair_with_first_turn_sink(false);

        let id = SessionId::new_v4();
        ship(
            &mut agent_w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w".into(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::MessageAppended {
                message: Message::user("already-resumed message"),
            },
        )
        .await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;

        // Sleep a bit to ensure the pump has had time to process; the
        // sink must still be empty.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            sink.calls().is_empty(),
            "resumed session should never fire the first-turn hook, got {:?}",
            sink.calls()
        );
    }

    #[tokio::test]
    async fn pump_fires_first_turn_sink_at_most_once_even_with_multiple_turn_completes() {
        // Double-fire guard: if two `TurnComplete` frames arrive back to
        // back (shouldn't happen, but the CAS is the invariant), the
        // sink must still see at most one call.
        let (_session, mut agent_w, _agent_r, sink) = session_pair_with_first_turn_sink(true);

        let id = SessionId::new_v4();
        ship(
            &mut agent_w,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: "/w".into(),
            },
        )
        .await;
        ship(
            &mut agent_w,
            &AgentEvent::MessageAppended {
                message: Message::user("hi"),
            },
        )
        .await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;
        ship(&mut agent_w, &AgentEvent::TurnComplete).await;

        // Wait for at least one call, then sleep a bit more to catch
        // any spurious second calls.
        for _ in 0..50 {
            if !sink.calls().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            sink.calls().len(),
            1,
            "expected a single fire, got {:?}",
            sink.calls()
        );
    }

    #[tokio::test]
    async fn replace_agent_re_arms_alive() {
        // After the old pump observes a closed stream, `alive` flips
        // false. A replacement must re-arm it so send_message / cancel
        // don't return `Dead` until the *new* pump ends.
        let (session, agent_w, _agent_r) = session_pair(true);
        drop(agent_w);
        for _ in 0..50 {
            if !session.is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!session.is_alive(), "alive should have flipped after close");

        let (_new_agent_writer, new_client_reader) = duplex(64 * 1024);
        let (new_client_writer, _new_agent_reader) = duplex(64 * 1024);
        let (new_client, new_stream) =
            AgentClient::new(BufReader::new(new_client_reader), new_client_writer);
        session.replace_agent(new_client, new_stream);

        assert!(session.is_alive(), "replace_agent should re-arm alive");
    }
}
