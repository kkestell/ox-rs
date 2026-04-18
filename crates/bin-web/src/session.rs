//! Per-session runtime state shared by HTTP handlers and the pump task.
//!
//! An `ActiveSession` bundles the bits a single session needs:
//!
//! - `agent`: the `AgentClient` holding the subprocess's stdin pipe. The
//!   client owns a `kill_on_drop` handle on the child, so dropping the
//!   `ActiveSession` kills the agent.
//! - `history`: every `AgentEvent` the agent has emitted, for replay on
//!   SSE subscribe.
//! - `tx`: broadcast channel for live fan-out to SSE subscribers.
//! - `runtime`: the pure [`SessionRuntime`] state machine — used for
//!   the `begin_send` double-send guard.
//! - `alive`: flipped to false when the event stream closes, so
//!   handlers can return 410 without probing the agent.
//!
//! The **pump task** spawned by [`ActiveSession::start`] owns the
//! `AgentEventStream` and the per-field `Arc`s. It does not hold the
//! `ActiveSession` itself — so when the registry drops its Arc, the
//! session drops, the agent dies, the stream returns `None`, and the
//! pump exits on its own.
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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use agent_host::{
    AgentClient, AgentEventStream, SessionRuntime, ShouldSend, apply_event, begin_send,
};
use domain::SessionId;
use protocol::{AgentCommand, AgentEvent};
use tokio::sync::broadcast;

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
}

pub struct ActiveSession {
    /// The session ID as reported by the agent's `Ready` frame. Set
    /// exactly once by the pump the first time it sees a `Ready`.
    /// Handlers and `SessionRegistry::create` poll this before routing.
    session_id: OnceLock<SessionId>,
    /// Send half of the IPC to the subprocess. Dropping the `AgentClient`
    /// drops the `_child` it wraps, which triggers `kill_on_drop`. We
    /// therefore never wrap this in an extra `Arc` — exactly one owner
    /// means shutdown is deterministic.
    agent: AgentClient,
    history: Arc<Mutex<Vec<AgentEvent>>>,
    tx: broadcast::Sender<AgentEvent>,
    runtime: Arc<Mutex<SessionRuntime>>,
    alive: Arc<AtomicBool>,
    /// Signals `await_ready` that `session_id` has been populated. Used
    /// during session creation so the handler can block on the first
    /// `Ready` frame before returning the HTTP response.
    ready_notify: Arc<tokio::sync::Notify>,
    /// Handle to the pump task. Dropped in [`ActiveSession::drop`] with
    /// an explicit `abort()` so that `DELETE /api/sessions/:id` closes
    /// every live SSE subscriber deterministically — without this, we
    /// would have to wait for the agent's stdout to close before the
    /// pump exited and dropped its broadcast sender clone.
    pump: tokio::task::AbortHandle,
}

impl Drop for ActiveSession {
    fn drop(&mut self) {
        // Kick the pump so the broadcast channel's last sender drops,
        // which closes every `BroadcastStream` subscribed to this
        // session. The abort also drops the `AgentEventStream` the
        // pump owned, but the `AgentClient` we still hold has already
        // been freed by this point — its `kill_on_drop` child (if any)
        // will SIGKILL the subprocess as the whole struct unwinds.
        self.pump.abort();
    }
}

impl ActiveSession {
    /// Build the session and spawn its pump task. The stream is consumed
    /// by the pump; the client is retained to serve `send_message` and
    /// `cancel`.
    pub fn start(agent: AgentClient, stream: AgentEventStream) -> Arc<Self> {
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        let history = Arc::new(Mutex::new(Vec::new()));
        let runtime = Arc::new(Mutex::new(SessionRuntime::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let ready_notify = Arc::new(tokio::sync::Notify::new());
        let session_id: OnceLock<SessionId> = OnceLock::new();

        // `new_cyclic` gives the pump a `Weak<ActiveSession>` so it can
        // populate `session_id` on the first `Ready` without keeping
        // the session alive. The pump's `AbortHandle` lives inside the
        // session so Drop can kick it on `DELETE /api/sessions/:id`.
        Arc::new_cyclic(|weak| {
            let pump = spawn_pump(
                stream,
                history.clone(),
                tx.clone(),
                runtime.clone(),
                alive.clone(),
                weak.clone(),
                ready_notify.clone(),
            );
            Self {
                session_id,
                agent,
                history,
                tx,
                runtime,
                alive,
                ready_notify,
                pump,
            }
        })
    }

    /// First `Ready` frame from the agent, once the pump has observed
    /// one. Returns `None` if the agent never sent `Ready`.
    #[allow(dead_code)]
    pub fn session_id(&self) -> Option<SessionId> {
        self.session_id.get().copied()
    }

    /// True until the pump observes a closed event stream.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Wait for the agent's first `Ready` frame. Returns `Some(id)`
    /// once observed, or `None` if the stream closed before any
    /// `Ready` arrived.
    pub async fn await_ready(&self) -> Option<SessionId> {
        loop {
            if let Some(id) = self.session_id.get().copied() {
                return Some(id);
            }
            if !self.is_alive() {
                return None;
            }
            // Re-check after notify to avoid a TOCTOU between `get` and
            // `notified().await`. The pump signals `ready_notify` when
            // it populates `session_id` or flips `alive=false`, so
            // waking up means one of the above is now true.
            self.ready_notify.notified().await;
        }
    }

    /// Snapshot + follow handshake. The caller receives the entire
    /// buffered event history plus a live receiver that will carry
    /// every event the pump publishes after this call returns. The
    /// history lock is held across both steps so the pump cannot
    /// publish a new event between the snapshot and the subscribe.
    pub fn subscribe(&self) -> (Vec<AgentEvent>, broadcast::Receiver<AgentEvent>) {
        let hist = self.history.lock().expect("session history mutex poisoned");
        let snapshot = hist.clone();
        let rx = self.tx.subscribe();
        drop(hist);
        (snapshot, rx)
    }

    /// Place a `SendMessage` command on the agent's stdin, guarded by
    /// [`begin_send`] so a second concurrent send while a turn is in
    /// flight returns `AlreadyTurning` without touching the wire.
    pub fn send_message(&self, input: String) -> SendOutcome {
        if !self.is_alive() {
            return SendOutcome::Dead;
        }
        {
            let mut rt = self.runtime.lock().expect("session runtime mutex poisoned");
            match begin_send(&mut rt) {
                ShouldSend::Skip => return SendOutcome::AlreadyTurning,
                ShouldSend::Send => {}
            }
        }
        if self
            .agent
            .send(AgentCommand::SendMessage { input })
            .is_err()
        {
            // Writer task dropped — roll back the `waiting` flag so a
            // subsequent retry isn't silently blocked. We flip `alive`
            // too so handlers report a consistent story: no wire IPC is
            // possible on this session any more.
            let mut rt = self.runtime.lock().expect("session runtime mutex poisoned");
            rt.waiting = false;
            drop(rt);
            self.alive.store(false, Ordering::SeqCst);
            return SendOutcome::Dead;
        }
        SendOutcome::Ok
    }

    /// Place a `Cancel` command on the agent's stdin. Idempotent — if
    /// the agent has already exited, the send silently no-ops; the
    /// handler still returns 204 to match the plan's contract.
    pub fn cancel(&self) {
        let _ = self.agent.send(AgentCommand::Cancel);
    }
}

/// Drive events from the agent's stream into history, the broadcast
/// channel, and the runtime state machine. Runs until the stream
/// closes or until the owning session is dropped and aborts us.
fn spawn_pump(
    mut stream: AgentEventStream,
    history: Arc<Mutex<Vec<AgentEvent>>>,
    tx: broadcast::Sender<AgentEvent>,
    runtime: Arc<Mutex<SessionRuntime>>,
    alive: Arc<AtomicBool>,
    session_weak: std::sync::Weak<ActiveSession>,
    ready_notify: Arc<tokio::sync::Notify>,
) -> tokio::task::AbortHandle {
    let join = tokio::spawn(async move {
        loop {
            let evt = match stream.recv().await {
                Some(e) => e,
                None => break,
            };

            // Record the `Ready` id on the owning session. `set` is a
            // no-op after the first call, so repeated `Ready` frames
            // (the plan allows it) don't clobber the cell.
            if let AgentEvent::Ready { session_id, .. } = &evt
                && let Some(session) = session_weak.upgrade()
            {
                let _ = session.session_id.set(*session_id);
                ready_notify.notify_waiters();
            }

            // Publish step: append to history and fan out on the
            // broadcast channel as one atomic unit. `subscribe()`
            // acquires the same lock, so the snapshot→follow handshake
            // never drops or duplicates events.
            {
                let mut hist = history.lock().expect("session history mutex poisoned");
                hist.push(evt.clone());
                // `send` returns Err(SendError) when there are no
                // subscribers. That's fine — history still holds the
                // event for future subscribers to replay.
                let _ = tx.send(evt.clone());
            }

            // State-machine step: apply the event to `runtime` under
            // its own lock. Held briefly, never across an await.
            {
                let mut rt = runtime.lock().expect("session runtime mutex poisoned");
                apply_event(&mut rt, evt);
            }
        }

        alive.store(false, Ordering::SeqCst);
        // Wake anyone waiting in `await_ready` so they observe the
        // closed stream and return `None` instead of hanging.
        ready_notify.notify_waiters();
        // Dropping `tx` here drops the last owned sender; existing
        // subscribers observe a closed channel on their next `recv`,
        // which SSE interprets as "session ended."
        drop(tx);
    });
    join.abort_handle()
}
