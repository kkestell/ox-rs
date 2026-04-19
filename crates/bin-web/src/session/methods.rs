use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use agent_host::{
    AgentClient, AgentEventStream, BeginClose, CloseRequestSink, FirstTurnSink, SessionRuntime,
    ShouldSend, begin_close, begin_send, clear_closing,
};
use domain::SessionId;
use protocol::AgentCommand;
use tokio::sync::broadcast;

use super::pump::spawn_pump;
use super::{ActiveSession, BROADCAST_CAPACITY, CloseStart, SendOutcome};

impl ActiveSession {
    /// Build the session and spawn its pump task. The stream is consumed
    /// by the pump; the client is retained to serve `send_message` and
    /// `cancel`.
    ///
    /// `fresh` is `true` for brand-new sessions (no resume target) and
    /// `false` for resumed ones; the slug-rename hook uses this to fire
    /// exactly once per new session. `close_sink` is the handle the
    /// pump will call on `RequestClose` frames. `first_turn_sink` is the
    /// handle the pump will call on the first `TurnComplete` of a fresh
    /// session, so the lifecycle coordinator can slug-rename the
    /// worktree + branch.
    pub fn start(
        agent: AgentClient,
        stream: AgentEventStream,
        close_sink: Arc<dyn CloseRequestSink>,
        first_turn_sink: Arc<dyn FirstTurnSink>,
        fresh: bool,
    ) -> Arc<Self> {
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        let history = Arc::new(Mutex::new(Vec::new()));
        let runtime = Arc::new(Mutex::new(SessionRuntime::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let fresh = Arc::new(AtomicBool::new(fresh));
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
                first_turn_sink.clone(),
                close_sink.clone(),
                false,
            );
            Self {
                session_id,
                agent: Mutex::new(agent),
                history,
                tx,
                runtime,
                alive,
                fresh,
                ready_notify,
                pump: Mutex::new(pump),
                close_sink,
                first_turn_sink,
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

    /// True for sessions that have not yet received their first
    /// `TurnComplete`. The slug-rename hook reads this and flips it
    /// to false exactly once.
    #[allow(dead_code)]
    pub fn is_fresh(&self) -> bool {
        self.fresh.load(Ordering::SeqCst)
    }

    /// Clear the `fresh` flag. Called by the slug-rename hook before
    /// it calls `replace_agent`, so the new pump observes
    /// `is_fresh() == false` from the start.
    #[allow(dead_code)]
    pub fn mark_not_fresh(&self) {
        self.fresh.store(false, Ordering::SeqCst);
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

    /// Atomically publish an `AgentEvent::Error` on this session:
    /// append to history and fan out on the broadcast channel as one
    /// unit under the shared history lock. The lifecycle coordinator
    /// calls this when an agent-initiated close (merge / abandon tool)
    /// is refused, so live SSE subscribers and later replay both see
    /// the rejection message.
    ///
    /// The session is left untouched beyond the error frame. Per the
    /// plan, an agent that emitted `RequestClose` is about to exit
    /// anyway; the registry entry goes dead naturally once the pump
    /// observes the subprocess's EOF.
    pub fn broadcast_error(&self, message: String) {
        let evt = protocol::AgentEvent::Error { message };
        let mut hist = self.history.lock().expect("session history mutex poisoned");
        hist.push(evt.clone());
        let _ = self.tx.send(evt);
    }

    /// Snapshot + follow handshake. The caller receives the entire
    /// buffered event history plus a live receiver that will carry
    /// every event the pump publishes after this call returns. The
    /// history lock is held across both steps so the pump cannot
    /// publish a new event between the snapshot and the subscribe.
    pub fn subscribe(
        &self,
    ) -> (
        Vec<protocol::AgentEvent>,
        broadcast::Receiver<protocol::AgentEvent>,
    ) {
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
                ShouldSend::Closing => return SendOutcome::Closing,
                ShouldSend::Send => {}
            }
        }
        let send_result = {
            let agent = self.agent.lock().expect("session agent mutex poisoned");
            agent.send(AgentCommand::SendMessage { input })
        };
        if send_result.is_err() {
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
        let agent = self.agent.lock().expect("session agent mutex poisoned");
        let _ = agent.send(AgentCommand::Cancel);
    }

    pub fn resolve_tool_approval(&self, request_id: String, approved: bool) -> SendOutcome {
        if !self.is_alive() {
            return SendOutcome::Dead;
        }
        let send_result = {
            let agent = self.agent.lock().expect("session agent mutex poisoned");
            agent.send(AgentCommand::ResolveToolApproval {
                request_id,
                approved,
            })
        };
        if send_result.is_err() {
            self.alive.store(false, Ordering::SeqCst);
            return SendOutcome::Dead;
        }
        SendOutcome::Ok
    }

    /// Snapshot of the receive-side runtime's "turn in flight" flag.
    /// Used by the merge/abandon flow to refuse a close while the agent
    /// is mid-turn: tearing a worktree out from under an active tool
    /// call would surface as "no such file" on the next filesystem op
    /// and leave the transcript in a confusing half-state. Cheap read —
    /// takes the runtime mutex briefly and never across an await.
    #[allow(dead_code)]
    pub fn is_turn_in_progress(&self) -> bool {
        let rt = self.runtime.lock().expect("session runtime mutex poisoned");
        rt.is_turn_in_progress()
    }

    /// Mark the session as closing only if it is idle. Once set, later
    /// `send_message` calls are rejected before they can enqueue a command.
    pub fn begin_close(&self) -> CloseStart {
        let mut rt = self.runtime.lock().expect("session runtime mutex poisoned");
        match begin_close(&mut rt) {
            BeginClose::Closing => CloseStart::Closing,
            BeginClose::TurnInProgress => CloseStart::TurnInProgress,
            BeginClose::AlreadyClosing => CloseStart::AlreadyClosing,
        }
    }

    /// Release the closing marker after a rejected close preflight.
    pub fn clear_closing(&self) {
        let mut rt = self.runtime.lock().expect("session runtime mutex poisoned");
        clear_closing(&mut rt);
    }

    /// Swap the agent subprocess without tearing down the session.
    ///
    /// The slug-rename flow uses this: after renaming the branch and
    /// moving the worktree, we spawn a new `ox-agent` with
    /// `--workspace-root=<new-path>` and `--resume=<session-id>`, then
    /// call `replace_agent` so live SSE subscribers keep their
    /// connection and the broadcast / history state persists across
    /// the respawn.
    ///
    /// Steps (in order, matching the plan's R8 contract):
    /// 1. Abort the current pump (drops its clone of `tx`, but `self.tx`
    ///    still holds a clone, so existing subscribers stay open).
    /// 2. Replace the stored `AgentClient`; dropping the old one triggers
    ///    `kill_on_drop` on the old subprocess.
    /// 3. Re-arm `alive` — the old pump set it `false` when its stream
    ///    ended.
    /// 4. Spawn a fresh pump over `new_stream`, sharing `history`, `tx`,
    ///    `runtime`, `alive`, `fresh`, and `ready_notify`.
    ///
    /// Idempotency vs. the slug-rename hook is handled upstream in
    /// `spawn_pump`: the CAS-flip of `fresh` happens *before* the
    /// coordinator is called, so the replacement pump spawned here sees
    /// `fresh=false` and cannot re-fire the hook on its first
    /// `TurnComplete`.
    pub fn replace_agent(self: &Arc<Self>, new_client: AgentClient, new_stream: AgentEventStream) {
        // Abort the current pump first so it stops appending to
        // history before we re-arm `alive`. Abort is cooperative —
        // the task will drop on its next await point.
        self.pump
            .lock()
            .expect("session pump mutex poisoned")
            .abort();

        // Swap the client under the mutex. The old client's `Drop`
        // runs as soon as we release the mutex — its `_child` handle
        // has `kill_on_drop(true)`, so the previous subprocess dies.
        {
            let mut agent = self.agent.lock().expect("session agent mutex poisoned");
            *agent = new_client;
        }

        // Re-arm `alive` before spawning the new pump so any handler
        // observing the atomic between these two operations sees the
        // session as live. The new pump will flip it to false only
        // when its stream closes in turn.
        self.alive.store(true, Ordering::SeqCst);

        let new_pump = spawn_pump(
            new_stream,
            self.history.clone(),
            self.tx.clone(),
            self.runtime.clone(),
            self.alive.clone(),
            Arc::downgrade(self),
            self.ready_notify.clone(),
            self.first_turn_sink.clone(),
            self.close_sink.clone(),
            true,
        );
        *self.pump.lock().expect("session pump mutex poisoned") = new_pump;
    }
}
