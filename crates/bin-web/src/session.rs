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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use agent_host::{
    AgentClient, AgentEventStream, BeginClose, CloseRequestSink, FirstTurnSink, SessionRuntime,
    ShouldSend, apply_event, begin_close, begin_send, clear_closing,
};
use domain::{ContentBlock, Role, SessionId};
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
            .expect("session pump mutex poisoned")
            .abort();
    }
}

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
        let evt = AgentEvent::Error { message };
        let mut hist = self.history.lock().expect("session history mutex poisoned");
        hist.push(evt.clone());
        let _ = self.tx.send(evt);
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

/// Drive events from the agent's stream into history, the broadcast
/// channel, and the runtime state machine. Runs until the stream
/// closes or until the owning session is dropped and aborts us.
///
/// Nine handles is a lot, but they all thread together as one unit and
/// grouping them under a struct just renames the problem — the struct
/// would have no other use and every callsite would populate every
/// field. Keep them as named parameters for readability.
#[allow(clippy::too_many_arguments)]
fn spawn_pump(
    mut stream: AgentEventStream,
    history: Arc<Mutex<Vec<AgentEvent>>>,
    tx: broadcast::Sender<AgentEvent>,
    runtime: Arc<Mutex<SessionRuntime>>,
    alive: Arc<AtomicBool>,
    session_weak: std::sync::Weak<ActiveSession>,
    ready_notify: Arc<tokio::sync::Notify>,
    first_turn_sink: Arc<dyn FirstTurnSink>,
    close_sink: Arc<dyn CloseRequestSink>,
    suppress_startup_replay: bool,
) -> tokio::task::AbortHandle {
    let join = tokio::spawn(async move {
        let mut filtering_startup_replay = suppress_startup_replay;
        let mut replay_message_index = 0usize;
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

            // Replacement agents are launched with `--resume`, so their
            // startup sequence replays persisted history. This session has
            // already published those frames; suppress the matching prefix so
            // live subscribers don't see duplicate transcript messages.
            if filtering_startup_replay {
                match &evt {
                    AgentEvent::Ready { .. } => continue,
                    AgentEvent::MessageAppended { message } => {
                        let is_replayed = {
                            let rt = runtime.lock().expect("session runtime mutex poisoned");
                            rt.messages
                                .get(replay_message_index)
                                .is_some_and(|known| messages_match(known, message))
                        };
                        if is_replayed {
                            replay_message_index += 1;
                            continue;
                        }
                        filtering_startup_replay = false;
                    }
                    _ => {
                        filtering_startup_replay = false;
                    }
                }
            }

            // `RequestClose` is a control frame from the agent asking
            // the host to merge or abandon this session. Route it to
            // the close sink and skip the normal publish / state-
            // machine / first-turn path — it isn't user-visible and
            // `apply_event` has nothing meaningful to do with it. The
            // sink call is fire-and-forget: the pump keeps draining
            // until the agent's stdout closes, which it will shortly
            // since the agent exits its command loop after emitting
            // this frame.
            if let AgentEvent::RequestClose { intent } = evt {
                let Some(session) = session_weak.upgrade() else {
                    // Session was dropped between the recv and now —
                    // there is no one to close, so we discard. The
                    // agent is already on its way out.
                    continue;
                };
                let Some(id) = session.session_id.get().copied() else {
                    // Defensive: `RequestClose` should only arrive
                    // after the agent's `Ready`, which populates
                    // `session_id`. Without an id the sink has no
                    // session to dispatch against, so we skip rather
                    // than invent one.
                    drop(session);
                    continue;
                };
                drop(session);
                let sink = close_sink.clone();
                tokio::spawn(async move {
                    sink.request_close(id, intent).await;
                });
                continue;
            }

            // Cheap flag computed before `evt` is consumed by
            // `apply_event` below, so the slug-rename hook can fire
            // without re-cloning the event.
            let is_turn_complete = matches!(evt, AgentEvent::TurnComplete);

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

            // Slug-rename hook: on the first `TurnComplete` observed
            // while the session is fresh, snapshot history, extract the
            // first user message, and fire the sink. The CAS-flip of
            // `fresh` ensures the hook fires at most once per session
            // — even if a crash-restart somehow replays the first turn
            // after resume, `fresh` starts as `false` on the
            // replacement agent's pump (the lifecycle coordinator clears
            // it before calling `replace_agent`).
            if is_turn_complete
                && let Some(session) = session_weak.upgrade()
                && session
                    .fresh
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                let Some(id) = session.session_id.get().copied() else {
                    // Defensive: `TurnComplete` should never precede
                    // `Ready`. If it does, skip the hook — there's
                    // nothing the coordinator can correlate without
                    // an id.
                    continue;
                };
                let snapshot = {
                    let hist = history.lock().expect("session history mutex poisoned");
                    hist.clone()
                };
                // Drop the Arc<ActiveSession> before spawning the
                // fire-and-forget so the task doesn't hold the session
                // alive past its own completion.
                drop(session);
                let Some(first_message) = extract_first_user_message(&snapshot) else {
                    // No user message in history is a logic bug — a
                    // TurnComplete without a preceding user frame
                    // shouldn't be possible. Skip rather than invent a
                    // slug from nothing.
                    continue;
                };
                let sink = first_turn_sink.clone();
                // Fire-and-forget: any slug-rename work (LLM call, git
                // operations) runs on its own task so the pump keeps
                // draining frames.
                tokio::spawn(async move {
                    sink.on_first_turn_complete(id, first_message).await;
                });
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

fn messages_match(left: &domain::Message, right: &domain::Message) -> bool {
    left.role == right.role
        && left.content == right.content
        && left.token_count == right.token_count
}

/// Walk `history` front-to-back and return the concatenated text of
/// the first `MessageAppended` whose role is `User`. Non-text content
/// blocks are skipped; text blocks are joined with newlines so a
/// multi-block user message reads naturally in the LLM prompt used to
/// derive the slug. Returns `None` if no user message is present.
fn extract_first_user_message(history: &[AgentEvent]) -> Option<String> {
    for evt in history {
        if let AgentEvent::MessageAppended { message } = evt
            && matches!(message.role, Role::User)
        {
            let text = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return Some(text);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use agent_host::{
        AgentClient,
        fake::{FakeFirstTurnSink, NoopCloseRequestSink, NoopFirstTurnSink},
    };
    use domain::Message;
    use protocol::{AgentEvent, read_frame, write_frame};
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
