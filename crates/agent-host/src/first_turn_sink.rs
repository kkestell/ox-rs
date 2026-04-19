//! Port for firing the slug-rename (or any other post-first-turn) hook
//! from the session pump into the lifecycle coordinator without a
//! cyclic reference.
//!
//! When the pump observes `AgentEvent::TurnComplete` for a fresh
//! session, it snapshots history, extracts the first user message, and
//! calls this sink so the server can rename the worktree + branch and
//! respawn the agent under the new path. The pump itself cannot do any
//! of that work — git + LLM + registry writes all live at a higher
//! layer — so it hands off through `Arc<dyn FirstTurnSink>`.
//!
//! The same reasoning as `CloseRequestSink` applies: wiring the
//! lifecycle directly into `ActiveSession` would create a strong cycle
//! (lifecycle → registry → sessions → pump → lifecycle). `Weak` alone
//! would force every pump to care about shutdown order; a sink trait
//! keeps the pump ignorant of who owns the implementation.
//!
//! Production wires `SessionLifecycle` through this. Tests substitute
//! `fake::FakeFirstTurnSink` to record calls, or
//! `fake::NoopFirstTurnSink` when a session's first-turn hook is
//! simply out-of-scope for the test.

use std::future::Future;
use std::pin::Pin;

use domain::SessionId;

pub trait FirstTurnSink: Send + Sync + 'static {
    /// Called by the pump on the first `TurnComplete` observed while
    /// the session is fresh. The pump fires this at most once per
    /// session (it CAS-flips the `fresh` flag before calling), but
    /// implementations should still be defensive: a server restart
    /// that resumes a session that was mid-slug-rename could surface
    /// the same first message again.
    ///
    /// Must not block — implementations should punt any slow work
    /// (LLM call, git operations, agent respawn) onto their own task
    /// so the pump can keep draining frames.
    fn on_first_turn_complete(
        &self,
        id: SessionId,
        first_message: String,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}
