//! Port for routing `RequestClose` events from a session pump into the
//! lifecycle coordinator without a cyclic reference.
//!
//! When the agent emits `AgentEvent::RequestClose { intent }`, the pump
//! task running inside [`crate::AgentEventStream`]'s consumer wants to
//! tell the server "please merge/abandon this session." The natural
//! target is the lifecycle coordinator, but wiring
//! `Arc<SessionLifecycle>` into `ActiveSession` creates a strong cycle
//! (lifecycle holds a handle to the registry; the registry holds the
//! sessions; each session holds a pump; the pump would hold the
//! lifecycle).
//!
//! Instead, the pump takes `Arc<dyn CloseRequestSink>`. Production wires
//! `SessionLifecycle` through this; tests use
//! `fake::FakeCloseRequestSink` to record calls. This module defines
//! only the trait; the implementations live at the layer that owns
//! them.

use async_trait::async_trait;
use domain::{CloseIntent, SessionId};

#[async_trait]
pub trait CloseRequestSink: Send + Sync + 'static {
    /// Handle a close request for `id`. Must not block the caller —
    /// implementations should spawn any slow work (git merge,
    /// filesystem work) onto their own tasks. Returning from this call
    /// does not imply the close has finished, only that the sink has
    /// accepted the request.
    async fn request_close(&self, id: SessionId, intent: CloseIntent);
}
