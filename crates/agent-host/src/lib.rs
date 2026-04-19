//! Agent host library.
//!
//! IPC + persistence + per-session state machine for the host side of
//! `ox-agent`. Exposes the [`AgentClient`] handle, the layout model
//! persistence type, the [`SessionRuntime`] state machine, and the
//! [`AgentSpawner`] trait that lets tests substitute an in-memory
//! subprocess. The crate is framework-agnostic: nothing here knows
//! about GTK, axum, or HTTP.

mod client;
mod close_request_sink;
pub mod fake;
mod first_turn_sink;
mod git;
mod layout;
mod paths;
mod session_runtime;
mod slug_generator;
mod spawner;
mod workspace_context;

pub use client::{AgentClient, AgentEventStream, AgentSpawnConfig};
pub use close_request_sink::CloseRequestSink;
pub use first_turn_sink::FirstTurnSink;
pub use git::{Git, MergeOutcome, WorktreeStatus};
pub use layout::{Layout, LayoutRepository, normalize_sizes};
pub use paths::workspace_slug;
pub use session_runtime::{
    BeginClose, SessionRuntime, ShouldSend, apply_event, begin_close, begin_send, clear_closing,
};
pub use slug_generator::SlugGenerator;
pub use spawner::AgentSpawner;
pub use workspace_context::WorkspaceContext;
