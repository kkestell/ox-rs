//! Agent host library.
//!
//! IPC + persistence + per-session state machine for the host side of
//! `ox-agent`. Exposes the [`AgentClient`] handle, the [`LayoutStore`]
//! persistence type, the [`SessionRuntime`] state machine, and the
//! [`AgentSpawner`] trait that lets tests substitute an in-memory
//! subprocess. The crate is framework-agnostic: nothing here knows
//! about GTK, axum, or HTTP.
//!
//! `StreamAccumulator` is re-exported from `app` so consumers do not
//! need to depend on the application layer just to assemble streamed
//! deltas into a renderable message snapshot.

mod client;
mod layout;
mod session_runtime;
mod spawner;

pub use app::StreamAccumulator;
pub use client::{AgentClient, AgentEventStream, AgentSpawnConfig};
pub use layout::{Layout, LayoutStore, normalize_sizes};
pub use session_runtime::{SessionRuntime, ShouldSend, apply_event, begin_send};
pub use spawner::{AgentSpawner, ProcessSpawner};
