//! Agent host library.
//!
//! IPC + persistence + slash-command parsing for the GUI side of `ox-agent`.
//! Exposes the [`AgentClient`] handle, the [`WorkspaceLayouts`] persistence
//! type, and the pure [`classify_input`] helper. Per-split UI state and the
//! multi-split workspace orchestration intentionally live in the GUI binary
//! — this crate is framework-agnostic so the IPC and on-disk shapes never
//! pick up dependencies on a particular UI toolkit.
//!
//! `StreamAccumulator` is re-exported from `app` so consumers do not need to
//! depend on the application layer just to assemble streamed deltas into a
//! renderable message snapshot.

mod client;
mod command;
mod layout;

pub use app::StreamAccumulator;
pub use client::{AgentClient, AgentEventStream, AgentSpawnConfig, SplitId};
pub use command::{SplitAction, classify_input};
pub use layout::{RestoreLayout, SavedWorkspaceLayout, WorkspaceLayouts, normalize_split_fracs};
