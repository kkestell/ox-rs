//! Agent host library.
//!
//! Owns the GUI-side orchestration of one or more `ox-agent` subprocesses:
//! the IPC client, per-split state machine, workspace layout persistence,
//! slash-command classifier, and multi-split workspace state. Consumed by
//! desktop binaries — does not implement any app ports and does not depend
//! on a UI framework.

mod client;
mod command;
mod layout;
mod split;
mod workspace;

pub use client::{AgentClient, AgentEventStream, AgentSpawnConfig, SplitId};
pub use command::{SplitAction, classify_input};
pub use layout::{RestoreLayout, SavedWorkspaceLayout, WorkspaceLayouts, normalize_split_fracs};
pub use split::AgentSplit;
pub use workspace::{
    CloseOutcome, SendError, Snapshot, SpawnedSplit, SplitSnapshot, UnknownSplit, WorkspaceState,
};
