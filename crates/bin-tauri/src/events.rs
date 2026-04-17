//! Typed Tauri event emission helpers.
//!
//! These are the one-way pushes from Rust to the webview. The frontend
//! subscribes with `listen<Payload>(name, cb)` and reconstructs its
//! client-side mirror from the payloads. Wire events (`AgentEvent`) are
//! forwarded verbatim as `"agent_event"`; workspace-level events carry
//! authoritative snapshots so the frontend never has to guess at state.

use agent_host::{Snapshot, SplitId};
use protocol::AgentEvent;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

pub const AGENT_EVENT: &str = "agent_event";
pub const SPLIT_ADDED: &str = "split_added";
pub const SPLIT_CLOSED: &str = "split_closed";
pub const WORKSPACE_REPLACED: &str = "workspace_replaced";
pub const CONFIRM_QUIT_REQUESTED: &str = "confirm_quit_requested";
pub const SHOW_ABOUT: &str = "show_about";

#[derive(Debug, Clone, Serialize)]
pub struct AgentEventPayload<'a> {
    pub split_id: SplitId,
    pub event: &'a AgentEvent,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitAddedPayload {
    pub split_id: SplitId,
    pub snapshot: Snapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitClosedPayload {
    pub split_id: SplitId,
    pub snapshot: Snapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceReplacedPayload {
    pub snapshot: Snapshot,
}

pub fn emit_agent_event(app: &AppHandle, split_id: SplitId, event: &AgentEvent) {
    // Silently drop emit errors: they fire when the window has already
    // gone away during shutdown, at which point logging noise helps nobody.
    let _ = app.emit(AGENT_EVENT, AgentEventPayload { split_id, event });
}

pub fn emit_split_added(app: &AppHandle, split_id: SplitId, snapshot: Snapshot) {
    let _ = app.emit(SPLIT_ADDED, SplitAddedPayload { split_id, snapshot });
}

pub fn emit_split_closed(app: &AppHandle, split_id: SplitId, snapshot: Snapshot) {
    let _ = app.emit(SPLIT_CLOSED, SplitClosedPayload { split_id, snapshot });
}

pub fn emit_workspace_replaced(app: &AppHandle, snapshot: Snapshot) {
    let _ = app.emit(WORKSPACE_REPLACED, WorkspaceReplacedPayload { snapshot });
}

pub fn emit_confirm_quit_requested(app: &AppHandle) {
    let _ = app.emit(CONFIRM_QUIT_REQUESTED, ());
}

pub fn emit_show_about(app: &AppHandle) {
    let _ = app.emit(SHOW_ABOUT, ());
}
