//! Shared state stored inside the Tauri `AppHandle`.
//!
//! `AppState` wraps a `tokio::sync::Mutex<WorkspaceState>` so command
//! handlers and drain tasks can mutate the same workspace without data
//! races. The async mutex matters: the drain task awaits on
//! `AgentEventStream::recv` *outside* the lock, then acquires the lock only
//! for the brief synchronous `apply_event` + emit, so lock hold times stay
//! small.
//!
//! `quit_confirmed` is a separate atomic because the `WindowEvent::CloseRequested`
//! handler may run while a command handler already holds the workspace
//! mutex, and tipping "the user confirmed quit" must succeed regardless of
//! that.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use agent_host::WorkspaceState;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub workspace: Arc<Mutex<WorkspaceState>>,
    pub quit_confirmed: Arc<AtomicBool>,
}

impl AppState {
    pub fn new(workspace: WorkspaceState) -> Self {
        Self {
            workspace: Arc::new(Mutex::new(workspace)),
            quit_confirmed: Arc::new(AtomicBool::new(false)),
        }
    }
}
