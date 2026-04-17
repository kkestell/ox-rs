//! Per-split drain task.
//!
//! One `tokio::task` per split owns that split's `AgentEventStream`. The
//! task awaits events off the wire, then acquires the workspace mutex just
//! long enough to apply the event to authoritative state and emit the
//! corresponding Tauri event. Awaiting on `recv().await` lives *outside*
//! the lock so a slow stream never blocks command handlers.
//!
//! Emit-inside-lock is deliberate: it guarantees the frontend sees events
//! in the same order authoritative state changes land. Contention is zero
//! because only one drain task exists per split and `apply_event` is fast.
//!
//! On stream shutdown (`recv() == None`) the task synthesizes an
//! `AgentEvent::Error { "agent disconnected" }` so the frontend renders a
//! terminal error rather than a stuck "waiting…" state, then exits.

use agent_host::{AgentEventStream, SplitId};
use protocol::AgentEvent;
use tauri::AppHandle;

use crate::events::emit_agent_event;
use crate::state::AppState;

pub fn spawn_drain(
    app: AppHandle,
    state: AppState,
    split_id: SplitId,
    mut stream: AgentEventStream,
) {
    tokio::spawn(async move {
        loop {
            match stream.recv().await {
                Some(event) => apply_and_emit(&app, &state, split_id, event).await,
                None => {
                    let event = AgentEvent::Error {
                        message: "agent disconnected".into(),
                    };
                    apply_and_emit(&app, &state, split_id, event).await;
                    break;
                }
            }
        }
    });
}

async fn apply_and_emit(app: &AppHandle, state: &AppState, split_id: SplitId, event: AgentEvent) {
    let mut workspace = state.workspace.lock().await;
    // If the split has already been closed (user hit `/close` or the
    // workspace was replaced), drop the event — the state's `UnknownSplit`
    // return is the signal.
    if workspace.apply_event(split_id, event.clone()).is_err() {
        return;
    }
    emit_agent_event(app, split_id, &event);
}
