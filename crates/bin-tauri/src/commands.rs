//! Tauri command handlers.
//!
//! Handlers are glue: acquire the workspace mutex, call one or two methods
//! on `WorkspaceState`, emit the matching event, and return. All policy
//! lives in `agent-host::WorkspaceState`. Nothing here reaches into
//! `AgentSplit` directly — that's the point of the `WorkspaceState`
//! encapsulation.
//!
//! Shared actions that both a command *and* the native menu trigger live in
//! this module as `do_*` helpers. The menu handler is sync and dispatches
//! onto the async runtime; Tauri commands are already async. Both call into
//! the same helper so behavior doesn't drift between the two entry points.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use agent_host::{Snapshot, SpawnedSplit, SplitAction, SplitId, classify_input};
use protocol::AgentCommand;
use serde::Serialize;
use tauri::{AppHandle, State};

use crate::drain::spawn_drain;
use crate::events::{
    emit_confirm_quit_requested, emit_split_added, emit_split_closed, emit_workspace_replaced,
};
use crate::state::AppState;

/// Classification result the frontend uses to decide what to do after a
/// `submit` succeeded — clear the textarea, focus the new split, dismiss
/// input, etc. The side effects (new drain task, event emission) have
/// already happened in the handler; this is a hint, not a command.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubmitOutcome {
    SentMessage,
    NewSplit { split_id: SplitId },
    ClosedSplit,
    CancelRequested,
    QuitRequested,
}

#[derive(Debug, Clone, Serialize)]
pub struct AppInfo {
    pub name: &'static str,
    pub version: &'static str,
}

#[tauri::command]
pub async fn get_snapshot(state: State<'_, AppState>) -> Result<Snapshot, String> {
    let ws = state.workspace.lock().await;
    Ok(ws.snapshot())
}

/// Handle a line of user input. Slash commands (`/new`, `/close`, `/quit`)
/// are intercepted; everything else is forwarded to the agent as a
/// `SendMessage`. See [`agent_host::classify_input`] for the rules.
#[tauri::command]
pub async fn submit(
    app: AppHandle,
    state: State<'_, AppState>,
    split_id: SplitId,
    input: String,
) -> Result<SubmitOutcome, String> {
    match classify_input(&input) {
        SplitAction::Send => {
            let mut ws = state.workspace.lock().await;
            ws.send(split_id, AgentCommand::SendMessage { input })
                .map_err(|e| e.to_string())?;
            Ok(SubmitOutcome::SentMessage)
        }
        SplitAction::New => {
            let new_id = add_and_drain_split(&app, state.inner()).await?;
            Ok(SubmitOutcome::NewSplit { split_id: new_id })
        }
        SplitAction::QuitApp => {
            do_request_quit(app, state.inner().clone()).await;
            Ok(SubmitOutcome::QuitRequested)
        }
        SplitAction::CloseSplit => {
            if close_and_emit_or_quit(&app, state.inner(), split_id).await? {
                Ok(SubmitOutcome::QuitRequested)
            } else {
                Ok(SubmitOutcome::ClosedSplit)
            }
        }
        SplitAction::Cancel => {
            let mut ws = state.workspace.lock().await;
            ws.send(split_id, AgentCommand::Cancel)
                .map_err(|e| e.to_string())?;
            Ok(SubmitOutcome::CancelRequested)
        }
    }
}

#[tauri::command]
pub async fn cancel_split(state: State<'_, AppState>, split_id: SplitId) -> Result<(), String> {
    let mut ws = state.workspace.lock().await;
    ws.send(split_id, AgentCommand::Cancel)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn close_split(
    app: AppHandle,
    state: State<'_, AppState>,
    split_id: SplitId,
) -> Result<(), String> {
    close_and_emit_or_quit(&app, state.inner(), split_id).await?;
    Ok(())
}

#[tauri::command]
pub async fn new_split(app: AppHandle, state: State<'_, AppState>) -> Result<SplitId, String> {
    add_and_drain_split(&app, state.inner()).await
}

#[tauri::command]
pub async fn open_workspace(
    app: AppHandle,
    state: State<'_, AppState>,
    path: PathBuf,
) -> Result<(), String> {
    do_open_workspace(app, state.inner().clone(), path)
        .await
        .map_err(|e| e.to_string())
}

/// Replace the current split fractions. The normalized values (which may
/// differ from the input if it was invalid or out-of-shape) are returned so
/// the frontend can snap its CSS grid to what the backend actually stored.
#[tauri::command]
pub async fn set_split_fractions(
    state: State<'_, AppState>,
    fractions: Vec<f32>,
) -> Result<Vec<f32>, String> {
    let mut ws = state.workspace.lock().await;
    Ok(ws.set_fractions(&fractions))
}

#[tauri::command]
pub async fn request_quit(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    do_request_quit(app, state.inner().clone()).await;
    Ok(())
}

/// Final-stage quit: the user acknowledged the "turn in progress" prompt.
/// Save the workspace layout, flip the confirm flag so the reentrant
/// `CloseRequested` handler lets the window close, and exit.
#[tauri::command]
pub async fn confirm_quit(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    {
        let ws = state.workspace.lock().await;
        if let Err(e) = ws.save_layout() {
            eprintln!("failed to save workspace layout on confirm quit: {e:#}");
        }
    }
    state.quit_confirmed.store(true, Ordering::SeqCst);
    app.exit(0);
    Ok(())
}

#[tauri::command]
pub async fn cancel_quit() -> Result<(), String> {
    // Nothing to do on the backend: `quit_confirmed` is never set by the
    // confirm dialog, only by `confirm_quit`. Dismissing the modal is a
    // frontend-only concern. Keeping this as an explicit (empty) command
    // makes the frontend-side flow symmetric with `confirm_quit`.
    Ok(())
}

#[tauri::command]
pub fn get_app_info() -> AppInfo {
    AppInfo {
        name: "Ox",
        version: env!("CARGO_PKG_VERSION"),
    }
}

// ---- Shared helpers reused by the menu handler --------------------------

/// Spawn a new split, wire up its drain task, emit the `split_added` event.
/// Used by both the `new_split` Tauri command and the `/new` slash command
/// path in `submit`.
async fn add_and_drain_split(app: &AppHandle, state: &AppState) -> Result<SplitId, String> {
    let (split_id, stream) = {
        let mut ws = state.workspace.lock().await;
        ws.add_split().map_err(|e| e.to_string())?
    };
    spawn_drain(app.clone(), state.clone(), split_id, stream);
    let snapshot = state.workspace.lock().await.snapshot();
    emit_split_added(app, split_id, snapshot);
    Ok(split_id)
}

/// Close a split. If it was the last, hand off to the quit path and return
/// `true`. Otherwise emit `split_closed` with a refreshed snapshot so the
/// frontend can rebuild its layout.
async fn close_and_emit_or_quit(
    app: &AppHandle,
    state: &AppState,
    split_id: SplitId,
) -> Result<bool, String> {
    let outcome = {
        let mut ws = state.workspace.lock().await;
        ws.close_split(split_id).map_err(|e| e.to_string())?
    };
    if outcome.last_split_closed {
        do_request_quit(app.clone(), state.clone()).await;
        Ok(true)
    } else {
        let snapshot = state.workspace.lock().await.snapshot();
        emit_split_closed(app, split_id, snapshot);
        Ok(false)
    }
}

/// Replace the workspace root. Saves the outgoing layout, spawns drains for
/// every new split, and emits `workspace_replaced` so the frontend can
/// rebuild. Used by both the `open_workspace` Tauri command and File > Open
/// in the native menu.
pub async fn do_open_workspace(
    app: AppHandle,
    state: AppState,
    path: PathBuf,
) -> anyhow::Result<()> {
    let spawned: Vec<SpawnedSplit> = {
        let mut ws = state.workspace.lock().await;
        ws.replace_workspace(path)?
    };
    for (split_id, stream) in spawned {
        spawn_drain(app.clone(), state.clone(), split_id, stream);
    }
    let snapshot = state.workspace.lock().await.snapshot();
    emit_workspace_replaced(&app, snapshot);
    Ok(())
}

/// Request a quit. If any split is mid-turn, emit `confirm_quit_requested`
/// so the frontend surfaces a modal; otherwise save the layout and exit
/// immediately. Used by File > Quit, `/quit`, and the `request_quit`
/// command.
pub async fn do_request_quit(app: AppHandle, state: AppState) {
    let busy = {
        let ws = state.workspace.lock().await;
        ws.any_turn_in_progress()
    };
    if busy {
        emit_confirm_quit_requested(&app);
        return;
    }
    {
        let ws = state.workspace.lock().await;
        if let Err(e) = ws.save_layout() {
            eprintln!("failed to save workspace layout on quit: {e:#}");
        }
    }
    // Set the confirm flag *before* `exit` so the re-entrant close handler
    // (which may fire synchronously on some platforms) doesn't re-trigger
    // the quit-confirmation path.
    state.quit_confirmed.store(true, Ordering::SeqCst);
    app.exit(0);
}
