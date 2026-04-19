use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use domain::SessionId;
use serde::Deserialize;

use crate::state::AppState;

use super::merge_rejection_response;

pub(super) async fn list_sessions(State(state): State<AppState>) -> Response {
    let snapshot = state.registry.snapshot().await;
    Json(snapshot).into_response()
}

pub(super) async fn create_session(State(state): State<AppState>) -> Response {
    // Route through the lifecycle coordinator so every fresh session
    // gets its own git worktree + branch.
    match state.lifecycle.create_and_spawn(&state.registry).await {
        Ok(session_id) => Json(CreatedSession { session_id }).into_response(),
        Err(err) => {
            eprintln!("ox: create_session failed: {err:#}");
            (
                StatusCode::BAD_GATEWAY,
                format!("failed to start agent: {err}"),
            )
                .into_response()
        }
    }
}

/// POST /api/sessions/{id}/merge
///
/// Merge the session branch back into the main workspace's base branch
/// and tear down the session. Status code mapping mirrors the plan:
///
/// - `204 No Content` — merge + teardown succeeded, session is gone.
/// - `404 Not Found` — no session with that id.
/// - `409 Conflict` — structured `{"reason": "..."}` body; the frontend
///   uses `reason` to choose which dialog to surface:
///   `already_closing`, `turn_in_progress`, `worktree_dirty`,
///   `main_dirty`, or `merge_conflict`.
/// - `500 Internal Server Error` — git subprocess or disk failure; the
///   server-side log carries details. The client has no useful action
///   beyond retrying.
pub(super) async fn merge_session(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
) -> Response {
    match state.lifecycle.merge(id, &state.registry).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(rej) => merge_rejection_response(rej, /* is_abandon */ false),
    }
}

#[derive(Deserialize, Default)]
pub(super) struct AbandonBody {
    /// `true` means the caller has already confirmed they are willing to
    /// discard uncommitted work. `#[serde(default)]` so the body is
    /// optional — a bare `POST /abandon` with no body means "abandon
    /// only if clean."
    #[serde(default)]
    confirm: bool,
}

/// POST /api/sessions/{id}/abandon
///
/// Discard the session's worktree + branch without merging. Shares the
/// status-code ladder with `merge_session`, with one carve-out: the
/// `WorktreeDirty` rejection returns a richer body
/// (`{"reason": "uncommitted_changes", "requires_confirmation": true}`)
/// so the frontend can show a confirm dialog and retry with
/// `{"confirm": true}`. A body-less POST is treated as `confirm=false`.
pub(super) async fn abandon_session(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
    body: Option<Json<AbandonBody>>,
) -> Response {
    let AbandonBody { confirm } = body.map(|Json(b)| b).unwrap_or_default();
    match state.lifecycle.abandon(id, confirm, &state.registry).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(rej) => merge_rejection_response(rej, /* is_abandon */ true),
    }
}

#[derive(serde::Serialize)]
struct CreatedSession {
    session_id: SessionId,
}
