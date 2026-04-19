use app::config::{Model, Provider, ProviderType};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use domain::SessionId;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

use super::{json_error_message, merge_rejection_response};

pub(super) async fn list_sessions(State(state): State<AppState>) -> Response {
    let snapshot = state.registry.snapshot().await;
    Json(snapshot).into_response()
}

pub(super) async fn create_session(State(state): State<AppState>) -> Response {
    match state.lifecycle.create_and_spawn(&state.registry).await {
        Ok(session_id) => Json(CreatedSession { session_id }).into_response(),
        Err(err) => {
            eprintln!("ox: create_session failed: {err:#}");
            json_error_message(
                StatusCode::BAD_GATEWAY,
                "agent_start_failed",
                format!("{err:#}"),
            )
        }
    }
}

pub(super) async fn get_providers(State(state): State<AppState>) -> Response {
    let providers: Vec<ProviderJson> = state
        .providers
        .providers
        .iter()
        .map(ProviderJson::from_provider)
        .collect();
    Json(ProvidersJson { providers }).into_response()
}

#[derive(Deserialize)]
pub(super) struct PatchModelBody {
    model: String,
}

pub(super) async fn patch_model(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
    Json(body): Json<PatchModelBody>,
) -> Response {
    let Some((provider, _model)) = state.providers.model(&body.model) else {
        return json_error_message(
            StatusCode::BAD_REQUEST,
            "unknown_model",
            format!("unknown model {:?}", body.model),
        );
    };
    if provider.provider_type != ProviderType::OpenRouter {
        return json_error_message(
            StatusCode::BAD_REQUEST,
            "unwired_model",
            format!("model {:?} is not backed by a wired provider", body.model),
        );
    }
    if state.registry.get(id).is_none() {
        return StatusCode::NOT_FOUND.into_response();
    }
    match state.registry.set_session_model(id, body.model).await {
        Ok(()) => Json(state.registry.snapshot().await).into_response(),
        Err(err) => json_error_message(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_update_failed",
            format!("{err:#}"),
        ),
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

#[derive(Serialize)]
struct ProvidersJson {
    providers: Vec<ProviderJson>,
}

#[derive(Serialize)]
struct ProviderJson {
    id: String,
    name: String,
    #[serde(rename = "type")]
    provider_type: ProviderType,
    base_url: Option<String>,
    models: Vec<ModelJson>,
}

#[derive(Serialize)]
struct ModelJson {
    id: String,
    name: String,
    context_in: u32,
    wired: bool,
}

impl ProviderJson {
    fn from_provider(provider: &Provider) -> Self {
        let wired = provider.provider_type == ProviderType::OpenRouter;
        Self {
            id: provider.id.clone(),
            name: provider.name.clone(),
            provider_type: provider.provider_type,
            base_url: provider.base_url.clone(),
            models: provider
                .models
                .iter()
                .map(|model| ModelJson::from_model(model, wired))
                .collect(),
        }
    }
}

impl ModelJson {
    fn from_model(model: &Model, wired: bool) -> Self {
        Self {
            id: model.id.clone(),
            name: model.name.clone(),
            context_in: model.context_in,
            wired,
        }
    }
}
