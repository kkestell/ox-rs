use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use domain::SessionId;
use protocol::AgentCommand;
use serde::Deserialize;

use crate::registry::CommandDispatch;
use crate::state::AppState;

use super::json_error;

#[derive(Deserialize)]
pub(super) struct SendMessageBody {
    input: String,
}

pub(super) async fn post_message(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
    Json(body): Json<SendMessageBody>,
) -> Response {
    if body.input.trim().is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "input_required");
    }
    match state
        .registry
        .send_command(
            id,
            AgentCommand::SendMessage {
                input: body.input,
                model: String::new(),
            },
        )
        .await
    {
        CommandDispatch::Ok => StatusCode::NO_CONTENT.into_response(),
        CommandDispatch::NotFound => StatusCode::NOT_FOUND.into_response(),
        CommandDispatch::Dead => StatusCode::GONE.into_response(),
        CommandDispatch::AlreadyTurning => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "already_turning"})),
        )
            .into_response(),
        CommandDispatch::Closing => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "closing"})),
        )
            .into_response(),
    }
}

pub(super) async fn post_cancel(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
) -> Response {
    match state.registry.send_command(id, AgentCommand::Cancel).await {
        CommandDispatch::Ok => StatusCode::NO_CONTENT.into_response(),
        CommandDispatch::NotFound => StatusCode::NOT_FOUND.into_response(),
        // Cancel is idempotent even against a dead agent — the plan
        // calls out "204 otherwise." Treat `Dead` and `AlreadyTurning`
        // the same way.
        _ => StatusCode::NO_CONTENT.into_response(),
    }
}

#[derive(Deserialize)]
pub(super) struct ToolApprovalBody {
    approved: bool,
}

pub(super) async fn post_tool_approval(
    State(state): State<AppState>,
    Path((id, request_id)): Path<(SessionId, String)>,
    Json(body): Json<ToolApprovalBody>,
) -> Response {
    match state
        .registry
        .resolve_tool_approval(id, request_id, body.approved)
        .await
    {
        CommandDispatch::Ok => StatusCode::NO_CONTENT.into_response(),
        CommandDispatch::NotFound => StatusCode::NOT_FOUND.into_response(),
        CommandDispatch::Dead => StatusCode::GONE.into_response(),
        CommandDispatch::AlreadyTurning => StatusCode::NO_CONTENT.into_response(),
        CommandDispatch::Closing => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "closing"})),
        )
            .into_response(),
    }
}
