use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use domain::SessionId;
use serde::Deserialize;

use crate::state::AppState;

#[derive(Deserialize)]
pub(super) struct LayoutBody {
    #[serde(default)]
    order: Vec<SessionId>,
    #[serde(default)]
    sizes: Vec<f32>,
}

pub(super) async fn put_layout(
    State(state): State<AppState>,
    Json(body): Json<LayoutBody>,
) -> Response {
    let layout = agent_host::Layout::new(body.order, body.sizes);
    match state.registry.put_layout(layout).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => {
            eprintln!("ox: put_layout failed: {err:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("layout persist failed: {err}"),
            )
                .into_response()
        }
    }
}
