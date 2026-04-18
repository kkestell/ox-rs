//! Axum route wiring.
//!
//! All HTTP endpoints the frontend consumes. One module per concern
//! isn't warranted — the full set is short and reading them side-by-
//! side makes the request/response shapes obvious.
//!
//! The router is built from an `AppState` containing the shared
//! `Arc<SessionRegistry>` so every handler operates on the same
//! live map of sessions.

use axum::Router;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, extract::DefaultBodyLimit};
use domain::SessionId;
use protocol::AgentCommand;
use serde::Deserialize;

use crate::assets;
use crate::registry::CommandDispatch;
use crate::sse;
use crate::state::AppState;

/// Build the router. `state` is cloned into axum once; every handler
/// receives it via the `State` extractor.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_html))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route("/api/sessions/{id}", delete(delete_session))
        .route("/api/sessions/{id}/messages", post(post_message))
        .route("/api/sessions/{id}/cancel", post(post_cancel))
        .route("/api/sessions/{id}/events", get(sse::sse_handler))
        .route("/api/layout", put(put_layout))
        // Large tool outputs can push `MessageAppended` payloads into
        // the multi-megabyte range. Axum's default body limit is 2MB
        // — the `/messages` body is just the user's input so the
        // default is plenty there, but we disable it on the `/layout`
        // route in case a future client ships something beefier and
        // never hit the server default on any route.
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}

// -- static assets -----------------------------------------------------------

async fn index_html() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::INDEX_HTML))
        .expect("building index response")
}

async fn app_js() -> Response {
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::APP_JS))
        .expect("building app.js response")
}

async fn styles_css() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/css; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(assets::STYLES_CSS))
        .expect("building styles.css response")
}

// -- /api/sessions -----------------------------------------------------------

async fn list_sessions(State(state): State<AppState>) -> Response {
    let snapshot = state.registry.snapshot();
    Json(snapshot).into_response()
}

async fn create_session(State(state): State<AppState>) -> Response {
    match state.registry.create().await {
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

async fn delete_session(State(state): State<AppState>, Path(id): Path<SessionId>) -> Response {
    if state.registry.remove(id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

#[derive(serde::Serialize)]
struct CreatedSession {
    session_id: SessionId,
}

// -- /api/sessions/:id/messages --------------------------------------------

#[derive(Deserialize)]
struct SendMessageBody {
    input: String,
}

async fn post_message(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
    Json(body): Json<SendMessageBody>,
) -> Response {
    if body.input.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "input required").into_response();
    }
    match state
        .registry
        .send_command(id, AgentCommand::SendMessage { input: body.input })
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
    }
}

async fn post_cancel(State(state): State<AppState>, Path(id): Path<SessionId>) -> Response {
    match state.registry.send_command(id, AgentCommand::Cancel).await {
        CommandDispatch::Ok => StatusCode::NO_CONTENT.into_response(),
        CommandDispatch::NotFound => StatusCode::NOT_FOUND.into_response(),
        // Cancel is idempotent even against a dead agent — the plan
        // calls out "204 otherwise." Treat `Dead` and `AlreadyTurning`
        // the same way.
        _ => StatusCode::NO_CONTENT.into_response(),
    }
}

// -- /api/layout ------------------------------------------------------------

#[derive(Deserialize)]
struct LayoutBody {
    #[serde(default)]
    order: Vec<SessionId>,
    #[serde(default)]
    sizes: Vec<f32>,
}

async fn put_layout(State(state): State<AppState>, Json(body): Json<LayoutBody>) -> Response {
    let layout = agent_host::Layout::new(body.order, body.sizes);
    match state.registry.put_layout(layout) {
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

#[cfg(test)]
mod tests {
    //! Router-level tests. Each test drives the real axum router via
    //! `tower::ServiceExt::oneshot`, backed by a `SessionRegistry` wired
    //! to a `DuplexSpawner`. The tests cover the full HTTP contract
    //! listed in the plan: status codes, body shapes, and the
    //! wire-level side-effects (did a frame actually reach the agent?).

    use std::sync::Arc;
    use std::time::Duration;

    use agent_host::{AgentSpawnConfig, Layout};
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use domain::SessionId;
    use http_body_util::BodyExt;
    use protocol::AgentCommand;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::time::timeout;
    use tower::ServiceExt;

    use crate::registry::SessionRegistry;
    use crate::state::AppState;
    use crate::test_support::{AgentHandles, DuplexSpawner, empty_layout, unique_temp_dir};

    use super::*;

    /// Build a ready-to-serve router plus the channel the test uses to
    /// receive agent-side handles whenever `POST /api/sessions` fires.
    async fn make_app() -> (
        Router,
        Arc<SessionRegistry>,
        UnboundedReceiver<AgentHandles>,
    ) {
        let workspace_root = unique_temp_dir("routes-ws");
        let (spawner, rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: std::path::PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "routes/test".into(),
            sessions_dir: std::path::PathBuf::from("/nonexistent/sessions"),
            resume: None,
            env: vec![],
        };
        let registry = SessionRegistry::new(
            spawner,
            spawn_config,
            empty_layout(),
            workspace_root.clone(),
        );
        let app = router(AppState {
            registry: registry.clone(),
        });
        (app, registry, rx)
    }

    /// Drive `POST /api/sessions` end-to-end: the handler calls
    /// `registry.create()`, the test harness plays the role of the
    /// agent by writing a `Ready` frame on the duplex, and the response
    /// body carries the session id the agent reported. Returns the
    /// session id, the HTTP status, and the still-live agent handles.
    async fn create_via_router(
        app: Router,
        rx: &mut UnboundedReceiver<AgentHandles>,
    ) -> (StatusCode, SessionId, AgentHandles) {
        let req = Request::builder()
            .method("POST")
            .uri("/api/sessions")
            .body(Body::empty())
            .unwrap();
        let svc_fut = tokio::spawn(async move { app.oneshot(req).await.unwrap() });
        let mut handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("spawner dropped");
        let id = SessionId::new_v4();
        handles.send_ready(id).await;
        let resp = timeout(Duration::from_secs(2), svc_fut)
            .await
            .expect("oneshot timed out")
            .expect("join failure");
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let returned: SessionId = serde_json::from_value(body["session_id"].clone()).unwrap();
        assert_eq!(returned, id);
        (status, id, handles)
    }

    // -- GET /api/sessions -------------------------------------------------

    #[tokio::test]
    async fn list_sessions_returns_empty_snapshot_on_fresh_registry() {
        let (app, _reg, _rx) = make_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body["sessions"].as_array().unwrap().is_empty());
        assert!(body["layout"]["order"].as_array().unwrap().is_empty());
    }

    // -- POST /api/sessions -----------------------------------------------

    #[tokio::test]
    async fn create_session_returns_id_and_registers_it() {
        let (app, registry, mut rx) = make_app().await;
        let (status, id, _handles) = create_via_router(app.clone(), &mut rx).await;
        assert_eq!(status, StatusCode::OK);
        assert!(registry.get(id).is_some());

        // The snapshot following a create should include the session
        // even though no layout PUT has happened yet — the "extras"
        // bucket of `snapshot()`.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
        let sessions = body["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0]["session_id"].as_str().unwrap(), id.to_string());
    }

    #[tokio::test]
    async fn create_session_returns_502_when_agent_spawn_fails() {
        let workspace_root = unique_temp_dir("routes-fail");
        let spawner = DuplexSpawner::failing();
        let spawn_config = AgentSpawnConfig {
            binary: std::path::PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "routes/test".into(),
            sessions_dir: std::path::PathBuf::from("/nonexistent/sessions"),
            resume: None,
            env: vec![],
        };
        let registry = SessionRegistry::new(spawner, spawn_config, empty_layout(), workspace_root);
        let app = router(AppState { registry });
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    // -- DELETE /api/sessions/:id -----------------------------------------

    #[tokio::test]
    async fn delete_session_returns_204_for_known_and_404_for_unknown() {
        let (app, _reg, mut rx) = make_app().await;
        let (_, id, _handles) = create_via_router(app.clone(), &mut rx).await;

        let del = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/sessions/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del.status(), StatusCode::NO_CONTENT);

        // Second delete — same id — must now 404.
        let del2 = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/sessions/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del2.status(), StatusCode::NOT_FOUND);
    }

    // -- POST /api/sessions/:id/messages ----------------------------------

    #[tokio::test]
    async fn post_message_rejects_empty_input_with_400() {
        let (app, _reg, mut rx) = make_app().await;
        let (_, id, _handles) = create_via_router(app.clone(), &mut rx).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"   "}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_message_places_frame_on_wire_and_returns_204() {
        let (app, _reg, mut rx) = make_app().await;
        let (_, id, mut handles) = create_via_router(app.clone(), &mut rx).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // The frame must land on the wire.
        let cmd = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("next_command timeout");
        match cmd {
            Some(AgentCommand::SendMessage { input }) => assert_eq!(input, "hello"),
            other => panic!("expected SendMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn post_message_returns_404_for_unknown_session() {
        let (app, _reg, _rx) = make_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/messages", SessionId::new_v4()))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn post_message_returns_409_on_double_send() {
        let (app, _reg, mut rx) = make_app().await;
        let (_, id, _handles) = create_via_router(app.clone(), &mut rx).await;

        let first = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"one"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::NO_CONTENT);

        let second = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"two"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::CONFLICT);
        let body: serde_json::Value =
            serde_json::from_slice(&second.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["reason"], "already_turning");
    }

    #[tokio::test]
    async fn post_message_returns_410_on_dead_session() {
        let (app, registry, mut rx) = make_app().await;
        let (_, id, handles) = create_via_router(app.clone(), &mut rx).await;

        // Kill the agent side so `alive` flips false. This mirrors
        // what happens in production when the subprocess exits.
        drop(handles);
        for _ in 0..50 {
            if !registry.get(id).unwrap().is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"hi"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::GONE);
    }

    // -- POST /api/sessions/:id/cancel ------------------------------------

    #[tokio::test]
    async fn post_cancel_returns_204_and_places_cancel_frame() {
        let (app, _reg, mut rx) = make_app().await;
        let (_, id, mut handles) = create_via_router(app.clone(), &mut rx).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/cancel"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let cmd = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("next_command timeout");
        assert!(matches!(cmd, Some(AgentCommand::Cancel)));
    }

    #[tokio::test]
    async fn post_cancel_returns_404_for_unknown_session() {
        let (app, _reg, _rx) = make_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/cancel", SessionId::new_v4()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // -- PUT /api/layout --------------------------------------------------

    #[tokio::test]
    async fn put_layout_persists_order_and_sizes() {
        let (app, registry, mut rx) = make_app().await;
        let (_, first, _h1) = create_via_router(app.clone(), &mut rx).await;
        let (_, second, _h2) = create_via_router(app.clone(), &mut rx).await;

        let body = serde_json::json!({
            "order": [second, first],
            "sizes": [0.25, 0.75],
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/layout")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        let snap = registry.snapshot();
        assert_eq!(snap.layout.order, vec![second, first]);
        assert!((snap.layout.sizes[0] - 0.25).abs() < 1e-5);
        assert!((snap.layout.sizes[1] - 0.75).abs() < 1e-5);
    }

    #[tokio::test]
    async fn put_layout_filters_unknown_ids_without_error() {
        let (app, registry, mut rx) = make_app().await;
        let (_, known, _handles) = create_via_router(app.clone(), &mut rx).await;
        let stranger = SessionId::new_v4();

        let body = serde_json::json!({
            "order": [stranger, known],
            "sizes": [0.5, 0.5],
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/layout")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Only the known session survives the filter.
        let stored = registry.snapshot().layout;
        assert_eq!(stored.order, vec![known]);
    }

    // -- end-to-end -------------------------------------------------------

    #[tokio::test]
    async fn end_to_end_create_send_cancel_close_single_session() {
        // The plan's integration acceptance: one session goes through
        // every handler in order. Along the way we assert that the
        // frames arrive on the wire in send order.
        let (app, registry, mut rx) = make_app().await;
        let (_, id, mut handles) = create_via_router(app.clone(), &mut rx).await;

        // Send a message, confirm the frame.
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"go"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let first = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("send frame");
        assert!(matches!(first, Some(AgentCommand::SendMessage { .. })));

        // Cancel, confirm the frame.
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/cancel"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let second = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("cancel frame");
        assert!(matches!(second, Some(AgentCommand::Cancel)));

        // Close the session; registry no longer lists it.
        let del = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/sessions/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del.status(), StatusCode::NO_CONTENT);
        assert!(registry.get(id).is_none());
    }

    #[tokio::test]
    async fn put_layout_reorders_and_then_snapshot_includes_new_fresh_session() {
        // Exercise the "two-pane create + reorder" flow the frontend uses.
        let (app, registry, mut rx) = make_app().await;
        let (_, first, _h1) = create_via_router(app.clone(), &mut rx).await;
        let (_, second, _h2) = create_via_router(app.clone(), &mut rx).await;

        // Swap the order.
        let body = serde_json::json!({
            "order": [second, first],
            "sizes": [0.5, 0.5],
        });
        app.clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/layout")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        // A third session arrives; it lands in the "extras" tail.
        let (_, third, _h3) = create_via_router(app, &mut rx).await;
        let snap = registry.snapshot();
        let order: Vec<SessionId> = snap.sessions.iter().map(|s| s.session_id).collect();
        assert_eq!(order, vec![second, first, third]);
    }

    // Helper usage of Layout to silence the unused-import warning in
    // the rare case no other test in this module exercises it.
    #[allow(dead_code)]
    fn _touch_layout_type() -> Layout {
        Layout::new(vec![], vec![])
    }
}
