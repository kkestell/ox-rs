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
use axum::routing::{get, post, put};
use axum::{Json, extract::DefaultBodyLimit};
use domain::SessionId;
use protocol::AgentCommand;
use serde::Deserialize;

use crate::assets;
use crate::lifecycle::MergeRejection;
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
        .route("/api/sessions/{id}/merge", post(merge_session))
        .route("/api/sessions/{id}/abandon", post(abandon_session))
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
    // Route through the lifecycle coordinator so every fresh session
    // gets its own git worktree + branch. The registry's legacy
    // `create()` is still available for the `--resume` CLI path and
    // fallback tests, but user-initiated creation always goes through
    // the lifecycle from now on.
    match state.lifecycle.create_and_spawn().await {
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
async fn merge_session(State(state): State<AppState>, Path(id): Path<SessionId>) -> Response {
    match state.lifecycle.merge(id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(rej) => merge_rejection_response(rej, /* is_abandon */ false),
    }
}

#[derive(Deserialize, Default)]
struct AbandonBody {
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
async fn abandon_session(
    State(state): State<AppState>,
    Path(id): Path<SessionId>,
    body: Option<Json<AbandonBody>>,
) -> Response {
    let AbandonBody { confirm } = body.map(|Json(b)| b).unwrap_or_default();
    match state.lifecycle.abandon(id, confirm).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(rej) => merge_rejection_response(rej, /* is_abandon */ true),
    }
}

/// Shared translator from `MergeRejection` to an HTTP response. `is_abandon`
/// swaps the `WorktreeDirty` body for the richer "needs confirmation"
/// shape the abandon UI expects.
fn merge_rejection_response(rej: MergeRejection, is_abandon: bool) -> Response {
    match rej {
        MergeRejection::NotFound => StatusCode::NOT_FOUND.into_response(),
        MergeRejection::AlreadyClosing => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "already_closing"})),
        )
            .into_response(),
        MergeRejection::TurnInProgress => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "turn_in_progress"})),
        )
            .into_response(),
        MergeRejection::WorktreeDirty => {
            let body = if is_abandon {
                serde_json::json!({
                    "reason": "uncommitted_changes",
                    "requires_confirmation": true,
                })
            } else {
                serde_json::json!({"reason": "worktree_dirty"})
            };
            (StatusCode::CONFLICT, Json(body)).into_response()
        }
        MergeRejection::MainDirty => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "main_dirty"})),
        )
            .into_response(),
        MergeRejection::Conflict => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "merge_conflict"})),
        )
            .into_response(),
        MergeRejection::Internal(msg) => {
            eprintln!("ox: session close failed: {msg}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("session close failed: {msg}"),
            )
                .into_response()
        }
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
        CommandDispatch::Closing => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"reason": "closing"})),
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

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use adapter_storage::DiskSessionStore;
    use agent_host::fake::{FakeGit, GitCall};
    use agent_host::{AgentSpawnConfig, Layout, MergeOutcome, WorktreeStatus};
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use domain::SessionId;
    use http_body_util::BodyExt;
    use protocol::{AgentCommand, AgentEvent};
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::time::timeout;
    use tower::ServiceExt;

    use crate::registry::SessionRegistry;
    use crate::state::AppState;
    use crate::test_support::{
        AgentHandles, DuplexSpawner, empty_layout, test_lifecycle, test_lifecycle_for_workspace,
        unique_temp_dir,
    };

    use super::*;

    /// Test harness bundle for merge/abandon assertions. Tests that only
    /// care about the registry-facing routes can destructure the first
    /// three fields; merge/abandon tests also use the fake git handle to
    /// script statuses / merge outcomes and the session store to seed
    /// worktree paths that the lifecycle's `try_load` can find.
    struct Harness {
        app: Router,
        registry: Arc<SessionRegistry>,
        rx: UnboundedReceiver<AgentHandles>,
        git: Arc<FakeGit>,
        store: Arc<DiskSessionStore>,
        /// The same workspace root the registry and lifecycle both
        /// reference; the `FakeGit::merge` call target resolves against
        /// this path.
        workspace_root: PathBuf,
    }

    /// Build a ready-to-serve router plus all the handles a merge/abandon
    /// test could need. Registry and lifecycle share one workspace root
    /// so the lifecycle's `git.merge` target matches what the registry
    /// considers its main checkout. The registry's `close_sink` is the
    /// lifecycle itself — production wiring — so pump-driven
    /// `RequestClose` frames dispatch through the real merge/abandon
    /// paths without a second harness seam.
    async fn make_harness() -> Harness {
        let workspace_root = unique_temp_dir("routes-ws");
        let (lifecycle, git, store) = test_lifecycle_for_workspace(workspace_root.clone());
        let (spawner, rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: std::path::PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "routes/test".into(),
            sessions_dir: std::path::PathBuf::from("/nonexistent/sessions"),
            resume: None,
            session_id: None,
            env: vec![],
        };
        let close_sink: std::sync::Arc<dyn agent_host::CloseRequestSink> = lifecycle.clone();
        let registry = SessionRegistry::new(
            spawner,
            spawn_config,
            empty_layout(),
            workspace_root.clone(),
            close_sink,
            std::sync::Arc::new(agent_host::fake::NoopFirstTurnSink),
        );
        // `POST /api/sessions` routes through the lifecycle coordinator,
        // which needs its registry back-reference wired up before it can
        // spawn anything. In production this happens in `bin-web::run`;
        // here we mirror it after both halves are constructed so the
        // test's HTTP requests exercise the real coordinator path.
        lifecycle.set_registry(std::sync::Arc::downgrade(&registry));
        let app = router(AppState {
            registry: registry.clone(),
            lifecycle,
        });
        Harness {
            app,
            registry,
            rx,
            git,
            store,
            workspace_root,
        }
    }

    /// Registry-only harness — a subset of [`make_harness`] with only the
    /// fields existing registry-focused tests consume. Keeps the
    /// destructure pattern short where the extra handles aren't used.
    async fn make_app() -> (
        Router,
        Arc<SessionRegistry>,
        UnboundedReceiver<AgentHandles>,
    ) {
        let h = make_harness().await;
        (h.app, h.registry, h.rx)
    }

    /// Drive `POST /api/sessions` end-to-end: the handler calls
    /// `lifecycle.create_and_spawn`, which pre-allocates a session id
    /// and passes it through to the spawn config. The test harness
    /// plays the agent by echoing that id in the `Ready` frame.
    /// Returns the session id, the HTTP status, and the still-live
    /// agent handles.
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
        // The lifecycle pre-allocated this id; a fresh agent subprocess
        // would honor it too (the agent's Ready frame echoes whatever
        // `--session-id` it was launched with).
        let id = handles
            .config
            .session_id
            .expect("lifecycle pre-allocates session id on fresh spawn");
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
        let mut harness = make_harness().await;
        let app = harness.app.clone();
        let registry = harness.registry.clone();
        let (status, id, handles) = create_via_router(app.clone(), &mut harness.rx).await;
        assert_eq!(status, StatusCode::OK);
        assert!(registry.get(id).is_some());

        let saved = harness
            .store
            .try_load(id)
            .await
            .expect("load initial session record")
            .expect("initial session record exists");
        assert_eq!(saved.id, id);
        assert_eq!(saved.workspace_root, harness.workspace_root);
        assert_eq!(saved.worktree_path, handles.config.workspace_root);
        assert!(
            saved.messages.is_empty(),
            "freshly-created sessions should not invent transcript history"
        );

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
            session_id: None,
            env: vec![],
        };
        let registry = SessionRegistry::new(
            spawner,
            spawn_config,
            empty_layout(),
            workspace_root,
            std::sync::Arc::new(agent_host::fake::NoopCloseRequestSink),
            std::sync::Arc::new(agent_host::fake::NoopFirstTurnSink),
        );
        let lifecycle = test_lifecycle();
        lifecycle.set_registry(std::sync::Arc::downgrade(&registry));
        let app = router(AppState {
            registry,
            lifecycle,
        });
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

    #[tokio::test]
    async fn post_message_returns_409_while_session_is_closing() {
        let (app, registry, mut rx) = make_app().await;
        let (_, id, mut handles) = create_via_router(app.clone(), &mut rx).await;
        registry.get(id).expect("session").begin_close();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"too late"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "closing");

        let frame = timeout(Duration::from_millis(100), handles.next_command()).await;
        assert!(frame.is_err(), "send while closing wrote to the agent");
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

    // -- POST /api/sessions/:id/merge -------------------------------------

    /// Walk a harness through the common "create + seed store + script
    /// git" preamble that every merge/abandon rejection test shares.
    /// Returns the session id, its worktree path, and the live agent
    /// handles so tests that care about wire frames can still interact
    /// with the duplex pipe.
    async fn seed_live_session(harness: &mut Harness) -> (SessionId, PathBuf, AgentHandles) {
        let (_, id, handles) = create_via_router(harness.app.clone(), &mut harness.rx).await;
        let worktree_path = handles.config.workspace_root.clone();
        // Use a recognisable branch name so assertions can verify the
        // correct branch was targeted end-to-end. FakeGit's default is
        // `"main"` which would collide with the workspace's base branch.
        harness
            .git
            .set_current_branch(worktree_path.clone(), "ox/abc12345");
        (id, worktree_path, handles)
    }

    /// Read a 409 response's body and return its decoded JSON. Panics
    /// on a missing body or non-JSON shape — tests invoke this only
    /// when they know the handler returned a structured reason.
    async fn decode_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn merge_clean_session_returns_204_and_tears_down() {
        let mut harness = make_harness().await;
        let (id, worktree_path, _handles) = seed_live_session(&mut harness).await;

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(harness.registry.get(id).is_none());

        // The full teardown sequence must have run: status → merge →
        // remove → delete. `current_branch` also ran first but we
        // assert only on the decisive steps.
        let calls = harness.git.calls();
        let status_idx = calls
            .iter()
            .position(|c| matches!(c, GitCall::Status(p) if p == &worktree_path))
            .expect("Status call");
        let merge_idx = calls
            .iter()
            .position(|c| matches!(c, GitCall::Merge { branch, .. } if branch == "ox/abc12345"))
            .expect("Merge call");
        let remove_idx = calls
            .iter()
            .position(|c| matches!(c, GitCall::RemoveWorktree { .. }))
            .expect("RemoveWorktree call");
        let delete_idx = calls
            .iter()
            .position(|c| matches!(c, GitCall::DeleteBranch { branch, force: false, .. } if branch == "ox/abc12345"))
            .expect("DeleteBranch(force=false) call");
        assert!(
            status_idx < merge_idx && merge_idx < remove_idx && remove_idx < delete_idx,
            "expected ordered status→merge→remove→delete, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn merge_rejects_dirty_worktree_with_409_and_skips_teardown() {
        let mut harness = make_harness().await;
        let (id, worktree_path, mut handles) = seed_live_session(&mut harness).await;
        harness.git.set_status(worktree_path, WorktreeStatus::Dirty);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "worktree_dirty");

        // Session is still in the registry; no merge or teardown ran.
        assert!(harness.registry.get(id).is_some());
        let calls = harness.git.calls();
        assert!(
            !calls.iter().any(|c| matches!(c, GitCall::Merge { .. })),
            "no Merge call expected, got {calls:?}"
        );
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no RemoveWorktree call expected, got {calls:?}"
        );

        let send = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"still open"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(send.status(), StatusCode::NO_CONTENT);
        match timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("send after rejected merge timed out")
        {
            Some(AgentCommand::SendMessage { input }) => assert_eq!(input, "still open"),
            other => panic!("expected SendMessage after rejected merge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn merge_reports_main_dirty_and_skips_teardown() {
        let mut harness = make_harness().await;
        let (id, _worktree_path, _handles) = seed_live_session(&mut harness).await;
        harness.git.enqueue_merge_outcome(MergeOutcome::MainDirty);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "main_dirty");

        // Session survives; worktree + branch were NOT removed.
        assert!(harness.registry.get(id).is_some());
        let calls = harness.git.calls();
        assert!(
            calls.iter().any(|c| matches!(c, GitCall::Merge { .. })),
            "Merge call expected, got {calls:?}"
        );
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no teardown expected after main-dirty, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn merge_reports_conflict_and_skips_teardown() {
        let mut harness = make_harness().await;
        let (id, _worktree_path, _handles) = seed_live_session(&mut harness).await;
        harness.git.enqueue_merge_outcome(MergeOutcome::Conflicts);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "merge_conflict");

        assert!(harness.registry.get(id).is_some());
        let calls = harness.git.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no teardown expected after conflict, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn merge_unknown_session_returns_404() {
        let harness = make_harness().await;
        let missing = SessionId::new_v4();

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{missing}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        // NotFound is detected before the close lock or the session
        // store are consulted, so nothing should have hit git.
        assert!(harness.git.calls().is_empty());
    }

    #[tokio::test]
    async fn merge_rejects_session_with_turn_in_progress() {
        let mut harness = make_harness().await;
        let (id, _worktree_path, _handles) = seed_live_session(&mut harness).await;

        // Fire a SendMessage to flip `waiting=true` without letting the
        // fake agent respond, so `is_turn_in_progress()` observes a
        // live turn on the next call.
        let send = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"busy"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(send.status(), StatusCode::NO_CONTENT);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "turn_in_progress");

        // Session is still around; no git work happened on this request.
        assert!(harness.registry.get(id).is_some());
        assert!(
            !harness
                .git
                .calls()
                .iter()
                .any(|c| matches!(c, GitCall::Merge { .. })),
            "no Merge call expected when turn is in progress"
        );
    }

    // -- POST /api/sessions/:id/abandon -----------------------------------

    #[tokio::test]
    async fn abandon_clean_session_without_confirm_returns_204() {
        let mut harness = make_harness().await;
        let (id, worktree_path, _handles) = seed_live_session(&mut harness).await;

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/abandon"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(harness.registry.get(id).is_none());

        // Abandon must never try to merge, and must force-delete the
        // branch (force=true) since an abandoned session may have
        // commits that never landed on main.
        let calls = harness.git.calls();
        assert!(
            !calls.iter().any(|c| matches!(c, GitCall::Merge { .. })),
            "abandon must not call Merge, got {calls:?}"
        );
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { worktree_path: wp, .. } if wp == &worktree_path)),
            "abandon must remove the worktree, got {calls:?}"
        );
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::DeleteBranch { branch, force: true, .. } if branch == "ox/abc12345"
            )),
            "abandon must force-delete the branch, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn abandon_dirty_without_confirm_returns_409_with_confirmation_prompt() {
        let mut harness = make_harness().await;
        let (id, worktree_path, _handles) = seed_live_session(&mut harness).await;
        harness.git.set_status(worktree_path, WorktreeStatus::Dirty);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/abandon"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        // The frontend keys both fields; missing `requires_confirmation`
        // would cause the UI to treat the reject as unrecoverable.
        assert_eq!(body["reason"], "uncommitted_changes");
        assert_eq!(body["requires_confirmation"], true);

        assert!(harness.registry.get(id).is_some());
        assert!(
            !harness
                .git
                .calls()
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no teardown before confirmation"
        );
    }

    #[tokio::test]
    async fn abandon_dirty_with_confirm_returns_204_and_force_teardown() {
        let mut harness = make_harness().await;
        let (id, worktree_path, _handles) = seed_live_session(&mut harness).await;
        harness.git.set_status(worktree_path, WorktreeStatus::Dirty);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/abandon"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"confirm":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(harness.registry.get(id).is_none());

        let calls = harness.git.calls();
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::DeleteBranch { branch, force: true, .. } if branch == "ox/abc12345"
            )),
            "abandon --confirm must force-delete the branch, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn abandon_rejects_session_with_turn_in_progress() {
        let mut harness = make_harness().await;
        let (id, _worktree_path, _handles) = seed_live_session(&mut harness).await;

        let send = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/messages"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"busy"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(send.status(), StatusCode::NO_CONTENT);

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/abandon"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"confirm":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body = decode_json(resp).await;
        assert_eq!(body["reason"], "turn_in_progress");
        assert!(harness.registry.get(id).is_some());
    }

    #[tokio::test]
    async fn abandon_unknown_session_returns_404() {
        let harness = make_harness().await;
        let missing = SessionId::new_v4();

        let resp = harness
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{missing}/abandon"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // -- RequestClose via pump --------------------------------------------
    //
    // The merge / abandon HTTP endpoints invoke `SessionLifecycle` directly.
    // These tests exercise the other path: an agent tool emits
    // `AgentEvent::RequestClose { intent }` on the wire, the pump routes it
    // to the registered `CloseRequestSink`, and the lifecycle dispatches to
    // the same merge / abandon flow. The harness wires the lifecycle as the
    // registry's real close sink, so these tests cover the full production
    // shape.

    /// Ship `event` on the agent side of a live session's duplex pipe.
    /// `write_frame` already flushes internally so no extra step is
    /// needed. Shared by the RequestClose tests to keep them readable.
    async fn ship_event(handles: &mut AgentHandles, event: &AgentEvent) {
        protocol::write_frame(&mut handles.writer, event)
            .await
            .expect("write_frame");
    }

    /// Poll until `predicate` returns true or the timeout elapses.
    /// Returns true if the predicate was satisfied within the window.
    /// Used to wait for the fire-and-forget RequestClose dispatch to
    /// land — the sink call runs on its own task and the test can't
    /// block on its completion directly.
    async fn wait_until<F: FnMut() -> bool>(mut predicate: F, max_ms: u64) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_millis(max_ms);
        while std::time::Instant::now() < deadline {
            if predicate() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        predicate()
    }

    #[tokio::test]
    async fn request_close_merge_drives_merge_and_tears_down_session() {
        // A clean session + a Merge `RequestClose` must walk the same
        // merge path the HTTP handler does: status → merge → remove →
        // delete → registry.remove(id) → session JSON deleted.
        let mut harness = make_harness().await;
        let (id, worktree_path, mut handles) = seed_live_session(&mut harness).await;

        ship_event(
            &mut handles,
            &AgentEvent::RequestClose {
                intent: domain::CloseIntent::Merge,
            },
        )
        .await;

        // Fire-and-forget: wait for the sink dispatch to remove the
        // session from the registry. Poll because the spawned task
        // runs independently of our `ship_event` call.
        let removed = wait_until(|| harness.registry.get(id).is_none(), 2_000).await;
        assert!(
            removed,
            "session should be removed from registry after Merge"
        );

        // The session JSON is gone after a successful merge.
        let loaded = harness.store.try_load(id).await.expect("try_load");
        assert!(
            loaded.is_none(),
            "session JSON should be deleted after Merge"
        );

        // Every decisive git call in the teardown ran against the
        // expected worktree + branch.
        let calls = harness.git.calls();
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, GitCall::Merge { branch, .. } if branch == "ox/abc12345")),
            "expected Merge on ox/abc12345, got {calls:?}"
        );
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::RemoveWorktree { worktree_path: wp, .. } if wp == &worktree_path
            )),
            "expected RemoveWorktree on {worktree_path:?}, got {calls:?}"
        );
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::DeleteBranch { branch, force: false, .. } if branch == "ox/abc12345"
            )),
            "expected unforced DeleteBranch on ox/abc12345, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn request_close_abandon_with_confirm_on_dirty_worktree_tears_down() {
        // Abandon must work on dirty worktrees when the agent tool
        // explicitly passed `confirm=true`. The DeleteBranch must be
        // forced because the branch may carry commits that never
        // landed on main.
        let mut harness = make_harness().await;
        let (id, worktree_path, mut handles) = seed_live_session(&mut harness).await;
        harness
            .git
            .set_status(&worktree_path, WorktreeStatus::Dirty);

        ship_event(
            &mut handles,
            &AgentEvent::RequestClose {
                intent: domain::CloseIntent::Abandon { confirm: true },
            },
        )
        .await;

        let removed = wait_until(|| harness.registry.get(id).is_none(), 2_000).await;
        assert!(
            removed,
            "session should be removed from registry after confirmed Abandon"
        );

        let loaded = harness.store.try_load(id).await.expect("try_load");
        assert!(
            loaded.is_none(),
            "session JSON should be deleted after Abandon"
        );

        let calls = harness.git.calls();
        // Abandon must never call Merge.
        assert!(
            !calls.iter().any(|c| matches!(c, GitCall::Merge { .. })),
            "abandon must not call Merge, got {calls:?}"
        );
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::DeleteBranch { branch, force: true, .. } if branch == "ox/abc12345"
            )),
            "abandon must force-delete ox/abc12345, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn request_close_merge_on_dirty_worktree_broadcasts_error_and_keeps_session() {
        // The sink must handle rejection by broadcasting an
        // `AgentEvent::Error` on the session and leaving the session
        // in place — the agent is about to exit, but the dead registry
        // entry still lets the user see why the close failed.
        let mut harness = make_harness().await;
        let (id, worktree_path, mut handles) = seed_live_session(&mut harness).await;
        harness
            .git
            .set_status(&worktree_path, WorktreeStatus::Dirty);

        // Subscribe before shipping the frame so we catch the
        // error-broadcast when it lands.
        let session = harness.registry.get(id).expect("session");
        let (_snapshot, mut rx) = session.subscribe();

        ship_event(
            &mut handles,
            &AgentEvent::RequestClose {
                intent: domain::CloseIntent::Merge,
            },
        )
        .await;

        // Drain the broadcast until an Error arrives or we time out.
        // `rx.recv()` may hand us earlier frames (`MessageAppended`
        // from a prior turn, etc.), so loop until we see the Error.
        let mut saw_error = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            match timeout(Duration::from_millis(200), rx.recv()).await {
                Ok(Ok(AgentEvent::Error { message })) => {
                    assert!(
                        message.contains("merge refused"),
                        "error message should label the intent, got {message:?}"
                    );
                    assert!(
                        message.contains("uncommitted"),
                        "error message should describe the rejection, got {message:?}"
                    );
                    saw_error = true;
                    break;
                }
                Ok(Ok(_)) => continue,
                _ => {}
            }
        }
        assert!(saw_error, "expected a broadcast Error frame");

        // Session stays in the registry — the agent is about to exit
        // on its own, but the dead pane is left for the user.
        assert!(harness.registry.get(id).is_some());

        // No teardown ran.
        let calls = harness.git.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no teardown expected on rejected merge, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn request_close_abandon_without_confirm_on_dirty_worktree_broadcasts_error() {
        // Symmetric to the merge-rejection test above, but for an
        // agent-side `abandon` tool call on a dirty worktree. Because
        // `RequestClose { Abandon { confirm: false } }` carries the
        // agent's own consent, the sink surfaces `abandon refused: …`
        // rather than `merge refused: …`. This guards against the two
        // intent labels getting swapped in the dispatch switch.
        let mut harness = make_harness().await;
        let (id, worktree_path, mut handles) = seed_live_session(&mut harness).await;
        harness
            .git
            .set_status(&worktree_path, WorktreeStatus::Dirty);

        let session = harness.registry.get(id).expect("session");
        let (_snapshot, mut rx) = session.subscribe();

        ship_event(
            &mut handles,
            &AgentEvent::RequestClose {
                intent: domain::CloseIntent::Abandon { confirm: false },
            },
        )
        .await;

        let mut saw_error = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            match timeout(Duration::from_millis(200), rx.recv()).await {
                Ok(Ok(AgentEvent::Error { message })) => {
                    assert!(
                        message.contains("abandon refused"),
                        "error message should label the intent, got {message:?}"
                    );
                    assert!(
                        message.contains("uncommitted"),
                        "error message should describe the rejection, got {message:?}"
                    );
                    saw_error = true;
                    break;
                }
                Ok(Ok(_)) => continue,
                _ => {}
            }
        }
        assert!(saw_error, "expected a broadcast Error frame");

        // Session stays in the registry and no teardown ran.
        assert!(harness.registry.get(id).is_some());
        let calls = harness.git.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "no teardown expected on rejected abandon, got {calls:?}"
        );
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
    async fn end_to_end_create_send_cancel_merge_single_session() {
        // Plan's integration acceptance: one session goes through every
        // handler in order. Along the way we assert that the frames
        // arrive on the wire in send order, and that the final merge
        // tears down the session + its worktree + its branch.
        let Harness {
            app,
            registry,
            mut rx,
            git,
            store: _store,
            workspace_root: _workspace_root,
        } = make_harness().await;
        let (_, id, mut handles) = create_via_router(app.clone(), &mut rx).await;

        // The lifecycle's `create_and_spawn` already recorded an
        // `AddWorktree` on the fake git and saved the initial session
        // record that merge will use to recover the worktree path.
        let worktree_path = handles.config.workspace_root.clone();
        // Script `current_branch` so the merge operates on a recognizable
        // session branch name the final assertion can look for.
        git.set_current_branch(worktree_path.clone(), "ox/abc12345");

        // Send a message, confirm the SendMessage frame.
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

        // Cancel, confirm the Cancel frame.
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

        // The cancel left the session with `waiting=true` — the real
        // agent would respond to `Cancel` with a `TurnCancelled` frame
        // that clears the state machine. Replay that here so the merge
        // can clear the turn-in-progress check. Poll briefly until the
        // pump has processed the frame.
        handles.send_event(&AgentEvent::TurnCancelled).await;
        for _ in 0..50 {
            if !registry
                .get(id)
                .expect("session still alive")
                .is_turn_in_progress()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Merge the session: clean worktree + Merged outcome → 204 and
        // the registry no longer lists the session.
        let merge = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{id}/merge"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(merge.status(), StatusCode::NO_CONTENT);
        assert!(registry.get(id).is_none());

        // The merge recorded the expected git sequence: status →
        // merge → remove_worktree → delete_branch(force=false).
        // `current_branch` also ran at the top of begin_close_flow but
        // the test only asserts presence of the decisive steps.
        let calls = git.calls();
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, GitCall::Merge { branch, .. } if branch == "ox/abc12345")),
            "expected a Merge call on ox/abc12345, got {calls:?}"
        );
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, GitCall::RemoveWorktree { .. })),
            "expected a RemoveWorktree call, got {calls:?}"
        );
        assert!(
            calls.iter().any(|c| matches!(
                c,
                GitCall::DeleteBranch { force: false, branch, .. } if branch == "ox/abc12345"
            )),
            "expected an unforced DeleteBranch on ox/abc12345, got {calls:?}"
        );
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
