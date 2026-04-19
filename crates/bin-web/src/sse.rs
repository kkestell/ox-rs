//! Server-sent events for `GET /api/sessions/:id/events`.
//!
//! The handler first yields every event already buffered in the
//! session's history (so a refresh mid-stream rehydrates the full
//! transcript), then follows the session's broadcast channel until
//! the client disconnects or the channel closes.
//!
//! The replay-then-follow order is load-bearing — without it, a
//! subscriber that joins mid-turn would start from the current
//! `StreamDelta` and miss the `MessageAppended(user)` that preceded
//! it. Replaying the buffered history keeps the frontend's
//! `apply_event` pipeline identical to the first-paint path.
//!
//! The stream is assembled as an async iterator of `axum::sse::Event`s
//! where each event's `data` field is the raw `AgentEvent` JSON —
//! no envelope, no event name, matching the wire format the agent
//! emits. That way the client's SSE handler and any transcript-on-
//! disk debugging look the same.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use domain::SessionId;
use futures_util::Stream;
use futures_util::stream::{self, StreamExt};
use protocol::AgentEvent;
use std::convert::Infallible;
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};

use crate::state::AppState;

pub async fn sse_handler(State(state): State<AppState>, Path(id): Path<SessionId>) -> Response {
    let Some(session) = state.registry.get(id) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let (history, rx) = session.subscribe();

    let replay = stream::iter(history.into_iter().map(event_to_sse));
    let follow = BroadcastStream::new(rx).filter_map(|item| async move {
        match item {
            Ok(evt) => Some(event_to_sse(evt)),
            // A lagged subscriber has missed events — we can't replay
            // a partial turn mid-SSE, so close the stream and let the
            // client reconnect. The `filter_map` returning `None`
            // here would silently swallow — we return `None` so the
            // chained stream simply ends, which the browser treats as
            // an EventSource close and triggers its own reconnect
            // logic. The new connection will snapshot fresh history.
            Err(BroadcastStreamRecvError::Lagged(_)) => None,
        }
    });

    let stream = replay.chain(follow);

    Sse::new(stream)
        // Keep-alive comments every 15s so proxies and lazy clients
        // don't consider an idle session (no live events for a while)
        // a half-open socket.
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

/// Encode one `AgentEvent` as an SSE event. We serialize JSON once
/// here rather than on the client side so the browser's `event.data`
/// is ready to `JSON.parse` without an extra hop.
fn event_to_sse(event: AgentEvent) -> Result<Event, Infallible> {
    let json = serde_json::to_string(&event).unwrap_or_else(|e| {
        // Serialization of a wire type should never fail; if it does,
        // emit a structured Error frame so the client sees something
        // instead of a silent drop.
        fallback_error_json(format!("sse encode failed: {e}"))
    });
    Ok(Event::default().data(json))
}

fn fallback_error_json(message: String) -> String {
    serde_json::to_string(&AgentEvent::Error { message }).unwrap_or_else(|_| {
        // Hard fallback for an impossible second serialization failure.
        r#"{"type":"error","message":"sse encode failed"}"#.to_owned()
    })
}

/// Streams returned by [`sse_handler`] — exposed at module scope so the
/// function signature stays readable. The concrete type is hidden
/// behind `impl Stream` at the Sse boundary in practice.
#[allow(dead_code)]
type _Marker = Box<dyn Stream<Item = Result<Event, Infallible>> + Send>;

#[cfg(test)]
mod tests {
    //! SSE handshake tests. The goal isn't to parse the wire format
    //! end-to-end — `axum::response::sse::Sse` already guarantees
    //! spec-compliant framing — but to assert the application-level
    //! invariant that makes the plan work: every subscriber observes
    //! the complete history, in order, even when it joins mid-turn.

    use std::sync::Arc;
    use std::time::Duration;

    use agent_host::AgentSpawnConfig;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use domain::SessionId;
    use http_body_util::BodyExt;
    use protocol::AgentEvent;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::time::timeout;
    use tower::ServiceExt;

    use crate::registry::SessionRegistry;
    use crate::routes::router;
    use crate::state::AppState;
    use crate::test_support::{
        AgentHandles, DuplexSpawner, empty_layout, test_catalog, test_lifecycle, test_providers,
        unique_temp_dir,
    };

    #[test]
    fn fallback_error_json_escapes_special_characters() {
        let encoded = super::fallback_error_json("quotes \" and slash \\ survive".into());
        let event: AgentEvent = serde_json::from_str(&encoded).unwrap();
        match event {
            AgentEvent::Error { message } => {
                assert_eq!(message, "quotes \" and slash \\ survive");
            }
            other => panic!("expected Error fallback, got {other:?}"),
        }
    }

    async fn make_app() -> (
        Router,
        Arc<SessionRegistry>,
        UnboundedReceiver<AgentHandles>,
    ) {
        let workspace_root = unique_temp_dir("sse-ws");
        let (spawner, rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: std::path::PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            sessions_dir: std::path::PathBuf::from("/nonexistent/sessions"),
            resume: None,
            session_id: None,
            env: vec![],
        };
        let registry = SessionRegistry::new(
            spawner,
            spawn_config,
            empty_layout().await,
            workspace_root.clone(),
            test_catalog(),
            "test/model".into(),
            std::sync::Arc::new(agent_host::fake::NoopCloseRequestSink),
            std::sync::Arc::new(agent_host::fake::NoopFirstTurnSink),
        );
        let lifecycle = test_lifecycle();
        let app = router(AppState {
            registry: registry.clone(),
            lifecycle,
            providers: test_providers(),
        });
        (app, registry, rx)
    }

    async fn create_and_send_ready(
        app: Router,
        rx: &mut UnboundedReceiver<AgentHandles>,
    ) -> (SessionId, AgentHandles) {
        let req = Request::builder()
            .method("POST")
            .uri("/api/sessions")
            .body(Body::empty())
            .unwrap();
        let svc = tokio::spawn(async move { app.oneshot(req).await.unwrap() });
        let mut handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        // Use the lifecycle-preallocated id — the wire contract is that
        // a fresh agent launched with `--session-id <id>` reports that
        // id in its Ready frame.
        let id = handles
            .config
            .session_id
            .expect("lifecycle pre-allocates session id on fresh spawn");
        handles.send_ready(id).await;
        let resp = timeout(Duration::from_secs(2), svc).await.unwrap().unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let returned: SessionId = serde_json::from_value(body["session_id"].clone()).unwrap();
        assert_eq!(returned, id);
        (id, handles)
    }

    /// Pull one SSE frame off the raw response body. The wire format is
    /// `data: <json>\n\n`; we slice out the JSON payload and parse it
    /// as an `AgentEvent` so assertions can match on variants directly.
    async fn next_sse_event(body: &mut http_body_util::BodyStream<Body>) -> Option<AgentEvent> {
        use futures_util::StreamExt;
        let mut buf = Vec::new();
        while let Some(frame) = body.next().await {
            let frame = frame.ok()?;
            if let Some(data) = frame.data_ref() {
                buf.extend_from_slice(data);
            }
            // Each SSE event ends in `\n\n`. Strip the `data: ` prefix
            // and any `: keep-alive` comment lines, which the axum SSE
            // layer emits periodically.
            while let Some(end) = buf.windows(2).position(|w| w == b"\n\n") {
                let raw = buf.drain(..end + 2).collect::<Vec<u8>>();
                let text = std::str::from_utf8(&raw).ok()?;
                for line in text.lines() {
                    if let Some(json) = line.strip_prefix("data: ") {
                        return serde_json::from_str(json).ok();
                    }
                }
                // Pure keep-alive comment — skip and look at next.
            }
        }
        None
    }

    // -- replay / follow ---------------------------------------------------

    #[tokio::test]
    async fn sse_replays_buffered_history_before_following() {
        let (app, registry, mut rx) = make_app().await;
        let (id, mut handles) = create_and_send_ready(app.clone(), &mut rx).await;

        // Emit a second, application-level event before anyone subscribes
        // so it has to come through replay (history), not follow.
        handles.send_event(&AgentEvent::TurnComplete).await;
        // Spin briefly for the pump to observe it.
        for _ in 0..50 {
            let session = registry.get(id).unwrap();
            let (history, _) = session.subscribe();
            if history.len() >= 2 {
                break;
            }
            drop(history);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = http_body_util::BodyStream::new(resp.into_body());
        let first = timeout(Duration::from_secs(2), next_sse_event(&mut body))
            .await
            .expect("first SSE frame")
            .expect("parse Ready");
        assert!(matches!(first, AgentEvent::Ready { .. }));
        let second = timeout(Duration::from_secs(2), next_sse_event(&mut body))
            .await
            .expect("second SSE frame")
            .expect("parse TurnComplete");
        assert!(matches!(second, AgentEvent::TurnComplete));
    }

    #[tokio::test]
    async fn sse_follows_live_events_after_replay() {
        let (app, _registry, mut rx) = make_app().await;
        let (id, mut handles) = create_and_send_ready(app.clone(), &mut rx).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = http_body_util::BodyStream::new(resp.into_body());

        // The one already-buffered event is Ready.
        let ready = timeout(Duration::from_secs(2), next_sse_event(&mut body))
            .await
            .expect("first frame")
            .expect("parse Ready");
        assert!(matches!(ready, AgentEvent::Ready { .. }));

        // Now push a live event; the subscriber should receive it via
        // the broadcast follow.
        handles.send_event(&AgentEvent::TurnComplete).await;
        let live = timeout(Duration::from_secs(2), next_sse_event(&mut body))
            .await
            .expect("live frame")
            .expect("parse live TurnComplete");
        assert!(matches!(live, AgentEvent::TurnComplete));
    }

    #[tokio::test]
    async fn sse_returns_404_for_unknown_session() {
        let (app, _reg, _rx) = make_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{}/events", SessionId::new_v4()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn multiple_subscribers_each_receive_full_history() {
        // Two subscribers, the second one joining strictly after a
        // mid-turn event. Both must see the full sequence in order —
        // that's the "snapshot + follow" contract the plan calls
        // load-bearing for the replay→follow transition.
        let (app, registry, mut rx) = make_app().await;
        let (id, mut handles) = create_and_send_ready(app.clone(), &mut rx).await;

        // Open subscriber A.
        let resp_a = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body_a = http_body_util::BodyStream::new(resp_a.into_body());
        let first_a = timeout(Duration::from_secs(2), next_sse_event(&mut body_a))
            .await
            .expect("A first")
            .expect("A parse ready");
        assert!(matches!(first_a, AgentEvent::Ready { .. }));

        // Push an event and wait for the pump to record it in history.
        handles.send_event(&AgentEvent::TurnComplete).await;
        for _ in 0..50 {
            let session = registry.get(id).unwrap();
            let (history, _) = session.subscribe();
            if history.len() >= 2 {
                break;
            }
            drop(history);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Subscriber A sees it on the live follow.
        let live_a = timeout(Duration::from_secs(2), next_sse_event(&mut body_a))
            .await
            .expect("A live")
            .expect("A parse turn complete");
        assert!(matches!(live_a, AgentEvent::TurnComplete));

        // Subscriber B opens now — it must replay Ready + TurnComplete
        // from history before we feed any more events.
        let resp_b = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body_b = http_body_util::BodyStream::new(resp_b.into_body());
        let first_b = timeout(Duration::from_secs(2), next_sse_event(&mut body_b))
            .await
            .expect("B first")
            .expect("B parse ready");
        assert!(matches!(first_b, AgentEvent::Ready { .. }));
        let second_b = timeout(Duration::from_secs(2), next_sse_event(&mut body_b))
            .await
            .expect("B second")
            .expect("B parse turn complete");
        assert!(matches!(second_b, AgentEvent::TurnComplete));
    }

    #[tokio::test]
    async fn sse_closes_when_session_is_removed_from_registry() {
        // The invariant this test exercises: when a session is dropped
        // from the registry (by any code path — a successful merge, a
        // force abandon, or the graceful-shutdown hook), its
        // `ActiveSession` is dropped, which drops the broadcast sender,
        // which terminates every live `EventSource` stream.
        //
        // We call `registry.remove` directly rather than routing through
        // HTTP. The merge/abandon handlers' full matrix is covered in
        // `routes.rs::tests`; here we care only about the SSE lifecycle.
        let (app, registry, mut rx) = make_app().await;
        let (id, _handles) = create_and_send_ready(app.clone(), &mut rx).await;

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{id}/events"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = http_body_util::BodyStream::new(resp.into_body());
        let _ready = timeout(Duration::from_secs(2), next_sse_event(&mut body))
            .await
            .expect("ready");

        // Drop the session from the registry. The follow stream must
        // terminate shortly after — a stuck subscriber would hang until
        // keep-alive fires, which is fine for a network hiccup but a
        // bug for an explicit session teardown.
        assert!(registry.remove(id));

        let closed = timeout(Duration::from_secs(2), next_sse_event(&mut body)).await;
        match closed {
            Ok(None) => {}
            Ok(Some(evt)) => panic!("expected stream close, got {evt:?}"),
            Err(_) => panic!("stream did not close within timeout"),
        }
    }
}
