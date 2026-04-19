//! OpenRouter-backed [`ModelCatalog`].
//!
//! Fetches the full `/api/v1/models` list once at construction and caches
//! the `id → context_length` map in memory. Intentionally separate from
//! the streaming chat provider: this is a one-shot discovery call with
//! different failure semantics — any error here is fatal to startup.
//!
//! The split between [`parse`] and [`OpenRouterCatalog::fetch`] keeps the
//! JSON decoding pure and unit-testable with the captured fixture in
//! `experiments/openrouter_models.json`, while `fetch` handles the live
//! HTTP side.

use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;

/// Errors produced when building an [`OpenRouterCatalog`].
///
/// Distinct variants let the host log a useful message naming the
/// endpoint and the failure mode. All variants are fatal at startup —
/// there is no graceful-degrade fallback for a missing catalog.
#[derive(Debug)]
pub enum CatalogError {
    Transport(reqwest::Error),
    Status(u16),
    Parse(serde_json::Error),
    EmptyCatalog,
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(err) => write!(f, "transport error: {err}"),
            Self::Status(code) => write!(f, "unexpected HTTP status {code}"),
            Self::Parse(err) => write!(f, "invalid JSON: {err}"),
            Self::EmptyCatalog => write!(f, "response contained no models with context_length"),
        }
    }
}

impl std::error::Error for CatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(err) => Some(err),
            Self::Parse(err) => Some(err),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for CatalogError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transport(err)
    }
}

impl From<serde_json::Error> for CatalogError {
    fn from(err: serde_json::Error) -> Self {
        Self::Parse(err)
    }
}

/// Wire-format minimum: we only keep the two fields we actually use so
/// the catalog is resilient to OpenRouter adding fields over time. The
/// fixture response carries dozens of per-model fields (pricing,
/// supported_parameters, etc.) that we deliberately ignore.
#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    /// Missing on a handful of provider-listed models; skip rather than
    /// fail the whole load. A single unusable entry shouldn't kill the
    /// host when the other hundreds are fine.
    #[serde(default)]
    context_length: Option<u32>,
}

/// In-memory cache of `model id → context_length` resolved from
/// OpenRouter's `/models` endpoint at startup. Immutable once built.
#[derive(Debug)]
pub struct OpenRouterCatalog {
    entries: HashMap<String, u32>,
}

impl OpenRouterCatalog {
    /// Build a catalog from the raw `/models` JSON body. Pure function;
    /// exposed so unit tests can exercise the parse without a live
    /// network round-trip.
    pub fn parse(body: &str) -> Result<Self, CatalogError> {
        let entries = parse(body)?;
        if entries.is_empty() {
            return Err(CatalogError::EmptyCatalog);
        }
        Ok(Self { entries })
    }

    /// Fetch and parse the catalog over HTTP. Sends the bearer token
    /// even though the endpoint is currently unauthenticated — this
    /// matches the rest of the OpenRouter adapter and avoids a future
    /// surprise if the endpoint ever tightens access.
    pub async fn fetch(
        client: &reqwest::Client,
        base_url: &str,
        api_key: &str,
    ) -> Result<Self, CatalogError> {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let response = client.get(&url).bearer_auth(api_key).send().await?;

        let status = response.status();
        if !status.is_success() {
            return Err(CatalogError::Status(status.as_u16()));
        }

        let body = response.text().await?;
        Self::parse(&body)
    }
}

impl app::ModelCatalog for OpenRouterCatalog {
    fn context_window(&self, model: &str) -> Option<u32> {
        self.entries.get(model).copied()
    }
}

/// Raw parse. Kept as a free function so it can be tested in isolation
/// from the `EmptyCatalog` check (callers that want the "no models"
/// distinction can inspect the map size themselves).
fn parse(body: &str) -> Result<HashMap<String, u32>, CatalogError> {
    // The `/models` response is either a bare array or wrapped in a
    // `{"data": [...]}` envelope depending on OpenRouter API version.
    // Accept both shapes so the catalog keeps working across a wire
    // tweak without needing a host restart to pick up the new parser.
    let models: Vec<ModelEntry> = match serde_json::from_str::<Envelope>(body) {
        Ok(env) => env.data,
        Err(_) => serde_json::from_str(body)?,
    };

    let mut map = HashMap::with_capacity(models.len());
    for entry in models {
        if let Some(len) = entry.context_length {
            map.insert(entry.id, len);
        }
    }
    Ok(map)
}

#[derive(Debug, Deserialize)]
struct Envelope {
    data: Vec<ModelEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use app::ModelCatalog;

    // Captured OpenRouter `/models` response; large (hundreds of models)
    // so we test against real-world variance (entries with/without
    // context_length, various tokenizers, etc.) rather than a trimmed
    // fixture that could mask a parser bug.
    const FIXTURE: &str = include_str!("../../../../experiments/openrouter_models.json");

    #[test]
    fn parse_fixture_resolves_known_model() {
        let catalog = OpenRouterCatalog::parse(FIXTURE).expect("fixture must parse");
        assert_eq!(
            catalog.context_window("meta-llama/llama-3.1-8b-instruct"),
            Some(16384),
        );
    }

    #[test]
    fn parse_fixture_map_size_matches_models_with_context_length() {
        // The parser silently skips entries without context_length, so
        // the final map size reflects that subset. Cross-check by
        // decoding the same JSON with serde_json::Value and counting
        // entries that have the field — this keeps the test robust to
        // a future fixture regeneration that adds or drops models.
        let raw: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let array = raw.as_array().expect("fixture is a top-level array");
        let expected = array
            .iter()
            .filter(|m| m.get("context_length").and_then(|v| v.as_u64()).is_some())
            .count();

        let catalog = OpenRouterCatalog::parse(FIXTURE).unwrap();
        assert_eq!(catalog.entries.len(), expected);
    }

    #[test]
    fn unknown_model_returns_none() {
        let catalog = OpenRouterCatalog::parse(FIXTURE).unwrap();
        assert!(catalog.context_window("not-a-real-model").is_none());
    }

    #[test]
    fn parse_rejects_invalid_json() {
        let err = OpenRouterCatalog::parse("not json").unwrap_err();
        assert!(matches!(err, CatalogError::Parse(_)));
    }

    #[test]
    fn parse_rejects_empty_array() {
        // `[]` is syntactically valid JSON but carries zero usable
        // entries — the host cannot function with an empty catalog, so
        // surface it as a distinct error rather than building a silent
        // map-of-nothing that poisons every session-create.
        let err = OpenRouterCatalog::parse("[]").unwrap_err();
        assert!(matches!(err, CatalogError::EmptyCatalog));
    }

    #[test]
    fn parse_rejects_empty_object() {
        // Same reasoning as `[]`, but through the envelope branch.
        let err = OpenRouterCatalog::parse(r#"{"data":[]}"#).unwrap_err();
        assert!(matches!(err, CatalogError::EmptyCatalog));
    }

    #[test]
    fn parse_accepts_data_envelope() {
        let body = r#"{"data":[{"id":"test/model","context_length":42}]}"#;
        let catalog = OpenRouterCatalog::parse(body).unwrap();
        assert_eq!(catalog.context_window("test/model"), Some(42));
    }

    #[test]
    fn parse_skips_entries_missing_context_length() {
        // One good entry plus a broken one — the broken one should be
        // silently dropped, not fail the whole parse. Keeps a single
        // malformed provider from taking the host down at startup.
        let body = r#"[
            {"id":"good/model","context_length":100},
            {"id":"bad/model"}
        ]"#;
        let catalog = OpenRouterCatalog::parse(body).unwrap();
        assert_eq!(catalog.context_window("good/model"), Some(100));
        assert!(catalog.context_window("bad/model").is_none());
    }

    // -- fetch error paths -------------------------------------------------
    //
    // `fetch` is thin glue over reqwest + `parse`, but the glue has three
    // distinct branches worth pinning: a transport failure (DNS/connect),
    // a non-2xx status (upstream error), and a success that decodes to
    // `EmptyCatalog`. A tiny one-shot HTTP server, served from a local
    // `TcpListener`, gives the last two branches a deterministic response
    // without a network mock dependency. The transport case reaches a
    // listener that accepts but closes before writing so reqwest returns
    // a decode/IO error.

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spin up a TCP listener on an OS-assigned port, hand back the
    /// `base_url` callers should pass to `fetch`, and run one handler
    /// iteration via `handler`. The handler receives the accepted
    /// stream and the raw request bytes so the test can assert on the
    /// path/auth header if it wants. Returns once the handler resolves.
    async fn serve_once<F, Fut>(handler: F) -> String
    where
        F: FnOnce(tokio::net::TcpStream) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((stream, _)) = listener.accept().await {
                handler(stream).await;
            }
        });
        format!("http://{addr}")
    }

    /// Read the request headers (up to the blank line) so the handler
    /// can release the connection cleanly before writing its response.
    /// Without this, reqwest occasionally surfaces the response as a
    /// transport error because the server closed mid-body.
    async fn drain_request(stream: &mut tokio::net::TcpStream) {
        let mut buf = [0u8; 1024];
        let mut acc = Vec::new();
        while !acc.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = stream.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            acc.extend_from_slice(&buf[..n]);
        }
    }

    #[tokio::test]
    async fn fetch_maps_non_success_status_to_status_error() {
        let base = serve_once(|mut stream| async move {
            drain_request(&mut stream).await;
            let _ = stream
                .write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\n\
                      content-length: 0\r\n\
                      connection: close\r\n\r\n",
                )
                .await;
        })
        .await;

        let client = reqwest::Client::new();
        let err = OpenRouterCatalog::fetch(&client, &base, "irrelevant")
            .await
            .expect_err("5xx should fail");
        assert!(
            matches!(err, CatalogError::Status(500)),
            "expected Status(500), got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_maps_empty_success_body_to_empty_catalog() {
        let base = serve_once(|mut stream| async move {
            drain_request(&mut stream).await;
            let body = "[]";
            let response = format!(
                "HTTP/1.1 200 OK\r\n\
                 content-type: application/json\r\n\
                 content-length: {}\r\n\
                 connection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        })
        .await;

        let client = reqwest::Client::new();
        let err = OpenRouterCatalog::fetch(&client, &base, "irrelevant")
            .await
            .expect_err("empty body should fail");
        assert!(
            matches!(err, CatalogError::EmptyCatalog),
            "expected EmptyCatalog, got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_maps_transport_failure_to_transport_error() {
        // Bind a listener and immediately drop it — the socket goes into
        // TIME_WAIT but connection attempts to the port will get
        // `ECONNREFUSED`. That's cleaner than pointing at `127.0.0.1:1`
        // or a made-up hostname whose failure mode is platform-specific.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let base = format!("http://{addr}");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let err = OpenRouterCatalog::fetch(&client, &base, "irrelevant")
            .await
            .expect_err("connect refused should fail");
        assert!(
            matches!(err, CatalogError::Transport(_)),
            "expected Transport(_), got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_parses_success_body_into_catalog() {
        // Round-trip sanity: if the server responds with a well-formed
        // body, fetch returns a populated catalog with the expected
        // context window. Complements the parse-only tests by pinning
        // the fetch+parse composition.
        let base = serve_once(|mut stream| async move {
            drain_request(&mut stream).await;
            let body = r#"{"data":[{"id":"wire/model","context_length":64000}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\n\
                 content-type: application/json\r\n\
                 content-length: {}\r\n\
                 connection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        })
        .await;

        let client = reqwest::Client::new();
        let catalog = OpenRouterCatalog::fetch(&client, &base, "irrelevant")
            .await
            .expect("well-formed response should succeed");
        assert_eq!(catalog.context_window("wire/model"), Some(64000));
    }
}
