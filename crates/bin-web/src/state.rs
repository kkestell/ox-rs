//! `AppState` — the single `Clone`-able handle threaded through axum's
//! `State` extractor to every handler.
//!
//! Handlers that need the registry clone the `Arc` out of the state
//! cheaply; axum itself clones `AppState` once per request. The whole
//! server model is the registry, so the state struct is a one-field
//! wrapper rather than a grab-bag of unrelated globals.

use std::sync::Arc;

use crate::registry::SessionRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<SessionRegistry>,
}
