//! `AppState` — the single `Clone`-able handle threaded through axum's
//! `State` extractor to every handler.
//!
//! Handlers that need the registry or lifecycle coordinator clone the
//! `Arc` out of the state cheaply; axum itself clones `AppState` once
//! per request. Both handles live side by side because a handler
//! typically does registry lookups **and** lifecycle orchestration in
//! the same call (e.g. `POST /:id/merge` resolves the session via the
//! registry, then delegates close orchestration to the lifecycle).

use std::sync::Arc;

use app::config::ProvidersConfig;

use crate::lifecycle::SessionLifecycle;
use crate::registry::SessionRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<SessionRegistry>,
    pub providers: Arc<ProvidersConfig>,
    /// Lifecycle coordinator. The `POST /sessions`, `/merge`, and
    /// `/abandon` handlers dispatch through this; the registry handles
    /// session-map lookups for everything else.
    pub lifecycle: Arc<SessionLifecycle>,
}
