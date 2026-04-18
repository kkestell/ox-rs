//! Host-side session record storage port.
//!
//! The application-layer `SessionStore` is intentionally generic and uses
//! return-position futures, which makes it a poor trait object for host
//! lifecycle policy. This narrower port exposes only the operations the host
//! needs while keeping concrete disk storage in an adapter crate.

use anyhow::Result;
use async_trait::async_trait;
use domain::{Session, SessionId};

#[async_trait]
pub trait SessionRecords: Send + Sync + 'static {
    async fn try_load(&self, id: SessionId) -> Result<Option<Session>>;
    async fn save(&self, session: &Session) -> Result<()>;
    async fn delete(&self, id: SessionId) -> Result<()>;
}
