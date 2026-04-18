//! `AgentSpawner` — indirection over the bytes-to-agent step.
//!
//! Tests substitute an in-memory implementation that pairs
//! [`AgentClient::new`] with a `tokio::io::duplex` partner the test drives
//! directly. That seam lets route-level tests run without touching a
//! subprocess — they play the role of the agent entirely in Rust.
//!
//! The trait deliberately takes `&self` (not `&mut self`) and requires
//! `Send + Sync + 'static`: the registry holds `Arc<dyn AgentSpawner>` and
//! calls it from handler tasks that may run on different workers.

use anyhow::Result;

use crate::client::{AgentClient, AgentEventStream, AgentSpawnConfig};

pub trait AgentSpawner: Send + Sync + 'static {
    fn spawn(&self, config: AgentSpawnConfig) -> Result<(AgentClient, AgentEventStream)>;
}
