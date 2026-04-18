//! Shared test fixtures for the `bin-web` crate.
//!
//! Everything here is gated behind `#[cfg(test)]` so no test-only
//! allocations or channels leak into the release binary. Modules add
//! `use crate::test_support::*;` inside their `#[cfg(test)] mod tests`
//! blocks.
//!
//! The centerpiece is [`DuplexSpawner`], an [`AgentSpawner`]
//! implementation that hands the "agent side" of a pair of
//! [`tokio::io::duplex`] pipes back to the test. That gives tests full
//! control over every wire interaction an `ox-agent` subprocess would
//! normally have, without ever invoking a subprocess.

#![cfg(test)]

use std::path::PathBuf;
use std::sync::Arc;

use agent_host::{AgentClient, AgentEventStream, AgentSpawnConfig, AgentSpawner, LayoutStore};
use anyhow::{Result, anyhow};
use domain::SessionId;
use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
use tokio::io::{BufReader, DuplexStream, duplex};
use tokio::sync::mpsc;

use crate::registry::SessionRegistry;

/// Handles the test side holds after a spawn: the bytes headed to the
/// "agent" and the bytes the "agent" emitted. `config` records what
/// the client asked for so tests can assert on `resume`, `model`, etc.
pub struct AgentHandles {
    pub reader: BufReader<DuplexStream>,
    pub writer: DuplexStream,
    pub config: AgentSpawnConfig,
}

impl AgentHandles {
    /// Convenience: write an `AgentEvent::Ready` with the given id
    /// so a test can unblock an `await_ready` await.
    pub async fn send_ready(&mut self, id: SessionId) {
        write_frame(
            &mut self.writer,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: self.config.workspace_root.clone(),
            },
        )
        .await
        .expect("writing Ready to duplex");
    }

    /// Write an arbitrary `AgentEvent` to the agent side. Panics on
    /// I/O failure (tests should see a clean channel).
    pub async fn send_event(&mut self, event: &AgentEvent) {
        write_frame(&mut self.writer, event)
            .await
            .expect("writing AgentEvent to duplex");
    }

    /// Read the next `AgentCommand` the client emitted. Returns `None`
    /// on a clean EOF — the send pipe closed.
    pub async fn next_command(&mut self) -> Option<AgentCommand> {
        read_frame(&mut self.reader).await.ok().flatten()
    }
}

/// Spawner that pairs an `AgentClient` with a `tokio::io::duplex` the
/// test controls. Every `spawn` call produces a fresh pair of pipes;
/// the agent side lands on the mpsc receiver handed back from `new`.
pub struct DuplexSpawner {
    tx: mpsc::UnboundedSender<AgentHandles>,
    /// When set, every `spawn` call errors immediately instead of
    /// handing out a duplex pair. Used to simulate "agent binary
    /// missing" / "permission denied" for the `restore` fallback test.
    fail: bool,
}

impl DuplexSpawner {
    pub fn new() -> (Arc<Self>, mpsc::UnboundedReceiver<AgentHandles>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Arc::new(Self { tx, fail: false }), rx)
    }

    pub fn failing() -> Arc<Self> {
        let (tx, _rx) = mpsc::unbounded_channel();
        Arc::new(Self { tx, fail: true })
    }
}

impl AgentSpawner for DuplexSpawner {
    fn spawn(&self, config: AgentSpawnConfig) -> Result<(AgentClient, AgentEventStream)> {
        if self.fail {
            return Err(anyhow!("DuplexSpawner configured to fail"));
        }
        let (agent_writer, client_reader) = duplex(64 * 1024);
        let (client_writer, agent_reader) = duplex(64 * 1024);
        let (client, stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        let handles = AgentHandles {
            reader: BufReader::new(agent_reader),
            writer: agent_writer,
            config,
        };
        self.tx
            .send(handles)
            .map_err(|_| anyhow!("DuplexSpawner receiver dropped"))?;
        Ok((client, stream))
    }
}

/// Build a test `SessionRegistry` wired to a `DuplexSpawner`. The
/// returned tuple is `(registry, agent_handles_rx, layout_file_path,
/// workspace_root)`. The caller keeps the receiver alive to pick up
/// the agent side of every `spawn` call.
pub async fn test_registry(
    layout: LayoutStore,
) -> (
    Arc<SessionRegistry>,
    mpsc::UnboundedReceiver<AgentHandles>,
    PathBuf,
) {
    let workspace_root = unique_temp_dir("ws-root");
    let (spawner, rx) = DuplexSpawner::new();
    let spawn_config = AgentSpawnConfig {
        binary: PathBuf::from("/nonexistent/ox-agent"),
        workspace_root: workspace_root.clone(),
        model: "test/model".into(),
        sessions_dir: PathBuf::from("/nonexistent/sessions"),
        resume: None,
        env: vec![],
    };
    let registry = SessionRegistry::new(spawner, spawn_config, layout, workspace_root.clone());
    (registry, rx, workspace_root)
}

/// Unique scratch directory so parallel tests don't collide. Caller is
/// responsible for removing it, but since these are inside
/// `std::env::temp_dir()` we let the OS clean up.
pub fn unique_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ox-web-test-{label}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Empty `LayoutStore` pointed at a scratch JSON path. Used when a
/// test doesn't care about pre-existing layout state.
pub fn empty_layout() -> LayoutStore {
    let path = unique_temp_dir("layout").join("workspaces.json");
    LayoutStore::load(path).expect("empty LayoutStore")
}
