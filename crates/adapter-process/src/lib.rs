//! Process-backed host adapter.
//!
//! This crate owns the concrete `tokio::process::Command` wiring for
//! launching `ox-agent`. The inward `agent-host` crate only sees generic
//! async I/O plus the `AgentSpawner` port.

use agent_host::{AgentClient, AgentEventStream, AgentSpawnConfig, AgentSpawner};
use anyhow::{Context, Result};
use tokio::io::BufReader;
use tokio::process::Command;

/// Production spawner for `ox-agent`.
pub struct ProcessSpawner;

impl AgentSpawner for ProcessSpawner {
    fn spawn(&self, config: AgentSpawnConfig) -> Result<(AgentClient, AgentEventStream)> {
        let mut cmd = Command::new(&config.binary);
        cmd.arg("--workspace-root")
            .arg(&config.workspace_root)
            .arg("--sessions-dir")
            .arg(&config.sessions_dir);
        if let Some(id) = config.resume {
            cmd.arg("--resume").arg(id.to_string());
        } else if let Some(id) = config.session_id {
            cmd.arg("--session-id").arg(id.to_string());
        }
        for (k, v) in &config.env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning {}", config.binary.display()))?;
        let stdin = child.stdin.take().context("child stdin missing")?;
        let stdout = child.stdout.take().context("child stdout missing")?;

        let (client, stream) = AgentClient::new(BufReader::new(stdout), stdin);
        Ok((client.with_drop_guard(child), stream))
    }
}
