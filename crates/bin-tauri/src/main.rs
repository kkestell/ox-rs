//! `ox-tauri` — the desktop AI coding assistant GUI, Tauri edition.
//!
//! The composition root: locates the `ox-agent` binary, builds an
//! [`AgentSpawnConfig`] from CLI args + env + defaults, and hands off to
//! [`bin_tauri::run`] which owns the Tauri runtime. All session work —
//! LLM streaming, persistence, tool execution — happens inside
//! `ox-agent` subprocesses.

use std::path::PathBuf;

use agent_host::AgentSpawnConfig;
use anyhow::{Context, Result};
use app::SecretStore;
use clap::Parser;
use domain::SessionId;

#[derive(Parser, Debug)]
#[command(name = "ox-tauri", about = "Desktop AI coding assistant (Tauri)")]
struct Cli {
    /// Resume a previous session by its UUID. Passed through to the agent
    /// as `--resume <id>`.
    #[arg(long)]
    resume: Option<SessionId>,

    /// Override the OpenRouter model ID.
    #[arg(long, default_value = "qwen/qwen3-235b-a22b-2507")]
    model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read the API key up front so a missing env var fails before a window
    // appears — less confusing than opening a broken GUI.
    let secrets = adapter_secrets::EnvSecretStore;
    let api_key = secrets
        .get("OPENROUTER_API_KEY")?
        .context("OPENROUTER_API_KEY is not set — export it and try again")?;

    let sessions_dir = dirs::home_dir()
        .context("could not determine home directory")?
        .join(".ox/sessions");
    let layout_state_path = dirs::home_dir()
        .context("could not determine home directory")?
        .join(".ox/workspaces.json");

    let workspace_root =
        std::env::current_dir().context("could not determine working directory")?;

    let agent_binary = locate_agent_binary().context(
        "could not find the ox-agent binary next to ox-tauri; \
         build the workspace (cargo build) or ensure both binaries live in the same directory",
    )?;

    let spawn_config = AgentSpawnConfig {
        binary: agent_binary,
        workspace_root,
        model: cli.model,
        sessions_dir,
        resume: cli.resume,
        env: vec![("OPENROUTER_API_KEY".to_owned(), api_key)],
    };

    bin_tauri::run(spawn_config, layout_state_path)
}

/// Locate `ox-agent` next to the currently-running executable. No `$PATH`
/// fallback — an older installed copy on `$PATH` would silently win over
/// a freshly-built one, which is the kind of bug that wastes an hour.
fn locate_agent_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating current executable")?;
    let dir = exe
        .parent()
        .context("current executable has no parent directory")?;
    let candidate = dir.join(if cfg!(windows) {
        "ox-agent.exe"
    } else {
        "ox-agent"
    });
    if candidate.exists() {
        Ok(candidate)
    } else {
        anyhow::bail!(
            "no ox-agent binary at {} — rebuild the workspace",
            candidate.display()
        )
    }
}
