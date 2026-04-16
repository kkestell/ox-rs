//! `ox-gui` — the desktop AI coding assistant GUI.
//!
//! A thin composition root: locates the `ox-agent` binary, builds an
//! [`AgentSpawnConfig`] from CLI args + env + defaults, spawns the initial
//! agent, wraps its client in an [`AgentSplit`], hands `vec![split]` plus a
//! cloneable spawn-config template to [`OxApp`], and runs the egui window.
//! The user can spawn additional agents at runtime with `/new`. On
//! shutdown, prints a resume command per active session so the user can
//! pick up where they left off.
//!
//! All session work — LLM streaming, session persistence, tool execution —
//! happens inside `ox-agent` subprocesses. This binary holds display state
//! and nothing more.

use std::path::PathBuf;

use adapter_egui::{AgentClient, AgentSpawnConfig, AgentSplit, OxApp};
use anyhow::{Context, Result};
use app::SecretStore;
use clap::Parser;
use domain::SessionId;

#[derive(Parser, Debug)]
#[command(name = "ox-gui", about = "Desktop AI coding assistant")]
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

    // Read the API key now rather than lazily inside the agent so a missing
    // env var fails before a window appears — less confusing than opening a
    // broken GUI.
    let secrets = adapter_secrets::EnvSecretStore;
    let api_key = secrets
        .get("OPENROUTER_API_KEY")?
        .context("OPENROUTER_API_KEY is not set — export it and try again")?;

    let sessions_dir = dirs::home_dir()
        .context("could not determine home directory")?
        .join(".ox/sessions");

    let workspace_root =
        std::env::current_dir().context("could not determine working directory")?;

    let agent_binary = locate_agent_binary().context(
        "could not find the ox-agent binary next to ox-gui; \
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

    // The agent spawns inside the tokio runtime — its client's reader/
    // writer tasks live there. The GUI thread is egui's main thread and
    // never blocks on async work; it only calls `try_recv` each frame.
    let rt = tokio::runtime::Runtime::new()?;
    let _guard = rt.enter();

    let client = AgentClient::spawn(spawn_config.clone())?;
    let split = AgentSplit::new(client);
    // Pass a template config with `resume: None` so `/new` always starts
    // fresh sessions. The original config (which may have `--resume`) was
    // consumed by the initial spawn above.
    let mut template = spawn_config;
    template.resume = None;
    let (app, session_id_mirror) = OxApp::new(vec![split], template, env!("CARGO_PKG_VERSION"));
    app.run()?;

    // After the GUI closes, read the session IDs the app mirrored on its
    // last frame. These are the IDs the user can re-open to pick up where
    // they left off. For a quick-exit before the agent's `Ready` frame,
    // the ID will be `None` — which is correct, because the agent never
    // persisted a session in that case.
    let ids = session_id_mirror
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();
    for id in ids.into_iter().flatten() {
        eprintln!("ox-gui --resume {id}");
    }

    Ok(())
}

/// Locate `ox-agent` next to the currently-running executable.
///
/// `cargo build` puts both `ox-gui` and `ox-agent` in the same target
/// directory, and `cargo install` does the same for the installed binaries.
/// Falling back to `$PATH` lookup would be convenient but invites confusion
/// if the user has an older copy on their PATH than the one they just
/// built, so we keep it strict.
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
