//! `ox` — desktop GUI binary.
//!
//! Composition root: resolves the `ox-agent` binary path, builds an
//! `AgentSpawnConfig` from CLI args + env, constructs a single-threaded
//! tokio runtime the agent subprocesses' IPC tasks will run on, and
//! hands off to `app::build_application` which owns the adw::Application.
//! Every UI-touching allocation lives downstream of that call, on the
//! GTK main thread.

use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;
use agent_host::{AgentSpawnConfig, WorkspaceLayouts};
use anyhow::{Context, Result};
use clap::Parser;
use domain::SessionId;

mod actions;
mod app;
mod events;
mod input;
mod modals;
mod objects;
mod split_view;
mod splits;
mod transcript;
mod window;

#[derive(Parser, Debug)]
#[command(name = "ox", about = "Desktop AI coding assistant (GTK)")]
struct Cli {
    /// Resume a previous session by its UUID. Passed through to the
    /// agent as `--resume <id>`. Wins over saved workspace layout.
    #[arg(long)]
    resume: Option<SessionId>,

    /// Override the OpenRouter model ID. Carried over from `bin-agent`'s
    /// default so fresh launches use the same model in both binaries.
    #[arg(long, default_value = "qwen/qwen3-235b-a22b-2507")]
    model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    ensure_graphical_session()?;

    // Fail fast on a missing API key: better to bail here than launch a
    // window and have every spawn immediately error.
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .context("OPENROUTER_API_KEY is not set — export it and try again")?;

    let home = dirs::home_dir().context("could not determine home directory")?;
    let sessions_dir = home.join(".ox/sessions");
    let layout_state_path = home.join(".ox/workspaces.json");

    let workspace_root =
        std::env::current_dir().context("could not determine working directory")?;

    let agent_binary = locate_agent_binary().context(
        "could not find the ox-agent binary next to ox; \
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

    // Multi-thread runtime so the IPC reader/writer tasks have worker
    // threads driving them. A `current_thread` runtime would require the
    // GTK main thread to call `block_on` to advance the executor, which
    // would freeze the UI — so we let tokio spin up its own workers and
    // only `enter()` the runtime briefly when we call `tokio::spawn`.
    // Two workers is plenty for a dozen subprocesses: the tasks are all
    // blocked on IPC reads/writes, not CPU-bound work.
    let runtime = Rc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("building tokio runtime")?,
    );

    let layouts = WorkspaceLayouts::load(&layout_state_path).unwrap_or_else(|e| {
        // A corrupted layout file mustn't wedge the whole app. Drop it
        // on the floor and start fresh; the next `save_layout` will
        // overwrite the bad file with a valid one.
        eprintln!(
            "ignoring workspace layout file {}: {e:#}",
            layout_state_path.display()
        );
        WorkspaceLayouts::default()
    });

    let application = app::build_application(runtime, spawn_config, layouts, layout_state_path);
    let exit = application.run_with_args::<&str>(&[]);
    std::process::exit(exit.into());
}

/// Bail with a clear error if this process has no display to connect to.
/// Prevents a confusing GTK abort buried deep in `Application::run`.
fn ensure_graphical_session() -> Result<()> {
    let has_wayland = std::env::var("WAYLAND_DISPLAY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let has_x11 = std::env::var("DISPLAY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if has_wayland || has_x11 {
        Ok(())
    } else {
        anyhow::bail!(
            "no graphical session detected — set WAYLAND_DISPLAY or DISPLAY and try again"
        )
    }
}

/// Locate `ox-agent` next to the currently-running executable. No
/// `$PATH` fallback so a stale installed copy can never silently win
/// over a freshly-built binary — that's the kind of bug that costs a
/// whole afternoon to diagnose.
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
