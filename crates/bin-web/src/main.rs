//! `ox` — the web-native session workspace.
//!
//! Boots an axum server on `127.0.0.1:<port>`, restores the saved layout
//! for the current workspace root, spawns one `ox-agent` subprocess per
//! saved session. If no sessions are restored, the workspace starts
//! empty and waits for the user to create one. The server opens a
//! browser tab pointed at the URL (unless `--no-open`) and keeps
//! running until SIGINT; on shutdown it flushes the current layout to
//! disk and lets `kill_on_drop` tear down every child.
//!
//! This binary is intentionally thin. All server logic lives in the
//! sibling modules (`routes`, `sse`, `registry`, `session`). `main.rs`
//! owns three concerns:
//!
//! 1. Resolve CLI + env + filesystem paths into an `AgentSpawnConfig`.
//! 2. Wire the tokio runtime, router, TCP listener, and signal handler.
//! 3. Kick the browser if the user wants it.

mod assets;
mod lifecycle;
mod registry;
mod routes;
mod session;
mod sse;
mod startup;
mod state;
#[cfg(test)]
mod test_support;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use adapter_git::CliGit;
use adapter_llm::OpenRouterSlugGenerator;
use adapter_process::ProcessSpawner;
use adapter_storage::{DiskLayoutRepository, DiskSessionStore};
use agent_host::{
    AgentSpawnConfig, CloseRequestSink, FirstTurnSink, Git, LayoutRepository, SessionRecords,
    SlugGenerator, workspace_slug,
};
use anyhow::{Context, Result};
use clap::Parser;
use domain::SessionId;
use tokio::net::TcpListener;

use crate::lifecycle::SessionLifecycle;
use crate::registry::SessionRegistry;
use crate::routes::router;
use crate::startup::assert_workspace_ready;
use crate::state::AppState;

/// Default OpenRouter model for new sessions. Matches the default the
/// deleted GTK binary used — keeps existing user flows working.
const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4.5";

/// Default port for the local HTTP server.
const DEFAULT_PORT: u16 = 3737;

#[derive(Parser, Debug)]
#[command(name = "ox", about = "Local web UI for ox sessions")]
struct Cli {
    /// Resume a single session by id, bypassing the saved layout.
    #[arg(long)]
    resume: Option<SessionId>,

    /// Workspace directory to open (defaults to current directory).
    #[arg(long)]
    workspace: Option<PathBuf>,

    /// Default model for new sessions.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Port to bind the HTTP server to on 127.0.0.1.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// Skip launching the system browser.
    #[arg(long)]
    no_open: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Two worker threads matches the blocking-I/O footprint of the
    // server: one for accept/routing, one spare for a handler running
    // `spawn_blocking` during startup. The pump tasks and SSE streams
    // are cheap green tasks and don't need more threads.
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ox: failed to build tokio runtime: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ox: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    // Workspace root is the current working directory. Keeping the
    // server scoped to one root means the layout file can use CWD as
    // the key without any user-facing flag.
    let workspace_root = match cli.workspace {
        Some(p) => p.canonicalize().context("resolving --workspace path")?,
        None => std::env::current_dir().context("resolving current working directory")?,
    };

    // Startup gate: refuse to launch outside a git working copy. The
    // worktree / merge flows built on top of `SessionLifecycle` assume
    // these preconditions for every session, so the earlier they fail
    // the less confusing the error becomes.
    let git: Arc<dyn Git> = Arc::new(CliGit::new());
    let workspace_ctx = assert_workspace_ready(git.as_ref(), &workspace_root).await?;

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .context("OPENROUTER_API_KEY is not set — export it before launching ox")?;

    // The slug generator needs the same OpenRouter credentials the
    // agent does, plus a model id. We clone the key so the spawn_config
    // can still move `api_key` into its env list below.
    let slug_api_key = api_key.clone();
    let slug_model = cli.model.clone();

    let agent_binary = resolve_agent_binary().context("locating the ox-agent binary")?;

    let home = dirs::home_dir().context("resolving the user's home directory")?;
    let ox_dir = home.join(".ox");
    let slug = workspace_slug(&workspace_root);
    let sessions_dir = ox_dir.join("workspaces").join(&slug).join("sessions");
    let layout_path = ox_dir.join("workspaces.json");

    let layout: Arc<dyn LayoutRepository> =
        Arc::new(match DiskLayoutRepository::load(layout_path.clone()) {
            Ok(l) => l,
            Err(err) => {
                // A corrupt layout file should not prevent the server from
                // starting. Move the bad file aside and reload from the
                // (now-missing) canonical path — `DiskLayoutRepository::load`
                // returns an empty store when the file doesn't exist.
                let bak = layout_path.with_extension("bak");
                eprintln!(
                    "ox: failed to load {}: {err:#}; moving aside to {} and starting empty",
                    layout_path.display(),
                    bak.display()
                );
                let _ = std::fs::rename(&layout_path, &bak);
                DiskLayoutRepository::load(layout_path.clone())
                    .context("reloading layout after corrupt-file fallback")?
            }
        });

    let spawn_config = AgentSpawnConfig {
        binary: agent_binary,
        workspace_root: workspace_root.clone(),
        model: cli.model.clone(),
        sessions_dir,
        resume: None,
        session_id: None,
        env: vec![("OPENROUTER_API_KEY".into(), api_key)],
    };

    let spawner = Arc::new(ProcessSpawner);

    // Two-phase init to break the registry↔lifecycle cycle:
    //
    //   1. Build the lifecycle with its `Weak<SessionRegistry>` empty.
    //   2. Build the registry, handing it `lifecycle` as its close sink.
    //   3. Call `lifecycle.set_registry(..)` so the coordinator can
    //      reach the registry through a weak-ref.
    //
    let slug_generator: Arc<dyn SlugGenerator> =
        Arc::new(OpenRouterSlugGenerator::new(slug_api_key, slug_model));
    let session_store = Arc::new(
        DiskSessionStore::new(&spawn_config.sessions_dir)
            .context("creating the session store directory")?,
    );
    let session_records: Arc<dyn SessionRecords> = session_store.clone();
    let lifecycle =
        SessionLifecycle::new(git, slug_generator, session_records.clone(), workspace_ctx);
    let close_sink: Arc<dyn CloseRequestSink> = lifecycle.clone();
    let first_turn_sink: Arc<dyn FirstTurnSink> = lifecycle.clone();

    let registry = if let Some(id) = cli.resume {
        // `--resume <id>` bypasses the saved layout and opens exactly
        // one pane for that id. The rest of the saved workspace is
        // untouched — a subsequent normal launch restores it.
        let reg = SessionRegistry::new(
            spawner.clone(),
            spawn_config.clone(),
            layout.clone(),
            workspace_root.clone(),
            close_sink.clone(),
            first_turn_sink.clone(),
        );
        resume_single(&reg, id, session_records.as_ref()).await?;
        reg
    } else {
        SessionRegistry::restore(
            spawner.clone(),
            spawn_config.clone(),
            layout,
            workspace_root.clone(),
            close_sink.clone(),
            first_turn_sink.clone(),
            session_records.clone(),
        )
        .await?
    };

    // Phase 3: close the init cycle. After this, any lifecycle method
    // that upgrades the weak-ref can reach the registry.
    lifecycle.set_registry(Arc::downgrade(&registry));

    let state = AppState {
        registry: registry.clone(),
        lifecycle: lifecycle.clone(),
    };
    let app = router(state);

    let addr = format!("127.0.0.1:{}", cli.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    let url = format!("http://{addr}");
    println!("ox: listening on {url}");

    if !cli.no_open
        && let Err(err) = open_browser(&url)
    {
        eprintln!("ox: could not open browser ({err:#}); visit {url} manually");
    }

    // Graceful shutdown: wait for the signal, then tear down every
    // live session so their broadcast senders drop. Without this step,
    // axum waits forever on the open SSE response bodies.
    let registry_for_shutdown = registry.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            registry_for_shutdown.shutdown();
        })
        .await
        .context("axum serve")?;

    // Best-effort final layout write. The drag-end and session-create
    // handlers already persist on every change, so this is only
    // relevant when the server has run entirely idle since the last
    // mutation — but it's cheap insurance against a missed write.
    if let Err(err) = registry.persist_current_layout() {
        eprintln!("ox: failed to persist layout on shutdown: {err:#}");
    }

    Ok(())
}

/// Resolve the `ox-agent` binary by looking next to the current
/// executable (`target/debug/` in dev; install dir in production).
/// Users can override the search by putting `ox-agent` first on
/// `PATH` and setting `OX_AGENT_BINARY` — we honor that env var for
/// ad-hoc testing without having to shuffle binaries around.
fn resolve_agent_binary() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("OX_AGENT_BINARY") {
        return Ok(PathBuf::from(explicit));
    }
    let exe = std::env::current_exe().context("resolving current executable path")?;
    let dir = exe
        .parent()
        .context("current executable has no parent directory")?;
    let candidate = dir.join("ox-agent");
    if candidate.exists() {
        return Ok(candidate);
    }
    // Fall back to whatever `ox-agent` resolves to on PATH — the OS
    // will error cleanly at spawn time if it isn't there.
    Ok(PathBuf::from("ox-agent"))
}

/// Restore a single named session. Used by `--resume <id>`.
async fn resume_single(
    registry: &SessionRegistry,
    id: SessionId,
    session_store: &dyn SessionRecords,
) -> Result<()> {
    // `restore`'s fallback path produces a one-pane registry already;
    // we replicate that shape here by creating the resumed session and
    // persisting a minimal layout so a later plain `ox` launch finds
    // something to resume.
    let _new_id = registry
        .create_resumed(id, session_store)
        .await
        .with_context(|| format!("resuming session {id}"))?;
    Ok(())
}

/// Resolve when the first of SIGINT or SIGTERM arrives. axum awaits
/// this future to stop accepting new connections; in-flight requests
/// finish on their own before `serve` returns.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = term.recv() => {},
            _ = int.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    println!("ox: shutdown signal received; draining");
}

/// Launch the system browser at `url`. Uses the platform-native opener
/// program and discards its output — if it fails, the caller already
/// printed the URL so the user can paste it manually.
fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "linux")]
    let program = "xdg-open";
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(target_os = "windows")]
    let program = "cmd";

    let mut cmd = std::process::Command::new(program);
    #[cfg(target_os = "windows")]
    cmd.args(["/c", "start", "", url]);
    #[cfg(not(target_os = "windows"))]
    cmd.arg(url);

    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("spawning {program}"))?;
    Ok(())
}
