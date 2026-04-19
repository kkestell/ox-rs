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
use adapter_llm::{OpenRouterSlugGenerator, ProvidersCatalog};
use adapter_process::ProcessSpawner;
use adapter_storage::{DiskLayoutRepository, DiskSessionStore};
use agent_host::{
    AgentSpawnConfig, CloseRequestSink, FirstTurnSink, Git, LayoutRepository, SlugGenerator,
    workspace_slug,
};
use anyhow::{Context, Result};
use app::config::{ProvidersConfig, Settings};
use app::{ModelCatalog, SessionStore};
use clap::Parser;
use domain::SessionId;
use tokio::net::TcpListener;

use crate::lifecycle::{
    ChannelCloseSink, ChannelFirstTurnSink, CloseRequestMsg, FirstTurnMsg, SessionLifecycle,
};
use crate::registry::SessionRegistry;
use crate::routes::router;
use crate::startup::assert_workspace_ready;
use crate::state::AppState;

/// Default port for the local HTTP server.
const DEFAULT_PORT: u16 = 3737;

#[derive(Parser, Debug)]
#[command(name = "ox", about = "Local web UI for Ox sessions")]
struct Cli {
    /// Resume a single session by id, bypassing the saved layout.
    #[arg(long)]
    resume: Option<SessionId>,

    /// Workspace directory to open (defaults to current directory).
    #[arg(long)]
    workspace: Option<PathBuf>,

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

    let agent_binary = resolve_agent_binary().context("locating the ox-agent binary")?;

    let home = dirs::home_dir().context("resolving the user's home directory")?;
    let ox_dir = home.join(".ox");
    let providers_path = ox_dir.join("providers.json");
    let settings_path = ox_dir.join("settings.json");
    if ProvidersConfig::write_shipped_default_if_missing(&providers_path)? {
        eprintln!(
            "ox: wrote shipped providers config to {}",
            providers_path.display()
        );
    }
    if Settings::write_shipped_default_if_missing(&settings_path)? {
        eprintln!(
            "ox: wrote shipped settings config to {}",
            settings_path.display()
        );
    }
    let providers = Arc::new(ProvidersConfig::load(&providers_path)?);
    let settings = Settings::load(&settings_path)?;
    settings.validate(&providers)?;
    let default_model = settings.default_model.clone();
    if !providers.is_wired_model(&default_model) {
        anyhow::bail!(
            "default model {:?} is not backed by a wired provider",
            default_model
        );
    }
    let catalog: Arc<dyn ModelCatalog> = Arc::new(ProvidersCatalog::new(providers.clone()));

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .context("OPENROUTER_API_KEY is not set — export it before launching ox")?;
    let slug_api_key = api_key.clone();
    let slug_model = default_model.clone();

    let slug = workspace_slug(&workspace_root);
    let sessions_dir = ox_dir.join("workspaces").join(&slug).join("sessions");
    let layout_path = ox_dir.join("workspaces.json");

    let layout: Arc<dyn LayoutRepository> = Arc::new(
        match DiskLayoutRepository::load(layout_path.clone()).await {
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
                    .await
                    .context("reloading layout after corrupt-file fallback")?
            }
        },
    );

    let spawn_config = AgentSpawnConfig {
        binary: agent_binary,
        workspace_root: workspace_root.clone(),
        sessions_dir,
        resume: None,
        session_id: None,
        env: vec![("OPENROUTER_API_KEY".into(), api_key)],
    };

    let spawner = Arc::new(ProcessSpawner);

    // Single-phase init: the registry holds channel-backed sinks, the
    // lifecycle holds no back-reference to the registry, and two
    // consumer tasks drain the channels and call lifecycle methods
    // with a weak registry handle. The registry drops naturally on
    // shutdown; the consumer tasks exit when their `Weak` upgrade
    // fails or when every sender (one per session) has been dropped.
    let slug_generator: Arc<dyn SlugGenerator> =
        Arc::new(OpenRouterSlugGenerator::new(slug_api_key, slug_model));
    let session_store = Arc::new(
        DiskSessionStore::new(&spawn_config.sessions_dir)
            .context("creating the session store directory")?,
    );
    let session_store_dyn: Arc<dyn SessionStore> = session_store.clone();
    let lifecycle = SessionLifecycle::new(
        git,
        slug_generator,
        session_store_dyn.clone(),
        workspace_ctx,
    );

    let (close_tx, close_rx) = tokio::sync::mpsc::unbounded_channel::<CloseRequestMsg>();
    let (first_turn_tx, first_turn_rx) = tokio::sync::mpsc::unbounded_channel::<FirstTurnMsg>();
    let close_sink: Arc<dyn CloseRequestSink> = Arc::new(ChannelCloseSink::new(close_tx));
    let first_turn_sink: Arc<dyn FirstTurnSink> =
        Arc::new(ChannelFirstTurnSink::new(first_turn_tx));

    let registry = if let Some(id) = cli.resume {
        // `--resume <id>` bypasses the saved layout and opens exactly
        // one pane for that id. The rest of the saved workspace is
        // untouched — a subsequent normal launch restores it.
        let reg = SessionRegistry::new(
            spawner.clone(),
            spawn_config.clone(),
            layout.clone(),
            workspace_root.clone(),
            catalog.clone(),
            default_model.clone(),
            close_sink.clone(),
            first_turn_sink.clone(),
        );
        resume_single(&reg, id, session_store_dyn.as_ref()).await?;
        reg
    } else {
        SessionRegistry::restore(
            spawner.clone(),
            spawn_config.clone(),
            layout,
            workspace_root.clone(),
            catalog.clone(),
            default_model.clone(),
            close_sink.clone(),
            first_turn_sink.clone(),
            session_store_dyn.clone(),
        )
        .await?
    };

    spawn_close_consumer(close_rx, lifecycle.clone(), Arc::downgrade(&registry));
    spawn_first_turn_consumer(first_turn_rx, lifecycle.clone(), Arc::downgrade(&registry));

    let state = AppState {
        registry: registry.clone(),
        lifecycle: lifecycle.clone(),
        providers: providers.clone(),
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
    if let Err(err) = registry.persist_current_layout().await {
        eprintln!("ox: failed to persist layout on shutdown: {err:#}");
    }

    Ok(())
}

/// Drain close-request messages and dispatch them to the lifecycle
/// against a live registry. The task exits when the registry is
/// dropped (weak-upgrade fails) or when every session-held sender has
/// been dropped and the channel is closed.
fn spawn_close_consumer(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<CloseRequestMsg>,
    lifecycle: Arc<SessionLifecycle>,
    registry: std::sync::Weak<SessionRegistry>,
) {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let Some(reg) = registry.upgrade() else {
                break;
            };
            lifecycle
                .handle_close_request(msg.id, msg.intent, &reg)
                .await;
        }
    });
}

/// Drain first-turn messages and dispatch them to the lifecycle's
/// slug-rename hook. Exit conditions mirror the close consumer.
fn spawn_first_turn_consumer(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<FirstTurnMsg>,
    lifecycle: Arc<SessionLifecycle>,
    registry: std::sync::Weak<SessionRegistry>,
) {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let Some(reg) = registry.upgrade() else {
                break;
            };
            lifecycle
                .handle_first_turn(msg.id, msg.first_message, &reg)
                .await;
        }
    });
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
    session_store: &dyn SessionStore,
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
