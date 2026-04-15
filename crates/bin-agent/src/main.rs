//! `ox-agent` — the headless session runner.
//!
//! One agent process == one session. The GUI spawns this binary with the
//! `SessionId` (if resuming), model, sessions directory, and workspace root on
//! the command line, and the API key via env. The agent then drives its
//! `SessionRunner` over NDJSON frames on stdin/stdout. Everything about the
//! session — loading, saving, tool execution, LLM streaming — lives here, not
//! in the GUI.
//!
//! Errors during startup print to stderr and set a non-zero exit code; once
//! the agent is in its main loop, per-turn errors surface as
//! `AgentEvent::Error` frames and the loop keeps running.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use app::tools::{EditFileTool, ReadFileTool, WriteFileTool};
use app::{SecretStore, SessionRunner, Tool, ToolRegistry};
use clap::Parser;
use domain::SessionId;
use protocol::{AgentEvent, write_frame};
use tokio::io::{AsyncWriteExt, BufReader};

mod driver;

#[derive(Parser, Debug)]
#[command(name = "ox-agent", about = "Headless session runner driven over stdio")]
struct AgentCli {
    /// Workspace root the agent's file tools resolve relative paths against.
    #[arg(long)]
    workspace_root: PathBuf,

    /// OpenRouter model ID (e.g. `deepseek/deepseek-r1`). Passed through
    /// verbatim to the provider.
    #[arg(long)]
    model: String,

    /// Directory where session JSON files are stored.
    #[arg(long)]
    sessions_dir: PathBuf,

    /// Resume a pre-existing session by its UUID. Without this flag a new
    /// session is created lazily when the first `SendMessage` arrives.
    #[arg(long)]
    resume: Option<SessionId>,
}

fn main() -> ExitCode {
    let cli = AgentCli::parse();
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ox-agent: failed to start tokio runtime: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    match rt.block_on(run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Best-effort: emit an `Error` frame so the GUI observes a
            // structured failure rather than "agent died silently," then also
            // echo to stderr for when ox-agent is driven by hand.
            let mut stdout = tokio::io::stdout();
            let msg = format!("{e:#}");
            let _ = rt.block_on(write_frame(
                &mut stdout,
                &AgentEvent::Error {
                    message: msg.clone(),
                },
            ));
            let _ = rt.block_on(stdout.flush());
            eprintln!("ox-agent: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Top-level async entry point. Wires adapters, builds the `SessionRunner`,
/// and hands control to the `driver` loop reading from stdin / writing to
/// stdout.
async fn run(cli: AgentCli) -> Result<()> {
    let secrets = adapter_secrets::EnvSecretStore;
    let api_key = secrets
        .get("OPENROUTER_API_KEY")?
        .context("OPENROUTER_API_KEY is not set — export it before launching ox-agent")?;

    // Build the runner that processes turns. The driver holds a *second*
    // `DiskSessionStore` handle so it can independently preload history for
    // `--resume`; both handles point at the same directory, and
    // `DiskSessionStore` is stateless beyond its root path.
    let llm = adapter_llm::OpenRouterProvider::new(api_key, cli.model.clone());

    // File tools share a single `LocalFileSystem` so concurrent tool calls in
    // the same turn see a consistent filesystem view.
    let fs = Arc::new(adapter_fs::LocalFileSystem);
    let mut tools = ToolRegistry::new();
    tools.register(
        Arc::new(ReadFileTool::new(fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>,
    );
    tools.register(
        Arc::new(WriteFileTool::new(fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>,
    );
    tools.register(
        Arc::new(EditFileTool::new(fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>,
    );

    let store = adapter_storage::DiskSessionStore::new(cli.sessions_dir.clone())?;
    let history_store = adapter_storage::DiskSessionStore::new(cli.sessions_dir)?;
    let runner = SessionRunner::new(llm, store, tools);

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);

    driver::agent_driver(
        &runner,
        &history_store,
        cli.workspace_root,
        cli.resume,
        reader,
        stdout,
    )
    .await
}
