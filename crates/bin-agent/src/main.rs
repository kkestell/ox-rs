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

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use app::tools::{BashTool, EditFileTool, ReadFileTool, WriteFileTool};
use app::{GlobTool, GrepTool, SecretStore, SessionRunner, Tool, ToolRegistry};
use chrono::Local;
use clap::Parser;
use domain::SessionId;
use protocol::{AgentEvent, write_frame};
use tokio::io::{AsyncWriteExt, BufReader};

mod driver;

fn build_system_prompt(workspace_root: &Path, model: &str) -> String {
    format!(
        "\
You are an AI coding assistant. You help users understand, write, and modify \
code within their projects.

Environment:
- Workspace root: {workspace_root}
- Platform: {os}
- Model: {model}
- Today's date: {date}

Paths you report to the user must match the workspace root above. Do not \
invent or assume paths — if you don't know something about the environment, \
use a tool to check.

Be concise and direct. Do not narrate what you are about to do or summarize \
what you just did — just do the work and show the result.

Use tools frugally. When the user asks you to do something, do it once and \
correctly. Do not re-run the same command with different flags, do not pipe \
output through head/tail/grep to manage length, and do not run redundant \
verification passes. Tool output that exceeds the inline threshold is \
automatically spilled to a file and a preview is shown — you never need to \
work around output length yourself.

Prefer answering from context when possible. Only call a tool when you \
genuinely need information you don't already have or need to perform an action.",
        workspace_root = workspace_root.display(),
        os = std::env::consts::OS,
        model = model,
        date = Local::now().format("%Y-%m-%d"),
    )
}

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
            // Broken pipe means the GUI side closed our stdout pipe — the
            // expected shutdown path when the host tears the agent down.
            // Don't pollute stderr or attempt to write a final Error frame
            // (that write would also fail with EPIPE).
            if is_broken_pipe(&e) {
                return ExitCode::SUCCESS;
            }
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

/// True when an anyhow error chain bottoms out at a `BrokenPipe` I/O error.
/// The host-tears-down-the-agent path goes through this; we treat it as a
/// clean exit so a normal shutdown doesn't print a confusing stderr line.
fn is_broken_pipe(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe)
    })
}

/// Top-level async entry point. Wires adapters, builds the `SessionRunner`,
/// and hands control to the `driver` loop reading from stdin / writing to
/// stdout.
async fn run(cli: AgentCli) -> Result<()> {
    let secrets = adapter_secrets::EnvSecretStore;
    let api_key = secrets
        .get("OPENROUTER_API_KEY")?
        .context("OPENROUTER_API_KEY is not set — export it before launching ox-agent")?;

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
    tools
        .register(Arc::new(GlobTool::new(fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>);
    tools
        .register(Arc::new(GrepTool::new(fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>);

    let shell = Arc::new(adapter_fs::BashShell::new(cli.workspace_root.clone()));
    tools.register(
        Arc::new(BashTool::new(shell, fs.clone(), cli.workspace_root.clone())) as Arc<dyn Tool>,
    );

    let store = adapter_storage::DiskSessionStore::new(cli.sessions_dir)?;
    let system_prompt = build_system_prompt(&cli.workspace_root, &cli.model);
    let runner = SessionRunner::new(llm, store, tools, system_prompt);

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);

    driver::agent_driver(&runner, cli.workspace_root, cli.resume, reader, stdout).await
}
