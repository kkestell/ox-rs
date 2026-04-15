use std::sync::Arc;

use anyhow::{Context, Result};
use app::tools::{EditFileTool, ReadFileTool, ToolRegistry, WriteFileTool};
use app::{SecretStore, SessionRunner, SessionStore, Tool};
use clap::Parser;
use domain::SessionId;
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "ox", about = "Desktop AI coding assistant")]
struct Cli {
    /// Resume a previous session by its ID.
    #[arg(long)]
    resume: Option<SessionId>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read the API key before creating the runtime — fail fast on missing config.
    let secrets = adapter_secrets::EnvSecretStore;
    let api_key = secrets
        .get("OPENROUTER_API_KEY")?
        .context("OPENROUTER_API_KEY is not set — export it and try again")?;

    // DeepSeek R1 — reasoning-capable flagship, good for exercising the
    // thinking/tool-call rendering path.
    let model = "deepseek/deepseek-r1".to_owned();

    let sessions_dir = dirs::home_dir()
        .context("could not determine home directory")?
        .join(".ox/sessions");
    let store = adapter_storage::DiskSessionStore::new(sessions_dir)?;

    let workspace_root =
        std::env::current_dir().context("could not determine working directory")?;

    // If resuming, load the session now so we can pre-populate the GUI with
    // history and hand the session ID to the backend. Fail fast if the
    // session doesn't exist — no point opening a window for a bad ID.
    let rt = tokio::runtime::Runtime::new()?;
    let (initial_messages, resume_id) = match cli.resume {
        Some(id) => {
            let session = rt
                .block_on(store.load(id))
                .with_context(|| format!("could not load session {id}"))?;
            (session.messages, Some(id))
        }
        None => (Vec::new(), None),
    };

    let llm = adapter_llm::OpenRouterProvider::new(api_key, model);

    // Wire the file-editing tools. An `Arc<LocalFileSystem>` is shared so
    // each tool observes the same filesystem; the workspace root is cloned
    // into each tool for path resolution (absolute paths bypass it).
    let fs = Arc::new(adapter_fs::LocalFileSystem);
    let mut tools = ToolRegistry::new();
    tools
        .register(Arc::new(ReadFileTool::new(fs.clone(), workspace_root.clone())) as Arc<dyn Tool>);
    tools.register(
        Arc::new(WriteFileTool::new(fs.clone(), workspace_root.clone())) as Arc<dyn Tool>,
    );
    tools
        .register(Arc::new(EditFileTool::new(fs.clone(), workspace_root.clone())) as Arc<dyn Tool>);

    let runner = SessionRunner::new(llm, store, tools);

    // Channels between the GUI and the async backend controller.
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (evt_tx, evt_rx) = mpsc::unbounded_channel();

    // The tokio runtime drives the backend; eframe blocks the main thread.
    let handle = rt.spawn(adapter_egui::backend::run_backend(
        runner,
        cmd_rx,
        evt_tx,
        workspace_root,
        resume_id,
    ));

    let app = adapter_egui::OxApp::new(cmd_tx, evt_rx, initial_messages);
    app.run()?;

    // After the GUI exits, collect the final session ID from the backend so
    // the user knows how to pick up where they left off.
    if let Ok(Some(id)) = rt.block_on(handle) {
        eprintln!("ox --resume {id}");
    }

    Ok(())
}
