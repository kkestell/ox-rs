use anyhow::{Context, Result};
use app::{SecretStore, SessionRunner, SessionStore};
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

    // Cheap model suitable for development and testing.
    let model = "mistralai/mistral-nemo".to_owned();

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
    let runner = SessionRunner::new(llm, store);

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
