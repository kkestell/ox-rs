use anyhow::{Context, Result};
use app::{SecretStore, SessionRunner};
use tokio::sync::mpsc;

fn main() -> Result<()> {
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

    let llm = adapter_llm::OpenRouterProvider::new(api_key, model);
    let runner = SessionRunner::new(llm, store);

    let workspace_root =
        std::env::current_dir().context("could not determine working directory")?;

    // Channels between the GUI and the async backend controller.
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (evt_tx, evt_rx) = mpsc::unbounded_channel();

    // The tokio runtime drives the backend; eframe blocks the main thread.
    let rt = tokio::runtime::Runtime::new()?;
    rt.spawn(adapter_egui::backend::run_backend(
        runner,
        cmd_rx,
        evt_tx,
        workspace_root,
    ));

    let app = adapter_egui::OxApp::new(cmd_tx, evt_rx);
    app.run()
}
