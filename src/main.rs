use anyhow::Result;

fn main() -> Result<()> {
    // Composition root: wire adapters to ports, start the app.
    //
    // let llm = adapter_llm::OpenRouterProvider::new(api_key, model);
    // let store = adapter_storage::DiskSessionStore::new(sessions_dir)?;
    // let fs = adapter_fs::LocalFileSystem;
    // let shell = adapter_fs::BashShell;
    // let runner = app::SessionRunner::new(llm, store);
    // let id = domain::SessionId::new_v4();
    // let gui = adapter_egui::OxApp::new();
    // gui.run()

    println!("ox");
    Ok(())
}
