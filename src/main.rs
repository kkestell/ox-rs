use anyhow::Result;

fn main() -> Result<()> {
    // Composition root: wire adapters to ports, start the app.
    //
    // let llm = adapter_llm::AnthropicProvider::new(api_key);
    // let store = adapter_storage::DiskSessionStore::new(sessions_dir)?;
    // let fs = adapter_fs::LocalFileSystem;
    // let shell = adapter_fs::BashShell;
    // let use_case = app::ContinueSession::new(llm, store);
    // let tui = adapter_tui::TuiApp::new();
    // tui.run()

    println!("ox");
    Ok(())
}
