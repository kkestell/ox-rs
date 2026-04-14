use std::path::Path;

use anyhow::Result;

pub struct LocalFileSystem;

impl app::FileSystem for LocalFileSystem {
    fn read(&self, path: &Path) -> Result<String> {
        Ok(std::fs::read_to_string(path)?)
    }

    fn write(&self, path: &Path, content: &str) -> Result<()> {
        Ok(std::fs::write(path, content)?)
    }
}

pub struct BashShell;

impl app::Shell for BashShell {
    async fn run(&self, _command: &str) -> Result<app::CommandOutput> {
        // TODO: implement shell execution
        todo!()
    }
}
