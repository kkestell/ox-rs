use std::path::Path;

use anyhow::{Context, Result};

pub struct LocalFileSystem;

impl app::FileSystem for LocalFileSystem {
    async fn read(&self, path: &Path) -> Result<String> {
        Ok(tokio::fs::read_to_string(path).await?)
    }

    /// Writes `content` to `path`, creating any missing parent directories
    /// first. This matches the expectation that a write-file tool should
    /// succeed when the LLM targets a new subdirectory — requiring callers
    /// to mkdir-then-write first would be a needless round trip, and no
    /// caller today relies on the strict "fail if parent missing" variant.
    async fn write(&self, path: &Path, content: &str) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating parent directories for {}", path.display()))?;
        }
        tokio::fs::write(path, content)
            .await
            .with_context(|| format!("writing to {}", path.display()))?;
        Ok(())
    }
}

pub struct BashShell;

impl app::Shell for BashShell {
    async fn run(&self, _command: &str) -> Result<app::CommandOutput> {
        // TODO: implement shell execution
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use app::FileSystem;
    use tempfile::TempDir;

    use super::*;

    fn fs() -> LocalFileSystem {
        LocalFileSystem
    }

    #[tokio::test]
    async fn write_creates_missing_parent_dirs() {
        let tmp = TempDir::new().unwrap();
        let nested: PathBuf = tmp.path().join("a/b/c/file.txt");
        fs().write(&nested, "hello").await.unwrap();
        assert_eq!(std::fs::read_to_string(&nested).unwrap(), "hello");
    }

    #[tokio::test]
    async fn write_overwrites_existing_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f.txt");
        fs().write(&path, "first").await.unwrap();
        fs().write(&path, "second").await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }

    #[tokio::test]
    async fn read_returns_file_contents() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("round_trip.txt");
        tokio::fs::write(&path, "expected content").await.unwrap();
        assert_eq!(fs().read(&path).await.unwrap(), "expected content");
    }

    #[tokio::test]
    async fn read_missing_file_returns_error() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope.txt");
        assert!(fs().read(&missing).await.is_err());
    }
}
