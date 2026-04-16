use std::path::{Path, PathBuf};

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

    async fn walk_glob(&self, root: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let full_pattern = root.join(pattern);
        let full_pattern = full_pattern
            .to_str()
            .context("glob pattern contains invalid UTF-8")?;
        let mut results: Vec<PathBuf> = glob::glob(full_pattern)
            .map_err(|e| anyhow::anyhow!("invalid glob pattern: {e}"))?
            .filter_map(|entry| {
                // Skip entries that fail to read (e.g. permission errors).
                let path = entry.ok()?;
                // Only include files, not directories.
                if path.is_file() { Some(path) } else { None }
            })
            .collect();
        results.sort();
        Ok(results)
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

    // -- walk_glob tests --

    /// Helper: create a file at `path` with dummy content.
    async fn touch(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.unwrap();
        }
        tokio::fs::write(path, "x").await.unwrap();
    }

    #[tokio::test]
    async fn walk_glob_matches_simple_pattern() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("a.rs")).await;
        touch(&tmp.path().join("b.txt")).await;
        touch(&tmp.path().join("c.rs")).await;

        let results = fs().walk_glob(tmp.path(), "*.rs").await.unwrap();
        let names: Vec<_> = results.iter().map(|p| p.file_name().unwrap()).collect();
        assert_eq!(names, vec!["a.rs", "c.rs"]);
    }

    #[tokio::test]
    async fn walk_glob_matches_recursive_pattern() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("src/main.rs")).await;
        touch(&tmp.path().join("src/lib.rs")).await;
        touch(&tmp.path().join("tests/it.rs")).await;
        touch(&tmp.path().join("README.md")).await;

        let results = fs().walk_glob(tmp.path(), "**/*.rs").await.unwrap();
        assert_eq!(results.len(), 3);
        // All results should be .rs files.
        assert!(results.iter().all(|p| p.extension().unwrap() == "rs"));
    }

    #[tokio::test]
    async fn walk_glob_returns_empty_on_no_match() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("a.txt")).await;

        let results = fs().walk_glob(tmp.path(), "*.rs").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn walk_glob_excludes_directories() {
        let tmp = TempDir::new().unwrap();
        // Create a directory that matches the glob but isn't a file.
        tokio::fs::create_dir_all(tmp.path().join("subdir.rs"))
            .await
            .unwrap();
        touch(&tmp.path().join("file.rs")).await;

        let results = fs().walk_glob(tmp.path(), "*.rs").await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].ends_with("file.rs"));
    }

    #[tokio::test]
    async fn walk_glob_invalid_pattern_returns_error() {
        let tmp = TempDir::new().unwrap();
        let result = fs().walk_glob(tmp.path(), "[invalid").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn walk_glob_results_are_sorted() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("c.txt")).await;
        touch(&tmp.path().join("a.txt")).await;
        touch(&tmp.path().join("b.txt")).await;

        let results = fs().walk_glob(tmp.path(), "*.txt").await.unwrap();
        let names: Vec<_> = results
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[tokio::test]
    async fn walk_glob_nonexistent_root_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does_not_exist");
        // glob::glob on a non-existent prefix returns an empty iterator,
        // not an error — so walk_glob returns an empty vec.
        let results = fs().walk_glob(&missing, "*.rs").await.unwrap();
        assert!(results.is_empty());
    }
}
