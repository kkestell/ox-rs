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

pub struct BashShell {
    workspace_root: PathBuf,
}

impl BashShell {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }
}

impl app::Shell for BashShell {
    async fn run(&self, command: &str, timeout: std::time::Duration) -> Result<app::CommandOutput> {
        use tokio::io::AsyncReadExt;

        let mut child = tokio::process::Command::new("/bin/bash")
            .args(["-c", command])
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("failed to spawn /bin/bash")?;

        // Take ownership of the pipe handles so we can read them concurrently.
        // Reading both via tokio::join! avoids pipe-buffer deadlocks that would
        // occur if we read one to completion before touching the other.
        let mut stdout_pipe = child.stdout.take().expect("stdout piped");
        let mut stderr_pipe = child.stderr.take().expect("stderr piped");

        let io_and_wait = async {
            let mut stdout_buf = Vec::new();
            let mut stderr_buf = Vec::new();
            let (stdout_res, stderr_res) = tokio::join!(
                stdout_pipe.read_to_end(&mut stdout_buf),
                stderr_pipe.read_to_end(&mut stderr_buf),
            );
            stdout_res.context("reading stdout")?;
            stderr_res.context("reading stderr")?;
            let status = child.wait().await.context("waiting for child")?;
            Ok::<_, anyhow::Error>((stdout_buf, stderr_buf, status))
        };

        match tokio::time::timeout(timeout, io_and_wait).await {
            Ok(Ok((stdout_buf, stderr_buf, status))) => Ok(app::CommandOutput {
                stdout: String::from_utf8_lossy(&stdout_buf).into_owned(),
                stderr: String::from_utf8_lossy(&stderr_buf).into_owned(),
                exit_code: status.code().unwrap_or(-1),
                timed_out: false,
            }),
            Ok(Err(e)) => Err(e),
            Err(_elapsed) => {
                // Timeout — kill the process and recover any partial output.
                let _ = child.kill().await;
                Ok(app::CommandOutput {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: -1,
                    timed_out: true,
                })
            }
        }
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

    // -- BashShell tests --

    use app::Shell;

    fn shell(dir: &std::path::Path) -> BashShell {
        BashShell::new(dir.to_path_buf())
    }

    /// Default timeout for BashShell integration tests — generous enough that
    /// even slow CI runners won't flake.
    const TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    #[tokio::test]
    async fn bash_echo_captures_stdout() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("echo hello", TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert_eq!(out.exit_code, 0);
        assert!(!out.timed_out);
    }

    #[tokio::test]
    async fn bash_pwd_uses_workspace_root() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path()).run("pwd", TEST_TIMEOUT).await.unwrap();
        // Canonicalize both sides because tmpdir may be behind a symlink
        // (e.g. /tmp -> /private/tmp on macOS).
        let expected = tmp.path().canonicalize().unwrap();
        let actual = PathBuf::from(out.stdout.trim()).canonicalize().unwrap();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("echo oops >&2", TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(out.stderr.trim(), "oops");
        assert!(out.stdout.trim().is_empty());
    }

    #[tokio::test]
    async fn bash_nonzero_exit_code() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("exit 42", TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(out.exit_code, 42);
        assert!(!out.timed_out);
    }

    #[tokio::test]
    async fn bash_timeout_kills_process() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("sleep 999", std::time::Duration::from_millis(100))
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, -1);
        assert!(out.stdout.is_empty());
        assert!(out.stderr.is_empty());
    }

    #[tokio::test]
    async fn bash_concurrent_stdout_stderr_no_deadlock() {
        let tmp = TempDir::new().unwrap();
        // Write enough data to both pipes that sequential reads would
        // deadlock on a full pipe buffer (~64KB on Linux).
        let out = shell(tmp.path())
            .run(
                "for i in $(seq 1 5000); do echo out_$i; echo err_$i >&2; done",
                TEST_TIMEOUT,
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(!out.timed_out);
        // Spot-check that output from both pipes was captured.
        assert!(out.stdout.contains("out_5000"));
        assert!(out.stderr.contains("err_5000"));
    }
}
