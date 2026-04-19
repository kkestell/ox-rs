use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub struct LocalFileSystem;

impl app::FileSystem for LocalFileSystem {
    async fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
        tokio::fs::canonicalize(path)
            .await
            .with_context(|| format!("canonicalizing {}", path.display()))
    }

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

    async fn walk_glob(
        &self,
        root: &Path,
        pattern: &str,
        max_bytes: usize,
    ) -> Result<app::WalkResult> {
        let root = root.to_path_buf();
        let pattern = pattern.to_owned();
        tokio::task::spawn_blocking(move || walk_glob_sync(root, pattern, max_bytes))
            .await
            .context("joining blocking glob task")?
    }
}

fn walk_glob_sync(root: PathBuf, pattern: String, max_bytes: usize) -> Result<app::WalkResult> {
    let full_pattern = root.join(pattern);
    let full_pattern = full_pattern
        .to_str()
        .context("glob pattern contains invalid UTF-8")?;
    let entries =
        glob::glob(full_pattern).map_err(|e| anyhow::anyhow!("invalid glob pattern: {e}"))?;

    let mut results: Vec<PathBuf> = Vec::new();
    let mut cumulative_bytes: usize = 0;
    let mut truncated = false;

    // Collect all matching files first so we can sort, then enforce the byte
    // cap. Sorting before truncation keeps output deterministic.
    let mut all_files: Vec<PathBuf> = entries
        .filter_map(|entry| {
            let path = entry.ok()?;
            if path.is_file() { Some(path) } else { None }
        })
        .collect();
    all_files.sort();

    for path in all_files {
        let path_bytes = path.to_string_lossy().len();
        if cumulative_bytes + path_bytes > max_bytes {
            truncated = true;
            break;
        }
        cumulative_bytes += path_bytes;
        results.push(path);
    }

    Ok(app::WalkResult {
        paths: results,
        truncated,
    })
}

async fn read_bounded(
    mut pipe: impl tokio::io::AsyncRead + Send + Unpin + 'static,
    max_bytes: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut truncated = false;
    loop {
        let n = pipe.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        if !truncated {
            let remaining = max_bytes.saturating_sub(buf.len());
            if remaining == 0 {
                truncated = true;
            } else {
                let take = n.min(remaining);
                buf.extend_from_slice(&tmp[..take]);
                if take < n {
                    truncated = true;
                }
            }
        }
        // Continue draining even after truncation so the child doesn't block
        // on a full pipe buffer.
    }
    Ok((buf, truncated))
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
    async fn run(
        &self,
        command: &str,
        timeout: std::time::Duration,
        max_bytes: usize,
    ) -> Result<app::CommandOutput> {
        let mut child = tokio::process::Command::new("/bin/bash")
            .args(["-c", command])
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("failed to spawn /bin/bash")?;

        // Take ownership of the pipe handles so reader tasks can keep
        // draining after a timeout kill and still hand back partial output.
        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");
        let stdout_reader = tokio::spawn(read_bounded(stdout_pipe, max_bytes));
        let stderr_reader = tokio::spawn(read_bounded(stderr_pipe, max_bytes));

        let (exit_code, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
            Ok(status) => (
                status.context("waiting for child")?.code().unwrap_or(-1),
                false,
            ),
            Err(_elapsed) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                (-1, true)
            }
        };

        let (stdout_buf, stdout_trunc) = stdout_reader
            .await
            .context("joining stdout reader")?
            .context("reading stdout")?;
        let (stderr_buf, stderr_trunc) = stderr_reader
            .await
            .context("joining stderr reader")?
            .context("reading stderr")?;

        Ok(app::CommandOutput {
            stdout: String::from_utf8_lossy(&stdout_buf).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_buf).into_owned(),
            exit_code,
            timed_out,
            truncated: stdout_trunc || stderr_trunc,
        })
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

    /// Large byte cap so existing tests don't trigger truncation.
    const WALK_MAX_BYTES: usize = 10 * 1024 * 1024;

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

        let result = fs()
            .walk_glob(tmp.path(), "*.rs", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert!(!result.truncated);
        let names: Vec<_> = result
            .paths
            .iter()
            .map(|p| p.file_name().unwrap())
            .collect();
        assert_eq!(names, vec!["a.rs", "c.rs"]);
    }

    #[tokio::test]
    async fn walk_glob_matches_recursive_pattern() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("src/main.rs")).await;
        touch(&tmp.path().join("src/lib.rs")).await;
        touch(&tmp.path().join("tests/it.rs")).await;
        touch(&tmp.path().join("README.md")).await;

        let result = fs()
            .walk_glob(tmp.path(), "**/*.rs", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(result.paths.len(), 3);
        assert!(result.paths.iter().all(|p| p.extension().unwrap() == "rs"));
    }

    #[tokio::test]
    async fn walk_glob_returns_empty_on_no_match() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("a.txt")).await;

        let result = fs()
            .walk_glob(tmp.path(), "*.rs", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert!(result.paths.is_empty());
    }

    #[tokio::test]
    async fn walk_glob_excludes_directories() {
        let tmp = TempDir::new().unwrap();
        // Create a directory that matches the glob but isn't a file.
        tokio::fs::create_dir_all(tmp.path().join("subdir.rs"))
            .await
            .unwrap();
        touch(&tmp.path().join("file.rs")).await;

        let result = fs()
            .walk_glob(tmp.path(), "*.rs", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(result.paths.len(), 1);
        assert!(result.paths[0].ends_with("file.rs"));
    }

    #[tokio::test]
    async fn walk_glob_invalid_pattern_returns_error() {
        let tmp = TempDir::new().unwrap();
        let result = fs().walk_glob(tmp.path(), "[invalid", WALK_MAX_BYTES).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn walk_glob_results_are_sorted() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("c.txt")).await;
        touch(&tmp.path().join("a.txt")).await;
        touch(&tmp.path().join("b.txt")).await;

        let result = fs()
            .walk_glob(tmp.path(), "*.txt", WALK_MAX_BYTES)
            .await
            .unwrap();
        let names: Vec<_> = result
            .paths
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
        let result = fs()
            .walk_glob(&missing, "*.rs", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert!(result.paths.is_empty());
    }

    // -- BashShell tests --

    use app::Shell;

    fn shell(dir: &std::path::Path) -> BashShell {
        BashShell::new(dir.to_path_buf())
    }

    /// Default timeout for BashShell integration tests — generous enough that
    /// even slow CI runners won't flake.
    const TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    /// Large byte cap so existing tests don't trigger truncation.
    const TEST_MAX_BYTES: usize = 10 * 1024 * 1024;

    #[tokio::test]
    async fn bash_echo_captures_stdout() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("echo hello", TEST_TIMEOUT, TEST_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert_eq!(out.exit_code, 0);
        assert!(!out.timed_out);
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn bash_pwd_uses_workspace_root() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("pwd", TEST_TIMEOUT, TEST_MAX_BYTES)
            .await
            .unwrap();
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
            .run("echo oops >&2", TEST_TIMEOUT, TEST_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(out.stderr.trim(), "oops");
        assert!(out.stdout.trim().is_empty());
    }

    #[tokio::test]
    async fn bash_nonzero_exit_code() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("exit 42", TEST_TIMEOUT, TEST_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(out.exit_code, 42);
        assert!(!out.timed_out);
    }

    #[tokio::test]
    async fn bash_timeout_kills_process() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run(
                "sleep 999",
                std::time::Duration::from_millis(100),
                TEST_MAX_BYTES,
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, -1);
        assert!(out.stdout.is_empty());
        assert!(out.stderr.is_empty());
    }

    #[tokio::test]
    async fn bash_timeout_preserves_partial_stdout_and_stderr() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run(
                "printf partial-out; printf partial-err >&2; sleep 999",
                std::time::Duration::from_millis(500),
                TEST_MAX_BYTES,
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, -1);
        assert_eq!(out.stdout, "partial-out");
        assert_eq!(out.stderr, "partial-err");
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn bash_timeout_reports_truncation_for_capped_partial_output() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run(
                "head -c 4096 /dev/zero | tr '\\0' x; sleep 999",
                std::time::Duration::from_millis(500),
                32,
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, -1);
        assert!(out.truncated);
        assert_eq!(out.stdout.len(), 32);
        assert!(out.stdout.chars().all(|ch| ch == 'x'));
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
                TEST_MAX_BYTES,
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(!out.timed_out);
        // Spot-check that output from both pipes was captured.
        assert!(out.stdout.contains("out_5000"));
        assert!(out.stderr.contains("err_5000"));
    }

    // -- BashShell max_bytes tests --

    #[tokio::test]
    async fn bash_stdout_under_cap_returns_normally() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("echo hello", TEST_TIMEOUT, 1024)
            .await
            .unwrap();
        assert_eq!(out.stdout.trim(), "hello");
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn bash_stdout_exceeding_cap_is_truncated() {
        let tmp = TempDir::new().unwrap();
        // Generate ~50KB of stdout but cap at 1KB.
        let out = shell(tmp.path())
            .run("seq 1 10000", TEST_TIMEOUT, 1024)
            .await
            .unwrap();
        assert!(out.truncated, "should be truncated");
        assert!(
            out.stdout.len() <= 1024,
            "stdout should be at most 1024 bytes, got {}",
            out.stdout.len()
        );
        // Partial data preserved — should contain early lines.
        assert!(out.stdout.contains('1'));
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn bash_stderr_exceeding_cap_is_truncated() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("seq 1 10000 >&2", TEST_TIMEOUT, 1024)
            .await
            .unwrap();
        assert!(out.truncated, "should be truncated");
        assert!(
            out.stderr.len() <= 1024,
            "stderr should be at most 1024 bytes, got {}",
            out.stderr.len()
        );
    }

    #[tokio::test]
    async fn bash_both_streams_truncated() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run(
                "for i in $(seq 1 10000); do echo out_$i; echo err_$i >&2; done",
                TEST_TIMEOUT,
                1024,
            )
            .await
            .unwrap();
        assert!(out.truncated);
        assert!(out.stdout.len() <= 1024);
        assert!(out.stderr.len() <= 1024);
    }

    #[tokio::test]
    async fn bash_timeout_works_independently_of_max_bytes() {
        let tmp = TempDir::new().unwrap();
        let out = shell(tmp.path())
            .run("sleep 999", std::time::Duration::from_millis(100), 1024)
            .await
            .unwrap();
        assert!(out.timed_out);
        assert!(!out.truncated);
    }

    // -- walk_glob max_bytes tests --

    #[tokio::test]
    async fn walk_glob_under_cap_returns_all() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("a.txt")).await;
        touch(&tmp.path().join("b.txt")).await;
        touch(&tmp.path().join("c.txt")).await;

        let result = fs()
            .walk_glob(tmp.path(), "*.txt", WALK_MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(result.paths.len(), 3);
        assert!(!result.truncated);
    }

    #[tokio::test]
    async fn walk_glob_exceeding_cap_truncates() {
        let tmp = TempDir::new().unwrap();
        // Create many files. Each path is ~60+ bytes (tmpdir prefix + filename).
        for i in 0..100 {
            touch(&tmp.path().join(format!("file_{i:04}.txt"))).await;
        }

        // Use a very small cap — should only fit a few paths.
        let result = fs().walk_glob(tmp.path(), "*.txt", 200).await.unwrap();
        assert!(result.truncated, "should be truncated");
        assert!(
            result.paths.len() < 100,
            "should have fewer than 100 paths, got {}",
            result.paths.len()
        );
        // Results should still be sorted.
        let names: Vec<_> = result
            .paths
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
}
