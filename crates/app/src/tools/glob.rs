//! `glob` tool.
//!
//! Finds files by name pattern within the workspace. Returns
//! workspace-relative paths sorted alphabetically, truncated at a cap to
//! keep context-window cost bounded.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::ports::FileSystem;
use crate::stream::ToolDef;

/// Maximum number of file paths returned before truncation.
const MAX_RESULTS: usize = 200;

pub struct GlobTool<F> {
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<F> GlobTool<F> {
    pub fn new(fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self { fs, workspace_root }
    }
}

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    /// Directory to search in. Absolute or workspace-relative. Defaults to
    /// the workspace root when omitted.
    #[serde(default)]
    path: Option<String>,
}

impl<F: FileSystem + Send + Sync + 'static> Tool for GlobTool<F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "glob".into(),
            description: "Find files by name pattern. Returns workspace-relative paths, \
                sorted alphabetically. Use glob syntax: `*.rs` for the current directory, \
                `**/*.rs` for recursive search."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. `**/*.rs`, `src/*.toml`)."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in, absolute or workspace-relative. Defaults to workspace root."
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: GlobArgs =
                serde_json::from_str(args).context("glob: invalid JSON arguments")?;
            require_non_empty("pattern", &parsed.pattern)?;

            let root = match &parsed.path {
                Some(p) => resolve_path(&self.workspace_root, p),
                None => self.workspace_root.clone(),
            };

            let matches = self
                .fs
                .walk_glob(&root, &parsed.pattern)
                .await
                .with_context(|| {
                    format!("glob: walk_glob failed for pattern {:?}", parsed.pattern)
                })?;

            if matches.is_empty() {
                return Ok("No files matched.".into());
            }

            let total = matches.len();
            let truncated = total > MAX_RESULTS;
            let display: Vec<String> = matches
                .iter()
                .take(MAX_RESULTS)
                .map(|p| display_path(&self.workspace_root, p))
                .collect();

            let mut out = if truncated {
                format!("Found {total} files (showing first {MAX_RESULTS}):\n")
            } else {
                format!("Found {total} file{}:\n", if total == 1 { "" } else { "s" })
            };
            out.push_str(&display.join("\n"));
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;

    fn tool(fs: Arc<FakeFileSystem>, root: &str) -> GlobTool<FakeFileSystem> {
        GlobTool::new(fs, PathBuf::from(root))
    }

    #[tokio::test]
    async fn returns_workspace_relative_paths_sorted() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/src/c.rs", "");
        fs.insert("/ws/src/a.rs", "");
        fs.insert("/ws/src/b.rs", "");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"**/*.rs"}"#).await.unwrap();
        assert!(out.contains("Found 3 files:"));
        let lines: Vec<_> = out.lines().skip(1).collect();
        assert_eq!(lines, vec!["src/a.rs", "src/b.rs", "src/c.rs"]);
    }

    #[tokio::test]
    async fn respects_path_parameter() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/src/main.rs", "");
        fs.insert("/ws/tests/test.rs", "");
        let t = tool(fs, "/ws");

        let out = t
            .execute(r#"{"pattern":"*.rs","path":"src"}"#)
            .await
            .unwrap();
        assert!(out.contains("Found 1 file:"));
        assert!(out.contains("src/main.rs"));
        assert!(!out.contains("tests/test.rs"));
    }

    #[tokio::test]
    async fn defaults_to_workspace_root() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "");
        fs.insert("/ws/b.txt", "");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"*.txt"}"#).await.unwrap();
        assert!(out.contains("Found 2 files:"));
    }

    #[tokio::test]
    async fn absolute_path_parameter() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/other/f.rs", "");
        fs.insert("/ws/g.rs", "");
        let t = tool(fs, "/ws");

        let out = t
            .execute(r#"{"pattern":"*.rs","path":"/other"}"#)
            .await
            .unwrap();
        assert!(out.contains("Found 1 file:"));
        // Path outside workspace shows as absolute.
        assert!(out.contains("/other/f.rs"));
    }

    #[tokio::test]
    async fn no_matches_returns_message() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"*.rs"}"#).await.unwrap();
        assert_eq!(out, "No files matched.");
    }

    #[tokio::test]
    async fn invalid_pattern_returns_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");

        let result = t.execute(r#"{"pattern":"[invalid"}"#).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn truncates_at_cap() {
        let fs = Arc::new(FakeFileSystem::new());
        // Insert more files than MAX_RESULTS.
        for i in 0..250 {
            fs.insert(format!("/ws/file_{i:04}.txt"), "");
        }
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"*.txt"}"#).await.unwrap();
        assert!(out.contains("Found 250 files (showing first 200):"));
        // Count the file lines (skip the header).
        let file_lines = out.lines().skip(1).count();
        assert_eq!(file_lines, 200);
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute("not json").await.is_err());
    }

    #[tokio::test]
    async fn empty_pattern_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute(r#"{"pattern":""}"#).await.is_err());
    }
}
