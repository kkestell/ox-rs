//! `grep` tool.
//!
//! Searches file contents by regex across the workspace. Walks files via
//! `walk_glob`, reads each one, and collects matching lines with file path
//! and line number. Output is truncated at a cap to keep context-window
//! cost bounded.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::ports::FileSystem;
use crate::stream::ToolDef;

/// Maximum number of matching lines returned before truncation.
const MAX_MATCHES: usize = 200;

/// Default glob pattern when no file filter is provided — matches
/// every file recursively.
const DEFAULT_GLOB: &str = "**/*";

pub struct GrepTool<F> {
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<F> GrepTool<F> {
    pub fn new(fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self { fs, workspace_root }
    }
}

#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    /// Directory to search in. Absolute or workspace-relative. Defaults to
    /// the workspace root when omitted.
    #[serde(default)]
    path: Option<String>,
    /// Glob pattern to filter which files are searched (e.g. `"*.rs"`).
    /// Defaults to `"**/*"` (all files).
    #[serde(default)]
    glob: Option<String>,
}

impl<F: FileSystem + Send + Sync + 'static> Tool for GrepTool<F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "grep".into(),
            description: "Search file contents by regex. Returns matching lines with file \
                paths and line numbers. Supports an optional `glob` filter to restrict \
                which files are searched."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in, absolute or workspace-relative. Defaults to workspace root."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. `*.rs`). Defaults to `**/*`."
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
            let parsed: GrepArgs =
                serde_json::from_str(args).context("grep: invalid JSON arguments")?;
            require_non_empty("pattern", &parsed.pattern)?;

            let re = Regex::new(&parsed.pattern)
                .with_context(|| format!("grep: invalid regex {:?}", parsed.pattern))?;

            let root = match &parsed.path {
                Some(p) => resolve_path(&self.workspace_root, p),
                None => self.workspace_root.clone(),
            };
            let glob_pattern = parsed.glob.as_deref().unwrap_or(DEFAULT_GLOB);

            let files = self
                .fs
                .walk_glob(&root, glob_pattern)
                .await
                .with_context(|| {
                    format!("grep: walk_glob failed for pattern {:?}", glob_pattern)
                })?;

            // Collect matches across all files: (display_path, line_number, line_content).
            let mut matches: Vec<(String, usize, String)> = Vec::new();
            let mut files_with_errors: Vec<String> = Vec::new();
            // Tracks whether we hit the cap and stopped early, as opposed
            // to naturally finding exactly MAX_MATCHES results.
            let mut hit_cap = false;

            for file_path in &files {
                let content = match self.fs.read(file_path).await {
                    Ok(c) => c,
                    Err(_) => {
                        files_with_errors.push(display_path(&self.workspace_root, file_path));
                        continue;
                    }
                };

                // Skip binary-looking files: if the first 8KB contain a NUL
                // byte, the file is likely binary and regex matches would be
                // nonsensical.
                let probe = &content[..content.len().min(8192)];
                if probe.contains('\0') {
                    continue;
                }

                let dp = display_path(&self.workspace_root, file_path);
                for (line_idx, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        matches.push((dp.clone(), line_idx + 1, line.to_string()));
                        // Collect one past the cap so we can distinguish
                        // "exactly MAX_MATCHES" from "more than MAX_MATCHES".
                        if matches.len() > MAX_MATCHES {
                            hit_cap = true;
                            break;
                        }
                    }
                }
                if hit_cap {
                    break;
                }
            }

            if matches.is_empty() && files_with_errors.is_empty() {
                return Ok("No matches found.".into());
            }

            // If we collected an overflow sentinel, remove it before display.
            if hit_cap {
                matches.truncate(MAX_MATCHES);
            }
            let total = matches.len();

            let mut out = String::new();
            if hit_cap {
                out.push_str(&format!(
                    "Found {MAX_MATCHES}+ matches (showing first {MAX_MATCHES}):\n"
                ));
            } else {
                out.push_str(&format!(
                    "Found {total} match{}:\n",
                    if total == 1 { "" } else { "es" }
                ));
            }

            for (path, line_num, content) in &matches {
                out.push_str(&format!("{path}:{line_num}: {content}\n"));
            }

            // Note any files that failed to read.
            if !files_with_errors.is_empty() {
                out.push_str(&format!(
                    "\n[skipped {} file{} due to read errors: {}]",
                    files_with_errors.len(),
                    if files_with_errors.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    files_with_errors.join(", "),
                ));
            }

            // Trim trailing whitespace.
            Ok(out.trim_end().to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;

    fn tool(fs: Arc<FakeFileSystem>, root: &str) -> GrepTool<FakeFileSystem> {
        GrepTool::new(fs, PathBuf::from(root))
    }

    #[tokio::test]
    async fn finds_matches_across_files() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.rs", "fn main() {}\nfn helper() {}\n");
        fs.insert("/ws/b.rs", "fn test() {}\nlet x = 1;\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"fn \\w+"}"#).await.unwrap();
        assert!(out.contains("Found 3 matches:"));
        assert!(out.contains("a.rs:1:"));
        assert!(out.contains("a.rs:2:"));
        assert!(out.contains("b.rs:1:"));
    }

    #[tokio::test]
    async fn respects_glob_filter() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/src/main.rs", "fn main() {}\n");
        fs.insert("/ws/src/lib.rs", "fn lib() {}\n");
        fs.insert("/ws/README.md", "fn not_code() {}\n");
        let t = tool(fs, "/ws");

        let out = t
            .execute(r#"{"pattern":"fn","glob":"**/*.rs"}"#)
            .await
            .unwrap();
        assert!(out.contains("Found 2 matches:"));
        assert!(!out.contains("README.md"));
    }

    #[tokio::test]
    async fn respects_path_parameter() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/src/main.rs", "fn main() {}\n");
        fs.insert("/ws/tests/test.rs", "fn test() {}\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"fn","path":"src"}"#).await.unwrap();
        assert!(out.contains("Found 1 match:"));
        assert!(out.contains("src/main.rs"));
        assert!(!out.contains("tests/"));
    }

    #[tokio::test]
    async fn defaults_to_workspace_root_and_all_files() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "hello\n");
        fs.insert("/ws/b.txt", "hello world\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"hello"}"#).await.unwrap();
        assert!(out.contains("Found 2 matches:"));
    }

    #[tokio::test]
    async fn no_matches_returns_message() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "hello\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"goodbye"}"#).await.unwrap();
        assert_eq!(out, "No matches found.");
    }

    #[tokio::test]
    async fn invalid_regex_returns_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");

        let result = t.execute(r#"{"pattern":"[invalid"}"#).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn truncates_at_cap() {
        let fs = Arc::new(FakeFileSystem::new());
        // Build a file with more lines than MAX_MATCHES, all matching.
        let content: String = (0..250).map(|i| format!("match_{i}\n")).collect();
        fs.insert("/ws/big.txt", &content);
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"match_"}"#).await.unwrap();
        assert!(out.contains("200+ matches"));
        assert!(out.contains("showing first 200"));
        // Count data lines (skip the header).
        let data_lines = out.lines().skip(1).count();
        assert_eq!(data_lines, 200);
    }

    #[tokio::test]
    async fn skips_unreadable_files_with_note() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/ok.txt", "found it\n");
        // Ghost path: walk_glob discovers it, but read() will fail.
        fs.insert_ghost("/ws/gone.txt");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"found"}"#).await.unwrap();
        assert!(out.contains("Found 1 match:"));
        assert!(out.contains("ok.txt:1:"));
        assert!(out.contains("[skipped 1 file due to read errors: gone.txt]"));
    }

    #[tokio::test]
    async fn skips_binary_files() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/text.txt", "searchme\n");
        fs.insert("/ws/binary.bin", "searchme\0\x01\x02\x03");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"searchme"}"#).await.unwrap();
        assert!(out.contains("Found 1 match:"));
        assert!(out.contains("text.txt"));
        assert!(!out.contains("binary.bin"));
    }

    #[tokio::test]
    async fn output_includes_line_numbers() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "aaa\nbbb\nccc\nbbb\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"bbb"}"#).await.unwrap();
        assert!(out.contains("f.txt:2:"));
        assert!(out.contains("f.txt:4:"));
    }

    #[tokio::test]
    async fn empty_workspace_returns_no_matches() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"anything"}"#).await.unwrap();
        assert_eq!(out, "No matches found.");
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

    #[tokio::test]
    async fn exactly_at_cap_is_not_truncated() {
        let fs = Arc::new(FakeFileSystem::new());
        // Exactly MAX_MATCHES lines, all matching — no truncation.
        let content: String = (0..200).map(|i| format!("line_{i}\n")).collect();
        fs.insert("/ws/exact.txt", &content);
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"line_"}"#).await.unwrap();
        assert!(out.contains("Found 200 matches:"), "got: {out}");
        assert!(!out.contains("200+"), "got: {out}");
    }

    #[tokio::test]
    async fn absolute_path_outside_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/other/f.rs", "fn hello() {}\n");
        fs.insert("/ws/g.rs", "fn world() {}\n");
        let t = tool(fs, "/ws");

        let out = t
            .execute(r#"{"pattern":"fn","path":"/other"}"#)
            .await
            .unwrap();
        assert!(out.contains("Found 1 match:"));
        // Path outside workspace shows as absolute.
        assert!(out.contains("/other/f.rs:1:"));
        assert!(!out.contains("g.rs"));
    }
}
