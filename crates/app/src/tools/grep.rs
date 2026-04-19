//! `grep` tool.
//!
//! Searches file contents by regex across the workspace. Walks files via
//! `walk_glob`, reads each one, and collects matching lines with file path
//! and line number. Large result sets are spilled to a temp file under
//! `.ox/tmp/` with a preview shown inline.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

use domain::ToolDef;

use super::spill::{self, PREVIEW_LINES};
use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::approval::{ApprovalRequirement, MissingPathPolicy, path_approval_requirement};
use crate::ports::FileSystem;

/// Byte cap for walk_glob — generous because the file list is rarely the
/// memory problem; the match accumulation is.
const WALK_MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MB

/// Byte budget for accumulated match output. Stops collecting when this
/// limit is reached to prevent unbounded memory consumption.
const MATCH_MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MB

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
            description: "Search file contents by regex. Use this instead of grep or rg \
                via bash. Returns matching lines with file paths and line numbers. \
                Supports an optional `glob` filter to restrict which files are searched."
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

    fn approval_requirement<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<ApprovalRequirement>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: GrepArgs =
                serde_json::from_str(args).context("grep: invalid JSON arguments")?;
            require_non_empty("pattern", &parsed.pattern)?;
            let path = parsed.path.as_deref().unwrap_or(".");
            path_approval_requirement(
                self.fs.as_ref(),
                &self.workspace_root,
                path,
                MissingPathPolicy::MustExist,
            )
            .await
        })
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

            let walk = self
                .fs
                .walk_glob(&root, glob_pattern, WALK_MAX_BYTES)
                .await
                .with_context(|| {
                    format!("grep: walk_glob failed for pattern {:?}", glob_pattern)
                })?;
            let files = walk.paths;

            // Accumulate formatted match lines, tracking byte size.
            let mut match_output = String::new();
            let mut match_count: usize = 0;
            let mut files_with_errors: Vec<String> = Vec::new();
            let mut hit_byte_cap = false;

            'outer: for file_path in &files {
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
                        let formatted = format!("{dp}:{}: {line}\n", line_idx + 1);
                        if match_output.len() + formatted.len() > MATCH_MAX_BYTES {
                            hit_byte_cap = true;
                            break 'outer;
                        }
                        match_output.push_str(&formatted);
                        match_count += 1;
                    }
                }
            }

            if match_count == 0 && files_with_errors.is_empty() {
                return Ok("No matches found.".into());
            }

            // Build the full result text, then decide whether to spill.
            let mut out = String::new();
            out.push_str(&format!(
                "Found {match_count} match{}:\n",
                if match_count == 1 { "" } else { "es" }
            ));
            out.push_str(&match_output);

            if hit_byte_cap {
                out.push_str("[match collection stopped — byte cap reached]\n");
            }

            // Note any files that failed to read.
            if !files_with_errors.is_empty() {
                out.push_str(&format!(
                    "\n[skipped {} file{} due to read errors: {}]\n",
                    files_with_errors.len(),
                    if files_with_errors.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    files_with_errors.join(", "),
                ));
            }

            let trimmed = out.trim_end().to_string();

            if spill::needs_spill(&trimmed) {
                let info =
                    spill::spill(self.fs.as_ref(), &self.workspace_root, &trimmed, "grep").await?;
                let preview = spill::preview(&trimmed, PREVIEW_LINES);
                Ok(format!(
                    "Found {match_count} matches (showing first {PREVIEW_LINES} lines, full output: {}):\n{}",
                    info.display_path,
                    preview.trim_end(),
                ))
            } else {
                Ok(trimmed)
            }
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
    async fn large_result_set_spills() {
        let fs = Arc::new(FakeFileSystem::new());
        // 500 matching lines — exceeds INLINE_MAX_LINES (200).
        let content: String = (0..500).map(|i| format!("match_{i}\n")).collect();
        fs.insert("/ws/big.txt", &content);
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"match_"}"#).await.unwrap();
        assert!(out.contains("500 matches"), "got: {out}");
        assert!(
            out.contains("[full output:") || out.contains("full output:"),
            "got: {out}"
        );
        assert!(out.contains(".ox/tmp/grep-"), "got: {out}");
        // Preview should show first PREVIEW_LINES lines, not all 500+.
        assert!(out.contains("match_0"), "got: {out}");
        assert!(!out.contains("match_499"), "got: {out}");
    }

    #[tokio::test]
    async fn at_inline_threshold_does_not_spill() {
        let fs = Arc::new(FakeFileSystem::new());
        // 199 matches + 1 header line = 200 formatted lines, exactly at threshold.
        let content: String = (0..199).map(|i| format!("line_{i}\n")).collect();
        fs.insert("/ws/exact.txt", &content);
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"pattern":"line_"}"#).await.unwrap();
        assert!(out.contains("Found 199 matches:"), "got: {out}");
        assert!(!out.contains(".ox/tmp/"), "got: {out}");
    }

    #[tokio::test]
    async fn skips_unreadable_files_with_note() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/ok.txt", "found it\n");
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
        assert!(out.contains("/other/f.rs:1:"));
        assert!(!out.contains("g.rs"));
    }

    #[tokio::test]
    async fn approval_not_required_for_path_under_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "hello\n");
        let t = tool(fs, "/ws");
        let requirement = t
            .approval_requirement(r#"{"pattern":"hello"}"#)
            .await
            .unwrap();
        assert_eq!(
            requirement,
            crate::approval::ApprovalRequirement::NotRequired
        );
    }

    #[tokio::test]
    async fn approval_required_for_path_outside_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/other/a.txt", "hello\n");
        let t = tool(fs, "/ws");
        let requirement = t
            .approval_requirement(r#"{"pattern":"hello","path":"../other"}"#)
            .await
            .unwrap();
        assert!(matches!(
            requirement,
            crate::approval::ApprovalRequirement::Required { .. }
        ));
    }
}
