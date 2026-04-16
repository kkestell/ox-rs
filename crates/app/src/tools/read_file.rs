//! `read_file` tool.
//!
//! Reads a file from the workspace (or an absolute path) and returns its
//! contents prefixed with `{N}:{hash}|` hashlines so the LLM can turn around
//! and call `edit_file` without a second read.
//!
//! `offset`/`limit` match the semantics of the C# reference implementation:
//! `offset` is 0-indexed from the top of the file, `limit` defaults to 2000
//! lines. When the returned window isn't the whole file, a truncation notice
//! is appended so the LLM knows more content exists.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::hashlines::{render_with_hashlines, split_lines};
use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::ports::FileSystem;
use crate::stream::ToolDef;

/// Matches the C# reference default; long enough for almost every source
/// file while keeping the context-window cost bounded for huge files.
const DEFAULT_LIMIT: usize = 2000;

pub struct ReadFileTool<F> {
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<F> ReadFileTool<F> {
    pub fn new(fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self { fs, workspace_root }
    }
}

#[derive(Debug, Deserialize)]
struct ReadArgs {
    file_path: String,
    /// 0-indexed offset into the file's line list. Optional; defaults to 0.
    #[serde(default)]
    offset: Option<usize>,
    /// Maximum number of lines to return. Optional; defaults to `DEFAULT_LIMIT`.
    #[serde(default)]
    limit: Option<usize>,
}

impl<F: FileSystem + Send + Sync + 'static> Tool for ReadFileTool<F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "read_file".into(),
            description: "Read a file from the workspace. Returns its contents with each line \
                prefixed by a `{line_number}:{hash}|` hashline tag that you can copy \
                verbatim into an `edit_file` anchor. Supports optional `offset` \
                (0-indexed) and `limit` (default 2000) for large files."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to read, absolute or workspace-relative."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "0-indexed line offset."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to return."
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: ReadArgs =
                serde_json::from_str(args).context("read_file: invalid JSON arguments")?;
            require_non_empty("file_path", &parsed.file_path)?;

            let path = resolve_path(&self.workspace_root, &parsed.file_path);
            let content = self
                .fs
                .read(&path)
                .await
                .with_context(|| format!("read_file: could not read {}", path.display()))?;

            // Empty files round-trip as empty — no hashlines, no truncation
            // notice. Keeps the output simple and the `edit_file` contract
            // trivial: "file has zero lines" means "nothing to anchor".
            if content.is_empty() {
                return Ok(String::new());
            }

            let total_lines = split_lines(&content).len();
            let offset = parsed.offset.unwrap_or(0);
            // Schema declares `minimum: 1`, but serde doesn't enforce JSON
            // Schema constraints — clamp here so a stray `limit: 0` from the
            // model can't produce a nonsense "lines 1-0 of N" notice.
            let limit = parsed.limit.unwrap_or(DEFAULT_LIMIT).max(1);

            // When the offset is past EOF we return empty with a notice
            // rather than failing — lets the LLM probe a file's length
            // without a dedicated "line count" tool.
            if offset >= total_lines {
                return Ok(format!(
                    "[empty: offset {offset} is past the end of the file ({total_lines} lines)]"
                ));
            }

            let end = (offset + limit).min(total_lines);
            let slice = slice_lines(&content, offset, end);

            let rendered = render_with_hashlines(&slice, offset + 1);

            // Only attach a notice when the window isn't the whole file; in
            // the common case of a small file we return clean hashlined text.
            if offset == 0 && end == total_lines {
                Ok(rendered)
            } else {
                // `rendered` may end with a newline (slice preserved the
                // file's trailing-newline shape). Strip one if present so
                // the notice appears immediately below the last line rather
                // than separated by a blank line.
                let rendered = rendered.strip_suffix('\n').unwrap_or(&rendered);
                Ok(format!(
                    "{rendered}\n[truncated: showing lines {}-{} of {total_lines} in {}]",
                    offset + 1,
                    end,
                    display_path(&self.workspace_root, &path),
                ))
            }
        })
    }
}

/// Return the substring of `text` containing lines `[start, end)`
/// (0-indexed). Preserves the trailing-newline shape of the slice the same
/// way `render_with_hashlines` does: if the original file had a trailing
/// newline AND we're including its last line, the slice keeps that newline.
fn slice_lines(text: &str, start: usize, end: usize) -> String {
    let lines = split_lines(text);
    let end = end.min(lines.len());
    if start >= end {
        return String::new();
    }
    let had_trailing_newline = text.ends_with('\n');
    let selected = &lines[start..end];
    let mut out = String::new();
    for (i, line) in selected.iter().enumerate() {
        out.push_str(line);
        let is_last_of_slice = i + 1 == selected.len();
        let is_last_of_file = start + i + 1 == lines.len();
        // Attach a newline except on the final line of the slice when that
        // line is also the final line of a file without a trailing newline.
        if !is_last_of_slice || !is_last_of_file || had_trailing_newline {
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;
    use crate::tools::hashlines::hash_line;

    fn tool(fs: Arc<FakeFileSystem>, root: &str) -> ReadFileTool<FakeFileSystem> {
        ReadFileTool::new(fs, PathBuf::from(root))
    }

    #[tokio::test]
    async fn returns_hashlined_output_for_small_file() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/a.txt", "alpha\nbeta\n");
        let t = tool(fs, "/ws");

        let out = t.execute(r#"{"file_path":"a.txt"}"#).await.unwrap();
        // Two lines, each with a hashline prefix.
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with(&format!("1:{}| alpha", hash_line("alpha"))));
        assert!(lines[1].starts_with(&format!("2:{}| beta", hash_line("beta"))));
        // No truncation notice — whole file fit.
        assert!(!out.contains("[truncated"));
    }

    #[tokio::test]
    async fn absolute_path_bypasses_workspace_root() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/other/place.txt", "hi\n");
        let t = tool(fs, "/ws");
        let out = t
            .execute(r#"{"file_path":"/other/place.txt"}"#)
            .await
            .unwrap();
        assert!(out.contains("hi"));
    }

    #[tokio::test]
    async fn empty_file_returns_empty_string() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/empty.txt", "");
        let t = tool(fs, "/ws");
        let out = t.execute(r#"{"file_path":"empty.txt"}"#).await.unwrap();
        assert_eq!(out, "");
    }

    #[tokio::test]
    async fn missing_file_returns_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        let result = t.execute(r#"{"file_path":"nope.txt"}"#).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn offset_and_limit_window_the_output() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/big.txt", "a\nb\nc\nd\ne\n");
        let t = tool(fs, "/ws");

        // Skip the first two lines, take two lines.
        let out = t
            .execute(r#"{"file_path":"big.txt","offset":2,"limit":2}"#)
            .await
            .unwrap();

        let lines: Vec<_> = out.lines().collect();
        // 2 content lines + 1 truncation notice line.
        assert_eq!(lines.len(), 3, "got {lines:?}");
        // Line numbers in hashlines are 1-indexed ABSOLUTE — line 3, line 4.
        assert!(lines[0].starts_with("3:"), "got {:?}", lines[0]);
        assert!(lines[1].starts_with("4:"), "got {:?}", lines[1]);
        // Truncation notice mentions the window and the total.
        assert!(lines[2].contains("[truncated"), "got {:?}", lines[2]);
        assert!(lines[2].contains("3-4"), "got {:?}", lines[2]);
        assert!(lines[2].contains("of 5"), "got {:?}", lines[2]);
    }

    #[tokio::test]
    async fn offset_past_end_returns_empty_notice() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/short.txt", "only\n");
        let t = tool(fs, "/ws");

        let out = t
            .execute(r#"{"file_path":"short.txt","offset":10}"#)
            .await
            .unwrap();
        assert!(out.contains("past the end"), "got {out}");
    }

    #[tokio::test]
    async fn reading_whole_file_does_not_add_truncation_notice() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "alpha\nbeta\n");
        let t = tool(fs, "/ws");
        let out = t
            .execute(r#"{"file_path":"f.txt","offset":0,"limit":2}"#)
            .await
            .unwrap();
        assert!(!out.contains("[truncated"), "got {out}");
    }

    #[tokio::test]
    async fn file_without_trailing_newline_still_round_trips() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/no_nl.txt", "alpha\nbeta");
        let t = tool(fs, "/ws");
        let out = t.execute(r#"{"file_path":"no_nl.txt"}"#).await.unwrap();
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("| beta"));
        // Output should not carry an artificial trailing newline, matching
        // the source file's shape.
        assert!(!out.ends_with('\n'));
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute("not json").await.is_err());
    }

    #[tokio::test]
    async fn empty_file_path_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute(r#"{"file_path":""}"#).await.is_err());
    }
}
