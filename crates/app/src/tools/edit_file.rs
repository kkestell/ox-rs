//! `edit_file` tool.
//!
//! Applies a list of `replace` / `insert_after` operations to a file, using
//! hashline anchors to verify the LLM's view of the file is current before
//! mutating it.
//!
//! ## Operations
//!
//! - `replace { start, end, content }` — replace the inclusive range of lines
//!   from `start` to `end` with `content`. `start`/`end` are hashline anchors
//!   (`"{N}:{hash}"`). `content == ""` means "delete these lines." A single-line
//!   replace has `start == end`.
//! - `insert_after { anchor, content }` — insert `content` immediately after
//!   the line named by `anchor`. The anchor may be a hashline or one of the
//!   sentinels `"start"` / `"end"` to prepend or append.
//!
//! ## Hash mismatch is a hard fail
//!
//! Any anchor whose hash doesn't match the current file content rejects the
//! *entire* edit (we never partially apply). The error message names the
//! bad anchor and shows the current hash, so the LLM can correct itself on
//! the next turn without a second `read_file` call.
//!
//! ## Bottom-up application
//!
//! Edits are sorted by their start line and applied bottom-up so earlier line
//! numbers aren't shifted by later edits. Overlapping edits are rejected.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::hashlines::{Anchor, parse_anchor, split_lines, verify_anchor};
use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::approval::{ApprovalRequirement, MissingPathPolicy, path_approval_requirement};
use crate::ports::FileSystem;
use crate::stream::ToolDef;

pub struct EditFileTool<F> {
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<F> EditFileTool<F> {
    pub fn new(fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self { fs, workspace_root }
    }
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    file_path: String,
    edits: Vec<EditOp>,
}

/// The on-wire edit representation. `#[serde(tag = "op")]` matches the
/// schema we publish to the LLM, where the discriminator field is literally
/// called `op`.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum EditOp {
    Replace {
        start: String,
        end: String,
        content: String,
    },
    InsertAfter {
        anchor: String,
        content: String,
    },
}

/// A resolved edit ready to apply: half-open `[start, end)` indices into the
/// current `lines` vector, plus the replacement text already split into lines
/// with no trailing empty line.
///
/// `insert_after` becomes a zero-width "replace" at the insertion point.
#[derive(Debug)]
struct ResolvedEdit {
    /// Inclusive start index (0-based).
    start: usize,
    /// Exclusive end index (0-based).
    end: usize,
    /// Replacement lines, already split.
    replacement: Vec<String>,
    /// For diagnostic messages on overlap.
    description: String,
}

impl<F: FileSystem + Send + Sync + 'static> Tool for EditFileTool<F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "edit_file".into(),
            description: "Apply line-range edits to a file using hashline anchors from a prior \
                `read_file`. Each edit is either a `replace` (swap a line range for new \
                content, or delete by passing empty content) or an `insert_after` (add \
                content right after a line, or use `start`/`end` as sentinels). Anchors \
                must match the file's current hashlines exactly — any mismatch aborts \
                the whole edit so stale reads can't silently corrupt the file."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File to edit, absolute or workspace-relative."
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of edits to apply in a single atomic call.",
                        "items": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "op": { "const": "replace" },
                                        "start": {
                                            "type": "string",
                                            "description": "Hashline anchor for the first line of the range."
                                        },
                                        "end": {
                                            "type": "string",
                                            "description": "Hashline anchor for the last line of the range (inclusive). Equal to `start` for a single-line replace."
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "Replacement text. Empty string deletes the range."
                                        }
                                    },
                                    "required": ["op", "start", "end", "content"]
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "op": { "const": "insert_after" },
                                        "anchor": {
                                            "type": "string",
                                            "description": "Hashline anchor, or the sentinel `start`/`end`."
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "Text to insert. A trailing newline is added if missing."
                                        }
                                    },
                                    "required": ["op", "anchor", "content"]
                                }
                            ]
                        }
                    }
                },
                "required": ["file_path", "edits"]
            }),
        }
    }

    fn approval_requirement<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<ApprovalRequirement>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: EditArgs =
                serde_json::from_str(args).context("edit_file: invalid JSON arguments")?;
            require_non_empty("file_path", &parsed.file_path)?;
            path_approval_requirement(
                self.fs.as_ref(),
                &self.workspace_root,
                &parsed.file_path,
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
            let parsed: EditArgs =
                serde_json::from_str(args).context("edit_file: invalid JSON arguments")?;
            require_non_empty("file_path", &parsed.file_path)?;
            if parsed.edits.is_empty() {
                bail!("edit_file: edits must not be empty");
            }

            let path = resolve_path(&self.workspace_root, &parsed.file_path);
            let original = self
                .fs
                .read(&path)
                .await
                .with_context(|| format!("edit_file: could not read {}", path.display()))?;

            let had_trailing_newline = original.ends_with('\n');
            let lines = split_lines(&original);

            // Resolve every edit against the *current* file state. Errors
            // here are hash mismatches, out-of-range anchors, and malformed
            // anchor strings — all recoverable by the LLM on the next turn.
            let mut resolved: Vec<ResolvedEdit> = Vec::with_capacity(parsed.edits.len());
            for (i, edit) in parsed.edits.iter().enumerate() {
                resolved.push(resolve_edit(&lines, edit, i)?);
            }

            // Sort by start index so we can detect overlaps cheaply and apply
            // bottom-up without re-indexing.
            resolved.sort_by_key(|e| e.start);
            detect_overlaps(&resolved)?;

            // Apply bottom-up: every edit modifies a half-open range that
            // doesn't shift ranges below it in the vector.
            let mut new_lines: Vec<String> = lines.iter().map(|s| (*s).to_owned()).collect();
            for edit in resolved.iter().rev() {
                new_lines.splice(edit.start..edit.end, edit.replacement.iter().cloned());
            }

            // Re-join. We preserve the original file's trailing-newline
            // behavior: if it had one, put one back at the end; otherwise
            // don't introduce one.
            let mut output = new_lines.join("\n");
            if had_trailing_newline && !output.is_empty() {
                output.push('\n');
            }

            self.fs
                .write(&path, &output)
                .await
                .with_context(|| format!("edit_file: could not write {}", path.display()))?;

            Ok(format!(
                "Applied {} edit{} to {}",
                parsed.edits.len(),
                if parsed.edits.len() == 1 { "" } else { "s" },
                display_path(&self.workspace_root, &path)
            ))
        })
    }
}

/// Resolve one on-wire `EditOp` into a `ResolvedEdit` indexed against
/// `lines`. Anchors are verified here; hash mismatches become errors.
fn resolve_edit(lines: &[&str], edit: &EditOp, index: usize) -> Result<ResolvedEdit> {
    match edit {
        EditOp::Replace {
            start,
            end,
            content,
        } => {
            let start_anchor = parse_anchor(start)
                .with_context(|| format!("edit {index}: invalid start anchor"))?;
            let end_anchor =
                parse_anchor(end).with_context(|| format!("edit {index}: invalid end anchor"))?;
            // Replace refuses the `start`/`end` sentinels — those make sense
            // only for `insert_after`. A replace has to name concrete lines.
            if matches!(start_anchor, Anchor::Start | Anchor::End)
                || matches!(end_anchor, Anchor::Start | Anchor::End)
            {
                bail!(
                    "edit {index}: replace anchors must be line references, not `start`/`end` sentinels"
                );
            }
            let start_idx = verify_anchor(lines, &start_anchor)
                .with_context(|| format!("edit {index}: replace start anchor"))?;
            let end_idx = verify_anchor(lines, &end_anchor)
                .with_context(|| format!("edit {index}: replace end anchor"))?;
            if end_idx < start_idx {
                bail!("edit {index}: replace end is before start ({end_idx} < {start_idx})");
            }
            let replacement = split_content_lines(content);
            Ok(ResolvedEdit {
                start: start_idx,
                end: end_idx + 1,
                replacement,
                description: format!("replace lines {}..={}", start_idx + 1, end_idx + 1),
            })
        }
        EditOp::InsertAfter { anchor, content } => {
            let parsed_anchor = parse_anchor(anchor)
                .with_context(|| format!("edit {index}: invalid insert anchor"))?;
            let insert_idx = match parsed_anchor {
                // `Start` means "insert before the first line" — index 0.
                Anchor::Start => 0,
                // `End` means "append" — the line count.
                Anchor::End => lines.len(),
                Anchor::Line { .. } => {
                    verify_anchor(lines, &parsed_anchor)
                        .with_context(|| format!("edit {index}: insert anchor"))?
                        + 1
                }
            };
            let replacement = split_content_lines(content);
            Ok(ResolvedEdit {
                start: insert_idx,
                end: insert_idx,
                replacement,
                description: format!("insert at line {}", insert_idx + 1),
            })
        }
    }
}

/// Split an edit's replacement content into the per-line representation used
/// by the splice. An empty string yields an empty vec — the splice becomes
/// a pure deletion. A trailing newline in `content` is dropped so it doesn't
/// round-trip into an empty final line of the spliced range.
fn split_content_lines(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let trimmed = content.strip_suffix('\n').unwrap_or(content);
    trimmed.split('\n').map(|s| s.to_owned()).collect()
}

/// Reject edits whose half-open ranges intersect. `resolved` must be sorted
/// by `start` ascending. Two edits `(a.start, a.end)` and `(b.start, b.end)`
/// overlap when `a.end > b.start`. A zero-width `insert_after` at position
/// `p` lies strictly between range ends — an overlap requires one range to
/// *cover* the insertion point, not just abut it, so we use strict `>`.
fn detect_overlaps(resolved: &[ResolvedEdit]) -> Result<()> {
    for pair in resolved.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if a.end > b.start {
            bail!(
                "overlapping edits: '{}' and '{}'",
                a.description,
                b.description
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;
    use crate::tools::hashlines::hash_line;

    fn tool(fs: Arc<FakeFileSystem>, root: &str) -> EditFileTool<FakeFileSystem> {
        EditFileTool::new(fs, PathBuf::from(root))
    }

    fn anchor(line: usize, content: &str) -> String {
        format!("{line}:{}", hash_line(content))
    }

    #[tokio::test]
    async fn single_line_replace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "alpha\nbeta\ngamma\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(2, "beta"),
                    "end": anchor(2, "beta"),
                    "content": "BETA"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();

        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("alpha\nBETA\ngamma\n")
        );
    }

    #[tokio::test]
    async fn multi_line_replace_across_range() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\nd\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(2, "b"),
                    "end": anchor(3, "c"),
                    "content": "X\nY\nZ"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("a\nX\nY\nZ\nd\n")
        );
    }

    #[tokio::test]
    async fn replace_with_empty_content_deletes_lines() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(2, "b"),
                    "end": anchor(2, "b"),
                    "content": ""
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(fs.get(Path::new("/ws/f.txt")).as_deref(), Some("a\nc\n"));
    }

    #[tokio::test]
    async fn insert_after_numeric_anchor() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "one\ntwo\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "insert_after",
                    "anchor": anchor(1, "one"),
                    "content": "inserted"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("one\ninserted\ntwo\n")
        );
    }

    #[tokio::test]
    async fn insert_after_start_prepends() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "insert_after",
                    "anchor": "start",
                    "content": "HEADER"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("HEADER\na\nb\n")
        );
    }

    #[tokio::test]
    async fn insert_after_end_appends() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "insert_after",
                    "anchor": "end",
                    "content": "FOOTER"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("a\nb\nFOOTER\n")
        );
    }

    #[tokio::test]
    async fn insert_content_without_trailing_newline_still_yields_clean_newlines() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "one\ntwo\n");
        let t = tool(fs.clone(), "/ws");

        // No trailing newline in content; the join logic supplies newlines.
        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "insert_after",
                    "anchor": anchor(1, "one"),
                    "content": "A"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        let out = fs.get(Path::new("/ws/f.txt")).unwrap();
        assert_eq!(out, "one\nA\ntwo\n");
    }

    #[tokio::test]
    async fn hash_mismatch_produces_descriptive_error_and_leaves_file_unchanged() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "alpha\nbeta\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": "1:xxx",
                    "end": "1:xxx",
                    "content": "broken"
                }
            ]
        })
        .to_string();

        // `{:#}` prints the full error chain; the top layer is just the
        // outer context ("edit 0: replace start anchor"), but the chain
        // contains the underlying hash-mismatch diagnostic the LLM needs.
        let err = format!("{:#}", t.execute(&args).await.unwrap_err());
        assert!(err.contains("line 1"), "got {err}");
        assert!(err.contains("xxx"), "got {err}");
        assert!(err.contains(&hash_line("alpha")), "got {err}");

        // File was NOT modified.
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("alpha\nbeta\n")
        );
    }

    #[tokio::test]
    async fn overlapping_edits_are_rejected() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\nd\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(1, "a"),
                    "end": anchor(3, "c"),
                    "content": "X"
                },
                {
                    "op": "replace",
                    "start": anchor(2, "b"),
                    "end": anchor(2, "b"),
                    "content": "Y"
                }
            ]
        })
        .to_string();

        let err = t.execute(&args).await.unwrap_err().to_string();
        assert!(err.to_ascii_lowercase().contains("overlap"), "got {err}");
        // File stays put.
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("a\nb\nc\nd\n")
        );
    }

    #[tokio::test]
    async fn multiple_non_overlapping_edits_apply_correctly() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\nd\ne\n");
        let t = tool(fs.clone(), "/ws");

        // Submit out-of-order to confirm sorting works.
        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(5, "e"),
                    "end": anchor(5, "e"),
                    "content": "E"
                },
                {
                    "op": "replace",
                    "start": anchor(1, "a"),
                    "end": anchor(1, "a"),
                    "content": "A"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("A\nb\nc\nd\nE\n")
        );
    }

    #[tokio::test]
    async fn anchor_past_end_of_file_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "only\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": "10:abc",
                    "end": "10:abc",
                    "content": "nope"
                }
            ]
        })
        .to_string();

        assert!(t.execute(&args).await.is_err());
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute("not json").await.is_err());
    }

    #[tokio::test]
    async fn duplicate_lines_disambiguated_by_line_number() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "same\nother\nsame\n");
        let t = tool(fs.clone(), "/ws");

        // Replace only the THIRD line (line 3 "same"), not the first.
        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(3, "same"),
                    "end": anchor(3, "same"),
                    "content": "UPDATED"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("same\nother\nUPDATED\n")
        );
    }

    #[tokio::test]
    async fn empty_edits_array_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "x\n");
        let t = tool(fs, "/ws");
        let args = r#"{"file_path":"f.txt","edits":[]}"#;
        assert!(t.execute(args).await.is_err());
    }

    #[tokio::test]
    async fn replace_with_sentinel_anchor_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "x\n");
        let t = tool(fs, "/ws");

        let args = r#"{"file_path":"f.txt","edits":[{"op":"replace","start":"start","end":"end","content":"Y"}]}"#;
        assert!(t.execute(args).await.is_err());
    }

    #[tokio::test]
    async fn replace_with_reversed_range_is_an_error() {
        // end anchor must not point to a line earlier than start. The
        // resolver rejects this before any write happens.
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(3, "c"),
                    "end": anchor(1, "a"),
                    "content": "X"
                }
            ]
        })
        .to_string();

        assert!(t.execute(&args).await.is_err());
        // File untouched.
        assert_eq!(fs.get(Path::new("/ws/f.txt")).as_deref(), Some("a\nb\nc\n"));
    }

    #[tokio::test]
    async fn replace_content_with_trailing_newline_does_not_add_blank_line() {
        // Guards `split_content_lines`'s trailing-newline strip. Without it,
        // content "NEW\n" would splice in ["NEW", ""] and produce an extra
        // blank line — an easy regression to introduce.
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "a\nb\nc\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(2, "b"),
                    "end": anchor(2, "b"),
                    "content": "NEW\n"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("a\nNEW\nc\n")
        );
    }

    #[tokio::test]
    async fn two_inserts_after_same_anchor_stack_in_call_order() {
        // Both edits resolve to the same insertion point (zero-width range),
        // so `detect_overlaps` accepts them. Sort-by-start is stable, and
        // bottom-up application applies the later edit first, so the two
        // inserts end up appearing in the order they were submitted.
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "one\ntwo\n");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "insert_after",
                    "anchor": anchor(1, "one"),
                    "content": "FIRST"
                },
                {
                    "op": "insert_after",
                    "anchor": anchor(1, "one"),
                    "content": "SECOND"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        assert_eq!(
            fs.get(Path::new("/ws/f.txt")).as_deref(),
            Some("one\nFIRST\nSECOND\ntwo\n")
        );
    }

    #[tokio::test]
    async fn file_without_trailing_newline_preserves_shape() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "alpha\nbeta");
        let t = tool(fs.clone(), "/ws");

        let args = serde_json::json!({
            "file_path": "f.txt",
            "edits": [
                {
                    "op": "replace",
                    "start": anchor(1, "alpha"),
                    "end": anchor(1, "alpha"),
                    "content": "ALPHA"
                }
            ]
        })
        .to_string();

        t.execute(&args).await.unwrap();
        let out = fs.get(Path::new("/ws/f.txt")).unwrap();
        assert_eq!(out, "ALPHA\nbeta"); // No trailing newline introduced.
    }

    #[tokio::test]
    async fn approval_not_required_for_file_under_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "content\n");
        let t = tool(fs, "/ws");
        let requirement = t
            .approval_requirement(r#"{"file_path":"f.txt","edits":[{"op":"replace","start":"1:abc","end":"1:abc","content":"X"}]}"#)
            .await
            .unwrap();
        assert_eq!(
            requirement,
            crate::approval::ApprovalRequirement::NotRequired
        );
    }

    #[tokio::test]
    async fn approval_required_for_file_outside_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/other/f.txt", "content\n");
        let t = tool(fs, "/ws");
        let requirement = t
            .approval_requirement(r#"{"file_path":"../other/f.txt","edits":[{"op":"replace","start":"1:abc","end":"1:abc","content":"X"}]}"#)
            .await
            .unwrap();
        assert!(matches!(
            requirement,
            crate::approval::ApprovalRequirement::Required { .. }
        ));
    }
}
