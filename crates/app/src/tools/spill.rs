//! Shared spill-to-file utility for tools that produce unbounded output.
//!
//! When a tool's formatted output exceeds the inline threshold (200 lines or
//! 50 KB), the full content is written to a temp file under
//! `$WORKSPACE/.ox/tmp/` and the inline result shows a preview (first 50
//! lines) plus the file path. The agent can then use `read_file`, `grep`, etc.
//! to explore the full result without cramming it into the context window.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use super::display_path;
use crate::ports::FileSystem;

/// Maximum lines before spilling to file.
pub(crate) const INLINE_MAX_LINES: usize = 200;

/// Maximum bytes before spilling to file.
pub(crate) const INLINE_MAX_BYTES: usize = 50 * 1024; // 50 KB

/// Number of preview lines shown inline when content spills.
pub(crate) const PREVIEW_LINES: usize = 50;

/// Returns `true` if the content exceeds either inline threshold.
pub(crate) fn needs_spill(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.len() > INLINE_MAX_BYTES || s.lines().count() > INLINE_MAX_LINES
}

/// Byte-offset slice of the first `max_lines` lines. No allocation — returns
/// a `&str` into the original content.
pub(crate) fn preview(s: &str, max_lines: usize) -> &str {
    let mut end = 0;
    let mut lines_seen = 0;
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            lines_seen += 1;
            if lines_seen >= max_lines {
                end = i + 1; // include the newline
                break;
            }
        }
        end = i + ch.len_utf8();
    }
    &s[..end]
}

/// Metadata returned after a successful spill.
pub(crate) struct SpillInfo {
    /// Absolute path to the written temp file. Used by tests to verify the
    /// spill file was written to the correct location.
    #[allow(dead_code)]
    pub path: std::path::PathBuf,
    /// Workspace-relative display path (e.g. `.ox/tmp/bash-stdout-1713200000000.txt`).
    pub display_path: String,
    /// Total line count of the spilled content.
    pub total_lines: usize,
    /// Total byte size of the spilled content.
    pub total_bytes: usize,
}

/// Build the temp file path: `{workspace_root}/.ox/tmp/{label}-{unix_millis}.txt`.
fn spill_path(workspace_root: &Path, label: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    workspace_root
        .join(".ox")
        .join("tmp")
        .join(format!("{label}-{millis}.txt"))
}

/// Write the full content to a spill file and return metadata. The caller is
/// responsible for checking `needs_spill` first — this function writes
/// unconditionally.
pub(crate) async fn spill<F: FileSystem>(
    fs: &F,
    workspace_root: &Path,
    content: &str,
    label: &str,
) -> Result<SpillInfo> {
    let path = spill_path(workspace_root, label);
    fs.write(&path, content).await?;

    Ok(SpillInfo {
        display_path: display_path(workspace_root, &path),
        path,
        total_lines: content.lines().count(),
        total_bytes: content.len(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;

    // -- needs_spill --

    #[test]
    fn empty_content_never_spills() {
        assert!(!needs_spill(""));
    }

    #[test]
    fn small_content_does_not_spill() {
        let s = "hello\nworld\n";
        assert!(!needs_spill(s));
    }

    #[test]
    fn over_line_threshold_spills() {
        let s: String = (0..INLINE_MAX_LINES + 1)
            .map(|i| format!("line {i}\n"))
            .collect();
        assert!(needs_spill(&s));
    }

    #[test]
    fn over_byte_threshold_spills() {
        // Under line count but over byte count.
        let s = "x".repeat(INLINE_MAX_BYTES + 1);
        assert!(needs_spill(&s));
    }

    #[test]
    fn exactly_at_line_threshold_does_not_spill() {
        let s: String = (0..INLINE_MAX_LINES)
            .map(|i| format!("line {i}\n"))
            .collect();
        assert!(!needs_spill(&s));
    }

    #[test]
    fn one_over_line_threshold_spills() {
        let s: String = (0..INLINE_MAX_LINES + 1)
            .map(|i| format!("line {i}\n"))
            .collect();
        assert!(needs_spill(&s));
    }

    #[test]
    fn exactly_at_byte_threshold_does_not_spill() {
        let s = "x".repeat(INLINE_MAX_BYTES);
        assert!(!needs_spill(&s));
    }

    // -- preview --

    #[test]
    fn preview_returns_first_n_lines() {
        let content = "a\nb\nc\nd\ne\n";
        let p = preview(content, 3);
        assert_eq!(p, "a\nb\nc\n");
    }

    #[test]
    fn preview_of_short_content_returns_full_content() {
        let content = "just two\nlines\n";
        let p = preview(content, 50);
        assert_eq!(p, content);
    }

    #[test]
    fn preview_of_content_without_trailing_newline() {
        let content = "a\nb\nc";
        let p = preview(content, 2);
        assert_eq!(p, "a\nb\n");
    }

    #[test]
    fn preview_of_empty_string() {
        assert_eq!(preview("", 50), "");
    }

    // -- spill --

    #[tokio::test]
    async fn spill_writes_full_content() {
        let fs = Arc::new(FakeFileSystem::new());
        let content = "line 1\nline 2\nline 3\n";
        let info = spill(fs.as_ref(), Path::new("/ws"), content, "test")
            .await
            .unwrap();
        let written = fs.get(&info.path).unwrap();
        assert_eq!(written, content);
    }

    #[tokio::test]
    async fn spill_returns_correct_metadata() {
        let fs = Arc::new(FakeFileSystem::new());
        let content: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let info = spill(fs.as_ref(), Path::new("/ws"), &content, "grep")
            .await
            .unwrap();
        assert_eq!(info.total_lines, 100);
        assert_eq!(info.total_bytes, content.len());
        assert!(info.display_path.starts_with(".ox/tmp/grep-"));
        assert!(info.display_path.ends_with(".txt"));
    }

    #[tokio::test]
    async fn spill_path_is_under_workspace() {
        let fs = Arc::new(FakeFileSystem::new());
        let info = spill(fs.as_ref(), Path::new("/ws"), "data", "bash-stdout")
            .await
            .unwrap();
        assert!(info.path.starts_with("/ws/.ox/tmp/"));
    }
}
