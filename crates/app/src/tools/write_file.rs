//! `write_file` tool.
//!
//! Writes raw content to a path, creating parent directories as needed. The
//! success string includes the byte count and the workspace-relative path so
//! the LLM can summarize what it just did without having to remember the
//! original arguments.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{Tool, display_path, require_non_empty, resolve_path};
use crate::ports::FileSystem;
use crate::stream::ToolDef;

pub struct WriteFileTool<F> {
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<F> WriteFileTool<F> {
    pub fn new(fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self { fs, workspace_root }
    }
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    file_path: String,
    content: String,
}

impl<F: FileSystem + Send + Sync + 'static> Tool for WriteFileTool<F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "write_file".into(),
            description: "Write `content` to `file_path`, creating parent directories as needed. \
                Overwrites any existing file. Use for creating new files or replacing \
                entire files; for targeted edits to an existing file, use `edit_file`."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Destination path, absolute or workspace-relative."
                    },
                    "content": {
                        "type": "string",
                        "description": "Raw file contents to write."
                    }
                },
                "required": ["file_path", "content"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: WriteArgs =
                serde_json::from_str(args).context("write_file: invalid JSON arguments")?;
            require_non_empty("file_path", &parsed.file_path)?;

            let path = resolve_path(&self.workspace_root, &parsed.file_path);
            let bytes = parsed.content.len();
            self.fs
                .write(&path, &parsed.content)
                .with_context(|| format!("write_file: could not write {}", path.display()))?;

            Ok(format!(
                "Wrote {bytes} bytes to {}",
                display_path(&self.workspace_root, &path)
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeFileSystem;

    fn tool(fs: Arc<FakeFileSystem>, root: &str) -> WriteFileTool<FakeFileSystem> {
        WriteFileTool::new(fs, PathBuf::from(root))
    }

    #[tokio::test]
    async fn creates_new_file_under_workspace_root() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs.clone(), "/ws");
        let out = t
            .execute(r#"{"file_path":"notes.txt","content":"hello"}"#)
            .await
            .unwrap();

        assert!(out.contains("5 bytes"), "got {out}");
        assert!(out.contains("notes.txt"), "got {out}");
        assert_eq!(
            fs.get(std::path::Path::new("/ws/notes.txt")).as_deref(),
            Some("hello")
        );
    }

    #[tokio::test]
    async fn overwrites_existing_file() {
        let fs = Arc::new(FakeFileSystem::new());
        fs.insert("/ws/f.txt", "old");
        let t = tool(fs.clone(), "/ws");

        t.execute(r#"{"file_path":"f.txt","content":"new content"}"#)
            .await
            .unwrap();
        assert_eq!(
            fs.get(std::path::Path::new("/ws/f.txt")).as_deref(),
            Some("new content")
        );
    }

    #[tokio::test]
    async fn records_parent_dir_creation_for_new_subdirectory() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs.clone(), "/ws");
        t.execute(r#"{"file_path":"a/b/c.txt","content":"x"}"#)
            .await
            .unwrap();

        // The fake's created_dirs reflects the intent that the real impl
        // runs create_dir_all on the parent before writing.
        let dirs = fs.created_dirs();
        assert!(
            dirs.iter().any(|p| p.ends_with("a/b")),
            "expected parent dir recorded, got {dirs:?}"
        );
    }

    #[tokio::test]
    async fn absolute_path_is_honored() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs.clone(), "/ws");
        t.execute(r#"{"file_path":"/other/x.txt","content":"y"}"#)
            .await
            .unwrap();
        assert_eq!(
            fs.get(std::path::Path::new("/other/x.txt")).as_deref(),
            Some("y")
        );
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(t.execute("garbage").await.is_err());
    }

    #[tokio::test]
    async fn empty_file_path_is_an_error() {
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(fs, "/ws");
        assert!(
            t.execute(r#"{"file_path":"","content":"x"}"#)
                .await
                .is_err()
        );
    }
}
