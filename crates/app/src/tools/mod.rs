//! The `Tool` port-like trait and the `ToolRegistry` collection.
//!
//! Tools live in the app layer rather than in `ports.rs` because a tool is
//! not infrastructure — it's business logic that *depends* on infrastructure
//! ports (notably `FileSystem`). The concrete file-editing tools are defined
//! in sibling modules and take a generic `F: FileSystem` so they can be
//! unit-tested against `FakeFileSystem` without any filesystem I/O.
//!
//! ## Object safety
//!
//! Unlike `LlmProvider` and `SessionStore` — each of which has exactly one
//! in-process implementation in the composition root, so we can keep them as
//! generic trait bounds — the registry holds a *heterogeneous* collection of
//! tools. That forces `Arc<dyn Tool>`, which in turn forces the async method
//! to return a boxed future (RPIT-in-trait is not `dyn`-compatible).
//!
//! This is a deliberately scoped deviation from the rest of the codebase's
//! "no `dyn`, no `async-trait`" style, not a new convention.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, bail};

use crate::stream::ToolDef;

pub mod bash;
pub mod edit_file;
pub mod glob;
pub mod grep;
pub mod hashlines;
pub mod read_file;
pub mod write_file;

pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

/// A callable function the LLM can invoke during a turn.
///
/// `def` returns the tool's schema — sent to the LLM as part of every stream
/// request. `execute` receives the raw JSON arguments string and produces a
/// textual result that is wrapped in a `Role::Tool` message and fed back to
/// the LLM on the next iteration of the tool loop.
///
/// The return type is `String` rather than anything structured because every
/// model provider collapses tool results to a string on the wire anyway.
/// Tools do their own JSON parsing on input (via `serde_json::from_str`) and
/// produce human-readable (and LLM-readable) output.
pub trait Tool: Send + Sync {
    fn def(&self) -> ToolDef;

    /// Execute the tool with raw JSON arguments. Returns a success string or
    /// an error. An error is translated by the caller into a
    /// `ToolResult { is_error: true }` message — the loop continues, giving
    /// the LLM a chance to self-correct.
    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

/// A name-indexed collection of tools.
///
/// The registry is a concrete (non-generic) type so `SessionRunner`'s type
/// parameters don't multiply. It owns `Arc<dyn Tool>` entries — cheap to
/// clone for dispatch, and consistent with how most hexagonal systems hand
/// around trait objects that don't participate in a lifetime dance.
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Append a tool. Registration order is preserved by `defs()`, which
    /// matters because providers often surface tools to the model in the
    /// order the schema lists them.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Snapshot of every registered tool's schema, in registration order.
    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.def()).collect()
    }

    /// `true` if no tools are registered — used by `SessionRunner` to skip
    /// the tool-call loop entirely when no tools are wired up, preserving
    /// the current single-turn behavior for the no-tools composition case.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Dispatch to the tool named `name`, passing `args` verbatim.
    /// Unknown-name errors are recoverable — the caller should translate
    /// them into a tool-result message and let the loop keep going.
    pub async fn execute(&self, name: &str, args: &str) -> Result<String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.def().name == name)
            .ok_or_else(|| anyhow::anyhow!("unknown tool: {name}"))?;
        tool.execute(args).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper used by `SessionRunner` and tests to pull out the tool-call
/// triples from an assistant message. Not public API.
pub(crate) fn extract_tool_calls(msg: &domain::Message) -> Vec<(String, String, String)> {
    msg.content
        .iter()
        .filter_map(|b| match b {
            domain::ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some((id.clone(), name.clone(), arguments.clone())),
            _ => None,
        })
        .collect()
}

/// Resolve `file_path` relative to `workspace_root` if it's not absolute.
/// Used by every file tool so the LLM can pass either style.
pub(crate) fn resolve_path(
    workspace_root: &std::path::Path,
    file_path: &str,
) -> std::path::PathBuf {
    let p = std::path::Path::new(file_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        workspace_root.join(p)
    }
}

/// Best-effort conversion of an absolute path back to a workspace-relative
/// display string. Falls back to the absolute path if `path` doesn't sit
/// under `workspace_root` — the tool result is for a human/LLM reader, not
/// for further automation, so a plain absolute path is fine as a fallback.
pub(crate) fn display_path(workspace_root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(workspace_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

/// Small guard: `bail_if_empty` keeps each tool's JSON-field validation
/// compact without a helper that would be overkill for a single line.
#[inline]
pub(crate) fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeTool;

    #[tokio::test]
    async fn execute_dispatches_by_name() {
        let a = Arc::new(FakeTool::new("a"));
        a.push_ok("from-a");
        let b = Arc::new(FakeTool::new("b"));
        b.push_ok("from-b");

        let mut reg = ToolRegistry::new();
        reg.register(a.clone() as Arc<dyn Tool>);
        reg.register(b.clone() as Arc<dyn Tool>);

        assert_eq!(reg.execute("b", "{}").await.unwrap(), "from-b");
        assert_eq!(reg.execute("a", "{}").await.unwrap(), "from-a");
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error() {
        let reg = ToolRegistry::new();
        let err = reg.execute("missing", "{}").await.unwrap_err().to_string();
        assert!(err.contains("missing"), "got {err}");
    }

    #[test]
    fn defs_preserves_registration_order() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(FakeTool::new("first")) as Arc<dyn Tool>);
        reg.register(Arc::new(FakeTool::new("second")) as Arc<dyn Tool>);
        reg.register(Arc::new(FakeTool::new("third")) as Arc<dyn Tool>);

        let names: Vec<_> = reg.defs().into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    #[test]
    fn empty_is_true_until_a_tool_is_registered() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(Arc::new(FakeTool::new("t")) as Arc<dyn Tool>);
        assert!(!reg.is_empty());
    }
}
