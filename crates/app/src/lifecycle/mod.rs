//! Session-lifecycle controls exposed as LLM tools.
//!
//! Unlike the content-manipulation tools under `app::tools`, these don't
//! touch the filesystem or shell — they flip an in-process signal that the
//! agent's driver drains after each turn's terminal frame. Keeping them in a
//! dedicated module signals their role: they *close* the agent rather than
//! act on its content.
//!
//! The wiring is:
//!
//! 1. `bin-agent::main::run` constructs a shared `Arc<CloseSignal>` and
//!    hands a clone to both `MergeTool` and `AbandonTool` before registering
//!    them. The same `Arc` is passed to `agent_driver` so it can drain the
//!    signal after each terminal frame.
//! 2. The LLM invokes `merge` or `abandon`; the tool stores the
//!    corresponding `CloseIntent` on the signal and returns a
//!    human-readable ack that flows back through the tool loop so the LLM
//!    can acknowledge the close to the user.
//! 3. After the turn's terminal frame, the driver takes the signal; on
//!    `Some(intent)` it emits a single `AgentEvent::RequestClose { intent }`
//!    and exits. The host-side lifecycle coordinator then runs the actual
//!    merge/abandon against the worktree.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use domain::CloseIntent;
use serde::Deserialize;

use crate::stream::ToolDef;
use crate::tools::Tool;

/// A one-shot slot for the next close intent. Tools call `set`; the driver
/// calls `take` after the turn's terminal frame and emits a `RequestClose`
/// frame if the result is `Some`.
///
/// Repeated `set` calls overwrite — last write wins. This matches the
/// natural semantics: a second tool call within the same turn is the LLM
/// changing its mind (e.g. calling `merge` then correcting to `abandon`),
/// and only the latest intent should reach the server. The displaced
/// intent is returned from `set` so the caller can flag it to the model.
pub struct CloseSignal {
    slot: Mutex<Option<CloseIntent>>,
}

impl CloseSignal {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// Replace the current intent and return whatever was there before.
    /// Tools use the return value to warn the LLM about a duplicate call
    /// in the same turn.
    pub fn set(&self, intent: CloseIntent) -> Option<CloseIntent> {
        self.slot.lock().unwrap().replace(intent)
    }

    /// Consume the current intent. Subsequent `take` calls without another
    /// `set` return `None` — the signal is single-shot per set/take pair.
    pub fn take(&self) -> Option<CloseIntent> {
        self.slot.lock().unwrap().take()
    }
}

impl Default for CloseSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool that flags the session for a `Merge` close after the turn ends.
pub struct MergeTool {
    signal: Arc<CloseSignal>,
}

impl MergeTool {
    pub fn new(signal: Arc<CloseSignal>) -> Self {
        Self { signal }
    }
}

impl Tool for MergeTool {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "merge".into(),
            description: "Request that this session be merged and closed after \
                the current turn ends. The server will merge the session's \
                branch into the workspace's base branch, remove the worktree, \
                and delete the session. Rejected if the worktree or main \
                checkout is dirty, or if the merge conflicts. Prefer this \
                over `abandon` when the user wants the work preserved. Takes \
                no arguments."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        _args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let previous = self.signal.set(CloseIntent::Merge);
            Ok(close_ack_string("merge", previous))
        })
    }
}

/// Tool that flags the session for an `Abandon` close after the turn ends.
pub struct AbandonTool {
    signal: Arc<CloseSignal>,
}

impl AbandonTool {
    pub fn new(signal: Arc<CloseSignal>) -> Self {
        Self { signal }
    }
}

#[derive(Debug, Default, Deserialize)]
struct AbandonArgs {
    /// When `true`, the server drops the worktree and branch even if the
    /// worktree has uncommitted changes. Defaults to `false` (server rejects
    /// dirty abandons and expects the UI to re-send with `confirm=true`).
    #[serde(default)]
    confirm: bool,
}

impl Tool for AbandonTool {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "abandon".into(),
            description: "Request that this session be abandoned (discarded) \
                after the current turn ends. The server will remove the \
                worktree and delete the branch without merging. If the \
                worktree has uncommitted changes, the call is rejected \
                unless `confirm` is `true`. Use this when the user wants to \
                throw away the work. Optional `confirm` boolean; defaults to \
                `false`."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Force the abandon even if the worktree is dirty."
                    }
                },
                "additionalProperties": false
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            // `serde_json::from_str("")` is an error, but an empty argument
            // string means "no arguments" from the LLM's perspective — some
            // providers send that for a zero-property tool. Treat it the same
            // as `{}`. `{}` itself is valid JSON and deserializes into the
            // default via `#[serde(default)]` on `confirm`.
            let parsed: AbandonArgs = if args.trim().is_empty() {
                AbandonArgs::default()
            } else {
                serde_json::from_str(args).context("abandon: invalid JSON arguments")?
            };
            let intent = CloseIntent::Abandon {
                confirm: parsed.confirm,
            };
            let previous = self.signal.set(intent);
            Ok(close_ack_string("abandon", previous))
        })
    }
}

/// Build the tool's return string, noting when the signal was already set.
///
/// The ack text flows back through the tool loop so the LLM can phrase its
/// reply to the user accurately: it learns the close is *queued* rather
/// than executed, and sees a hint when it has stacked a second intent on
/// top of an earlier one.
fn close_ack_string(name: &str, previous: Option<CloseIntent>) -> String {
    let base = match name {
        "merge" => "Merge requested. The session will close when this turn ends.",
        "abandon" => "Abandon requested. The session will close when this turn ends.",
        _ => "Close requested. The session will close when this turn ends.",
    };
    match previous {
        None => base.to_owned(),
        Some(CloseIntent::Merge) => {
            format!("{base} (Replaces an earlier `merge` request for this turn.)")
        }
        Some(CloseIntent::Abandon { .. }) => {
            format!("{base} (Replaces an earlier `abandon` request for this turn.)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CloseSignal --------------------------------------------------------

    #[test]
    fn close_signal_starts_empty() {
        let sig = CloseSignal::new();
        assert!(sig.take().is_none());
    }

    #[test]
    fn close_signal_take_is_consume_once() {
        let sig = CloseSignal::new();
        assert!(sig.set(CloseIntent::Merge).is_none());
        assert_eq!(sig.take(), Some(CloseIntent::Merge));
        // Second take returns None until set is called again.
        assert!(sig.take().is_none());
    }

    #[test]
    fn close_signal_set_overwrites_and_returns_previous() {
        let sig = CloseSignal::new();
        assert!(sig.set(CloseIntent::Merge).is_none());
        let previous = sig.set(CloseIntent::Abandon { confirm: true });
        assert_eq!(previous, Some(CloseIntent::Merge));
        assert_eq!(
            sig.take(),
            Some(CloseIntent::Abandon { confirm: true }),
            "last write wins",
        );
    }

    // ---- MergeTool ----------------------------------------------------------

    #[tokio::test]
    async fn merge_tool_sets_merge_intent_and_returns_ack() {
        let sig = Arc::new(CloseSignal::new());
        let tool = MergeTool::new(sig.clone());
        let ack = tool.execute("{}").await.unwrap();
        assert_eq!(
            ack,
            "Merge requested. The session will close when this turn ends."
        );
        assert_eq!(sig.take(), Some(CloseIntent::Merge));
    }

    #[tokio::test]
    async fn merge_tool_ignores_extra_args() {
        // Models sometimes pad tool-call args with stray fields; the tool
        // should still succeed because the schema declares no required keys.
        let sig = Arc::new(CloseSignal::new());
        let tool = MergeTool::new(sig.clone());
        let ack = tool.execute(r#"{"unused": 42}"#).await.unwrap();
        assert!(ack.contains("Merge requested"));
        assert_eq!(sig.take(), Some(CloseIntent::Merge));
    }

    #[tokio::test]
    async fn merge_tool_repeated_call_annotates_duplicate() {
        let sig = Arc::new(CloseSignal::new());
        let tool = MergeTool::new(sig.clone());
        let first = tool.execute("{}").await.unwrap();
        assert!(!first.contains("Replaces"));

        let second = tool.execute("{}").await.unwrap();
        assert!(
            second.contains("Replaces an earlier `merge` request"),
            "got {second:?}"
        );
        // Signal still resolves to Merge — last write wins.
        assert_eq!(sig.take(), Some(CloseIntent::Merge));
    }

    // ---- AbandonTool --------------------------------------------------------

    #[tokio::test]
    async fn abandon_tool_without_confirm_sets_false() {
        let sig = Arc::new(CloseSignal::new());
        let tool = AbandonTool::new(sig.clone());
        let ack = tool.execute("{}").await.unwrap();
        assert_eq!(
            ack,
            "Abandon requested. The session will close when this turn ends."
        );
        assert_eq!(sig.take(), Some(CloseIntent::Abandon { confirm: false }));
    }

    #[tokio::test]
    async fn abandon_tool_with_confirm_true_is_forwarded() {
        let sig = Arc::new(CloseSignal::new());
        let tool = AbandonTool::new(sig.clone());
        let ack = tool.execute(r#"{"confirm": true}"#).await.unwrap();
        assert!(ack.contains("Abandon requested"));
        assert_eq!(sig.take(), Some(CloseIntent::Abandon { confirm: true }));
    }

    #[tokio::test]
    async fn abandon_tool_with_confirm_false_is_forwarded() {
        let sig = Arc::new(CloseSignal::new());
        let tool = AbandonTool::new(sig.clone());
        let _ = tool.execute(r#"{"confirm": false}"#).await.unwrap();
        assert_eq!(sig.take(), Some(CloseIntent::Abandon { confirm: false }));
    }

    #[tokio::test]
    async fn abandon_tool_accepts_empty_arg_string() {
        // Some providers emit an empty string when the model calls a tool
        // with no arguments. serde_json rejects that; the tool must not.
        let sig = Arc::new(CloseSignal::new());
        let tool = AbandonTool::new(sig.clone());
        let ack = tool.execute("").await.unwrap();
        assert!(ack.contains("Abandon requested"));
        assert_eq!(sig.take(), Some(CloseIntent::Abandon { confirm: false }));
    }

    #[tokio::test]
    async fn abandon_tool_rejects_malformed_json() {
        let sig = Arc::new(CloseSignal::new());
        let tool = AbandonTool::new(sig.clone());
        let err = tool
            .execute("not json at all")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("abandon"), "got {err}");
        // A parse failure must leave the signal untouched — the LLM sees the
        // tool error and can retry, and the driver will not queue a close.
        assert!(sig.take().is_none());
    }

    // ---- Cross-tool interactions -------------------------------------------

    #[tokio::test]
    async fn abandon_overwrites_prior_merge_and_warns() {
        let sig = Arc::new(CloseSignal::new());
        let merge = MergeTool::new(sig.clone());
        let abandon = AbandonTool::new(sig.clone());

        let _ = merge.execute("{}").await.unwrap();
        let ack = abandon.execute(r#"{"confirm": true}"#).await.unwrap();
        assert!(
            ack.contains("Replaces an earlier `merge` request"),
            "got {ack:?}"
        );
        assert_eq!(sig.take(), Some(CloseIntent::Abandon { confirm: true }));
    }

    #[tokio::test]
    async fn merge_overwrites_prior_abandon_and_warns() {
        let sig = Arc::new(CloseSignal::new());
        let merge = MergeTool::new(sig.clone());
        let abandon = AbandonTool::new(sig.clone());

        let _ = abandon.execute(r#"{"confirm": false}"#).await.unwrap();
        let ack = merge.execute("{}").await.unwrap();
        assert!(
            ack.contains("Replaces an earlier `abandon` request"),
            "got {ack:?}"
        );
        assert_eq!(sig.take(), Some(CloseIntent::Merge));
    }

    // ---- Schemas -----------------------------------------------------------

    #[test]
    fn merge_tool_schema_is_empty_object() {
        let def = MergeTool::new(Arc::new(CloseSignal::new())).def();
        assert_eq!(def.name, "merge");
        // Schema should accept `{}` — no required properties, no extras.
        assert_eq!(def.parameters["type"], "object");
        assert_eq!(def.parameters["additionalProperties"], false);
    }

    #[test]
    fn abandon_tool_schema_declares_confirm_boolean() {
        let def = AbandonTool::new(Arc::new(CloseSignal::new())).def();
        assert_eq!(def.name, "abandon");
        assert_eq!(def.parameters["type"], "object");
        assert_eq!(
            def.parameters["properties"]["confirm"]["type"], "boolean",
            "schema must expose `confirm` so providers advertise it to the model"
        );
        assert_eq!(def.parameters["additionalProperties"], false);
    }
}
