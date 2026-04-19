//! `todo_write` tool — the LLM's task-list scratchpad.
//!
//! Stateless by design. Each call replaces the full list; the transcript
//! itself stores the running history via the `ContentBlock::ToolCall` the
//! tool was invoked through. Nothing in the app layer — no store, no port,
//! no event — needs to persist a list between calls. The web UI parses the
//! same raw arguments JSON at render time and draws a glyph-prefixed list
//! inline at the tool-call's chronological position.
//!
//! No filesystem, network, or process side effects, so approval is always
//! `NotRequired`.

use std::future::Future;
use std::pin::Pin;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::Tool;
use crate::approval::ApprovalRequirement;
use crate::stream::ToolDef;

/// Serde validator for the status enum. `rename_all = "snake_case"` matches
/// the C# reference's wire format (`pending`, `in_progress`, `completed`);
/// any other string deserializes to a `serde_json` error whose message
/// names the offending value, which is what we want the LLM to see.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single todo item. Fields are populated by serde during validation; the
/// tool itself never reads them — rendering happens in the frontend off the
/// raw JSON. `#[allow(dead_code)]` is deliberate: the whole point of this
/// struct is to make serde reject malformed items before the call succeeds.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TodoItem {
    content: String,
    status: TodoStatus,
}

/// Argument envelope. Having the `todos` field keeps error messages from
/// serde specific ("missing field `todos`") rather than "missing data at
/// root".
#[derive(Debug, Deserialize)]
struct TodoArgs {
    todos: Vec<TodoItem>,
}

/// Unit struct — no dependencies, no state. Constructing it via
/// `Arc::new(TodoWriteTool)` in the composition root is enough.
pub struct TodoWriteTool;

impl Tool for TodoWriteTool {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "todo_write".into(),
            description: "Update the task list displayed in the conversation. \
                Use this for multi-step tasks (3+ steps) to track progress. \
                Each call replaces the entire list — always send all items, \
                not just changed ones. Mark items \"in_progress\" before \
                starting and \"completed\" when done. Keep at most one item \
                \"in_progress\" at a time. Send an empty list to clear the \
                task list when all work is complete."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete todo list. Every call replaces the entire list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Brief task description in imperative form."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "pending = not started, in_progress = currently working, completed = done."
                                }
                            },
                            "required": ["content", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["todos"],
                "additionalProperties": false
            }),
        }
    }

    fn approval_requirement<'a>(
        &'a self,
        _args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<ApprovalRequirement>> + Send + 'a>> {
        // No side effects beyond shaping the transcript, so never prompt.
        Box::pin(async move { Ok(ApprovalRequirement::NotRequired) })
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: TodoArgs =
                serde_json::from_str(args).context("todo_write: invalid JSON arguments")?;
            // The ack text echoes the item count so the LLM has a signal that
            // the list it just sent was accepted — including for the
            // "send [] to clear" case.
            Ok(format!("Todo list updated ({} items).", parsed.todos.len()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn valid_multi_item_list_returns_count_ack() {
        let args = r#"{
            "todos": [
                {"content": "plan", "status": "completed"},
                {"content": "build", "status": "in_progress"},
                {"content": "ship", "status": "pending"}
            ]
        }"#;
        let out = TodoWriteTool.execute(args).await.unwrap();
        assert_eq!(out, "Todo list updated (3 items).");
    }

    #[tokio::test]
    async fn empty_list_is_supported_and_counts_zero() {
        // Sending `[]` is the "clear the panel" signal documented in the
        // system prompt. The tool must accept it rather than rejecting on an
        // "empty list" rule the schema does not declare.
        let out = TodoWriteTool.execute(r#"{"todos": []}"#).await.unwrap();
        assert_eq!(out, "Todo list updated (0 items).");
    }

    #[tokio::test]
    async fn unknown_status_string_is_an_error_naming_the_value() {
        // serde's own error message for a rename_all = snake_case enum with
        // an unknown variant includes the offending variant name — that's
        // the behavior we want the LLM to see so it can self-correct.
        let args = r#"{
            "todos": [{"content": "plan", "status": "blocked"}]
        }"#;
        let err = TodoWriteTool.execute(args).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            chain.contains("blocked"),
            "error chain should name the offending status value, got: {chain}",
        );
    }

    #[tokio::test]
    async fn missing_todos_field_is_an_error() {
        let err = TodoWriteTool.execute(r#"{}"#).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(chain.contains("todos"), "got: {chain}");
    }

    #[tokio::test]
    async fn item_missing_content_is_an_error() {
        let args = r#"{"todos": [{"status": "pending"}]}"#;
        let err = TodoWriteTool.execute(args).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(chain.contains("content"), "got: {chain}");
    }

    #[tokio::test]
    async fn item_missing_status_is_an_error() {
        let args = r#"{"todos": [{"content": "plan"}]}"#;
        let err = TodoWriteTool.execute(args).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(chain.contains("status"), "got: {chain}");
    }

    #[tokio::test]
    async fn malformed_json_is_an_error() {
        let err = TodoWriteTool.execute("not json at all").await.unwrap_err();
        // The top-level context we attach is the user-facing hint; verifying
        // it ensures a parse failure bubbles up through our wrapper rather
        // than a bare serde message.
        let msg = err.to_string();
        assert!(msg.contains("todo_write"), "got: {msg}");
    }

    #[tokio::test]
    async fn approval_requirement_is_not_required_even_for_bogus_args() {
        // The approval gate runs before execute, so it must short-circuit
        // to NotRequired without parsing — a malformed JSON argument should
        // still return NotRequired, not an error. This keeps the approval
        // contract consistent with the tool's "no side effects" guarantee.
        let r = TodoWriteTool
            .approval_requirement("this is not JSON")
            .await
            .unwrap();
        assert_eq!(r, ApprovalRequirement::NotRequired);

        let r = TodoWriteTool
            .approval_requirement(r#"{"todos": []}"#)
            .await
            .unwrap();
        assert_eq!(r, ApprovalRequirement::NotRequired);
    }

    #[tokio::test]
    async fn empty_content_string_is_accepted() {
        // Mirrors the C# reference: the schema requires `content` to be
        // present and a string, but does not forbid empty strings. The
        // validator stays out of the way so the prompt/LLM owns content
        // quality decisions. Pinning this behavior so a future "improvement"
        // that rejects empties has to come with a deliberate plan change.
        let args = r#"{"todos": [{"content": "", "status": "pending"}]}"#;
        let out = TodoWriteTool.execute(args).await.unwrap();
        assert_eq!(out, "Todo list updated (1 items).");
    }

    #[tokio::test]
    async fn duplicate_items_are_accepted() {
        // The list is positional; the LLM is free to include repeats. No
        // dedup, no uniqueness constraint — again matching the C# reference.
        let args = r#"{
            "todos": [
                {"content": "task", "status": "pending"},
                {"content": "task", "status": "pending"}
            ]
        }"#;
        let out = TodoWriteTool.execute(args).await.unwrap();
        assert_eq!(out, "Todo list updated (2 items).");
    }

    #[tokio::test]
    async fn multiple_in_progress_items_are_accepted() {
        // "At most one in_progress at a time" is prompt guidance, not a
        // validator rule. Pinning this so the schema contract stays clean
        // — if we ever want to enforce it, that becomes a conscious change.
        let args = r#"{
            "todos": [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"}
            ]
        }"#;
        let out = TodoWriteTool.execute(args).await.unwrap();
        assert_eq!(out, "Todo list updated (2 items).");
    }

    #[tokio::test]
    async fn non_string_content_is_rejected() {
        // If someone later widens `TodoItem::content` to `serde_json::Value`
        // or swaps in a custom deserializer, this test ensures the contract
        // "content is a string" stays pinned at execute time — not just at
        // the schema layer (which the LLM is free to ignore). The real
        // regression signal here is `unwrap_err()`: a looser impl would
        // return `Ok` for a numeric content, failing at this line.
        let args = r#"{"todos": [{"content": 42, "status": "pending"}]}"#;
        let err = TodoWriteTool.execute(args).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        // `expected` is serde's pervasive phrasing for type-mismatch errors
        // ("expected a string", "expected value", ...). Asserting on it
        // pins that a type-level rejection (not some unrelated error) is
        // what came through, while staying stable across serde minor
        // versions that may tweak exact wording.
        assert!(
            chain.contains("todo_write") && chain.contains("expected"),
            "got: {chain}",
        );
    }

    #[tokio::test]
    async fn non_string_status_is_rejected() {
        // Same intent as the content test above — a numeric `status` must
        // be rejected, not silently coerced, so the enum gate isn't
        // bypassable by changing the JSON type. `unwrap_err()` is the real
        // regression detector; the chain assertion just verifies the error
        // came through our wrapper as a serde type-mismatch rather than,
        // say, a panic or an unrelated I/O failure.
        let args = r#"{"todos": [{"content": "x", "status": 1}]}"#;
        let err = TodoWriteTool.execute(args).await.unwrap_err();
        let chain: String = err
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            chain.contains("todo_write") && chain.contains("expected"),
            "got: {chain}",
        );
    }

    #[test]
    fn tool_def_declares_expected_schema_shape() {
        let def = TodoWriteTool.def();
        assert_eq!(def.name, "todo_write");

        // Top-level: strict object that requires `todos`. `additionalProperties:
        // false` matters — it prevents the LLM from smuggling unrelated keys
        // past the schema gate and noticing nothing happened to them.
        assert_eq!(def.parameters["type"], "object");
        assert_eq!(def.parameters["additionalProperties"], false);
        assert_eq!(def.parameters["required"][0], "todos");

        // Item-level constraints. A regression that dropped any of these
        // would silently weaken the schema the LLM sees — serde would still
        // catch malformed inputs at execute time, but the tool's public
        // contract (what the model is told to send) would drift from the
        // validator. Pin them individually so such a drift fails loudly.
        let items = &def.parameters["properties"]["todos"]["items"];
        assert_eq!(items["type"], "object");
        assert_eq!(items["additionalProperties"], false);
        let item_required: Vec<&str> = items["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            item_required.contains(&"content") && item_required.contains(&"status"),
            "items must declare both content and status as required, got {item_required:?}",
        );
        assert_eq!(items["properties"]["content"]["type"], "string");
        assert_eq!(items["properties"]["status"]["type"], "string");

        // Status is a closed enum — the LLM can't sneak in a new state by
        // calling it something new. Schema-level gate in addition to the
        // serde-level gate on the Rust side.
        let status_enum = &items["properties"]["status"]["enum"];
        let values: Vec<&str> = status_enum
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(values, vec!["pending", "in_progress", "completed"]);
    }
}
