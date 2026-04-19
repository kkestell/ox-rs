use serde::{Deserialize, Serialize};

use crate::ContentBlock;
use crate::stream::Usage;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    /// Token usage for the LLM call that produced this message. Only
    /// populated on assistant messages emitted by a provider that reports
    /// usage (OpenRouter's `Finished` chunk). User and tool messages, plus
    /// assistant messages from providers without usage reporting, carry
    /// `None`. The full `Usage` struct is preserved — not just a single
    /// token count — so UI layers can pick whichever field they need
    /// (e.g. `prompt_tokens` for the context-usage chip) without the
    /// domain type locking in one interpretation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
            usage: None,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            usage: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: tool_call_id.into(),
                content: content.into(),
                is_error,
            }],
            usage: None,
        }
    }

    /// Concatenates all `Text` blocks into a single string.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns references to all `ToolCall` blocks.
    pub fn tool_calls(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolCall { .. }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_assistant_message_with_usage() {
        // Assistant messages emitted by a provider with usage reporting
        // must preserve the full `Usage` struct across disk/IPC
        // serialization so the browser can render prompt/completion
        // counts after a page reload.
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "hi".into() }],
            usage: Some(Usage {
                prompt_tokens: 123,
                completion_tokens: 45,
                reasoning_tokens: 6,
            }),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, msg.role);
        assert_eq!(back.content, msg.content);
        assert_eq!(back.usage, msg.usage);
    }

    #[test]
    fn roundtrip_user_message_without_usage() {
        // User and tool-result messages must round-trip with
        // `usage: None`. `skip_serializing_if` keeps the JSON tidy,
        // and `#[serde(default)]` lets older messages (or hand-written
        // fixtures) omit the field entirely without breaking.
        let msg = Message::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(
            !json.contains("\"usage\""),
            "None usage must be omitted from the wire form: {json}"
        );
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.usage, None);
    }

    #[test]
    fn deserialize_message_missing_usage_field() {
        // Hand-crafted JSON without the `usage` key must deserialize
        // to `None`. This exercises `#[serde(default)]` and guards
        // against a future refactor that silently flips the attribute.
        let json = r#"{"role":"user","content":[{"type":"text","text":"hi"}]}"#;
        let back: Message = serde_json::from_str(json).unwrap();
        assert_eq!(back.role, Role::User);
        assert_eq!(back.usage, None);
    }
}
