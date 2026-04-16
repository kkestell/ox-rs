use anyhow::Result;
use app::ToolDef;
use domain::{ContentBlock, Message, Role};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct RequestBody {
    pub model: String,
    pub messages: Vec<WireMessage>,
    pub stream: bool,
    /// Tells the model that reasoning is desired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    /// Older flag some models still require.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
}

#[derive(Debug, Serialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: WireFunction,
}

#[derive(Debug, Serialize)]
pub struct WireFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct WireTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: WireToolFunction,
}

#[derive(Debug, Serialize)]
pub struct WireToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Response types (SSE chunks)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SseChunk {
    #[serde(default)]
    pub choices: Vec<SseChoice>,
    pub usage: Option<SseUsage>,
}

#[derive(Debug, Deserialize)]
pub struct SseChoice {
    pub delta: SseDelta,
}

#[derive(Debug, Deserialize)]
pub struct SseDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_details: Option<Vec<SseReasoningDetail>>,
    #[serde(default)]
    pub tool_calls: Option<Vec<SseToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct SseReasoningDetail {
    #[serde(rename = "type")]
    pub detail_type: Option<String>,
    pub data: Option<String>,
    pub format: Option<String>,
    pub signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SseToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<SseToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub struct SseToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversion: domain types → wire request
// ---------------------------------------------------------------------------

impl RequestBody {
    pub fn from_messages(
        model: &str,
        messages: &[Message],
        system_prompt: &str,
        tools: &[ToolDef],
    ) -> Result<Self> {
        let mut wire_messages: Vec<WireMessage> = Vec::new();

        // Prepend the system prompt as a system-role message when provided.
        // This stays out of the domain model — it's a wire-level concern.
        if !system_prompt.is_empty() {
            wire_messages.push(WireMessage {
                role: "system".to_owned(),
                content: Some(system_prompt.to_owned()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        wire_messages.extend(
            messages
                .iter()
                .map(wire_message)
                .collect::<Result<Vec<_>>>()?,
        );

        let wire_tools: Vec<WireTool> = tools
            .iter()
            .map(|t| WireTool {
                tool_type: "function".to_owned(),
                function: WireToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect();

        Ok(Self {
            model: model.to_owned(),
            messages: wire_messages,
            stream: true,
            reasoning: Some(serde_json::json!({})),
            include_reasoning: Some(true),
            tools: wire_tools,
        })
    }
}

/// Convert a domain `Message` into the wire format OpenRouter expects.
/// Reasoning blocks are intentionally omitted — the API does not accept them
/// as input, only produces them in responses.
fn wire_message(msg: &Message) -> Result<WireMessage> {
    match msg.role {
        Role::User => Ok(WireMessage {
            role: "user".to_owned(),
            content: Some(msg.text()),
            tool_calls: None,
            tool_call_id: None,
        }),
        Role::Assistant => {
            let tool_calls: Vec<WireToolCall> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(WireToolCall {
                        id: id.clone(),
                        call_type: "function".to_owned(),
                        function: WireFunction {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    }),
                    _ => None,
                })
                .collect();

            Ok(WireMessage {
                role: "assistant".to_owned(),
                content: {
                    let t = msg.text();
                    if t.is_empty() { None } else { Some(t) }
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            })
        }
        Role::Tool => {
            // A tool message must have exactly one ToolResult block — producing
            // an empty tool_call_id / empty content would silently poison the
            // wire request, so we fail loudly instead.
            let (call_id, content) = msg
                .content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::ToolResult {
                        tool_call_id,
                        content,
                        ..
                    } => Some((tool_call_id.clone(), content.clone())),
                    _ => None,
                })
                .ok_or_else(|| anyhow::anyhow!("Role::Tool message has no ToolResult block"))?;

            Ok(WireMessage {
                role: "tool".to_owned(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: Some(call_id),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_conversion() {
        let msg = Message::user("hello");
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.role, "user");
        assert_eq!(wire.content.as_deref(), Some("hello"));
        assert!(wire.tool_calls.is_none());
        assert!(wire.tool_call_id.is_none());
    }

    #[test]
    fn assistant_text_only() {
        let msg = Message::assistant(vec![ContentBlock::Text {
            text: "reply".into(),
        }]);
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.role, "assistant");
        assert_eq!(wire.content.as_deref(), Some("reply"));
        assert!(wire.tool_calls.is_none());
    }

    #[test]
    fn assistant_with_tool_calls_and_no_text() {
        let msg = Message::assistant(vec![ContentBlock::ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"a.rs"}"#.into(),
        }]);
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.role, "assistant");
        assert!(wire.content.is_none());
        let calls = wire.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn tool_result_conversion() {
        let msg = Message::tool_result("call_1", "file contents here", false);
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.role, "tool");
        assert_eq!(wire.content.as_deref(), Some("file contents here"));
        assert_eq!(wire.tool_call_id.as_deref(), Some("call_1"));
        assert!(wire.tool_calls.is_none());
    }

    #[test]
    fn tool_result_with_empty_content_is_valid() {
        let msg = Message::tool_result("call_1", "", false);
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.role, "tool");
        assert_eq!(wire.content.as_deref(), Some(""));
        assert_eq!(wire.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn wire_message_errors_on_role_tool_without_tool_result_block() {
        let msg = Message {
            role: Role::Tool,
            content: vec![ContentBlock::Text {
                text: "wrong".into(),
            }],
            token_count: 0,
        };
        let err = wire_message(&msg).unwrap_err();
        assert!(
            err.to_string().contains("no ToolResult block"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reasoning_blocks_are_excluded_from_wire() {
        let msg = Message::assistant(vec![
            ContentBlock::Reasoning {
                content: "thinking...".into(),
                signature: Some("sig".into()),
                encrypted: None,
                format: None,
            },
            ContentBlock::Text {
                text: "answer".into(),
            },
        ]);
        let wire = wire_message(&msg).unwrap();
        assert_eq!(wire.content.as_deref(), Some("answer"));
        assert!(wire.tool_calls.is_none());
    }

    #[test]
    fn request_body_includes_tools() {
        let tools = vec![ToolDef {
            name: "read_file".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let body = RequestBody::from_messages("test-model", &[], "", &tools).unwrap();
        assert_eq!(body.model, "test-model");
        assert!(body.stream);
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].function.name, "read_file");
    }

    #[test]
    fn system_prompt_prepended_as_first_message() {
        let msgs = vec![Message::user("hello")];
        let body =
            RequestBody::from_messages("m", &msgs, "You are a coding assistant.", &[]).unwrap();
        assert_eq!(body.messages.len(), 2);
        assert_eq!(body.messages[0].role, "system");
        assert_eq!(
            body.messages[0].content.as_deref(),
            Some("You are a coding assistant.")
        );
        assert_eq!(body.messages[1].role, "user");
    }

    #[test]
    fn empty_system_prompt_produces_no_system_message() {
        let msgs = vec![Message::user("hello")];
        let body = RequestBody::from_messages("m", &msgs, "", &[]).unwrap();
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
    }
}
