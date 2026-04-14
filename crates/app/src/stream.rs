use std::collections::BTreeMap;

use domain::{ContentBlock, Message, Role};

/// In-flight streaming events from an LLM provider. These represent incremental
/// chunks as they arrive over the wire — never serialized, only consumed by
/// the accumulator to produce at-rest `Message` values.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ReasoningDelta(String),
    /// Encrypted reasoning arrives as a single opaque blob, not a delta stream.
    EncryptedReasoning {
        data: String,
        format: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentDelta {
        index: usize,
        delta: String,
    },
    /// Anthropic-style signature for reasoning verification, separate from content.
    ReasoningSignature(String),
    Finished {
        usage: Usage,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
}

/// Definition of a tool the model can call.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// StreamAccumulator — bridges in-flight events to at-rest Message
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

/// Consumes `StreamEvent`s and assembles a completed `Message`.
///
/// Content blocks are ordered: reasoning first, then text, then tool calls.
/// This matches the logical structure of a model turn even when deltas arrive
/// interleaved.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    text: String,
    reasoning: String,
    signature: Option<String>,
    encrypted: Option<(String, String)>, // (data, format)
    tool_calls: BTreeMap<usize, ToolCallAccum>,
    usage: Usage,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::TextDelta(delta) => self.text.push_str(&delta),
            StreamEvent::ReasoningDelta(delta) => self.reasoning.push_str(&delta),
            StreamEvent::EncryptedReasoning { data, format } => {
                self.encrypted = Some((data, format));
            }
            StreamEvent::ToolCallStart { index, id, name } => {
                self.tool_calls.insert(
                    index,
                    ToolCallAccum {
                        id,
                        name,
                        arguments: String::new(),
                    },
                );
            }
            StreamEvent::ToolCallArgumentDelta { index, delta } => {
                if let Some(tc) = self.tool_calls.get_mut(&index) {
                    tc.arguments.push_str(&delta);
                }
            }
            StreamEvent::ReasoningSignature(sig) => {
                self.signature = Some(sig);
            }
            StreamEvent::Finished { usage } => {
                self.usage = usage;
            }
        }
    }

    pub fn into_message(self) -> Message {
        let mut blocks = Vec::new();

        // Reasoning block: either readable text or encrypted blob (not both)
        if !self.reasoning.is_empty() {
            blocks.push(ContentBlock::Reasoning {
                content: self.reasoning,
                signature: self.signature.clone(),
                encrypted: None,
                format: None,
            });
        } else if let Some((data, format)) = self.encrypted {
            blocks.push(ContentBlock::Reasoning {
                content: String::new(),
                signature: self.signature.clone(),
                encrypted: Some(data),
                format: Some(format),
            });
        }

        // Text block
        if !self.text.is_empty() {
            blocks.push(ContentBlock::Text { text: self.text });
        }

        // Tool calls, ordered by index
        for (_idx, tc) in self.tool_calls {
            blocks.push(ContentBlock::ToolCall {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            });
        }

        Message {
            role: Role::Assistant,
            content: blocks,
            token_count: self.usage.completion_tokens as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_response() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta("Hello, ".into()));
        acc.push(StreamEvent::TextDelta("world!".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "Hello, world!");
        assert_eq!(msg.token_count, 5);
        assert!(msg.tool_calls().is_empty());
    }

    #[test]
    fn single_tool_call() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "read_file".into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"{"path":"#.into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#""foo.rs"}"#.into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 8,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 1);
        match &calls[0] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
                assert_eq!(arguments, r#"{"path":"foo.rs"}"#);
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn parallel_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ToolCallStart {
            index: 0,
            id: "call_a".into(),
            name: "tool_a".into(),
        });
        acc.push(StreamEvent::ToolCallStart {
            index: 1,
            id: "call_b".into(),
            name: "tool_b".into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"{"x":1}"#.into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 1,
            delta: r#"{"y":2}"#.into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 10,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 2);
        // BTreeMap guarantees index ordering
        match &calls[0] {
            ContentBlock::ToolCall { name, .. } => assert_eq!(name, "tool_a"),
            _ => panic!("expected ToolCall"),
        }
        match &calls[1] {
            ContentBlock::ToolCall { name, .. } => assert_eq!(name, "tool_b"),
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn reasoning_plus_text() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta("Let me think...".into()));
        acc.push(StreamEvent::ReasoningDelta(" done.".into()));
        acc.push(StreamEvent::TextDelta("The answer is 42.".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 12,
                reasoning_tokens: 8,
            },
        });

        let msg = acc.into_message();
        assert_eq!(msg.text(), "The answer is 42.");
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::Reasoning {
                content,
                signature,
                encrypted,
                format,
            } => {
                assert_eq!(content, "Let me think... done.");
                assert!(signature.is_none());
                assert!(encrypted.is_none());
                assert!(format.is_none());
            }
            _ => panic!("expected Reasoning"),
        }
    }

    #[test]
    fn encrypted_reasoning_preservation() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::EncryptedReasoning {
            data: "base64blob==".into(),
            format: "anthropic-claude-v1".into(),
        });
        acc.push(StreamEvent::TextDelta("visible answer".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 3,
                reasoning_tokens: 10,
            },
        });

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::Reasoning {
                content,
                encrypted,
                format,
                ..
            } => {
                assert!(content.is_empty());
                assert_eq!(encrypted.as_deref(), Some("base64blob=="));
                assert_eq!(format.as_deref(), Some("anthropic-claude-v1"));
            }
            _ => panic!("expected Reasoning"),
        }
    }

    #[test]
    fn anthropic_signature_attachment() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta("thinking".into()));
        acc.push(StreamEvent::ReasoningSignature("sig123".into()));
        acc.push(StreamEvent::TextDelta("answer".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 2,
                reasoning_tokens: 5,
            },
        });

        let msg = acc.into_message();
        match &msg.content[0] {
            ContentBlock::Reasoning { signature, .. } => {
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            _ => panic!("expected Reasoning"),
        }
    }

    #[test]
    fn interleaved_text_and_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta("I'll read that file.".into()));
        acc.push(StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "read".into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: "{}".into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 6,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        // Text comes before tool calls regardless of arrival order
        assert_eq!(msg.content.len(), 2);
        assert!(matches!(&msg.content[0], ContentBlock::Text { .. }));
        assert!(matches!(&msg.content[1], ContentBlock::ToolCall { .. }));
    }

    #[test]
    fn empty_response() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 0,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.content.is_empty());
        assert_eq!(msg.token_count, 0);
    }

    #[test]
    fn orphan_argument_delta_is_silently_dropped() {
        let mut acc = StreamAccumulator::new();
        // Argument delta arrives for an index that was never started.
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 99,
            delta: "garbage".into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                reasoning_tokens: 0,
            },
        });

        let msg = acc.into_message();
        assert!(msg.tool_calls().is_empty());
        assert!(msg.content.is_empty());
    }

    #[test]
    fn encrypted_reasoning_with_signature() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::EncryptedReasoning {
            data: "encrypted_blob".into(),
            format: "anthropic-claude-v1".into(),
        });
        acc.push(StreamEvent::ReasoningSignature("sig_abc".into()));
        acc.push(StreamEvent::TextDelta("visible".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 1,
                reasoning_tokens: 10,
            },
        });

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::Reasoning {
                content,
                encrypted,
                signature,
                format,
            } => {
                assert!(content.is_empty());
                assert_eq!(encrypted.as_deref(), Some("encrypted_blob"));
                assert_eq!(format.as_deref(), Some("anthropic-claude-v1"));
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            _ => panic!("expected Reasoning"),
        }
    }

    #[test]
    fn readable_reasoning_takes_priority_over_encrypted() {
        let mut acc = StreamAccumulator::new();
        // Both readable and encrypted arrive (shouldn't happen in practice,
        // but the accumulator should prefer readable).
        acc.push(StreamEvent::ReasoningDelta("readable thinking".into()));
        acc.push(StreamEvent::EncryptedReasoning {
            data: "blob".into(),
            format: "unknown".into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                reasoning_tokens: 5,
            },
        });

        let msg = acc.into_message();
        // Only one reasoning block — the readable one.
        let reasoning_blocks: Vec<_> = msg
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::Reasoning { .. }))
            .collect();
        assert_eq!(reasoning_blocks.len(), 1);
        match &reasoning_blocks[0] {
            ContentBlock::Reasoning {
                content, encrypted, ..
            } => {
                assert_eq!(content, "readable thinking");
                assert!(encrypted.is_none());
            }
            _ => unreachable!(),
        }
    }
}
