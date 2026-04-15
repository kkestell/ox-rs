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
        Message {
            role: Role::Assistant,
            content: self.assemble_blocks(),
            token_count: self.usage.completion_tokens as usize,
        }
    }

    /// Non-consuming snapshot of the accumulator as a `Message`. Enables live
    /// rendering of in-flight state (e.g. in a GUI) using the same ordering
    /// and block-shape guarantees as `into_message`, so there is no risk of
    /// drift between the in-progress view and the final stored message.
    pub fn snapshot(&self) -> Message {
        Message {
            role: Role::Assistant,
            content: self.assemble_blocks(),
            token_count: self.usage.completion_tokens as usize,
        }
    }

    /// Shared block-assembly logic. Kept on `&self` with explicit clones so
    /// it can back both the consuming `into_message` and the non-consuming
    /// `snapshot` without duplicating ordering rules.
    fn assemble_blocks(&self) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        // Reasoning block: either readable text or encrypted blob (not both).
        // Readable wins if both arrived — the user-visible trace beats the
        // opaque blob that only the provider cares about on re-send.
        if !self.reasoning.is_empty() {
            blocks.push(ContentBlock::Reasoning {
                content: self.reasoning.clone(),
                signature: self.signature.clone(),
                encrypted: None,
                format: None,
            });
        } else if let Some((data, format)) = &self.encrypted {
            blocks.push(ContentBlock::Reasoning {
                content: String::new(),
                signature: self.signature.clone(),
                encrypted: Some(data.clone()),
                format: Some(format.clone()),
            });
        }

        // Text block
        if !self.text.is_empty() {
            blocks.push(ContentBlock::Text {
                text: self.text.clone(),
            });
        }

        // Tool calls, ordered by index (BTreeMap guarantees stable ordering).
        for tc in self.tool_calls.values() {
            blocks.push(ContentBlock::ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            });
        }

        blocks
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

    // -- snapshot: non-consuming equivalent of into_message --
    //
    // These tests pin down the contract that drives live GUI rendering:
    // a snapshot of the accumulator must produce exactly what `into_message`
    // would produce from the same events, at any point in the stream. Any
    // drift between the two would cause the final message to "flicker" when
    // the accumulator is consumed and its result replaces the live view.

    /// Helper: the two messages must agree on role, content blocks, and
    /// token count. `Message` doesn't derive `PartialEq`, so we compare
    /// field-by-field.
    fn assert_messages_equal(a: &Message, b: &Message) {
        assert_eq!(a.role, b.role);
        assert_eq!(a.content, b.content);
        assert_eq!(a.token_count, b.token_count);
    }

    #[test]
    fn snapshot_matches_into_message_for_text_only() {
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

        let snap = acc.snapshot();
        let consumed = acc.into_message();
        assert_messages_equal(&snap, &consumed);
    }

    #[test]
    fn snapshot_matches_into_message_for_reasoning_plus_text() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta("think ".into()));
        acc.push(StreamEvent::ReasoningDelta("harder".into()));
        acc.push(StreamEvent::ReasoningSignature("sig".into()));
        acc.push(StreamEvent::TextDelta("answer".into()));
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 3,
                reasoning_tokens: 5,
            },
        });

        let snap = acc.snapshot();
        let consumed = acc.into_message();
        assert_messages_equal(&snap, &consumed);
    }

    #[test]
    fn snapshot_matches_into_message_for_tool_calls_including_parallel() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta("dispatching...".into()));
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

        let snap = acc.snapshot();
        let consumed = acc.into_message();
        assert_messages_equal(&snap, &consumed);

        // Independent hand-rolled assertions so that a bug in the shared
        // `assemble_blocks` helper (wrong ordering, dropped blocks, wrong
        // token count) would be visible here rather than hidden behind
        // parity agreement between `snapshot` and `into_message`.
        assert_eq!(snap.token_count, 10);
        assert_eq!(snap.content.len(), 3);
        match &snap.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "dispatching..."),
            _ => panic!("expected Text first"),
        }
        match &snap.content[1] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_a");
                assert_eq!(name, "tool_a");
                assert_eq!(arguments, r#"{"x":1}"#);
            }
            _ => panic!("expected ToolCall at index 1"),
        }
        match &snap.content[2] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_b");
                assert_eq!(name, "tool_b");
                assert_eq!(arguments, r#"{"y":2}"#);
            }
            _ => panic!("expected ToolCall at index 2"),
        }
    }

    #[test]
    fn snapshot_mid_stream_reflects_partial_state() {
        // A snapshot taken before `Finished` must still reflect every event
        // that has arrived, including a tool call whose arguments haven't
        // finished streaming. This is the live-rendering path.
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta("partial reasoning".into()));
        acc.push(StreamEvent::TextDelta("partial ".into()));
        acc.push(StreamEvent::ToolCallStart {
            index: 0,
            id: "call_x".into(),
            name: "search".into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"{"query":"hel"#.into(),
        });

        let snap = acc.snapshot();
        assert_eq!(snap.role, Role::Assistant);
        // Blocks in canonical order: reasoning, text, tool call.
        assert_eq!(snap.content.len(), 3);
        match &snap.content[0] {
            ContentBlock::Reasoning { content, .. } => {
                assert_eq!(content, "partial reasoning");
            }
            _ => panic!("expected Reasoning first"),
        }
        match &snap.content[1] {
            ContentBlock::Text { text } => assert_eq!(text, "partial "),
            _ => panic!("expected Text second"),
        }
        match &snap.content[2] {
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                assert_eq!(name, "search");
                assert_eq!(arguments, r#"{"query":"hel"#);
            }
            _ => panic!("expected ToolCall third"),
        }
        // No Finished event yet — token count should still be zero.
        assert_eq!(snap.token_count, 0);

        // Continue streaming and snapshot again — arguments should grow.
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"lo"}"#.into(),
        });
        let snap2 = acc.snapshot();
        match &snap2.content[2] {
            ContentBlock::ToolCall { arguments, .. } => {
                assert_eq!(arguments, r#"{"query":"hello"}"#);
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn snapshot_of_empty_accumulator_returns_assistant_with_no_blocks() {
        let acc = StreamAccumulator::new();
        let snap = acc.snapshot();
        assert_eq!(snap.role, Role::Assistant);
        assert!(snap.content.is_empty());
        assert_eq!(snap.token_count, 0);
    }

    #[test]
    fn snapshot_preserves_encrypted_only_reasoning() {
        // When only an encrypted blob arrives, `snapshot` must still surface
        // the `Reasoning` block with empty `content` and the blob intact —
        // the GUI renderer uses the empty `content` to decide whether to
        // show anything, but the block must still be persisted so the
        // provider can re-verify on the next turn.
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::EncryptedReasoning {
            data: "opaque-blob".into(),
            format: "provider-x".into(),
        });
        acc.push(StreamEvent::ReasoningSignature("sig-1".into()));
        acc.push(StreamEvent::TextDelta("final answer".into()));

        let snap = acc.snapshot();
        assert_eq!(snap.content.len(), 2);
        match &snap.content[0] {
            ContentBlock::Reasoning {
                content,
                encrypted,
                format,
                signature,
            } => {
                assert!(content.is_empty());
                assert_eq!(encrypted.as_deref(), Some("opaque-blob"));
                assert_eq!(format.as_deref(), Some("provider-x"));
                assert_eq!(signature.as_deref(), Some("sig-1"));
            }
            _ => panic!("expected Reasoning first"),
        }
    }

    #[test]
    fn snapshot_prefers_readable_over_encrypted_reasoning() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::EncryptedReasoning {
            data: "opaque".into(),
            format: "provider-x".into(),
        });
        acc.push(StreamEvent::ReasoningDelta("readable".into()));
        acc.push(StreamEvent::TextDelta("done".into()));

        let snap = acc.snapshot();
        let reasoning: Vec<_> = snap
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::Reasoning { .. }))
            .collect();
        assert_eq!(reasoning.len(), 1);
        match reasoning[0] {
            ContentBlock::Reasoning {
                content, encrypted, ..
            } => {
                assert_eq!(content, "readable");
                assert!(encrypted.is_none());
            }
            _ => unreachable!(),
        }
    }
}
