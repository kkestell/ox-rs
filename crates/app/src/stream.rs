use std::cell::OnceCell;
use std::collections::BTreeMap;

use domain::{ContentBlock, Message, Role, StreamEvent, Usage};

/// Definition of a tool the model can call.
///
/// Stays in the app layer rather than in `domain` because the LLM tool schema
/// is an *application* concern — `ToolRegistry` produces these from `Tool`
/// implementations, and provider adapters translate them to provider-specific
/// request shapes. Nothing persisted or serialized cares about `ToolDef`.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Non-consuming, borrowed view of a `StreamAccumulator`'s current state.
///
/// Role is always `Role::Assistant`. The content slice borrows the
/// accumulator's internal cache and is only rebuilt when new events arrive.
/// This avoids per-frame cloning during live streaming in the GUI.
pub struct Snapshot<'a> {
    pub content: &'a [ContentBlock],
    pub token_count: usize,
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
///
/// Not `Sync` — must remain owned or behind `&mut`. The `OnceCell` cache
/// requires `&self` to populate and `&mut self` to invalidate, which is
/// compatible with the current ownership model (moved between threads, never
/// shared via `&` across threads).
pub struct StreamAccumulator {
    text: String,
    reasoning: String,
    signature: Option<String>,
    encrypted: Option<(String, String)>, // (data, format)
    tool_calls: BTreeMap<usize, ToolCallAccum>,
    usage: Usage,
    /// Lazily built block cache, invalidated on every `push`. Avoids
    /// rebuilding the block vec on every `snapshot()` call within a single
    /// frame when the GUI calls it multiple times between events.
    cached_blocks: OnceCell<Vec<ContentBlock>>,
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            signature: None,
            encrypted: None,
            tool_calls: BTreeMap::new(),
            usage: Usage::default(),
            cached_blocks: OnceCell::new(),
        }
    }
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: StreamEvent) {
        // Any new event invalidates the cached block vec so the next
        // snapshot() rebuilds it with fresh data.
        self.cached_blocks.take();

        match event {
            StreamEvent::TextDelta { delta } => self.text.push_str(&delta),
            StreamEvent::ReasoningDelta { delta } => self.reasoning.push_str(&delta),
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
            StreamEvent::ReasoningSignature { signature } => {
                self.signature = Some(signature);
            }
            StreamEvent::Finished { usage } => {
                self.usage = usage;
            }
        }
    }

    pub fn into_message(mut self) -> Message {
        let blocks = self
            .cached_blocks
            .take()
            .unwrap_or_else(|| self.assemble_blocks());
        Message {
            role: Role::Assistant,
            content: blocks,
            token_count: self.usage.completion_tokens as usize,
        }
    }

    /// Non-consuming, borrowed view of the accumulator's current state.
    ///
    /// Uses a lazily built cache: the first call after a `push` rebuilds the
    /// block vec; subsequent calls return the same slice until the next
    /// `push` invalidates the cache. This is the hot path for live GUI
    /// rendering — called once per egui frame while the model is streaming.
    pub fn snapshot(&self) -> Snapshot<'_> {
        let content = self.cached_blocks.get_or_init(|| self.assemble_blocks());
        Snapshot {
            content,
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
        acc.push(StreamEvent::TextDelta {
            delta: "Hello, ".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "world!".into(),
        });
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
        acc.push(StreamEvent::ReasoningDelta {
            delta: "Let me think...".into(),
        });
        acc.push(StreamEvent::ReasoningDelta {
            delta: " done.".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "The answer is 42.".into(),
        });
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
        acc.push(StreamEvent::TextDelta {
            delta: "visible answer".into(),
        });
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
        acc.push(StreamEvent::ReasoningDelta {
            delta: "thinking".into(),
        });
        acc.push(StreamEvent::ReasoningSignature {
            signature: "sig123".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "answer".into(),
        });
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
        acc.push(StreamEvent::TextDelta {
            delta: "I'll read that file.".into(),
        });
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
        acc.push(StreamEvent::ReasoningSignature {
            signature: "sig_abc".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "visible".into(),
        });
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
        acc.push(StreamEvent::ReasoningDelta {
            delta: "readable thinking".into(),
        });
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

    /// Helper: a snapshot must agree with `into_message` on content blocks and
    /// token count. The snapshot borrows the accumulator, so we take the
    /// snapshot first, copy its fields, drop the borrow, then consume.
    fn assert_snapshot_matches_message(acc: StreamAccumulator) {
        let snap_content = acc.snapshot().content.to_vec();
        let snap_token_count = acc.snapshot().token_count;
        let msg = acc.into_message();
        assert_eq!(snap_content, msg.content);
        assert_eq!(snap_token_count, msg.token_count);
    }

    #[test]
    fn snapshot_matches_into_message_for_text_only() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta {
            delta: "Hello, ".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "world!".into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                reasoning_tokens: 0,
            },
        });

        assert_snapshot_matches_message(acc);
    }

    #[test]
    fn snapshot_matches_into_message_for_reasoning_plus_text() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta {
            delta: "think ".into(),
        });
        acc.push(StreamEvent::ReasoningDelta {
            delta: "harder".into(),
        });
        acc.push(StreamEvent::ReasoningSignature {
            signature: "sig".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "answer".into(),
        });
        acc.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 3,
                reasoning_tokens: 5,
            },
        });

        assert_snapshot_matches_message(acc);
    }

    #[test]
    fn snapshot_matches_into_message_for_tool_calls_including_parallel() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta {
            delta: "dispatching...".into(),
        });
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

        // Independent hand-rolled assertions so that a bug in the shared
        // `assemble_blocks` helper (wrong ordering, dropped blocks, wrong
        // token count) would be visible here rather than hidden behind
        // parity agreement between `snapshot` and `into_message`.
        {
            let snap = acc.snapshot();
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

        // Now verify parity with into_message.
        assert_snapshot_matches_message(acc);
    }

    #[test]
    fn snapshot_mid_stream_reflects_partial_state() {
        // A snapshot taken before `Finished` must still reflect every event
        // that has arrived, including a tool call whose arguments haven't
        // finished streaming. This is the live-rendering path.
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::ReasoningDelta {
            delta: "partial reasoning".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "partial ".into(),
        });
        acc.push(StreamEvent::ToolCallStart {
            index: 0,
            id: "call_x".into(),
            name: "search".into(),
        });
        acc.push(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"{"query":"hel"#.into(),
        });

        {
            let snap = acc.snapshot();
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
        }

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
        acc.push(StreamEvent::ReasoningSignature {
            signature: "sig-1".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "final answer".into(),
        });

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
        acc.push(StreamEvent::ReasoningDelta {
            delta: "readable".into(),
        });
        acc.push(StreamEvent::TextDelta {
            delta: "done".into(),
        });

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

    #[test]
    fn snapshot_caches_blocks_until_next_push() {
        let mut acc = StreamAccumulator::new();
        acc.push(StreamEvent::TextDelta {
            delta: "hello".into(),
        });

        // Two consecutive snapshots without an intervening push should
        // return the same cached block vec (pointer-equal slices).
        {
            let snap1 = acc.snapshot();
            let snap2 = acc.snapshot();
            assert!(
                std::ptr::eq(snap1.content, snap2.content),
                "consecutive snapshots must share the same cached slice"
            );
        }

        // A push invalidates the cache — the next snapshot rebuilds.
        acc.push(StreamEvent::TextDelta {
            delta: " world".into(),
        });
        {
            let snap3 = acc.snapshot();
            assert_eq!(snap3.content.len(), 1);
            match &snap3.content[0] {
                ContentBlock::Text { text } => assert_eq!(text, "hello world"),
                _ => panic!("expected Text"),
            }
            // The rebuilt cache is itself stable for subsequent calls.
            let snap4 = acc.snapshot();
            assert!(
                std::ptr::eq(snap3.content, snap4.content),
                "post-push snapshots must share the new cached slice"
            );
        }
    }
}
