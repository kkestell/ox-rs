use serde::{Deserialize, Serialize};

mod accumulator;
mod snapshot;

pub use accumulator::StreamAccumulator;
pub use snapshot::Snapshot;

/// In-flight streaming events from an LLM provider. These represent incremental
/// chunks as they arrive over the wire.
///
/// Serde-derived so the same events can cross a process boundary as NDJSON
/// frames in the GUI ↔ agent protocol. The wire tag matches the `ContentBlock`
/// convention (`#[serde(tag = "type", rename_all = "snake_case")]`) so the two
/// shapes look uniform on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Every variant is a struct variant (even single-field ones) so that the
    /// internal-tag serde convention works uniformly — `#[serde(tag = "type")]`
    /// rejects newtype variants that wrap a primitive, and mixing struct and
    /// newtype variants in one enum is an easy footgun.
    TextDelta {
        delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
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
    ReasoningSignature {
        signature: String,
    },
    Finished {
        usage: Usage,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
}

/// Definition of a tool the model can call.
///
/// The LLM tool schema lives in the domain because provider adapters translate
/// these shapes to provider-specific request bodies and the agent subprocess
/// hands them across the wire. Keeping the definition here lets the app layer
/// construct them without pulling in wire-layer details.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a `StreamEvent` through JSON and assert equality. Every
    /// variant must survive the round-trip without losing fields — the
    /// accumulator depends on `EncryptedReasoning`'s `data`/`format` and
    /// `ReasoningSignature`'s payload being preserved verbatim so that the
    /// at-rest `Message` produced on both sides of the wire matches.
    fn roundtrip(evt: StreamEvent) {
        let json = serde_json::to_string(&evt).expect("serialize");
        let back: StreamEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, evt, "round-trip mismatch for {evt:?} -> {json}");
    }

    #[test]
    fn roundtrip_text_delta() {
        roundtrip(StreamEvent::TextDelta {
            delta: "hello".into(),
        });
    }

    #[test]
    fn roundtrip_reasoning_delta() {
        roundtrip(StreamEvent::ReasoningDelta {
            delta: "thinking...".into(),
        });
    }

    #[test]
    fn roundtrip_encrypted_reasoning() {
        roundtrip(StreamEvent::EncryptedReasoning {
            data: "base64blob==".into(),
            format: "anthropic-claude-v1".into(),
        });
    }

    #[test]
    fn roundtrip_tool_call_start() {
        roundtrip(StreamEvent::ToolCallStart {
            index: 3,
            id: "call_1".into(),
            name: "read_file".into(),
        });
    }

    #[test]
    fn roundtrip_tool_call_argument_delta() {
        roundtrip(StreamEvent::ToolCallArgumentDelta {
            index: 0,
            delta: r#"{"path":"a.rs"}"#.into(),
        });
    }

    #[test]
    fn roundtrip_reasoning_signature() {
        roundtrip(StreamEvent::ReasoningSignature {
            signature: "sig_abc".into(),
        });
    }

    #[test]
    fn serialized_wire_format_uses_type_tag() {
        // Pin down the on-wire shape so an accidental change to the serde
        // attribute (e.g. switching to external tagging) fails loudly.
        let json = serde_json::to_string(&StreamEvent::TextDelta { delta: "hi".into() }).unwrap();
        assert_eq!(json, r#"{"type":"text_delta","delta":"hi"}"#);
    }

    #[test]
    fn roundtrip_finished() {
        roundtrip(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                reasoning_tokens: 3,
            },
        });
    }

    #[test]
    fn usage_roundtrip() {
        let u = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            reasoning_tokens: 5,
        };
        let json = serde_json::to_string(&u).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, u);
    }
}
