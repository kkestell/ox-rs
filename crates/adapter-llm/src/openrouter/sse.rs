use anyhow::{Context, Result};
use domain::{StreamEvent, Usage};
use futures::StreamExt;
use reqwest::Response;

use super::wire::{SseChunk, SseToolCallDelta};

/// Parse an SSE byte stream from a reqwest `Response` and yield `StreamEvent`s
/// through a tokio mpsc channel. Designed to run as a spawned task.
pub async fn parse_sse_stream(
    response: Response,
    tx: tokio::sync::mpsc::Sender<Result<StreamEvent>>,
) {
    if let Err(e) = parse_sse_stream_inner(response, &tx).await {
        // If sending the error fails, the receiver has been dropped — nothing to do.
        let _ = tx.send(Err(e)).await;
    }
}

async fn parse_sse_stream_inner(
    response: Response,
    tx: &tokio::sync::mpsc::Sender<Result<StreamEvent>>,
) -> Result<()> {
    let mut byte_stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("reading SSE byte chunk")?;
        let text = String::from_utf8_lossy(&chunk);
        // Normalize \r\n → \n so the \n\n boundary split works regardless
        // of whether the server uses \r\n line endings.
        buffer.push_str(&text.replace("\r\n", "\n"));

        // SSE events are separated by double newlines.
        while let Some(boundary) = buffer.find("\n\n") {
            let event_block = buffer[..boundary].to_owned();
            buffer = buffer[boundary + 2..].to_owned();

            for line in event_block.lines() {
                let line = line.trim();
                if !line.starts_with("data:") {
                    continue;
                }
                let data = line["data:".len()..].trim();

                if data == "[DONE]" {
                    return Ok(());
                }

                let chunk: SseChunk = serde_json::from_str(data)
                    .with_context(|| format!("parsing SSE JSON: {data}"))?;

                let events = chunk_to_events(&chunk);
                for event in events {
                    tx.send(Ok(event)).await.ok();
                }
            }
        }
    }

    Ok(())
}

/// Map a single parsed SSE chunk into zero or more `StreamEvent`s.
fn chunk_to_events(chunk: &SseChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for choice in &chunk.choices {
        let delta = &choice.delta;

        // Readable reasoning text (redundant with reasoning_details.text for
        // readable models, absent for encrypted-only models).
        if let Some(ref reasoning) = delta.reasoning
            && !reasoning.is_empty()
        {
            events.push(StreamEvent::ReasoningDelta {
                delta: reasoning.clone(),
            });
        }

        // Inspect reasoning_details only for encrypted blobs and signatures.
        if let Some(ref details) = delta.reasoning_details {
            for detail in details {
                if detail.detail_type.as_deref() == Some("reasoning.encrypted")
                    && let (Some(data), Some(format)) = (&detail.data, &detail.format)
                {
                    events.push(StreamEvent::EncryptedReasoning {
                        data: data.clone(),
                        format: format.clone(),
                    });
                }
                if let Some(ref sig) = detail.signature
                    && !sig.is_empty()
                {
                    events.push(StreamEvent::ReasoningSignature {
                        signature: sig.clone(),
                    });
                }
            }
        }

        // Text content delta.
        if let Some(ref content) = delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta {
                delta: content.clone(),
            });
        }

        // Tool call deltas.
        if let Some(ref tool_calls) = delta.tool_calls {
            for tc in tool_calls {
                events.extend(tool_call_delta_to_events(tc));
            }
        }
    }

    // Usage appears on the final chunk.
    if let Some(ref usage) = chunk.usage {
        events.push(StreamEvent::Finished {
            usage: Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                reasoning_tokens: usage.reasoning_tokens,
            },
        });
    }

    events
}

fn tool_call_delta_to_events(tc: &SseToolCallDelta) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    // A delta with id + function.name means a new tool call is starting.
    if let (Some(id), Some(func)) = (&tc.id, &tc.function)
        && let Some(ref name) = func.name
    {
        events.push(StreamEvent::ToolCallStart {
            index: tc.index,
            id: id.clone(),
            name: name.clone(),
        });
    }

    // Argument fragments.
    if let Some(ref func) = tc.function
        && let Some(ref args) = func.arguments
        && !args.is_empty()
    {
        events.push(StreamEvent::ToolCallArgumentDelta {
            index: tc.index,
            delta: args.clone(),
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_from_chunk() {
        let json = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta { delta: t } if t == "hello"));
    }

    #[test]
    fn reasoning_delta_from_chunk() {
        let json = r#"{"choices":[{"delta":{"reasoning":"thinking..."}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ReasoningDelta { delta: r } if r == "thinking...")
        );
    }

    #[test]
    fn encrypted_reasoning_from_details() {
        let json = r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"blob==","format":"anthropic-claude-v1"}]}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::EncryptedReasoning { data, format }
            if data == "blob==" && format == "anthropic-claude-v1")
        );
    }

    #[test]
    fn signature_from_details() {
        let json = r#"{"choices":[{"delta":{"reasoning_details":[{"signature":"sig123"}]}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ReasoningSignature { signature: s } if s == "sig123")
        );
    }

    #[test]
    fn tool_call_start_and_args() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 2); // ToolCallStart + ToolCallArgumentDelta
        assert!(
            matches!(&events[0], StreamEvent::ToolCallStart { index: 0, id, name }
            if id == "call_1" && name == "read_file")
        );
        assert!(
            matches!(&events[1], StreamEvent::ToolCallArgumentDelta { index: 0, delta }
            if delta == r#"{"path":"#)
        );
    }

    #[test]
    fn usage_emits_finished() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"reasoning_tokens":3}}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Finished { usage } => {
                assert_eq!(usage.prompt_tokens, 10);
                assert_eq!(usage.completion_tokens, 5);
                assert_eq!(usage.reasoning_tokens, 3);
            }
            _ => panic!("expected Finished"),
        }
    }

    #[test]
    fn empty_fields_produce_no_events() {
        let json = r#"{"choices":[{"delta":{"content":"","reasoning":""}}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let events = chunk_to_events(&chunk);
        assert!(events.is_empty());
    }
}
