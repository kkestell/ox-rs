//! Wire protocol between the GUI process (`ox-tauri`) and the `ox-agent` process.
//!
//! The two processes exchange newline-delimited JSON frames over stdin/stdout.
//! The GUI sends `AgentCommand`s into the agent's stdin; the agent sends
//! `AgentEvent`s back over its stdout. One JSON value per line — no length
//! prefix, no multi-line frames, so either side can parse with a line buffer
//! and a single `serde_json::from_str`.
//!
//! ### Why this crate is tiny
//!
//! The protocol crate exists *only* to define the wire enums and their framing
//! helpers. It deliberately has no request/response helpers, no state machine,
//! no retry logic — those are consumer concerns and live in the GUI's
//! `agent_client` and the agent's driver loop. A fat protocol crate creates
//! temptation to couple the two processes' state models together, which would
//! defeat the reason we split them in the first place.

use std::path::PathBuf;

use anyhow::{Context, Result};
use domain::{CloseIntent, Message, SessionId, StreamEvent};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// Commands the GUI sends to the agent.
///
/// Kept extensible (`#[non_exhaustive]`) so a future `Cancel` or
/// `ApproveToolCall` can be added without a protocol bump — the agent should
/// treat any unknown command as a malformed frame and continue.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentCommand {
    /// User submitted a chat message. The agent appends it to the session
    /// and drives a turn.
    SendMessage {
        input: String,
        model: String,
    },
    /// Request cancellation of the in-progress turn. The agent sets a
    /// cooperative cancel flag; actual cancellation happens at the next
    /// check point (between stream events, before tool calls). Idempotent
    /// — sending multiple `Cancel` commands is harmless.
    Cancel,
    ResolveToolApproval {
        request_id: String,
        approved: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolApprovalRequest {
    pub request_id: String,
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub reason: String,
}

/// Events the agent streams back to the GUI.
///
/// The protocol separates three concerns:
/// 1. `Ready` — a one-shot handshake that confirms the agent started and
///    published the session ID the GUI will display in its exit message.
/// 2. Per-turn events — `StreamDelta` for in-flight tokens/reasoning/tool-
///    call argument chunks, `MessageAppended` for every committed message
///    (user input, assistant replies, tool results).
/// 3. Turn terminators — `TurnComplete` on success, `Error` on failure.
///
/// On resume, `MessageAppended` also carries historical messages the agent
/// replays after `Ready` so the GUI can render the existing conversation
/// through the *same* code path it uses for live messages. Uniform handling
/// is load-bearing — two separate "populate history" and "append message"
/// code paths drift apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentEvent {
    /// Sent once at startup. No history is bundled here — the agent replays
    /// history as a sequence of `MessageAppended` frames after `Ready` so the
    /// GUI doesn't need a second "initial state" code path.
    Ready {
        session_id: SessionId,
        workspace_root: PathBuf,
    },
    /// An incremental stream event from the LLM.
    StreamDelta {
        event: StreamEvent,
    },
    /// A message was just committed to the session. Used both for live turn
    /// messages and for historical replay on resume.
    MessageAppended {
        message: Message,
    },
    ToolApprovalRequested {
        requests: Vec<ToolApprovalRequest>,
    },
    ToolApprovalResolved {
        request_id: String,
        approved: bool,
    },
    /// The current turn ended successfully. The GUI may re-enable input.
    TurnComplete,
    /// The current turn was cancelled at the user's request. Partial
    /// content (if any) has already been committed via `MessageAppended`
    /// before this event. The GUI should re-enable input and show a
    /// cancellation indicator.
    TurnCancelled,
    /// Fatal error for the current turn (or, pre-`Ready`, for startup).
    /// No `TurnComplete` follows.
    Error {
        message: String,
    },
    /// The agent has executed a lifecycle tool (`merge` or `abandon`) and
    /// is now requesting that the host close the session. Emitted by the
    /// driver immediately after the turn's terminal frame; the agent exits
    /// its command loop right after, so the host sees EOF on stdout next.
    /// Not user-visible — the host's pump routes this to a
    /// `CloseRequestSink`, not to the session's broadcast channel.
    RequestClose {
        intent: CloseIntent,
    },
}

// ---------------------------------------------------------------------------
// Framing helpers
// ---------------------------------------------------------------------------

/// Serialize `frame` as JSON and write it followed by a newline.
///
/// Flushes after the newline so readers on the other end of a pipe don't
/// block on a partial line. Both sides call this for every outgoing frame so
/// a write never interleaves with another in the pipe.
pub async fn write_frame<W, T>(writer: &mut W, frame: &T) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let mut line = serde_json::to_vec(frame).context("serializing protocol frame")?;
    line.push(b'\n');
    writer
        .write_all(&line)
        .await
        .context("writing protocol frame")?;
    writer.flush().await.context("flushing protocol frame")?;
    Ok(())
}

/// Read a single newline-delimited JSON frame from `reader`.
///
/// Returns:
/// - `Ok(Some(T))` on a parsed frame.
/// - `Ok(None)` on a clean EOF before any bytes arrive — the far side closed
///   its writer, which the consumer should treat as "graceful shutdown", not
///   an error.
/// - `Err(_)` on I/O failure or on a line that fails JSON parsing. Callers
///   that want to tolerate malformed frames (`AgentClient` does, to keep
///   one bad line from killing a tab) can catch the error and keep reading.
pub async fn read_frame<R, T>(reader: &mut R) -> Result<Option<T>>
where
    R: AsyncBufReadExt + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .context("reading protocol frame")?;
    if n == 0 {
        return Ok(None); // clean EOF
    }
    // `read_line` leaves the newline in the buffer; strip it so error messages
    // show only the offending JSON, not a trailing \n.
    let trimmed = line.trim_end_matches(['\n', '\r']);
    let frame: T = serde_json::from_str(trimmed)
        .with_context(|| format!("parsing protocol frame: {trimmed}"))?;
    Ok(Some(frame))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use domain::{ContentBlock, Message, Role, Usage};
    use tokio::io::{AsyncWriteExt, BufReader};

    use super::*;

    /// Round-trip `T` through JSON and compare the JSON representation before
    /// and after. We compare the re-serialized JSON because neither enum
    /// derives `PartialEq` (nested types don't either, and adding it would
    /// cost more than it's worth). If the shape is stable the two strings
    /// are identical.
    fn roundtrip_json<T>(value: T) -> String
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let json = serde_json::to_string(&value).unwrap();
        let back: T = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
        json
    }

    // -- serde round-trip ----------------------------------------------------

    #[test]
    fn roundtrip_send_message_command() {
        let json = roundtrip_json(AgentCommand::SendMessage {
            input: "hello".into(),
            model: "test/model".into(),
        });
        assert_eq!(
            json,
            r#"{"type":"send_message","input":"hello","model":"test/model"}"#
        );
    }

    #[test]
    fn roundtrip_resolve_tool_approval_command() {
        let json = roundtrip_json(AgentCommand::ResolveToolApproval {
            request_id: "call_1".into(),
            approved: true,
        });
        assert_eq!(
            json,
            r#"{"type":"resolve_tool_approval","request_id":"call_1","approved":true}"#
        );
    }

    #[test]
    fn roundtrip_ready_event() {
        roundtrip_json(AgentEvent::Ready {
            session_id: SessionId::new_v4(),
            workspace_root: PathBuf::from("/home/user/project"),
        });
    }

    #[test]
    fn roundtrip_stream_delta_text() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::TextDelta {
                delta: "hello".into(),
            },
        });
    }

    #[test]
    fn roundtrip_tool_approval_requested() {
        let json = roundtrip_json(AgentEvent::ToolApprovalRequested {
            requests: vec![ToolApprovalRequest {
                request_id: "call_1".into(),
                tool_call_id: "call_1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"cargo test"}"#.into(),
                reason: "Tool requires user approval".into(),
            }],
        });
        assert!(json.contains(r#""type":"tool_approval_requested""#));
    }

    #[test]
    fn roundtrip_tool_approval_resolved() {
        let json = roundtrip_json(AgentEvent::ToolApprovalResolved {
            request_id: "call_1".into(),
            approved: false,
        });
        assert_eq!(
            json,
            r#"{"type":"tool_approval_resolved","request_id":"call_1","approved":false}"#
        );
    }

    #[test]
    fn roundtrip_stream_delta_reasoning() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::ReasoningDelta {
                delta: "thinking".into(),
            },
        });
    }

    #[test]
    fn roundtrip_stream_delta_encrypted_reasoning() {
        // Encrypted reasoning carries an opaque base64 blob that the provider
        // re-verifies against its own signature. Both `data` and `format`
        // must round-trip intact — the accumulator's block-order contract
        // depends on seeing these verbatim.
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::EncryptedReasoning {
                data: "base64blob==".into(),
                format: "anthropic-claude-v1".into(),
            },
        });
    }

    #[test]
    fn roundtrip_stream_delta_tool_call_start() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::ToolCallStart {
                index: 2,
                id: "call_1".into(),
                name: "read_file".into(),
            },
        });
    }

    #[test]
    fn roundtrip_stream_delta_tool_call_arg_delta() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: r#"{"path":"a.rs"}"#.into(),
            },
        });
    }

    #[test]
    fn roundtrip_stream_delta_reasoning_signature() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::ReasoningSignature {
                signature: "sig_abc".into(),
            },
        });
    }

    #[test]
    fn roundtrip_stream_delta_finished() {
        roundtrip_json(AgentEvent::StreamDelta {
            event: StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    reasoning_tokens: 2,
                },
            },
        });
    }

    #[test]
    fn roundtrip_message_appended_user() {
        roundtrip_json(AgentEvent::MessageAppended {
            message: Message::user("hi"),
        });
    }

    #[test]
    fn roundtrip_message_appended_assistant_with_blocks() {
        roundtrip_json(AgentEvent::MessageAppended {
            message: Message::assistant(vec![
                ContentBlock::Reasoning {
                    content: "think".into(),
                    signature: Some("sig".into()),
                    encrypted: None,
                    format: None,
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
                ContentBlock::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ]),
        });
    }

    #[test]
    fn roundtrip_message_appended_tool_result() {
        let msg = Message::tool_result("call_1", "output", false);
        assert_eq!(msg.role, Role::Tool);
        roundtrip_json(AgentEvent::MessageAppended { message: msg });
    }

    #[test]
    fn roundtrip_cancel_command() {
        let json = roundtrip_json(AgentCommand::Cancel);
        assert_eq!(json, r#"{"type":"cancel"}"#);
    }

    #[test]
    fn roundtrip_turn_cancelled() {
        let json = roundtrip_json(AgentEvent::TurnCancelled);
        assert_eq!(json, r#"{"type":"turn_cancelled"}"#);
    }

    #[test]
    fn roundtrip_turn_complete() {
        let json = roundtrip_json(AgentEvent::TurnComplete);
        assert_eq!(json, r#"{"type":"turn_complete"}"#);
    }

    #[test]
    fn roundtrip_error() {
        let json = roundtrip_json(AgentEvent::Error {
            message: "stream interrupted".into(),
        });
        assert_eq!(json, r#"{"type":"error","message":"stream interrupted"}"#);
    }

    #[test]
    fn roundtrip_request_close_merge() {
        let json = roundtrip_json(AgentEvent::RequestClose {
            intent: CloseIntent::Merge,
        });
        assert_eq!(
            json,
            r#"{"type":"request_close","intent":{"kind":"merge"}}"#
        );
    }

    #[test]
    fn roundtrip_request_close_abandon_with_confirm() {
        let json = roundtrip_json(AgentEvent::RequestClose {
            intent: CloseIntent::Abandon { confirm: true },
        });
        assert_eq!(
            json,
            r#"{"type":"request_close","intent":{"kind":"abandon","confirm":true}}"#
        );
    }

    // -- write_frame / read_frame round-trip --------------------------------

    #[tokio::test]
    async fn write_then_read_single_frame() {
        let (mut client, server) = tokio::io::duplex(4096);
        write_frame(
            &mut client,
            &AgentCommand::SendMessage {
                input: "hi".into(),
                model: "test/model".into(),
            },
        )
        .await
        .unwrap();
        drop(client);

        let mut reader = BufReader::new(server);
        let parsed: Option<AgentCommand> = read_frame(&mut reader).await.unwrap();
        let Some(AgentCommand::SendMessage { input, model }) = parsed else {
            panic!("expected SendMessage");
        };
        assert_eq!(input, "hi");
        assert_eq!(model, "test/model");
    }

    #[tokio::test]
    async fn write_frame_terminates_with_newline() {
        // Readers on the other side of a pipe block on partial lines, so we
        // need a hard guarantee that every frame ends with \n.
        let (mut client, mut server) = tokio::io::duplex(4096);
        write_frame(
            &mut client,
            &AgentEvent::Error {
                message: "boom".into(),
            },
        )
        .await
        .unwrap();
        drop(client);
        let mut raw = Vec::new();
        use tokio::io::AsyncReadExt;
        server.read_to_end(&mut raw).await.unwrap();
        assert!(raw.ends_with(b"\n"), "frame must end with newline");
    }

    #[tokio::test]
    async fn read_frame_returns_none_on_clean_eof() {
        let (client, server) = tokio::io::duplex(64);
        drop(client); // writer gone — reader sees EOF immediately.
        let mut reader = BufReader::new(server);
        let frame: Option<AgentCommand> = read_frame(&mut reader).await.unwrap();
        assert!(frame.is_none());
    }

    #[tokio::test]
    async fn read_frame_errors_on_malformed_json() {
        let (mut client, server) = tokio::io::duplex(64);
        client.write_all(b"this is not json\n").await.unwrap();
        drop(client);
        let mut reader = BufReader::new(server);
        let result: Result<Option<AgentCommand>> = read_frame(&mut reader).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_frame_preserves_frame_order_across_multiple_lines() {
        let (mut client, server) = tokio::io::duplex(1024);
        write_frame(
            &mut client,
            &AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "first".into(),
                },
            },
        )
        .await
        .unwrap();
        write_frame(&mut client, &AgentEvent::TurnComplete)
            .await
            .unwrap();
        drop(client);

        let mut reader = BufReader::new(server);
        let f1: Option<AgentEvent> = read_frame(&mut reader).await.unwrap();
        let f2: Option<AgentEvent> = read_frame(&mut reader).await.unwrap();
        let f3: Option<AgentEvent> = read_frame(&mut reader).await.unwrap();

        assert!(matches!(
            f1,
            Some(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta { delta }
            }) if delta == "first"
        ));
        assert!(matches!(f2, Some(AgentEvent::TurnComplete)));
        assert!(f3.is_none()); // EOF after both frames.
    }

    #[tokio::test]
    async fn write_and_read_handles_multi_megabyte_payload() {
        // A 2 MiB text block must round-trip without wedging the reader.
        // `BufReader::read_line` grows its internal buffer as needed; the
        // risk here is a forgotten flush or an embedded newline in the JSON.
        // `serde_json` escapes `\n` in strings, so this is a real round-trip
        // check that our framing invariants hold under load.
        let huge = "x".repeat(2 * 1024 * 1024);
        let (mut client, server) = tokio::io::duplex(8 * 1024);

        let writer = tokio::spawn(async move {
            write_frame(
                &mut client,
                &AgentEvent::MessageAppended {
                    message: Message::user(&huge),
                },
            )
            .await
            .unwrap();
            drop(client);
        });

        let mut reader = BufReader::new(server);
        let frame: Option<AgentEvent> = read_frame(&mut reader).await.unwrap();
        writer.await.unwrap();

        let Some(AgentEvent::MessageAppended { message }) = frame else {
            panic!("expected MessageAppended");
        };
        assert_eq!(message.text().len(), 2 * 1024 * 1024);
    }
}
