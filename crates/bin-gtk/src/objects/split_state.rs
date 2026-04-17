//! Receive- and send-side state machine for one agent split.
//!
//! `SplitState` is the plain in-memory state every split tracks; `apply_event`
//! mirrors the deleted `agent_host::AgentSplit::handle_event` body verbatim,
//! and `begin_send` is the send-side guard previously inlined in
//! `WorkspaceState::send`. Both are kept free-standing so the state machine
//! tests run with no GTK / glib in scope — `SplitObject` just forwards to
//! these functions and reconciles its own UI properties afterward.

use agent_host::StreamAccumulator;
use domain::{Message, SessionId};
use protocol::AgentEvent;

#[derive(Default)]
pub struct SplitState {
    pub messages: Vec<Message>,
    pub streaming: Option<StreamAccumulator>,
    pub waiting: bool,
    pub error: Option<String>,
    pub session_id: Option<SessionId>,
    pub cancelled: bool,
}

/// Outcome of [`begin_send`]. The caller flips properties only when `Send`
/// is returned — `Skip` is the silent double-Enter guard from the previous
/// `WorkspaceState::send`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShouldSend {
    Send,
    Skip,
}

impl SplitState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether quit-confirmation should ask the user before tearing the
    /// agent down. True if a turn is in flight or already streaming.
    pub fn is_turn_in_progress(&self) -> bool {
        self.waiting || self.streaming.is_some()
    }
}

/// Receive-side state transition. Pure; no GTK, no IPC, no I/O.
pub fn apply_event(state: &mut SplitState, event: AgentEvent) {
    match event {
        AgentEvent::Ready { session_id, .. } => {
            // Record the ID; do NOT touch messages/waiting/streaming. Replayed
            // history arrives as a sequence of MessageAppended frames below.
            state.session_id = Some(session_id);
        }
        AgentEvent::StreamDelta { event } => {
            state
                .streaming
                .get_or_insert_with(StreamAccumulator::new)
                .push(event);
        }
        AgentEvent::MessageAppended { message } => {
            // Single code path for both live commits and historical replay.
            // Drop the accumulator so a subsequent renderer doesn't show the
            // streaming view atop the just-committed message.
            state.messages.push(message);
            state.streaming = None;
            state.error = None;
        }
        AgentEvent::TurnComplete => {
            state.waiting = false;
            state.streaming = None;
            state.error = None;
        }
        AgentEvent::TurnCancelled => {
            state.waiting = false;
            // Promote any partial streaming content into a committed message
            // so a cancelled turn still shows what the model produced before
            // the cancel arrived.
            if let Some(acc) = state.streaming.take() {
                let msg = acc.into_message();
                if !msg.content.is_empty() {
                    state.messages.push(msg);
                }
            }
            state.cancelled = true;
            state.error = None;
        }
        AgentEvent::Error { message } => {
            state.streaming = None;
            state.error = Some(message);
            state.waiting = false;
        }
        // AgentEvent is `#[non_exhaustive]`; future variants must be a no-op
        // rather than crash the GUI.
        _ => {}
    }
}

/// Send-side state flip. Returns `Skip` when a turn is already in flight
/// (double-Enter guard); otherwise sets `waiting`, clears `error`, clears
/// `cancelled`, and returns `Send`. The IPC dispatch happens in the GObject
/// wrapper after this returns `Send`.
pub fn begin_send(state: &mut SplitState) -> ShouldSend {
    if state.waiting {
        return ShouldSend::Skip;
    }
    state.waiting = true;
    state.error = None;
    state.cancelled = false;
    ShouldSend::Send
}

#[cfg(test)]
mod tests {
    //! State-machine tests, ported from the deleted
    //! `agent_host::split::tests` plus a pair of new `begin_send` cases.
    //! No GTK; runs with `cargo test -p bin-gtk` headless.

    use std::path::PathBuf;

    use domain::{ContentBlock, Message, Role, StreamEvent, Usage};

    use super::*;

    #[test]
    fn ready_records_session_id_and_leaves_other_fields_untouched() {
        let mut state = SplitState::new();
        let id = SessionId::new_v4();
        apply_event(
            &mut state,
            AgentEvent::Ready {
                session_id: id,
                workspace_root: PathBuf::from("/w"),
            },
        );
        assert_eq!(state.session_id, Some(id));
        assert!(state.messages.is_empty(), "Ready must not touch messages");
        assert!(!state.waiting, "Ready must not flip waiting");
        assert!(state.streaming.is_none(), "Ready must not touch streaming");
        assert!(state.error.is_none(), "Ready must not set an error");
    }

    #[test]
    fn message_appended_extends_history_without_touching_streaming() {
        let mut state = SplitState::new();
        apply_event(
            &mut state,
            AgentEvent::MessageAppended {
                message: Message::user("hi"),
            },
        );
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].text(), "hi");
        assert!(!state.waiting, "history replay should not set waiting");
        assert!(state.streaming.is_none());
    }

    #[test]
    fn message_appended_drops_any_inflight_stream_accumulator() {
        // Realistic ordering: StreamDelta arrives, then the agent commits
        // the final message via MessageAppended. The accumulator must be
        // dropped so the renderer does not render the same tokens twice.
        let mut state = SplitState::new();
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            },
        );
        assert!(state.streaming.is_some());
        apply_event(
            &mut state,
            AgentEvent::MessageAppended {
                message: Message::user("final"),
            },
        );
        assert!(
            state.streaming.is_none(),
            "streaming mirror must be dropped on commit"
        );
    }

    #[test]
    fn turn_complete_clears_waiting_and_streaming_and_error() {
        let mut state = SplitState::new();
        state.waiting = true;
        state.error = Some("old".into());
        state.streaming = Some(StreamAccumulator::new());
        apply_event(&mut state, AgentEvent::TurnComplete);
        assert!(!state.waiting);
        assert!(state.streaming.is_none());
        assert!(state.error.is_none());
    }

    #[test]
    fn error_frame_records_message_and_clears_streaming() {
        let mut state = SplitState::new();
        state.waiting = true;
        state.streaming = Some(StreamAccumulator::new());
        apply_event(
            &mut state,
            AgentEvent::Error {
                message: "model overloaded".into(),
            },
        );
        assert_eq!(state.error.as_deref(), Some("model overloaded"));
        assert!(!state.waiting);
        assert!(state.streaming.is_none());
    }

    #[test]
    fn stream_delta_seeds_accumulator_on_first_event() {
        let mut state = SplitState::new();
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "hello".into(),
                },
            },
        );
        assert!(state.streaming.is_some());
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: " world".into(),
                },
            },
        );
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::Finished {
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 2,
                        reasoning_tokens: 0,
                    },
                },
            },
        );
        let snap = state.streaming.as_ref().unwrap().snapshot();
        assert_eq!(snap.token_count, 2);
        match &snap.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello world"),
            _ => panic!("expected Text block"),
        }
    }

    #[test]
    fn turn_cancelled_sets_cancelled_and_clears_waiting_and_streaming() {
        let mut state = SplitState::new();
        state.waiting = true;
        state.streaming = Some(StreamAccumulator::new());
        apply_event(&mut state, AgentEvent::TurnCancelled);
        assert!(!state.waiting, "waiting must be cleared");
        assert!(state.streaming.is_none(), "streaming must be cleared");
        assert!(state.error.is_none(), "error must be cleared");
        assert!(state.cancelled, "cancelled must be set");
    }

    #[test]
    fn turn_cancelled_commits_streaming_content_as_message() {
        let mut state = SplitState::new();
        state.waiting = true;
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial output".into(),
                },
            },
        );
        assert!(state.streaming.is_some());
        apply_event(&mut state, AgentEvent::TurnCancelled);
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, Role::Assistant);
        assert_eq!(state.messages[0].text(), "partial output");
        assert!(state.streaming.is_none());
    }

    #[test]
    fn turn_cancelled_with_no_streaming_appends_no_message() {
        let mut state = SplitState::new();
        state.waiting = true;
        apply_event(&mut state, AgentEvent::TurnCancelled);
        assert!(state.messages.is_empty(), "no message should be appended");
        assert!(state.cancelled);
    }

    #[test]
    fn turn_cancelled_with_empty_accumulator_does_not_commit() {
        // StreamAccumulator::new() with no pushes yields an empty content
        // vec — the cancel handler must not append a zero-block message.
        let mut state = SplitState::new();
        state.waiting = true;
        state.streaming = Some(StreamAccumulator::new());
        apply_event(&mut state, AgentEvent::TurnCancelled);
        assert!(
            state.messages.is_empty(),
            "must not commit empty assistant message"
        );
        assert!(state.streaming.is_none());
        assert!(state.cancelled);
    }

    #[test]
    fn turn_cancelled_after_message_appended_does_not_double_commit() {
        // Realistic wire sequence: the agent commits the partial assistant
        // message via MessageAppended *before* sending TurnCancelled. The
        // MessageAppended handler clears the streaming accumulator, so the
        // TurnCancelled handler should find no accumulator and not append
        // a duplicate message.
        let mut state = SplitState::new();
        state.waiting = true;
        apply_event(
            &mut state,
            AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            },
        );
        apply_event(
            &mut state,
            AgentEvent::MessageAppended {
                message: Message::assistant(vec![ContentBlock::Text {
                    text: "partial".into(),
                }]),
            },
        );
        assert!(
            state.streaming.is_none(),
            "MessageAppended should clear the accumulator"
        );
        apply_event(&mut state, AgentEvent::TurnCancelled);
        assert_eq!(
            state.messages.len(),
            1,
            "message must not be double-committed"
        );
        assert_eq!(state.messages[0].text(), "partial");
        assert!(state.cancelled);
        assert!(!state.waiting);
    }

    // -- begin_send -------------------------------------------------------

    #[test]
    fn begin_send_skips_when_already_waiting() {
        let mut state = SplitState::new();
        state.waiting = true;
        state.error = Some("preserved".into());
        state.cancelled = true;
        let outcome = begin_send(&mut state);
        assert_eq!(outcome, ShouldSend::Skip);
        assert!(state.waiting);
        // Skip path mutates nothing — error and cancelled stay as they were.
        assert_eq!(state.error.as_deref(), Some("preserved"));
        assert!(state.cancelled);
    }

    #[test]
    fn begin_send_flips_waiting_and_clears_error_and_cancelled_when_idle() {
        let mut state = SplitState::new();
        state.error = Some("prev".into());
        state.cancelled = true;
        let outcome = begin_send(&mut state);
        assert_eq!(outcome, ShouldSend::Send);
        assert!(state.waiting);
        assert!(state.error.is_none());
        assert!(!state.cancelled);
    }
}
