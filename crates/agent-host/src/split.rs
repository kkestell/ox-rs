//! Per-agent split state.
//!
//! One `AgentSplit` per running `ox-agent` subprocess. Holds the
//! [`AgentClient`] handle (for sending commands), the committed-message
//! history, and a transient `StreamAccumulator` that mirrors the in-flight
//! assistant turn so a UI layer can render live tokens without waiting for
//! the final commit.
//!
//! The receive side of the IPC channel lives elsewhere — `WorkspaceState`
//! (and above it, a per-split drain task) pulls events off the
//! [`AgentEventStream`] and routes them here via [`AgentSplit::handle_event`].
//!
//! Invariant: after `AgentEvent::Ready`, `waiting == false` and
//! `streaming == None`. Replayed-history `MessageAppended` frames append to
//! `messages` without flipping `waiting` or touching `streaming`, so they
//! look to the renderer identical to live-appended messages.

use app::StreamAccumulator;
use domain::{Message, SessionId};
use protocol::{AgentCommand, AgentEvent};
use tokio::sync::mpsc;

use crate::client::{AgentClient, SplitId};

pub struct AgentSplit {
    client: AgentClient,
    pub(crate) messages: Vec<Message>,
    pub(crate) streaming: Option<StreamAccumulator>,
    pub(crate) waiting: bool,
    pub(crate) error: Option<String>,
    pub(crate) session_id: Option<SessionId>,
    /// Set when the most recent turn was cancelled by the user. Cleared
    /// on the next `SendMessage`. A UI layer renders a "Cancelled" label
    /// when this is true.
    pub(crate) cancelled: bool,
}

impl AgentSplit {
    pub fn new(client: AgentClient) -> Self {
        Self {
            client,
            messages: Vec::new(),
            streaming: None,
            waiting: false,
            error: None,
            session_id: None,
            cancelled: false,
        }
    }

    /// Stable UI-local identity assigned when the underlying client was built.
    pub fn id(&self) -> SplitId {
        self.client.id()
    }

    /// Session ID reported by the agent via `Ready`. `None` until the
    /// handshake completes.
    pub fn session_id(&self) -> Option<SessionId> {
        self.session_id
    }

    /// Forward a command to the split's agent. Returns `Err` only if the
    /// writer task has already shut down (agent dead or the client is being
    /// torn down).
    pub fn send(
        &self,
        cmd: AgentCommand,
    ) -> std::result::Result<(), mpsc::error::SendError<AgentCommand>> {
        self.client.send(cmd)
    }

    /// `true` if the split is waiting for an agent reply or already
    /// receiving stream deltas. Used to decide whether quit confirmation
    /// is needed.
    pub fn is_turn_in_progress(&self) -> bool {
        self.waiting || self.streaming.is_some()
    }

    /// Apply a single event from the agent to this split's state. Mirrors
    /// the previous egui loop's `poll_events` body one event at a time.
    pub fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Ready { session_id, .. } => {
                // Record the ID but do NOT touch `messages`, `waiting`,
                // or `streaming`. History arrives as a sequence of
                // `MessageAppended` frames handled below.
                self.session_id = Some(session_id);
            }
            AgentEvent::StreamDelta { event } => {
                self.streaming
                    .get_or_insert_with(StreamAccumulator::new)
                    .push(event);
            }
            AgentEvent::MessageAppended { message } => {
                // Single code path for committed messages — used for live
                // turn messages and for historical replay on resume. Drop
                // the accumulator on each commit so the live-view branch
                // stops rendering the message we just appended.
                self.messages.push(message);
                self.streaming = None;
                self.error = None;
            }
            AgentEvent::TurnComplete => {
                self.waiting = false;
                self.streaming = None;
                self.error = None;
            }
            AgentEvent::TurnCancelled => {
                self.waiting = false;
                // Commit any partial streaming content as a message before
                // discarding the accumulator.
                if let Some(acc) = self.streaming.take() {
                    let msg = acc.into_message();
                    if !msg.content.is_empty() {
                        self.messages.push(msg);
                    }
                }
                self.cancelled = true;
                self.error = None;
            }
            AgentEvent::Error { message } => {
                self.streaming = None;
                self.error = Some(message);
                self.waiting = false;
            }
            // AgentEvent is `#[non_exhaustive]`; any future variant we have
            // not taught the host about yet should be a no-op, not crash.
            _ => {}
        }
    }

    /// Read-only view of the committed messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Snapshot of the live streaming accumulator, if one is in flight.
    pub fn streaming_content(&self) -> Option<Vec<domain::ContentBlock>> {
        self.streaming
            .as_ref()
            .map(|a| a.snapshot().content.to_vec())
    }

    pub fn is_waiting(&self) -> bool {
        self.waiting
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

#[cfg(test)]
mod tests {
    //! `AgentSplit::handle_event` tests.
    //!
    //! Each test pins down one documented invariant:
    //!
    //! - `Ready` records `session_id` and *only* that — `messages`,
    //!   `waiting`, `streaming`, `error` stay untouched.
    //! - `MessageAppended` is the single entry point for both historical
    //!   replay and live turn messages; it appends unconditionally.
    //! - `TurnComplete` clears `waiting` and `streaming` and `error`.
    //! - `TurnCancelled` commits any partial streaming content as a
    //!   message (if non-empty) and flips `cancelled`.
    //! - `Error` clears `waiting` and `streaming` and records the message.
    //!
    //! Because `handle_event` is synchronous and in-memory now, tests drive
    //! the state machine directly with hand-built `AgentEvent`s. No async
    //! pipes, no timeouts, no polling loops.

    use std::path::PathBuf;

    use domain::{ContentBlock, Message, Role, StreamEvent, Usage};
    use tokio::io::{BufReader, duplex};

    use super::*;

    /// Build an `AgentSplit` with a throwaway client. The duplex pipes are
    /// kept alive via `_leaked` so the background reader/writer tasks don't
    /// see unexpected EOF while a test is running; no test in this module
    /// reads events off the stream.
    fn make_split() -> AgentSplit {
        let (_agent_writer, client_reader) = duplex(4096);
        let (client_writer, _agent_reader) = duplex(4096);
        // Leak the agent-side halves. Dropping them would trigger
        // disconnect-path behavior in the reader/writer tasks and add
        // noise to tests that only care about `handle_event`.
        std::mem::forget(_agent_writer);
        std::mem::forget(_agent_reader);
        let (client, _stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        // Drop the stream — tests feed events into `handle_event` directly.
        AgentSplit::new(client)
    }

    fn run(f: impl FnOnce()) {
        // AgentClient::new spawns tokio tasks internally, so a runtime must
        // be in scope even though `handle_event` itself is synchronous.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = rt.enter();
        f();
    }

    #[test]
    fn ready_records_session_id_and_leaves_other_fields_untouched() {
        run(|| {
            let mut split = make_split();
            let id = SessionId::new_v4();
            split.handle_event(AgentEvent::Ready {
                session_id: id,
                workspace_root: PathBuf::from("/w"),
            });

            assert_eq!(split.session_id(), Some(id));
            assert!(split.messages.is_empty(), "Ready must not touch messages");
            assert!(!split.waiting, "Ready must not flip waiting");
            assert!(split.streaming.is_none(), "Ready must not touch streaming");
            assert!(split.error.is_none(), "Ready must not set an error");
        });
    }

    #[test]
    fn message_appended_extends_history_without_touching_streaming() {
        run(|| {
            let mut split = make_split();
            split.handle_event(AgentEvent::MessageAppended {
                message: Message::user("hi"),
            });

            assert_eq!(split.messages.len(), 1);
            assert_eq!(split.messages[0].text(), "hi");
            assert!(!split.waiting, "history replay should not set waiting");
            assert!(split.streaming.is_none());
        });
    }

    #[test]
    fn message_appended_drops_any_inflight_stream_accumulator() {
        // Realistic ordering: StreamDelta arrives, then the agent commits
        // the final message via MessageAppended. The accumulator must be
        // dropped so the renderer does not render the same tokens twice.
        run(|| {
            let mut split = make_split();
            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            });
            assert!(split.streaming.is_some());
            split.handle_event(AgentEvent::MessageAppended {
                message: Message::user("final"),
            });

            assert!(
                split.streaming.is_none(),
                "streaming mirror must be dropped on commit"
            );
        });
    }

    #[test]
    fn turn_complete_clears_waiting_and_streaming_and_error() {
        run(|| {
            let mut split = make_split();
            split.waiting = true;
            split.error = Some("old".into());
            split.streaming = Some(StreamAccumulator::new());

            split.handle_event(AgentEvent::TurnComplete);
            assert!(!split.waiting);
            assert!(split.streaming.is_none());
            assert!(split.error.is_none());
        });
    }

    #[test]
    fn error_frame_records_message_and_clears_streaming() {
        run(|| {
            let mut split = make_split();
            split.waiting = true;
            split.streaming = Some(StreamAccumulator::new());

            split.handle_event(AgentEvent::Error {
                message: "model overloaded".into(),
            });
            assert_eq!(split.error.as_deref(), Some("model overloaded"));
            assert!(!split.waiting);
            assert!(split.streaming.is_none());
        });
    }

    #[test]
    fn stream_delta_seeds_accumulator_on_first_event() {
        run(|| {
            let mut split = make_split();
            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "hello".into(),
                },
            });
            assert!(split.streaming.is_some());

            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: " world".into(),
                },
            });
            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::Finished {
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 2,
                        reasoning_tokens: 0,
                    },
                },
            });

            let snap = split.streaming.as_ref().unwrap().snapshot();
            assert_eq!(snap.token_count, 2);
            match &snap.content[0] {
                ContentBlock::Text { text } => assert_eq!(text, "hello world"),
                _ => panic!("expected Text block"),
            }
        });
    }

    #[test]
    fn turn_cancelled_sets_cancelled_and_clears_waiting_and_streaming() {
        run(|| {
            let mut split = make_split();
            split.waiting = true;
            split.streaming = Some(StreamAccumulator::new());

            split.handle_event(AgentEvent::TurnCancelled);
            assert!(!split.waiting, "waiting must be cleared");
            assert!(split.streaming.is_none(), "streaming must be cleared");
            assert!(split.error.is_none(), "error must be cleared");
            assert!(split.cancelled, "cancelled must be set");
        });
    }

    #[test]
    fn turn_cancelled_commits_streaming_content_as_message() {
        run(|| {
            let mut split = make_split();
            split.waiting = true;
            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial output".into(),
                },
            });
            assert!(split.streaming.is_some());

            split.handle_event(AgentEvent::TurnCancelled);
            assert_eq!(split.messages.len(), 1);
            assert_eq!(split.messages[0].role, Role::Assistant);
            assert_eq!(split.messages[0].text(), "partial output");
            assert!(split.streaming.is_none());
        });
    }

    #[test]
    fn turn_cancelled_with_no_streaming_appends_no_message() {
        run(|| {
            let mut split = make_split();
            split.waiting = true;

            split.handle_event(AgentEvent::TurnCancelled);
            assert!(split.messages.is_empty(), "no message should be appended");
            assert!(split.cancelled);
        });
    }

    #[test]
    fn turn_cancelled_after_message_appended_does_not_double_commit() {
        // Realistic wire sequence: the agent commits the partial assistant
        // message via MessageAppended *before* sending TurnCancelled. The
        // MessageAppended handler clears the streaming accumulator, so the
        // TurnCancelled handler should find no accumulator and not append
        // a duplicate message.
        run(|| {
            let mut split = make_split();
            split.waiting = true;

            split.handle_event(AgentEvent::StreamDelta {
                event: StreamEvent::TextDelta {
                    delta: "partial".into(),
                },
            });
            split.handle_event(AgentEvent::MessageAppended {
                message: Message::assistant(vec![ContentBlock::Text {
                    text: "partial".into(),
                }]),
            });
            assert!(
                split.streaming.is_none(),
                "MessageAppended should clear the accumulator"
            );

            split.handle_event(AgentEvent::TurnCancelled);
            assert_eq!(
                split.messages.len(),
                1,
                "message must not be double-committed"
            );
            assert_eq!(split.messages[0].text(), "partial");
            assert!(split.cancelled);
            assert!(!split.waiting);
        });
    }
}
