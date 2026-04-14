use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Mutex;

use anyhow::Result;
use domain::{Message, Session, SessionId, SessionSummary};
use futures::stream::{self, Stream};

use crate::LlmProvider;
use crate::ports::SessionStore;
use crate::stream::{StreamEvent, ToolDef, Usage};

/// A queued response: either a sequence of stream events (success) or an
/// error message that `stream()` will return as `Err(...)`.
enum QueuedResponse {
    Events(Vec<StreamEvent>),
    Error(String),
}

/// Test double for `LlmProvider`. Queue responses ahead of time; each
/// `stream()` call pops the next one. Panics if the queue is empty — a
/// missing response is a test bug, not a graceful failure.
pub struct FakeLlmProvider {
    responses: Mutex<VecDeque<QueuedResponse>>,
}

impl FakeLlmProvider {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
        }
    }

    /// Queue a raw sequence of events.
    pub fn push_response(&self, events: Vec<StreamEvent>) {
        self.responses
            .lock()
            .unwrap()
            .push_back(QueuedResponse::Events(events));
    }

    /// Queue an error response. The next `stream()` call will return `Err`.
    pub fn push_error(&self, msg: impl Into<String>) {
        self.responses
            .lock()
            .unwrap()
            .push_back(QueuedResponse::Error(msg.into()));
    }

    /// Convenience: queue a simple text response.
    pub fn push_text(&self, text: &str) {
        self.push_response(vec![
            StreamEvent::TextDelta(text.to_owned()),
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: text.len() as u32,
                    reasoning_tokens: 0,
                },
            },
        ]);
    }

    /// Convenience: queue a single tool call response.
    pub fn push_tool_call(&self, id: &str, name: &str, arguments: &str) {
        self.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: id.to_owned(),
                name: name.to_owned(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: arguments.to_owned(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                },
            },
        ]);
    }
}

impl LlmProvider for FakeLlmProvider {
    async fn stream(
        &self,
        _messages: &[Message],
        _tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let queued = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("FakeLlmProvider: no responses queued — did you forget to call push_*?");

        match queued {
            QueuedResponse::Events(events) => {
                Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
            }
            QueuedResponse::Error(msg) => Err(anyhow::anyhow!("{msg}")),
        }
    }
}

// ---------------------------------------------------------------------------
// FakeSessionStore
// ---------------------------------------------------------------------------

/// Test double for `SessionStore`. Stores sessions in a `HashMap` behind a
/// `Mutex`. Useful for verifying that use cases save and load correctly without
/// touching the filesystem.
pub struct FakeSessionStore {
    sessions: Mutex<HashMap<SessionId, Session>>,
}

impl FakeSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Seed the store with a pre-existing session (for resume tests).
    pub fn insert(&self, session: Session) {
        self.sessions.lock().unwrap().insert(session.id, session);
    }

    /// Snapshot of the currently stored session, if any.
    pub fn get(&self, id: SessionId) -> Option<Session> {
        self.sessions.lock().unwrap().get(&id).cloned()
    }
}

impl SessionStore for FakeSessionStore {
    async fn load(&self, id: SessionId) -> Result<Session> {
        self.sessions
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("FakeSessionStore: no session with id {id}"))
    }

    async fn save(&self, session: &Session) -> Result<()> {
        self.sessions
            .lock()
            .unwrap()
            .insert(session.id, session.clone());
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionSummary>> {
        let guard = self.sessions.lock().unwrap();
        let summaries = guard
            .values()
            .map(|s| SessionSummary {
                id: s.id,
                message_count: s.messages.len(),
            })
            .collect();
        Ok(summaries)
    }
}

#[cfg(test)]
mod tests {
    use domain::{ContentBlock, Role};
    use futures::StreamExt;

    use super::*;
    use crate::stream::StreamAccumulator;

    #[tokio::test]
    async fn fake_emits_text_response() {
        let fake = FakeLlmProvider::new();
        fake.push_text("hello world");

        let mut stream = fake.stream(&[], &[]).await.unwrap();
        let mut acc = StreamAccumulator::new();
        while let Some(event) = stream.next().await {
            acc.push(event.unwrap());
        }
        let msg = acc.into_message();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "hello world");
    }

    #[tokio::test]
    async fn fake_emits_tool_call() {
        let fake = FakeLlmProvider::new();
        fake.push_tool_call("call_1", "read_file", r#"{"path":"a.rs"}"#);

        let mut stream = fake.stream(&[], &[]).await.unwrap();
        let mut acc = StreamAccumulator::new();
        while let Some(event) = stream.next().await {
            acc.push(event.unwrap());
        }
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
                assert_eq!(arguments, r#"{"path":"a.rs"}"#);
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[tokio::test]
    async fn multi_turn_consumes_in_order() {
        let fake = FakeLlmProvider::new();
        fake.push_text("first");
        fake.push_text("second");

        // First call
        let mut s1 = fake.stream(&[], &[]).await.unwrap();
        let mut acc1 = StreamAccumulator::new();
        while let Some(e) = s1.next().await {
            acc1.push(e.unwrap());
        }
        assert_eq!(acc1.into_message().text(), "first");

        // Second call
        let mut s2 = fake.stream(&[], &[]).await.unwrap();
        let mut acc2 = StreamAccumulator::new();
        while let Some(e) = s2.next().await {
            acc2.push(e.unwrap());
        }
        assert_eq!(acc2.into_message().text(), "second");
    }

    #[tokio::test]
    #[should_panic(expected = "no responses queued")]
    async fn panics_when_no_responses_queued() {
        let fake = FakeLlmProvider::new();
        let _ = fake.stream(&[], &[]).await;
    }

    // -- FakeSessionStore tests --

    #[tokio::test]
    async fn store_save_load_roundtrip() {
        let store = FakeSessionStore::new();
        let id = SessionId::new_v4();
        let mut session = Session::new(id, "/tmp/project".into());
        session.push_message(Message::user("hello"));

        store.save(&session).await.unwrap();
        let loaded = store.load(id).await.unwrap();

        assert_eq!(loaded.id, id);
        assert_eq!(loaded.workspace_root.to_str().unwrap(), "/tmp/project");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "hello");
    }

    #[tokio::test]
    async fn store_load_nonexistent_returns_error() {
        let store = FakeSessionStore::new();
        let result = store.load(SessionId::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn store_list_returns_summaries() {
        let store = FakeSessionStore::new();

        let id1 = SessionId::new_v4();
        let mut s1 = Session::new(id1, "/a".into());
        s1.push_message(Message::user("hi"));
        s1.push_message(Message::user("there"));
        store.save(&s1).await.unwrap();

        let id2 = SessionId::new_v4();
        let s2 = Session::new(id2, "/b".into());
        store.save(&s2).await.unwrap();

        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 2);

        // Find each summary by ID and check message counts.
        let sum1 = summaries.iter().find(|s| s.id == id1).unwrap();
        assert_eq!(sum1.message_count, 2);
        let sum2 = summaries.iter().find(|s| s.id == id2).unwrap();
        assert_eq!(sum2.message_count, 0);
    }
}
