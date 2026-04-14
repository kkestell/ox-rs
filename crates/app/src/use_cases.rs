use std::path::PathBuf;

use anyhow::Result;
use domain::{Message, Session, SessionId};
use futures::StreamExt;

use crate::ports::{LlmProvider, SessionStore};
use crate::stream::{StreamAccumulator, StreamEvent};

/// Runs conversation turns against an LLM, handling session creation, message
/// accumulation, and persistence. Exposes two entry points:
///
/// - `start` — creates a brand-new session and sends the first user message.
/// - `resume` — loads an existing session and appends a new turn.
///
/// Both delegate to the private `run_turn` method for the stream-accumulate
/// sequence so that streaming logic lives in exactly one place.
pub struct SessionRunner<L, S> {
    llm: L,
    store: S,
}

impl<L: LlmProvider, S: SessionStore> SessionRunner<L, S> {
    pub fn new(llm: L, store: S) -> Self {
        Self { llm, store }
    }

    /// Create a new session and run the first turn. The caller is responsible
    /// for generating the `SessionId` — this keeps the app layer pure and
    /// deterministic.
    ///
    /// The `on_event` callback fires for each stream event as it arrives,
    /// before accumulation — see `run_turn` for details.
    pub async fn start(
        &self,
        id: SessionId,
        workspace_root: PathBuf,
        input: &str,
        on_event: impl FnMut(&StreamEvent) + Send,
    ) -> Result<(SessionId, Message)> {
        let mut session = Session::new(id, workspace_root);
        let response = self.run_turn(&mut session, input, on_event).await?;
        self.store.save(&session).await?;
        Ok((id, response))
    }

    /// Load an existing session and run the next turn.
    ///
    /// The `on_event` callback fires for each stream event as it arrives,
    /// before accumulation — see `run_turn` for details.
    pub async fn resume(
        &self,
        id: SessionId,
        input: &str,
        on_event: impl FnMut(&StreamEvent) + Send,
    ) -> Result<Message> {
        let mut session = self.store.load(id).await?;
        let response = self.run_turn(&mut session, input, on_event).await?;
        self.store.save(&session).await?;
        Ok(response)
    }

    /// Core turn loop: append the user message, stream the LLM response,
    /// accumulate it into a `Message`, and append the assistant reply to
    /// the session. The caller is responsible for saving.
    ///
    /// The `on_event` callback fires for each stream event before it is
    /// accumulated, giving callers a chance to forward incremental updates
    /// (e.g. to a GUI) without breaking SessionRunner's ownership of the
    /// turn lifecycle.
    async fn run_turn(
        &self,
        session: &mut Session,
        input: &str,
        mut on_event: impl FnMut(&StreamEvent) + Send,
    ) -> Result<Message> {
        session.push_message(Message::user(input));

        let mut event_stream = self.llm.stream(&session.messages, &[]).await?;
        let mut acc = StreamAccumulator::new();
        while let Some(event) = event_stream.next().await {
            let event = event?;
            on_event(&event);
            acc.push(event);
        }
        let response = acc.into_message();

        session.push_message(response.clone());
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use domain::Role;

    use super::*;
    use crate::fake::{FakeLlmProvider, FakeSessionStore};

    fn make_runner(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
    ) -> SessionRunner<FakeLlmProvider, FakeSessionStore> {
        SessionRunner::new(llm, store)
    }

    // -- start --

    #[tokio::test]
    async fn start_returns_id_and_assistant_message() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store);
        let id = SessionId::new_v4();
        let (returned_id, msg) = runner
            .start(id, "/project".into(), "Hello", |_| {})
            .await
            .unwrap();

        assert_eq!(returned_id, id);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "Hello back!");
    }

    #[tokio::test]
    async fn start_saves_session_with_two_messages_and_workspace_root() {
        let llm = FakeLlmProvider::new();
        llm.push_text("reply");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store);
        let id = SessionId::new_v4();
        runner
            .start(id, "/my/project".into(), "hi", |_| {})
            .await
            .unwrap();

        // Inspect the saved session through the store.
        let saved = runner.store.get(id).expect("session should be saved");
        assert_eq!(saved.id, id);
        assert_eq!(saved.workspace_root.to_str().unwrap(), "/my/project");
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[0].text(), "hi");
        assert_eq!(saved.messages[1].role, Role::Assistant);
        assert_eq!(saved.messages[1].text(), "reply");
    }

    // -- resume --

    #[tokio::test]
    async fn resume_loads_and_appends_messages() {
        let llm = FakeLlmProvider::new();
        llm.push_text("continued");
        let store = FakeSessionStore::new();

        // Seed the store with a session that already has one exchange.
        let id = SessionId::new_v4();
        let mut existing = Session::new(id, "/project".into());
        existing.push_message(Message::user("turn 1"));
        existing.push_message(Message::assistant(vec![domain::ContentBlock::Text {
            text: "response 1".into(),
        }]));
        store.insert(existing);

        let runner = make_runner(llm, store);
        let msg = runner.resume(id, "turn 2", |_| {}).await.unwrap();

        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "continued");

        // Session should now have 4 messages: original 2 + user turn 2 + assistant reply.
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 4);
        assert_eq!(saved.messages[2].text(), "turn 2");
        assert_eq!(saved.messages[3].text(), "continued");
    }

    #[tokio::test]
    async fn resume_nonexistent_session_returns_error() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store);

        let result = runner.resume(SessionId::new_v4(), "hi", |_| {}).await;
        assert!(result.is_err());
    }

    // -- streaming callback --

    #[tokio::test]
    async fn start_invokes_callback_for_each_event() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::TextDelta("Hello, ".into()),
            StreamEvent::TextDelta("world!".into()),
            StreamEvent::Finished {
                usage: crate::stream::Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store);
        let id = SessionId::new_v4();
        let mut collected = Vec::new();
        let (_, msg) = runner
            .start(id, "/project".into(), "hi", |evt| {
                collected.push(evt.clone());
            })
            .await
            .unwrap();

        // Callback received all three events in order.
        assert_eq!(collected.len(), 3);
        assert!(matches!(&collected[0], StreamEvent::TextDelta(s) if s == "Hello, "));
        assert!(matches!(&collected[1], StreamEvent::TextDelta(s) if s == "world!"));
        assert!(matches!(&collected[2], StreamEvent::Finished { .. }));

        // Final message is still correctly assembled.
        assert_eq!(msg.text(), "Hello, world!");
    }

    #[tokio::test]
    async fn resume_invokes_callback_for_each_event() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::TextDelta("continued".into()),
            StreamEvent::Finished {
                usage: crate::stream::Usage {
                    prompt_tokens: 0,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                },
            },
        ]);
        let store = FakeSessionStore::new();

        let id = SessionId::new_v4();
        let mut existing = Session::new(id, "/project".into());
        existing.push_message(Message::user("turn 1"));
        existing.push_message(Message::assistant(vec![domain::ContentBlock::Text {
            text: "response 1".into(),
        }]));
        store.insert(existing);

        let runner = make_runner(llm, store);
        let mut collected = Vec::new();
        let msg = runner
            .resume(id, "turn 2", |evt| {
                collected.push(evt.clone());
            })
            .await
            .unwrap();

        assert_eq!(collected.len(), 2);
        assert!(matches!(&collected[0], StreamEvent::TextDelta(s) if s == "continued"));
        assert!(matches!(&collected[1], StreamEvent::Finished { .. }));
        assert_eq!(msg.text(), "continued");
    }

    // -- mid-stream error --

    #[tokio::test]
    async fn mid_stream_error_invokes_callback_for_events_before_failure() {
        let llm = FakeLlmProvider::new();
        llm.push_error_after(
            vec![StreamEvent::TextDelta("partial ".into())],
            "connection lost",
        );
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store);
        let id = SessionId::new_v4();
        let mut collected = Vec::new();
        let result = runner
            .start(id, "/project".into(), "hi", |evt| {
                collected.push(evt.clone());
            })
            .await;

        // Callback was invoked for the event before the error.
        assert_eq!(collected.len(), 1);
        assert!(matches!(&collected[0], StreamEvent::TextDelta(s) if s == "partial "));

        // The overall turn failed.
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("connection lost"),
            "error should propagate"
        );
    }

    // -- multi-turn integration --

    #[tokio::test]
    async fn start_then_resume_accumulates_all_messages() {
        let llm = FakeLlmProvider::new();
        llm.push_text("first reply");
        llm.push_text("second reply");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store);
        let id = SessionId::new_v4();

        // Turn 1: start
        let (_, msg1) = runner
            .start(id, "/project".into(), "hello", |_| {})
            .await
            .unwrap();
        assert_eq!(msg1.text(), "first reply");

        // Turn 2: resume
        let msg2 = runner.resume(id, "followup", |_| {}).await.unwrap();
        assert_eq!(msg2.text(), "second reply");

        // Final session should have 4 messages total.
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 4);
        assert_eq!(saved.messages[0].text(), "hello");
        assert_eq!(saved.messages[1].text(), "first reply");
        assert_eq!(saved.messages[2].text(), "followup");
        assert_eq!(saved.messages[3].text(), "second reply");
    }
}
