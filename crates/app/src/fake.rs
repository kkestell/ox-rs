use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;

use anyhow::Result;
use domain::Message;
use futures::stream::{self, Stream};

use crate::LlmProvider;
use crate::stream::{StreamEvent, ToolDef, Usage};

/// Test double for `LlmProvider`. Queue responses ahead of time; each
/// `stream()` call pops the next one. Panics if the queue is empty — a
/// missing response is a test bug, not a graceful failure.
pub struct FakeLlmProvider {
    responses: Mutex<VecDeque<Vec<StreamEvent>>>,
}

impl FakeLlmProvider {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
        }
    }

    /// Queue a raw sequence of events.
    pub fn push_response(&self, events: Vec<StreamEvent>) {
        self.responses.lock().unwrap().push_back(events);
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
        let events = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("FakeLlmProvider: no responses queued — did you forget to call push_*?");

        Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
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
}
