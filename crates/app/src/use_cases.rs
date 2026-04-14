use anyhow::Result;
use domain::{Message, SessionId};
use futures::StreamExt;

use crate::ports::{LlmProvider, SessionStore};
use crate::stream::StreamAccumulator;

pub struct ContinueSession<L, S> {
    llm: L,
    store: S,
}

impl<L: LlmProvider, S: SessionStore> ContinueSession<L, S> {
    pub fn new(llm: L, store: S) -> Self {
        Self { llm, store }
    }

    pub async fn execute(&self, id: SessionId, input: &str) -> Result<Message> {
        let mut session = self.store.load(id).await?;
        session.push_message(Message::user(input));

        let mut event_stream = self.llm.stream(&session.messages, &[]).await?;
        let mut acc = StreamAccumulator::new();
        while let Some(event) = event_stream.next().await {
            acc.push(event?);
        }
        let response = acc.into_message();

        session.push_message(response.clone());
        self.store.save(&session).await?;
        Ok(response)
    }
}
