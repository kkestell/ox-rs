pub mod sse;
pub mod wire;

use std::pin::Pin;

use anyhow::{Context, Result};
use app::stream::{StreamEvent, ToolDef};
use domain::Message;
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;

use wire::RequestBody;

pub struct OpenRouterProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

impl app::LlmProvider for OpenRouterProvider {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let body = RequestBody::from_messages(&self.model, messages, tools);

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("sending request to OpenRouter")?
            .error_for_status()
            .context("OpenRouter returned an error status")?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(sse::parse_sse_stream(response, tx));

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
