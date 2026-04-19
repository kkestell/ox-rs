use std::pin::Pin;

use anyhow::Result;
use domain::{Message, StreamEvent, ToolDef};
use futures::stream::Stream;

pub struct OllamaProvider {
    _client: reqwest::Client,
    _base_url: String,
}

impl OllamaProvider {
    pub fn new(base_url: String) -> Self {
        Self {
            _client: reqwest::Client::new(),
            _base_url: base_url,
        }
    }
}

impl app::LlmProvider for OllamaProvider {
    async fn stream(
        &self,
        _messages: &[Message],
        _system_prompt: &str,
        _tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        todo!()
    }
}
