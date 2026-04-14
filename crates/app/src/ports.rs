use std::path::Path;
use std::pin::Pin;

use anyhow::Result;
use domain::{Message, Session, SessionId, SessionSummary};
use futures::stream::Stream;

use crate::stream::{StreamEvent, ToolDef};

// Driven (outbound) ports

pub trait LlmProvider {
    /// Start a streaming completion. The outer Future resolves once the HTTP
    /// connection is established; the inner Stream yields incremental events.
    fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> impl Future<Output = Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>>> + Send;
}

pub trait SessionStore {
    fn load(&self, id: SessionId) -> impl Future<Output = Result<Session>> + Send;
    fn save(&self, session: &Session) -> impl Future<Output = Result<()>> + Send;
    fn list(&self) -> impl Future<Output = Result<Vec<SessionSummary>>> + Send;
}

pub trait SecretStore {
    fn get(&self, key: &str) -> Result<Option<String>>;
}

pub trait FileSystem {
    fn read(&self, path: &Path) -> Result<String>;
    fn write(&self, path: &Path, content: &str) -> Result<()>;
}

pub trait Shell {
    fn run(&self, command: &str) -> impl Future<Output = Result<CommandOutput>> + Send;
}

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}
