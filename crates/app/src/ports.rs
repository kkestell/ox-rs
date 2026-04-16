use std::path::{Path, PathBuf};
use std::pin::Pin;

use anyhow::Result;
use domain::{Message, Session, SessionId, SessionSummary, StreamEvent};
use futures::stream::Stream;

use crate::stream::ToolDef;

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

pub trait FileSystem: Send + Sync {
    fn read(&self, path: &Path) -> impl Future<Output = Result<String>> + Send;
    fn write(&self, path: &Path, content: &str) -> impl Future<Output = Result<()>> + Send;

    /// Return all file paths under `root` that match `pattern` (a glob
    /// expression like `**/*.rs`). Results are sorted and contain only files,
    /// not directories. `pattern` is interpreted relative to `root`.
    fn walk_glob(
        &self,
        root: &Path,
        pattern: &str,
    ) -> impl Future<Output = Result<Vec<PathBuf>>> + Send;
}

pub trait Shell: Send + Sync {
    fn run(
        &self,
        command: &str,
        timeout: std::time::Duration,
    ) -> impl Future<Output = Result<CommandOutput>> + Send;
}

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}
