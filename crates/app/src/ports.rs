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
        system_prompt: &str,
        tools: &[ToolDef],
    ) -> impl Future<Output = Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>>> + Send;
}

pub trait SessionStore {
    fn load(&self, id: SessionId) -> impl Future<Output = Result<Session>> + Send;
    fn save(&self, session: &Session) -> impl Future<Output = Result<()>> + Send;
    fn list(&self) -> impl Future<Output = Result<Vec<SessionSummary>>> + Send;
    /// Remove a session's on-disk record. Missing sessions are treated as
    /// success so callers can invoke `delete` idempotently during merge /
    /// abandon flows without having to race a concurrent removal.
    fn delete(&self, id: SessionId) -> impl Future<Output = Result<()>> + Send;
}

pub trait SecretStore {
    fn get(&self, key: &str) -> Result<Option<String>>;
}

pub trait FileSystem: Send + Sync {
    fn canonicalize(&self, path: &Path) -> impl Future<Output = Result<PathBuf>> + Send;
    fn read(&self, path: &Path) -> impl Future<Output = Result<String>> + Send;
    fn write(&self, path: &Path, content: &str) -> impl Future<Output = Result<()>> + Send;

    /// Return file paths under `root` that match `pattern` (a glob expression
    /// like `**/*.rs`). Results are sorted and contain only files, not
    /// directories. `pattern` is interpreted relative to `root`.
    ///
    /// `max_bytes` bounds the cumulative size of collected path strings to
    /// prevent unbounded memory consumption. When the limit is reached,
    /// collection stops and `WalkResult::truncated` is set to `true`.
    fn walk_glob(
        &self,
        root: &Path,
        pattern: &str,
        max_bytes: usize,
    ) -> impl Future<Output = Result<WalkResult>> + Send;
}

/// Result of a bounded `walk_glob` call. The named struct exists because the
/// caller cannot distinguish "all results returned" from "results were capped"
/// by inspecting the `Vec` alone.
#[derive(Debug, Clone)]
pub struct WalkResult {
    pub paths: Vec<PathBuf>,
    pub truncated: bool,
}

pub trait Shell: Send + Sync {
    fn run(
        &self,
        command: &str,
        timeout: std::time::Duration,
        max_bytes: usize,
    ) -> impl Future<Output = Result<CommandOutput>> + Send;
}

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
    /// True if stdout or stderr was truncated because it exceeded `max_bytes`.
    pub truncated: bool,
}
