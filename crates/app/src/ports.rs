use std::path::{Path, PathBuf};
use std::pin::Pin;

use anyhow::Result;
use domain::{Message, Session, SessionId, SessionSummary, StreamEvent};
use futures::stream::Stream;

use domain::ToolDef;

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

pub trait SessionStore: Send + Sync + 'static {
    /// Load a session by id, returning `None` when the record does not
    /// exist on disk. Boxed futures keep the trait `dyn`-compatible so
    /// lifecycle code can hold `Arc<dyn SessionStore>`.
    fn try_load(
        &self,
        id: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Session>>> + Send + '_>>;
    fn save<'a>(
        &'a self,
        session: &'a Session,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
    fn list(&self) -> Pin<Box<dyn Future<Output = Result<Vec<SessionSummary>>> + Send + '_>>;
    /// Remove a session's on-disk record. Missing sessions are treated as
    /// success so callers can invoke `delete` idempotently during merge /
    /// abandon flows without having to race a concurrent removal.
    fn delete(&self, id: SessionId) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
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
