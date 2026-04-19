use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use domain::{Message, Session, SessionId, SessionSummary, StreamEvent, ToolDef, Usage};
use futures::stream::{self, Stream};

use crate::LlmProvider;
use crate::ports::{CommandOutput, FileSystem, SessionStore};
use crate::tools::Tool;

/// A queued response: a sequence of stream events (success), a connection-time
/// error, a mid-stream error, or a channel for fine-grained timing control.
enum QueuedResponse {
    Events(Vec<StreamEvent>),
    Error(String),
    /// Some events succeed, then the stream yields an error.
    MidStreamError {
        events: Vec<StreamEvent>,
        error: String,
    },
    /// A live channel the test pushes events through. Gives tests precise
    /// control over when each event arrives relative to external state
    /// changes (e.g. setting a `CancelToken`).
    Channel(futures::channel::mpsc::Receiver<Result<StreamEvent>>),
}

/// Test double for `LlmProvider`. Queue responses ahead of time; each
/// `stream()` call pops the next one. Panics if the queue is empty — a
/// missing response is a test bug, not a graceful failure.
pub struct FakeLlmProvider {
    responses: Mutex<VecDeque<QueuedResponse>>,
    /// Every `system_prompt` value passed to `stream()`, in call order.
    system_prompts: Mutex<Vec<String>>,
}

impl Default for FakeLlmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeLlmProvider {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            system_prompts: Mutex::new(Vec::new()),
        }
    }

    /// All `system_prompt` values passed to `stream()` so far, in call order.
    pub fn system_prompts(&self) -> Vec<String> {
        self.system_prompts.lock().unwrap().clone()
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
            StreamEvent::TextDelta {
                delta: text.to_owned(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: text.len() as u32,
                    reasoning_tokens: 0,
                },
            },
        ]);
    }

    /// Queue a mid-stream error: the given events succeed, then the stream
    /// yields an error. Simulates failures like a dropped connection after
    /// partial output.
    pub fn push_error_after(&self, events: Vec<StreamEvent>, msg: impl Into<String>) {
        self.responses
            .lock()
            .unwrap()
            .push_back(QueuedResponse::MidStreamError {
                events,
                error: msg.into(),
            });
    }

    /// Queue a channel-based response. Returns the sender side; the test
    /// pushes `Ok(StreamEvent)` or `Err(...)` items through it. The stream
    /// ends when the sender is dropped.
    pub fn push_channel(&self) -> futures::channel::mpsc::Sender<Result<StreamEvent>> {
        let (tx, rx) = futures::channel::mpsc::channel(64);
        self.responses
            .lock()
            .unwrap()
            .push_back(QueuedResponse::Channel(rx));
        tx
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
        system_prompt: &str,
        _tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        self.system_prompts
            .lock()
            .unwrap()
            .push(system_prompt.to_owned());
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
            QueuedResponse::MidStreamError { events, error } => {
                // Yield the successful events, then an error item.
                let items: Vec<Result<StreamEvent>> = events
                    .into_iter()
                    .map(Ok)
                    .chain(std::iter::once(Err(anyhow::anyhow!("{error}"))))
                    .collect();
                Ok(Box::pin(stream::iter(items)))
            }
            QueuedResponse::Channel(rx) => Ok(Box::pin(rx)),
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

impl Default for FakeSessionStore {
    fn default() -> Self {
        Self::new()
    }
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
    fn try_load(
        &self,
        id: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Session>>> + Send + '_>> {
        Box::pin(async move { Ok(self.sessions.lock().unwrap().get(&id).cloned()) })
    }

    fn save<'a>(
        &'a self,
        session: &'a Session,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.sessions
                .lock()
                .unwrap()
                .insert(session.id, session.clone());
            Ok(())
        })
    }

    fn list(&self) -> Pin<Box<dyn Future<Output = Result<Vec<SessionSummary>>> + Send + '_>> {
        Box::pin(async move {
            let guard = self.sessions.lock().unwrap();
            let summaries = guard
                .values()
                .map(|s| SessionSummary { id: s.id })
                .collect();
            Ok(summaries)
        })
    }

    fn delete(
        &self,
        id: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            // Mirrors `DiskSessionStore::delete`: removing a missing id is a
            // success so tests can exercise idempotent delete paths without
            // having to probe for existence first.
            self.sessions.lock().unwrap().remove(&id);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// FakeFileSystem
// ---------------------------------------------------------------------------

/// Test double for `FileSystem`. Backed by an in-memory `HashMap` keyed by
/// absolute path. Writes record parent directories in a set so tests can
/// assert that a write either targeted an existing directory or created one.
///
/// Paths are stored verbatim — no canonicalization. Callers must be consistent
/// about absolute-vs-relative paths, which matches how the real tools resolve
/// paths against the workspace root before handing them to the `FileSystem`.
pub struct FakeFileSystem {
    files: Mutex<HashMap<PathBuf, String>>,
    /// Every parent directory that `write` has ensured exists. A real
    /// filesystem doesn't track this separately, but tests benefit from a
    /// way to assert "yes, this write implicitly created that dir tree."
    created_dirs: Mutex<Vec<PathBuf>>,
    /// Paths that `walk_glob` includes in results but `read` knows nothing
    /// about. Simulates files that are discovered by directory walking but
    /// fail to read (e.g. permission errors, deleted between walk and read).
    ghost_paths: Mutex<Vec<PathBuf>>,
}

impl FakeFileSystem {
    pub fn new() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
            created_dirs: Mutex::new(Vec::new()),
            ghost_paths: Mutex::new(Vec::new()),
        }
    }

    /// Seed the fake with a file at `path`.
    pub fn insert(&self, path: impl Into<PathBuf>, content: impl Into<String>) {
        self.files
            .lock()
            .unwrap()
            .insert(path.into(), content.into());
    }

    /// Snapshot of a file's current contents, if present.
    pub fn get(&self, path: &Path) -> Option<String> {
        self.files.lock().unwrap().get(path).cloned()
    }

    /// Set of parent directories that `write` has "created" via this fake.
    pub fn created_dirs(&self) -> Vec<PathBuf> {
        self.created_dirs.lock().unwrap().clone()
    }

    /// Register a path that `walk_glob` will include in results but `read`
    /// will fail on. Used to test tool behavior when a file disappears or
    /// becomes unreadable between discovery and read.
    pub fn insert_ghost(&self, path: impl Into<PathBuf>) {
        self.ghost_paths.lock().unwrap().push(path.into());
    }
}

impl Default for FakeFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystem for FakeFileSystem {
    async fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
        Ok(crate::approval::normalize_lexical(path))
    }

    async fn read(&self, path: &Path) -> Result<String> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("FakeFileSystem: no file at {}", path.display()))
    }

    async fn write(&self, path: &Path, content: &str) -> Result<()> {
        // Mirror `LocalFileSystem::write`'s parent-dir-creation behavior so
        // tests that exercise "writes into a new subdirectory" produce the
        // same observable outcome in-memory as on disk.
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            self.created_dirs.lock().unwrap().push(parent.to_path_buf());
        }
        self.files
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), content.to_owned());
        Ok(())
    }

    async fn walk_glob(
        &self,
        root: &Path,
        pattern: &str,
        max_bytes: usize,
    ) -> Result<crate::ports::WalkResult> {
        let pat = glob::Pattern::new(pattern)
            .map_err(|e| anyhow::anyhow!("invalid glob pattern: {e}"))?;
        let matches_filter = |path: &PathBuf| -> bool {
            let Some(relative) = path.strip_prefix(root).ok() else {
                return false;
            };
            pat.matches_path(relative)
        };
        let files = self.files.lock().unwrap();
        let ghosts = self.ghost_paths.lock().unwrap();
        let mut all: Vec<PathBuf> = files
            .keys()
            .chain(ghosts.iter())
            .filter(|path| matches_filter(path))
            .cloned()
            .collect();
        all.sort();

        let mut results = Vec::new();
        let mut cumulative_bytes: usize = 0;
        let mut truncated = false;
        for path in all {
            let path_bytes = path.to_string_lossy().len();
            if cumulative_bytes + path_bytes > max_bytes {
                truncated = true;
                break;
            }
            cumulative_bytes += path_bytes;
            results.push(path);
        }

        Ok(crate::ports::WalkResult {
            paths: results,
            truncated,
        })
    }
}

// ---------------------------------------------------------------------------
// FakeTool
// ---------------------------------------------------------------------------

/// Test double for `Tool`. Executions consume canned results in FIFO order.
/// Panics on extra calls — missing a canned result in a test is a bug.
///
/// Keeps a minimal call log for tests that need to assert dispatch order or
/// that arguments survived JSON round-tripping through the LLM.
pub struct FakeTool {
    def: ToolDef,
    results: Mutex<VecDeque<Result<String, String>>>,
    calls: Mutex<Vec<String>>,
    requires_approval: bool,
}

impl FakeTool {
    pub fn new(name: &str) -> Self {
        Self {
            def: ToolDef {
                name: name.to_owned(),
                description: format!("fake tool '{name}'"),
                parameters: serde_json::json!({"type": "object"}),
            },
            results: Mutex::new(VecDeque::new()),
            calls: Mutex::new(Vec::new()),
            requires_approval: false,
        }
    }

    pub fn new_requiring_approval(name: &str) -> Self {
        let mut tool = Self::new(name);
        tool.requires_approval = true;
        tool
    }

    /// Queue a successful result.
    pub fn push_ok(&self, output: impl Into<String>) {
        self.results.lock().unwrap().push_back(Ok(output.into()));
    }

    /// Queue an error result. Surfaces as `Err` from `execute`, which the
    /// tool loop translates into a `ToolResult { is_error: true }`.
    pub fn push_err(&self, msg: impl Into<String>) {
        self.results.lock().unwrap().push_back(Err(msg.into()));
    }

    /// All arguments passed to `execute` so far, in call order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl Tool for FakeTool {
    fn def(&self) -> ToolDef {
        self.def.clone()
    }

    fn approval_requirement<'a>(
        &'a self,
        _args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<crate::approval::ApprovalRequirement>> + Send + 'a>>
    {
        let requires = self.requires_approval;
        Box::pin(async move {
            if requires {
                Ok(crate::approval::ApprovalRequirement::Required {
                    reason: "test tool requires approval".into(),
                })
            } else {
                Ok(crate::approval::ApprovalRequirement::NotRequired)
            }
        })
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push(args.to_owned());
            match self
                .results
                .lock()
                .unwrap()
                .pop_front()
                .expect("FakeTool: no results queued — did you forget to call push_ok/push_err?")
            {
                Ok(s) => Ok(s),
                Err(e) => Err(anyhow::anyhow!("{e}")),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// FakeShell
// ---------------------------------------------------------------------------

/// Test double for `Shell`. Queued `CommandOutput` results consumed in FIFO
/// order. Records every (command, timeout, max_bytes) triple for assertion.
pub struct FakeShell {
    results: Mutex<VecDeque<Result<CommandOutput, String>>>,
    calls: Mutex<Vec<(String, std::time::Duration, usize)>>,
}

impl Default for FakeShell {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeShell {
    pub fn new() -> Self {
        Self {
            results: Mutex::new(VecDeque::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Queue a successful result.
    pub fn push_output(&self, output: CommandOutput) {
        self.results.lock().unwrap().push_back(Ok(output));
    }

    /// Queue an error result.
    pub fn push_err(&self, msg: impl Into<String>) {
        self.results.lock().unwrap().push_back(Err(msg.into()));
    }

    /// All (command, timeout, max_bytes) triples passed to `run` so far.
    pub fn calls(&self) -> Vec<(String, std::time::Duration, usize)> {
        self.calls.lock().unwrap().clone()
    }
}

impl crate::ports::Shell for FakeShell {
    async fn run(
        &self,
        command: &str,
        timeout: std::time::Duration,
        max_bytes: usize,
    ) -> Result<CommandOutput> {
        self.calls
            .lock()
            .unwrap()
            .push((command.to_owned(), timeout, max_bytes));
        match self
            .results
            .lock()
            .unwrap()
            .pop_front()
            .expect("FakeShell: no results queued — did you forget to call push_output/push_err?")
        {
            Ok(output) => Ok(output),
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    }
}

/// Build a `ToolRegistry` pre-populated with the given fakes. Convenience for
/// tests that want to stand up a registry without repeating the registration
/// boilerplate — takes the fakes by `Arc` so callers can still observe call
/// logs on the originals after the registry consumes them.
pub fn tool_registry_with(tools: Vec<Arc<dyn Tool>>) -> crate::tools::ToolRegistry {
    let mut reg = crate::tools::ToolRegistry::new();
    for t in tools {
        reg.register(t);
    }
    reg
}

#[cfg(test)]
mod tests {
    use domain::{ContentBlock, Role, StreamAccumulator};
    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn fake_emits_text_response() {
        let fake = FakeLlmProvider::new();
        fake.push_text("hello world");

        let mut stream = fake.stream(&[], "", &[]).await.unwrap();
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

        let mut stream = fake.stream(&[], "", &[]).await.unwrap();
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
        let mut s1 = fake.stream(&[], "", &[]).await.unwrap();
        let mut acc1 = StreamAccumulator::new();
        while let Some(e) = s1.next().await {
            acc1.push(e.unwrap());
        }
        assert_eq!(acc1.into_message().text(), "first");

        // Second call
        let mut s2 = fake.stream(&[], "", &[]).await.unwrap();
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
        let _ = fake.stream(&[], "", &[]).await;
    }

    // -- FakeSessionStore tests --

    #[tokio::test]
    async fn store_save_load_roundtrip() {
        let store = FakeSessionStore::new();
        let id = SessionId::new_v4();
        let mut session = Session::new(id, "/tmp/project".into(), "/tmp/project/wt".into());
        session.push_message(Message::user("hello"));

        store.save(&session).await.unwrap();
        let loaded = store.try_load(id).await.unwrap().expect("session exists");

        assert_eq!(loaded.id, id);
        assert_eq!(loaded.workspace_root.to_str().unwrap(), "/tmp/project");
        assert_eq!(loaded.worktree_path.to_str().unwrap(), "/tmp/project/wt");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "hello");
    }

    #[tokio::test]
    async fn store_try_load_nonexistent_returns_none() {
        let store = FakeSessionStore::new();
        let result = store.try_load(SessionId::new_v4()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn store_list_returns_summaries() {
        let store = FakeSessionStore::new();

        let id1 = SessionId::new_v4();
        let s1 = Session::new(id1, "/a".into(), "/a/wt1".into());
        store.save(&s1).await.unwrap();

        let id2 = SessionId::new_v4();
        let s2 = Session::new(id2, "/b".into(), "/b/wt2".into());
        store.save(&s2).await.unwrap();

        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 2);
        assert!(summaries.iter().any(|s| s.id == id1));
        assert!(summaries.iter().any(|s| s.id == id2));
    }

    #[tokio::test]
    async fn store_delete_removes_and_is_idempotent() {
        let store = FakeSessionStore::new();
        let id = SessionId::new_v4();
        let session = Session::new(id, "/a".into(), "/a/wt".into());
        store.save(&session).await.unwrap();

        store.delete(id).await.unwrap();
        assert!(
            store.try_load(id).await.unwrap().is_none(),
            "loaded a deleted session"
        );

        // Second delete of the same id succeeds — matches the port contract.
        store.delete(id).await.unwrap();
        // Deleting an id that was never inserted also succeeds.
        store.delete(SessionId::new_v4()).await.unwrap();
    }
}
