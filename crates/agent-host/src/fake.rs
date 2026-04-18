//! Test doubles for the `agent-host` ports.
//!
//! These exist so downstream crates (`bin-web`, the lifecycle
//! coordinator) can be unit-tested without shelling out to git, without
//! spawning subprocesses, and without reaching an LLM. Each fake is
//! deliberately minimal: it records what the code under test did, or
//! hands back a scripted response, and that's it. No retries, no
//! locking, no production surface.
//!
//! Everything here is behind a crate-level opt-in so release builds of
//! downstream binaries don't pay for the test scaffolding (the fakes
//! compile only when the consumer enables `test-support` on its dev
//! dependency on `agent-host`, or when `agent-host`'s own tests run —
//! the module is always compiled for this crate's own unit tests).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Result, bail};
use async_trait::async_trait;
use domain::{CloseIntent, SessionId};

use crate::close_request_sink::CloseRequestSink;
use crate::first_turn_sink::FirstTurnSink;
use crate::git::{Git, MergeOutcome, WorktreeStatus};
use crate::slug_generator::SlugGenerator;

// ---------------------------------------------------------------------------
// FakeGit
// ---------------------------------------------------------------------------

/// Test double for [`Git`]. Backed by `Mutex<FakeGitState>` so tests can
/// script outcomes per path and inspect the call log after.
///
/// Sensible defaults: `assert_repo` returns `Ok(())` for any workspace,
/// `current_branch` returns `"main"`, `status` reports `Clean`, `merge`
/// reports `Merged`. Tests override any of these via the setter methods.
pub struct FakeGit {
    state: Mutex<FakeGitState>,
}

struct FakeGitState {
    /// Paths that should be rejected by `assert_repo`. All others pass.
    non_repo: Vec<PathBuf>,
    /// Paths whose HEAD should report detached (empty string in the
    /// short name → error returned).
    detached: Vec<PathBuf>,
    /// Per-workspace-root override for `current_branch`. Missing → `"main"`.
    branches: HashMap<PathBuf, String>,
    /// Per-worktree override for `status`. Missing → `Clean`.
    statuses: HashMap<PathBuf, WorktreeStatus>,
    /// Next-call override for `merge`. Pops one per call; after empty,
    /// defaults to `Merged`.
    merge_outcomes: Vec<MergeOutcome>,
    /// Every invocation in arrival order. Tests assert on this.
    calls: Vec<GitCall>,
}

/// Observable record of a [`Git`] call. Tests read the ordered log to
/// verify "did the lifecycle coordinator run the merge steps in this
/// order?" without having to mock every intermediate state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitCall {
    AssertRepo(PathBuf),
    CurrentBranch(PathBuf),
    AddWorktree {
        workspace_root: PathBuf,
        worktree_path: PathBuf,
        branch: String,
        base_branch: String,
    },
    Status(PathBuf),
    RenameBranch {
        workspace_root: PathBuf,
        old: String,
        new: String,
    },
    MoveWorktree {
        workspace_root: PathBuf,
        old_path: PathBuf,
        new_path: PathBuf,
    },
    Merge {
        workspace_root: PathBuf,
        branch: String,
    },
    RemoveWorktree {
        workspace_root: PathBuf,
        worktree_path: PathBuf,
    },
    DeleteBranch {
        workspace_root: PathBuf,
        branch: String,
        force: bool,
    },
}

impl Default for FakeGit {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeGit {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(FakeGitState {
                non_repo: Vec::new(),
                detached: Vec::new(),
                branches: HashMap::new(),
                statuses: HashMap::new(),
                merge_outcomes: Vec::new(),
                calls: Vec::new(),
            }),
        }
    }

    /// Mark `path` as not-a-repo; `assert_repo` will error for it.
    pub fn reject_as_non_repo(&self, path: impl Into<PathBuf>) {
        self.state.lock().unwrap().non_repo.push(path.into());
    }

    /// Mark `path` as detached-HEAD; `current_branch` will error for it.
    pub fn mark_detached(&self, path: impl Into<PathBuf>) {
        self.state.lock().unwrap().detached.push(path.into());
    }

    /// Set the branch `current_branch` reports for `workspace_root`.
    pub fn set_current_branch(&self, workspace_root: impl Into<PathBuf>, branch: &str) {
        self.state
            .lock()
            .unwrap()
            .branches
            .insert(workspace_root.into(), branch.to_owned());
    }

    /// Set the status `status()` reports for `worktree_path`.
    pub fn set_status(&self, worktree_path: impl Into<PathBuf>, status: WorktreeStatus) {
        self.state
            .lock()
            .unwrap()
            .statuses
            .insert(worktree_path.into(), status);
    }

    /// Queue the next `merge()` result. FIFO — push once per expected
    /// call. After the queue empties, defaults to `Merged`.
    pub fn enqueue_merge_outcome(&self, outcome: MergeOutcome) {
        self.state.lock().unwrap().merge_outcomes.push(outcome);
    }

    /// Snapshot of every call made so far, in arrival order.
    pub fn calls(&self) -> Vec<GitCall> {
        self.state.lock().unwrap().calls.clone()
    }

    fn record(&self, call: GitCall) {
        self.state.lock().unwrap().calls.push(call);
    }
}

#[async_trait]
impl Git for FakeGit {
    async fn assert_repo(&self, workspace_root: &Path) -> Result<()> {
        self.record(GitCall::AssertRepo(workspace_root.to_path_buf()));
        if self
            .state
            .lock()
            .unwrap()
            .non_repo
            .iter()
            .any(|p| p == workspace_root)
        {
            bail!(
                "FakeGit: {} is not a git repository",
                workspace_root.display()
            );
        }
        Ok(())
    }

    async fn current_branch(&self, workspace_root: &Path) -> Result<String> {
        self.record(GitCall::CurrentBranch(workspace_root.to_path_buf()));
        let state = self.state.lock().unwrap();
        if state.detached.iter().any(|p| p == workspace_root) {
            bail!("FakeGit: HEAD at {} is detached", workspace_root.display());
        }
        Ok(state
            .branches
            .get(workspace_root)
            .cloned()
            .unwrap_or_else(|| "main".to_owned()))
    }

    async fn add_worktree(
        &self,
        workspace_root: &Path,
        worktree_path: &Path,
        branch: &str,
        base_branch: &str,
    ) -> Result<()> {
        self.record(GitCall::AddWorktree {
            workspace_root: workspace_root.to_path_buf(),
            worktree_path: worktree_path.to_path_buf(),
            branch: branch.to_owned(),
            base_branch: base_branch.to_owned(),
        });
        Ok(())
    }

    async fn status(&self, worktree_path: &Path) -> Result<WorktreeStatus> {
        self.record(GitCall::Status(worktree_path.to_path_buf()));
        Ok(self
            .state
            .lock()
            .unwrap()
            .statuses
            .get(worktree_path)
            .copied()
            .unwrap_or(WorktreeStatus::Clean))
    }

    async fn rename_branch(&self, workspace_root: &Path, old: &str, new: &str) -> Result<()> {
        self.record(GitCall::RenameBranch {
            workspace_root: workspace_root.to_path_buf(),
            old: old.to_owned(),
            new: new.to_owned(),
        });
        Ok(())
    }

    async fn move_worktree(
        &self,
        workspace_root: &Path,
        old_path: &Path,
        new_path: &Path,
    ) -> Result<()> {
        self.record(GitCall::MoveWorktree {
            workspace_root: workspace_root.to_path_buf(),
            old_path: old_path.to_path_buf(),
            new_path: new_path.to_path_buf(),
        });
        Ok(())
    }

    async fn merge(&self, workspace_root: &Path, branch: &str) -> Result<MergeOutcome> {
        self.record(GitCall::Merge {
            workspace_root: workspace_root.to_path_buf(),
            branch: branch.to_owned(),
        });
        let mut state = self.state.lock().unwrap();
        Ok(state.merge_outcomes.pop().unwrap_or(MergeOutcome::Merged))
    }

    async fn remove_worktree(&self, workspace_root: &Path, worktree_path: &Path) -> Result<()> {
        self.record(GitCall::RemoveWorktree {
            workspace_root: workspace_root.to_path_buf(),
            worktree_path: worktree_path.to_path_buf(),
        });
        Ok(())
    }

    async fn delete_branch(&self, workspace_root: &Path, branch: &str, force: bool) -> Result<()> {
        self.record(GitCall::DeleteBranch {
            workspace_root: workspace_root.to_path_buf(),
            branch: branch.to_owned(),
            force,
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FakeSlugGenerator
// ---------------------------------------------------------------------------

/// Test double for [`SlugGenerator`]. Returns a configured response for
/// exact-match first-message strings; unknown messages → `None`,
/// matching how production fallback paths see a "couldn't slugify this"
/// signal.
pub struct FakeSlugGenerator {
    responses: Mutex<HashMap<String, Option<String>>>,
    calls: Mutex<Vec<String>>,
}

impl Default for FakeSlugGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeSlugGenerator {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(HashMap::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Script a response: when `generate(first_message)` is called with
    /// exactly this string, return `slug`. Use `None` to simulate a
    /// failed generation.
    pub fn set_response(&self, first_message: &str, slug: Option<String>) {
        self.responses
            .lock()
            .unwrap()
            .insert(first_message.to_owned(), slug);
    }

    /// Ordered list of every message `generate` was called with.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl SlugGenerator for FakeSlugGenerator {
    async fn generate(&self, first_message: &str) -> Option<String> {
        self.calls.lock().unwrap().push(first_message.to_owned());
        self.responses
            .lock()
            .unwrap()
            .get(first_message)
            .cloned()
            .unwrap_or(None)
    }
}

// ---------------------------------------------------------------------------
// FakeCloseRequestSink / NoopCloseRequestSink
// ---------------------------------------------------------------------------

/// Test double for [`CloseRequestSink`]. Records every `request_close`
/// call so tests can assert "the pump called the sink exactly once with
/// the expected intent."
pub struct FakeCloseRequestSink {
    calls: Mutex<Vec<(SessionId, CloseIntent)>>,
}

impl Default for FakeCloseRequestSink {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeCloseRequestSink {
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<(SessionId, CloseIntent)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl CloseRequestSink for FakeCloseRequestSink {
    async fn request_close(&self, id: SessionId, intent: CloseIntent) {
        self.calls.lock().unwrap().push((id, intent));
    }
}

/// Drop-in sink that does nothing. Used by the intermediate refactor
/// steps — R9 wires the sink through the registry before the pump is
/// taught to call it, and `ActiveSession::start` call sites need
/// *some* implementation in the meantime.
pub struct NoopCloseRequestSink;

#[async_trait]
impl CloseRequestSink for NoopCloseRequestSink {
    async fn request_close(&self, _id: SessionId, _intent: CloseIntent) {
        // Intentionally empty.
    }
}

// ---------------------------------------------------------------------------
// FakeFirstTurnSink / NoopFirstTurnSink
// ---------------------------------------------------------------------------

/// Test double for [`FirstTurnSink`]. Records every
/// `on_first_turn_complete` call so tests can assert "the pump called
/// the sink exactly once with the expected first message."
pub struct FakeFirstTurnSink {
    calls: Mutex<Vec<(SessionId, String)>>,
}

impl Default for FakeFirstTurnSink {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeFirstTurnSink {
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<(SessionId, String)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl FirstTurnSink for FakeFirstTurnSink {
    async fn on_first_turn_complete(&self, id: SessionId, first_message: String) {
        self.calls.lock().unwrap().push((id, first_message));
    }
}

/// Drop-in sink that does nothing. Used by session tests that
/// don't care about slug rename and by intermediate refactor
/// points that predate the production wiring.
pub struct NoopFirstTurnSink;

#[async_trait]
impl FirstTurnSink for NoopFirstTurnSink {
    async fn on_first_turn_complete(&self, _id: SessionId, _first_message: String) {
        // Intentionally empty.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- FakeGit -------------------------------------------------------------

    #[tokio::test]
    async fn assert_repo_passes_by_default_and_records_the_call() {
        let git = FakeGit::new();
        git.assert_repo(Path::new("/tmp/proj")).await.unwrap();
        assert_eq!(
            git.calls(),
            vec![GitCall::AssertRepo(PathBuf::from("/tmp/proj"))]
        );
    }

    #[tokio::test]
    async fn assert_repo_errors_for_paths_marked_non_repo() {
        let git = FakeGit::new();
        git.reject_as_non_repo("/tmp/not-a-repo");
        let err = git
            .assert_repo(Path::new("/tmp/not-a-repo"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a git repository"));
    }

    #[tokio::test]
    async fn current_branch_returns_configured_value_else_main() {
        let git = FakeGit::new();
        git.set_current_branch("/a", "develop");
        assert_eq!(
            git.current_branch(Path::new("/a")).await.unwrap(),
            "develop"
        );
        assert_eq!(git.current_branch(Path::new("/b")).await.unwrap(), "main");
    }

    #[tokio::test]
    async fn current_branch_errors_for_detached_head() {
        let git = FakeGit::new();
        git.mark_detached("/detached");
        let err = git
            .current_branch(Path::new("/detached"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("detached"));
    }

    #[tokio::test]
    async fn status_returns_configured_value_else_clean() {
        let git = FakeGit::new();
        git.set_status("/dirty", WorktreeStatus::Dirty);
        assert_eq!(
            git.status(Path::new("/dirty")).await.unwrap(),
            WorktreeStatus::Dirty
        );
        assert_eq!(
            git.status(Path::new("/clean")).await.unwrap(),
            WorktreeStatus::Clean
        );
    }

    #[tokio::test]
    async fn merge_respects_queued_outcomes_then_defaults_to_merged() {
        let git = FakeGit::new();
        git.enqueue_merge_outcome(MergeOutcome::Conflicts);
        assert_eq!(
            git.merge(Path::new("/w"), "ox/abc").await.unwrap(),
            MergeOutcome::Conflicts
        );
        // Second call exhausts the queue; defaults to Merged.
        assert_eq!(
            git.merge(Path::new("/w"), "ox/abc").await.unwrap(),
            MergeOutcome::Merged
        );
    }

    #[tokio::test]
    async fn call_log_captures_each_operation_in_order() {
        let git = FakeGit::new();
        git.add_worktree(Path::new("/w"), Path::new("/w/.ox/wt"), "ox/abc", "main")
            .await
            .unwrap();
        git.rename_branch(Path::new("/w"), "ox/abc", "ox/slug-abc")
            .await
            .unwrap();
        git.move_worktree(
            Path::new("/w"),
            Path::new("/w/.ox/wt"),
            Path::new("/w/.ox/wt2"),
        )
        .await
        .unwrap();
        git.remove_worktree(Path::new("/w"), Path::new("/w/.ox/wt2"))
            .await
            .unwrap();
        git.delete_branch(Path::new("/w"), "ox/slug-abc", false)
            .await
            .unwrap();

        let calls = git.calls();
        assert_eq!(calls.len(), 5);
        assert!(matches!(calls[0], GitCall::AddWorktree { .. }));
        assert!(matches!(calls[1], GitCall::RenameBranch { .. }));
        assert!(matches!(calls[2], GitCall::MoveWorktree { .. }));
        assert!(matches!(calls[3], GitCall::RemoveWorktree { .. }));
        assert!(matches!(calls[4], GitCall::DeleteBranch { .. }));
    }

    // -- FakeSlugGenerator ---------------------------------------------------

    #[tokio::test]
    async fn slug_generator_returns_scripted_response() {
        let generator = FakeSlugGenerator::new();
        generator.set_response("add login button", Some("add-login-button".into()));
        assert_eq!(
            generator.generate("add login button").await,
            Some("add-login-button".into())
        );
        assert_eq!(generator.calls(), vec!["add login button".to_owned()]);
    }

    #[tokio::test]
    async fn slug_generator_returns_none_for_unscripted_messages() {
        let generator = FakeSlugGenerator::new();
        assert_eq!(generator.generate("no rule for this").await, None);
    }

    // -- FakeCloseRequestSink ------------------------------------------------

    #[tokio::test]
    async fn close_sink_records_intents_in_arrival_order() {
        let sink = FakeCloseRequestSink::new();
        let id = SessionId::new_v4();
        sink.request_close(id, CloseIntent::Merge).await;
        sink.request_close(id, CloseIntent::Abandon { confirm: true })
            .await;

        assert_eq!(
            sink.calls(),
            vec![
                (id, CloseIntent::Merge),
                (id, CloseIntent::Abandon { confirm: true })
            ]
        );
    }

    #[tokio::test]
    async fn noop_sink_is_a_silent_drop() {
        let sink = NoopCloseRequestSink;
        sink.request_close(SessionId::new_v4(), CloseIntent::Merge)
            .await;
        // Nothing observable to assert — the point is that this compiles
        // and drives to completion.
    }

    // -- FakeFirstTurnSink ---------------------------------------------------

    #[tokio::test]
    async fn first_turn_sink_records_calls_in_arrival_order() {
        let sink = FakeFirstTurnSink::new();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        sink.on_first_turn_complete(id1, "add login button".into())
            .await;
        sink.on_first_turn_complete(id2, "fix the crash".into())
            .await;

        assert_eq!(
            sink.calls(),
            vec![
                (id1, "add login button".to_owned()),
                (id2, "fix the crash".to_owned()),
            ]
        );
    }

    #[tokio::test]
    async fn noop_first_turn_sink_drops_silently() {
        let sink = NoopFirstTurnSink;
        sink.on_first_turn_complete(SessionId::new_v4(), "hello".into())
            .await;
    }
}
