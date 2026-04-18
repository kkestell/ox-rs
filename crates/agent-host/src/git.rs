//! Port for git operations the lifecycle coordinator orchestrates.
//!
//! The trait lives in `agent-host` (not `app`) because only the host
//! layer talks to git — sessions, worktrees, and merges are host
//! concerns. Mirrors how `AgentSpawner` is also in `agent-host`.
//!
//! Methods are `async` + boxed via `#[async_trait]` so callers can use
//! `Arc<dyn Git>` as a heterogeneous handle (production: `CliGit`;
//! tests: `fake::FakeGit`).

use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;

/// Worktree cleanliness as reported by `git status --porcelain`. A
/// worktree is `Dirty` if there's anything staged, unstaged, or
/// untracked — the merge/abandon flows refuse to destroy work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorktreeStatus {
    Clean,
    Dirty,
}

/// Result of a `git merge` attempt. Callers distinguish "happy path"
/// from the two recoverable failure modes (dirty main checkout, merge
/// conflict) by variant. Any other error — a bad command, a filesystem
/// problem — surfaces through the `Result::Err` that wraps this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOutcome {
    /// Merge succeeded; the base branch now includes the session branch.
    Merged,
    /// The merge started but conflicted. The adapter runs `git merge
    /// --abort` before returning so the main checkout is left clean.
    Conflicts,
    /// The main working copy had uncommitted changes at merge time. We
    /// refuse to merge in this state — running `git merge` would risk
    /// scribbling over the user's in-progress work.
    MainDirty,
}

#[async_trait]
pub trait Git: Send + Sync + 'static {
    /// Fail fast if `workspace_root` is not a git working copy, or if
    /// HEAD is detached. The startup gate uses this to reject non-repo
    /// launches before any other setup runs.
    async fn assert_repo(&self, workspace_root: &Path) -> Result<()>;

    /// Short name of the workspace's current branch (e.g. `"main"`).
    /// Errors if HEAD is detached or the command fails.
    async fn current_branch(&self, workspace_root: &Path) -> Result<String>;

    /// `git worktree add -b <branch> <worktree_path> <base_branch>` —
    /// creates the branch and checks it out in one step.
    async fn add_worktree(
        &self,
        workspace_root: &Path,
        worktree_path: &Path,
        branch: &str,
        base_branch: &str,
    ) -> Result<()>;

    /// `git status --porcelain` inside `worktree_path`. Any output →
    /// `Dirty`; empty output → `Clean`.
    async fn status(&self, worktree_path: &Path) -> Result<WorktreeStatus>;

    /// `git branch -m <old> <new>`. Works on a branch checked out in a
    /// worktree — updates the worktree's HEAD in place, no reset needed.
    async fn rename_branch(&self, workspace_root: &Path, old: &str, new: &str) -> Result<()>;

    /// `git worktree move <old> <new>`. Requires the worktree to be
    /// clean; callers ensure that (slug rename runs right after the
    /// first turn, before the user can dirty it).
    async fn move_worktree(
        &self,
        workspace_root: &Path,
        old_path: &Path,
        new_path: &Path,
    ) -> Result<()>;

    /// Run `git merge <branch>` inside `workspace_root` (the main
    /// checkout — never a worktree, or it would merge the wrong way).
    /// The implementation inspects the main checkout for cleanliness
    /// before starting, and aborts a conflicting merge before returning.
    async fn merge(&self, workspace_root: &Path, branch: &str) -> Result<MergeOutcome>;

    /// `git worktree remove --force <path>`. `--force` lets us tolerate
    /// the worktree being already gone (removed out-of-band) as well as
    /// discarding uncommitted changes on confirmed abandon.
    async fn remove_worktree(&self, workspace_root: &Path, worktree_path: &Path) -> Result<()>;

    /// `git branch -d <branch>` (or `-D` when `force=true`). Used to
    /// delete the session branch after merge (unforced) or abandon
    /// (forced on a dirty confirm).
    async fn delete_branch(&self, workspace_root: &Path, branch: &str, force: bool) -> Result<()>;
}
