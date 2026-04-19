//! Port for git operations the lifecycle coordinator orchestrates.
//!
//! The trait lives in `agent-host` (not `app`) because only the host
//! layer talks to git — sessions, worktrees, and merges are host
//! concerns. Mirrors how `AgentSpawner` is also in `agent-host`.
//!
//! Methods return `Pin<Box<dyn Future>>` so callers can use
//! `Arc<dyn Git>` as a heterogeneous handle (production: `CliGit`;
//! tests: `fake::FakeGit`).

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use anyhow::Result;

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

pub trait Git: Send + Sync + 'static {
    /// Fail fast if `workspace_root` is not a git working copy, or if
    /// HEAD is detached. The startup gate uses this to reject non-repo
    /// launches before any other setup runs.
    fn assert_repo<'a>(
        &'a self,
        workspace_root: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Short name of the workspace's current branch (e.g. `"main"`).
    /// Errors if HEAD is detached or the command fails.
    fn current_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

    /// `git worktree add -b <branch> <worktree_path> <base_branch>` —
    /// creates the branch and checks it out in one step.
    fn add_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        worktree_path: &'a Path,
        branch: &'a str,
        base_branch: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// `git status --porcelain` inside `worktree_path`. Any output →
    /// `Dirty`; empty output → `Clean`.
    fn status<'a>(
        &'a self,
        worktree_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<WorktreeStatus>> + Send + 'a>>;

    /// `git branch -m <old> <new>`. Works on a branch checked out in a
    /// worktree — updates the worktree's HEAD in place, no reset needed.
    fn rename_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
        old: &'a str,
        new: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// `git worktree move <old> <new>`. Requires the worktree to be
    /// clean; callers ensure that (slug rename runs right after the
    /// first turn, before the user can dirty it).
    fn move_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        old_path: &'a Path,
        new_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Run `git merge <branch>` inside `workspace_root` (the main
    /// checkout — never a worktree, or it would merge the wrong way).
    /// The implementation inspects the main checkout for cleanliness
    /// before starting, and aborts a conflicting merge before returning.
    fn merge<'a>(
        &'a self,
        workspace_root: &'a Path,
        branch: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<MergeOutcome>> + Send + 'a>>;

    /// `git worktree remove --force <path>`. `--force` lets us tolerate
    /// the worktree being already gone (removed out-of-band) as well as
    /// discarding uncommitted changes on confirmed abandon.
    fn remove_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        worktree_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// `git branch -d <branch>` (or `-D` when `force=true`). Used to
    /// delete the session branch after merge (unforced) or abandon
    /// (forced on a dirty confirm).
    fn delete_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
        branch: &'a str,
        force: bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}
