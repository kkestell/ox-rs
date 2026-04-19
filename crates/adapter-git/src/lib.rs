//! Production [`Git`](agent_host::Git) adapter that shells out to the
//! `git` CLI. Kept in its own crate for the same reason the other
//! adapters are — swap it for a fake in tests, keep the port crate
//! (`agent-host`) free of process-spawning code.
//!
//! Every method follows the same shape:
//!
//! 1. Build a `tokio::process::Command` for `git` with `-C
//!    <workspace-root>` so we never rely on `std::env::set_current_dir`
//!    (the process is multi-threaded and shared with axum).
//! 2. Capture stdout and stderr.
//! 3. Return `Ok(...)` on success or bubble a rich `anyhow` error with
//!    the full stderr so the caller can log a useful diagnostic.
//!
//! `GIT_TERMINAL_PROMPT=0` is set for every invocation so a
//! misconfigured credential helper cannot freeze the server waiting for
//! a username/password on the terminal.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::ExitStatus;

use agent_host::{Git, MergeOutcome, WorktreeStatus};
use anyhow::{Context, Result, anyhow, bail};
use tokio::process::Command;

/// Stateless handle — all per-call state lives in arguments. Constructed
/// once in the server's startup path and passed around as `Arc<dyn Git>`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CliGit;

impl CliGit {
    pub fn new() -> Self {
        Self
    }
}

/// One shell-out's captured output. `stdout` and `stderr` are decoded as
/// UTF-8 lossily — git output is ASCII in practice, and we only need the
/// text for error messages and single-line parses (branch name, status
/// porcelain), so the lossy conversion is safe.
struct Captured {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

/// Spawn `git` with the given args rooted at `cwd`, capture both
/// streams, and return the raw outcome. The caller decides whether the
/// command's exit status is a failure — some call sites treat non-zero
/// as a recoverable condition (`git merge` on conflict, say) rather
/// than an error to bubble up.
async fn run_git(cwd: &Path, args: &[&str]) -> Result<Captured> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .await
        .with_context(|| format!("spawning git {} in {}", args.join(" "), cwd.display()))?;
    Ok(Captured {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// Run a `git` invocation and bail with a rich error if it exits non-
/// zero. Used everywhere a non-zero exit is always a bug: add_worktree,
/// rename_branch, move_worktree, etc. `merge` and `assert_repo` have
/// their own nuanced handling and call `run_git` directly.
async fn run_git_checked(cwd: &Path, args: &[&str]) -> Result<Captured> {
    let cap = run_git(cwd, args).await?;
    if !cap.status.success() {
        bail!(
            "git {} failed in {} (exit {}): {}",
            args.join(" "),
            cwd.display(),
            cap.status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into()),
            cap.stderr.trim()
        );
    }
    Ok(cap)
}

impl Git for CliGit {
    fn assert_repo<'a>(
        &'a self,
        workspace_root: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // `rev-parse --is-inside-work-tree` prints `true` when run inside
            // a checked-out work tree and errors otherwise — exactly the
            // signal we want. `--show-toplevel` would also work but fails
            // *silently* with an empty stdout in bare repos, which confuses
            // the error message.
            let cap = run_git(workspace_root, &["rev-parse", "--is-inside-work-tree"]).await?;
            if !cap.status.success() {
                bail!(
                    "{} is not a git repository ({})",
                    workspace_root.display(),
                    cap.stderr.trim()
                );
            }
            if cap.stdout.trim() != "true" {
                bail!(
                    "{} is a git directory but not a working copy",
                    workspace_root.display()
                );
            }
            // `symbolic-ref HEAD` fails when HEAD is detached. Surface that
            // as a distinct error so the startup gate can steer the user to
            // `git checkout <branch>` instead of `git init`.
            let head = run_git(workspace_root, &["symbolic-ref", "-q", "HEAD"]).await?;
            if !head.status.success() {
                bail!(
                    "HEAD at {} is detached — check out a branch before launching ox",
                    workspace_root.display()
                );
            }
            Ok(())
        })
    }

    fn current_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let cap = run_git(workspace_root, &["symbolic-ref", "--short", "-q", "HEAD"]).await?;
            if !cap.status.success() {
                bail!(
                    "HEAD at {} is detached: {}",
                    workspace_root.display(),
                    cap.stderr.trim()
                );
            }
            let name = cap.stdout.trim();
            if name.is_empty() {
                bail!(
                    "git reported an empty branch name at {}",
                    workspace_root.display()
                );
            }
            let head = run_git(
                workspace_root,
                &["rev-parse", "--verify", "--quiet", "HEAD"],
            )
            .await?;
            if !head.status.success() {
                bail!(
                    "branch {name} at {} has no commits; create an initial commit before launching ox",
                    workspace_root.display()
                );
            }
            Ok(name.to_owned())
        })
    }

    fn add_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        worktree_path: &'a Path,
        branch: &'a str,
        base_branch: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // `git worktree add` requires the parent directory to exist;
            // create it here so callers don't have to mirror the adapter's
            // layout assumptions.
            if let Some(parent) = worktree_path.parent() {
                tokio::fs::create_dir_all(parent).await.with_context(|| {
                    format!(
                        "creating parent directory for worktree {}",
                        worktree_path.display()
                    )
                })?;
            }
            let worktree_str = worktree_path.to_str().ok_or_else(|| {
                anyhow!(
                    "worktree path {} is not valid UTF-8",
                    worktree_path.display()
                )
            })?;
            run_git_checked(
                workspace_root,
                &["worktree", "add", "-b", branch, worktree_str, base_branch],
            )
            .await?;
            Ok(())
        })
    }

    fn status<'a>(
        &'a self,
        worktree_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<WorktreeStatus>> + Send + 'a>> {
        Box::pin(async move {
            // `--porcelain=v1` emits one line per change; any output means
            // there's something uncommitted. Untracked files count — we
            // don't want to drop a "temporary" file the user is editing.
            let cap = run_git_checked(worktree_path, &["status", "--porcelain"]).await?;
            if cap.stdout.trim().is_empty() {
                Ok(WorktreeStatus::Clean)
            } else {
                Ok(WorktreeStatus::Dirty)
            }
        })
    }

    fn rename_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
        old: &'a str,
        new: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            run_git_checked(workspace_root, &["branch", "-m", old, new]).await?;
            Ok(())
        })
    }

    fn move_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        old_path: &'a Path,
        new_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let old_str = old_path.to_str().ok_or_else(|| {
                anyhow!(
                    "old worktree path {} is not valid UTF-8",
                    old_path.display()
                )
            })?;
            let new_str = new_path.to_str().ok_or_else(|| {
                anyhow!(
                    "new worktree path {} is not valid UTF-8",
                    new_path.display()
                )
            })?;
            // Same parent-dir concern as `add_worktree` — `git worktree move`
            // will not create intermediate directories.
            if let Some(parent) = new_path.parent() {
                tokio::fs::create_dir_all(parent).await.with_context(|| {
                    format!(
                        "creating parent directory for worktree move target {}",
                        new_path.display()
                    )
                })?;
            }
            run_git_checked(workspace_root, &["worktree", "move", old_str, new_str]).await?;
            Ok(())
        })
    }

    fn merge<'a>(
        &'a self,
        workspace_root: &'a Path,
        branch: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<MergeOutcome>> + Send + 'a>> {
        Box::pin(async move {
            // Guard: the main checkout must be clean. `git merge` would
            // happily churn through unrelated work and leave the user with a
            // mess to untangle.
            let main_status = self.status(workspace_root).await?;
            if main_status == WorktreeStatus::Dirty {
                return Ok(MergeOutcome::MainDirty);
            }
            // `--no-edit` keeps the merge non-interactive; `--no-ff`
            // preserves the session branch as a distinct ancestor so the
            // history tells the "this was an ox session" story. Conflicts
            // still surface as a non-zero exit.
            let merge =
                run_git(workspace_root, &["merge", "--no-edit", "--no-ff", branch]).await?;
            if merge.status.success() {
                return Ok(MergeOutcome::Merged);
            }
            // A non-zero exit with MERGE_HEAD present means a conflict —
            // abort so main is left clean. Any other non-zero exit is a hard
            // error (missing branch, broken ref, etc.) and bubbles up.
            let merge_head =
                run_git(workspace_root, &["rev-parse", "--verify", "MERGE_HEAD"]).await?;
            if merge_head.status.success() {
                let _ = run_git(workspace_root, &["merge", "--abort"]).await?;
                return Ok(MergeOutcome::Conflicts);
            }
            bail!(
                "git merge {} failed in {}: {}",
                branch,
                workspace_root.display(),
                merge.stderr.trim()
            );
        })
    }

    fn remove_worktree<'a>(
        &'a self,
        workspace_root: &'a Path,
        worktree_path: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Tolerate a missing worktree dir — abandon can race with the
            // user deleting the directory by hand, and the coordinator
            // shouldn't error on an already-clean state.
            if !worktree_path.exists() {
                // Still prune the administrative bookkeeping; git otherwise
                // keeps a stale entry in `git worktree list`.
                let _ = run_git(workspace_root, &["worktree", "prune"]).await?;
                return Ok(());
            }
            let worktree_str = worktree_path.to_str().ok_or_else(|| {
                anyhow!(
                    "worktree path {} is not valid UTF-8",
                    worktree_path.display()
                )
            })?;
            run_git_checked(
                workspace_root,
                &["worktree", "remove", "--force", worktree_str],
            )
            .await?;
            Ok(())
        })
    }

    fn delete_branch<'a>(
        &'a self,
        workspace_root: &'a Path,
        branch: &'a str,
        force: bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let flag = if force { "-D" } else { "-d" };
            run_git_checked(workspace_root, &["branch", flag, branch]).await?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    //! Integration-style tests against a real `git` binary. Each test
    //! allocates a fresh `tempfile::TempDir`, initializes a repo with a
    //! known layout, and exercises one CliGit method. We assert on the
    //! observable state after the call — `git worktree list`, branch
    //! existence, working-tree cleanliness — rather than on command
    //! invocations, because the goal is to pin adapter behaviour against
    //! whatever `git` returns in practice.
    //!
    //! `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars are set per-commit
    //! instead of running `git config user.name` so the repo's config
    //! file stays pristine and test setup doesn't accidentally depend on
    //! the developer's global gitconfig.
    //!
    //! Tests are annotated `#[tokio::test]` with `flavor = "current_thread"`
    //! so each test runs in its own runtime and stays insulated from the
    //! default multi-thread runtime's scheduling.

    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use tempfile::TempDir;

    /// Layout the tests share: a `root/repo/` main checkout and a
    /// sibling `root/worktrees/` directory for any linked worktrees.
    /// Worktrees must live **outside** the main working tree, otherwise
    /// their parent directory shows up as an untracked entry in `git
    /// status --porcelain` run from main — which would poison every
    /// merge test. Production uses the same shape (the main repo lives
    /// at the user's workspace root and worktrees live under
    /// `~/.ox/workspaces/<slug>/worktrees/`).
    struct Scratch {
        _tmp: TempDir,
        repo: PathBuf,
        worktrees: PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            let tmp = TempDir::new().unwrap();
            let repo = tmp.path().join("repo");
            let worktrees = tmp.path().join("worktrees");
            std::fs::create_dir_all(&repo).unwrap();
            std::fs::create_dir_all(&worktrees).unwrap();
            init_repo(&repo);
            Self {
                _tmp: tmp,
                repo,
                worktrees,
            }
        }

        fn worktree(&self, name: &str) -> PathBuf {
            self.worktrees.join(name)
        }
    }

    /// Run a blocking `git` command during test setup. Setup can't use
    /// `tokio::process::Command` without wrapping every helper in
    /// `#[tokio::test]`, and the asserts on output cleanliness are
    /// simpler with a plain `Output`.
    fn git_setup(cwd: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            // Commit author/committer env comes from a single place so
            // we never accidentally fail on a dev machine missing a
            // `user.email` git config.
            .env("GIT_AUTHOR_NAME", "Ox Test")
            .env("GIT_AUTHOR_EMAIL", "test@ox.local")
            .env("GIT_COMMITTER_NAME", "Ox Test")
            .env("GIT_COMMITTER_EMAIL", "test@ox.local")
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .output()
            .expect("spawn git");
        assert!(
            output.status.success(),
            "git {} failed in {}: {}\n{}",
            args.join(" "),
            cwd.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Create a repo at `root` with a single initial commit on `main`.
    /// Returns the initialized root path so callers can chain helpers.
    fn init_repo(root: &Path) -> PathBuf {
        // `-b main` forces the initial branch name regardless of the
        // host's `init.defaultBranch` setting.
        git_setup(root, &["init", "-b", "main"]);
        std::fs::write(root.join("README.md"), "hello\n").expect("write README");
        git_setup(root, &["add", "README.md"]);
        git_setup(root, &["commit", "-m", "initial"]);
        root.to_path_buf()
    }

    fn init_unborn_repo(root: &Path) -> PathBuf {
        git_setup(root, &["init", "-b", "main"]);
        root.to_path_buf()
    }

    /// Make a second commit modifying `README.md`, used to force merge
    /// scenarios that diverge from the base branch.
    fn commit_change(root: &Path, message: &str, content: &str) {
        std::fs::write(root.join("README.md"), content).expect("write README");
        git_setup(root, &["add", "README.md"]);
        git_setup(root, &["commit", "-m", message]);
    }

    // -- assert_repo / current_branch -----------------------------------

    #[tokio::test]
    async fn assert_repo_succeeds_inside_a_fresh_repo() {
        let s = Scratch::new();
        CliGit::new().assert_repo(&s.repo).await.unwrap();
    }

    #[tokio::test]
    async fn assert_repo_rejects_a_non_repo_directory() {
        let tmp = TempDir::new().unwrap();
        let err = CliGit::new()
            .assert_repo(tmp.path())
            .await
            .expect_err("non-repo should be rejected");
        assert!(
            err.to_string().contains("not a git repository")
                || err.to_string().contains("not a working copy"),
            "unexpected error: {err}"
        );
    }

    /// Shell the current HEAD sha out of a repo as a helper for the
    /// detached-head tests. Going through `std::process` avoids
    /// reintroducing a runtime-dependence in the setup helpers.
    fn head_sha(cwd: &Path) -> String {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        assert!(out.status.success(), "rev-parse HEAD");
        String::from_utf8_lossy(&out.stdout).trim().to_owned()
    }

    #[tokio::test]
    async fn assert_repo_rejects_detached_head() {
        let s = Scratch::new();
        let sha = head_sha(&s.repo);
        git_setup(&s.repo, &["checkout", "--detach", &sha]);
        let err = CliGit::new()
            .assert_repo(&s.repo)
            .await
            .expect_err("detached head should be rejected");
        assert!(
            err.to_string().contains("detached"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn current_branch_returns_the_checked_out_branch_name() {
        let s = Scratch::new();
        let name = CliGit::new()
            .current_branch(&s.repo)
            .await
            .expect("current_branch");
        assert_eq!(name, "main");
    }

    #[tokio::test]
    async fn current_branch_errors_on_detached_head() {
        let s = Scratch::new();
        let sha = head_sha(&s.repo);
        git_setup(&s.repo, &["checkout", "--detach", &sha]);
        let err = CliGit::new()
            .current_branch(&s.repo)
            .await
            .expect_err("detached head should error");
        assert!(
            err.to_string().contains("detached"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn current_branch_errors_on_unborn_branch() {
        let tmp = TempDir::new().unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        init_unborn_repo(&repo);

        let err = CliGit::new()
            .current_branch(&repo)
            .await
            .expect_err("unborn branch should error");
        assert!(
            err.to_string().contains("has no commits"),
            "unexpected error: {err}"
        );
    }

    // -- add_worktree / status ------------------------------------------

    #[tokio::test]
    async fn add_worktree_creates_branch_and_checks_it_out() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();

        // Worktree dir exists and holds the expected README.
        assert!(wt.join("README.md").exists(), "worktree has README");
        // Branch exists in the main repo.
        let branches = std::process::Command::new("git")
            .arg("-C")
            .arg(&s.repo)
            .args(["branch", "--list"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&branches.stdout).contains("ox/abc"),
            "branch ox/abc should exist"
        );
    }

    #[tokio::test]
    async fn status_reports_clean_and_dirty_for_worktree() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        assert_eq!(
            CliGit::new().status(&wt).await.unwrap(),
            WorktreeStatus::Clean,
            "fresh worktree is clean"
        );

        // Introduce an untracked file — porcelain reports it.
        std::fs::write(wt.join("scratch.txt"), "hi").unwrap();
        assert_eq!(
            CliGit::new().status(&wt).await.unwrap(),
            WorktreeStatus::Dirty,
            "untracked file makes status dirty"
        );
    }

    // -- rename_branch / move_worktree ---------------------------------

    #[tokio::test]
    async fn rename_branch_changes_branch_name_in_place() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        CliGit::new()
            .rename_branch(&s.repo, "ox/abc", "ox/slug-abc")
            .await
            .unwrap();

        let branches = std::process::Command::new("git")
            .arg("-C")
            .arg(&s.repo)
            .args(["branch", "--list"])
            .output()
            .unwrap();
        let out = String::from_utf8_lossy(&branches.stdout);
        assert!(out.contains("ox/slug-abc"));
        assert!(!out.contains("ox/abc\n") && !out.contains(" ox/abc "));
    }

    #[tokio::test]
    async fn move_worktree_relocates_checked_out_directory() {
        let s = Scratch::new();
        let wt_old = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt_old, "ox/abc", "main")
            .await
            .unwrap();
        let wt_new = s.worktree("renamed");
        CliGit::new()
            .move_worktree(&s.repo, &wt_old, &wt_new)
            .await
            .unwrap();

        assert!(!wt_old.exists(), "old path gone");
        assert!(wt_new.join("README.md").exists(), "new path populated");
    }

    // -- merge ----------------------------------------------------------

    #[tokio::test]
    async fn merge_fast_forward_reports_merged() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        // Advance the branch in the worktree so merge has something to do.
        commit_change(&wt, "session work", "session-change\n");

        let outcome = CliGit::new().merge(&s.repo, "ox/abc").await.unwrap();
        assert_eq!(outcome, MergeOutcome::Merged);
        // Main now carries the session's change.
        let main = std::fs::read_to_string(s.repo.join("README.md")).unwrap();
        assert_eq!(main, "session-change\n");
    }

    #[tokio::test]
    async fn merge_reports_main_dirty_without_running_merge() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        commit_change(&wt, "session work", "session-change\n");
        // Now dirty the main checkout with an untracked file.
        std::fs::write(s.repo.join("scratch"), "x").unwrap();

        let outcome = CliGit::new().merge(&s.repo, "ox/abc").await.unwrap();
        assert_eq!(outcome, MergeOutcome::MainDirty);
        // Main's working copy is untouched — README wasn't updated.
        let main = std::fs::read_to_string(s.repo.join("README.md")).unwrap();
        assert_eq!(main, "hello\n");
    }

    #[tokio::test]
    async fn merge_conflict_is_aborted_and_reported() {
        let s = Scratch::new();
        // Branch off first, then diverge both sides to force a conflict.
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        commit_change(&wt, "branch side", "branch\n");
        commit_change(&s.repo, "main side", "main\n");

        let outcome = CliGit::new().merge(&s.repo, "ox/abc").await.unwrap();
        assert_eq!(outcome, MergeOutcome::Conflicts);
        // The merge must be fully aborted — main is clean and no merge
        // is in progress. `rev-parse --verify MERGE_HEAD` fails when
        // there's nothing being merged.
        let merge_head = std::process::Command::new("git")
            .arg("-C")
            .arg(&s.repo)
            .args(["rev-parse", "--verify", "MERGE_HEAD"])
            .output()
            .unwrap();
        assert!(!merge_head.status.success(), "MERGE_HEAD must be cleared");
        assert_eq!(
            CliGit::new().status(&s.repo).await.unwrap(),
            WorktreeStatus::Clean,
            "main working copy is left clean after abort"
        );
    }

    // -- remove_worktree / delete_branch -------------------------------

    #[tokio::test]
    async fn remove_worktree_tolerates_missing_directory() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        // User removes the worktree dir by hand.
        std::fs::remove_dir_all(&wt).unwrap();
        // Adapter treats this as success.
        CliGit::new()
            .remove_worktree(&s.repo, &wt)
            .await
            .expect("missing worktree is tolerable");
        // The branch still exists — the adapter only prunes bookkeeping,
        // not branches.
        let branches = std::process::Command::new("git")
            .arg("-C")
            .arg(&s.repo)
            .args(["branch", "--list"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&branches.stdout).contains("ox/abc"),
            "branch untouched"
        );
    }

    #[tokio::test]
    async fn remove_worktree_removes_a_real_worktree_with_force() {
        let s = Scratch::new();
        let wt = s.worktree("abc");
        CliGit::new()
            .add_worktree(&s.repo, &wt, "ox/abc", "main")
            .await
            .unwrap();
        // Dirty the worktree — `--force` should discard the change.
        std::fs::write(wt.join("scratch.txt"), "x").unwrap();
        CliGit::new()
            .remove_worktree(&s.repo, &wt)
            .await
            .expect("remove with force");
        assert!(!wt.exists(), "worktree dir gone");
    }

    #[tokio::test]
    async fn delete_branch_removes_merged_and_force_removes_unmerged() {
        let s = Scratch::new();
        // Merged branch — `-d` suffices.
        git_setup(&s.repo, &["branch", "feature/merged"]);
        CliGit::new()
            .delete_branch(&s.repo, "feature/merged", false)
            .await
            .expect("delete merged branch");

        // Unmerged branch — requires `-D`.
        git_setup(&s.repo, &["branch", "feature/unmerged"]);
        // Make feature/unmerged diverge from main.
        git_setup(&s.repo, &["checkout", "feature/unmerged"]);
        commit_change(&s.repo, "diverge", "different\n");
        git_setup(&s.repo, &["checkout", "main"]);

        // `-d` fails because the branch isn't merged.
        let err = CliGit::new()
            .delete_branch(&s.repo, "feature/unmerged", false)
            .await
            .expect_err("unmerged branch must refuse plain -d");
        assert!(
            err.to_string().contains("not fully merged")
                || err.to_string().contains("The branch")
                || err.to_string().contains("not yet merged"),
            "unexpected error: {err}"
        );
        // `-D` succeeds.
        CliGit::new()
            .delete_branch(&s.repo, "feature/unmerged", true)
            .await
            .expect("force delete unmerged branch");
    }
}
