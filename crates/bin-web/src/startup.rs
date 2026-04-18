//! Workspace validation the server runs **before** it wires the
//! registry or opens a socket.
//!
//! The single public function, [`assert_workspace_ready`], verifies
//! that `workspace_root` is a git repository on a named branch and
//! returns a populated [`WorkspaceContext`] that captures the branch
//! name for later merge/abandon operations. Pulled out of `main.rs`
//! because the git-check branches (non-repo vs. detached HEAD) deserve
//! focused unit tests against `FakeGit` — and the helper also drives
//! the "this isn't a repo" error message that ends up in the user's
//! terminal.

use std::path::Path;

use agent_host::{Git, WorkspaceContext};
use anyhow::{Context, Result};

/// Validate that `workspace_root` is a git work tree on a concrete
/// branch, returning a [`WorkspaceContext`] rooted at it. Errors carry
/// a remediation hint so the user can unstick themselves without
/// reading the source: `git init` for non-repos, `git checkout` for
/// detached HEAD.
pub async fn assert_workspace_ready(
    git: &dyn Git,
    workspace_root: &Path,
) -> Result<WorkspaceContext> {
    git.assert_repo(workspace_root).await.with_context(|| {
        format!(
            "ox workspaces must be git repositories. Run `git init` in {} \
             (or cd into an existing repo) and try again.",
            workspace_root.display()
        )
    })?;
    let base_branch = git.current_branch(workspace_root).await.with_context(|| {
        format!(
            "ox needs a named branch checked out in {}. \
             Run `git checkout <branch>` before launching.",
            workspace_root.display()
        )
    })?;
    Ok(WorkspaceContext::new(
        workspace_root.to_path_buf(),
        base_branch,
    ))
}

#[cfg(test)]
mod tests {
    //! Branch coverage for the startup gate. Each test drives
    //! `assert_workspace_ready` with a `FakeGit` scripted into one of the
    //! three observable outcomes: happy path, non-repo, detached HEAD.
    //! The fake records calls so we can also verify the two trait
    //! methods are invoked in the expected order — callers downstream
    //! rely on `assert_repo` running first, before `current_branch`
    //! might spuriously error for a different reason.

    use super::*;
    use agent_host::fake::{FakeGit, GitCall};
    use std::path::PathBuf;

    #[tokio::test]
    async fn returns_context_for_valid_repo_on_named_branch() {
        let git = FakeGit::new();
        git.set_current_branch("/ws", "develop");
        let ctx = assert_workspace_ready(&git, Path::new("/ws"))
            .await
            .expect("valid repo");
        assert_eq!(ctx.workspace_root, Path::new("/ws"));
        assert_eq!(ctx.base_branch, "develop");
        assert_eq!(
            git.calls(),
            vec![
                GitCall::AssertRepo(PathBuf::from("/ws")),
                GitCall::CurrentBranch(PathBuf::from("/ws")),
            ]
        );
    }

    #[tokio::test]
    async fn surfaces_helpful_error_for_non_repo() {
        let git = FakeGit::new();
        git.reject_as_non_repo("/not-a-repo");
        let err = assert_workspace_ready(&git, Path::new("/not-a-repo"))
            .await
            .expect_err("non-repo should error");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("ox workspaces must be git repositories"),
            "missing user-facing hint: {chain}"
        );
        assert!(
            chain.contains("git init"),
            "hint should reference `git init`: {chain}"
        );
    }

    #[tokio::test]
    async fn surfaces_helpful_error_for_detached_head() {
        let git = FakeGit::new();
        git.mark_detached("/ws");
        let err = assert_workspace_ready(&git, Path::new("/ws"))
            .await
            .expect_err("detached HEAD should error");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("named branch"),
            "missing user-facing hint: {chain}"
        );
        assert!(
            chain.contains("git checkout"),
            "hint should reference `git checkout`: {chain}"
        );
    }

    #[tokio::test]
    async fn short_circuits_on_assert_repo_failure() {
        // Detached check runs via `current_branch`. If `assert_repo`
        // already rejected the path we must never reach `current_branch`,
        // otherwise the detached-HEAD error would shadow the more
        // accurate "not a repo" message.
        let git = FakeGit::new();
        git.reject_as_non_repo("/none");
        let _ = assert_workspace_ready(&git, Path::new("/none")).await;
        assert_eq!(
            git.calls(),
            vec![GitCall::AssertRepo(PathBuf::from("/none"))]
        );
    }

    /// End-to-end sanity check against a real `git` binary. Complements
    /// the `FakeGit`-driven branch tests by confirming the error
    /// surface against `CliGit`'s actual stderr output — the plan calls
    /// out "integration test under `bin-web::tests` that runs the
    /// startup path against a non-repo tempdir." A tempdir is not a
    /// git repo, so the user-facing hint must come through verbatim.
    #[tokio::test]
    async fn real_cli_git_rejects_non_repo_tempdir() {
        use adapter_git::CliGit;
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let git = CliGit::new();
        let err = assert_workspace_ready(&git, tmp.path())
            .await
            .expect_err("tempdir is not a repo");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("ox workspaces must be git repositories"),
            "missing user-facing hint: {chain}"
        );
    }
}
