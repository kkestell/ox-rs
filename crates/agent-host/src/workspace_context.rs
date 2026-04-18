//! Workspace-level information the lifecycle coordinator carries around
//! while orchestrating git operations.
//!
//! Every merge, every worktree-add, and every branch rename happens
//! against the same pair: the main workspace root (the user's CWD at
//! launch) and the branch it was on at startup (the base branch). Bundle
//! those together so the pair travels as one value instead of two loose
//! fields scattered across the lifecycle coordinator and registry.
//!
//! Constructed once in `bin-web::run` after the startup git checks; cloned
//! freely afterwards (both fields are owned, but clones are cheap —
//! `PathBuf` + `String`).

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct WorkspaceContext {
    /// The main repository working-copy root the server was launched in.
    /// Session worktrees fork from this checkout; merges run against it.
    pub workspace_root: PathBuf,
    /// The branch the workspace was on at startup. Session branches are
    /// created from this base branch, and merges target it. Kept as a
    /// short name (the output of `git symbolic-ref --short HEAD`), not a
    /// fully-qualified ref.
    pub base_branch: String,
}

impl WorkspaceContext {
    pub fn new(workspace_root: PathBuf, base_branch: String) -> Self {
        Self {
            workspace_root,
            base_branch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_populates_fields_verbatim() {
        let ctx = WorkspaceContext::new(PathBuf::from("/tmp/proj"), "main".into());
        assert_eq!(ctx.workspace_root, PathBuf::from("/tmp/proj"));
        assert_eq!(ctx.base_branch, "main");
    }

    #[test]
    fn clone_produces_an_independent_copy() {
        // Cloning is cheap and used all over the coordinator; the field
        // types are owned, so this is a real-data copy, not a handle
        // share. Assert that mutating one does not affect the other.
        let original = WorkspaceContext::new(PathBuf::from("/a"), "main".into());
        let mut clone = original.clone();
        clone.base_branch = "feature".into();
        assert_eq!(original.base_branch, "main");
        assert_eq!(clone.base_branch, "feature");
    }
}
