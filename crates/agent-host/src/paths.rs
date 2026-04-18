//! Path derivations shared across the session lifecycle.
//!
//! The lone export today is [`workspace_slug`], which encodes a main
//! workspace's on-disk path into a single, filesystem-safe directory
//! segment. The slug is used as `~/.ox/workspaces/<slug>/worktrees/<id>`
//! — one directory per workspace root the user launches `ox` from — so
//! we need it to be:
//!
//! - **Canonical.** `/tmp/project` and `/private/tmp/project` (the same
//!   directory on macOS, via a well-known symlink) must map to the
//!   same slug. We call `Path::canonicalize` up front and fall back to
//!   the raw path only when canonicalization fails (e.g. the directory
//!   is being probed before it exists in a test).
//! - **A single path segment.** Separators, colons, and anything else
//!   outside the portable `a-z0-9_.-` set collapse to a single `-` so
//!   the slug stays one directory level deep. Runs of `-` fold and
//!   leading/trailing `-` are trimmed.
//! - **Stable.** No random salt, no hashing. If two launches see the
//!   same canonical path, they get the same slug forever — otherwise
//!   the "restart resumes saved sessions" contract would silently break.

use std::path::Path;

/// Derive a filesystem-safe directory segment from `workspace_root`.
///
/// The function is total and infallible: even a path that canonicalize
/// can't resolve, or one composed entirely of exotic characters, yields
/// the fallback `"workspace"`. Callers combine the returned slug with
/// a fixed layout (`~/.ox/workspaces/<slug>/worktrees/...`) so the
/// fallback still produces a valid path; it just won't collide-resist
/// across truly pathological inputs, which is acceptable for a local
/// single-user tool.
pub fn workspace_slug(workspace_root: &Path) -> String {
    let canonical = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let raw = canonical.to_string_lossy();

    let mut out = String::with_capacity(raw.len());
    // `last_was_sep = true` at the start trims any leading separator run
    // (e.g. the leading `/` on a Unix absolute path) without needing a
    // post-pass.
    let mut last_was_sep = true;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('-');
            last_was_sep = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }

    if out.is_empty() {
        "workspace".to_owned()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugifies_a_simple_absolute_path() {
        // The leading `/` is dropped, separators become `-`, the output
        // is already lowercase ASCII.
        let slug = workspace_slug(Path::new("/home/alice/projects/ox"));
        assert_eq!(slug, "home-alice-projects-ox");
    }

    #[test]
    fn preserves_underscores_and_dots_but_lowercases_letters() {
        let slug = workspace_slug(Path::new("/Users/Bob/my_app.v2"));
        assert_eq!(slug, "users-bob-my_app.v2");
    }

    #[test]
    fn exotic_characters_collapse_to_single_dashes() {
        // Colons (Windows drive letters, timestamps), spaces, and unicode
        // all collapse to `-`. Successive non-slug characters are folded
        // so the output never contains `--`.
        let slug = workspace_slug(Path::new("/a/b c:d/:ö/"));
        assert_eq!(slug, "a-b-c-d");
    }

    #[test]
    fn empty_or_entirely_exotic_paths_fall_back_to_placeholder() {
        // If every character gets collapsed away, the slug can't be empty
        // — `~/.ox/workspaces//worktrees/...` would be broken. `workspace`
        // is an obvious placeholder.
        let slug = workspace_slug(Path::new("/"));
        assert_eq!(slug, "workspace");
        let slug = workspace_slug(Path::new("/:ö/:ä/"));
        assert_eq!(slug, "workspace");
    }

    #[test]
    #[cfg(unix)]
    fn canonical_equivalents_produce_the_same_slug() {
        // The guarantee the rest of the system leans on: if two paths
        // refer to the same canonical location — e.g. `/tmp/x` and
        // `/private/tmp/x` on macOS, or a hand-made symlink on Linux —
        // they must produce the same slug, otherwise a restart would
        // fail to find its own workspaces.json row.
        //
        // To keep the test portable across Linux and macOS we construct
        // our own symlink inside a tempdir and verify that the symlinked
        // path and the real path yield equal slugs.
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).expect("create real");
        let link = tmp.path().join("alias");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        assert_eq!(workspace_slug(&real), workspace_slug(&link));
    }

    #[test]
    #[cfg(unix)]
    fn relative_paths_also_canonicalize_to_the_same_slug() {
        // A workspace root reached via a relative `.` prefix (for
        // example, when the user launches `ox` and the CWD is noted
        // with the trailing `.` segment some shells keep) must slugify
        // identically to the bare absolute form.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dotted = tmp.path().join(".");
        assert_eq!(workspace_slug(tmp.path()), workspace_slug(&dotted));
    }

    #[test]
    fn nonexistent_paths_still_produce_stable_slugs() {
        // `canonicalize` fails on a path that doesn't exist; the helper
        // falls back to the raw path so callers can derive the slug
        // before the directory is created.
        let slug = workspace_slug(Path::new("/nope/definitely/does/not/exist"));
        assert_eq!(slug, "nope-definitely-does-not-exist");
    }
}
