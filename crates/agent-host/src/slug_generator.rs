//! Port for deriving a short, kebab-case slug from a user's first
//! message to a session.
//!
//! The slug is used to rename `ox/<short-uuid>` to
//! `ox/<slug>-<short-uuid>` after the first turn completes, so sessions
//! get a human-readable name without any user input. Production
//! (`CliSlugGenerator` in `bin-web`) wraps an LLM call with a short
//! timeout and validates the output shape; tests use the
//! `fake::FakeSlugGenerator` double.
//!
//! The trait returns `Option<String>` (not `Result`) so callers can
//! treat all failure modes — timeout, bad shape, network error —
//! uniformly as "skip the rename and keep the short-UUID name." The
//! slug rename is a nice-to-have; it must never block session
//! progress.

use async_trait::async_trait;

#[async_trait]
pub trait SlugGenerator: Send + Sync + 'static {
    /// Produce a kebab-case slug from `first_message`, or `None` if the
    /// message can't be slugified for any reason. Implementations must
    /// enforce their own timeout — the caller doesn't supply one.
    async fn generate(&self, first_message: &str) -> Option<String>;
}
