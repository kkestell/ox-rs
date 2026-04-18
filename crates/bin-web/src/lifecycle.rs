//! Session lifecycle coordinator.
//!
//! `SessionLifecycle` owns all the moving parts that orchestrate a
//! session across its birth-to-close arc:
//!
//! - the [`Git`] adapter (worktree + branch operations),
//! - the [`SlugGenerator`] (turns the first user message into a slug
//!   used for renaming the worktree dir + branch),
//! - the [`DiskSessionStore`] (deletes the session JSON when merge /
//!   abandon completes),
//! - the [`WorkspaceContext`] (main workspace root + base branch),
//! - per-session close locks so concurrent merge / abandon requests
//!   for the same session serialize cleanly.
//!
//! The coordinator also implements [`CloseRequestSink`], which is how
//! the per-session pump in `session.rs` routes
//! `AgentEvent::RequestClose` frames back here without holding a
//! strong back-pointer to the registry.
//!
//! # Why a separate type
//!
//! Piling merge / abandon / close-routing / slug-rename / session-JSON
//! deletion onto `SessionRegistry` would turn the registry into the
//! central coupling point for every future workspace feature. The
//! registry's three existing concerns (sessions map, layout store,
//! spawner) stay stable; the coordinator mutates the sessions map in
//! response to lifecycle events.
//!
//! # Breaking the init cycle
//!
//! `SessionLifecycle` needs the registry (to `remove` sessions after a
//! successful merge / abandon) and the registry needs the lifecycle
//! (as its `CloseRequestSink`). To break the cycle, construction is
//! two-phase:
//!
//! 1. Build the lifecycle with its registry-weak-ref empty.
//! 2. Build the registry, handing it `lifecycle.clone() as Arc<dyn
//!    CloseRequestSink>`.
//! 3. Call `lifecycle.set_registry(Arc::downgrade(&registry))` so the
//!    coordinator can reach the registry through a `Weak`.
//!
//! Every lifecycle method that touches the registry calls
//! [`SessionLifecycle::registry`], which upgrades the weak-ref and
//! returns `None` only when the registry has been dropped (a shutdown-
//! only condition).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use adapter_storage::DiskSessionStore;
use agent_host::{
    CloseRequestSink, FirstTurnSink, Git, MergeOutcome, SlugGenerator, WorkspaceContext,
    WorktreeStatus, workspace_slug,
};
use anyhow::{Context, Result, anyhow};
use app::SessionStore;
use async_trait::async_trait;
use domain::{CloseIntent, SessionId};

use crate::registry::SessionRegistry;

/// Per-session close-in-flight flag. The coordinator takes the lock
/// for the whole merge / abandon / close-request flow; concurrent
/// callers observe `CloseState::Closing` and drop their request. The
/// enum exists (vs a bare `bool`) so later steps can add richer state
/// — e.g. `Closing { intent: CloseIntent, started_at: Instant }` for
/// debug surfaces — without churning every call site.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseState {
    Idle,
    Closing,
}

/// Reasons the merge / abandon flow can refuse to close a session.
///
/// Each variant maps to a specific HTTP status + structured reason
/// string in the route handler; keeping them as typed variants (rather
/// than a free-form `anyhow::Error`) keeps that mapping exhaustive at
/// the match site.
///
/// `Internal` is the catch-all for filesystem / git-subprocess failures
/// that shouldn't happen in a healthy deployment (e.g. the git binary
/// vanished mid-request). Its payload is a human-readable message the
/// handler logs and surfaces in a 500 response.
#[derive(Debug)]
pub enum MergeRejection {
    /// The registry has no session with this id.
    NotFound,
    /// Another close is already in flight for this session.
    AlreadyClosing,
    /// The agent is mid-turn — closing now would strand the tool call
    /// on a disappearing worktree.
    TurnInProgress,
    /// The session's worktree has uncommitted or untracked work. For
    /// merge this is always a hard refusal; for abandon it is soft
    /// unless the caller passes `confirm=true`.
    WorktreeDirty,
    /// The main workspace checkout has uncommitted changes; the merge
    /// would risk scribbling over the user's in-progress work.
    MainDirty,
    /// The merge started but conflicted. The adapter has already run
    /// `git merge --abort`, so the main checkout is clean.
    Conflict,
    /// Any other error (git subprocess failed, disk I/O failed, etc.).
    /// Carries a human-readable message for logging / 500 response.
    Internal(String),
}

pub struct SessionLifecycle {
    /// Git adapter — shelled-out `git` in production, `FakeGit` in
    /// tests. Held behind `Arc<dyn Git>` so swapping implementations
    /// does not ripple through the coordinator's public surface.
    git: Arc<dyn Git>,
    /// Slug generator — an `OpenRouterProvider`-backed call in
    /// production, `FakeSlugGenerator` in tests.
    slug_generator: Arc<dyn SlugGenerator>,
    /// Session store — used **only** by the coordinator to delete the
    /// on-disk session JSON after a successful merge / abandon. The
    /// agent subprocess owns all other session-store reads and writes.
    /// A concrete `Arc<DiskSessionStore>` instead of `Arc<dyn
    /// SessionStore>` because the `SessionStore` trait uses
    /// return-position `impl Trait` and so is not dyn-compatible; the
    /// coordinator has only one production impl and future tests can
    /// introduce a narrower port if they need to fake deletion.
    session_store: Arc<DiskSessionStore>,
    /// Main workspace root + base branch, snapshotted at startup. Every
    /// worktree-add and merge runs against these values; they do not
    /// change over the server's lifetime.
    workspace: WorkspaceContext,
    /// Per-session close-in-flight flags. A fresh entry is inserted
    /// when a close begins and removed on the `CloseGuard`'s drop.
    /// Concurrent merge/abandon calls for the same id observe a
    /// `Closing` entry and return `AlreadyClosing`.
    closing: Mutex<HashMap<SessionId, CloseState>>,
    /// Backreference to the registry, set after the registry is
    /// constructed (two-phase init — see module docs). `Weak` so the
    /// coordinator does not keep the registry alive past shutdown; the
    /// registry is the graph's root.
    registry: OnceLock<Weak<SessionRegistry>>,
}

impl SessionLifecycle {
    /// Build the coordinator. `set_registry` must be called once
    /// afterwards to complete the back-reference — until then, methods
    /// that touch the registry will no-op.
    pub fn new(
        git: Arc<dyn Git>,
        slug_generator: Arc<dyn SlugGenerator>,
        session_store: Arc<DiskSessionStore>,
        workspace: WorkspaceContext,
    ) -> Arc<Self> {
        Arc::new(Self {
            git,
            slug_generator,
            session_store,
            workspace,
            closing: Mutex::new(HashMap::new()),
            registry: OnceLock::new(),
        })
    }

    /// Install the registry back-reference. Second calls are a silent
    /// no-op (the `OnceLock` is write-once); the server's startup path
    /// calls this exactly once after the registry is built.
    pub fn set_registry(&self, registry: Weak<SessionRegistry>) {
        let _ = self.registry.set(registry);
    }

    /// Upgrade the stored `Weak<SessionRegistry>` to an `Arc`, or
    /// `None` if the registry has been dropped. Every method that
    /// mutates the registry funnels through this so a post-shutdown
    /// lifecycle call is a no-op instead of a panic.
    fn registry(&self) -> Option<Arc<SessionRegistry>> {
        self.registry.get().and_then(|weak| weak.upgrade())
    }

    /// Create a fresh session backed by a dedicated git worktree on its
    /// own `ox/<short>` branch, then spawn an `ox-agent` pointed at it.
    ///
    /// Steps:
    /// 1. Pre-allocate a `SessionId` so the worktree directory, branch
    ///    name, and eventual `{id}.json` file all share one id.
    /// 2. Slugify the main workspace root to a stable, filesystem-safe
    ///    segment so concurrent `ox` launches from different projects
    ///    never collide on `~/.ox/workspaces/…`.
    /// 3. Create the worktree checkout at
    ///    `~/.ox/workspaces/<workspace-slug>/worktrees/<short>/` on a
    ///    branch named `ox/<short>`, branched from `workspace.base_branch`.
    /// 4. Spawn the agent via the registry with the worktree as its
    ///    `workspace_root` and the pre-allocated id as `config.session_id`.
    ///
    /// A failed `git worktree add` leaves nothing on disk (the call is
    /// atomic from the caller's perspective). A failed spawn leaves the
    /// worktree and branch behind — the user can clean those up with
    /// `git worktree remove` / `git branch -D` if desired; we do not try
    /// to roll back partial work here because a retry with a fresh id is
    /// cheaper than implementing two-phase undo.
    pub async fn create_and_spawn(&self) -> Result<SessionId> {
        let registry = self
            .registry()
            .ok_or_else(|| anyhow!("registry has been dropped"))?;

        let session_id = SessionId::new_v4();
        let short = short_prefix(session_id);
        let slug = workspace_slug(&self.workspace.workspace_root);

        let home = dirs::home_dir().context("resolving the user's home directory")?;
        let worktree_path: PathBuf = home
            .join(".ox")
            .join("workspaces")
            .join(&slug)
            .join("worktrees")
            .join(&short);

        let branch = format!("ox/{short}");
        self.git
            .add_worktree(
                &self.workspace.workspace_root,
                &worktree_path,
                &branch,
                &self.workspace.base_branch,
            )
            .await
            .with_context(|| {
                format!(
                    "creating worktree at {} on branch {branch}",
                    worktree_path.display()
                )
            })?;

        registry
            .spawn_for_worktree(worktree_path, session_id, None)
            .await
    }

    /// Merge `id`'s session branch back into the main workspace's base
    /// branch and tear down the session. All-or-nothing from the
    /// caller's perspective: either every step succeeds and the session
    /// is gone, or nothing observable changes beyond the rejection.
    ///
    /// Steps, in order (matches the plan's merge contract):
    /// 1. Session exists in the registry (else `NotFound`).
    /// 2. No other close is in flight (else `AlreadyClosing`). Acquiring
    ///    the close lock is done via [`begin_close`], which returns an
    ///    RAII guard that releases the lock on drop.
    /// 3. The agent is idle — no turn in flight (else `TurnInProgress`).
    ///    Checked against [`ActiveSession::is_turn_in_progress`], which
    ///    reflects the pump's view of the `SessionRuntime` state machine.
    /// 4. Worktree is clean (else `WorktreeDirty`). Dirty state means
    ///    the agent has uncommitted edits; merging silently would
    ///    discard them when we remove the worktree below.
    /// 5. `git merge` against the main checkout — branching on the
    ///    three recoverable outcomes (`MainDirty`, `Conflicts`,
    ///    `Merged`).
    /// 6. Remove the worktree, delete the branch (unforced; merge
    ///    succeeded so branch is already merged), delete the session
    ///    JSON, drop from the registry.
    ///
    /// Note the TOCTOU window between (3) and (5): a new `POST
    /// /messages` can race in after the turn-in-progress check clears
    /// and land a `SendMessage` on the agent's stdin mid-merge. We
    /// accept the race per the plan — the common case (idle session,
    /// user clicks merge) is still guarded; the racy case leaves the
    /// session closed with an interrupted final turn, which matches the
    /// user's intent anyway.
    pub async fn merge(&self, id: SessionId) -> Result<(), MergeRejection> {
        let (session, worktree_path, branch, _guard) = self.begin_close_flow(id).await?;

        let status = self
            .git
            .status(&worktree_path)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git status failed: {err:#}")))?;
        if status == WorktreeStatus::Dirty {
            return Err(MergeRejection::WorktreeDirty);
        }

        let outcome = self
            .git
            .merge(&self.workspace.workspace_root, &branch)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git merge failed: {err:#}")))?;
        match outcome {
            MergeOutcome::Merged => {}
            MergeOutcome::MainDirty => return Err(MergeRejection::MainDirty),
            MergeOutcome::Conflicts => return Err(MergeRejection::Conflict),
        }

        // Drop our handle on the session *before* we drop it from the
        // registry — otherwise the registry's internal Arc count still
        // holds the session alive after remove(), postponing the agent
        // kill until our Arc goes out of scope at the end of the
        // function.
        drop(session);

        self.teardown(id, &branch, &worktree_path, /* force_branch */ false)
            .await
    }

    /// Discard `id`'s session work: remove the worktree, delete the
    /// branch without merging, drop the session. Parallel to [`merge`]
    /// except no `git merge` call and the dirty-worktree check is
    /// soft-gated by `confirm`.
    ///
    /// If `confirm=false` and the worktree has uncommitted work,
    /// returns [`MergeRejection::WorktreeDirty`] so the client can
    /// prompt the user. On `confirm=true`, the worktree + branch are
    /// force-removed regardless of cleanliness; the user has explicitly
    /// asked to discard the work.
    ///
    /// `delete_branch` is always called with `force=true` for abandon:
    /// an abandoned session branch may have commits that never landed
    /// on main, so `git branch -d` (the unforced version) would refuse.
    pub async fn abandon(&self, id: SessionId, confirm: bool) -> Result<(), MergeRejection> {
        let (session, worktree_path, branch, _guard) = self.begin_close_flow(id).await?;

        let status = self
            .git
            .status(&worktree_path)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git status failed: {err:#}")))?;
        if status == WorktreeStatus::Dirty && !confirm {
            return Err(MergeRejection::WorktreeDirty);
        }

        drop(session);

        self.teardown(id, &branch, &worktree_path, /* force_branch */ true)
            .await
    }

    /// Shared preamble for merge and abandon: resolve the session,
    /// claim the close lock, confirm the agent is idle, and load the
    /// session's worktree path + branch name so the caller can continue
    /// with the status / merge / teardown steps.
    ///
    /// The returned `Arc<ActiveSession>` keeps the session alive until
    /// the caller drops it; callers should drop it right before the
    /// final `registry.remove(id)` so the agent kill is prompt rather
    /// than waiting for the function's own Arc to go out of scope.
    async fn begin_close_flow(
        &self,
        id: SessionId,
    ) -> Result<
        (
            Arc<crate::session::ActiveSession>,
            PathBuf,
            String,
            CloseGuard<'_>,
        ),
        MergeRejection,
    > {
        let registry = self
            .registry()
            .ok_or_else(|| MergeRejection::Internal("registry has been dropped".to_owned()))?;
        let session = registry.get(id).ok_or(MergeRejection::NotFound)?;

        let guard = self.begin_close(id).ok_or(MergeRejection::AlreadyClosing)?;

        // The turn-idle check must run *after* we own the close lock —
        // otherwise a concurrent merge/abandon could see us as idle,
        // start closing too, and we'd race past the check together.
        if session.is_turn_in_progress() {
            return Err(MergeRejection::TurnInProgress);
        }

        // The session JSON is the authoritative source for worktree
        // location: the slug-rename flow updates it when the worktree
        // directory moves, and the path stored there is what the agent
        // subprocess actually operates under.
        let saved = self
            .session_store
            .try_load(id)
            .await
            .map_err(|err| MergeRejection::Internal(format!("loading session: {err:#}")))?
            .ok_or_else(|| {
                MergeRejection::Internal(format!(
                    "session {id} has no on-disk record; cannot determine worktree path"
                ))
            })?;
        let worktree_path = saved.worktree_path.clone();

        // Discover the branch from the worktree's HEAD — handles both
        // pre- and post-slug-rename names (`ox/<short>` vs
        // `ox/<slug>-<short>`) without the coordinator having to
        // remember which state any given session is in.
        let branch = self
            .git
            .current_branch(&worktree_path)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git current-branch: {err:#}")))?;

        Ok((session, worktree_path, branch, guard))
    }

    /// Shared teardown for merge / abandon after the pre-flight checks
    /// and any git work have succeeded: remove the worktree, delete the
    /// branch, delete the session JSON, drop from the registry. Any
    /// step failing surfaces as `Internal` — at this point we've
    /// already started destroying state, so there's no "recoverable"
    /// path to offer the caller.
    async fn teardown(
        &self,
        id: SessionId,
        branch: &str,
        worktree_path: &Path,
        force_branch: bool,
    ) -> Result<(), MergeRejection> {
        self.git
            .remove_worktree(&self.workspace.workspace_root, worktree_path)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git worktree remove: {err:#}")))?;

        self.git
            .delete_branch(&self.workspace.workspace_root, branch, force_branch)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git branch delete: {err:#}")))?;

        self.session_store
            .delete(id)
            .await
            .map_err(|err| MergeRejection::Internal(format!("delete session JSON: {err:#}")))?;

        if let Some(reg) = self.registry() {
            reg.remove(id);
        }

        Ok(())
    }

    /// Claim the per-session close lock. Returns `Some(guard)` if no
    /// other close is in flight, or `None` if the session is already
    /// closing. The guard releases the lock on drop — callers don't
    /// have to remember, and early returns from later rejections
    /// unwind correctly.
    fn begin_close(&self, id: SessionId) -> Option<CloseGuard<'_>> {
        let mut map = self.closing.lock().expect("close map poisoned");
        match map.get(&id) {
            Some(CloseState::Closing) => None,
            _ => {
                map.insert(id, CloseState::Closing);
                Some(CloseGuard {
                    closing: &self.closing,
                    id,
                })
            }
        }
    }
}

/// RAII release of the close lock entry. Removes the session's
/// `CloseState::Closing` record when dropped so the next
/// merge/abandon call on the same id isn't rejected.
///
/// The guard removes the entry on *any* drop — success or failure.
/// That's correct: on success the session is also removed from the
/// registry so the map entry has nothing to guard; on failure we want
/// the lock released so the caller can retry.
struct CloseGuard<'a> {
    closing: &'a Mutex<HashMap<SessionId, CloseState>>,
    id: SessionId,
}

impl Drop for CloseGuard<'_> {
    fn drop(&mut self) {
        let mut map = self.closing.lock().expect("close map poisoned");
        map.remove(&self.id);
    }
}

/// The first 8 hex characters of a UUID — used verbatim as the short id
/// that names the worktree directory and the `ox/<short>` branch. 8 hex
/// chars gives 4.3B possibilities, comfortable for a per-user tool where
/// a collision is locally observable (worktree-add would fail).
fn short_prefix(id: SessionId) -> String {
    let s = id.to_string();
    s[..8].to_owned()
}

/// Dispatch a host-side close request from the agent's `merge` /
/// `abandon` tools. The per-session close lock is acquired inside
/// [`SessionLifecycle::merge`] / [`SessionLifecycle::abandon`] via
/// [`SessionLifecycle::begin_close_flow`]; an already-closing session
/// surfaces as `MergeRejection::AlreadyClosing` here.
///
/// On success the session has been torn down (worktree removed, branch
/// deleted, JSON erased, registry entry dropped) and there's nothing
/// more to do. On rejection we broadcast an `AgentEvent::Error` to the
/// session so the running SSE subscriber sees why the close was
/// refused, and leave the session in place — the agent is about to
/// exit after emitting `RequestClose`, so the registry entry will go
/// dead naturally when the pump observes EOF. Users can still inspect
/// the dead pane in the UI for triage.
///
/// `NotFound` is a legitimate "this race resolved itself" — the
/// agent sent `RequestClose` just as a concurrent HTTP merge/abandon
/// tore the session down. Log and return without broadcasting
/// (there's no session channel to broadcast on).
#[async_trait]
impl CloseRequestSink for SessionLifecycle {
    async fn request_close(&self, id: SessionId, intent: CloseIntent) {
        let result = match intent {
            CloseIntent::Merge => self.merge(id).await,
            CloseIntent::Abandon { confirm } => self.abandon(id, confirm).await,
        };

        let Err(rejection) = result else {
            return;
        };

        let (intent_label, reason) = match intent {
            CloseIntent::Merge => ("merge", rejection_message(&rejection)),
            CloseIntent::Abandon { .. } => ("abandon", rejection_message(&rejection)),
        };
        let message = format!("{intent_label} refused: {reason}");

        // `NotFound` means the registry dropped the session already;
        // there is no broadcast channel to reach. Just log and return.
        if matches!(rejection, MergeRejection::NotFound) {
            eprintln!("ox: close for session {id} skipped: {message}");
            return;
        }

        if let Some(registry) = self.registry()
            && let Some(session) = registry.get(id)
        {
            session.broadcast_error(message.clone());
        }
        eprintln!("ox: {message} (session {id})");
    }
}

/// Human-readable reason for a `MergeRejection`, used in the
/// broadcast-Error message the lifecycle emits on agent-initiated
/// close failures. Route handlers do not use this helper — they map
/// rejections to structured HTTP bodies the frontend keys off of.
fn rejection_message(rej: &MergeRejection) -> String {
    match rej {
        MergeRejection::NotFound => "session no longer exists".to_owned(),
        MergeRejection::AlreadyClosing => "another close is already in progress".to_owned(),
        MergeRejection::TurnInProgress => "cannot close while a turn is still running".to_owned(),
        MergeRejection::WorktreeDirty => "worktree has uncommitted changes".to_owned(),
        MergeRejection::MainDirty => "main workspace has uncommitted changes".to_owned(),
        MergeRejection::Conflict => "merge produced conflicts; resolve by hand".to_owned(),
        MergeRejection::Internal(msg) => msg.clone(),
    }
}

/// Slug-rename hook. On the first `TurnComplete` of a fresh session the
/// pump calls this with the text of the first user message. We ask the
/// slug generator for a kebab-case label, rename the git branch from
/// `ox/<short>` to `ox/<slug>-<short>`, move the worktree directory to
/// match, persist the new path in the session JSON, and respawn the
/// agent pointed at the new worktree path.
///
/// Every step is best-effort: a generator timeout, a branch-rename
/// conflict, or a respawn failure all log and return. The session stays
/// on its original short-UUID name but remains fully functional.
///
/// Idempotency: the pump CAS-flips `fresh` before calling so this hook
/// fires at most once per `ActiveSession`. On a process restart, the
/// resumed session starts with `fresh=false`, so the pump's guard
/// prevents a re-fire without any work here.
#[async_trait]
impl FirstTurnSink for SessionLifecycle {
    async fn on_first_turn_complete(&self, id: SessionId, first_message: String) {
        let slug = match self.slug_generator.generate(&first_message).await {
            Some(s) => s,
            None => {
                // Generator timed out / returned invalid shape / errored.
                // Keep the short-UUID name.
                return;
            }
        };

        let short = short_prefix(id);
        let old_branch = format!("ox/{short}");
        let new_branch = format!("ox/{slug}-{short}");

        // Load the session to find its current worktree path. The agent
        // saves the session JSON on every TurnComplete, so by the time
        // the hook fires the file must exist — but we still treat
        // missing-file as a skippable failure rather than a panic.
        let session = match self.session_store.try_load(id).await {
            Ok(Some(s)) => s,
            Ok(None) => {
                eprintln!("ox: slug rename skipped for session {id}: no session file on disk");
                return;
            }
            Err(err) => {
                eprintln!("ox: slug rename: failed to load session {id}: {err:#}");
                return;
            }
        };
        let old_worktree_path = session.worktree_path.clone();
        // The new worktree sits in the same parent directory (the
        // per-workspace `worktrees/` folder); only the leaf name changes
        // from `<short>` to `<slug>-<short>`.
        let new_worktree_path = match old_worktree_path.parent() {
            Some(parent) => parent.join(format!("{slug}-{short}")),
            None => {
                eprintln!(
                    "ox: slug rename: worktree path {} has no parent; skipping",
                    old_worktree_path.display()
                );
                return;
            }
        };

        if let Err(err) = self
            .git
            .rename_branch(&self.workspace.workspace_root, &old_branch, &new_branch)
            .await
        {
            eprintln!(
                "ox: slug rename: renaming branch {old_branch} → {new_branch} failed: {err:#}"
            );
            return;
        }

        if let Err(err) = self
            .git
            .move_worktree(
                &self.workspace.workspace_root,
                &old_worktree_path,
                &new_worktree_path,
            )
            .await
        {
            eprintln!(
                "ox: slug rename: moving worktree {} → {} failed: {err:#}",
                old_worktree_path.display(),
                new_worktree_path.display()
            );
            // Best-effort rollback of the branch rename so the on-disk
            // state stays consistent with the session JSON. A failed
            // rollback is only logged — the session is still usable on
            // its original short-UUID path.
            if let Err(rb_err) = self
                .git
                .rename_branch(&self.workspace.workspace_root, &new_branch, &old_branch)
                .await
            {
                eprintln!(
                    "ox: slug rename: branch rollback {new_branch} → {old_branch} failed: {rb_err:#}"
                );
            }
            return;
        }

        let mut updated = session;
        updated.worktree_path = new_worktree_path.clone();
        if let Err(err) = self.session_store.save(&updated).await {
            eprintln!("ox: slug rename: saving updated session JSON failed: {err:#}");
            // At this point the branch+worktree moved but the JSON
            // still points at the old path. A restart would skip this
            // session's resume because `old_worktree_path` no longer
            // exists on disk, stranding it until manual cleanup. We
            // don't attempt a double-rollback; the user can rerun the
            // session from the renamed branch by hand.
            return;
        }

        let Some(registry) = self.registry() else {
            // Registry has been dropped (shutdown in flight). The JSON
            // is already updated so a future launch will pick up the
            // new path.
            return;
        };

        if let Err(err) = registry
            .spawn_new_agent_for_existing(id, new_worktree_path)
            .await
        {
            eprintln!("ox: slug rename: respawning agent for session {id} failed: {err:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use agent_host::fake::{FakeGit, FakeSlugGenerator, GitCall};
    use domain::SessionId;

    use super::*;

    fn test_lifecycle() -> (Arc<SessionLifecycle>, tempfile::TempDir) {
        // A tempdir is required because `DiskSessionStore::new` eagerly
        // creates the directory it points at; the lifecycle does not
        // actually touch the store in any of these tests, but we still
        // need a well-formed handle.
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = DiskSessionStore::new(tmp.path()).expect("store");
        let lifecycle = SessionLifecycle::new(
            Arc::new(FakeGit::new()),
            Arc::new(FakeSlugGenerator::new()),
            Arc::new(store),
            WorkspaceContext::new(PathBuf::from("/ws"), "main".into()),
        );
        (lifecycle, tmp)
    }

    /// Builds a lifecycle wired to the shared fakes it returns, with a
    /// real [`DiskSessionStore`] rooted inside a temp directory. Tests
    /// that need to assert on git calls, generator invocations, or the
    /// on-disk session JSON use this instead of `test_lifecycle`.
    fn test_lifecycle_with_fakes() -> (
        Arc<SessionLifecycle>,
        Arc<FakeGit>,
        Arc<FakeSlugGenerator>,
        Arc<DiskSessionStore>,
        PathBuf,
        tempfile::TempDir,
    ) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace_root = tmp.path().join("repo");
        std::fs::create_dir_all(&workspace_root).expect("repo dir");
        let sessions_dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let store = Arc::new(DiskSessionStore::new(&sessions_dir).expect("store"));
        let git = Arc::new(FakeGit::new());
        let slug = Arc::new(FakeSlugGenerator::new());
        let lifecycle = SessionLifecycle::new(
            git.clone(),
            slug.clone(),
            store.clone(),
            WorkspaceContext::new(workspace_root.clone(), "main".into()),
        );
        (lifecycle, git, slug, store, workspace_root, tmp)
    }

    #[test]
    fn registry_is_none_before_set_registry() {
        let (lifecycle, _tmp) = test_lifecycle();
        assert!(
            lifecycle.registry().is_none(),
            "registry should be None before set_registry"
        );
    }

    #[test]
    fn set_registry_is_idempotent() {
        // `OnceLock::set` is write-once; a second call must not panic
        // and must not overwrite the first value. We can't easily build
        // a real `SessionRegistry` here (it would need a spawner etc.),
        // so the test is: call set_registry twice with dummy weak-refs
        // and confirm the first one wins. Since both are dummy Weaks
        // pointing at freed memory, both upgrade to None; we assert the
        // call does not panic.
        let (lifecycle, _tmp) = test_lifecycle();
        let dummy: Weak<SessionRegistry> = Weak::new();
        lifecycle.set_registry(dummy.clone());
        lifecycle.set_registry(dummy); // second call is a silent no-op
        assert!(lifecycle.registry().is_none());
    }

    #[tokio::test]
    async fn close_sink_without_registry_is_a_silent_no_op() {
        // Before `set_registry` lands the weak-ref, merge/abandon both
        // surface `Internal("registry has been dropped")`. That maps
        // to a broadcast-Error — but there is no session to broadcast
        // on, so the call must drive to completion without panicking
        // and without touching the (absent) registry.
        let (lifecycle, _tmp) = test_lifecycle();
        let sink: Arc<dyn CloseRequestSink> = lifecycle.clone();
        sink.request_close(SessionId::new_v4(), CloseIntent::Merge)
            .await;
        sink.request_close(SessionId::new_v4(), CloseIntent::Abandon { confirm: true })
            .await;
    }

    #[test]
    fn rejection_message_covers_every_variant() {
        // Each `MergeRejection` variant must produce a message whose
        // decisive keyword identifies the cause — the UI surfaces this
        // verbatim through its Error banner, so swapping two variants'
        // text would silently mis-diagnose the failure. A future
        // refactor that rewrites these strings must update the
        // assertions here in lockstep.
        let cases: &[(MergeRejection, &str)] = &[
            (MergeRejection::NotFound, "no longer exists"),
            (MergeRejection::AlreadyClosing, "another close"),
            (MergeRejection::TurnInProgress, "turn"),
            (MergeRejection::WorktreeDirty, "uncommitted"),
            (MergeRejection::MainDirty, "main"),
            (MergeRejection::Conflict, "conflict"),
        ];
        for (rej, needle) in cases {
            let msg = rejection_message(rej);
            assert!(
                msg.to_ascii_lowercase().contains(needle),
                "expected '{needle}' in message for {rej:?}, got {msg:?}"
            );
        }
        // The Internal variant should propagate its payload verbatim so
        // the server log and the broadcast reason match.
        assert_eq!(
            rejection_message(&MergeRejection::Internal("boom".into())),
            "boom"
        );
    }

    #[tokio::test]
    async fn slug_none_skips_git_and_respawn() {
        // When the generator returns `None` the session must stay on
        // its `ox/<short>` branch. No rename, no move, no session
        // store write. The generator itself is still called once with
        // the first message — that is how callers distinguish "asked,
        // got no-op" from "skipped entirely."
        let (lifecycle, git, slug, _store, _ws, _tmp) = test_lifecycle_with_fakes();
        // No response scripted → defaults to None.
        let id = SessionId::new_v4();
        let sink: Arc<dyn FirstTurnSink> = lifecycle.clone();
        sink.on_first_turn_complete(id, "fix the login bug".into())
            .await;

        assert_eq!(slug.calls(), vec!["fix the login bug".to_owned()]);
        assert!(
            git.calls().is_empty(),
            "no git operations expected when slug is None, got {:?}",
            git.calls()
        );
    }

    #[tokio::test]
    async fn slug_success_renames_branch_moves_worktree_and_saves_session() {
        // Happy path for the slug rename. Seed the session store with a
        // session whose worktree_path is under `<tmp>/worktrees/<short>`,
        // script the generator to return `fix-login`, and confirm:
        //   - rename_branch ran with `ox/<short>` → `ox/fix-login-<short>`
        //   - move_worktree ran with old → new paths
        //   - the session JSON now points at the new worktree_path
        //   - calls appeared in the documented order (branch first,
        //     then worktree — branch rename is cheap and reversible).
        //
        // The registry respawn step is NOT exercised here (the
        // lifecycle's `registry()` is `None`, so the call short-
        // circuits). A separate test could cover the respawn once a
        // registry can be wired in; today it isn't — but the respawn
        // is a thin passthrough to `SessionRegistry::spawn_new_agent_for_existing`,
        // which has its own tests.
        let (lifecycle, git, slug, store, workspace_root, _tmp) = test_lifecycle_with_fakes();
        let id = SessionId::new_v4();
        let short = short_prefix(id);
        let old_worktree = workspace_root.join("worktrees").join(&short);
        let expected_new = workspace_root
            .join("worktrees")
            .join(format!("fix-login-{short}"));

        let first_msg = "help me fix the login flow".to_owned();
        slug.set_response(&first_msg, Some("fix-login".into()));

        use app::SessionStore;
        let seed = domain::Session::new(id, workspace_root.clone(), old_worktree.clone());
        store.save(&seed).await.expect("seed session");

        let sink: Arc<dyn FirstTurnSink> = lifecycle.clone();
        sink.on_first_turn_complete(id, first_msg.clone()).await;

        let calls = git.calls();
        assert_eq!(
            calls.len(),
            2,
            "expected exactly branch-rename + worktree-move, got {calls:?}"
        );
        match &calls[0] {
            GitCall::RenameBranch {
                workspace_root: ws,
                old,
                new,
            } => {
                assert_eq!(ws, &workspace_root);
                assert_eq!(old, &format!("ox/{short}"));
                assert_eq!(new, &format!("ox/fix-login-{short}"));
            }
            other => panic!("expected RenameBranch first, got {other:?}"),
        }
        match &calls[1] {
            GitCall::MoveWorktree {
                workspace_root: ws,
                old_path,
                new_path,
            } => {
                assert_eq!(ws, &workspace_root);
                assert_eq!(old_path, &old_worktree);
                assert_eq!(new_path, &expected_new);
            }
            other => panic!("expected MoveWorktree second, got {other:?}"),
        }

        let reloaded = store.try_load(id).await.expect("reload").expect("exists");
        assert_eq!(reloaded.worktree_path, expected_new);
        assert_eq!(
            reloaded.workspace_root, workspace_root,
            "workspace_root must stay the main-repo root across a slug rename"
        );
    }

    #[tokio::test]
    async fn slug_rename_skips_when_session_not_on_disk() {
        // If the hook fires before the agent has persisted the session
        // (unlikely but possible — the agent saves on TurnComplete;
        // the hook fires after the pump observes that same frame, so
        // a race with disk flush is theoretically possible), the hook
        // must not attempt a rename. `try_load` returns `Ok(None)` in
        // that case, and the generator's output is discarded.
        let (lifecycle, git, slug, _store, _ws, _tmp) = test_lifecycle_with_fakes();
        slug.set_response("anything", Some("anything-slug".into()));
        let id = SessionId::new_v4();

        let sink: Arc<dyn FirstTurnSink> = lifecycle.clone();
        sink.on_first_turn_complete(id, "anything".into()).await;

        assert!(
            git.calls().is_empty(),
            "no git operations expected when the session isn't on disk yet"
        );
    }
}
