//! Session lifecycle coordinator.
//!
//! `SessionLifecycle` owns all the moving parts that orchestrate a
//! session across its birth-to-close arc:
//!
//! - the [`Git`] adapter (worktree + branch operations),
//! - the [`SlugGenerator`] (turns the first user message into a slug
//!   used for renaming the worktree dir + branch),
//! - the [`SessionStore`] port (loads/saves/deletes host-visible
//!   session JSON),
//! - the [`WorkspaceContext`] (main workspace root + base branch),
//! - per-session close locks so concurrent merge / abandon requests
//!   for the same session serialize cleanly.
//!
//! The coordinator is decoupled from [`SessionRegistry`] at
//! construction time: every method that touches the registry takes
//! `&SessionRegistry` as a parameter. Agent-initiated close requests
//! and first-turn notifications reach the coordinator via the
//! [`ChannelCloseSink`] and [`ChannelFirstTurnSink`] types, which push
//! messages onto unbounded channels; the composition root owns
//! consumer tasks that drain the receivers and dispatch back to
//! [`SessionLifecycle::handle_close_request`] /
//! [`SessionLifecycle::handle_first_turn`] with a live registry handle.
//!
//! # Why a separate type
//!
//! Piling merge / abandon / close-routing / slug-rename / session-JSON
//! deletion onto `SessionRegistry` would turn the registry into the
//! central coupling point for every future workspace feature. The
//! registry's three existing concerns (sessions map, layout store,
//! spawner) stay stable; the coordinator mutates the sessions map in
//! response to lifecycle events.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_host::{
    Git, MergeOutcome, SlugGenerator, WorkspaceContext, WorktreeStatus, workspace_slug,
};
use anyhow::{Context, Result};
use app::SessionStore;
use domain::{CloseIntent, Session, SessionId};

use crate::registry::SessionRegistry;
use crate::session::CloseStart;

mod close_guard;
mod sinks;

use close_guard::CloseGuard;
pub use sinks::{ChannelCloseSink, ChannelFirstTurnSink, CloseRequestMsg, FirstTurnMsg};

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
    /// Host-facing session store. The agent subprocess owns normal turn
    /// persistence; lifecycle policy uses the port for close preflight,
    /// teardown deletion, and slug-rename path updates.
    session_store: Arc<dyn SessionStore>,
    /// Main workspace root + base branch, snapshotted at startup. Every
    /// worktree-add and merge runs against these values; they do not
    /// change over the server's lifetime.
    workspace: WorkspaceContext,
    /// Per-session close-in-flight flags. A fresh entry is inserted
    /// when a close begins and removed on the `CloseGuard`'s drop.
    /// Concurrent merge/abandon calls for the same id observe a
    /// `Closing` entry and return `AlreadyClosing`.
    closing: Mutex<HashMap<SessionId, CloseState>>,
}

impl SessionLifecycle {
    /// Build the coordinator. The registry is passed into each
    /// lifecycle method at call time; there is no back-reference to
    /// install.
    pub fn new(
        git: Arc<dyn Git>,
        slug_generator: Arc<dyn SlugGenerator>,
        session_store: Arc<dyn SessionStore>,
        workspace: WorkspaceContext,
    ) -> Arc<Self> {
        Arc::new(Self {
            git,
            slug_generator,
            session_store,
            workspace,
            closing: Mutex::new(HashMap::new()),
        })
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
    pub async fn create_and_spawn(&self, registry: &SessionRegistry) -> Result<SessionId> {
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

        let id = registry
            .spawn_for_worktree(worktree_path.clone(), session_id, None)
            .await?;

        let session = Session::new(
            id,
            self.workspace.workspace_root.clone(),
            worktree_path,
            registry.default_model().to_owned(),
        );
        self.session_store
            .save(&session)
            .await
            .with_context(|| format!("saving initial session record for {id}"))?;

        Ok(id)
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
    /// 3. The agent is idle and is marked closing. Later sends now
    ///    receive `409 Conflict` instead of racing into teardown.
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
    pub async fn merge(
        &self,
        id: SessionId,
        registry: &SessionRegistry,
    ) -> Result<(), MergeRejection> {
        let (session, worktree_path, branch, mut _guard) =
            self.begin_close_flow(id, registry).await?;

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
        _guard.keep_session_closing();

        self.teardown(
            id,
            &branch,
            &worktree_path,
            /* force_branch */ false,
            registry,
        )
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
    pub async fn abandon(
        &self,
        id: SessionId,
        confirm: bool,
        registry: &SessionRegistry,
    ) -> Result<(), MergeRejection> {
        let (session, worktree_path, branch, mut _guard) =
            self.begin_close_flow(id, registry).await?;

        let status = self
            .git
            .status(&worktree_path)
            .await
            .map_err(|err| MergeRejection::Internal(format!("git status failed: {err:#}")))?;
        if status == WorktreeStatus::Dirty && !confirm {
            return Err(MergeRejection::WorktreeDirty);
        }

        drop(session);
        _guard.keep_session_closing();

        self.teardown(
            id,
            &branch,
            &worktree_path,
            /* force_branch */ true,
            registry,
        )
        .await
    }

    /// Dispatch an agent-initiated close request. Called by the consumer
    /// task that drains the close-request channel.
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
    pub async fn handle_close_request(
        &self,
        id: SessionId,
        intent: CloseIntent,
        registry: &SessionRegistry,
    ) {
        let result = match intent {
            CloseIntent::Merge => self.merge(id, registry).await,
            CloseIntent::Abandon { confirm } => self.abandon(id, confirm, registry).await,
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

        if let Some(session) = registry.get(id) {
            session.broadcast_error(message.clone());
        }
        eprintln!("ox: {message} (session {id})");
    }

    /// Slug-rename hook, invoked by the consumer task that drains the
    /// first-turn channel. On the first `TurnComplete` of a fresh
    /// session we ask the slug generator for a kebab-case label, rename
    /// the git branch from `ox/<short>` to `ox/<slug>-<short>`, move the
    /// worktree directory to match, persist the new path in the session
    /// JSON, and respawn the agent pointed at the new worktree path.
    ///
    /// Every step is best-effort: a generator timeout, a branch-rename
    /// conflict, or a respawn failure all log and return. The session stays
    /// on its original short-UUID name but remains fully functional.
    ///
    /// Idempotency: the pump CAS-flips `fresh` before pushing onto the
    /// channel so this hook fires at most once per `ActiveSession`. On a
    /// process restart, the resumed session starts with `fresh=false`, so
    /// the pump's guard prevents a re-fire without any work here.
    pub async fn handle_first_turn(
        &self,
        id: SessionId,
        first_message: String,
        registry: &SessionRegistry,
    ) {
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

        if let Err(err) = registry
            .spawn_new_agent_for_existing(id, new_worktree_path)
            .await
        {
            eprintln!("ox: slug rename: respawning agent for session {id} failed: {err:#}");
        }
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
        registry: &SessionRegistry,
    ) -> Result<
        (
            Arc<crate::session::ActiveSession>,
            PathBuf,
            String,
            CloseGuard<'_>,
        ),
        MergeRejection,
    > {
        let session = registry.get(id).ok_or(MergeRejection::NotFound)?;

        let mut guard = self.begin_close(id).ok_or(MergeRejection::AlreadyClosing)?;

        // The session marker closes the send/close race: once set, a
        // later POST /messages returns conflict before touching the
        // agent's stdin. The close guard clears the marker on every
        // rejected preflight path.
        match session.begin_close() {
            CloseStart::Closing => guard.protect_session(session.clone()),
            CloseStart::TurnInProgress => return Err(MergeRejection::TurnInProgress),
            CloseStart::AlreadyClosing => return Err(MergeRejection::AlreadyClosing),
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
        registry: &SessionRegistry,
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

        registry.remove(id);

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
                    session: None,
                    clear_session_on_drop: true,
                })
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
