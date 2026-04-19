//! `SessionRegistry` — the server's entire session-state model.
//!
//! The registry owns:
//!
//! - a map of live sessions keyed by [`SessionId`],
//! - the workspace root this server instance is scoped to,
//! - the [`LayoutRepository`] that persists pane order / sizes across runs,
//! - the [`AgentSpawner`] used to birth new subprocesses (or, in
//!   tests, to hand back duplex-backed clients).
//!
//! It does **not** model "focus" or "the active pane" — those are pure
//! client concerns in the web UI. It does **not** model a workspace
//! beyond `workspace_root` — multiple workspaces per server is out of
//! scope.
//!
//! Locking discipline:
//!
//! - `sessions` is an `RwLock`; handler paths take a read guard, clone
//!   the `Arc<ActiveSession>`, and release the guard before awaiting.
//!   `create`/`remove`/`restore` take the write guard.
//! - `layout` is a `Mutex` on the on-disk store. Held briefly around
//!   reads and writes.
//! - Neither lock is held across an `.await` involving the agent
//!   (wire I/O, `await_ready`, etc.).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use agent_host::{
    AgentSpawnConfig, AgentSpawner, CloseRequestSink, FirstTurnSink, Layout, LayoutRepository,
};
use anyhow::{Context, Result, anyhow};
use app::SessionStore;
use domain::SessionId;
use protocol::AgentCommand;
use serde::Serialize;

use crate::session::{ActiveSession, SendOutcome};

/// JSON payload for `GET /api/sessions` — everything the frontend
/// needs on first paint. Snapshotted under the registry's read lock,
/// so any concurrent `create`/`remove` appears as "not yet" or "gone,"
/// never as half-applied state.
#[derive(Debug, Serialize)]
pub struct SnapshotJson {
    pub workspace_root: PathBuf,
    pub sessions: Vec<SessionSummary>,
    pub layout: Layout,
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub model: String,
}

/// Errors [`SessionRegistry::send_command`] maps to HTTP status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandDispatch {
    Ok,
    NotFound,
    Dead,
    AlreadyTurning,
    Closing,
}

pub struct SessionRegistry {
    sessions: RwLock<HashMap<SessionId, Arc<ActiveSession>>>,
    spawner: Arc<dyn AgentSpawner>,
    spawn_config: AgentSpawnConfig,
    layout: Arc<dyn LayoutRepository>,
    workspace_root: PathBuf,
    /// Sink for `AgentEvent::RequestClose` frames the pump observes.
    /// Passed through to every `ActiveSession` at creation so the pump
    /// can route close intents to the lifecycle coordinator without a
    /// cyclic reference back to the registry.
    close_sink: Arc<dyn CloseRequestSink>,
    /// Sink for the first `TurnComplete` of a fresh session. Passed
    /// through to every `ActiveSession` at creation so the pump can
    /// route the slug-rename trigger to the lifecycle coordinator.
    /// Shares the coordinator's `Arc` with `close_sink` — they are two
    /// trait projections of the same object in production.
    first_turn_sink: Arc<dyn FirstTurnSink>,
}

impl SessionRegistry {
    /// Build an empty registry. `restore` is the usual entry point on
    /// startup; `new` exists for tests that want an empty registry.
    pub fn new(
        spawner: Arc<dyn AgentSpawner>,
        spawn_config: AgentSpawnConfig,
        layout: Arc<dyn LayoutRepository>,
        workspace_root: PathBuf,
        close_sink: Arc<dyn CloseRequestSink>,
        first_turn_sink: Arc<dyn FirstTurnSink>,
    ) -> Arc<Self> {
        Arc::new(Self {
            sessions: RwLock::new(HashMap::new()),
            spawner,
            spawn_config,
            layout,
            workspace_root,
            close_sink,
            first_turn_sink,
        })
    }

    /// Attempt to resume every saved session for this workspace root.
    ///
    /// For each layout-row session id:
    /// - Look up its `{id}.json` via `session_store.try_load`. If no
    ///   file exists, the row is stale (session was never persisted or
    ///   was hand-deleted); skip it.
    /// - Read `worktree_path` from the loaded session and spawn an
    ///   agent pointed at that path with `resume: Some(id)`.
    /// - Any per-session failure (worktree missing, agent binary out of
    ///   date, etc.) is logged and the loop continues so one broken row
    ///   does not block the rest of the layout from restoring.
    ///
    /// If no session resumes, the registry is returned empty. The
    /// frontend will show the workspace with no panes until the user
    /// creates a session via `POST /api/sessions`.
    pub async fn restore(
        spawner: Arc<dyn AgentSpawner>,
        spawn_config: AgentSpawnConfig,
        layout: Arc<dyn LayoutRepository>,
        workspace_root: PathBuf,
        close_sink: Arc<dyn CloseRequestSink>,
        first_turn_sink: Arc<dyn FirstTurnSink>,
        session_store: Arc<dyn SessionStore>,
    ) -> Result<Arc<Self>> {
        let registry = Self::new(
            spawner,
            spawn_config,
            layout,
            workspace_root.clone(),
            close_sink,
            first_turn_sink,
        );

        let saved_order: Vec<SessionId> = match registry.layout.get(&workspace_root).await {
            Ok(Some(layout)) => layout.order,
            Ok(None) => Vec::new(),
            Err(err) => {
                eprintln!("ox: failed to load saved layout for restore: {err:#}");
                Vec::new()
            }
        };

        for id in saved_order {
            let session = match session_store.try_load(id).await {
                Ok(Some(s)) => s,
                Ok(None) => {
                    // Session file never written — the previous run died
                    // before the first `TurnComplete`. The layout row is
                    // stale; skip it and move on.
                    eprintln!("ox: skipping session {id}: no on-disk session file");
                    continue;
                }
                Err(err) => {
                    eprintln!("ox: failed to load session {id}: {err:#}");
                    continue;
                }
            };

            if !session.worktree_path.exists() {
                eprintln!(
                    "ox: skipping session {id}: worktree {} is missing",
                    session.worktree_path.display()
                );
                continue;
            }

            if let Err(err) = registry
                .spawn_for_worktree(session.worktree_path.clone(), id, Some(id))
                .await
            {
                eprintln!("ox: failed to resume session {id}: {err:#}");
            }
        }

        Ok(registry)
    }

    /// Spawn an agent that resumes a specific session id. Used by the
    /// `--resume <id>` CLI path. The session's saved `worktree_path` is
    /// loaded from the store so the agent comes back on the same branch
    /// it was last working on.
    pub async fn create_resumed(
        &self,
        id: SessionId,
        session_store: &dyn SessionStore,
    ) -> Result<SessionId> {
        let session = session_store
            .try_load(id)
            .await
            .with_context(|| format!("loading session {id} for --resume"))?
            .ok_or_else(|| anyhow!("no saved session with id {id}"))?;
        if !session.worktree_path.exists() {
            return Err(anyhow!(
                "session {id} worktree {} is missing",
                session.worktree_path.display()
            ));
        }
        self.spawn_for_worktree(session.worktree_path, id, Some(id))
            .await
    }

    /// Drop the session from the registry. The `Arc<ActiveSession>`
    /// held by the map is released; its `AgentClient` drops; the child
    /// dies via `kill_on_drop`. Returns true if the session existed.
    pub fn remove(&self, id: SessionId) -> bool {
        let mut sessions = self.sessions.write().expect("sessions lock poisoned");
        sessions.remove(&id).is_some()
    }

    /// Drop every session. Used by the graceful-shutdown hook: dropping
    /// each `ActiveSession` drops its broadcast `Sender`, which causes
    /// the follow half of every live SSE stream to terminate, which
    /// lets `axum::serve(..).with_graceful_shutdown(..)` finish instead
    /// of hanging on those long-lived responses forever.
    pub fn shutdown(&self) {
        let mut sessions = self.sessions.write().expect("sessions lock poisoned");
        sessions.clear();
    }

    /// Look up a session by id, bumping its `Arc` count for the
    /// handler's lifetime.
    pub fn get(&self, id: SessionId) -> Option<Arc<ActiveSession>> {
        let sessions = self.sessions.read().expect("sessions lock poisoned");
        sessions.get(&id).cloned()
    }

    /// `true` when no sessions are registered.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let sessions = self.sessions.read().expect("sessions lock poisoned");
        sessions.is_empty()
    }

    /// Snapshot for `GET /api/sessions`.
    pub async fn snapshot(&self) -> SnapshotJson {
        let layout = match self.layout.get(&self.workspace_root).await {
            Ok(Some(layout)) => layout,
            Ok(None) => Layout::default(),
            Err(err) => {
                eprintln!("ox: failed to read workspace layout: {err:#}");
                Layout::default()
            }
        };
        let sessions = self.sessions.read().expect("sessions lock poisoned");

        // Present sessions in layout order first, then append any
        // live sessions that aren't referenced by the layout (e.g.,
        // a freshly created session before the client has PUT the
        // new layout).
        let mut ordered: Vec<SessionSummary> = Vec::with_capacity(sessions.len());
        let mut seen = std::collections::HashSet::new();
        for id in &layout.order {
            if let Some(session) = sessions.get(id) {
                let _ = session; // Summary doesn't need session-level state today.
                ordered.push(SessionSummary {
                    session_id: *id,
                    model: self.spawn_config.model.clone(),
                });
                seen.insert(*id);
            }
        }
        for id in sessions.keys() {
            if !seen.contains(id) {
                ordered.push(SessionSummary {
                    session_id: *id,
                    model: self.spawn_config.model.clone(),
                });
            }
        }

        SnapshotJson {
            workspace_root: self.workspace_root.clone(),
            sessions: ordered,
            layout,
        }
    }

    /// Persist a client-authored layout. Unknown session ids are
    /// filtered out (the client can race the server on remove);
    /// [`LayoutRepository::put`] normalizes sizes.
    pub async fn put_layout(&self, mut layout: Layout) -> Result<()> {
        {
            let sessions = self.sessions.read().expect("sessions lock poisoned");
            // Filter order by known ids, preserving sizes by index.
            // After filtering we may have fewer sizes than entries,
            // which `LayoutRepository::put` re-normalizes to equal widths.
            let mut new_order = Vec::with_capacity(layout.order.len());
            let mut new_sizes = Vec::with_capacity(layout.order.len());
            for (i, id) in layout.order.iter().enumerate() {
                if sessions.contains_key(id) {
                    new_order.push(*id);
                    if let Some(sz) = layout.sizes.get(i).copied() {
                        new_sizes.push(sz);
                    }
                }
            }
            layout.order = new_order;
            layout.sizes = new_sizes;
        }

        self.layout.put(&self.workspace_root, layout).await
    }

    /// Persist the current registry state as a one-row layout with
    /// equal sizes. Used on graceful shutdown when the server wants
    /// to make sure a restart can find the current set of sessions.
    pub async fn persist_current_layout(&self) -> Result<()> {
        let ids: Vec<SessionId> = {
            let sessions = self.sessions.read().expect("sessions lock poisoned");
            sessions.keys().copied().collect()
        };
        if ids.is_empty() {
            return Ok(());
        }
        let n = ids.len();
        let layout = Layout::new(ids, vec![1.0 / n as f32; n]);
        self.layout.put(&self.workspace_root, layout).await
    }

    /// Dispatch a command to a session by id. Maps `SendOutcome`
    /// onto the richer `CommandDispatch` so the handler has all four
    /// HTTP paths (204/404/410/409) in one match.
    pub async fn send_command(&self, id: SessionId, cmd: AgentCommand) -> CommandDispatch {
        let session = match self.get(id) {
            Some(s) => s,
            None => return CommandDispatch::NotFound,
        };
        match cmd {
            AgentCommand::SendMessage { input } => match session.send_message(input) {
                SendOutcome::Ok => CommandDispatch::Ok,
                SendOutcome::Dead => CommandDispatch::Dead,
                SendOutcome::AlreadyTurning => CommandDispatch::AlreadyTurning,
                SendOutcome::Closing => CommandDispatch::Closing,
            },
            AgentCommand::Cancel => {
                session.cancel();
                CommandDispatch::Ok
            }
            // `AgentCommand` is `#[non_exhaustive]`; accept unknown
            // variants as 204 on the theory that the handler shouldn't
            // have routed them here. Callers use the narrow, typed
            // `send_message` / `cancel` methods instead.
            _ => CommandDispatch::Ok,
        }
    }

    pub async fn resolve_tool_approval(
        &self,
        id: SessionId,
        request_id: String,
        approved: bool,
    ) -> CommandDispatch {
        let session = match self.get(id) {
            Some(s) => s,
            None => return CommandDispatch::NotFound,
        };
        match session.resolve_tool_approval(request_id, approved) {
            SendOutcome::Ok => CommandDispatch::Ok,
            SendOutcome::Dead => CommandDispatch::Dead,
            SendOutcome::AlreadyTurning => CommandDispatch::Ok,
            SendOutcome::Closing => CommandDispatch::Closing,
        }
    }

    /// Shared entry point for spawning a session against a specific
    /// worktree. `create` / `restore` route through this today with
    /// `worktree_path = self.workspace_root`; the lifecycle coordinator
    /// will call it directly with the session's own worktree directory
    /// once R11 lands.
    ///
    /// `session_id` is advisory — the lifecycle coordinator pre-allocates
    /// it to name the worktree directory before spawning, but the agent
    /// still reports its own id via the `Ready` frame and that is the
    /// source of truth for the registry map. Keeping the parameter on
    /// the signature today means the coordinator can drop into the
    /// same call shape without churn when it lands.
    pub async fn spawn_for_worktree(
        &self,
        worktree_path: PathBuf,
        session_id: SessionId,
        resume: Option<SessionId>,
    ) -> Result<SessionId> {
        let mut config = self.spawn_config.clone();
        // The agent process's view of "workspace root" is the worktree —
        // that's what its tools resolve relative paths against and what
        // its shell's `current_dir` points at. The registry's own
        // `workspace_root` field continues to refer to the main
        // workspace; the two diverge once the lifecycle coordinator
        // starts creating per-session worktrees.
        config.workspace_root = worktree_path;
        config.resume = resume;
        // Pass the pre-allocated id through to the agent so the
        // on-disk `{id}.json`, the host-side worktree directory name,
        // and the `ox/<slug>` branch all share the same identifier.
        // The agent still announces its id via `Ready`, but on a fresh
        // spawn that announced value will equal `session_id`.
        config.session_id = Some(session_id);

        // A session is "fresh" iff we're not resuming a saved one. The
        // slug-rename hook checks this flag so it runs exactly once, on
        // the first `TurnComplete` after a user starts a brand-new
        // session.
        let fresh = resume.is_none();

        let (client, stream) = self.spawner.spawn(config).context("spawning agent")?;
        let session = ActiveSession::start(
            client,
            stream,
            self.close_sink.clone(),
            self.first_turn_sink.clone(),
            fresh,
        );

        let id = session
            .await_ready()
            .await
            .ok_or_else(|| anyhow!("agent exited before emitting Ready"))?;

        // Insert under the write lock. If for some reason the agent
        // re-announced a known id (shouldn't happen), the newer
        // session displaces the older — the older's Arc drops,
        // killing its child.
        let mut sessions = self.sessions.write().expect("sessions lock poisoned");
        sessions.insert(id, session);
        Ok(id)
    }

    /// Respawn an existing session's agent under a new worktree path.
    ///
    /// Used by the slug-rename hook after the worktree directory has
    /// been moved and the branch renamed: a fresh `ox-agent` is spawned
    /// with `--workspace-root=<new-path>` and `--resume=<id>`, and the
    /// resulting client+stream are installed into the existing
    /// [`ActiveSession`] via [`ActiveSession::replace_agent`]. SSE
    /// subscribers keep their connection and history survives across
    /// the swap.
    ///
    /// Returns an error if the session is not known, or if the spawner
    /// itself fails. The agent's `Ready` frame is NOT awaited here
    /// because the session's existing `session_id` is already known; the
    /// replacement pump records the new `Ready` into history like any
    /// other event.
    pub async fn spawn_new_agent_for_existing(
        &self,
        id: SessionId,
        new_worktree_path: PathBuf,
    ) -> Result<()> {
        let session = self
            .get(id)
            .ok_or_else(|| anyhow!("session {id} not found in registry"))?;

        let mut config = self.spawn_config.clone();
        config.workspace_root = new_worktree_path;
        config.resume = Some(id);
        config.session_id = Some(id);

        let (client, stream) = self
            .spawner
            .spawn(config)
            .context("spawning replacement agent")?;

        session.replace_agent(client, stream);
        Ok(())
    }

    /// Accessor for handlers and tests that need the workspace root
    /// out-of-band from the `snapshot()` JSON.
    #[allow(dead_code)]
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use adapter_storage::{DiskLayoutRepository, DiskSessionStore};
    use agent_host::{AgentSpawnConfig, Layout};
    use domain::SessionId;
    use protocol::AgentEvent;
    use tokio::time::timeout;

    use super::*;
    use crate::test_support::{DuplexSpawner, empty_layout, test_registry, unique_temp_dir};

    /// Drive `registry.spawn_for_worktree` to completion by receiving the
    /// agent-side handles, writing `Ready(id)`, and awaiting the
    /// registry's returned session id. Returns the id and the live
    /// agent handles so the test can keep interacting with them. The
    /// worktree path is the registry's workspace root — tests here do
    /// not exercise the per-session-worktree path; that's the
    /// lifecycle coordinator's job.
    async fn create_session(
        registry: Arc<SessionRegistry>,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::test_support::AgentHandles>,
    ) -> (SessionId, crate::test_support::AgentHandles) {
        let id = SessionId::new_v4();
        let worktree = registry.workspace_root().to_path_buf();
        let create = tokio::spawn({
            let r = registry.clone();
            async move { r.spawn_for_worktree(worktree, id, None).await }
        });
        let mut handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("spawner receiver dropped");
        handles.send_ready(id).await;
        let ret_id = timeout(Duration::from_secs(2), create)
            .await
            .expect("create timed out")
            .expect("join failure")
            .expect("spawn_for_worktree returned Err");
        assert_eq!(ret_id, id);
        (id, handles)
    }

    // -- spawn_for_worktree ----------------------------------------------

    #[tokio::test]
    async fn spawn_awaits_ready_then_registers_session() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, _handles) = create_session(registry.clone(), &mut rx).await;

        // Snapshot must now include the session in the order/sessions list.
        let snap = registry.snapshot().await;
        assert!(snap.sessions.iter().any(|s| s.session_id == id));
        assert!(registry.get(id).is_some());
    }

    #[tokio::test]
    async fn spawn_returns_error_if_agent_exits_before_ready() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let id = SessionId::new_v4();
        let worktree = registry.workspace_root().to_path_buf();
        let create = tokio::spawn({
            let r = registry.clone();
            async move { r.spawn_for_worktree(worktree, id, None).await }
        });
        let handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("spawner dropped");
        // Drop the handles without writing `Ready`. The agent stdout
        // side closes, the pump observes `None`, alive flips false,
        // await_ready resolves None, and spawn_for_worktree returns an Err.
        drop(handles);
        let result = timeout(Duration::from_secs(2), create)
            .await
            .expect("spawn_for_worktree timed out")
            .expect("join failure");
        assert!(result.is_err(), "expected spawn_for_worktree to fail");
    }

    // -- remove ----------------------------------------------------------

    #[tokio::test]
    async fn remove_returns_true_once_then_false() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, _handles) = create_session(registry.clone(), &mut rx).await;

        assert!(registry.remove(id));
        assert!(!registry.remove(id));
        assert!(registry.get(id).is_none());
    }

    #[tokio::test]
    async fn remove_drops_agent_pipe() {
        // Removing the session drops its Arc, which drops the AgentClient,
        // which closes the stdin pipe. The test reads the duplex until
        // EOF to verify the close propagates.
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, mut handles) = create_session(registry.clone(), &mut rx).await;

        registry.remove(id);

        // `next_command` returns None on clean EOF. Use a timeout so
        // a bug doesn't hang the test indefinitely.
        let next = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("next_command timed out after remove");
        assert!(next.is_none(), "expected EOF after remove, got {next:?}");
    }

    // -- snapshot --------------------------------------------------------

    #[tokio::test]
    async fn snapshot_orders_sessions_by_layout_then_extras() {
        let (registry, mut rx, ws) = test_registry(empty_layout().await).await;
        let (first, _h1) = create_session(registry.clone(), &mut rx).await;
        let (second, _h2) = create_session(registry.clone(), &mut rx).await;

        // Persist a layout that inverts insertion order.
        registry
            .put_layout(Layout::new(vec![second, first], vec![0.3, 0.7]))
            .await
            .expect("put_layout");

        let snap = registry.snapshot().await;
        assert_eq!(snap.workspace_root, ws);
        assert_eq!(
            snap.sessions
                .iter()
                .map(|s| s.session_id)
                .collect::<Vec<_>>(),
            vec![second, first]
        );
        // The layout field round-trips back unchanged (normalized).
        assert_eq!(snap.layout.order, vec![second, first]);
        assert_eq!(snap.layout.sizes.len(), 2);
    }

    // -- send_command ----------------------------------------------------

    #[tokio::test]
    async fn send_message_returns_not_found_for_unknown_id() {
        let (registry, _rx, _ws) = test_registry(empty_layout().await).await;
        let unknown = SessionId::new_v4();
        let outcome = registry
            .send_command(
                unknown,
                protocol::AgentCommand::SendMessage { input: "hi".into() },
            )
            .await;
        assert_eq!(outcome, CommandDispatch::NotFound);
    }

    #[tokio::test]
    async fn send_message_writes_one_frame_and_second_returns_already_turning() {
        // Plan's "double-send guard" acceptance test: the second send
        // while a turn is in flight must return AlreadyTurning and must
        // not place a second SendMessage on the agent's stdin.
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, mut handles) = create_session(registry.clone(), &mut rx).await;

        let first = registry
            .send_command(
                id,
                protocol::AgentCommand::SendMessage {
                    input: "one".into(),
                },
            )
            .await;
        assert_eq!(first, CommandDispatch::Ok);
        let second = registry
            .send_command(
                id,
                protocol::AgentCommand::SendMessage {
                    input: "two".into(),
                },
            )
            .await;
        assert_eq!(second, CommandDispatch::AlreadyTurning);

        // Exactly one frame must appear on the wire, matching "one".
        let frame1 = timeout(Duration::from_secs(2), handles.next_command())
            .await
            .expect("first frame timeout");
        match frame1 {
            Some(protocol::AgentCommand::SendMessage { input }) => {
                assert_eq!(input, "one");
            }
            other => panic!("expected SendMessage('one'), got {other:?}"),
        }

        // The second send must not have produced a frame. We check
        // with a short timeout — a buffered byte would appear
        // immediately, so 100ms is enough to catch any latent write.
        let second_frame = timeout(Duration::from_millis(100), handles.next_command()).await;
        assert!(
            second_frame.is_err(),
            "unexpected second frame: {second_frame:?}"
        );
    }

    #[tokio::test]
    async fn send_message_returns_dead_after_agent_closes() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, handles) = create_session(registry.clone(), &mut rx).await;

        // Dropping the handles closes the agent side of the duplex. The
        // pump observes EOF on its read half, flips alive to false.
        drop(handles);

        // Give the pump a moment to observe the close.
        // `is_alive` polls an atomic; we loop briefly instead of
        // sleeping a flat duration to keep the test fast when the flag
        // flips quickly.
        for _ in 0..50 {
            let session = registry.get(id).unwrap();
            if !session.is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let outcome = registry
            .send_command(
                id,
                protocol::AgentCommand::SendMessage {
                    input: "after-death".into(),
                },
            )
            .await;
        assert_eq!(outcome, CommandDispatch::Dead);
    }

    #[tokio::test]
    async fn cancel_is_idempotent_even_on_dead_session() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, handles) = create_session(registry.clone(), &mut rx).await;

        // Two live cancels succeed, as does a cancel after the agent exits.
        assert_eq!(
            registry
                .send_command(id, protocol::AgentCommand::Cancel)
                .await,
            CommandDispatch::Ok
        );
        drop(handles);
        // Wait for alive flip.
        for _ in 0..50 {
            if !registry.get(id).unwrap().is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            registry
                .send_command(id, protocol::AgentCommand::Cancel)
                .await,
            CommandDispatch::Ok
        );
    }

    // -- layout ----------------------------------------------------------

    #[tokio::test]
    async fn put_layout_drops_unknown_session_ids() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (known, _handles) = create_session(registry.clone(), &mut rx).await;
        let unknown = SessionId::new_v4();

        registry
            .put_layout(Layout::new(vec![unknown, known], vec![0.5, 0.5]))
            .await
            .expect("put_layout");

        let snap = registry.snapshot().await;
        // The unknown id is filtered; the known id survives.
        assert_eq!(snap.layout.order, vec![known]);
    }

    #[tokio::test]
    async fn put_layout_persists_to_disk() {
        // A PUT-then-reload cycle must round-trip order and (normalized)
        // sizes so a restart picks up the same pane tiling.
        let layout_path = unique_temp_dir("persist").join("workspaces.json");
        let store = DiskLayoutRepository::load(layout_path.clone()).await.unwrap();
        let workspace_root = unique_temp_dir("ws-persist");
        let (spawner, mut rx) = DuplexSpawner::new();
        let registry = SessionRegistry::new(
            spawner,
            AgentSpawnConfig {
                binary: PathBuf::from("/nonexistent/ox-agent"),
                workspace_root: workspace_root.clone(),
                model: "test/model".into(),
                sessions_dir: PathBuf::from("/nonexistent/sessions"),
                resume: None,
                session_id: None,
                env: vec![],
            },
            Arc::new(store),
            workspace_root.clone(),
            Arc::new(agent_host::fake::NoopCloseRequestSink),
            Arc::new(agent_host::fake::NoopFirstTurnSink),
        );
        let (id, _handles) = create_session(registry.clone(), &mut rx).await;
        registry
            .put_layout(Layout::new(vec![id], vec![1.0]))
            .await
            .expect("put_layout");

        // Reload and confirm the file contains the expected entry.
        let reloaded = DiskLayoutRepository::load(layout_path).await.unwrap();
        let got = reloaded
            .get(&workspace_root)
            .await
            .expect("layout read")
            .expect("layout row");
        assert_eq!(got.order, vec![id]);
        assert_eq!(got.sizes, vec![1.0]);
    }

    #[tokio::test]
    async fn put_layout_normalizes_non_finite_sizes() {
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (a, _ha) = create_session(registry.clone(), &mut rx).await;
        let (b, _hb) = create_session(registry.clone(), &mut rx).await;

        registry
            .put_layout(Layout::new(vec![a, b], vec![f32::NAN, 0.5]))
            .await
            .expect("put_layout");
        let snap = registry.snapshot().await;
        for s in &snap.layout.sizes {
            assert!(s.is_finite());
        }
        let sum: f32 = snap.layout.sizes.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    // -- restore ---------------------------------------------------------

    /// Seed the disk session store with a session whose `worktree_path`
    /// is a real directory created inside the tempdir (so the existence
    /// check in `restore` passes). Returns the store and the created
    /// worktree path so the test can assert the spawner saw it.
    async fn seed_saved_session(
        store: &DiskSessionStore,
        id: SessionId,
        workspace_root: &Path,
        label: &str,
    ) -> PathBuf {
        let worktree = unique_temp_dir(label);
        let session = domain::Session::new(id, workspace_root.to_path_buf(), worktree.clone());
        app::SessionStore::save(store, &session).await.unwrap();
        worktree
    }

    #[tokio::test]
    async fn restore_skips_session_whose_file_is_missing() {
        // A layout row referring to an id with no `{id}.json` on disk
        // is stale (previous run died before the first TurnComplete).
        // `restore` should skip it silently and return an empty registry.
        let layout_path = unique_temp_dir("restore-no-file").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore-no-file");
        let stale_id = SessionId::new_v4();
        {
            let store = DiskLayoutRepository::load(layout_path.clone())
                .await
                .unwrap();
            store
                .put(&workspace_root, Layout::new(vec![stale_id], vec![1.0]))
                .await
                .unwrap();
        }

        let layout = DiskLayoutRepository::load(layout_path).await.unwrap();
        // Empty session store — no `{id}.json` exists for stale_id.
        let sessions_dir = unique_temp_dir("sessions-no-file");
        let session_store = Arc::new(DiskSessionStore::new(&sessions_dir).unwrap());
        let (spawner, mut rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir,
            resume: None,
            session_id: None,
            env: vec![],
        };

        let registry = SessionRegistry::restore(
            spawner,
            spawn_config,
            Arc::new(layout),
            workspace_root,
            Arc::new(agent_host::fake::NoopCloseRequestSink),
            Arc::new(agent_host::fake::NoopFirstTurnSink),
            session_store,
        )
        .await
        .expect("restore ok");

        // No session resumed → registry is empty and the spawner was
        // never invoked (no handles arrived).
        assert!(registry.is_empty());
        assert!(rx.try_recv().is_err(), "spawner should not be called");
    }

    #[tokio::test]
    async fn restore_skips_session_with_missing_worktree_directory() {
        // Session file exists, but the `worktree_path` it points at has
        // been removed on disk (user hand-cleaned, or a parallel `git
        // worktree prune` happened). `restore` must treat this as a
        // skippable row rather than trying to resume into a missing
        // directory.
        let layout_path = unique_temp_dir("restore-no-wt").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore-no-wt");
        let sessions_dir = unique_temp_dir("sessions-no-wt");
        let session_store = Arc::new(DiskSessionStore::new(&sessions_dir).unwrap());
        let id = SessionId::new_v4();

        // Seed a session whose worktree_path is a non-existent directory.
        {
            let fake_wt = std::env::temp_dir().join("does-not-exist-intentionally");
            let session = domain::Session::new(id, workspace_root.clone(), fake_wt);
            app::SessionStore::save(session_store.as_ref(), &session)
                .await
                .unwrap();
        }
        {
            let layout = DiskLayoutRepository::load(layout_path.clone())
                .await
                .unwrap();
            layout
                .put(&workspace_root, Layout::new(vec![id], vec![1.0]))
                .await
                .unwrap();
        }

        let layout = DiskLayoutRepository::load(layout_path).await.unwrap();
        let (spawner, mut rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir,
            resume: None,
            session_id: None,
            env: vec![],
        };

        let registry = SessionRegistry::restore(
            spawner,
            spawn_config,
            Arc::new(layout),
            workspace_root,
            Arc::new(agent_host::fake::NoopCloseRequestSink),
            Arc::new(agent_host::fake::NoopFirstTurnSink),
            session_store,
        )
        .await
        .expect("restore ok");

        assert!(registry.is_empty());
        assert!(rx.try_recv().is_err(), "spawner should not be called");
    }

    #[tokio::test]
    async fn restore_spawns_saved_session_using_its_worktree_path() {
        // Happy path: saved session file points to an existing worktree
        // directory. `restore` must spawn an agent whose config
        // `workspace_root` equals the saved worktree path — that's the
        // guarantee that makes resume land on the right branch.
        let layout_path = unique_temp_dir("restore-happy").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore-happy");
        let sessions_dir = unique_temp_dir("sessions-happy");
        let session_store = Arc::new(DiskSessionStore::new(&sessions_dir).unwrap());
        let id = SessionId::new_v4();
        let worktree = seed_saved_session(&session_store, id, &workspace_root, "wt-happy").await;
        {
            let layout = DiskLayoutRepository::load(layout_path.clone())
                .await
                .unwrap();
            layout
                .put(&workspace_root, Layout::new(vec![id], vec![1.0]))
                .await
                .unwrap();
        }

        let layout = DiskLayoutRepository::load(layout_path).await.unwrap();
        let (spawner, mut rx) = DuplexSpawner::new();
        let spawn_config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir,
            resume: None,
            session_id: None,
            env: vec![],
        };

        let restore_fut = tokio::spawn(async move {
            SessionRegistry::restore(
                spawner,
                spawn_config,
                Arc::new(layout),
                workspace_root.clone(),
                Arc::new(agent_host::fake::NoopCloseRequestSink),
                Arc::new(agent_host::fake::NoopFirstTurnSink),
                session_store,
            )
            .await
        });

        let mut handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("handles channel dropped");
        // The spawner must have seen the saved worktree as the agent's
        // workspace_root, and `resume: Some(id)` so the agent rehydrates
        // the right session.
        assert_eq!(handles.config.workspace_root, worktree);
        assert_eq!(handles.config.resume, Some(id));
        handles.send_ready(id).await;

        let registry = timeout(Duration::from_secs(2), restore_fut)
            .await
            .expect("restore timed out")
            .expect("join")
            .expect("restore ok");
        assert!(!registry.is_empty());
        drop(handles);
    }

    #[tokio::test]
    async fn restore_continues_after_per_session_spawn_failure() {
        // Two rows in the saved layout. The spawner fails for every
        // call — both rows are skipped, and `restore` still returns
        // Ok with an empty registry. The caller (main.rs) decides
        // whether to bootstrap a fresh session afterwards.
        let layout_path = unique_temp_dir("restore-spawn-fail").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore-spawn-fail");
        let sessions_dir = unique_temp_dir("sessions-spawn-fail");
        let session_store = Arc::new(DiskSessionStore::new(&sessions_dir).unwrap());
        let a = SessionId::new_v4();
        let b = SessionId::new_v4();
        let _wt_a = seed_saved_session(&session_store, a, &workspace_root, "wt-fail-a").await;
        let _wt_b = seed_saved_session(&session_store, b, &workspace_root, "wt-fail-b").await;
        {
            let layout = DiskLayoutRepository::load(layout_path.clone())
                .await
                .unwrap();
            layout
                .put(&workspace_root, Layout::new(vec![a, b], vec![0.5, 0.5]))
                .await
                .unwrap();
        }

        let layout = DiskLayoutRepository::load(layout_path).await.unwrap();
        let spawner = DuplexSpawner::failing();
        let spawn_config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir,
            resume: None,
            session_id: None,
            env: vec![],
        };

        let registry = SessionRegistry::restore(
            spawner,
            spawn_config,
            Arc::new(layout),
            workspace_root,
            Arc::new(agent_host::fake::NoopCloseRequestSink),
            Arc::new(agent_host::fake::NoopFirstTurnSink),
            session_store,
        )
        .await
        .expect("restore should be resilient to spawn failures");
        assert!(registry.is_empty());
    }

    // -- persist_current_layout ------------------------------------------

    #[tokio::test]
    async fn persist_current_layout_writes_equal_sizes() {
        let layout_path = unique_temp_dir("persist-current").join("workspaces.json");
        let store = DiskLayoutRepository::load(layout_path.clone()).await.unwrap();
        let workspace_root = unique_temp_dir("ws-persist-current");
        let (spawner, mut rx) = DuplexSpawner::new();
        let registry = SessionRegistry::new(
            spawner,
            AgentSpawnConfig {
                binary: PathBuf::from("/nonexistent/ox-agent"),
                workspace_root: workspace_root.clone(),
                model: "test/model".into(),
                sessions_dir: PathBuf::from("/nonexistent/sessions"),
                resume: None,
                session_id: None,
                env: vec![],
            },
            Arc::new(store),
            workspace_root.clone(),
            Arc::new(agent_host::fake::NoopCloseRequestSink),
            Arc::new(agent_host::fake::NoopFirstTurnSink),
        );
        let (_a, _ha) = create_session(registry.clone(), &mut rx).await;
        let (_b, _hb) = create_session(registry.clone(), &mut rx).await;

        registry
            .persist_current_layout()
            .await
            .expect("persist_current_layout");

        let reloaded = DiskLayoutRepository::load(layout_path).await.unwrap();
        let got = reloaded
            .get(&workspace_root)
            .await
            .expect("layout read")
            .expect("row");
        assert_eq!(got.order.len(), 2);
        // Equal-size fallback: both entries should be ~0.5.
        for s in &got.sizes {
            assert!((*s - 0.5).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn pump_does_not_drop_events_during_subscribe_handshake() {
        // Publishing during the subscribe handshake must not lose or
        // duplicate events. The test alternates rapid publishes from
        // the agent side with subscribes from the handler side and
        // checks that every subscriber sees every event emitted before
        // it subscribed (via the replay snapshot) plus every event
        // after (via the broadcast).
        let (registry, mut rx, _ws) = test_registry(empty_layout().await).await;
        let (id, mut handles) = create_session(registry.clone(), &mut rx).await;
        let session = registry.get(id).expect("session");

        // Pre-populate some history so every subscriber has a non-
        // trivial replay to chew through.
        for i in 0..20u32 {
            handles
                .send_event(&AgentEvent::Error {
                    message: format!("pre-{i}"),
                })
                .await;
        }
        // Give the pump a moment to drain. We don't want the test to
        // race its own precondition.
        for _ in 0..50 {
            let (snapshot, _rx2) = session.subscribe();
            if snapshot.len() >= 21 {
                // 1 Ready + 20 errors
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Interleave publishes with subscribes. Each subscribe locks
        // history + clones it + subscribes; each publish locks history
        // + appends + broadcasts. The shared mutex guarantees every
        // post-subscribe publish arrives on the receiver; pre-subscribe
        // publishes arrive in the snapshot.
        let mut total_subscribers = vec![];
        for i in 20..25u32 {
            let (snapshot, mut rx) = session.subscribe();
            handles
                .send_event(&AgentEvent::Error {
                    message: format!("post-{i}"),
                })
                .await;
            // The new event shows up on the broadcast receiver.
            let received = timeout(Duration::from_secs(1), rx.recv())
                .await
                .expect("broadcast timeout")
                .expect("broadcast closed");
            match received {
                AgentEvent::Error { message } => {
                    assert_eq!(message, format!("post-{i}"));
                }
                other => panic!("unexpected event: {other:?}"),
            }
            total_subscribers.push(snapshot.len());
        }

        // Each successive snapshot must be at least as long as the
        // previous (history only grows). No duplicates — strict inequality
        // between iterations because each loop publishes one event.
        for pair in total_subscribers.windows(2) {
            assert!(pair[1] > pair[0], "snapshot history did not grow: {pair:?}");
        }
    }
}
