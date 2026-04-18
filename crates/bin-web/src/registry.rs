//! `SessionRegistry` — the server's entire session-state model.
//!
//! The registry owns:
//!
//! - a map of live sessions keyed by [`SessionId`],
//! - the workspace root this server instance is scoped to,
//! - the [`LayoutStore`] that persists pane order / sizes across runs,
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
use std::sync::{Arc, Mutex, RwLock};

use agent_host::{AgentSpawnConfig, AgentSpawner, Layout, LayoutStore};
use anyhow::{Context, Result, anyhow};
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
}

pub struct SessionRegistry {
    sessions: RwLock<HashMap<SessionId, Arc<ActiveSession>>>,
    spawner: Arc<dyn AgentSpawner>,
    spawn_config: AgentSpawnConfig,
    layout: Mutex<LayoutStore>,
    workspace_root: PathBuf,
}

impl SessionRegistry {
    /// Build an empty registry. `restore` is the usual entry point on
    /// startup; `new` exists for tests that want an empty registry.
    pub fn new(
        spawner: Arc<dyn AgentSpawner>,
        spawn_config: AgentSpawnConfig,
        layout: LayoutStore,
        workspace_root: PathBuf,
    ) -> Arc<Self> {
        Arc::new(Self {
            sessions: RwLock::new(HashMap::new()),
            spawner,
            spawn_config,
            layout: Mutex::new(layout),
            workspace_root,
        })
    }

    /// Attempt to resume every saved session for this workspace root.
    /// Sessions that fail to spawn (agent binary missing, session file
    /// deleted, etc.) are skipped; the registry falls back to a single
    /// fresh session if nothing resumes so the user always has a pane
    /// to type into.
    pub async fn restore(
        spawner: Arc<dyn AgentSpawner>,
        spawn_config: AgentSpawnConfig,
        layout: LayoutStore,
        workspace_root: PathBuf,
    ) -> Result<Arc<Self>> {
        let registry = Self::new(spawner, spawn_config, layout, workspace_root.clone());

        let saved_order: Vec<SessionId> = {
            let layout = registry.layout.lock().expect("layout mutex poisoned");
            layout
                .get(&workspace_root)
                .map(|l| l.order.clone())
                .unwrap_or_default()
        };

        let mut resumed: Vec<SessionId> = Vec::new();
        for id in saved_order {
            match registry.spawn_and_insert(Some(id)).await {
                Ok(new_id) => resumed.push(new_id),
                Err(err) => {
                    // A failed resume is non-fatal — the agent binary
                    // may be out of date, the session file may have
                    // been deleted by hand, etc. Log and move on so
                    // the rest of the saved layout can still restore.
                    eprintln!("ox: failed to resume session {id}: {err:#}");
                }
            }
        }

        if resumed.is_empty() {
            match registry.spawn_and_insert(None).await {
                Ok(id) => {
                    // Persist a one-pane layout so the next restart
                    // finds something to resume. Any client-initiated
                    // PUT will overwrite this with the real sizes.
                    let layout = Layout::new(vec![id], vec![1.0]);
                    registry.put_layout(layout)?;
                }
                Err(err) => {
                    return Err(err.context("failed to start the initial session"));
                }
            }
        }

        Ok(registry)
    }

    /// Spawn a fresh agent (no resume) and register it. Returns the
    /// session id reported by the agent's `Ready` frame.
    pub async fn create(&self) -> Result<SessionId> {
        self.spawn_and_insert(None).await
    }

    /// Spawn an agent that resumes a specific session id. Used by the
    /// CLI's `--resume` path to bypass the saved layout and open one
    /// named session instead.
    pub async fn create_resumed(&self, id: SessionId) -> Result<SessionId> {
        self.spawn_and_insert(Some(id)).await
    }

    /// Drop the session from the registry. The `Arc<ActiveSession>`
    /// held by the map is released; its `AgentClient` drops; the child
    /// dies via `kill_on_drop`. Returns true if the session existed.
    pub fn remove(&self, id: SessionId) -> bool {
        let mut sessions = self.sessions.write().expect("sessions lock poisoned");
        sessions.remove(&id).is_some()
    }

    /// Look up a session by id, bumping its `Arc` count for the
    /// handler's lifetime.
    pub fn get(&self, id: SessionId) -> Option<Arc<ActiveSession>> {
        let sessions = self.sessions.read().expect("sessions lock poisoned");
        sessions.get(&id).cloned()
    }

    /// Snapshot for `GET /api/sessions`.
    pub fn snapshot(&self) -> SnapshotJson {
        let sessions = self.sessions.read().expect("sessions lock poisoned");
        let layout = self
            .layout
            .lock()
            .expect("layout mutex poisoned")
            .get(&self.workspace_root)
            .cloned()
            .unwrap_or_default();

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
    /// [`LayoutStore::put`] normalizes sizes.
    pub fn put_layout(&self, mut layout: Layout) -> Result<()> {
        {
            let sessions = self.sessions.read().expect("sessions lock poisoned");
            // Filter order by known ids, preserving sizes by index.
            // After filtering we may have fewer sizes than entries,
            // which `LayoutStore::put` re-normalizes to equal widths.
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

        let mut store = self.layout.lock().expect("layout mutex poisoned");
        store.put(&self.workspace_root, layout)
    }

    /// Persist the current registry state as a one-row layout with
    /// equal sizes. Used on graceful shutdown when the server wants
    /// to make sure a restart can find the current set of sessions.
    pub fn persist_current_layout(&self) -> Result<()> {
        let ids: Vec<SessionId> = {
            let sessions = self.sessions.read().expect("sessions lock poisoned");
            sessions.keys().copied().collect()
        };
        if ids.is_empty() {
            return Ok(());
        }
        let n = ids.len();
        let layout = Layout::new(ids, vec![1.0 / n as f32; n]);
        let mut store = self.layout.lock().expect("layout mutex poisoned");
        store.put(&self.workspace_root, layout)
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

    /// Shared path for `create` and `restore`: spawn the agent,
    /// await `Ready`, and insert it into the map.
    async fn spawn_and_insert(&self, resume: Option<SessionId>) -> Result<SessionId> {
        let mut config = self.spawn_config.clone();
        config.resume = resume;

        let (client, stream) = self.spawner.spawn(config).context("spawning agent")?;
        let session = ActiveSession::start(client, stream);

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

    use agent_host::{AgentSpawnConfig, Layout, LayoutStore};
    use domain::SessionId;
    use protocol::AgentEvent;
    use tokio::time::timeout;

    use super::*;
    use crate::test_support::{DuplexSpawner, empty_layout, test_registry, unique_temp_dir};

    /// Drive `registry.create()` to completion by receiving the
    /// agent-side handles, writing `Ready(id)`, and awaiting the
    /// registry's returned session id. Returns the id and the live
    /// agent handles so the test can keep interacting with them.
    async fn create_session(
        registry: Arc<SessionRegistry>,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::test_support::AgentHandles>,
    ) -> (SessionId, crate::test_support::AgentHandles) {
        let create = tokio::spawn({
            let r = registry.clone();
            async move { r.create().await }
        });
        let mut handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("spawner receiver dropped");
        let id = SessionId::new_v4();
        handles.send_ready(id).await;
        let ret_id = timeout(Duration::from_secs(2), create)
            .await
            .expect("create timed out")
            .expect("join failure")
            .expect("create returned Err");
        assert_eq!(ret_id, id);
        (id, handles)
    }

    // -- create ----------------------------------------------------------

    #[tokio::test]
    async fn create_awaits_ready_then_registers_session() {
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
        let (id, _handles) = create_session(registry.clone(), &mut rx).await;

        // Snapshot must now include the session in the order/sessions list.
        let snap = registry.snapshot();
        assert!(snap.sessions.iter().any(|s| s.session_id == id));
        assert!(registry.get(id).is_some());
    }

    #[tokio::test]
    async fn create_returns_error_if_agent_exits_before_ready() {
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
        let create = tokio::spawn({
            let r = registry.clone();
            async move { r.create().await }
        });
        let handles = timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("spawner timed out")
            .expect("spawner dropped");
        // Drop the handles without writing `Ready`. The agent stdout
        // side closes, the pump observes `None`, alive flips false,
        // await_ready resolves None, and create returns an Err.
        drop(handles);
        let result = timeout(Duration::from_secs(2), create)
            .await
            .expect("create timed out")
            .expect("join failure");
        assert!(result.is_err(), "expected create to fail");
    }

    // -- remove ----------------------------------------------------------

    #[tokio::test]
    async fn remove_returns_true_once_then_false() {
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, ws) = test_registry(empty_layout()).await;
        let (first, _h1) = create_session(registry.clone(), &mut rx).await;
        let (second, _h2) = create_session(registry.clone(), &mut rx).await;

        // Persist a layout that inverts insertion order.
        registry
            .put_layout(Layout::new(vec![second, first], vec![0.3, 0.7]))
            .expect("put_layout");

        let snap = registry.snapshot();
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
        let (registry, _rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
        let (known, _handles) = create_session(registry.clone(), &mut rx).await;
        let unknown = SessionId::new_v4();

        registry
            .put_layout(Layout::new(vec![unknown, known], vec![0.5, 0.5]))
            .expect("put_layout");

        let snap = registry.snapshot();
        // The unknown id is filtered; the known id survives.
        assert_eq!(snap.layout.order, vec![known]);
    }

    #[tokio::test]
    async fn put_layout_persists_to_disk() {
        // A PUT-then-reload cycle must round-trip order and (normalized)
        // sizes so a restart picks up the same pane tiling.
        let layout_path = unique_temp_dir("persist").join("workspaces.json");
        let store = LayoutStore::load(layout_path.clone()).unwrap();
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
                env: vec![],
            },
            store,
            workspace_root.clone(),
        );
        let (id, _handles) = create_session(registry.clone(), &mut rx).await;
        registry
            .put_layout(Layout::new(vec![id], vec![1.0]))
            .expect("put_layout");

        // Reload and confirm the file contains the expected entry.
        let reloaded = LayoutStore::load(layout_path).unwrap();
        let got = reloaded.get(&workspace_root).expect("layout row");
        assert_eq!(got.order, vec![id]);
        assert_eq!(got.sizes, vec![1.0]);
    }

    #[tokio::test]
    async fn put_layout_normalizes_non_finite_sizes() {
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
        let (a, _ha) = create_session(registry.clone(), &mut rx).await;
        let (b, _hb) = create_session(registry.clone(), &mut rx).await;

        registry
            .put_layout(Layout::new(vec![a, b], vec![f32::NAN, 0.5]))
            .expect("put_layout");
        let snap = registry.snapshot();
        for s in &snap.layout.sizes {
            assert!(s.is_finite());
        }
        let sum: f32 = snap.layout.sizes.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    // -- restore ---------------------------------------------------------

    #[tokio::test]
    async fn restore_falls_back_to_fresh_session_when_every_resume_fails() {
        // A failing spawner simulates "agent binary missing": every
        // resume attempt returns Err. `restore` must still hand back a
        // registry, spawning a new fresh session if nothing resumed.
        //
        // The fallback path re-uses the same spawner, so we can't use
        // `DuplexSpawner::failing()` here — it fails on every spawn,
        // including the fallback. Instead we use a two-phase spawner
        // that fails the resumed calls and succeeds for a fresh spawn.
        let layout_path = unique_temp_dir("restore-fallback").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore");
        let stale_id = SessionId::new_v4();
        {
            let mut store = LayoutStore::load(layout_path.clone()).unwrap();
            store
                .put(&workspace_root, Layout::new(vec![stale_id], vec![1.0]))
                .unwrap();
        }

        let store = LayoutStore::load(layout_path.clone()).unwrap();
        let spawner = Arc::new(FailThenSucceedSpawner::new(1));
        let mut handles_rx = spawner.rx();
        let spawn_config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir: PathBuf::from("/nonexistent/sessions"),
            resume: None,
            env: vec![],
        };

        let restore = tokio::spawn(async move {
            SessionRegistry::restore(spawner, spawn_config, store, workspace_root.clone()).await
        });

        // The failing attempt doesn't push handles. The fresh fallback
        // does — we receive it and write Ready so `restore` resolves.
        let mut handles = timeout(Duration::from_secs(2), handles_rx.recv())
            .await
            .expect("fallback spawner timed out")
            .expect("channel dropped");
        let fresh_id = SessionId::new_v4();
        handles.send_ready(fresh_id).await;
        // Keep handles alive so `kill_on_drop` doesn't fire on the
        // fallback session before the test finishes.
        let registry = timeout(Duration::from_secs(2), restore)
            .await
            .expect("restore timed out")
            .expect("join failure")
            .expect("restore returned Err");
        let snap = registry.snapshot();
        assert_eq!(snap.sessions.len(), 1);
        assert_eq!(snap.sessions[0].session_id, fresh_id);
        drop(handles);
    }

    /// Spawner that fails on the first `n` calls, then succeeds with
    /// duplex pipes just like `DuplexSpawner`.
    struct FailThenSucceedSpawner {
        counter: std::sync::Mutex<usize>,
        fail_count: usize,
        tx: tokio::sync::mpsc::UnboundedSender<crate::test_support::AgentHandles>,
        rx: std::sync::Mutex<
            Option<tokio::sync::mpsc::UnboundedReceiver<crate::test_support::AgentHandles>>,
        >,
    }

    impl FailThenSucceedSpawner {
        fn new(fail_count: usize) -> Self {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            Self {
                counter: std::sync::Mutex::new(0),
                fail_count,
                tx,
                rx: std::sync::Mutex::new(Some(rx)),
            }
        }

        fn rx(&self) -> tokio::sync::mpsc::UnboundedReceiver<crate::test_support::AgentHandles> {
            self.rx.lock().unwrap().take().expect("rx already taken")
        }
    }

    impl agent_host::AgentSpawner for FailThenSucceedSpawner {
        fn spawn(
            &self,
            config: AgentSpawnConfig,
        ) -> anyhow::Result<(agent_host::AgentClient, agent_host::AgentEventStream)> {
            let mut n = self.counter.lock().unwrap();
            if *n < self.fail_count {
                *n += 1;
                return Err(anyhow::anyhow!("simulated spawn failure #{}", *n));
            }
            *n += 1;
            drop(n);
            let (agent_writer, client_reader) = tokio::io::duplex(64 * 1024);
            let (client_writer, agent_reader) = tokio::io::duplex(64 * 1024);
            let (client, stream) = agent_host::AgentClient::new(
                tokio::io::BufReader::new(client_reader),
                client_writer,
            );
            let handles = crate::test_support::AgentHandles {
                reader: tokio::io::BufReader::new(agent_reader),
                writer: agent_writer,
                config,
            };
            self.tx
                .send(handles)
                .map_err(|_| anyhow::anyhow!("rx dropped"))?;
            Ok((client, stream))
        }
    }

    #[tokio::test]
    async fn restore_errors_if_fallback_spawn_also_fails() {
        // When every resume fails *and* the fresh-fallback spawn also
        // fails, `restore` surfaces the error to the caller rather than
        // silently pretending everything is fine.
        let layout_path = unique_temp_dir("restore-all-fail").join("workspaces.json");
        let workspace_root = unique_temp_dir("ws-restore-all-fail");
        let stale_id = SessionId::new_v4();
        {
            let mut store = LayoutStore::load(layout_path.clone()).unwrap();
            store
                .put(&workspace_root, Layout::new(vec![stale_id], vec![1.0]))
                .unwrap();
        }

        let store = LayoutStore::load(layout_path).unwrap();
        let spawner = DuplexSpawner::failing();
        let config = AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: workspace_root.clone(),
            model: "test/model".into(),
            sessions_dir: PathBuf::from("/nonexistent/sessions"),
            resume: None,
            env: vec![],
        };
        let result = SessionRegistry::restore(spawner, config, store, workspace_root).await;
        assert!(result.is_err(), "restore should error when fallback fails");
    }

    // -- persist_current_layout ------------------------------------------

    #[tokio::test]
    async fn persist_current_layout_writes_equal_sizes() {
        let layout_path = unique_temp_dir("persist-current").join("workspaces.json");
        let store = LayoutStore::load(layout_path.clone()).unwrap();
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
                env: vec![],
            },
            store,
            workspace_root.clone(),
        );
        let (_a, _ha) = create_session(registry.clone(), &mut rx).await;
        let (_b, _hb) = create_session(registry.clone(), &mut rx).await;

        registry
            .persist_current_layout()
            .expect("persist_current_layout");

        let reloaded = LayoutStore::load(layout_path).unwrap();
        let got = reloaded.get(&workspace_root).expect("row");
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
        let (registry, mut rx, _ws) = test_registry(empty_layout()).await;
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
