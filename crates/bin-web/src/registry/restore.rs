use std::path::PathBuf;
use std::sync::Arc;

use agent_host::{AgentSpawnConfig, AgentSpawner, CloseRequestSink, FirstTurnSink, LayoutRepository};
use anyhow::{Context, Result, anyhow};
use app::SessionStore;
use domain::SessionId;

use super::SessionRegistry;

impl SessionRegistry {
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
}
