use std::path::PathBuf;

use agent_host::Layout;
use domain::SessionId;
use serde::Serialize;

use super::SessionRegistry;

/// JSON payload for `GET /api/sessions` — everything the frontend
/// needs on first paint. Snapshotted under the registry's read lock,
/// so any concurrent `create`/`remove` appears as "not yet" or "gone,"
/// never as half-applied state.
#[derive(Debug, Serialize)]
pub struct SnapshotJson {
    pub workspace_root: PathBuf,
    pub sessions: Vec<SessionSummary>,
    pub layout: Layout,
    /// Registry-wide token context window. Duplicates the value on each
    /// [`SessionSummary`] (they all share the same model today), but
    /// surfaced at the top level so a fresh workspace with no sessions
    /// can still populate the usage-chip denominator before the first
    /// session is created. Resolved from the registry's catalog against
    /// `spawn_config.model`; defaults to 0 if the catalog is silent
    /// (shouldn't happen — main.rs validates at startup — but treated
    /// as "unknown" here rather than panicking at snapshot time).
    pub context_window: u32,
    /// Registry-wide model slug. All sessions share the same model today;
    /// surfaced at the top level so the frontend can display it without
    /// waiting for the first session to be created.
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub model: String,
    /// Token context window for this session's model, sourced from the
    /// OpenRouter catalog. Resolved per-session at snapshot time so a
    /// future per-session model override lands on the wire without any
    /// further plumbing work. Rendered as the denominator of the
    /// frontend's usage chip.
    pub context_window: u32,
}

impl SessionRegistry {
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
        let default_model = self.default_model();
        // Every session today uses `spawn_config.model`; resolving once
        // here is equivalent to resolving per session and saves a
        // catalog lookup per entry. If per-session models arrive later,
        // this switches to per-session resolution without changing the
        // wire payload.
        let context_window = self
            .catalog()
            .context_window(default_model)
            .unwrap_or(0);

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
                    model: default_model.to_owned(),
                    context_window,
                });
                seen.insert(*id);
            }
        }
        for id in sessions.keys() {
            if !seen.contains(id) {
                ordered.push(SessionSummary {
                    session_id: *id,
                    model: default_model.to_owned(),
                    context_window,
                });
            }
        }

        SnapshotJson {
            workspace_root: self.workspace_root.clone(),
            sessions: ordered,
            layout,
            context_window,
            model: default_model.to_owned(),
        }
    }
}
