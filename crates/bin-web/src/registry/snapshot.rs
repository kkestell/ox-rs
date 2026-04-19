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
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub model: String,
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
}
