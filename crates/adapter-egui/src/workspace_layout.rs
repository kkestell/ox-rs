use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use domain::SessionId;
use serde::{Deserialize, Serialize};

const FRACTION_SUM_EPSILON: f32 = 0.01;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct WorkspaceLayouts {
    pub(super) workspaces: BTreeMap<String, SavedWorkspaceLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct SavedWorkspaceLayout {
    pub(super) sessions: Vec<SessionId>,
    pub(super) split_fracs: Vec<f32>,
    pub(super) focused: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RestoreLayout {
    pub(super) sessions: Vec<SessionId>,
    pub(super) split_fracs: Vec<f32>,
    pub(super) focused: usize,
}

impl WorkspaceLayouts {
    pub(super) fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading workspace layout file {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing workspace layout file {}", path.display()))
    }

    pub(super) fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating workspace layout directory {}", parent.display())
            })?;
        }
        let tmp_path = tmp_path_for(path);
        let text = serde_json::to_string_pretty(self).context("serializing workspace layouts")?;
        fs::write(&tmp_path, text)
            .with_context(|| format!("writing workspace layout file {}", tmp_path.display()))?;
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "replacing workspace layout file {} with {}",
                path.display(),
                tmp_path.display()
            )
        })?;
        Ok(())
    }

    pub(super) fn restore_for(&self, workspace_root: &Path) -> Option<RestoreLayout> {
        self.workspaces
            .get(&workspace_key(workspace_root))
            .and_then(SavedWorkspaceLayout::normalized_for_restore)
    }

    pub(super) fn restore_existing_for(
        &self,
        workspace_root: &Path,
        sessions_dir: &Path,
    ) -> Option<RestoreLayout> {
        let layout = self.restore_for(workspace_root)?;
        let sessions: Vec<SessionId> = layout
            .sessions
            .into_iter()
            .filter(|id| sessions_dir.join(format!("{id}.json")).exists())
            .collect();
        if sessions.is_empty() {
            return None;
        }
        Some(RestoreLayout {
            split_fracs: normalize_split_fracs(&layout.split_fracs, sessions.len()),
            focused: layout.focused.min(sessions.len() - 1),
            sessions,
        })
    }

    pub(super) fn save_current(
        &mut self,
        workspace_root: &Path,
        session_ids: impl IntoIterator<Item = Option<SessionId>>,
        split_fracs: &[f32],
        focused: usize,
    ) -> bool {
        let sessions: Vec<SessionId> = session_ids.into_iter().flatten().collect();
        if sessions.is_empty() {
            return false;
        }
        let split_fracs = normalize_split_fracs(split_fracs, sessions.len());
        let focused = focused.min(sessions.len() - 1);
        self.workspaces.insert(
            workspace_key(workspace_root),
            SavedWorkspaceLayout {
                sessions,
                split_fracs,
                focused,
            },
        );
        true
    }
}

impl SavedWorkspaceLayout {
    fn normalized_for_restore(&self) -> Option<RestoreLayout> {
        if self.sessions.is_empty() {
            return None;
        }
        Some(RestoreLayout {
            sessions: self.sessions.clone(),
            split_fracs: normalize_split_fracs(&self.split_fracs, self.sessions.len()),
            focused: self.focused.min(self.sessions.len() - 1),
        })
    }
}

pub(super) fn normalize_split_fracs(fracs: &[f32], len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let equal = || vec![1.0 / len as f32; len];
    if fracs.len() != len || fracs.iter().any(|f| !f.is_finite() || *f <= 0.0) {
        return equal();
    }
    let sum: f32 = fracs.iter().sum();
    if (sum - 1.0).abs() > FRACTION_SUM_EPSILON {
        return equal();
    }
    fracs.iter().map(|f| f / sum).collect()
}

fn workspace_key(workspace_root: &Path) -> String {
    workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!("{ext}.tmp"))
        .unwrap_or_else(|| "tmp".to_owned());
    tmp.set_extension(extension);
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ox-layout-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn normalize_rejects_mismatched_or_bad_fractions() {
        assert_eq!(normalize_split_fracs(&[0.25], 2), vec![0.5, 0.5]);
        assert_eq!(normalize_split_fracs(&[0.0, 1.0], 2), vec![0.5, 0.5]);
        assert_eq!(normalize_split_fracs(&[0.25, 0.25], 2), vec![0.5, 0.5]);
        assert_eq!(normalize_split_fracs(&[0.25, 0.75], 2), vec![0.25, 0.75]);
    }

    #[test]
    fn save_current_drops_unknown_session_ids_and_clamps_focus() {
        let workspace = temp_dir("save-current");
        let mut layouts = WorkspaceLayouts::default();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();

        assert!(layouts.save_current(
            &workspace,
            [Some(id1), None, Some(id2)],
            &[0.2, 0.3, 0.5],
            99,
        ));
        let restored = layouts.restore_for(&workspace).unwrap();
        assert_eq!(restored.sessions, vec![id1, id2]);
        assert_eq!(restored.split_fracs, vec![0.5, 0.5]);
        assert_eq!(restored.focused, 1);
    }

    #[test]
    fn save_current_does_not_overwrite_with_empty_sessions() {
        let workspace = temp_dir("empty-save");
        let mut layouts = WorkspaceLayouts::default();
        let id = SessionId::new_v4();
        assert!(layouts.save_current(&workspace, [Some(id)], &[1.0], 0));
        assert!(!layouts.save_current(&workspace, [None], &[1.0], 0));
        assert_eq!(layouts.restore_for(&workspace).unwrap().sessions, vec![id]);
    }

    #[test]
    fn load_missing_file_returns_empty_layouts() {
        let dir = temp_dir("missing");
        let layouts = WorkspaceLayouts::load(&dir.join("workspaces.json")).unwrap();
        assert!(layouts.workspaces.is_empty());
    }

    #[test]
    fn save_and_load_round_trips_layouts() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("workspaces.json");
        let workspace = dir.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let mut layouts = WorkspaceLayouts::default();
        let id = SessionId::new_v4();
        layouts.save_current(&workspace, [Some(id)], &[1.0], 0);

        layouts.save(&path).unwrap();
        let loaded = WorkspaceLayouts::load(&path).unwrap();

        assert_eq!(loaded.restore_for(&workspace).unwrap().sessions, vec![id]);
    }

    #[test]
    fn empty_saved_sessions_are_not_restored() {
        let workspace = temp_dir("empty-restore");
        let mut layouts = WorkspaceLayouts::default();
        layouts.workspaces.insert(
            workspace_key(&workspace),
            SavedWorkspaceLayout {
                sessions: Vec::new(),
                split_fracs: Vec::new(),
                focused: 0,
            },
        );

        assert!(layouts.restore_for(&workspace).is_none());
    }

    #[test]
    fn restore_existing_filters_missing_session_files() {
        let dir = temp_dir("existing-filter");
        let workspace = dir.join("workspace");
        let sessions_dir = dir.join("sessions");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&sessions_dir).unwrap();
        let keep = SessionId::new_v4();
        let missing = SessionId::new_v4();
        fs::write(sessions_dir.join(format!("{keep}.json")), "{}").unwrap();

        let mut layouts = WorkspaceLayouts::default();
        layouts.save_current(&workspace, [Some(keep), Some(missing)], &[0.25, 0.75], 1);

        let restored = layouts
            .restore_existing_for(&workspace, &sessions_dir)
            .unwrap();
        assert_eq!(restored.sessions, vec![keep]);
        assert_eq!(restored.split_fracs, vec![1.0]);
        assert_eq!(restored.focused, 0);
    }

    #[test]
    fn restore_existing_ignores_layout_when_all_session_files_are_missing() {
        let dir = temp_dir("existing-none");
        let workspace = dir.join("workspace");
        let sessions_dir = dir.join("sessions");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&sessions_dir).unwrap();

        let mut layouts = WorkspaceLayouts::default();
        layouts.save_current(&workspace, [Some(SessionId::new_v4())], &[1.0], 0);

        assert!(
            layouts
                .restore_existing_for(&workspace, &sessions_dir)
                .is_none()
        );
    }
}
