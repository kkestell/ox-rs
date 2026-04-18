//! Workspace layout persistence.
//!
//! `LayoutStore` is a flat `workspace-root → Layout` map serialized as
//! JSON at `~/.ox/workspaces.json`. Each [`Layout`] records the session
//! IDs that made up a workspace's panes and the horizontal fraction each
//! pane occupied, so a later launch can rebuild the tiled UI.
//!
//! No `focused` field: focus is pure client-side UI state (the browser
//! doesn't need the server to remember which tab it last clicked on).
//! Files written by the deleted GTK app carrying `focused` round-trip
//! cleanly — unknown fields are ignored by serde.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use domain::SessionId;
use serde::{Deserialize, Serialize};

/// Tolerance used when deciding whether persisted sizes already sum to
/// 1.0. Values outside this band are rejected and replaced with equal
/// sizes so a corrupted entry can't produce pathological widths.
const SIZE_SUM_EPSILON: f32 = 0.01;

/// The on-disk aggregate. Lives behind `LayoutStore` so every mutation
/// flows through `put`, which persists the file atomically.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct LayoutFile {
    /// Using `BTreeMap` keeps the on-disk order stable and the diffs
    /// reviewable when eyeballing the JSON by hand.
    #[serde(default)]
    workspaces: BTreeMap<String, Layout>,
}

/// A single workspace's saved layout. `order` and `sizes` line up
/// positionally: `sizes[i]` is the width fraction of the pane showing
/// `order[i]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Layout {
    #[serde(default)]
    pub order: Vec<SessionId>,
    #[serde(default)]
    pub sizes: Vec<f32>,
}

impl Layout {
    pub fn new(order: Vec<SessionId>, sizes: Vec<f32>) -> Self {
        Self { order, sizes }
    }
}

/// File-backed store of [`Layout`] entries keyed by workspace root.
///
/// The store caches the decoded file in memory so `get` is a cheap
/// lookup, and writes the whole blob on each `put`. The data is small
/// (a few UUIDs plus floats per workspace) so that trade is fine.
#[derive(Debug)]
pub struct LayoutStore {
    path: PathBuf,
    data: LayoutFile,
}

impl LayoutStore {
    /// Load the store from `path`. A missing file yields an empty store;
    /// a corrupt file yields an error — callers decide whether to fall
    /// back to an empty store (the server logs and keeps running).
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                data: LayoutFile::default(),
            });
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("reading workspace layout file {}", path.display()))?;
        let data: LayoutFile = serde_json::from_str(&text)
            .with_context(|| format!("parsing workspace layout file {}", path.display()))?;
        Ok(Self { path, data })
    }

    /// Path the store persists to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Look up the saved layout for `workspace_root`, if any. Returns
    /// `None` when nothing has been persisted for that root yet.
    pub fn get(&self, workspace_root: &Path) -> Option<&Layout> {
        self.data.workspaces.get(&workspace_key(workspace_root))
    }

    /// Replace the layout for `workspace_root` and persist the whole
    /// store atomically (tmp-write then rename). Sizes are normalized
    /// before writing so callers don't need to pre-validate user input.
    pub fn put(&mut self, workspace_root: &Path, mut layout: Layout) -> Result<()> {
        normalize_sizes(&mut layout.sizes, layout.order.len());
        self.data
            .workspaces
            .insert(workspace_key(workspace_root), layout);
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating workspace layout directory {}", parent.display())
            })?;
        }
        let tmp_path = tmp_path_for(&self.path);
        let text =
            serde_json::to_string_pretty(&self.data).context("serializing workspace layouts")?;
        fs::write(&tmp_path, text)
            .with_context(|| format!("writing workspace layout file {}", tmp_path.display()))?;
        fs::rename(&tmp_path, &self.path).with_context(|| {
            format!(
                "replacing workspace layout file {} with {}",
                self.path.display(),
                tmp_path.display()
            )
        })?;
        Ok(())
    }
}

/// Normalize `sizes` in place so it has `len` non-negative entries that
/// sum to 1.0. Pathological inputs (empty, wrong length, any NaN/inf,
/// any negative, sum far from 1) collapse to equal sizes. When the
/// input is well-formed, each entry is rescaled to make the sum exactly
/// 1.0.
pub fn normalize_sizes(sizes: &mut Vec<f32>, len: usize) {
    if len == 0 {
        sizes.clear();
        return;
    }
    let equal = || vec![1.0 / len as f32; len];
    if sizes.len() != len || sizes.iter().any(|f| !f.is_finite() || *f < 0.0) {
        *sizes = equal();
        return;
    }
    let sum: f32 = sizes.iter().sum();
    if sum <= 0.0 || (sum - 1.0).abs() > SIZE_SUM_EPSILON {
        *sizes = equal();
        return;
    }
    for f in sizes.iter_mut() {
        *f /= sum;
    }
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
    fn normalize_sizes_rejects_mismatched_length_or_bad_values() {
        let mut v = vec![0.25];
        normalize_sizes(&mut v, 2);
        assert_eq!(v, vec![0.5, 0.5]);

        let mut v = vec![f32::NAN, 0.5];
        normalize_sizes(&mut v, 2);
        assert_eq!(v, vec![0.5, 0.5]);

        let mut v = vec![-0.2, 1.2];
        normalize_sizes(&mut v, 2);
        assert_eq!(v, vec![0.5, 0.5]);

        // Input already near unity — left as-is (rescaled to exact 1.0).
        let mut v = vec![0.25, 0.75];
        normalize_sizes(&mut v, 2);
        assert_eq!(v, vec![0.25, 0.75]);
    }

    #[test]
    fn normalize_sizes_rescales_to_exact_sum_of_one() {
        let mut v = vec![0.2, 0.3, 0.5005]; // slightly off unity, inside epsilon
        normalize_sizes(&mut v, 3);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    #[test]
    fn normalize_sizes_with_zero_len_clears() {
        let mut v = vec![0.5, 0.5];
        normalize_sizes(&mut v, 0);
        assert!(v.is_empty());
    }

    #[test]
    fn get_returns_none_for_unknown_workspace() {
        let dir = temp_dir("unknown");
        let store = LayoutStore::load(dir.join("workspaces.json")).unwrap();
        assert!(store.get(&dir).is_none());
    }

    #[test]
    fn put_and_get_round_trip_via_memory() {
        let dir = temp_dir("roundtrip-mem");
        let mut store = LayoutStore::load(dir.join("workspaces.json")).unwrap();
        let id = SessionId::new_v4();
        store.put(&dir, Layout::new(vec![id], vec![1.0])).unwrap();
        let got = store.get(&dir).unwrap().clone();
        assert_eq!(got.order, vec![id]);
        assert_eq!(got.sizes, vec![1.0]);
    }

    #[test]
    fn put_persists_to_disk_and_reloads() {
        let dir = temp_dir("persist");
        let path = dir.join("workspaces.json");
        let workspace = dir.join("workspace");
        fs::create_dir_all(&workspace).unwrap();

        let id = SessionId::new_v4();
        {
            let mut store = LayoutStore::load(path.clone()).unwrap();
            store
                .put(&workspace, Layout::new(vec![id], vec![1.0]))
                .unwrap();
        }

        let reloaded = LayoutStore::load(path).unwrap();
        let got = reloaded.get(&workspace).unwrap();
        assert_eq!(got.order, vec![id]);
        assert_eq!(got.sizes, vec![1.0]);
    }

    #[test]
    fn put_normalizes_bad_sizes() {
        let dir = temp_dir("normalize-on-put");
        let mut store = LayoutStore::load(dir.join("workspaces.json")).unwrap();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        store
            .put(&dir, Layout::new(vec![id1, id2], vec![-1.0, 2.0]))
            .unwrap();
        let got = store.get(&dir).unwrap();
        assert_eq!(got.sizes, vec![0.5, 0.5]);
    }

    #[test]
    fn load_missing_file_returns_empty_store() {
        let dir = temp_dir("missing");
        let store = LayoutStore::load(dir.join("workspaces.json")).unwrap();
        assert!(store.get(&dir).is_none());
    }

    #[test]
    fn load_tolerates_extra_fields_from_older_schemas() {
        // Files written by the deleted GTK app had a `focused` field. The new
        // schema ignores it — unknown fields must not fail deserialization.
        let dir = temp_dir("legacy-schema");
        let path = dir.join("workspaces.json");
        let workspace = dir.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let key = workspace_key(&workspace);
        let id = SessionId::new_v4();
        let legacy = format!(
            r#"{{
                "workspaces": {{
                    "{key}": {{
                        "order": ["{id}"],
                        "sizes": [1.0],
                        "focused": 0,
                        "split_fracs": [1.0]
                    }}
                }}
            }}"#
        );
        fs::write(&path, legacy).unwrap();

        let store = LayoutStore::load(path).unwrap();
        let got = store.get(&workspace).unwrap();
        assert_eq!(got.order, vec![id]);
        assert_eq!(got.sizes, vec![1.0]);
    }

    #[test]
    fn load_corrupt_file_returns_err() {
        let dir = temp_dir("corrupt");
        let path = dir.join("workspaces.json");
        fs::write(&path, "not json at all").unwrap();
        let err = LayoutStore::load(path).unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("parsing workspace layout"), "{s}");
    }

    #[test]
    fn put_overwrites_existing_entry() {
        let dir = temp_dir("overwrite");
        let mut store = LayoutStore::load(dir.join("workspaces.json")).unwrap();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        store.put(&dir, Layout::new(vec![id1], vec![1.0])).unwrap();
        store.put(&dir, Layout::new(vec![id2], vec![1.0])).unwrap();
        let got = store.get(&dir).unwrap();
        assert_eq!(got.order, vec![id2]);
    }
}
