//! Workspace layout model and repository port.
//!
//! Each [`Layout`] records the session IDs that made up a workspace's panes
//! and the horizontal fraction each pane occupied, so a later launch can
//! rebuild the tiled UI. File-backed persistence is an outward adapter; this
//! crate keeps only the pure data shape, normalization rules, and port.
//!
//! No `focused` field: focus is pure client-side UI state (the browser
//! doesn't need the server to remember which tab it last clicked on).
//! Files written by the deleted GTK app carrying `focused` round-trip
//! cleanly — unknown fields are ignored by serde.

use std::path::Path;

use anyhow::Result;
use domain::SessionId;
use serde::{Deserialize, Serialize};

/// Tolerance used when deciding whether persisted sizes already sum to
/// 1.0. Values outside this band are rejected and replaced with equal
/// sizes so a corrupted entry can't produce pathological widths.
const SIZE_SUM_EPSILON: f32 = 0.01;

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

/// Persistence port for workspace layouts. Implementations return owned
/// values so callers are not coupled to cache lifetimes or lock guards.
pub trait LayoutRepository: Send + Sync + 'static {
    fn get(&self, workspace_root: &Path) -> Result<Option<Layout>>;
    fn put(&self, workspace_root: &Path, layout: Layout) -> Result<()>;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
