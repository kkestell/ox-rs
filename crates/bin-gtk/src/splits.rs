//! Splits: `gtk::Paned` tree builder for the horizontal split container.
//!
//! Ox lays out splits horizontally: N splits → a left-leaning `gtk::Paned`
//! tree where the N-th pane occupies the rightmost slot. Positions are
//! driven by the per-split fractions stored on `OxWindow`, multiplied by
//! the container's width at layout time.
//!
//! Rebuilding is always from-scratch: `rebuild_paned_tree` clears the
//! `split_container` `gtk::Box` and re-assembles it. Children are
//! `split_view::build_split_view` outputs — the per-split top-to-bottom
//! vertical stack with header, transcript, status, and input.
//!
//! Separator drags call `OxWindow::set_split_fractions` via a pure
//! `fractions_after_drag` helper so the drag math is testable headless.

use gtk::glib;
use gtk::prelude::*;

use crate::objects::SplitObject;
use crate::split_view;
use crate::window::OxWindow;

/// Minimum pixel width for a split. Below this, a drag is clamped so the
/// user can't squash a split into oblivion.
const MIN_SPLIT_PX: i32 = 260;

/// Return the split's cached root widget, building it on first call. The
/// cache lives on `SplitObject`; this function is the only place that
/// populates it. Subsequent calls (across paned-tree rebuilds) hand back
/// the same widget so reparenting is cheap and the WebView/input state
/// is preserved.
fn get_or_build_view(split: &SplitObject, window: &OxWindow) -> gtk::Box {
    if let Some(view) = split.cached_view() {
        return view;
    }
    let view = split_view::build_split_view(split, window);
    split.set_cached_view(view.clone());
    view
}

/// Replace the `split_container`'s children with a fresh paned tree. Safe
/// to call whenever the split count or fractions change.
///
/// Existing splits' root widgets are reused — the only widget tree that
/// gets built here is for splits that don't yet have a cached view (i.e.
/// the brand-new split when called from `add_split`). Without this reuse,
/// every `/new` would tear down every WebView, force a full HTML reload,
/// and replay the entire transcript per surviving split — visible to the
/// user as "the entire UI redraws."
pub fn rebuild_paned_tree(window: &OxWindow) {
    let container = window.split_container();

    let store = window.splits_store();
    let n = store.n_items();

    // Collect (split, view) pairs first. Building the view here (before
    // we tear down the old container) means a SplitObject's cached view
    // gets populated before any reparenting happens.
    let mut splits: Vec<SplitObject> = Vec::with_capacity(n as usize);
    let mut views: Vec<gtk::Box> = Vec::with_capacity(n as usize);
    for i in 0..n {
        let split: SplitObject = store
            .item(i)
            .expect("index within n_items")
            .downcast()
            .expect("SplitObject");
        let view = get_or_build_view(&split, window);
        splits.push(split);
        views.push(view);
    }

    // Detach every cached view from its current parent (likely a Paned
    // from the previous tree). GTK4 forbids attaching a widget that
    // already has a parent, so this must happen before we tear down the
    // container or build new panes around the same widgets.
    for view in &views {
        if view.parent().is_some() {
            view.unparent();
        }
    }

    // Now safe to drop the old paned tree. The Panes have no further
    // references and will be finalized.
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }

    if views.is_empty() {
        return;
    }

    if views.len() == 1 {
        views[0].set_hexpand(true);
        views[0].set_vexpand(true);
        container.append(&views[0]);
        return;
    }

    // Build a nested `gtk::Paned` tree: (((split0 | split1) | split2) | ...)
    // so the N-th split occupies the rightmost slot. `set_position` is
    // applied on an `map` signal so the container has a real width.
    let mut panes: Vec<gtk::Paned> = Vec::new();
    let first_pane = gtk::Paned::new(gtk::Orientation::Horizontal);
    first_pane.set_start_child(Some(&views[0]));
    first_pane.set_end_child(Some(&views[1]));
    first_pane.set_resize_start_child(true);
    first_pane.set_resize_end_child(true);
    first_pane.set_shrink_start_child(false);
    first_pane.set_shrink_end_child(false);
    panes.push(first_pane.clone());

    let mut current: gtk::Widget = first_pane.upcast();
    for view in views.iter().skip(2) {
        let next_pane = gtk::Paned::new(gtk::Orientation::Horizontal);
        next_pane.set_start_child(Some(&current));
        next_pane.set_end_child(Some(view));
        next_pane.set_resize_start_child(true);
        next_pane.set_resize_end_child(true);
        next_pane.set_shrink_start_child(false);
        next_pane.set_shrink_end_child(false);
        panes.push(next_pane.clone());
        current = next_pane.upcast();
    }

    container.append(&current);

    // Once the container is mapped and has a real width, translate each
    // fraction into an absolute pixel position. The inner-most `gtk::Paned`
    // gets the fraction for split 0; each outer one gets the running sum.
    let window_for_map = window.clone();
    let container_for_map = container.clone();
    container.connect_map(move |_| {
        install_paned_positions(&window_for_map, &container_for_map);
    });

    // Wire each paned's position-drag-end to `fractions_after_drag` → window.
    for (i, pane) in panes.iter().enumerate() {
        let window_for_drag = window.clone();
        let pane_for_drag = pane.clone();
        pane.connect_position_notify(move |_| {
            // A position_notify fires on every pixel of a drag. We don't
            // want to stash a full fractions update every pixel, so the
            // drag-end signal would be preferred — but gtk4-rs doesn't
            // expose that on `Paned`. Debounce via a one-shot idle so the
            // final position wins without flooding.
            let idx = i;
            let pane = pane_for_drag.clone();
            let window = window_for_drag.clone();
            glib::idle_add_local_once(move || {
                let alloc_width = window.split_container().width().max(1) as f32;
                let pane_position = pane.position() as f32 / alloc_width;
                let old_fracs = window.split_fracs();
                let new_fracs = fractions_after_drag(&old_fracs, idx, pane_position);
                window.set_split_fractions(new_fracs);
            });
        });
    }
}

/// Walk the paned tree and set the horizontal position of each `Paned`
/// based on the current fractions. Called on map (to land the initial
/// position) and indirectly via `rebuild_paned_tree` for reshapes.
fn install_paned_positions(window: &OxWindow, container: &gtk::Box) {
    let fracs = window.split_fracs();
    if fracs.len() < 2 {
        return;
    }
    let alloc_width = container.width().max(1) as f32;
    // Walk outward: the outer-most pane has its position at
    // `sum(fracs[0..n-1]) * width`; the next inner has
    // `sum(fracs[0..n-2]) * width`; and so on.
    let mut pane_opt: Option<gtk::Paned> = container
        .first_child()
        .and_then(|w| w.downcast::<gtk::Paned>().ok());
    let mut running = fracs.iter().sum::<f32>();
    for frac in fracs.iter().rev().skip(1) {
        running -= frac;
        let Some(pane) = pane_opt else {
            break;
        };
        let pos = (running * alloc_width).max(MIN_SPLIT_PX as f32) as i32;
        pane.set_position(pos);
        pane_opt = pane
            .start_child()
            .and_then(|w| w.downcast::<gtk::Paned>().ok());
    }
}

/// Pure helper: given the prior fractions and a drag event at pane index
/// `idx` with new fractional position `new_pos` (relative to the
/// container's total width), return the new fraction vector. The drag
/// changes `fracs[idx]` and `fracs[idx+1]` only; others are untouched.
///
/// The paned tree is left-leaning, so pane index `i` boundary sits at
/// `sum(fracs[0..=i])`. A drag to position `p` on pane `i` therefore
/// redistributes `fracs[i]` = `p - sum(fracs[0..i])` and
/// `fracs[i+1]` += `old_fracs[i] - new_fracs[i]`.
pub fn fractions_after_drag(old_fracs: &[f32], idx: usize, new_pos: f32) -> Vec<f32> {
    if idx + 1 >= old_fracs.len() {
        return old_fracs.to_vec();
    }
    let mut fracs = old_fracs.to_vec();
    let before: f32 = fracs[..idx].iter().sum();
    let combined = fracs[idx] + fracs[idx + 1];
    let min_frac = 0.05_f32; // soft floor; rebuild clamps to MIN_SPLIT_PX at layout time
    let new_left = (new_pos - before).clamp(min_frac, combined - min_frac);
    fracs[idx] = new_left;
    fracs[idx + 1] = combined - new_left;
    fracs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_redistributes_between_adjacent_fractions() {
        let fracs = vec![0.3, 0.3, 0.4];
        let out = fractions_after_drag(&fracs, 0, 0.5);
        assert!((out[0] - 0.5).abs() < 1e-4);
        assert!((out[1] - 0.1).abs() < 1e-4);
        assert_eq!(out[2], 0.4);
    }

    #[test]
    fn drag_clamps_to_minimum_fraction() {
        let fracs = vec![0.5, 0.5];
        let out = fractions_after_drag(&fracs, 0, 0.99);
        // Should clamp to combined - 0.05 = 0.95.
        assert!((out[0] - 0.95).abs() < 1e-4);
        assert!((out[1] - 0.05).abs() < 1e-4);
    }

    #[test]
    fn drag_on_rightmost_pane_is_noop() {
        let fracs = vec![0.25, 0.25, 0.5];
        let out = fractions_after_drag(&fracs, 2, 0.9);
        assert_eq!(out, fracs);
    }
}
