//! Per-split widget assembly.
//!
//! Builds the vertical `gtk::Box` that a `gtk::Paned` child slot holds:
//! a small header strip at the top (workspace root + session id), the
//! transcript `gtk::ScrolledWindow` in the middle, the status row and
//! input strip at the bottom. Labels for workspace root and session id
//! bind directly to the `SplitObject` via `bind_property` so property
//! changes flow into the UI with no manual notify wiring.
//!
//! A `GestureClick` on the root box forwards clicks to
//! `OxWindow::set_focused_split` so "click to focus" works without a
//! dedicated focus-chain rewrite.

use gtk::glib;
use gtk::prelude::*;

use crate::input;
use crate::objects::SplitObject;
use crate::transcript;
use crate::window::OxWindow;

pub fn build_split_view(split: &SplitObject, window: &OxWindow) -> gtk::Box {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.add_css_class("split");
    root.set_hexpand(true);
    root.set_vexpand(true);
    root.set_size_request(260, -1);

    root.append(&build_header_strip(split));
    let transcript_widget = transcript::build_transcript(split);
    transcript_widget.set_hexpand(true);
    transcript_widget.set_vexpand(true);
    root.append(&transcript_widget);
    root.append(&build_status_row(split));
    root.append(&input::build_input_strip(split, window));

    // Focused styling: bind the `split-focused` CSS class to the split's
    // `focused` property. The closure bridge avoids a separate handler by
    // wiring directly into GObject property transforms.
    split
        .bind_property("focused", &root, "css-classes")
        .transform_to(|_, focused: bool| {
            let mut classes = vec!["split".to_string()];
            if focused {
                classes.push("split-focused".to_string());
            }
            Some(
                classes
                    .into_iter()
                    .map(glib::GString::from)
                    .collect::<Vec<_>>(),
            )
        })
        .sync_create()
        .build();

    // Focus-on-click. Any click inside the split's root box (including
    // clicks that land in the input TextView) promotes the split to the
    // focused slot. The GestureClick runs in the Capture phase so it
    // observes the click before descendants consume it; we do NOT claim
    // the gesture, so the click still bubbles down to the TextView for
    // cursor placement / native selection. Bubble phase would have missed
    // any click on the TextView entirely because the TextView eats
    // button-press events for its own input handling.
    //
    // Weak references break the cycle root → controller → closure → root
    // (the SplitObject caches `root` and we capture the SplitObject here).
    let click = gtk::GestureClick::new();
    click.set_propagation_phase(gtk::PropagationPhase::Capture);
    let window_weak = window.downgrade();
    let split_weak = split.downgrade();
    click.connect_pressed(move |_, _, _, _| {
        let (Some(window), Some(split)) = (window_weak.upgrade(), split_weak.upgrade()) else {
            return;
        };
        window.set_focused_split(split.split_id_uuid());
    });
    root.add_controller(click);

    root
}

/// Top header strip: workspace root on the left, session id on the
/// right. Both bound via `bind_property` so they update the moment the
/// split's property changes (e.g. `Ready` lands a session id).
fn build_header_strip(split: &SplitObject) -> gtk::Box {
    let strip = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    strip.add_css_class("split-header");
    strip.set_margin_start(8);
    strip.set_margin_end(8);
    strip.set_margin_top(4);
    strip.set_margin_bottom(4);

    let workspace_label = gtk::Label::new(None);
    workspace_label.set_xalign(0.0);
    workspace_label.set_hexpand(true);
    workspace_label.add_css_class("split-workspace-label");
    workspace_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    split
        .bind_property("workspace-root", &workspace_label, "label")
        .sync_create()
        .build();

    let session_label = gtk::Label::new(None);
    session_label.set_xalign(1.0);
    session_label.add_css_class("split-session-label");
    session_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    split
        .bind_property("session-id", &session_label, "label")
        .sync_create()
        .build();

    strip.append(&workspace_label);
    strip.append(&session_label);
    strip
}

/// Bottom status row: waiting / error / cancelled labels. Each label's
/// `visible` property is bound to its backing flag so only one label
/// shows at a time (or none).
fn build_status_row(split: &SplitObject) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    row.add_css_class("split-status");
    row.set_margin_start(16);
    row.set_margin_end(16);
    row.set_margin_top(2);
    row.set_margin_bottom(2);

    let waiting = gtk::Label::new(Some("Thinking…"));
    waiting.add_css_class("status-waiting");
    waiting.set_xalign(0.0);
    split
        .bind_property("waiting", &waiting, "visible")
        .sync_create()
        .build();
    row.append(&waiting);

    let error = gtk::Label::new(None);
    error.add_css_class("status-error");
    error.set_xalign(0.0);
    error.set_wrap(true);
    error.set_wrap_mode(gtk::pango::WrapMode::WordChar);
    split
        .bind_property("error-text", &error, "label")
        .sync_create()
        .build();
    // Show the error row only when there is text to show.
    split
        .bind_property("error-text", &error, "visible")
        .transform_to(|_, text: String| Some(!text.is_empty()))
        .sync_create()
        .build();
    row.append(&error);

    let cancelled = gtk::Label::new(Some("Cancelled"));
    cancelled.add_css_class("status-cancelled");
    cancelled.set_xalign(0.0);
    split
        .bind_property("cancelled", &cancelled, "visible")
        .sync_create()
        .build();
    row.append(&cancelled);

    row
}
