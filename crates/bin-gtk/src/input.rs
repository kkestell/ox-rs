//! Per-split input strip: text view + hint label.
//!
//! The input strip is the only place in the app that owns a
//! `gtk::TextBuffer` for a draft; we stash it on the `SplitObject` so
//! the `submit-focused` action handler can read (and clear) the draft
//! text without having to walk widget trees.
//!
//! Key handling: Enter submits via `window.activate_action("submit-focused")`.
//! Shift+Enter inserts a newline (falls through). Escape cancels via
//! `window.activate_action("cancel-focused")`. Every other key is
//! propagated untouched.

use gtk::gdk;
use gtk::glib;
use gtk::prelude::*;

use crate::objects::SplitObject;
use crate::window::OxWindow;

pub fn build_input_strip(split: &SplitObject, window: &OxWindow) -> gtk::Box {
    let strip = gtk::Box::new(gtk::Orientation::Vertical, 0);
    strip.add_css_class("split-input");
    strip.set_margin_start(16);
    strip.set_margin_end(16);
    strip.set_margin_top(4);
    strip.set_margin_bottom(8);

    let text_view = gtk::TextView::new();
    text_view.set_wrap_mode(gtk::WrapMode::WordChar);
    text_view.set_accepts_tab(false);
    text_view.set_hexpand(true);
    text_view.set_top_margin(4);
    text_view.set_bottom_margin(4);
    text_view.add_css_class("draft-input");

    // Stash the buffer on the SplitObject so `submit_focused` can reach it.
    split.set_draft_buffer(text_view.buffer());

    // Promote this split to focused whenever its TextView actually receives
    // GTK keyboard focus. The split-root GestureClick (in `split_view.rs`)
    // catches *clicks*, but it does not fire for tab navigation or
    // programmatic focus changes — and after the paned tree is rebuilt,
    // GtkPaned's internal button handling can also swallow the click before
    // the Capture-phase gesture sees it. Hooking ::enter on the TextView
    // catches every path that lands the cursor in this split's input, which
    // is the exact moment the focused-tracker needs to follow.
    let focus = gtk::EventControllerFocus::new();
    let focus_window_weak = window.downgrade();
    let focus_split_weak = split.downgrade();
    focus.connect_enter(move |_| {
        let (Some(window), Some(split)) =
            (focus_window_weak.upgrade(), focus_split_weak.upgrade())
        else {
            return;
        };
        window.set_focused_split(split.split_id_uuid());
    });
    text_view.add_controller(focus);

    // Weak ref to break root → text_view → controller → closure → window
    // → splits_store → split → root cycle now that splits cache their root.
    let key = gtk::EventControllerKey::new();
    let window_weak = window.downgrade();
    key.connect_key_pressed(move |_, keyval, _keycode, state| {
        let Some(window) = window_weak.upgrade() else {
            return glib::Propagation::Proceed;
        };
        let shift = state.contains(gdk::ModifierType::SHIFT_MASK);
        match keyval {
            gdk::Key::Return | gdk::Key::KP_Enter => {
                if shift {
                    glib::Propagation::Proceed
                } else {
                    gtk::prelude::WidgetExt::activate_action(
                        &window,
                        "app.submit-focused",
                        None,
                    )
                    .ok();
                    glib::Propagation::Stop
                }
            }
            gdk::Key::Escape => {
                gtk::prelude::WidgetExt::activate_action(&window, "app.cancel-focused", None)
                    .ok();
                glib::Propagation::Stop
            }
            _ => glib::Propagation::Proceed,
        }
    });
    text_view.add_controller(key);

    strip.append(&text_view);

    strip
}
