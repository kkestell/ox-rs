//! Free functions that back `gio::SimpleAction` activate handlers.
//!
//! `app.rs` registers a `SimpleAction` per entry here; each handler
//! upgrades `Application::active_window()` to an `OxWindow` and calls
//! the matching function below. Keeping the logic out of the closures
//! themselves makes the behaviour independently readable and avoids
//! deeply nested captures.

use adw::prelude::*;
use agent_host::{SplitAction, classify_input};
use gtk::gio;
use gtk::glib;

use crate::modals;
use crate::window::OxWindow;

/// Entry point for Enter on the focused split's input. Reads the draft,
/// runs it through `classify_input`, and dispatches to whichever branch
/// the classification picks.
pub fn submit_focused(window: &OxWindow) {
    let Some(split) = window.focused_split() else {
        return;
    };
    let Some(buffer) = split.draft_buffer() else {
        return;
    };
    let (start, end) = buffer.bounds();
    let text = buffer.text(&start, &end, false).to_string();

    match classify_input(&text) {
        SplitAction::Send => {
            if text.trim().is_empty() {
                return;
            }
            buffer.set_text("");
            split.submit_draft(text);
        }
        SplitAction::New => {
            buffer.set_text("");
            if let Err(e) = window.add_split() {
                split.set_error_text(format!("{e:#}"));
            }
        }
        SplitAction::CloseSplit => {
            buffer.set_text("");
            close_focused_split(window);
        }
        SplitAction::QuitApp => {
            buffer.set_text("");
            window.request_quit();
        }
        SplitAction::Cancel => {
            // Classify never returns Cancel today (that's an Escape
            // handler concern), but keep the arm so the match stays
            // exhaustive if new slash commands land.
            split.cancel();
        }
    }
}

/// Escape on the focused split's input — tell the agent to abandon the
/// in-flight turn. Harmless if nothing is in flight.
pub fn cancel_focused(window: &OxWindow) {
    if let Some(split) = window.focused_split() {
        split.cancel();
    }
}

/// Menu: New Split. Route through `OxWindow::add_split`, surface any
/// spawn failure on the previously-focused split.
pub fn new_split(window: &OxWindow) {
    let prev_focused = window.focused_split();
    if let Err(e) = window.add_split()
        && let Some(split) = prev_focused
    {
        split.set_error_text(format!("{e:#}"));
    }
}

/// Menu: Close Split. If this is the last split, route into the
/// quit-confirmation flow so the app doesn't end up with an empty window.
pub fn close_split(window: &OxWindow) {
    close_focused_split(window);
}

fn close_focused_split(window: &OxWindow) {
    let Some(id) = window.focused() else {
        return;
    };
    let outcome = window.close_split(id);
    if outcome.last_split_closed {
        window.request_quit();
    }
}

/// Menu: Open Workspace. Presents a folder-picker and, on confirm,
/// calls `replace_workspace`. The picker future is spawned on the
/// GLib main loop so the action handler returns immediately.
pub fn open_workspace(window: &OxWindow) {
    let dialog = gtk::FileDialog::builder()
        .title("Open Workspace")
        .modal(true)
        .build();
    let window_for_future = window.clone();
    let parent = window.clone();
    glib::spawn_future_local(async move {
        match dialog.select_folder_future(Some(&parent)).await {
            Ok(folder) => {
                if let Some(path) = folder.path()
                    && let Err(e) = window_for_future.replace_workspace(path)
                    && let Some(split) = window_for_future.focused_split()
                {
                    split.set_error_text(format!("{e:#}"));
                }
            }
            Err(_) => {
                // User dismissed the dialog.
            }
        }
    });
}

/// Menu: Quit. Routes through `request_quit` which handles the
/// confirmation dialog when a turn is in flight.
pub fn quit(window: &OxWindow) {
    window.request_quit();
}

/// Menu: About. Presents the `adw::AboutDialog`.
pub fn about(window: &OxWindow) {
    modals::present_about(window);
}

/// Register every app-level action on the given `gio::ActionMap` (the
/// `adw::Application`). `dispatch` upgrades the active window to an
/// `OxWindow` and calls the supplied free function. Actions with no
/// window are silently dropped — that state should be impossible under
/// normal flow, but guarding avoids a panic on e.g. a keyboard shortcut
/// fired during window teardown.
pub fn register_actions(app: &adw::Application) {
    type Handler = fn(&OxWindow);
    let pairs: &[(&str, Handler)] = &[
        ("new-split", new_split),
        ("close-split", close_split),
        ("open-workspace", open_workspace),
        ("quit", quit),
        ("about", about),
        ("cancel-focused", cancel_focused),
        ("submit-focused", submit_focused),
    ];
    for (name, handler) in pairs {
        let action = gio::SimpleAction::new(name, None);
        let handler = *handler;
        let app_weak = app.downgrade();
        action.connect_activate(move |_, _| {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let Some(window) = app.active_window() else {
                return;
            };
            let Ok(ox) = window.downcast::<OxWindow>() else {
                return;
            };
            handler(&ox);
        });
        app.add_action(&action);
    }
}
