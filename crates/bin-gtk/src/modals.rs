//! Adwaita modal dialogs: quit confirmation and about.
//!
//! Every dialog here is presented on an `OxWindow` and dismisses by
//! itself — the caller only needs to wire the confirm callback.

use adw::prelude::*;

use crate::window::OxWindow;

/// Quit-confirmation alert. Invoked when the user attempts to close the
/// window while a turn is in flight. `on_confirm` runs only if the user
/// chooses "Quit"; dismissing or canceling leaves the window open.
pub fn present_quit_confirm<F>(window: &OxWindow, on_confirm: F)
where
    F: Fn() + 'static,
{
    let dialog = adw::AlertDialog::new(
        Some("Quit Ox?"),
        Some("A turn is in progress. Closing now will abandon it."),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("quit", "Quit");
    dialog.set_response_appearance("quit", adw::ResponseAppearance::Destructive);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");

    dialog.connect_response(None, move |_, response| {
        if response == "quit" {
            on_confirm();
        }
    });

    dialog.present(Some(window));
}

/// "About Ox" dialog. Pulls the version string out of the package's
/// Cargo.toml at compile time so the two never drift.
pub fn present_about(window: &OxWindow) {
    let dialog = adw::AboutDialog::new();
    dialog.set_application_name("Ox");
    dialog.set_version(env!("CARGO_PKG_VERSION"));
    dialog.set_developer_name("Kyle Kestell");
    dialog.set_comments("A desktop AI coding assistant.");
    dialog.present(Some(window));
}
