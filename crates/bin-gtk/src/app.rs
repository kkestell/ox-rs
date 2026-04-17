//! Application composition root.
//!
//! `build_application` wires up the `adw::Application`: CSS load, dark
//! color scheme, app-level actions (both menu-targeted and input-route
//! targets), keyboard accelerators, and the one-shot `activate` that
//! constructs the `OxWindow` from the seed state (runtime, spawn
//! config, layouts). Seed state is captured by value into the
//! `connect_activate` closure via an `Rc<OnceCell>` dance — `activate`
//! fires once per invocation, and on cold starts we cannot move the
//! state into the closure body without `'static`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;
use agent_host::{AgentSpawnConfig, WorkspaceLayouts};
use gtk::gdk;
use gtk::gio;
use tokio::runtime::Runtime;

use crate::actions;
use crate::window::OxWindow;

const STYLE_SHEET: &str = include_str!("../assets/style.css");

/// Seed state captured by the application closures. Wrapped in a
/// `RefCell<Option<_>>` so the first `activate` call can `take()` the
/// seed and hand it to the window; later activates (if any) reuse the
/// existing window.
struct Seed {
    runtime: Rc<Runtime>,
    spawn_config: AgentSpawnConfig,
    layouts: WorkspaceLayouts,
    layout_path: PathBuf,
}

pub fn build_application(
    runtime: Rc<Runtime>,
    spawn_config: AgentSpawnConfig,
    layouts: WorkspaceLayouts,
    layout_path: PathBuf,
) -> adw::Application {
    let app = adw::Application::builder()
        .application_id("dev.kestell.ox")
        .flags(gio::ApplicationFlags::NON_UNIQUE)
        .build();

    // `startup` runs once per process. CSS and the color scheme belong
    // here because a display is available but no window yet — setting
    // them on `activate` risks a flicker on windowed reshow.
    app.connect_startup(|_| {
        install_css();
        adw::StyleManager::default().set_color_scheme(adw::ColorScheme::ForceDark);
    });

    actions::register_actions(&app);

    // Keyboard accelerators for the menu actions. `activate_action`
    // routes from any widget in the tree into these handlers.
    app.set_accels_for_action("app.new-split", &["<Primary>n"]);
    app.set_accels_for_action("app.close-split", &["<Primary>w"]);
    app.set_accels_for_action("app.open-workspace", &["<Primary>o"]);
    app.set_accels_for_action("app.quit", &["<Primary>q"]);

    let seed = Rc::new(RefCell::new(Some(Seed {
        runtime,
        spawn_config,
        layouts,
        layout_path,
    })));

    app.connect_activate(move |app| {
        // If a window already exists, present it instead of building a
        // second. Matches standard `adw::Application` reactivation.
        if let Some(window) = app.active_window() {
            window.present();
            return;
        }
        let Some(seed) = seed.borrow_mut().take() else {
            return;
        };
        let window = OxWindow::new(
            app,
            seed.runtime,
            seed.spawn_config,
            seed.layouts,
            seed.layout_path,
        );
        if let Err(e) = window.restore_workspace() {
            eprintln!("failed to restore workspace: {e:#}");
        }
        window.present();
    });

    app
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(STYLE_SHEET);
    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}
