//! Native menu construction and dispatch.
//!
//! A plain OS-level menu (not a webview overlay) gives users the File and
//! Help menus they expect from a desktop app. Menu events are raised on the
//! Rust side; the handler translates each click into either a shared
//! `do_*` action from `commands.rs` or an event emission to the webview.
//! No business logic lives here — this module is pure dispatch.

use anyhow::Result;
use tauri::menu::{Menu, MenuBuilder, MenuEvent, MenuItemBuilder, SubmenuBuilder};
use tauri::{AppHandle, Manager, Wry};
use tauri_plugin_dialog::DialogExt;

use crate::commands::{do_open_workspace, do_request_quit};
use crate::events::emit_show_about;
use crate::state::AppState;

pub fn build(app: &AppHandle) -> Result<Menu<Wry>> {
    let open_workspace = MenuItemBuilder::with_id("open_workspace", "Open Workspace…")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit")
        .accelerator("CmdOrCtrl+Q")
        .build(app)?;
    let file = SubmenuBuilder::new(app, "File")
        .item(&open_workspace)
        .separator()
        .item(&quit)
        .build()?;

    let about = MenuItemBuilder::with_id("about", "About Ox").build(app)?;
    let help = SubmenuBuilder::new(app, "Help").item(&about).build()?;

    let menu = MenuBuilder::new(app).item(&file).item(&help).build()?;
    Ok(menu)
}

pub fn handle_menu_event(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        "open_workspace" => open_workspace_picker(app.clone()),
        "quit" => {
            let state = app.state::<AppState>().inner().clone();
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                do_request_quit(app, state).await;
            });
        }
        "about" => emit_show_about(app),
        _ => {}
    }
}

/// Launch the native directory picker. `pick_folder` is async (it returns
/// immediately; the callback fires once the user selects or cancels), so
/// the work happens inside the callback — not on the menu-event thread.
fn open_workspace_picker(app: AppHandle) {
    app.clone().dialog().file().pick_folder(move |result| {
        let Some(file_path) = result else {
            return;
        };
        let Some(path) = file_path.as_path().map(|p| p.to_path_buf()) else {
            // `FilePath::Url` happens on mobile; the desktop picker always
            // returns a real path. Guard anyway so a future mobile build
            // doesn't silently misbehave.
            return;
        };
        let state = app.state::<AppState>().inner().clone();
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(e) = do_open_workspace(app, state, path).await {
                eprintln!("open workspace failed: {e:#}");
            }
        });
    });
}
