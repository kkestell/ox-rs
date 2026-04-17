//! Tauri composition root.
//!
//! [`run`] wires the `WorkspaceState` into the Tauri `Builder`, registers
//! every `#[tauri::command]`, attaches the native menu, installs a window
//! close-requested handler, starts drain tasks for every split that came
//! back from `WorkspaceState::restore`, and hands control to the Tauri
//! runtime. No application logic lives here — this is plumbing.
//!
//! Runtime discipline: a tokio runtime is created up front and installed as
//! the tauri async runtime. `WorkspaceState::restore` is called *inside* the
//! runtime's enter-guard so `tokio::process::Command` spawns succeed. From
//! there on, every drain task and every Tauri command runs on the same
//! runtime.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

use agent_host::{AgentSpawnConfig, WorkspaceState};
use anyhow::{Context, Result};
use tauri::{Manager, WindowEvent};

use crate::events::emit_confirm_quit_requested;
use crate::state::AppState;

mod commands;
mod drain;
mod events;
mod menu;
mod state;

pub fn run(spawn_config: AgentSpawnConfig, layout_state_path: PathBuf) -> Result<()> {
    ensure_graphical_session()?;

    // Create tokio *before* restoring: the agent-host spawn path uses
    // `tokio::process::Command`, which needs an active runtime. Telling
    // `tauri::async_runtime` to use this same runtime keeps drain tasks and
    // Tauri commands on one scheduler.
    let rt = tokio::runtime::Runtime::new()?;
    tauri::async_runtime::set(rt.handle().clone());
    let _guard = rt.enter();

    let (workspace, spawned) = WorkspaceState::restore(spawn_config, layout_state_path)?;
    let app_state = AppState::new(workspace);

    // `setup` is `FnOnce(&mut App)` and the spawned-stream vector must move
    // into it — the streams aren't `Clone`. Wrapping in `Mutex<Option<_>>`
    // lets the closure `take()` the value the first (and only) time it runs.
    // A sync `std::sync::Mutex` is fine here because the lock is only ever
    // contended on a single call from the main thread.
    let initial_streams = Mutex::new(Some(spawned));
    let state_for_setup = app_state.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::submit,
            commands::cancel_split,
            commands::close_split,
            commands::new_split,
            commands::open_workspace,
            commands::set_split_fractions,
            commands::request_quit,
            commands::confirm_quit,
            commands::cancel_quit,
            commands::get_app_info,
        ])
        .on_menu_event(menu::handle_menu_event)
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                on_close_requested(window, api);
            }
        })
        .setup(move |app| {
            let handle = app.handle().clone();
            let menu = menu::build(&handle)?;
            app.set_menu(menu)?;

            if let Some(streams) = initial_streams.lock().unwrap().take() {
                for (split_id, stream) in streams {
                    drain::spawn_drain(handle.clone(), state_for_setup.clone(), split_id, stream);
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .context("tauri runtime exited with error")?;

    Ok(())
}

#[cfg(target_os = "linux")]
fn ensure_graphical_session() -> Result<()> {
    if has_graphical_session_vars(std::env::vars()) {
        return Ok(());
    }

    anyhow::bail!(
        "cannot start ox-tauri without a graphical session; \
         set DISPLAY or WAYLAND_DISPLAY, or run from a desktop terminal"
    );
}

#[cfg(not(target_os = "linux"))]
fn ensure_graphical_session() -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn has_graphical_session_vars(vars: impl IntoIterator<Item = (String, String)>) -> bool {
    vars.into_iter().any(|(key, value)| {
        matches!(key.as_str(), "DISPLAY" | "WAYLAND_DISPLAY") && !value.is_empty()
    })
}

/// Window close-requested handler. Mirrors the quit policy in
/// `commands::do_request_quit`: any in-flight turn prompts the frontend
/// for confirmation; an idle window saves the layout and exits.
///
/// Runs synchronously on the event-loop thread. The actual work happens in
/// a spawned task so we don't block the UI on a lock acquisition.
fn on_close_requested(window: &tauri::Window, api: &tauri::CloseRequestApi) {
    let app = window.app_handle().clone();
    let state = app.state::<AppState>().inner().clone();

    // Fast path: `confirm_quit` already ran and triggered `app.exit(0)`.
    // Don't re-prevent the close — the app is already tearing down.
    if state.quit_confirmed.load(Ordering::SeqCst) {
        return;
    }

    api.prevent_close();
    tauri::async_runtime::spawn(async move {
        let busy = {
            let ws = state.workspace.lock().await;
            ws.any_turn_in_progress()
        };
        if busy {
            emit_confirm_quit_requested(&app);
            return;
        }
        {
            let ws = state.workspace.lock().await;
            if let Err(e) = ws.save_layout() {
                eprintln!("failed to save workspace layout on close: {e:#}");
            }
        }
        state.quit_confirmed.store(true, Ordering::SeqCst);
        app.exit(0);
    });
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::has_graphical_session_vars;

    #[test]
    fn graphical_session_is_present_with_display() {
        assert!(has_graphical_session_vars([(
            "DISPLAY".to_owned(),
            ":0".to_owned(),
        )]));
    }

    #[test]
    fn graphical_session_is_present_with_wayland_display() {
        assert!(has_graphical_session_vars([(
            "WAYLAND_DISPLAY".to_owned(),
            "wayland-0".to_owned(),
        )]));
    }

    #[test]
    fn graphical_session_is_absent_when_display_vars_are_missing_or_empty() {
        assert!(!has_graphical_session_vars([
            ("DISPLAY".to_owned(), String::new()),
            ("PATH".to_owned(), "/usr/bin".to_owned()),
        ]));
    }
}
