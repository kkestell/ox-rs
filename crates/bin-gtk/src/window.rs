//! `OxWindow` — the per-application `adw::ApplicationWindow` subclass.
//!
//! Owns every running split's `SplitObject` (in a `gio::ListStore`), the
//! `AgentSpawnConfig` used to birth new agents, the shared tokio
//! `Runtime` handle the `AgentClient::spawn` calls route through, and the
//! workspace layout persistence.
//!
//! Methods on this type are the surface area that `actions::*` and menu
//! item handlers target. Everything UI-visible happens on the GTK main
//! thread; the runtime is only entered briefly inside `spawn` to let the
//! tokio reader/writer tasks get scheduled.
//!
//! The subprocess lifetime story: dropping a `SplitObject` drops its
//! `AgentClient`, which triggers `kill_on_drop` on the child process.
//! Closing a split therefore SIGKILLs the agent without any explicit
//! teardown sequence.

use std::cell::{Cell, OnceCell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;
use adw::subclass::prelude::*;
use agent_host::{
    AgentClient, AgentEventStream, AgentSpawnConfig, RestoreLayout, SplitId, WorkspaceLayouts,
    normalize_split_fracs,
};
use anyhow::{Context, Result};
use domain::SessionId;
use gtk::gio;
use gtk::glib;
use tokio::runtime::Runtime;

use crate::events;
use crate::objects::SplitObject;

/// Outcome of `close_split` — the caller (the `close-split` action) uses
/// this to decide whether closing the last split should instead trigger
/// the quit-confirm flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseOutcome {
    pub last_split_closed: bool,
}

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct OxWindow {
        /// Tokio runtime the `AgentClient::spawn` calls are entered into.
        /// `Rc` so we can hand it to action handlers; a multi-thread
        /// runtime would work too, but `current_thread` keeps IPC on one
        /// background thread and is plenty for a dozen subprocesses.
        pub runtime: OnceCell<Rc<Runtime>>,
        /// Template spawn config — per-spawn we clone and tweak (`resume`,
        /// `workspace_root`) before handing to `AgentClient::spawn`.
        pub spawn_config: OnceCell<AgentSpawnConfig>,
        /// Persisted layouts: workspace_root → saved split layout. Loaded
        /// once on construction; mutated when the workspace changes and
        /// flushed to disk from `save_layout`.
        pub layouts: RefCell<WorkspaceLayouts>,
        /// Path on disk where `layouts` is persisted. Stashed alongside
        /// the layouts themselves so we don't need to thread the path
        /// through every call site.
        pub layout_path: OnceCell<PathBuf>,
        /// The splits themselves. `gio::ListStore` so future work can back
        /// a model-view binding off it if we ever render the split list
        /// itself as a widget; today the splits are walked via `n_items`.
        pub splits: OnceCell<gio::ListStore>,
        /// Current workspace root. Mutated by `replace_workspace`; stable
        /// for the lifetime of a given "workspace session."
        pub workspace_root: RefCell<PathBuf>,
        /// `SplitId` of the focused split. `None` only transiently during
        /// `replace_workspace` swap; otherwise always points at a real
        /// entry in `splits`.
        pub focused: RefCell<Option<SplitId>>,
        /// Horizontal split fractions. Always has the same length as
        /// `splits.n_items()`; every mutation passes through
        /// `normalize_split_fracs`.
        pub split_fracs: RefCell<Vec<f32>>,
        /// The `gtk::Box` rendered as the window's content. `splits.rs`
        /// rebuilds its children from scratch whenever the split count or
        /// fractions change.
        pub split_container: OnceCell<gtk::Box>,
        /// Set when `confirm_quit` has decided to close. Drives
        /// `close_request` to propagate instead of re-prompting.
        pub quit_pending: Cell<bool>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for OxWindow {
        const NAME: &'static str = "OxOxWindow";
        type Type = super::OxWindow;
        type ParentType = adw::ApplicationWindow;
    }

    impl ObjectImpl for OxWindow {
        fn constructed(&self) {
            self.parent_constructed();
            // Splits store must exist before any widget code can append
            // rows. Created here — never replaced.
            let store = gio::ListStore::new::<SplitObject>();
            let _ = self.splits.set(store);
        }
    }

    impl WidgetImpl for OxWindow {}
    impl WindowImpl for OxWindow {
        fn close_request(&self) -> glib::Propagation {
            // First close attempt: delegate to `request_quit` which may
            // pop a confirmation dialog; stop propagation so the window
            // doesn't actually close yet. Once `confirm_quit` runs and
            // flips `quit_pending`, a re-entry here propagates.
            if self.quit_pending.get() {
                glib::Propagation::Proceed
            } else {
                self.obj().request_quit();
                glib::Propagation::Stop
            }
        }
    }
    impl ApplicationWindowImpl for OxWindow {}
    impl AdwApplicationWindowImpl for OxWindow {}
}

glib::wrapper! {
    pub struct OxWindow(ObjectSubclass<imp::OxWindow>)
        @extends adw::ApplicationWindow, gtk::ApplicationWindow, gtk::Window, gtk::Widget,
        @implements gio::ActionGroup, gio::ActionMap, gtk::Accessible, gtk::Buildable,
                    gtk::ConstraintTarget, gtk::Native, gtk::Root, gtk::ShortcutManager;
}

impl OxWindow {
    /// Build a window wired to the given runtime + spawn config + layouts.
    /// Does not spawn any agents — `restore_workspace` is the entry point
    /// that reads the layout store and birth-spawns the initial splits.
    pub fn new(
        app: &adw::Application,
        runtime: Rc<Runtime>,
        spawn_config: AgentSpawnConfig,
        layouts: WorkspaceLayouts,
        layout_path: PathBuf,
    ) -> Self {
        let obj: Self = glib::Object::builder().property("application", app).build();
        obj.set_default_size(1280, 800);
        obj.set_title(Some("Ox"));

        let imp = obj.imp();
        let workspace_root = spawn_config.workspace_root.clone();
        let _ = imp.runtime.set(runtime);
        let _ = imp.spawn_config.set(spawn_config);
        *imp.layouts.borrow_mut() = layouts;
        let _ = imp.layout_path.set(layout_path);
        *imp.workspace_root.borrow_mut() = workspace_root;

        obj.build_chrome();
        obj
    }

    /// Populate the `adw::ToolbarView` chrome: header bar with hamburger
    /// menu button, body area reserved for the split container. The
    /// split container itself stays empty until `restore_workspace` runs.
    fn build_chrome(&self) {
        let imp = self.imp();

        let toolbar = adw::ToolbarView::new();
        let header = adw::HeaderBar::new();

        let menu_button = gtk::MenuButton::new();
        menu_button.set_icon_name("open-menu-symbolic");
        menu_button.set_menu_model(Some(&build_menu_model()));
        header.pack_end(&menu_button);
        toolbar.add_top_bar(&header);

        let split_container = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        split_container.set_hexpand(true);
        split_container.set_vexpand(true);
        toolbar.set_content(Some(&split_container));
        let _ = imp.split_container.set(split_container);

        self.set_content(Some(&toolbar));
    }

    // ------------------------------------------------------------------
    // Accessors used by actions / splits / child widgets
    // ------------------------------------------------------------------

    /// Backing store. Handed out so `splits::rebuild_paned_tree` can walk
    /// it without needing an `OxWindow` subclass method per iteration.
    pub fn splits_store(&self) -> gio::ListStore {
        self.imp()
            .splits
            .get()
            .expect("splits store initialized in constructed")
            .clone()
    }

    pub fn split_container(&self) -> gtk::Box {
        self.imp()
            .split_container
            .get()
            .expect("split_container initialized in build_chrome")
            .clone()
    }

    pub fn workspace_root(&self) -> PathBuf {
        self.imp().workspace_root.borrow().clone()
    }

    pub fn focused(&self) -> Option<SplitId> {
        *self.imp().focused.borrow()
    }

    /// Split fractions snapshot; returned by value so callers don't hold
    /// a borrow across other window methods.
    pub fn split_fracs(&self) -> Vec<f32> {
        self.imp().split_fracs.borrow().clone()
    }

    /// Find the `SplitObject` matching the given id, if still present.
    pub fn split_by_id(&self, id: SplitId) -> Option<SplitObject> {
        let store = self.splits_store();
        for i in 0..store.n_items() {
            let obj: SplitObject = store
                .item(i)
                .expect("index within n_items")
                .downcast()
                .expect("splits_store holds SplitObject");
            if obj.split_id_uuid() == id {
                return Some(obj);
            }
        }
        None
    }

    /// Currently-focused split, if any.
    pub fn focused_split(&self) -> Option<SplitObject> {
        self.focused().and_then(|id| self.split_by_id(id))
    }

    // ------------------------------------------------------------------
    // Workspace lifecycle
    // ------------------------------------------------------------------

    /// Startup path: consults saved layouts, spawns the initial set of
    /// agents, wires up drain tasks, and triggers the initial paned
    /// rebuild. Mirrors the body of the old `WorkspaceState::restore`.
    ///
    /// Returns `Err` only when every spawn attempt (both the saved
    /// layout's and the fresh fallback) fails — i.e. the `ox-agent`
    /// binary itself is unreachable. A per-split spawn failure falls
    /// through to the fresh-spawn retry.
    pub fn restore_workspace(&self) -> Result<()> {
        let imp = self.imp();
        let spawn_config = imp.spawn_config.get().expect("spawn_config set").clone();

        // If `--resume` was passed explicitly on the CLI, it wins over any
        // saved layout. Otherwise consult the layout store.
        let restore: Option<RestoreLayout> = if spawn_config.resume.is_some() {
            None
        } else {
            imp.layouts
                .borrow()
                .restore_existing_for(&spawn_config.workspace_root, &spawn_config.sessions_dir)
        };
        let explicit_resume = spawn_config.resume;
        let spawn_configs = startup_spawn_configs(&spawn_config, restore.as_ref());

        let _guard = imp.runtime.get().expect("runtime set").enter();

        let mut spawned: Vec<(AgentClient, AgentEventStream)> = Vec::new();
        let mut restored_spawn_failed = false;
        for config in &spawn_configs {
            match AgentClient::spawn(config.clone()) {
                Ok(pair) => spawned.push(pair),
                Err(_err) if restore.is_some() && explicit_resume.is_none() => {
                    restored_spawn_failed = true;
                    break;
                }
                Err(err) => return Err(err).context("spawning initial agent"),
            }
        }

        // If a restored layout turned out to point at sessions the agent
        // can't resume, drop everything we spawned so far and start one
        // fresh agent instead.
        if restored_spawn_failed {
            drop(spawned);
            let mut fresh = spawn_config.clone();
            fresh.resume = None;
            let pair = AgentClient::spawn(fresh)
                .context("spawning fresh agent after restored workspace layout failed")?;
            spawned = vec![pair];
        }

        let n = spawned.len();
        let (split_fracs, focused_idx) = if restored_spawn_failed {
            (vec![1.0 / n as f32; n], 0)
        } else {
            (
                restore
                    .as_ref()
                    .map(|layout| normalize_split_fracs(&layout.split_fracs, n))
                    .unwrap_or_else(|| vec![1.0 / n as f32; n]),
                restore
                    .as_ref()
                    .map(|layout| layout.focused.min(n.saturating_sub(1)))
                    .unwrap_or(0),
            )
        };

        self.install_splits(spawned, split_fracs, focused_idx);
        Ok(())
    }

    /// Append a fresh split on the right edge. Focus moves to the new
    /// split; fractions redistribute equally.
    pub fn add_split(&self) -> Result<()> {
        let imp = self.imp();
        let mut config = imp.spawn_config.get().expect("spawn_config set").clone();
        config.workspace_root = self.workspace_root();
        config.resume = None;

        let _guard = imp.runtime.get().expect("runtime set").enter();
        let (client, stream) = AgentClient::spawn(config).context("spawning new agent")?;

        let split = SplitObject::new(client, &self.workspace_root());
        let new_id = split.split_id_uuid();
        let store = self.splits_store();
        store.append(&split);
        events::spawn_drain_task(&split, stream);

        let n = store.n_items() as usize;
        *imp.split_fracs.borrow_mut() = vec![1.0 / n as f32; n];
        self.set_focused_split(new_id);
        crate::splits::rebuild_paned_tree(self);
        Ok(())
    }

    /// Remove a split by id. Returns `last_split_closed: true` *without*
    /// actually removing when the target is the only remaining split —
    /// the caller is responsible for the app-quit path in that case.
    pub fn close_split(&self, id: SplitId) -> CloseOutcome {
        let store = self.splits_store();
        let n = store.n_items();
        let Some(idx) = self.index_of(id) else {
            return CloseOutcome {
                last_split_closed: false,
            };
        };
        if n == 1 {
            return CloseOutcome {
                last_split_closed: true,
            };
        }

        let imp = self.imp();
        // Reclaim the closed split's fraction into a neighbor so the
        // remaining splits fill the window without a sudden resize.
        {
            let mut fracs = imp.split_fracs.borrow_mut();
            if (idx as usize) < fracs.len() {
                let reclaimed = fracs.remove(idx as usize);
                let neighbor = if (idx as usize) < fracs.len() {
                    idx as usize
                } else {
                    fracs.len().saturating_sub(1)
                };
                if !fracs.is_empty() {
                    fracs[neighbor] += reclaimed;
                }
            }
        }

        store.remove(idx);
        self.adjust_focus_after_remove(idx as usize);
        crate::splits::rebuild_paned_tree(self);
        CloseOutcome {
            last_split_closed: false,
        }
    }

    /// Atomic spawn-then-swap replacement. Saves the outgoing layout
    /// first, then tries to spawn every new agent; on any spawn failure
    /// the old state is preserved and the focused split's error is set.
    pub fn replace_workspace(&self, new_root: PathBuf) -> Result<()> {
        self.save_layout();

        let imp = self.imp();
        let mut next_config = imp.spawn_config.get().expect("spawn_config set").clone();
        next_config.workspace_root = new_root.clone();
        next_config.resume = None;

        let restore: Option<RestoreLayout> = imp
            .layouts
            .borrow()
            .restore_existing_for(&new_root, &next_config.sessions_dir);
        let spawn_configs = restore_spawn_configs(&next_config, restore.as_ref());

        let _guard = imp.runtime.get().expect("runtime set").enter();
        let mut spawned: Vec<(AgentClient, AgentEventStream)> = Vec::new();
        for config in &spawn_configs {
            match AgentClient::spawn(config.clone()) {
                Ok(pair) => spawned.push(pair),
                Err(e) => {
                    if let Some(focused) = self.focused_split() {
                        focused.set_error_text(format!("failed to spawn agent: {e:#}"));
                    }
                    return Ok(());
                }
            }
        }

        *imp.workspace_root.borrow_mut() = new_root;
        let n = spawned.len();
        let split_fracs = restore
            .as_ref()
            .map(|layout| normalize_split_fracs(&layout.split_fracs, n))
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        let focused_idx = restore
            .as_ref()
            .map(|layout| layout.focused.min(n.saturating_sub(1)))
            .unwrap_or(0);

        // Swap only after spawns all succeeded.
        let store = self.splits_store();
        store.remove_all();
        self.install_splits(spawned, split_fracs, focused_idx);
        Ok(())
    }

    /// Set the focused split; notifies every split object so bound
    /// widgets can react to `focused` transitions.
    pub fn set_focused_split(&self, id: SplitId) {
        let imp = self.imp();
        *imp.focused.borrow_mut() = Some(id);
        let store = self.splits_store();
        for i in 0..store.n_items() {
            let split: SplitObject = store
                .item(i)
                .expect("index within n_items")
                .downcast()
                .expect("SplitObject");
            split.set_focused(split.split_id_uuid() == id);
        }
    }

    /// Store fresh split fractions (normalized) — called by the drag
    /// handler in `splits.rs` so live drags are reflected in saved
    /// layouts.
    pub fn set_split_fractions(&self, fracs: Vec<f32>) {
        let imp = self.imp();
        let n = self.splits_store().n_items() as usize;
        *imp.split_fracs.borrow_mut() = normalize_split_fracs(&fracs, n);
    }

    /// Flush the current layout to disk. Silently drops failures onto
    /// stderr — a layout save failure shouldn't block a quit.
    pub fn save_layout(&self) {
        let imp = self.imp();
        let path = imp.layout_path.get().expect("layout_path set").clone();
        let workspace_root = self.workspace_root();
        let store = self.splits_store();
        let focused_idx = self.focused_index().unwrap_or(0);
        let fracs = imp.split_fracs.borrow().clone();

        let session_ids: Vec<Option<SessionId>> = (0..store.n_items())
            .map(|i| {
                let split: SplitObject = store
                    .item(i)
                    .expect("index within n_items")
                    .downcast()
                    .expect("SplitObject");
                parse_session_id(&split.session_id())
            })
            .collect();

        let mut layouts = imp.layouts.borrow_mut();
        let saved = layouts.save_current(&workspace_root, session_ids, &fracs, focused_idx);
        if saved && let Err(e) = layouts.save(&path) {
            eprintln!("failed to save workspace layout: {e:#}");
        }
    }

    /// Quit-confirm entry point: if any split has a turn in flight,
    /// pops the confirmation alert dialog; otherwise closes immediately.
    pub fn request_quit(&self) {
        if self.any_turn_in_progress() {
            let window_for_cb = self.clone();
            crate::modals::present_quit_confirm(self, move || {
                window_for_cb.confirm_quit();
            });
        } else {
            self.confirm_quit();
        }
    }

    /// Commit to shutting down: flush layout, flip `quit_pending` so the
    /// next `close_request` propagates, then close.
    pub fn confirm_quit(&self) {
        self.save_layout();
        self.imp().quit_pending.set(true);
        self.close();
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// Shared tail of `restore_workspace` and `replace_workspace`: given
    /// a vector of freshly-spawned `(client, stream)` pairs, build the
    /// `SplitObject`s, append them to the store, wire drain tasks, and
    /// rebuild the paned layout.
    fn install_splits(
        &self,
        spawned: Vec<(AgentClient, AgentEventStream)>,
        split_fracs: Vec<f32>,
        focused_idx: usize,
    ) {
        let imp = self.imp();
        let workspace_root = self.workspace_root();
        let store = self.splits_store();

        *imp.split_fracs.borrow_mut() = split_fracs;
        let mut focused_id: Option<SplitId> = None;
        for (i, (client, stream)) in spawned.into_iter().enumerate() {
            let split = SplitObject::new(client, &workspace_root);
            let id = split.split_id_uuid();
            if i == focused_idx {
                focused_id = Some(id);
            }
            store.append(&split);
            events::spawn_drain_task(&split, stream);
        }
        if let Some(id) = focused_id {
            self.set_focused_split(id);
        }
        crate::splits::rebuild_paned_tree(self);
    }

    fn index_of(&self, id: SplitId) -> Option<u32> {
        let store = self.splits_store();
        for i in 0..store.n_items() {
            let split: SplitObject = store
                .item(i)
                .expect("index within n_items")
                .downcast()
                .expect("SplitObject");
            if split.split_id_uuid() == id {
                return Some(i);
            }
        }
        None
    }

    fn focused_index(&self) -> Option<usize> {
        self.focused()
            .and_then(|id| self.index_of(id).map(|i| i as usize))
    }

    /// Clamp `focused` to a sensible neighbor after a removal at
    /// `removed_idx`. Mirrors the old `adjust_focus_after_remove`.
    fn adjust_focus_after_remove(&self, removed_idx: usize) {
        let store = self.splits_store();
        let new_len = store.n_items() as usize;
        if new_len == 0 {
            *self.imp().focused.borrow_mut() = None;
            return;
        }
        let next_idx = removed_idx.min(new_len - 1);
        let split: SplitObject = store
            .item(next_idx as u32)
            .expect("next_idx within n_items")
            .downcast()
            .expect("SplitObject");
        self.set_focused_split(split.split_id_uuid());
    }

    /// Any split currently has a turn in flight or partial streaming.
    /// The quit-confirmation flow keys off this; only UI-facing bits are
    /// inspected (waiting flag, streaming accumulator presence).
    fn any_turn_in_progress(&self) -> bool {
        let store = self.splits_store();
        for i in 0..store.n_items() {
            let split: SplitObject = store
                .item(i)
                .expect("index within n_items")
                .downcast()
                .expect("SplitObject");
            if split.is_turn_in_progress() {
                return true;
            }
        }
        false
    }
}

/// Build the "hamburger menu" `gio::Menu` attached to the header bar.
/// Targets the app-level actions registered in `app.rs`.
fn build_menu_model() -> gio::Menu {
    let menu = gio::Menu::new();
    menu.append(Some("New Split"), Some("app.new-split"));
    menu.append(Some("Close Split"), Some("app.close-split"));
    menu.append(Some("Open Workspace…"), Some("app.open-workspace"));
    let meta = gio::Menu::new();
    meta.append(Some("About Ox"), Some("app.about"));
    meta.append(Some("Quit"), Some("app.quit"));
    menu.append_section(None, &meta);
    menu
}

fn parse_session_id(s: &str) -> Option<SessionId> {
    if s.is_empty() {
        return None;
    }
    s.parse().ok()
}

fn restore_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    match restore {
        Some(layout) => layout
            .sessions
            .iter()
            .map(|id| {
                let mut config = base_config.clone();
                config.resume = Some(*id);
                config
            })
            .collect(),
        None => {
            let mut config = base_config.clone();
            config.resume = None;
            vec![config]
        }
    }
}

fn startup_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    if base_config.resume.is_some() {
        vec![base_config.clone()]
    } else {
        restore_spawn_configs(base_config, restore)
    }
}
