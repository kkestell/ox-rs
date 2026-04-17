//! `SplitObject` — the per-split GObject the WebView and signal handlers
//! bind against.
//!
//! Owns the `AgentClient` for its split (so closing the split drops the
//! subprocess via `kill_on_drop`), the receive-side `SplitState` used for
//! invariant tracking and partial-streaming commits, and a handle to the
//! per-split `webkit6::WebView` used as the transcript surface. UI properties
//! (waiting, error_text, cancelled, focused, session_id) are exposed as glib
//! properties so labels and bottom-strip widgets can drive their visibility
//! via `bind_property` rather than imperative event hooks.
//!
//! `handle_event` and `submit_draft` both run on the GTK main thread; the
//! state is `RefCell`-locked rather than mutex-locked because no background
//! thread can ever call into either method (the drain task that pumps events
//! into `handle_event` runs under `glib::spawn_future_local`, also on the
//! main thread).

use std::cell::{Cell, RefCell};
use std::path::Path;

use agent_host::{AgentClient, SplitId};
use gtk::glib;
use gtk::prelude::*;
use gtk::subclass::prelude::*;
use protocol::{AgentCommand, AgentEvent};

use super::split_state::{ShouldSend, SplitState, apply_event, begin_send};
use crate::transcript;

mod imp {
    use super::*;

    #[derive(glib::Properties, Default)]
    #[properties(wrapper_type = super::SplitObject)]
    pub struct SplitObject {
        /// `SplitId` stringified. Construct-only — assigned from the
        /// underlying `AgentClient` at build time and stable for the
        /// split's lifetime.
        #[property(get, set, construct_only)]
        pub split_id: RefCell<String>,
        /// Workspace root rendered in the per-split header bar.
        #[property(get, set, construct_only)]
        pub workspace_root: RefCell<String>,
        /// Session UUID the agent reports via `Ready`. Empty until the
        /// handshake completes; non-empty once a session is associated.
        #[property(get, set)]
        pub session_id: RefCell<String>,
        #[property(get, set)]
        pub waiting: Cell<bool>,
        #[property(get, set)]
        pub error_text: RefCell<String>,
        #[property(get, set)]
        pub cancelled: Cell<bool>,
        #[property(get, set)]
        pub focused: Cell<bool>,

        /// Receive-side state machine. Same shape as the deleted
        /// `agent_host::AgentSplit` fields, kept behind a `RefCell` so the
        /// pure `apply_event` function can mutate it.
        pub state: RefCell<SplitState>,
        /// `AgentClient` handle. `Option` only because GObject `Default`
        /// requires it; in practice always `Some` after construction.
        pub client: RefCell<Option<AgentClient>>,
        /// Draft text buffer owned by the split's input `gtk::TextView`.
        /// Stashed here so the `submit-focused` action can read and clear
        /// the draft text without walking the widget tree.
        pub draft_buffer: RefCell<Option<gtk::TextBuffer>>,
        /// The transcript WebView. `None` only between SplitObject
        /// construction and the call to `transcript::build_transcript`,
        /// which sets it via `set_webview`.
        pub webview: RefCell<Option<webkit6::WebView>>,
        /// True once the WebView's load-changed has reached `Finished`.
        /// Until then, `handle_event` keeps mutating state but does not
        /// dispatch JS — the `replay_into_webview` callback re-seeds the
        /// page from accumulated state once the page is ready.
        pub webview_ready: Cell<bool>,
        /// Cached root widget. Built once by `split_view::build_split_view`
        /// and reused across paned-tree rebuilds — without this, every
        /// `add_split`/`close_split` would tear down and rebuild every
        /// existing split's WebView, forcing a full HTML reload and message
        /// replay per surviving split. Closures inside this widget tree must
        /// capture `SplitObject` weakly to avoid a `split → view → closure
        /// → split` retain cycle that would prevent close.
        pub view: RefCell<Option<gtk::Box>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for SplitObject {
        const NAME: &'static str = "OxSplitObject";
        type Type = super::SplitObject;
    }

    #[glib::derived_properties]
    impl ObjectImpl for SplitObject {}
}

glib::wrapper! {
    pub struct SplitObject(ObjectSubclass<imp::SplitObject>);
}

impl SplitObject {
    /// Build a SplitObject around an already-spawned `AgentClient`.
    pub fn new(client: AgentClient, workspace_root: &Path) -> Self {
        let split_id_str = client.id().0.to_string();
        let workspace_root_str = workspace_root.display().to_string();
        let obj: Self = glib::Object::builder()
            .property("split-id", &split_id_str)
            .property("workspace-root", &workspace_root_str)
            .property("session-id", String::new())
            .property("error-text", String::new())
            .build();
        *obj.imp().client.borrow_mut() = Some(client);
        obj
    }

    /// `SplitId` reconstructed from the `split-id` string property.
    pub fn split_id_uuid(&self) -> SplitId {
        let s = self.split_id();
        SplitId(uuid::Uuid::parse_str(&s).expect("split-id is always a UUID"))
    }

    /// True when the split has a turn in flight or partial streaming
    /// content. Used by quit-confirmation before tearing the agent down.
    pub fn is_turn_in_progress(&self) -> bool {
        self.imp().state.borrow().is_turn_in_progress()
    }

    /// Apply one event from the agent. Runs on the GTK main thread:
    /// mutates `SplitState`, then reconciles the notify-able properties
    /// and dispatches a JS update into the transcript WebView.
    pub fn handle_event(&self, event: AgentEvent) {
        let imp = self.imp();
        let pre_msg_len = imp.state.borrow().messages.len();

        // Pure state transition first; UI reconciliation second.
        apply_event(&mut imp.state.borrow_mut(), event);

        // Reconcile property surface (notify-driven bindings refresh on set).
        let (waiting, error, cancelled, session_id) = {
            let s = imp.state.borrow();
            (
                s.waiting,
                s.error.clone(),
                s.cancelled,
                s.session_id.map(|id| id.to_string()),
            )
        };
        self.set_waiting(waiting);
        self.set_error_text(error.unwrap_or_default());
        self.set_cancelled(cancelled);
        if let Some(id) = session_id
            && self.session_id() != id
        {
            self.set_session_id(id);
        }

        // Build the JS update for the transcript and dispatch it. If the
        // WebView isn't ready yet, skip — `replay_into_webview` will
        // re-seed the page from full state once load-changed fires.
        if !imp.webview_ready.get() {
            return;
        }
        let Some(webview) = imp.webview.borrow().clone() else {
            return;
        };

        let post_state = imp.state.borrow();
        let mut script = String::new();
        for m in &post_state.messages[pre_msg_len..] {
            script.push_str(&transcript::js_append_message(m.role.clone(), &m.content));
        }
        match post_state.streaming.as_ref() {
            Some(acc) => {
                let blocks = acc.snapshot().content.to_vec();
                script.push_str(&transcript::js_set_streaming(&blocks));
            }
            None => {
                script.push_str(transcript::js_clear_streaming());
            }
        }
        drop(post_state);

        if !script.is_empty() {
            transcript::run_js(&webview, &script);
        }
    }

    /// User typed Enter on the input. Consults `begin_send` for the
    /// double-Enter guard, flips the bottom-strip properties, and dispatches
    /// the IPC `SendMessage`. The user row itself is *not* appended here —
    /// the agent emits a `MessageAppended` for the user message as the first
    /// event of the turn and `handle_event` adds the row from there. That
    /// keeps the rendered transcript and the agent's persisted history
    /// strictly in sync, with one code path that handles both fresh sends
    /// and replayed history.
    pub fn submit_draft(&self, draft: String) {
        let imp = self.imp();

        let outcome = begin_send(&mut imp.state.borrow_mut());
        if outcome == ShouldSend::Skip {
            return;
        }

        self.set_waiting(true);
        self.set_error_text(String::new());
        self.set_cancelled(false);

        if let Some(client) = imp.client.borrow().as_ref() {
            let _ = client.send(AgentCommand::SendMessage { input: draft });
        }
    }

    /// Stash the draft buffer owned by this split's input `TextView`.
    pub fn set_draft_buffer(&self, buffer: gtk::TextBuffer) {
        *self.imp().draft_buffer.borrow_mut() = Some(buffer);
    }

    /// Clone of the current draft buffer, if the input strip has been built.
    pub fn draft_buffer(&self) -> Option<gtk::TextBuffer> {
        self.imp().draft_buffer.borrow().clone()
    }

    /// Stash the WebView instance the transcript module built. Called once
    /// during split-view assembly.
    pub fn set_webview(&self, webview: webkit6::WebView) {
        *self.imp().webview.borrow_mut() = Some(webview);
    }

    /// Re-seed the WebView from accumulated state. Called by
    /// `transcript::build_transcript` once the page reaches
    /// `LoadEvent::Finished` so events that landed during the page-load
    /// window are not lost.
    pub fn replay_into_webview(&self, webview: &webkit6::WebView) {
        let imp = self.imp();
        imp.webview_ready.set(true);

        let state = imp.state.borrow();
        let streaming_blocks = state
            .streaming
            .as_ref()
            .map(|acc| acc.snapshot().content.to_vec());
        let script = transcript::js_replay(&state.messages, streaming_blocks.as_deref());
        drop(state);
        transcript::run_js(webview, &script);
    }

    /// Tell the agent to abandon the in-flight turn.
    pub fn cancel(&self) {
        if let Some(client) = self.imp().client.borrow().as_ref() {
            let _ = client.send(AgentCommand::Cancel);
        }
    }

    /// Cached root widget, if `set_cached_view` has been called for this
    /// split. Used by `splits::rebuild_paned_tree` to reuse a split's
    /// widget tree across rebuilds.
    pub fn cached_view(&self) -> Option<gtk::Box> {
        self.imp().view.borrow().clone()
    }

    /// Stash the freshly-built root widget so subsequent rebuilds reuse it.
    pub fn set_cached_view(&self, view: gtk::Box) {
        *self.imp().view.borrow_mut() = Some(view);
    }
}
