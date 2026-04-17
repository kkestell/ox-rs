//! WebKitGTK-backed transcript view.
//!
//! The transcript is rendered as HTML inside a single `webkit6::WebView`
//! per split. This is the only way to get true cross-block text selection,
//! Ctrl+A select-all, and native browser copy semantics in a native GTK
//! app — every per-widget approach (Label, TextView, custom widget) has
//! per-widget selection that can't span multiple bubbles.
//!
//! All transcript mutation goes through `evaluate_javascript` calls into
//! the page; Rust pre-encodes ContentBlocks as JSON via serde and passes
//! them to JS helpers in the page that build/replace DOM nodes.
//!
//! The WebView is stashed on the `SplitObject` (alongside the draft buffer)
//! so `SplitObject::handle_event` can push updates without walking the
//! widget tree.

use domain::{ContentBlock, Message, Role};
use gtk::prelude::*;
use webkit6::prelude::*;

use crate::objects::SplitObject;

const PAGE_HTML: &str = include_str!("../assets/transcript.html");

/// Build the WebView used as the transcript surface for one split. The
/// returned widget is packed into the split's vertical box; the WebView
/// itself is stashed on `split` so `handle_event` can drive it via JS.
pub fn build_transcript(split: &SplitObject) -> gtk::Widget {
    let webview = webkit6::WebView::new();
    webview.set_hexpand(true);
    webview.set_vexpand(true);
    // Transparent background so the GTK window's color shows through
    // until the page is ready and renders its own background.
    webview.set_background_color(&gtk::gdk::RGBA::new(
        0x1d as f32 / 255.0,
        0x1d as f32 / 255.0,
        0x20 as f32 / 255.0,
        1.0,
    ));

    // Load the page template. `load_html` does not block — page-load is
    // async; queued JS calls before load completes are dropped silently
    // by WebKit. We guard against that by buffering any pending JS until
    // load-changed reaches Finished.
    webview.load_html(PAGE_HTML, None);

    split.set_webview(webview.clone());

    // Once the page is ready, replay whatever state has accumulated in
    // the meantime. In practice handle_event runs on the GTK main thread
    // (same thread as load-changed), so the only events the WebView can
    // miss are those between widget construction and the first idle pump.
    //
    // Weak ref: the SplitObject caches the view (which contains this
    // WebView), so a strong capture here would create a retain cycle.
    let split_weak = split.downgrade();
    webview.connect_load_changed(move |wv, event| {
        if event == webkit6::LoadEvent::Finished
            && let Some(split) = split_weak.upgrade()
        {
            split.replay_into_webview(wv);
        }
    });

    webview.upcast()
}

/// JS expression that appends a committed message row to the transcript.
/// `role` is "user" / "assistant" / "tool"; `blocks_json` is the JSON
/// representation of `Vec<ContentBlock>` (same shape `serde_json` produces).
pub fn js_append_message(role: Role, blocks: &[ContentBlock]) -> String {
    let role_str = role_to_str(role);
    let blocks_json = serde_json::to_string(blocks).unwrap_or_else(|_| "[]".into());
    format!("appendMessage({}, {});", json_str(role_str), blocks_json)
}

/// JS expression that replaces the in-flight streaming row's content.
/// The streaming row is always the assistant role.
pub fn js_set_streaming(blocks: &[ContentBlock]) -> String {
    let blocks_json = serde_json::to_string(blocks).unwrap_or_else(|_| "[]".into());
    format!("setStreaming({blocks_json});")
}

/// JS expression that drops the streaming row.
pub fn js_clear_streaming() -> &'static str {
    "clearStreaming();"
}

/// Build the JS that re-seeds the page from a vector of committed messages
/// plus an optional streaming snapshot. Used by the load-finished handler
/// so the page renders any state that landed before the page was ready.
pub fn js_replay(messages: &[Message], streaming: Option<&[ContentBlock]>) -> String {
    let mut out = String::from("clearAll();");
    for m in messages {
        out.push_str(&js_append_message(m.role.clone(), &m.content));
    }
    if let Some(blocks) = streaming {
        out.push_str(&js_set_streaming(blocks));
    }
    out
}

/// Run a JS snippet in the WebView. Failures are logged to stderr — the
/// WebView reports script errors via `result` callbacks; we don't have any
/// recovery path beyond noting that an update was lost.
pub fn run_js(webview: &webkit6::WebView, script: &str) {
    webview.evaluate_javascript(
        script,
        None,
        None,
        None::<&gtk::gio::Cancellable>,
        |result| {
            if let Err(e) = result {
                eprintln!("transcript JS evaluation failed: {e}");
            }
        },
    );
}

fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// JSON-encode a string literal so it can be inlined into a JS expression
/// without manual escaping.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}
