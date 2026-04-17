//! Per-split drain task: forwards `AgentEventStream` into `SplitObject`.
//!
//! The stream is an `mpsc::UnboundedReceiver<AgentEvent>` fed by the
//! `AgentClient`'s reader task (which runs on the tokio runtime). We pump
//! events into the split's `handle_event` from `glib::spawn_future_local`
//! so the mutation lands on the GTK main thread where every bind_property
//! and ListStore expects to be touched.
//!
//! `clone!(#[weak] split)` drops the closure silently once the SplitObject
//! is gone — closing a split tears down the client (which stops the reader
//! task) and then the weak capture turns the next recv branch into a no-op.

use agent_host::AgentEventStream;
use gtk::glib;
use gtk::glib::clone;
use protocol::AgentEvent;

use crate::objects::SplitObject;

/// Spawn the main-thread drain that funnels stream events into the split.
/// Takes the stream by value so the caller can't accidentally hold on to
/// it; the loop owns it until the agent hangs up.
pub fn spawn_drain_task(split: &SplitObject, mut stream: AgentEventStream) {
    glib::spawn_future_local(clone!(
        #[weak]
        split,
        async move {
            while let Some(event) = stream.recv().await {
                split.handle_event(event);
            }
            // Stream ended: reader task saw EOF and the writer task has
            // also dropped its sender. Surface that to the UI as an Error
            // event so the bottom strip can show a terminal state instead
            // of hanging on "waiting" forever.
            split.handle_event(AgentEvent::Error {
                message: "agent disconnected".into(),
            });
        }
    ));
}
