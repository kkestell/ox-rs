use std::future::Future;
use std::pin::Pin;

use agent_host::{CloseRequestSink, FirstTurnSink};
use domain::{CloseIntent, SessionId};
use tokio::sync::mpsc;

/// Agent-initiated close request message. Pushed onto the close
/// channel by a [`ChannelCloseSink`]; drained by the consumer task
/// that calls [`super::SessionLifecycle::handle_close_request`].
#[derive(Debug)]
pub struct CloseRequestMsg {
    pub id: SessionId,
    pub intent: CloseIntent,
}

/// First-turn completion message. Pushed by a
/// [`ChannelFirstTurnSink`]; drained by the consumer task that calls
/// [`super::SessionLifecycle::handle_first_turn`].
#[derive(Debug)]
pub struct FirstTurnMsg {
    pub id: SessionId,
    pub first_message: String,
}

/// [`CloseRequestSink`] that pushes requests onto an unbounded channel.
/// Sessions hold one of these behind `Arc<dyn CloseRequestSink>`; the
/// composition root owns a consumer task that drains the receiver and
/// dispatches to [`super::SessionLifecycle::handle_close_request`].
///
/// Send is sync (unbounded channels never block); the returned future
/// only exists to satisfy the trait's `dyn`-friendly signature.
pub struct ChannelCloseSink {
    tx: mpsc::UnboundedSender<CloseRequestMsg>,
}

impl ChannelCloseSink {
    pub fn new(tx: mpsc::UnboundedSender<CloseRequestMsg>) -> Self {
        Self { tx }
    }
}

impl CloseRequestSink for ChannelCloseSink {
    fn request_close(
        &self,
        id: SessionId,
        intent: CloseIntent,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let result = self.tx.send(CloseRequestMsg { id, intent });
        Box::pin(async move {
            if result.is_err() {
                eprintln!("ox: close request for session {id} dropped: consumer channel closed");
            }
        })
    }
}

/// [`FirstTurnSink`] that pushes notifications onto an unbounded
/// channel. Mirror of [`ChannelCloseSink`] for the slug-rename path.
pub struct ChannelFirstTurnSink {
    tx: mpsc::UnboundedSender<FirstTurnMsg>,
}

impl ChannelFirstTurnSink {
    pub fn new(tx: mpsc::UnboundedSender<FirstTurnMsg>) -> Self {
        Self { tx }
    }
}

impl FirstTurnSink for ChannelFirstTurnSink {
    fn on_first_turn_complete(
        &self,
        id: SessionId,
        first_message: String,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let result = self.tx.send(FirstTurnMsg { id, first_message });
        Box::pin(async move {
            if result.is_err() {
                eprintln!(
                    "ox: first-turn notification for session {id} dropped: consumer channel closed"
                );
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn channel_close_sink_surfaces_send_error_when_receiver_gone() {
        // Drop the receiver before sending. The sink must not panic —
        // it should return a future that logs and completes. This is
        // the degenerate case where the consumer task has already
        // exited (shutdown in flight).
        let (tx, rx) = mpsc::unbounded_channel::<CloseRequestMsg>();
        drop(rx);
        let sink: Arc<dyn CloseRequestSink> = Arc::new(ChannelCloseSink::new(tx));
        sink.request_close(SessionId::new_v4(), CloseIntent::Merge)
            .await;
    }

    #[tokio::test]
    async fn channel_first_turn_sink_surfaces_send_error_when_receiver_gone() {
        let (tx, rx) = mpsc::unbounded_channel::<FirstTurnMsg>();
        drop(rx);
        let sink: Arc<dyn FirstTurnSink> = Arc::new(ChannelFirstTurnSink::new(tx));
        sink.on_first_turn_complete(SessionId::new_v4(), "hello".into())
            .await;
    }
}
