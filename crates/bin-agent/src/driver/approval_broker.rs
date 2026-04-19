use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use app::{
    CancelToken, ToolApprovalDecision, ToolApprovalRequest as AppToolApprovalRequest, ToolApprover,
};
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use tokio::sync::{Mutex, oneshot};

#[derive(Default, Clone)]
pub(super) struct ApprovalBroker {
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<bool>>>>,
}

impl ApprovalBroker {
    pub(super) async fn resolve(&self, request_id: String, approved: bool) {
        if let Some(tx) = self.pending.lock().await.remove(&request_id) {
            let _ = tx.send(approved);
        }
    }
}

impl ToolApprover for ApprovalBroker {
    fn approve(
        &self,
        requests: Vec<AppToolApprovalRequest>,
        cancel: CancelToken,
    ) -> Pin<Box<dyn Stream<Item = Result<ToolApprovalDecision>> + Send + '_>> {
        let pending = self.pending.clone();
        // Register oneshot senders, then yield decisions as each receiver
        // resolves. FuturesUnordered drives all receivers concurrently so
        // whichever the user approves first flows to the runner first.
        Box::pin(
            futures::stream::once(async move {
                let mut receivers = Vec::with_capacity(requests.len());
                {
                    let mut locked = pending.lock().await;
                    for request in requests {
                        let (tx, rx) = oneshot::channel();
                        locked.insert(request.request_id.clone(), tx);
                        receivers.push((request.request_id, rx));
                    }
                }

                let futures: FuturesUnordered<_> = receivers
                    .into_iter()
                    .map(|(request_id, mut rx)| {
                        let cancel = cancel.clone();
                        let pending = pending.clone();
                        async move {
                            loop {
                                if cancel.is_cancelled() {
                                    pending.lock().await.remove(&request_id);
                                    return Ok(ToolApprovalDecision {
                                        request_id,
                                        approved: false,
                                    });
                                }
                                tokio::select! {
                                    decision = &mut rx => {
                                        return Ok(ToolApprovalDecision {
                                            request_id,
                                            approved: decision.unwrap_or(false),
                                        });
                                    }
                                    _ = tokio::time::sleep(Duration::from_millis(25)) => {}
                                }
                            }
                        }
                    })
                    .collect();
                futures
            })
            .flatten(),
        )
    }
}
