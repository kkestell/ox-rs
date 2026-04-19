use std::sync::Arc;

use anyhow::Result;
use app::{CancelToken, LlmProvider, SessionRunner, SessionStore, TurnEvent, TurnOutcome};
use domain::SessionId;
use protocol::{AgentEvent, write_frame};
use tokio::io::AsyncWrite;
use tokio::sync::mpsc;

use super::approval_broker::ApprovalBroker;

/// Context capsule for a single `SendMessage` turn. Bundles the
/// per-turn inputs `run_turn` forwards to the `SessionRunner` so the
/// driver's command loop stays readable.
pub(super) struct TurnRun<'a> {
    pub(super) workspace_root: &'a std::path::Path,
    pub(super) session_id: SessionId,
    pub(super) input: &'a str,
    pub(super) model: &'a str,
    pub(super) initialized: bool,
    pub(super) cancel: CancelToken,
    pub(super) approvals: Arc<ApprovalBroker>,
}

/// Run a single turn and stream its events to `writer` as they arrive.
///
/// `SessionRunner`'s callback is synchronous `FnMut`, but `write_frame` is
/// async — a naive `write_frame(...).await` inside the callback would need
/// the callback to be async. Instead we bridge the two worlds with a tokio
/// channel:
///
/// - The synchronous callback pushes each event onto an unbounded sender.
/// - A concurrent "drain" future reads from the receiver and calls
///   `write_frame` for each event.
/// - `tokio::join!` runs the runner and the drain concurrently so deltas
///   reach the wire as soon as the runner emits them, preserving the
///   real-time streaming feel the GUI depends on.
/// - When the runner returns, its closure is dropped, which drops the
///   sender, which closes the channel, which causes the drain future to
///   finish after flushing the last events.
pub(super) async fn run_turn<L, S, W>(
    runner: &SessionRunner<L, S>,
    turn: TurnRun<'_>,
    writer: &mut W,
) -> Result<TurnOutcome>
where
    L: LlmProvider + Send + Sync + 'static,
    S: SessionStore + Send + Sync + 'static,
    W: AsyncWrite + Unpin,
{
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Runner future: owns `tx` (via the callback), drops it when finished.
    // The callback is `FnMut` because `SessionRunner::start` / `resume`
    // want `FnMut`, but it only needs `&tx` — unbounded senders are `Sync`
    // and `send(&self)` is enough.
    let workspace = turn.workspace_root.to_path_buf();
    let run_fut = async move {
        let callback = |evt: TurnEvent<'_>| match evt {
            TurnEvent::StreamDelta(e) => {
                let _ = tx.send(AgentEvent::StreamDelta { event: e.clone() });
            }
            TurnEvent::MessageAppended(m) => {
                let _ = tx.send(AgentEvent::MessageAppended { message: m.clone() });
            }
            TurnEvent::ToolApprovalRequested { requests } => {
                let _ = tx.send(AgentEvent::ToolApprovalRequested { requests });
            }
            TurnEvent::ToolApprovalResolved {
                request_id,
                approved,
            } => {
                let _ = tx.send(AgentEvent::ToolApprovalResolved {
                    request_id,
                    approved,
                });
            }
        };
        if turn.initialized {
            runner
                .resume_with_model_and_approver(
                    turn.session_id,
                    turn.model,
                    turn.input,
                    turn.cancel,
                    turn.approvals.as_ref(),
                    callback,
                )
                .await
        } else {
            runner
                .start_with_model_and_approver(
                    turn.session_id,
                    workspace,
                    turn.model,
                    turn.input,
                    turn.cancel,
                    turn.approvals.as_ref(),
                    callback,
                )
                .await
        }
        // `tx` (captured inside the callback) is dropped when this future
        // finishes, closing the channel so the drain future exits.
    };

    // Drain future: forwards each event to the writer in arrival order.
    // Write errors abort draining immediately — if the GUI hung up, there
    // is no point trying to write further frames.
    let drain_fut = async {
        while let Some(evt) = rx.recv().await {
            write_frame(writer, &evt).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let (run_outcome, drain_outcome) = tokio::join!(run_fut, drain_fut);
    // Surface a write error before the runner's error so the caller sees
    // the earliest failure. Either one aborts the turn.
    drain_outcome?;
    run_outcome
}
