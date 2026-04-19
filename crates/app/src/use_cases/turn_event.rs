use domain::{Message, StreamEvent};

use crate::approval::ToolApprovalRequest;

/// Result of a completed turn — either it ran to natural completion or was
/// cancelled cooperatively via a `CancelToken`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Completed,
    Cancelled,
}

/// Events surfaced to the caller during a turn.
///
/// `StreamDelta` forwards raw in-flight stream events (token/reasoning/tool-
/// call argument chunks). `MessageAppended` fires for every message that is
/// committed to the session — the initial user input, each intermediate
/// assistant reply and tool-result in a multi-step tool loop, and the final
/// assistant reply.
///
/// The callback sees a `MessageAppended` *after* the message has been pushed
/// to `session.messages`, so UI layers can treat it as "this is now part of
/// history." That keeps the live-streaming view (driven by `StreamDelta`s
/// into a `StreamAccumulator`) and the committed view strictly separated:
/// the accumulator renders in-progress state, and once the message is
/// committed via `MessageAppended`, the accumulator is discarded.
#[derive(Debug)]
pub enum TurnEvent<'a> {
    StreamDelta(&'a StreamEvent),
    MessageAppended(&'a Message),
    ToolApprovalRequested { requests: Vec<ToolApprovalRequest> },
    ToolApprovalResolved { request_id: String, approved: bool },
}
