//! Conversation-turn orchestration.
//!
//! - [`turn_event`] — the `TurnEvent` / `TurnOutcome` types the caller
//!   observes through the event callback.
//! - [`tool_loop`] — the per-call helpers that execute a tool and
//!   commit its result (plus the `PlannedToolCall` classification the
//!   runner uses to separate ready / approval-gated / policy-rejected
//!   calls).
//! - [`runner`] — the `SessionRunner` type itself, including the
//!   streaming + tool-call loop in `run_turn_with_approver`.

mod runner;
mod tool_loop;
mod turn_event;

pub use runner::SessionRunner;
pub use turn_event::{TurnEvent, TurnOutcome};
