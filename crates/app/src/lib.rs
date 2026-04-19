pub mod approval;
pub mod cancel;
pub mod lifecycle;
mod ports;
pub mod tools;
mod use_cases;

#[cfg(any(test, feature = "test-support"))]
pub mod fake;

pub use approval::{
    ApprovalRequirement, MissingPathPolicy, NoApprovalRequired, TOOL_REJECTED_MESSAGE,
    ToolApprovalDecision, ToolApprovalRequest, ToolApprover,
};
pub use cancel::CancelToken;
pub use lifecycle::{AbandonTool, CloseSignal, MergeTool};
pub use ports::*;
pub use tools::{
    EditFileTool, GlobTool, GrepTool, ReadFileTool, TodoWriteTool, Tool, ToolRegistry,
    WriteFileTool,
};
pub use use_cases::{SessionRunner, TurnEvent, TurnOutcome};

// `StreamEvent`, `Usage`, `StreamAccumulator`, `Snapshot`, and `ToolDef` live
// in `domain` — they're serializable data shapes, not application behavior.
// Re-exported here so existing `app::*` callers don't have to chase the move.
pub use domain::{Snapshot, StreamAccumulator, StreamEvent, ToolDef, Usage};
