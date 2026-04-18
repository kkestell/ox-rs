//! How a session should be closed.
//!
//! `CloseIntent` travels three ways: the agent-side `MergeTool` /
//! `AbandonTool` set it on a shared `CloseSignal`; the driver drains the
//! signal after each terminal frame and ships it on an
//! `AgentEvent::RequestClose { intent }`; the host's lifecycle coordinator
//! receives it via `CloseRequestSink` and dispatches to `merge` /
//! `abandon`. All three live in different crates that already depend on
//! `domain`, which is why the type lives here.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CloseIntent {
    /// Merge the session's branch into the workspace's base branch and
    /// drop the session. Rejected if the worktree or main checkout is
    /// dirty, or if the merge conflicts.
    Merge,
    /// Discard the session's branch and worktree without merging. If the
    /// worktree is dirty, the caller must pass `confirm: true` or the
    /// operation is rejected.
    Abandon { confirm: bool },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_round_trips_through_serde() {
        let value = CloseIntent::Merge;
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(json, r#"{"kind":"merge"}"#);
        let back: CloseIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn abandon_confirm_true_round_trips() {
        let value = CloseIntent::Abandon { confirm: true };
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(json, r#"{"kind":"abandon","confirm":true}"#);
        let back: CloseIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn abandon_confirm_false_round_trips() {
        let value = CloseIntent::Abandon { confirm: false };
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(json, r#"{"kind":"abandon","confirm":false}"#);
        let back: CloseIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn unknown_kind_tag_fails_to_deserialize() {
        let err = serde_json::from_str::<CloseIntent>(r#"{"kind":"explode"}"#).unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }
}
