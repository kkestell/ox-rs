use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// Generate a new random session ID (v4 UUID).
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for SessionId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(SessionId)
    }
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: SessionId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    /// The main repository root this session belongs to — i.e. the CWD the
    /// user launched `ox` from. Stays stable across slug renames so the
    /// session can be looked up by the workspace that birthed it.
    pub workspace_root: PathBuf,
    /// The dedicated worktree checkout where this session does its work.
    /// Every tool invocation, shell command, and file edit is scoped to
    /// this path — never to `workspace_root`. On slug rename this path
    /// is updated to the new `ox/<slug>-<short-uuid>` directory.
    pub worktree_path: PathBuf,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(id: SessionId, workspace_root: PathBuf, worktree_path: PathBuf) -> Self {
        Self {
            id,
            workspace_root,
            worktree_path,
            messages: Vec::new(),
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentBlock, Role, stream::Usage};

    #[test]
    fn session_id_display_roundtrip() {
        let id = SessionId::new_v4();
        let s = id.to_string();
        let parsed: SessionId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn session_id_from_str_invalid() {
        assert!("not-a-uuid".parse::<SessionId>().is_err());
    }

    #[test]
    fn session_new_has_workspace_root() {
        let id = SessionId::new_v4();
        let root = PathBuf::from("/tmp/project");
        let worktree = PathBuf::from("/tmp/project/.ox/worktrees/ox/abcdef");
        let session = Session::new(id, root.clone(), worktree.clone());
        assert_eq!(session.id, id);
        assert_eq!(session.workspace_root, root);
        assert_eq!(session.worktree_path, worktree);
        assert!(session.messages.is_empty());
    }

    #[test]
    fn session_roundtrips_messages_with_and_without_usage() {
        // Session-level JSON must preserve each message's `Option<Usage>`
        // so a post-restart snapshot populates the frontend's usage chip
        // from the last assistant turn. Mix a user message (no usage),
        // an assistant message with usage, and an assistant message
        // without usage — all three patterns appear on disk.
        let id = SessionId::new_v4();
        let root = PathBuf::from("/tmp/ws");
        let worktree = PathBuf::from("/tmp/ws/.ox/wt");
        let mut session = Session::new(id, root, worktree);
        session.push_message(Message::user("hi"));
        session.push_message(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                reasoning_tokens: 0,
            }),
        });
        session.push_message(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "no usage reported".into(),
            }],
            usage: None,
        });

        let json = serde_json::to_string(&session).unwrap();
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 3);
        assert_eq!(back.messages[0].usage, None);
        assert_eq!(
            back.messages[1].usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                reasoning_tokens: 0,
            })
        );
        assert_eq!(back.messages[2].usage, None);
    }
}
