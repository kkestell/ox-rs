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
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub workspace_root: PathBuf,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(id: SessionId, workspace_root: PathBuf) -> Self {
        Self {
            id,
            workspace_root,
            messages: Vec::new(),
        }
    }

    /// Returns the total token count across all messages in this session.
    pub fn is_over_budget(&self, limit: usize) -> bool {
        self.messages.iter().map(|m| m.token_count).sum::<usize>() > limit
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let session = Session::new(id, root.clone());
        assert_eq!(session.id, id);
        assert_eq!(session.workspace_root, root);
        assert!(session.messages.is_empty());
    }
}
