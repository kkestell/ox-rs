use std::path::PathBuf;

use anyhow::{Context, Result};
use domain::{Session, SessionId, SessionSummary};

pub struct DiskSessionStore {
    dir: PathBuf,
}

impl DiskSessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Build the on-disk path for a given session: `{dir}/{id}.json`.
    fn session_path(&self, id: SessionId) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}

impl app::SessionStore for DiskSessionStore {
    async fn load(&self, id: SessionId) -> Result<Session> {
        let path = self.session_path(id);
        let data = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read session file {}", path.display()))?;
        let session: Session = serde_json::from_str(&data)
            .with_context(|| format!("failed to deserialize session {id}"))?;
        Ok(session)
    }

    async fn save(&self, session: &Session) -> Result<()> {
        let path = self.session_path(session.id);
        let data = serde_json::to_string_pretty(session)
            .with_context(|| format!("failed to serialize session {}", session.id))?;
        tokio::fs::write(&path, data)
            .await
            .with_context(|| format!("failed to write session file {}", path.display()))?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();

        let mut entries = tokio::fs::read_dir(&self.dir)
            .await
            .with_context(|| format!("failed to read session directory {}", self.dir.display()))?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only consider .json files whose stem is a valid UUID.
            // No file reads — the filename alone is authoritative.
            let is_json = path.extension().is_some_and(|ext| ext == "json");
            if !is_json {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(id) = stem.parse::<SessionId>() else {
                continue;
            };

            summaries.push(SessionSummary { id });
        }

        Ok(summaries)
    }
}

#[cfg(test)]
mod tests {
    use app::SessionStore;
    use domain::{ContentBlock, Message, Role, Session, SessionId};

    use super::*;

    /// Helper: create a store backed by a temp directory. Returns the store and
    /// the TempDir guard (dropping it cleans up the directory).
    fn temp_store() -> (DiskSessionStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = DiskSessionStore::new(tmp.path()).unwrap();
        (store, tmp)
    }

    /// Build a session containing one message of each role (user, assistant,
    /// tool) to exercise all ContentBlock variants through serialization.
    fn multi_role_session(id: SessionId) -> Session {
        let mut session = Session::new(id, "/tmp/project".into());
        session.push_message(Message::user("hello"));
        session.push_message(Message::assistant(vec![
            ContentBlock::Text {
                text: "I'll read that file.".into(),
            },
            ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"src/main.rs"}"#.into(),
            },
        ]));
        session.push_message(Message::tool_result("call_1", "fn main() {}", false));
        session
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let (store, _tmp) = temp_store();
        let id = SessionId::new_v4();
        let session = multi_role_session(id);

        store.save(&session).await.unwrap();
        let loaded = store.load(id).await.unwrap();

        assert_eq!(loaded.id, id);
        assert_eq!(loaded.workspace_root, session.workspace_root);
        assert_eq!(loaded.messages.len(), 3);

        // Verify each message role survived the round-trip.
        assert_eq!(loaded.messages[0].role, Role::User);
        assert_eq!(loaded.messages[0].text(), "hello");
        assert_eq!(loaded.messages[1].role, Role::Assistant);
        assert_eq!(loaded.messages[1].tool_calls().len(), 1);
        assert_eq!(loaded.messages[2].role, Role::Tool);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_error() {
        let (store, _tmp) = temp_store();
        let result = store.load(SessionId::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_empty_directory() {
        let (store, _tmp) = temp_store();
        let summaries = store.list().await.unwrap();
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn list_returns_all_saved_sessions() {
        let (store, _tmp) = temp_store();

        let id1 = SessionId::new_v4();
        let s1 = Session::new(id1, "/a".into());
        store.save(&s1).await.unwrap();

        let id2 = SessionId::new_v4();
        let s2 = Session::new(id2, "/b".into());
        store.save(&s2).await.unwrap();

        let id3 = SessionId::new_v4();
        let s3 = Session::new(id3, "/c".into());
        store.save(&s3).await.unwrap();

        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 3);

        let mut ids: Vec<_> = summaries.iter().map(|s| s.id).collect();
        ids.sort_by_key(|id| id.0);
        let mut expected = vec![id1, id2, id3];
        expected.sort_by_key(|id| id.0);
        assert_eq!(ids, expected);
    }

    #[tokio::test]
    async fn save_overwrites_existing() {
        let (store, _tmp) = temp_store();
        let id = SessionId::new_v4();

        let mut session = Session::new(id, "/tmp".into());
        session.push_message(Message::user("first"));
        store.save(&session).await.unwrap();

        // Mutate and save again — should overwrite, not duplicate.
        session.push_message(Message::user("second"));
        store.save(&session).await.unwrap();

        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[1].text(), "second");
    }

    #[tokio::test]
    async fn list_ignores_non_json_files() {
        let (store, tmp) = temp_store();

        // Save a real session so the directory isn't empty.
        let id = SessionId::new_v4();
        let session = Session::new(id, "/tmp".into());
        store.save(&session).await.unwrap();

        // Drop non-json files and a json file with a non-UUID name into the
        // directory — list should skip all of them.
        std::fs::write(tmp.path().join("notes.txt"), "not a session").unwrap();
        std::fs::write(tmp.path().join("backup.json.bak"), "{}").unwrap();
        std::fs::write(tmp.path().join("not-a-uuid.json"), "{}").unwrap();

        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, id);
    }

    /// A .json file with a valid UUID stem but malformed body is still returned
    /// by `list()` — because `list()` never reads the file contents.
    #[tokio::test]
    async fn list_succeeds_for_session_file_with_malformed_body() {
        let (store, tmp) = temp_store();

        // Write a file with a valid UUID stem but garbage JSON body.
        let id = SessionId::new_v4();
        std::fs::write(
            tmp.path().join(format!("{id}.json")),
            "this is not valid json",
        )
        .unwrap();

        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, id);

        // load() on the same ID should fail with a deserialization error.
        let err = store.load(id).await.unwrap_err();
        assert!(
            err.to_string().contains("deserialize"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn empty_session_round_trips() {
        let (store, _tmp) = temp_store();
        let id = SessionId::new_v4();
        let session = Session::new(id, "/empty".into());

        store.save(&session).await.unwrap();
        let loaded = store.load(id).await.unwrap();

        assert_eq!(loaded.id, id);
        assert!(loaded.messages.is_empty());
    }
}
