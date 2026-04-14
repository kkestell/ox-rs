use std::path::PathBuf;

use anyhow::Result;
use domain::{Session, SessionId, SessionSummary};

pub struct DiskSessionStore {
    _dir: PathBuf,
}

impl DiskSessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { _dir: dir })
    }
}

impl app::SessionStore for DiskSessionStore {
    async fn load(&self, _id: SessionId) -> Result<Session> {
        // TODO: read session JSON from disk
        todo!()
    }

    async fn save(&self, _session: &Session) -> Result<()> {
        // TODO: write session JSON to disk
        todo!()
    }

    async fn list(&self) -> Result<Vec<SessionSummary>> {
        // TODO: list session files
        todo!()
    }
}
