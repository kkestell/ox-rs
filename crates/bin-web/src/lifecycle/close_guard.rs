use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use domain::SessionId;

use super::CloseState;
use crate::session::ActiveSession;

/// RAII release of the close lock entry. Removes the session's
/// `CloseState::Closing` record when dropped so the next
/// merge/abandon call on the same id isn't rejected.
///
/// The guard removes the entry on *any* drop — success or failure.
/// That's correct: on success the session is also removed from the
/// registry so the map entry has nothing to guard; on failure we want
/// the lock released so the caller can retry.
pub(super) struct CloseGuard<'a> {
    pub(super) closing: &'a Mutex<HashMap<SessionId, CloseState>>,
    pub(super) id: SessionId,
    pub(super) session: Option<Arc<ActiveSession>>,
    pub(super) clear_session_on_drop: bool,
}

impl CloseGuard<'_> {
    pub(super) fn protect_session(&mut self, session: Arc<ActiveSession>) {
        self.session = Some(session);
    }

    pub(super) fn keep_session_closing(&mut self) {
        self.clear_session_on_drop = false;
    }
}

impl Drop for CloseGuard<'_> {
    fn drop(&mut self) {
        if self.clear_session_on_drop
            && let Some(session) = &self.session
        {
            session.clear_closing();
        }
        let mut map = self.closing.lock().expect("close map poisoned");
        map.remove(&self.id);
    }
}
