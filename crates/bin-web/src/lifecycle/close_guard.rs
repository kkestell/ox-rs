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
        let mut map = self.closing.lock().unwrap_or_else(|err| err.into_inner());
        map.remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use domain::SessionId;

    use super::*;

    #[test]
    fn drop_does_not_panic_when_close_map_is_poisoned() {
        let id = SessionId::new_v4();
        let closing = Mutex::new(HashMap::from([(id, CloseState::Closing)]));
        let poison_result = std::panic::catch_unwind(|| {
            let _guard = closing.lock().unwrap();
            panic!("poison close map");
        });
        assert!(poison_result.is_err());

        let drop_result = std::panic::catch_unwind(|| {
            drop(CloseGuard {
                closing: &closing,
                id,
                session: None,
                clear_session_on_drop: true,
            });
        });
        assert!(drop_result.is_ok());
        assert!(closing.lock().unwrap_err().into_inner().is_empty());
    }
}
