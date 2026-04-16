use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cooperative cancellation signal for an in-progress turn.
///
/// Backed by an `AtomicBool` so it can be shared between the driver (which
/// sets it on `Cancel` command) and the `SessionRunner` (which checks it
/// between stream events and tool calls). `Clone` is cheap — it's an `Arc`.
///
/// Defined in the `app` crate rather than `protocol` so the application
/// layer stays independent of the async runtime and the wire format.
#[derive(Clone)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation. Idempotent — calling this multiple times is
    /// harmless.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}
