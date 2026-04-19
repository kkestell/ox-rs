use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// Cooperative cancellation signal for an in-progress turn.
///
/// Backed by an `AtomicBool` so synchronous callers (the `SessionRunner`
/// loop, the driver's `Cancel` handler) can poll the flag cheaply, and a
/// [`Notify`] so async callers can suspend on [`CancelToken::cancelled`]
/// instead of polling. `Clone` is cheap — it's an `Arc`.
///
/// Defined in the `app` crate rather than `protocol` so the application
/// layer stays independent of the async runtime's wire format.
#[derive(Clone)]
pub struct CancelToken {
    inner: Arc<Inner>,
}

struct Inner {
    flag: AtomicBool,
    notify: Notify,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                flag: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    /// Signal cancellation. Idempotent — calling this multiple times is
    /// harmless. Wakes every task currently parked in [`cancelled`].
    ///
    /// [`cancelled`]: Self::cancelled
    pub fn cancel(&self) {
        self.inner.flag.store(true, Ordering::Release);
        self.inner.notify.notify_waiters();
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.flag.load(Ordering::Acquire)
    }

    /// Resolves once cancellation has been requested. Uses the standard
    /// register-then-recheck idiom so a `cancel()` that races with this
    /// call still wakes the waiter.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.inner.notify.notified();
        tokio::pin!(notified);
        // `enable` registers the future as a waiter. Subsequent
        // `notify_waiters` calls will deliver to it. Re-check the flag
        // afterwards to cover a `cancel()` that ran between the first
        // check and registration.
        notified.as_mut().enable();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_resolves_when_cancelled_before_call() {
        let token = CancelToken::new();
        token.cancel();
        // Already cancelled path: must not block.
        tokio::time::timeout(std::time::Duration::from_millis(50), token.cancelled())
            .await
            .expect("already-cancelled token must resolve immediately");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_wakes_when_cancel_called_concurrently() {
        // Spin the race many times on a multi-threaded runtime to exercise
        // the window between `is_cancelled()` and `notify.notified().enable()`
        // inside `cancelled()`. If the register-then-recheck idiom regressed,
        // at least one iteration in this loop is likely to wedge.
        for _ in 0..256 {
            let token = CancelToken::new();
            let waiter = {
                let token = token.clone();
                tokio::spawn(async move { token.cancelled().await })
            };
            token.cancel();
            tokio::time::timeout(std::time::Duration::from_millis(100), waiter)
                .await
                .expect("waiter must wake within 100ms of cancel")
                .expect("waiter task panicked");
        }
    }

    #[tokio::test]
    async fn cancelled_wakes_multiple_waiters() {
        let token = CancelToken::new();
        let a = {
            let token = token.clone();
            tokio::spawn(async move { token.cancelled().await })
        };
        let b = {
            let token = token.clone();
            tokio::spawn(async move { token.cancelled().await })
        };
        tokio::task::yield_now().await;
        token.cancel();
        for waiter in [a, b] {
            tokio::time::timeout(std::time::Duration::from_millis(50), waiter)
                .await
                .expect("each waiter must wake")
                .expect("waiter task panicked");
        }
    }
}
