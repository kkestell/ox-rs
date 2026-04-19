use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use agent_host::{AgentEventStream, CloseRequestSink, FirstTurnSink, SessionRuntime, apply_event};
use domain::{ContentBlock, Role};
use protocol::AgentEvent;
use tokio::sync::broadcast;

use super::ActiveSession;

/// Drive events from the agent's stream into history, the broadcast
/// channel, and the runtime state machine. Runs until the stream
/// closes or until the owning session is dropped and aborts us.
///
/// Nine handles is a lot, but they all thread together as one unit and
/// grouping them under a struct just renames the problem — the struct
/// would have no other use and every callsite would populate every
/// field. Keep them as named parameters for readability.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_pump(
    mut stream: AgentEventStream,
    history: Arc<Mutex<Vec<AgentEvent>>>,
    tx: broadcast::Sender<AgentEvent>,
    runtime: Arc<Mutex<SessionRuntime>>,
    alive: Arc<AtomicBool>,
    session_weak: std::sync::Weak<ActiveSession>,
    ready_notify: Arc<tokio::sync::Notify>,
    first_turn_sink: Arc<dyn FirstTurnSink>,
    close_sink: Arc<dyn CloseRequestSink>,
    suppress_startup_replay: bool,
) -> tokio::task::AbortHandle {
    let join = tokio::spawn(async move {
        let mut filtering_startup_replay = suppress_startup_replay;
        let mut replay_message_index = 0usize;
        loop {
            let evt = match stream.recv().await {
                Some(e) => e,
                None => break,
            };

            // Record the `Ready` id on the owning session. `set` is a
            // no-op after the first call, so repeated `Ready` frames
            // (the plan allows it) don't clobber the cell.
            if let AgentEvent::Ready { session_id, .. } = &evt
                && let Some(session) = session_weak.upgrade()
            {
                let _ = session.session_id.set(*session_id);
                ready_notify.notify_waiters();
            }

            // Replacement agents are launched with `--resume`, so their
            // startup sequence replays persisted history. This session has
            // already published those frames; suppress the matching prefix so
            // live subscribers don't see duplicate transcript messages.
            if filtering_startup_replay {
                match &evt {
                    AgentEvent::Ready { .. } => continue,
                    AgentEvent::MessageAppended { message } => {
                        let is_replayed = {
                            let rt = runtime.lock().expect("session runtime mutex poisoned");
                            rt.messages
                                .get(replay_message_index)
                                .is_some_and(|known| messages_match(known, message))
                        };
                        if is_replayed {
                            replay_message_index += 1;
                            continue;
                        }
                        filtering_startup_replay = false;
                    }
                    _ => {
                        filtering_startup_replay = false;
                    }
                }
            }

            // `RequestClose` is a control frame from the agent asking
            // the host to merge or abandon this session. Route it to
            // the close sink and skip the normal publish / state-
            // machine / first-turn path — it isn't user-visible and
            // `apply_event` has nothing meaningful to do with it. The
            // sink call is fire-and-forget: the pump keeps draining
            // until the agent's stdout closes, which it will shortly
            // since the agent exits its command loop after emitting
            // this frame.
            if let AgentEvent::RequestClose { intent } = evt {
                let Some(session) = session_weak.upgrade() else {
                    // Session was dropped between the recv and now —
                    // there is no one to close, so we discard. The
                    // agent is already on its way out.
                    continue;
                };
                let Some(id) = session.session_id.get().copied() else {
                    // Defensive: `RequestClose` should only arrive
                    // after the agent's `Ready`, which populates
                    // `session_id`. Without an id the sink has no
                    // session to dispatch against, so we skip rather
                    // than invent one.
                    drop(session);
                    continue;
                };
                drop(session);
                let sink = close_sink.clone();
                tokio::spawn(async move {
                    sink.request_close(id, intent).await;
                });
                continue;
            }

            // Cheap flag computed before `evt` is consumed by
            // `apply_event` below, so the slug-rename hook can fire
            // without re-cloning the event.
            let is_turn_complete = matches!(evt, AgentEvent::TurnComplete);

            // Publish step: append to history and fan out on the
            // broadcast channel as one atomic unit. `subscribe()`
            // acquires the same lock, so the snapshot→follow handshake
            // never drops or duplicates events.
            {
                let mut hist = history.lock().expect("session history mutex poisoned");
                hist.push(evt.clone());
                // `send` returns Err(SendError) when there are no
                // subscribers. That's fine — history still holds the
                // event for future subscribers to replay.
                let _ = tx.send(evt.clone());
            }

            // State-machine step: apply the event to `runtime` under
            // its own lock. Held briefly, never across an await.
            {
                let mut rt = runtime.lock().expect("session runtime mutex poisoned");
                apply_event(&mut rt, evt);
            }

            // Slug-rename hook: on the first `TurnComplete` observed
            // while the session is fresh, snapshot history, extract the
            // first user message, and fire the sink. The CAS-flip of
            // `fresh` ensures the hook fires at most once per session
            // — even if a crash-restart somehow replays the first turn
            // after resume, `fresh` starts as `false` on the
            // replacement agent's pump (the lifecycle coordinator clears
            // it before calling `replace_agent`).
            if is_turn_complete
                && let Some(session) = session_weak.upgrade()
                && session
                    .fresh
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                let Some(id) = session.session_id.get().copied() else {
                    // Defensive: `TurnComplete` should never precede
                    // `Ready`. If it does, skip the hook — there's
                    // nothing the coordinator can correlate without
                    // an id.
                    continue;
                };
                let snapshot = {
                    let hist = history.lock().expect("session history mutex poisoned");
                    hist.clone()
                };
                // Drop the Arc<ActiveSession> before spawning the
                // fire-and-forget so the task doesn't hold the session
                // alive past its own completion.
                drop(session);
                let Some(first_message) = extract_first_user_message(&snapshot) else {
                    // No user message in history is a logic bug — a
                    // TurnComplete without a preceding user frame
                    // shouldn't be possible. Skip rather than invent a
                    // slug from nothing.
                    continue;
                };
                let sink = first_turn_sink.clone();
                // Fire-and-forget: any slug-rename work (LLM call, git
                // operations) runs on its own task so the pump keeps
                // draining frames.
                tokio::spawn(async move {
                    sink.on_first_turn_complete(id, first_message).await;
                });
            }
        }

        alive.store(false, Ordering::SeqCst);
        // Wake anyone waiting in `await_ready` so they observe the
        // closed stream and return `None` instead of hanging.
        ready_notify.notify_waiters();
        // Dropping `tx` here drops the last owned sender; existing
        // subscribers observe a closed channel on their next `recv`,
        // which SSE interprets as "session ended."
        drop(tx);
    });
    join.abort_handle()
}

fn messages_match(left: &domain::Message, right: &domain::Message) -> bool {
    left.role == right.role
        && left.content == right.content
        && left.token_count == right.token_count
}

/// Walk `history` front-to-back and return the concatenated text of
/// the first `MessageAppended` whose role is `User`. Non-text content
/// blocks are skipped; text blocks are joined with newlines so a
/// multi-block user message reads naturally in the LLM prompt used to
/// derive the slug. Returns `None` if no user message is present.
fn extract_first_user_message(history: &[AgentEvent]) -> Option<String> {
    for evt in history {
        if let AgentEvent::MessageAppended { message } = evt
            && matches!(message.role, Role::User)
        {
            let text = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return Some(text);
        }
    }
    None
}
