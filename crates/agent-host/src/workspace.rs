//! Multi-split workspace orchestration.
//!
//! `WorkspaceState` owns every running [`AgentSplit`] in a single window:
//! their horizontal split fractions, the focused split, and the layout
//! persistence. It's the only surface the desktop binary touches — every
//! Tauri command handler and every drain task goes through methods on this
//! type rather than reaching into an `AgentSplit` directly.
//!
//! The split/stream pairing discipline is critical: methods that spawn a
//! fresh agent (`add_split`, `replace_workspace`, `restore`) hand the
//! matching [`AgentEventStream`] back to the caller along with the split's
//! [`SplitId`]. The caller wires the stream into a drain task that calls
//! [`WorkspaceState::apply_event`] for each event. Receivers never sit
//! behind the state mutex, so a slow `recv().await` can't block commands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use domain::{ContentBlock, Message, SessionId};
use protocol::{AgentCommand, AgentEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::client::{AgentClient, AgentEventStream, AgentSpawnConfig, SplitId};
use crate::layout::{RestoreLayout, WorkspaceLayouts, normalize_split_fracs};
use crate::split::AgentSplit;

/// Fresh-spawn shorthand: the id and the event stream for a newly-created
/// split. Returned everywhere a new subprocess is spawned so the caller can
/// wire a drain task.
pub type SpawnedSplit = (SplitId, AgentEventStream);

/// Outcome of [`WorkspaceState::close_split`]. The desktop binary uses
/// `last_split_closed` to branch into the app-quit path (quit-confirmation
/// if busy, direct close if idle) instead of leaving the window empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseOutcome {
    pub last_split_closed: bool,
}

/// Returned when a method is given a [`SplitId`] that is not present in the
/// workspace. Happens when a command race-conditions with a close — the
/// split already went away before the command arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownSplit(pub SplitId);

impl std::fmt::Display for UnknownSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no split with id {}", self.0)
    }
}

impl std::error::Error for UnknownSplit {}

/// Error from [`WorkspaceState::send`]. Either the split is unknown or the
/// client's writer task has already shut down (agent dead). The latter is
/// `Disconnected` rather than crashing so a racing `Cancel` after the
/// subprocess died is a harmless no-op from the caller's perspective.
#[derive(Debug)]
pub enum SendError {
    UnknownSplit(SplitId),
    Disconnected(mpsc::error::SendError<AgentCommand>),
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownSplit(id) => write!(f, "no split with id {id}"),
            Self::Disconnected(e) => write!(f, "agent writer disconnected: {e}"),
        }
    }
}

impl std::error::Error for SendError {}

/// Snapshot view of the whole workspace — everything the frontend needs to
/// rebuild its DOM from authoritative Rust state. Serializable so it can
/// cross the Tauri IPC boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub splits: Vec<SplitSnapshot>,
    pub split_fracs: Vec<f32>,
    pub focused: Option<SplitId>,
    pub workspace_root: PathBuf,
}

/// Per-split snapshot. The streaming content is flattened into a plain
/// `Vec<ContentBlock>` so the TS side doesn't need access to
/// `StreamAccumulator`'s internal shape — the accumulator is a Rust-only
/// rendering intermediate, not a wire type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitSnapshot {
    pub id: SplitId,
    pub session_id: Option<SessionId>,
    pub messages: Vec<Message>,
    pub streaming: Option<Vec<ContentBlock>>,
    pub waiting: bool,
    pub error: Option<String>,
    pub cancelled: bool,
}

pub struct WorkspaceState {
    splits: Vec<AgentSplit>,
    focused: usize,
    split_fracs: Vec<f32>,
    spawn_config: AgentSpawnConfig,
    layout_state_path: Option<PathBuf>,
}

impl WorkspaceState {
    /// Build a workspace from a vector of already-constructed splits. The
    /// streams for those splits have already been handed to drain tasks by
    /// the caller — this constructor takes only the sending halves.
    ///
    /// Used by tests and by the `restore` fast-path once spawned clients
    /// have been peeled apart.
    pub fn new(splits: Vec<AgentSplit>, spawn_config: AgentSpawnConfig) -> Self {
        Self::with_layout(splits, spawn_config, None, None, 0)
    }

    pub fn with_layout(
        splits: Vec<AgentSplit>,
        spawn_config: AgentSpawnConfig,
        layout_state_path: Option<PathBuf>,
        split_fracs: Option<Vec<f32>>,
        focused: usize,
    ) -> Self {
        assert!(
            !splits.is_empty(),
            "WorkspaceState requires at least one split"
        );
        let n = splits.len();
        let split_fracs = split_fracs
            .map(|fracs| normalize_split_fracs(&fracs, n))
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        let focused = focused.min(n - 1);
        Self {
            splits,
            focused,
            split_fracs,
            spawn_config,
            layout_state_path,
        }
    }

    /// Spawn the initial workspace at startup. Returns the workspace state
    /// plus an `AgentEventStream` per spawned split so the caller can start
    /// a drain task per split.
    ///
    /// Mirrors the fallback flow from the old `OxApp::restore`: if the
    /// caller provided an explicit `--resume`, that wins; otherwise we try
    /// the saved layout for this workspace root, filtering to sessions
    /// whose on-disk JSON still exists, and fall back to one fresh split if
    /// the saved layout points only at missing sessions.
    pub fn restore(
        mut spawn_config: AgentSpawnConfig,
        layout_state_path: PathBuf,
    ) -> Result<(Self, Vec<SpawnedSplit>)> {
        let layout = if spawn_config.resume.is_some() {
            None
        } else {
            load_workspace_layout(&layout_state_path)
                .restore_existing_for(&spawn_config.workspace_root, &spawn_config.sessions_dir)
        };
        let explicit_resume = spawn_config.resume;
        let restore = layout.as_ref();
        let spawn_configs = startup_spawn_configs(&spawn_config, restore);

        let mut splits = Vec::with_capacity(spawn_configs.len());
        let mut streams = Vec::with_capacity(spawn_configs.len());
        let mut restored_spawn_failed = false;
        for config in &spawn_configs {
            match AgentClient::spawn(config.clone()) {
                Ok((client, stream)) => {
                    splits.push(AgentSplit::new(client));
                    streams.push(stream);
                }
                Err(_err) if restore.is_some() && explicit_resume.is_none() => {
                    // A saved layout pointed at a session the agent can't
                    // actually resume (corrupt file, unavailable model,
                    // etc.). Fall through to a fresh-spawn retry below
                    // rather than propagating the error up — otherwise the
                    // first crash permanently wedges the workspace.
                    restored_spawn_failed = true;
                    break;
                }
                Err(err) => return Err(err).context("spawning initial agent"),
            }
        }

        let (split_fracs, focused) = if restored_spawn_failed {
            let mut fresh = spawn_config.clone();
            fresh.resume = None;
            let (client, stream) = AgentClient::spawn(fresh)
                .context("spawning fresh agent after restored workspace layout failed")?;
            splits = vec![AgentSplit::new(client)];
            streams = vec![stream];
            (None, 0)
        } else {
            (
                restore.map(|layout| layout.split_fracs.clone()),
                restore.map(|layout| layout.focused).unwrap_or(0),
            )
        };

        spawn_config.resume = None;
        let state = Self::with_layout(
            splits,
            spawn_config,
            Some(layout_state_path),
            split_fracs,
            focused,
        );
        let split_ids: Vec<SplitId> = state.splits.iter().map(AgentSplit::id).collect();
        Ok((state, split_ids.into_iter().zip(streams).collect()))
    }

    /// Number of active splits. Primarily for tests.
    pub fn split_count(&self) -> usize {
        self.splits.len()
    }

    /// Current split fractions. Borrowed view — callers that want to keep
    /// the slice around take `.to_vec()`.
    pub fn split_fracs(&self) -> &[f32] {
        &self.split_fracs
    }

    /// The active workspace root. Used by the desktop binary when deciding
    /// whether a File > Open path would replace the workspace or not.
    pub fn workspace_root(&self) -> &Path {
        &self.spawn_config.workspace_root
    }

    /// Any split has `waiting == true` or a live streaming accumulator. The
    /// quit-confirmation flow keys off this.
    pub fn any_turn_in_progress(&self) -> bool {
        self.splits.iter().any(AgentSplit::is_turn_in_progress)
    }

    /// Identity of the focused split, if any.
    pub fn focused(&self) -> Option<SplitId> {
        self.splits.get(self.focused).map(AgentSplit::id)
    }

    /// Set the focused split by id.
    pub fn set_focused(&mut self, id: SplitId) -> std::result::Result<(), UnknownSplit> {
        let idx = self.index_of(id).ok_or(UnknownSplit(id))?;
        self.focused = idx;
        Ok(())
    }

    /// Serializable view of the whole workspace.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            splits: self.splits.iter().map(split_snapshot).collect(),
            split_fracs: self.split_fracs.clone(),
            focused: self.focused(),
            workspace_root: self.spawn_config.workspace_root.clone(),
        }
    }

    /// Route one event to the split it belongs to. Drain tasks call this
    /// per received event.
    pub fn apply_event(
        &mut self,
        id: SplitId,
        event: AgentEvent,
    ) -> std::result::Result<(), UnknownSplit> {
        let idx = self.index_of(id).ok_or(UnknownSplit(id))?;
        self.splits[idx].handle_event(event);
        Ok(())
    }

    /// Forward a command to a split's agent.
    ///
    /// For `SendMessage`, this checks the split's `waiting` flag: if the
    /// split is already waiting on a reply, the command is silently
    /// dropped (same behavior as the old egui `send_message`). This
    /// prevents a rapid double-Enter from queuing a second send before the
    /// first turn finishes. Every other command (e.g. `Cancel`) is
    /// forwarded unconditionally so cancel-while-waiting always works.
    pub fn send(&mut self, id: SplitId, cmd: AgentCommand) -> std::result::Result<(), SendError> {
        let idx = self.index_of(id).ok_or(SendError::UnknownSplit(id))?;
        let split = &mut self.splits[idx];

        if let AgentCommand::SendMessage { .. } = cmd {
            if split.is_waiting() {
                return Ok(());
            }
            split.send(cmd).map_err(SendError::Disconnected)?;
            split.waiting = true;
            split.error = None;
            split.cancelled = false;
            return Ok(());
        }

        split.send(cmd).map_err(SendError::Disconnected)
    }

    /// Spawn a fresh agent and append it to the right of the current
    /// splits. Focus moves to the new split; fractions redistribute equally.
    pub fn add_split(&mut self) -> Result<SpawnedSplit> {
        let mut config = self.spawn_config.clone();
        config.resume = None;
        let (client, stream) = AgentClient::spawn(config).context("spawning new agent")?;
        let split = AgentSplit::new(client);
        let id = split.id();
        self.splits.push(split);
        self.focused = self.splits.len() - 1;
        let n = self.splits.len();
        self.split_fracs = vec![1.0 / n as f32; n];
        Ok((id, stream))
    }

    /// Close the split with the given id. If it's the last split, the
    /// caller is responsible for running the app-quit path (quit-confirm
    /// if busy, close otherwise) — this method just reports
    /// `last_split_closed: true`.
    pub fn close_split(&mut self, id: SplitId) -> std::result::Result<CloseOutcome, UnknownSplit> {
        let idx = self.index_of(id).ok_or(UnknownSplit(id))?;
        if self.splits.len() == 1 {
            return Ok(CloseOutcome {
                last_split_closed: true,
            });
        }
        self.remove_split_state(idx);
        Ok(CloseOutcome {
            last_split_closed: false,
        })
    }

    /// Replace `split_fracs` with new values. No-ops if the length doesn't
    /// match the split count or any value is non-finite/negative — bad
    /// input from a separator drag shouldn't corrupt the layout. Returns
    /// the normalized values that were actually stored.
    pub fn set_fractions(&mut self, fracs: &[f32]) -> Vec<f32> {
        let normalized = normalize_split_fracs(fracs, self.splits.len());
        self.split_fracs = normalized.clone();
        normalized
    }

    /// Save the current workspace layout to `layout_state_path`. Returns
    /// `Ok(true)` if anything was actually saved, `Ok(false)` if there was
    /// no path configured or no session IDs to record.
    pub fn save_layout(&self) -> Result<bool> {
        self.save_current_workspace_layout()
    }

    /// Replace the current workspace with a different root. Saves the
    /// outgoing layout first so switching back later restores the splits
    /// in-order. If any of the new agents fail to spawn, the old state is
    /// preserved and an error is recorded on the focused split so the UI
    /// can surface it.
    pub fn replace_workspace(&mut self, new_root: PathBuf) -> Result<Vec<SpawnedSplit>> {
        self.replace_workspace_with(new_root, |config| AgentClient::spawn(config.clone()))
    }

    /// Factory-based replace_workspace, used by unit tests that need to
    /// substitute a canned spawn result without a real subprocess.
    pub fn replace_workspace_with(
        &mut self,
        new_root: PathBuf,
        mut factory: impl FnMut(&AgentSpawnConfig) -> Result<(AgentClient, AgentEventStream)>,
    ) -> Result<Vec<SpawnedSplit>> {
        if let Err(e) = self.save_current_workspace_layout() {
            self.splits[self.focused].error =
                Some(format!("failed to save workspace layout: {e:#}"));
        }

        let mut next_config = self.spawn_config.clone();
        next_config.workspace_root = new_root.clone();
        next_config.resume = None;
        let layouts = self
            .layout_state_path
            .as_deref()
            .map(load_workspace_layout)
            .unwrap_or_default();
        let restore = layouts.restore_existing_for(&new_root, &next_config.sessions_dir);
        let spawn_configs = restore_spawn_configs(&next_config, restore.as_ref());
        let n = spawn_configs.len();

        let mut next_splits = Vec::with_capacity(n);
        let mut next_streams = Vec::with_capacity(n);
        for config in &spawn_configs {
            match factory(config) {
                Ok((client, stream)) => {
                    next_splits.push(AgentSplit::new(client));
                    next_streams.push(stream);
                }
                Err(e) => {
                    self.splits[self.focused].error = Some(format!("failed to spawn agent: {e:#}"));
                    return Ok(Vec::new());
                }
            }
        }

        self.spawn_config = next_config;
        self.splits = next_splits;
        self.split_fracs = restore
            .as_ref()
            .map(|layout| normalize_split_fracs(&layout.split_fracs, n))
            .unwrap_or_else(|| vec![1.0 / n as f32; n]);
        self.focused = restore
            .as_ref()
            .map(|layout| layout.focused.min(n - 1))
            .unwrap_or(0);

        let split_ids: Vec<SplitId> = self.splits.iter().map(AgentSplit::id).collect();
        Ok(split_ids.into_iter().zip(next_streams).collect())
    }

    fn save_current_workspace_layout(&self) -> Result<bool> {
        let Some(path) = &self.layout_state_path else {
            return Ok(false);
        };
        let mut layouts = load_workspace_layout(path);
        let saved = layouts.save_current(
            &self.spawn_config.workspace_root,
            self.splits.iter().map(AgentSplit::session_id),
            &self.split_fracs,
            self.focused,
        );
        if saved {
            layouts.save(path)?;
        }
        Ok(saved)
    }

    fn remove_split_state(&mut self, split_idx: usize) {
        let reclaimed = self.split_fracs.remove(split_idx);
        // Reclaim the closed split's fraction into a neighbor so the
        // remaining splits fill the window without a sudden resize. Prefer
        // the right neighbor; fall back to the left when removing the
        // rightmost split.
        let neighbor = if split_idx < self.split_fracs.len() {
            split_idx
        } else {
            split_idx - 1
        };
        self.split_fracs[neighbor] += reclaimed;

        self.splits.remove(split_idx);
        adjust_focus_after_remove(&mut self.focused, split_idx, self.splits.len());
    }

    fn index_of(&self, id: SplitId) -> Option<usize> {
        self.splits.iter().position(|split| split.id() == id)
    }
}

fn split_snapshot(split: &AgentSplit) -> SplitSnapshot {
    SplitSnapshot {
        id: split.id(),
        session_id: split.session_id(),
        messages: split.messages().to_vec(),
        streaming: split.streaming_content(),
        waiting: split.is_waiting(),
        error: split.error().map(str::to_owned),
        cancelled: split.is_cancelled(),
    }
}

fn load_workspace_layout(path: &Path) -> WorkspaceLayouts {
    WorkspaceLayouts::load(path).unwrap_or_else(|e| {
        // A corrupted layout file must not wedge the app. The desktop
        // binary will surface the `Err` returned from restore on the
        // focused split's error slot once rendering is ready.
        eprintln!("ignoring workspace layout file {}: {e:#}", path.display());
        WorkspaceLayouts::default()
    })
}

fn restore_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    match restore {
        Some(layout) => layout
            .sessions
            .iter()
            .map(|id| {
                let mut config = base_config.clone();
                config.resume = Some(*id);
                config
            })
            .collect(),
        None => {
            let mut config = base_config.clone();
            config.resume = None;
            vec![config]
        }
    }
}

fn startup_spawn_configs(
    base_config: &AgentSpawnConfig,
    restore: Option<&RestoreLayout>,
) -> Vec<AgentSpawnConfig> {
    if base_config.resume.is_some() {
        vec![base_config.clone()]
    } else {
        restore_spawn_configs(base_config, restore)
    }
}

/// Adjust `focused` after removing the split at `removed_idx`.
///
/// - If the removed split was before the focused one, decrement so focus
///   continues to point at the same logical split.
/// - If the focused split itself was removed, clamp to the last valid index.
fn adjust_focus_after_remove(focused: &mut usize, removed_idx: usize, new_len: usize) {
    if removed_idx < *focused {
        *focused -= 1;
    } else if *focused >= new_len {
        *focused = new_len.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    //! `WorkspaceState` tests.
    //!
    //! These exercise the public API — `apply_event`, `send`, `close_split`,
    //! `replace_workspace_with`, `save_layout`, `any_turn_in_progress`,
    //! `snapshot`, layout normalization, focus adjustment — against a
    //! state constructed directly from in-memory duplex pipes. No real
    //! agent subprocesses are spawned.

    use std::fs;

    use domain::StreamEvent;
    use tokio::io::{BufReader, duplex};

    use super::*;
    use crate::client::AgentClient;

    /// Build an `AgentSplit` over a throwaway duplex pair. The agent-side
    /// pipe halves are leaked so the reader/writer tasks don't emit
    /// disconnect errors while a test is running.
    fn make_split() -> (AgentSplit, SplitId, tokio::io::DuplexStream) {
        let (agent_writer, client_reader) = duplex(4096);
        let (client_writer, agent_reader) = duplex(4096);
        std::mem::forget(agent_reader);
        let (client, _stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        let id = client.id();
        // We drop `_stream` — tests feed events via `apply_event` directly,
        // not through the drain-task path.
        (AgentSplit::new(client), id, agent_writer)
    }

    /// Build a workspace with N pre-wired splits. Returns the state, the
    /// ids of each split (in order), and the agent-side writers so tests
    /// can inject frames when they need to exercise the read path.
    fn make_state(n: usize) -> (WorkspaceState, Vec<SplitId>, Vec<tokio::io::DuplexStream>) {
        assert!(n >= 1);
        let mut splits = Vec::with_capacity(n);
        let mut ids = Vec::with_capacity(n);
        let mut writers = Vec::with_capacity(n);
        for _ in 0..n {
            let (split, id, writer) = make_split();
            splits.push(split);
            ids.push(id);
            writers.push(writer);
        }
        let state = WorkspaceState::new(splits, dummy_spawn_config());
        (state, ids, writers)
    }

    fn dummy_spawn_config() -> AgentSpawnConfig {
        AgentSpawnConfig {
            binary: PathBuf::from("/nonexistent/ox-agent"),
            workspace_root: PathBuf::from("/tmp"),
            model: "test-model".into(),
            sessions_dir: PathBuf::from("/tmp/sessions"),
            resume: None,
            env: vec![],
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ox-workspace-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn run(f: impl FnOnce()) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = rt.enter();
        f();
    }

    // -- constructor invariants ---------------------------------------------

    #[test]
    #[should_panic(expected = "at least one split")]
    fn new_panics_with_no_splits() {
        run(|| {
            let _ = WorkspaceState::new(Vec::new(), dummy_spawn_config());
        });
    }

    #[test]
    fn with_layout_clamps_focus_and_normalizes_fractions() {
        run(|| {
            let (s0, _, _w0) = make_split();
            let (s1, _, _w1) = make_split();
            let state = WorkspaceState::with_layout(
                vec![s0, s1],
                dummy_spawn_config(),
                None,
                Some(vec![0.25, 0.25]), // doesn't sum to 1 — normalized to equal widths
                99,                     // out of range — clamped
            );
            assert_eq!(state.split_fracs(), &[0.5, 0.5]);
            // Focus clamped to last split.
            assert_eq!(state.focused, 1);
        });
    }

    // -- close_split / remove_split -----------------------------------------

    #[test]
    fn close_split_last_signals_last_split_closed() {
        run(|| {
            let (mut state, ids, _writers) = make_state(1);
            let outcome = state.close_split(ids[0]).unwrap();
            assert!(outcome.last_split_closed);
            // State still has the split — caller must handle app-quit.
            assert_eq!(state.split_count(), 1);
        });
    }

    #[test]
    fn close_split_middle_shrinks_and_redistributes_fraction() {
        run(|| {
            let (mut state, ids, _writers) = make_state(3);
            state.split_fracs = vec![0.2, 0.3, 0.5];
            let outcome = state.close_split(ids[1]).unwrap();
            assert!(!outcome.last_split_closed);
            assert_eq!(state.split_count(), 2);
            // The middle split's 0.3 went to the right neighbor (now at
            // index 1 in the post-removal vec).
            assert_eq!(state.split_fracs(), &[0.2, 0.8]);
        });
    }

    #[test]
    fn close_split_rightmost_reclaims_fraction_into_left_neighbor() {
        run(|| {
            let (mut state, ids, _writers) = make_state(3);
            state.split_fracs = vec![0.25, 0.25, 0.5];
            state.close_split(ids[2]).unwrap();
            assert_eq!(state.split_fracs(), &[0.25, 0.75]);
        });
    }

    #[test]
    fn close_split_preserves_focus_when_closing_before_focused() {
        run(|| {
            let (mut state, ids, _writers) = make_state(5);
            state.focused = 3;
            state.close_split(ids[1]).unwrap();
            // Focus was at the 4th split (index 3); after removing index 1
            // it moves to index 2 so it stays on the same logical split.
            assert_eq!(state.focused, 2);
        });
    }

    #[test]
    fn close_split_clamps_focus_when_removing_focused_last() {
        run(|| {
            let (mut state, ids, _writers) = make_state(2);
            state.focused = 1;
            state.close_split(ids[1]).unwrap();
            // Focus clamped to the surviving split.
            assert_eq!(state.focused, 0);
        });
    }

    #[test]
    fn close_split_returns_unknown_for_stale_id() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(1);
            let stale = SplitId::new();
            assert_eq!(state.close_split(stale), Err(UnknownSplit(stale)));
        });
    }

    // -- any_turn_in_progress -----------------------------------------------

    #[test]
    fn any_turn_in_progress_reflects_waiting_or_streaming() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(2);
            assert!(!state.any_turn_in_progress());

            state.splits[0].waiting = true;
            assert!(state.any_turn_in_progress());

            state.splits[0].waiting = false;
            state.splits[1].streaming = Some(app::StreamAccumulator::new());
            assert!(state.any_turn_in_progress());
        });
    }

    // -- apply_event --------------------------------------------------------

    #[test]
    fn apply_event_routes_to_correct_split() {
        run(|| {
            let (mut state, ids, _writers) = make_state(2);
            let id = SessionId::new_v4();
            state
                .apply_event(
                    ids[1],
                    AgentEvent::Ready {
                        session_id: id,
                        workspace_root: PathBuf::from("/w"),
                    },
                )
                .unwrap();
            // Only the target split was touched.
            assert_eq!(state.splits[1].session_id(), Some(id));
            assert_eq!(state.splits[0].session_id(), None);
        });
    }

    #[test]
    fn apply_event_returns_unknown_for_stale_id() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(1);
            let stale = SplitId::new();
            assert_eq!(
                state.apply_event(stale, AgentEvent::TurnComplete),
                Err(UnknownSplit(stale))
            );
        });
    }

    // -- send ----------------------------------------------------------------

    #[tokio::test]
    async fn send_forwards_send_message_to_the_correct_split() {
        // Build two splits where split 1's agent-side reader is observable
        // so the test can verify the command arrived on the right pipe.
        let (agent_writer0, client_reader0) = duplex(4096);
        let (client_writer0, agent_reader0) = duplex(4096);
        let (client0, _s0) = AgentClient::new(BufReader::new(client_reader0), client_writer0);
        let id0 = client0.id();

        let (agent_writer1, client_reader1) = duplex(4096);
        let (client_writer1, agent_reader1) = duplex(4096);
        let (client1, _s1) = AgentClient::new(BufReader::new(client_reader1), client_writer1);
        let id1 = client1.id();

        let mut state = WorkspaceState::new(
            vec![AgentSplit::new(client0), AgentSplit::new(client1)],
            dummy_spawn_config(),
        );

        state
            .send(
                id1,
                AgentCommand::SendMessage {
                    input: "hello split 1".into(),
                },
            )
            .unwrap();

        let mut reader = BufReader::new(agent_reader1);
        let frame: Option<AgentCommand> = protocol::read_frame(&mut reader).await.unwrap();
        match frame.unwrap() {
            AgentCommand::SendMessage { input } => assert_eq!(input, "hello split 1"),
            other => panic!("unexpected {other:?}"),
        }

        assert!(state.splits[state.index_of(id1).unwrap()].is_waiting());
        assert!(!state.splits[state.index_of(id0).unwrap()].is_waiting());

        // Keep the unused pipe halves alive until the end so the reader
        // tasks don't surface disconnect noise.
        drop(agent_writer0);
        drop(agent_writer1);
        drop(agent_reader0);
    }

    #[test]
    fn send_send_message_is_noop_when_waiting() {
        run(|| {
            let (mut state, ids, _writers) = make_state(1);
            state.splits[0].waiting = true;

            state
                .send(
                    ids[0],
                    AgentCommand::SendMessage {
                        input: "ignored".into(),
                    },
                )
                .unwrap();
            // Still waiting, still no error — the send was silently dropped.
            assert!(state.splits[0].is_waiting());
            assert!(state.splits[0].error().is_none());
        });
    }

    #[tokio::test]
    async fn send_forwards_cancel_even_when_waiting() {
        // Cancel while waiting is exactly when cancel is useful — the
        // waiting flag must not suppress it.
        let (agent_writer, client_reader) = duplex(4096);
        let (client_writer, agent_reader) = duplex(4096);
        let (client, _stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        let id = client.id();
        let mut split = AgentSplit::new(client);
        split.waiting = true;

        let mut state = WorkspaceState::new(vec![split], dummy_spawn_config());
        state.send(id, AgentCommand::Cancel).unwrap();

        let mut reader = BufReader::new(agent_reader);
        let frame: Option<AgentCommand> = protocol::read_frame(&mut reader).await.unwrap();
        assert!(matches!(frame.unwrap(), AgentCommand::Cancel));

        drop(agent_writer);
    }

    #[test]
    fn send_returns_unknown_for_stale_id() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(1);
            let stale = SplitId::new();
            match state.send(stale, AgentCommand::Cancel) {
                Err(SendError::UnknownSplit(got)) => assert_eq!(got, stale),
                other => panic!("expected UnknownSplit, got {other:?}"),
            }
        });
    }

    // -- replace_workspace_with ---------------------------------------------

    #[test]
    fn replace_workspace_updates_template_and_returns_new_splits() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(3);
            let new_root = PathBuf::from("/new/workspace");

            let mut seen = Vec::new();
            let returned = state
                .replace_workspace_with(new_root.clone(), |config| {
                    seen.push((config.workspace_root.clone(), config.resume));
                    let (agent_writer, client_reader) = duplex(4096);
                    let (client_writer, agent_reader) = duplex(4096);
                    std::mem::forget(agent_reader);
                    std::mem::forget(agent_writer);
                    Ok(AgentClient::new(
                        BufReader::new(client_reader),
                        client_writer,
                    ))
                })
                .unwrap();

            assert_eq!(state.spawn_config.workspace_root, new_root);
            assert_eq!(state.spawn_config.resume, None);
            assert_eq!(state.split_count(), 1);
            assert_eq!(returned.len(), 1);
            assert_eq!(
                seen,
                vec![(PathBuf::from("/new/workspace"), None::<SessionId>)]
            );
        });
    }

    #[test]
    fn replace_workspace_resets_focus_and_fracs_when_no_layout() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(2);
            state.focused = 1;
            state.split_fracs = vec![0.25, 0.75];

            state
                .replace_workspace_with(PathBuf::from("/new/workspace"), |_| {
                    let (aw, cr) = duplex(4096);
                    let (cw, ar) = duplex(4096);
                    std::mem::forget(aw);
                    std::mem::forget(ar);
                    Ok(AgentClient::new(BufReader::new(cr), cw))
                })
                .unwrap();

            assert_eq!(state.focused, 0);
            assert_eq!(state.split_fracs(), &[1.0]);
        });
    }

    #[test]
    fn replace_workspace_spawn_failure_preserves_existing_state() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(2);
            let old_root = state.spawn_config.workspace_root.clone();
            state.focused = 1;
            state.split_fracs = vec![0.4, 0.6];

            let returned = state
                .replace_workspace_with(PathBuf::from("/bad/workspace"), |_| anyhow::bail!("boom"))
                .unwrap();

            assert!(returned.is_empty());
            assert_eq!(state.spawn_config.workspace_root, old_root);
            assert_eq!(state.split_count(), 2);
            assert_eq!(state.focused, 1);
            assert_eq!(state.split_fracs(), &[0.4, 0.6]);
            assert!(
                state.splits[1].error().is_some_and(
                    |err| err.contains("failed to spawn agent") && err.contains("boom")
                )
            );
        });
    }

    #[test]
    fn replace_workspace_restores_saved_sessions_fractions_and_focus() {
        run(|| {
            let dir = temp_dir("replace-restore");
            let layout_path = dir.join("workspaces.json");
            let target_root = dir.join("target");
            fs::create_dir_all(&target_root).unwrap();

            let id1 = SessionId::new_v4();
            let id2 = SessionId::new_v4();
            let mut layouts = WorkspaceLayouts::default();
            layouts.save_current(&target_root, [Some(id1), Some(id2)], &[0.3, 0.7], 1);
            layouts.save(&layout_path).unwrap();

            let (mut state, _ids, _writers) = make_state(1);
            state.spawn_config.sessions_dir = dir.join("sessions");
            fs::create_dir_all(&state.spawn_config.sessions_dir).unwrap();
            fs::write(
                state.spawn_config.sessions_dir.join(format!("{id1}.json")),
                "{}",
            )
            .unwrap();
            fs::write(
                state.spawn_config.sessions_dir.join(format!("{id2}.json")),
                "{}",
            )
            .unwrap();
            state.layout_state_path = Some(layout_path);

            let mut seen = Vec::new();
            state
                .replace_workspace_with(target_root.clone(), |config| {
                    seen.push((config.workspace_root.clone(), config.resume));
                    let (aw, cr) = duplex(4096);
                    let (cw, ar) = duplex(4096);
                    std::mem::forget(aw);
                    std::mem::forget(ar);
                    Ok(AgentClient::new(BufReader::new(cr), cw))
                })
                .unwrap();

            assert_eq!(
                seen,
                vec![
                    (target_root.clone(), Some(id1)),
                    (target_root.clone(), Some(id2))
                ]
            );
            assert_eq!(state.split_count(), 2);
            assert_eq!(state.split_fracs(), &[0.3, 0.7]);
            assert_eq!(state.focused, 1);
            assert_eq!(state.spawn_config.workspace_root, target_root);
            assert_eq!(state.spawn_config.resume, None);
        });
    }

    #[test]
    fn replace_workspace_saves_current_layout_before_switching() {
        run(|| {
            let dir = temp_dir("replace-save-current");
            let layout_path = dir.join("workspaces.json");
            let old_root = dir.join("old");
            let new_root = dir.join("new");
            fs::create_dir_all(&old_root).unwrap();
            fs::create_dir_all(&new_root).unwrap();

            let (mut state, _ids, _writers) = make_state(2);
            state.layout_state_path = Some(layout_path.clone());
            state.spawn_config.workspace_root = old_root.clone();
            state.splits[0].session_id = Some(SessionId::new_v4());
            state.splits[1].session_id = Some(SessionId::new_v4());
            state.split_fracs = vec![0.4, 0.6];
            state.focused = 1;

            let old_ids: Vec<_> = state.splits.iter().map(AgentSplit::session_id).collect();
            state
                .replace_workspace_with(new_root, |_| {
                    let (aw, cr) = duplex(4096);
                    let (cw, ar) = duplex(4096);
                    std::mem::forget(aw);
                    std::mem::forget(ar);
                    Ok(AgentClient::new(BufReader::new(cr), cw))
                })
                .unwrap();

            let layouts = WorkspaceLayouts::load(&layout_path).unwrap();
            let restored = layouts.restore_for(&old_root).unwrap();
            assert_eq!(
                restored.sessions,
                old_ids.into_iter().flatten().collect::<Vec<_>>()
            );
            assert_eq!(restored.split_fracs, vec![0.4, 0.6]);
            assert_eq!(restored.focused, 1);
        });
    }

    // -- set_fractions / snapshot / set_focused ------------------------------

    #[test]
    fn set_fractions_normalizes_and_rejects_bad_input() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(2);
            // Equal-valid input passes through.
            let got = state.set_fractions(&[0.25, 0.75]);
            assert_eq!(got, &[0.25, 0.75]);
            assert_eq!(state.split_fracs(), &[0.25, 0.75]);

            // Negative / non-finite input falls back to equal widths.
            let got = state.set_fractions(&[f32::NAN, 1.0]);
            assert_eq!(got, &[0.5, 0.5]);
            assert_eq!(state.split_fracs(), &[0.5, 0.5]);

            // Wrong length also falls back to equal widths.
            let got = state.set_fractions(&[0.5]);
            assert_eq!(got, &[0.5, 0.5]);
        });
    }

    #[test]
    fn set_focused_updates_focus_for_known_id() {
        run(|| {
            let (mut state, ids, _writers) = make_state(3);
            assert_eq!(state.focused().unwrap(), ids[0]);
            state.set_focused(ids[2]).unwrap();
            assert_eq!(state.focused().unwrap(), ids[2]);
        });
    }

    #[test]
    fn set_focused_returns_unknown_for_stale_id() {
        run(|| {
            let (mut state, _ids, _writers) = make_state(1);
            let stale = SplitId::new();
            assert_eq!(state.set_focused(stale), Err(UnknownSplit(stale)));
        });
    }

    #[test]
    fn snapshot_reflects_current_state() {
        run(|| {
            let (mut state, ids, _writers) = make_state(2);
            state.splits[0].messages.push(Message::user("hello"));
            state.splits[1].waiting = true;
            state.splits[1].error = Some("broke".into());
            state.splits[1].cancelled = true;
            state.set_focused(ids[1]).unwrap();

            let snap = state.snapshot();
            assert_eq!(snap.splits.len(), 2);
            assert_eq!(snap.splits[0].id, ids[0]);
            assert_eq!(snap.splits[0].messages.len(), 1);
            assert!(!snap.splits[0].waiting);
            assert_eq!(snap.splits[1].id, ids[1]);
            assert!(snap.splits[1].waiting);
            assert_eq!(snap.splits[1].error.as_deref(), Some("broke"));
            assert!(snap.splits[1].cancelled);
            assert_eq!(snap.focused, Some(ids[1]));
        });
    }

    // -- layout save/load round-trip -----------------------------------------

    #[test]
    fn save_layout_writes_current_workspace() {
        run(|| {
            let dir = temp_dir("save-layout");
            let layout_path = dir.join("workspaces.json");
            let (mut state, _ids, _writers) = make_state(2);
            state.layout_state_path = Some(layout_path.clone());
            state.spawn_config.workspace_root = dir.clone();
            state.splits[0].session_id = Some(SessionId::new_v4());
            state.splits[1].session_id = Some(SessionId::new_v4());

            assert!(state.save_layout().unwrap());

            let layouts = WorkspaceLayouts::load(&layout_path).unwrap();
            let restored = layouts.restore_for(&dir).unwrap();
            assert_eq!(restored.sessions.len(), 2);
        });
    }

    #[test]
    fn save_layout_without_sessions_returns_false() {
        run(|| {
            let dir = temp_dir("save-nosession");
            let layout_path = dir.join("workspaces.json");
            let (mut state, _ids, _writers) = make_state(1);
            state.layout_state_path = Some(layout_path.clone());

            // No session_id set — save_current reports it didn't save and
            // the file is never written.
            assert!(!state.save_layout().unwrap());
            assert!(!layout_path.exists());
        });
    }

    // -- helper-function unit tests -----------------------------------------

    #[test]
    fn restore_spawn_configs_use_saved_sessions_in_order() {
        let base = dummy_spawn_config();
        let id1 = SessionId::new_v4();
        let id2 = SessionId::new_v4();
        let layout = RestoreLayout {
            sessions: vec![id1, id2],
            split_fracs: vec![0.25, 0.75],
            focused: 1,
        };

        let configs = restore_spawn_configs(&base, Some(&layout));

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].resume, Some(id1));
        assert_eq!(configs[1].resume, Some(id2));
        assert!(
            configs
                .iter()
                .all(|config| config.workspace_root == base.workspace_root)
        );
    }

    #[test]
    fn restore_spawn_configs_falls_back_to_fresh_without_layout() {
        let mut base = dummy_spawn_config();
        base.resume = Some(SessionId::new_v4());

        let configs = restore_spawn_configs(&base, None);

        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].resume, None);
    }

    #[test]
    fn startup_spawn_configs_preserve_explicit_resume_override() {
        let mut base = dummy_spawn_config();
        let explicit = SessionId::new_v4();
        let saved = SessionId::new_v4();
        base.resume = Some(explicit);
        let layout = RestoreLayout {
            sessions: vec![saved],
            split_fracs: vec![1.0],
            focused: 0,
        };

        let configs = startup_spawn_configs(&base, Some(&layout));

        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].resume, Some(explicit));
    }

    #[test]
    fn adjust_focus_after_remove_decrements_when_removing_before_focused() {
        let mut focused = 3;
        adjust_focus_after_remove(&mut focused, 1, 4);
        assert_eq!(focused, 2);
    }

    #[test]
    fn adjust_focus_after_remove_clamps_when_focused_is_removed_last() {
        let mut focused = 1;
        adjust_focus_after_remove(&mut focused, 1, 1);
        assert_eq!(focused, 0);
    }

    #[test]
    fn adjust_focus_after_remove_leaves_focus_alone_when_before_it_in_a_larger_window() {
        let mut focused = 1;
        adjust_focus_after_remove(&mut focused, 2, 3);
        assert_eq!(focused, 1);
    }

    // Sanity check that apply_event dispatches into handle_event — the
    // detailed per-event behavior is already pinned down in split.rs.
    #[test]
    fn apply_event_dispatches_into_split_handle_event() {
        run(|| {
            let (mut state, ids, _writers) = make_state(1);
            state
                .apply_event(
                    ids[0],
                    AgentEvent::StreamDelta {
                        event: StreamEvent::TextDelta { delta: "hi".into() },
                    },
                )
                .unwrap();
            assert!(state.splits[0].streaming.is_some());
        });
    }
}
