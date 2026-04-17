//! `AgentClient` — the GUI-side handle on a single `ox-agent` subprocess.
//!
//! A client owns the command-send side of the IPC channel; its paired
//! [`AgentEventStream`] owns the event-receive side. The halves are split
//! so a drain task can `await` on the stream without contending with the
//! shared `WorkspaceState` mutex the command sender lives behind.
//!
//! The split between [`AgentClient::new`] and [`AgentClient::spawn`] exists
//! for testing: `new` takes any `AsyncBufRead` / `AsyncWrite` pair so unit
//! tests drive a real client over `tokio::io::duplex`. `spawn` is the thin
//! `tokio::process::Command` wrapper around `new`.
//!
//! The agent is killed on drop (`kill_on_drop(true)`) so dropping the
//! `AgentClient` is the cancellation path: the subprocess always dies with
//! the GUI, not the other way around.

use std::path::PathBuf;

use anyhow::{Context, Result};
use domain::SessionId;
use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufRead, AsyncWrite, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Stable, UI-local identity for a split.
///
/// Generated GUI-side when an `AgentClient` is created. Never crosses the
/// IPC boundary to `ox-agent` — splits are a GUI concept. Lives here rather
/// than in `domain` because it is not persisted and does not round-trip
/// through the wire protocol: domain identities are reserved for values
/// that cross processes or storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SplitId(pub Uuid);

impl SplitId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SplitId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SplitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Config passed to [`AgentClient::spawn`]. Deliberately small — extra agent
/// knobs belong in the agent's env, not in a twelve-field "options" struct.
#[derive(Debug, Clone)]
pub struct AgentSpawnConfig {
    /// Path to the `ox-agent` binary.
    pub binary: PathBuf,
    /// Workspace root passed to the agent on the command line.
    pub workspace_root: PathBuf,
    /// OpenRouter model ID.
    pub model: String,
    /// Sessions directory the agent reads/writes.
    pub sessions_dir: PathBuf,
    /// Optional session to resume. `None` starts a fresh session.
    pub resume: Option<SessionId>,
    /// Extra environment variables. Inherited-by-default env vars (like PATH)
    /// are not cleared; these are merged on top. Used primarily to pass
    /// `OPENROUTER_API_KEY` into the child process without inheriting the
    /// host's entire environment.
    pub env: Vec<(String, String)>,
}

/// Send-side handle. Thread-safe by construction — `mpsc::UnboundedSender`
/// is `Sync`, so the state guard can hand out `&AgentClient` references
/// across tasks without cloning the client.
pub struct AgentClient {
    id: SplitId,
    cmd_tx: mpsc::UnboundedSender<AgentCommand>,
    /// Handle to the child process, if this client was created by `spawn`.
    /// Dropped along with the client, which triggers `kill_on_drop`.
    _child: Option<Child>,
}

/// Receive-side handle. Owned by a drain task in the desktop binary — kept
/// off the shared state mutex so `recv().await` never blocks a lock.
pub struct AgentEventStream {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl AgentEventStream {
    /// Await the next event from the agent. Returns `None` when the reader
    /// and writer tasks have both shut down (agent dead or torn down).
    pub async fn recv(&mut self) -> Option<AgentEvent> {
        self.rx.recv().await
    }
}

impl AgentClient {
    /// Build a client over any pair of reader + writer. Used by both
    /// `spawn` (which passes the child's stdout / stdin) and by unit tests
    /// (which pass `tokio::io::duplex` halves).
    ///
    /// Internally spawns one reader task and one writer task via the
    /// `spawn_reader` / `spawn_writer` helpers. `spawn` calls this function
    /// too, so both code paths share a single implementation — no risk of
    /// drift between test and production.
    pub fn new<R, W>(reader: R, writer: W) -> (Self, AgentEventStream)
    where
        R: AsyncBufRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<AgentCommand>();
        let (evt_tx, evt_rx) = mpsc::unbounded_channel::<AgentEvent>();
        spawn_reader(reader, evt_tx.clone());
        spawn_writer(writer, cmd_rx, evt_tx);
        let client = Self {
            id: SplitId::new(),
            cmd_tx,
            _child: None,
        };
        let stream = AgentEventStream { rx: evt_rx };
        (client, stream)
    }

    /// Spawn `ox-agent` as a subprocess and return a client wired to its
    /// stdio along with the event stream.
    ///
    /// `kill_on_drop(true)` ties the agent's lifetime to the client — the
    /// moment the `AgentClient` is dropped (window closed, split removed,
    /// etc.), the child is SIGKILL'd.
    pub fn spawn(config: AgentSpawnConfig) -> Result<(Self, AgentEventStream)> {
        let mut cmd = Command::new(&config.binary);
        cmd.arg("--workspace-root")
            .arg(&config.workspace_root)
            .arg("--model")
            .arg(&config.model)
            .arg("--sessions-dir")
            .arg(&config.sessions_dir);
        if let Some(id) = config.resume {
            cmd.arg("--resume").arg(id.to_string());
        }
        for (k, v) in &config.env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning {}", config.binary.display()))?;

        let stdin = child.stdin.take().context("child stdin missing")?;
        let stdout = child.stdout.take().context("child stdout missing")?;
        let reader = BufReader::new(stdout);

        // Route through `new` so the reader/writer tasks come from the
        // same code as the duplex-backed unit tests.
        let (mut client, stream) = Self::new(reader, stdin);
        client._child = Some(child);
        Ok((client, stream))
    }

    /// Stable identifier assigned when the client was built.
    pub fn id(&self) -> SplitId {
        self.id
    }

    /// Send a command to the agent. Returns `Err` only if the writer task
    /// has already shut down (agent dead or the `AgentClient` is being torn
    /// down).
    pub fn send(
        &self,
        cmd: AgentCommand,
    ) -> std::result::Result<(), mpsc::error::SendError<AgentCommand>> {
        self.cmd_tx.send(cmd)
    }
}

// Reader and writer task helpers. Both `new` and `spawn` route through
// `new`, which calls these helpers, so there's exactly one implementation of
// each task body. No risk of drift between test and production code paths.

fn spawn_reader<R>(mut reader: R, evt_tx: mpsc::UnboundedSender<AgentEvent>)
where
    R: AsyncBufRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        loop {
            match read_frame::<_, AgentEvent>(&mut reader).await {
                Ok(Some(evt)) => {
                    if evt_tx.send(evt).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    if evt_tx
                        .send(AgentEvent::Error {
                            message: format!("{e:#}"),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });
}

fn spawn_writer<W>(
    mut writer: W,
    mut cmd_rx: mpsc::UnboundedReceiver<AgentCommand>,
    evt_tx: mpsc::UnboundedSender<AgentEvent>,
) where
    W: AsyncWrite + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            if let Err(e) = write_frame(&mut writer, &cmd).await {
                let _ = evt_tx.send(AgentEvent::Error {
                    message: format!("agent write failed: {e:#}"),
                });
                break;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use domain::StreamEvent;
    use tokio::io::{AsyncWriteExt, BufReader};

    use super::*;

    /// Build a client over two duplex pipes and return the halves the test
    /// side uses to play the role of the agent. `agent_writer` is where the
    /// test injects `AgentEvent` frames the client should see; `agent_reader`
    /// is where the test reads `AgentCommand` frames the client emitted.
    fn make_client() -> (
        AgentClient,
        AgentEventStream,
        tokio::io::DuplexStream,
        tokio::io::DuplexStream,
    ) {
        let (agent_writer, client_reader) = tokio::io::duplex(4096);
        let (client_writer, agent_reader) = tokio::io::duplex(4096);
        let (client, stream) = AgentClient::new(BufReader::new(client_reader), client_writer);
        (client, stream, agent_writer, agent_reader)
    }

    /// Await the next event with a timeout. Replaces the old `try_recv`
    /// polling loop now that the receiver is an async `AgentEventStream`.
    async fn recv_event(stream: &mut AgentEventStream, timeout: Duration) -> Option<AgentEvent> {
        tokio::time::timeout(timeout, stream.recv())
            .await
            .ok()
            .flatten()
    }

    #[tokio::test]
    async fn events_forwarded_from_reader_to_channel() {
        let (_client, mut stream, mut agent_writer, _agent_reader) = make_client();

        let id = SessionId::new_v4();
        write_frame(
            &mut agent_writer,
            &AgentEvent::Ready {
                session_id: id,
                workspace_root: PathBuf::from("/x"),
            },
        )
        .await
        .unwrap();

        let evt = recv_event(&mut stream, Duration::from_secs(1)).await;
        let Some(AgentEvent::Ready {
            session_id,
            workspace_root,
        }) = evt
        else {
            panic!("expected Ready, got {evt:?}");
        };
        assert_eq!(session_id, id);
        assert_eq!(workspace_root, PathBuf::from("/x"));
    }

    #[tokio::test]
    async fn commands_written_to_writer_in_send_order() {
        let (client, _stream, _agent_writer, agent_reader) = make_client();

        client
            .send(AgentCommand::SendMessage {
                input: "first".into(),
            })
            .unwrap();
        client
            .send(AgentCommand::SendMessage {
                input: "second".into(),
            })
            .unwrap();

        let mut reader = BufReader::new(agent_reader);
        let f1: Option<AgentCommand> = read_frame(&mut reader).await.unwrap();
        let f2: Option<AgentCommand> = read_frame(&mut reader).await.unwrap();
        match f1.unwrap() {
            AgentCommand::SendMessage { input } => assert_eq!(input, "first"),
            other => panic!("unexpected {other:?}"),
        }
        match f2.unwrap() {
            AgentCommand::SendMessage { input } => assert_eq!(input, "second"),
            other => panic!("unexpected {other:?}"),
        }

        drop(client);
    }

    #[tokio::test]
    async fn clean_eof_emits_no_error_and_stream_closes() {
        // A well-behaved agent shutdown closes *both* pipes; on EOF the
        // reader task exits without emitting an error and, once the writer
        // task's sender also drops, `recv()` returns `None`.
        let (client, mut stream, agent_writer, _agent_reader) = make_client();
        drop(agent_writer);
        drop(client); // drops the writer-side sender along with the command channel

        // `recv()` must eventually return `None` (stream closed) rather than
        // a spurious `Error` frame.
        let evt = tokio::time::timeout(Duration::from_secs(1), stream.recv())
            .await
            .expect("recv should resolve once stream closes");
        assert!(evt.is_none(), "expected None on EOF, got {evt:?}");
    }

    #[tokio::test]
    async fn malformed_line_produces_error_and_reader_keeps_running() {
        let (_client, mut stream, mut agent_writer, _agent_reader) = make_client();

        agent_writer.write_all(b"this is not json\n").await.unwrap();
        write_frame(&mut agent_writer, &AgentEvent::TurnComplete)
            .await
            .unwrap();
        drop(agent_writer);

        // First event must be an Error — the reader tolerated the bad line.
        let first = recv_event(&mut stream, Duration::from_secs(1))
            .await
            .expect("first event");
        assert!(matches!(first, AgentEvent::Error { .. }), "{first:?}");

        // Second event is the valid TurnComplete that came after.
        let second = recv_event(&mut stream, Duration::from_secs(1))
            .await
            .expect("second event");
        assert!(matches!(second, AgentEvent::TurnComplete), "{second:?}");
    }

    #[tokio::test]
    async fn broken_pipe_on_send_surfaces_as_error_event() {
        // Dropping the agent reader kills the pipe the writer task uses.
        // After the next command, the writer task emits an Error and exits.
        let (client, mut stream, _agent_writer, agent_reader) = make_client();
        drop(agent_reader);

        // Give the OS a moment to register the broken-pipe state before the
        // first write hits the pipe.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Subsequent sends will break the pipe. Send a handful to force the
        // failure.
        for _ in 0..10 {
            let _ = client.send(AgentCommand::SendMessage { input: "hi".into() });
        }

        // Eventually an Error frame arrives on the event stream.
        let mut got_error = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(50), stream.recv()).await {
                Ok(Some(AgentEvent::Error { message })) => {
                    assert!(message.contains("agent write failed"), "{message}");
                    got_error = true;
                    break;
                }
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {}
            }
        }
        assert!(got_error, "expected an Error event after pipe closed");
    }

    #[tokio::test]
    async fn stream_delta_preserves_wire_payload() {
        // StreamEvent round-trips across the wire without field drift.
        let (_client, mut stream, mut agent_writer, _agent_reader) = make_client();
        write_frame(
            &mut agent_writer,
            &AgentEvent::StreamDelta {
                event: StreamEvent::EncryptedReasoning {
                    data: "blob".into(),
                    format: "fmt".into(),
                },
            },
        )
        .await
        .unwrap();

        let evt = recv_event(&mut stream, Duration::from_secs(1))
            .await
            .expect("event");
        match evt {
            AgentEvent::StreamDelta {
                event: StreamEvent::EncryptedReasoning { data: d, format: f },
            } => {
                assert_eq!(d, "blob");
                assert_eq!(f, "fmt");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn split_ids_are_unique_per_client() {
        // Minimal sanity check that the `SplitId` assigned by `new` is
        // fresh per client — we rely on this for routing in WorkspaceState.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = rt.enter();
        let (c1, _s1, _aw1, _ar1) = make_client();
        let (c2, _s2, _aw2, _ar2) = make_client();
        assert_ne!(c1.id(), c2.id());
    }
}
