//! `AgentClient` — the GUI's handle on a single `ox-agent` subprocess.
//!
//! Replaces the in-process `run_backend` controller with an IPC client that:
//! - spawns `ox-agent` as a tokio `Child` with `kill_on_drop(true)`, so the
//!   agent dies if the GUI exits for any reason;
//! - runs two background tasks per client — a reader that parses stdout into
//!   `AgentEvent`s and forwards them to a GUI-bound mpsc channel, and a
//!   writer that serializes `AgentCommand`s from the GUI onto stdin;
//! - exposes the same "channels as the public API" shape today's code uses,
//!   so the `OxApp` event loop stays structurally identical.
//!
//! The split between [`AgentClient::new`] and [`AgentClient::spawn`] matters
//! for testing: `new` takes any `AsyncBufRead` / `AsyncWrite` pair, so unit
//! tests can drive a real client over `tokio::io::duplex` without launching
//! a process. `spawn` is the thin `tokio::process::Command` wrapper around it.

use std::path::PathBuf;

use anyhow::{Context, Result};
use domain::SessionId;
use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
use tokio::io::{AsyncBufRead, AsyncWrite, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

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
    /// GUI's entire environment.
    pub env: Vec<(String, String)>,
}

/// The GUI-side handle on one running agent.
///
/// Public surface is the two channel endpoints and the [`Drop`] impl; the
/// reader and writer tasks run in the background until either the channels
/// close or the underlying I/O hangs up.
pub struct AgentClient {
    cmd_tx: mpsc::UnboundedSender<AgentCommand>,
    evt_rx: mpsc::UnboundedReceiver<AgentEvent>,
    /// Handle to the child process, if this client was created by `spawn`.
    /// Dropped along with the client, which triggers `kill_on_drop`.
    _child: Option<Child>,
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
    pub fn new<R, W>(reader: R, writer: W) -> Self
    where
        R: AsyncBufRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<AgentCommand>();
        let (evt_tx, evt_rx) = mpsc::unbounded_channel::<AgentEvent>();
        spawn_reader(reader, evt_tx.clone());
        spawn_writer(writer, cmd_rx, evt_tx);
        Self {
            cmd_tx,
            evt_rx,
            _child: None,
        }
    }

    /// Spawn `ox-agent` as a subprocess and return a client wired to its
    /// stdio. The child's stderr is inherited (flows to the GUI terminal).
    ///
    /// `kill_on_drop(true)` ties the agent's lifetime to the client — the
    /// moment the `AgentClient` is dropped (GUI shuts down, tab closed,
    /// etc.), the child is SIGKILL'd. That's the "cancel = kill" contract
    /// the plan commits to.
    pub fn spawn(config: AgentSpawnConfig) -> Result<Self> {
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
        let mut client = Self::new(reader, stdin);
        client._child = Some(child);
        Ok(client)
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

    /// Non-blocking poll for the next event.
    ///
    /// Returns:
    /// - `Ok(evt)` on a delivered event,
    /// - `Err(TryRecvError::Empty)` when no event is queued yet,
    /// - `Err(TryRecvError::Disconnected)` when the reader task has exited.
    pub fn try_recv(&mut self) -> std::result::Result<AgentEvent, mpsc::error::TryRecvError> {
        self.evt_rx.try_recv()
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
        tokio::io::DuplexStream,
        tokio::io::DuplexStream,
    ) {
        let (agent_writer, client_reader) = tokio::io::duplex(4096);
        let (client_writer, agent_reader) = tokio::io::duplex(4096);
        let client = AgentClient::new(BufReader::new(client_reader), client_writer);
        (client, agent_writer, agent_reader)
    }

    /// Pump `try_recv` with a short sleep between calls until a matching
    /// event arrives, the channel disconnects, or `timeout` elapses.
    async fn recv_event(client: &mut AgentClient, timeout: Duration) -> Option<AgentEvent> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            match client.try_recv() {
                Ok(evt) => return Some(evt),
                Err(mpsc::error::TryRecvError::Empty) => {
                    if tokio::time::Instant::now() >= deadline {
                        return None;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return None,
            }
        }
    }

    #[tokio::test]
    async fn events_forwarded_from_reader_to_channel() {
        let (mut client, mut agent_writer, _agent_reader) = make_client();

        // Agent sends a Ready frame; client should surface it on its channel.
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

        let evt = recv_event(&mut client, Duration::from_secs(1)).await;
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
        let (client, _agent_writer, agent_reader) = make_client();

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
    async fn clean_eof_emits_no_error_and_reader_task_exits() {
        // A well-behaved agent shutdown closes *both* pipes; on the GUI side
        // the reader task observes EOF and the writer task observes a
        // broken pipe. This test pins down the contract for the reader's
        // half of that: on EOF alone, no spurious `Error` event appears,
        // and the reader task exits cleanly. (The writer task keeps the
        // event channel open by design — it still has a sender clone —
        // until a command actually fails; that's tested elsewhere.)
        let (mut client, agent_writer, _agent_reader) = make_client();
        drop(agent_writer);

        // Give the reader task time to observe EOF.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // No event should have been produced by the reader on EOF — only
        // real events or parse errors do. A buffered event from the reader
        // task would surface here as `Ok(...)`.
        for _ in 0..10 {
            match client.try_recv() {
                Err(mpsc::error::TryRecvError::Empty) => {}
                Err(mpsc::error::TryRecvError::Disconnected) => return,
                Ok(evt) => panic!("reader should not emit events after EOF, got {evt:?}"),
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn malformed_line_produces_error_and_reader_keeps_running() {
        let (mut client, mut agent_writer, _agent_reader) = make_client();

        agent_writer.write_all(b"this is not json\n").await.unwrap();
        write_frame(&mut agent_writer, &AgentEvent::TurnComplete)
            .await
            .unwrap();
        drop(agent_writer);

        // First event must be an Error — the reader tolerated the bad line.
        let first = recv_event(&mut client, Duration::from_secs(1))
            .await
            .expect("first event");
        assert!(matches!(first, AgentEvent::Error { .. }), "{first:?}");

        // Second event is the valid TurnComplete that came after.
        let second = recv_event(&mut client, Duration::from_secs(1))
            .await
            .expect("second event");
        assert!(matches!(second, AgentEvent::TurnComplete), "{second:?}");
    }

    #[tokio::test]
    async fn broken_pipe_on_send_surfaces_as_error_event() {
        // Dropping the agent reader kills the pipe the writer task uses.
        // After the next command, the writer task emits an Error and exits.
        let (mut client, _agent_writer, agent_reader) = make_client();
        drop(agent_reader);

        // Give the OS a moment to register the broken-pipe state before the
        // first write hits the pipe.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // This command may succeed (buffered) the first time, but subsequent
        // sends will break the pipe. Send a handful to force the failure.
        for _ in 0..10 {
            let _ = client.send(AgentCommand::SendMessage { input: "hi".into() });
        }

        // Eventually an Error frame arrives on the event channel.
        let mut got_error = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            match client.try_recv() {
                Ok(AgentEvent::Error { message }) => {
                    assert!(message.contains("agent write failed"), "{message}");
                    got_error = true;
                    break;
                }
                Ok(_) => {}
                Err(mpsc::error::TryRecvError::Empty) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        assert!(got_error, "expected an Error event after pipe closed");
    }

    #[tokio::test]
    async fn stream_delta_preserves_wire_payload() {
        // StreamEvent round-trips across the wire without field drift.
        let (mut client, mut agent_writer, _agent_reader) = make_client();
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

        let evt = recv_event(&mut client, Duration::from_secs(1))
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
}
