//! `AgentClient` — the host-side handle on a single `ox-agent` subprocess.
//!
//! A client owns the command-send side of the IPC channel; its paired
//! [`AgentEventStream`] owns the event-receive side. The halves are split
//! so a drain task can `await` on the stream without contending with the
//! shared state the command sender lives behind.
//!
//! `AgentClient::new` takes any `AsyncBufRead` / `AsyncWrite` pair so unit
//! tests drive a real client over `tokio::io::duplex`, while production
//! adapters can wire those same generic halves to a subprocess, socket, or any
//! other transport.

use std::path::PathBuf;

use domain::SessionId;
use protocol::{AgentCommand, AgentEvent, read_frame, write_frame};
use tokio::io::{AsyncBufRead, AsyncWrite};
use tokio::sync::mpsc;

/// Config passed to an [`AgentSpawner`](crate::AgentSpawner). Deliberately small — extra agent
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
    /// Pre-allocated session id for a **fresh** session. When set and
    /// `resume` is `None`, the agent uses this id instead of generating
    /// one — keeping the host-controlled worktree directory (named after
    /// the id's short prefix) and the agent's `{id}.json` in lockstep.
    /// Ignored when `resume` is set.
    pub session_id: Option<SessionId>,
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
    cmd_tx: mpsc::UnboundedSender<AgentCommand>,
    /// Optional owner for transport lifetime resources supplied by an
    /// adapter. The process adapter stores its `tokio::process::Child`
    /// here so dropping the client also drops the child handle.
    _drop_guard: Option<Box<dyn Send>>,
}

/// Receive-side handle. Owned by a drain task in the host binary — kept
/// off any shared state mutex so `recv().await` never blocks a lock.
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
    /// Build a client over any pair of reader + writer. Production adapters
    /// pass the transport's read/write halves; unit tests pass
    /// `tokio::io::duplex` halves.
    ///
    /// Internally spawns one reader task and one writer task via the
    /// `spawn_reader` / `spawn_writer` helpers so test and production
    /// transports share the same IPC implementation.
    pub fn new<R, W>(reader: R, writer: W) -> (Self, AgentEventStream)
    where
        R: AsyncBufRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<AgentCommand>();
        let (evt_tx, evt_rx) = mpsc::unbounded_channel::<AgentEvent>();
        // Oneshot that closes when the reader task exits. The writer
        // task selects on `cmd_rx.recv()` and this oneshot so that an
        // EOF on the agent's stdout propagates to the writer side —
        // without it, a dead agent would leave the writer parked on
        // `cmd_rx.recv()` and its `evt_tx` clone would keep the event
        // stream open forever, masking the agent's death from the pump.
        let (reader_closed_tx, reader_closed_rx) = tokio::sync::oneshot::channel::<()>();
        spawn_reader(reader, evt_tx.clone(), reader_closed_tx);
        spawn_writer(writer, cmd_rx, evt_tx, reader_closed_rx);
        let client = Self {
            cmd_tx,
            _drop_guard: None,
        };
        let stream = AgentEventStream { rx: evt_rx };
        (client, stream)
    }

    /// Attach an adapter-owned lifetime guard to this client. The guard
    /// is dropped with the client; production process spawning uses this
    /// to keep the child handle tied to the command sender.
    pub fn with_drop_guard(mut self, guard: impl Send + 'static) -> Self {
        self._drop_guard = Some(Box::new(guard));
        self
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

// Reader and writer task helpers. Every transport routes through `new`, which
// calls these helpers, so there's exactly one implementation of each task body.

fn spawn_reader<R>(
    mut reader: R,
    evt_tx: mpsc::UnboundedSender<AgentEvent>,
    reader_closed_tx: tokio::sync::oneshot::Sender<()>,
) where
    R: AsyncBufRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        // Moving the sender into the task means the oneshot closes
        // the instant the task exits — success or panic — which the
        // writer task observes via its `reader_closed` branch.
        let _reader_closed_tx = reader_closed_tx;
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
    mut reader_closed: tokio::sync::oneshot::Receiver<()>,
) where
    W: AsyncWrite + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        loop {
            tokio::select! {
                maybe_cmd = cmd_rx.recv() => {
                    let Some(cmd) = maybe_cmd else { break };
                    if let Err(e) = write_frame(&mut writer, &cmd).await {
                        let _ = evt_tx.send(AgentEvent::Error {
                            message: format!("agent write failed: {e:#}"),
                        });
                        break;
                    }
                }
                // The oneshot resolves (with Err) the instant the
                // reader task exits and drops its sender. That
                // signals "the agent's stdout is gone" — there's no
                // point continuing to buffer commands, so we break
                // and let our `evt_tx` clone drop, which lets the
                // event stream return `None` to the pump.
                _ = &mut reader_closed => break,
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
}
