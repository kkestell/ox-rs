use protocol::{AgentCommand, read_frame};
use tokio::io::AsyncBufRead;
use tokio::sync::mpsc;

/// One frame's worth of progress from the command reader task.
///
/// The driver can't `read_frame` inline with its turn loop because that
/// would make reads non-cancellation-safe — a mid-turn `select!` could
/// abandon a partially-read NDJSON line. Instead a dedicated task reads
/// frames and forwards them through an unbounded channel as one of
/// these variants.
pub(super) enum ReaderEvent {
    Command(AgentCommand),
    Malformed(String),
    Eof,
}

/// Spawn a background task that pumps `AgentCommand` frames off `reader`
/// and forwards them as [`ReaderEvent`]s. The returned receiver closes
/// when the reader hits EOF (the task sends a final `Eof` before exiting)
/// or when the receiver itself is dropped.
pub(super) fn spawn_reader<R>(mut reader: R) -> mpsc::UnboundedReceiver<ReaderEvent>
where
    R: AsyncBufRead + Unpin + Send + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            match read_frame::<_, AgentCommand>(&mut reader).await {
                Ok(Some(cmd)) => {
                    if tx.send(ReaderEvent::Command(cmd)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = tx.send(ReaderEvent::Eof);
                    break;
                }
                Err(e) => {
                    if tx.send(ReaderEvent::Malformed(format!("{e:#}"))).is_err() {
                        break;
                    }
                }
            }
        }
    });
    rx
}
