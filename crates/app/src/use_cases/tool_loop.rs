use anyhow::Result;
use domain::{Message, Session};

use crate::approval::ToolApprovalRequest;
use crate::ports::SessionStore;
use crate::tools::ToolRegistry;
use crate::use_cases::TurnEvent;

/// Intermediate classification of a single model-requested tool call,
/// produced by the runner's `plan_tool_approvals` step so the turn loop
/// can fan ready tools out in order, gate approval-required tools behind
/// an explicit user decision, and short-circuit policy errors straight
/// into a `tool_result` message without invoking the tool at all.
pub(super) enum PlannedToolCall {
    Ready {
        id: String,
        name: String,
        arguments: String,
    },
    NeedsApproval {
        id: String,
        name: String,
        arguments: String,
        request: ToolApprovalRequest,
    },
    PolicyError {
        id: String,
        error: String,
    },
}

pub(super) async fn execute_and_commit<S: SessionStore>(
    tools: &ToolRegistry,
    store: &S,
    session: &mut Session,
    on_event: &mut (impl FnMut(TurnEvent<'_>) + Send),
    id: String,
    name: String,
    arguments: String,
) -> Result<()> {
    let (content, is_error) = match tools.execute(&name, &arguments).await {
        Ok(out) => (out, false),
        Err(e) => (format!("{e:#}"), true),
    };
    commit_tool_result(
        store,
        session,
        on_event,
        Message::tool_result(id, content, is_error),
    )
    .await
}

pub(super) async fn commit_tool_result<S: SessionStore>(
    store: &S,
    session: &mut Session,
    on_event: &mut (impl FnMut(TurnEvent<'_>) + Send),
    tool_msg: Message,
) -> Result<()> {
    session.push_message(tool_msg);
    on_event(TurnEvent::MessageAppended(
        session.messages.last().expect("just pushed"),
    ));
    store.save(session).await?;
    Ok(())
}
