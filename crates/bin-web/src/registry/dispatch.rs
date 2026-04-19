use domain::SessionId;
use protocol::AgentCommand;

use super::SessionRegistry;
use crate::session::SendOutcome;

/// Errors [`SessionRegistry::send_command`] maps to HTTP status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandDispatch {
    Ok,
    NotFound,
    Dead,
    AlreadyTurning,
    Closing,
}

impl SessionRegistry {
    /// Dispatch a command to a session by id. Maps `SendOutcome`
    /// onto the richer `CommandDispatch` so the handler has all four
    /// HTTP paths (204/404/410/409) in one match.
    pub async fn send_command(&self, id: SessionId, cmd: AgentCommand) -> CommandDispatch {
        let session = match self.get(id) {
            Some(s) => s,
            None => return CommandDispatch::NotFound,
        };
        match cmd {
            AgentCommand::SendMessage { input } => match session.send_message(input) {
                SendOutcome::Ok => CommandDispatch::Ok,
                SendOutcome::Dead => CommandDispatch::Dead,
                SendOutcome::AlreadyTurning => CommandDispatch::AlreadyTurning,
                SendOutcome::Closing => CommandDispatch::Closing,
            },
            AgentCommand::Cancel => {
                session.cancel();
                CommandDispatch::Ok
            }
            // `AgentCommand` is `#[non_exhaustive]`; accept unknown
            // variants as 204 on the theory that the handler shouldn't
            // have routed them here. Callers use the narrow, typed
            // `send_message` / `cancel` methods instead.
            _ => CommandDispatch::Ok,
        }
    }

    pub async fn resolve_tool_approval(
        &self,
        id: SessionId,
        request_id: String,
        approved: bool,
    ) -> CommandDispatch {
        let session = match self.get(id) {
            Some(s) => s,
            None => return CommandDispatch::NotFound,
        };
        match session.resolve_tool_approval(request_id, approved) {
            SendOutcome::Ok => CommandDispatch::Ok,
            SendOutcome::Dead => CommandDispatch::Dead,
            SendOutcome::AlreadyTurning => CommandDispatch::Ok,
            SendOutcome::Closing => CommandDispatch::Closing,
        }
    }
}
