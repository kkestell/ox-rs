use std::path::PathBuf;

use std::collections::HashMap;

use anyhow::{Result, bail};
use domain::{Message, Session, SessionId, StreamAccumulator, StreamEvent};
use futures::StreamExt;

use crate::approval::{
    ApprovalRequirement, NoApprovalRequired, TOOL_REJECTED_MESSAGE, ToolApprovalRequest,
    ToolApprover,
};
use crate::cancel::CancelToken;
use crate::ports::{LlmProvider, SessionStore};
use crate::tools::{ToolRegistry, extract_tool_calls};

/// Result of a completed turn — either it ran to natural completion or was
/// cancelled cooperatively via a `CancelToken`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Completed,
    Cancelled,
}

/// Events surfaced to the caller during a turn.
///
/// `StreamDelta` forwards raw in-flight stream events (token/reasoning/tool-
/// call argument chunks). `MessageAppended` fires for every message that is
/// committed to the session — the initial user input, each intermediate
/// assistant reply and tool-result in a multi-step tool loop, and the final
/// assistant reply.
///
/// The callback sees a `MessageAppended` *after* the message has been pushed
/// to `session.messages`, so UI layers can treat it as "this is now part of
/// history." That keeps the live-streaming view (driven by `StreamDelta`s
/// into a `StreamAccumulator`) and the committed view strictly separated:
/// the accumulator renders in-progress state, and once the message is
/// committed via `MessageAppended`, the accumulator is discarded.
#[derive(Debug)]
pub enum TurnEvent<'a> {
    StreamDelta(&'a StreamEvent),
    MessageAppended(&'a Message),
    ToolApprovalRequested { requests: Vec<ToolApprovalRequest> },
    ToolApprovalResolved { request_id: String, approved: bool },
}

/// Cap on tool-call loop iterations per turn.
///
/// A misbehaving model can chain tool calls indefinitely — e.g. read-edit-
/// read-edit-… without converging. 25 is generous enough for real work
/// (open file, grep, edit, read diff, edit again) while short enough that
/// a runaway turn surfaces as an error rather than burning tokens forever.
const MAX_LOOP_ITERATIONS: usize = 25;

/// Runs conversation turns against an LLM, handling session creation, message
/// accumulation, persistence, and the tool-call loop. Exposes two entry points:
///
/// - `start` — creates a brand-new session and sends the first user message.
/// - `resume` — loads an existing session and appends a new turn.
///
/// Both delegate to the private `run_turn` method so the streaming +
/// tool-execution loop lives in exactly one place.
pub struct SessionRunner<L, S> {
    llm: L,
    store: S,
    tools: ToolRegistry,
    system_prompt: String,
}

impl<L: LlmProvider, S: SessionStore> SessionRunner<L, S> {
    pub fn new(llm: L, store: S, tools: ToolRegistry, system_prompt: String) -> Self {
        Self {
            llm,
            store,
            tools,
            system_prompt,
        }
    }

    /// Load a session's messages by id. Used by the agent driver to replay
    /// history on `--resume` without holding a second `SessionStore` handle.
    pub async fn load_history(&self, id: SessionId) -> Result<Vec<Message>> {
        match self.store.try_load(id).await? {
            Some(session) => Ok(session.messages),
            None => bail!("session {id} not found"),
        }
    }

    /// Create a new session and run the first turn. The caller is responsible
    /// for generating the `SessionId` — this keeps the app layer pure and
    /// deterministic.
    ///
    /// The `on_event` callback fires for each `TurnEvent`; see `run_turn`.
    pub async fn start(
        &self,
        id: SessionId,
        workspace_root: PathBuf,
        input: &str,
        cancel: CancelToken,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        // The agent's `workspace_root` CLI argument is its CWD, which is
        // the session's worktree path for worktree-backed sessions. The
        // agent has no separate knowledge of the main repository root —
        // only the host (which tracks it via `WorkspaceContext`) does — so
        // both `Session` fields are populated from the same value here.
        let mut session = Session::new(id, workspace_root.clone(), workspace_root);
        self.run_turn(&mut session, input, cancel, on_event).await
    }

    pub async fn start_with_approver(
        &self,
        id: SessionId,
        workspace_root: PathBuf,
        input: &str,
        cancel: CancelToken,
        approver: &dyn ToolApprover,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        let mut session = Session::new(id, workspace_root.clone(), workspace_root);
        self.run_turn_with_approver(&mut session, input, cancel, approver, on_event)
            .await
    }

    /// Load an existing session and run the next turn.
    pub async fn resume(
        &self,
        id: SessionId,
        input: &str,
        cancel: CancelToken,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        let Some(mut session) = self.store.try_load(id).await? else {
            bail!("session {id} not found");
        };
        self.run_turn(&mut session, input, cancel, on_event).await
    }

    pub async fn resume_with_approver(
        &self,
        id: SessionId,
        input: &str,
        cancel: CancelToken,
        approver: &dyn ToolApprover,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        let Some(mut session) = self.store.try_load(id).await? else {
            bail!("session {id} not found");
        };
        self.run_turn_with_approver(&mut session, input, cancel, approver, on_event)
            .await
    }

    /// Core turn loop. Appends the user input, then iteratively:
    ///
    /// 1. Streams an assistant response, surfacing each event as a
    ///    `StreamDelta`.
    /// 2. Accumulates the response into a `Message`, commits it to the
    ///    session, and fires a `MessageAppended`.
    /// 3. If the response includes tool calls, executes each in order,
    ///    appending a `Role::Tool` message per result, then loops.
    /// 4. Otherwise returns.
    ///
    /// Tool errors (including unknown-tool and JSON-parse failures) become
    /// `ToolResult { is_error: true }` messages rather than bubbling up;
    /// the LLM then gets to see the error and try again. Only *structural*
    /// failures (LLM stream error, storage failure, iteration cap) propagate.
    ///
    /// The session is persisted after every message append so that a killed
    /// agent never loses work that was already streamed to the GUI.
    async fn run_turn(
        &self,
        session: &mut Session,
        input: &str,
        cancel: CancelToken,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        let approver = NoApprovalRequired;
        self.run_turn_with_approver(session, input, cancel, &approver, on_event)
            .await
    }

    async fn run_turn_with_approver(
        &self,
        session: &mut Session,
        input: &str,
        cancel: CancelToken,
        approver: &dyn ToolApprover,
        mut on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<TurnOutcome> {
        // Commit the user message before any network work — its presence in
        // `session.messages` is what keeps the final persisted session valid
        // if a later error aborts the turn.
        session.push_message(Message::user(input));
        on_event(TurnEvent::MessageAppended(
            session.messages.last().expect("just pushed"),
        ));
        self.store.save(session).await?;

        let defs = self.tools.defs();

        for _ in 0..MAX_LOOP_ITERATIONS {
            // Check cancellation before starting a new LLM stream. This
            // catches cancels that arrived during tool execution or between
            // iterations.
            if cancel.is_cancelled() {
                return Ok(TurnOutcome::Cancelled);
            }

            let mut event_stream = self
                .llm
                .stream(&session.messages, &self.system_prompt, defs)
                .await?;
            let mut acc = StreamAccumulator::new();
            while let Some(event) = event_stream.next().await {
                let event = event?;
                on_event(TurnEvent::StreamDelta(&event));
                acc.push(event);

                if cancel.is_cancelled() {
                    // Commit whatever the accumulator has collected so far.
                    // Skip the commit if nothing meaningful arrived — avoids
                    // polluting the session with a zero-content assistant
                    // message.
                    let partial = acc.into_message();
                    if !partial.content.is_empty() {
                        session.push_message(partial);
                        on_event(TurnEvent::MessageAppended(
                            session.messages.last().expect("just pushed"),
                        ));
                        self.store.save(session).await?;
                    }
                    return Ok(TurnOutcome::Cancelled);
                }
            }
            session.push_message(acc.into_message());
            let response = session.messages.last().expect("just pushed");
            on_event(TurnEvent::MessageAppended(response));
            let tool_calls = extract_tool_calls(response);
            self.store.save(session).await?;

            if tool_calls.is_empty() {
                return Ok(TurnOutcome::Completed);
            }

            // If the model is asking for a tool but we have no registry, we
            // can't make progress. Return an error so the caller can surface
            // the misconfiguration rather than silently looping.
            if self.tools.is_empty() {
                bail!("model returned tool calls but no tools are registered");
            }

            let approval_plan = self.plan_tool_approvals(tool_calls).await;
            if cancel.is_cancelled() {
                return Ok(TurnOutcome::Cancelled);
            }

            // Split the plan into entries that need a user decision and
            // entries that can be committed immediately (Ready tools + policy
            // errors). Ready tools run in plan order. Approval-gated tools
            // execute in *decision arrival order* so a user approving one
            // doesn't block on a different tool still waiting for input.
            let mut approval_requests: Vec<ToolApprovalRequest> = Vec::new();
            let mut pending: HashMap<String, (String, String, String)> = HashMap::new();

            for planned in approval_plan {
                if cancel.is_cancelled() {
                    return Ok(TurnOutcome::Cancelled);
                }
                match planned {
                    PlannedToolCall::Ready {
                        id,
                        name,
                        arguments,
                    } => {
                        execute_and_commit(
                            &self.tools,
                            &self.store,
                            session,
                            &mut on_event,
                            id,
                            name,
                            arguments,
                        )
                        .await?;
                    }
                    PlannedToolCall::PolicyError { id, error } => {
                        commit_tool_result(
                            &self.store,
                            session,
                            &mut on_event,
                            Message::tool_result(id, error, true),
                        )
                        .await?;
                    }
                    PlannedToolCall::NeedsApproval {
                        id,
                        name,
                        arguments,
                        request,
                    } => {
                        pending.insert(request.request_id.clone(), (id, name, arguments));
                        approval_requests.push(request);
                    }
                }
            }

            if !approval_requests.is_empty() {
                on_event(TurnEvent::ToolApprovalRequested {
                    requests: approval_requests.clone(),
                });
                let mut stream = approver.approve(approval_requests, cancel.clone());
                while let Some(decision) = stream.next().await {
                    let decision = decision?;
                    on_event(TurnEvent::ToolApprovalResolved {
                        request_id: decision.request_id.clone(),
                        approved: decision.approved,
                    });
                    let Some((id, name, arguments)) = pending.remove(&decision.request_id) else {
                        continue;
                    };
                    if cancel.is_cancelled() {
                        return Ok(TurnOutcome::Cancelled);
                    }
                    if decision.approved {
                        execute_and_commit(
                            &self.tools,
                            &self.store,
                            session,
                            &mut on_event,
                            id,
                            name,
                            arguments,
                        )
                        .await?;
                    } else {
                        commit_tool_result(
                            &self.store,
                            session,
                            &mut on_event,
                            Message::tool_result(id, TOOL_REJECTED_MESSAGE, true),
                        )
                        .await?;
                    }
                }
                drop(stream);

                // If the approver ended the stream without decisions for
                // every pending request — typically because cancellation
                // fired — treat the turn as cancelled so no tool runs
                // without an explicit user choice.
                if !pending.is_empty() {
                    return Ok(TurnOutcome::Cancelled);
                }
            }
        }

        bail!("tool-call loop exceeded {MAX_LOOP_ITERATIONS} iterations — model is not converging");
    }

    async fn plan_tool_approvals(
        &self,
        tool_calls: Vec<(String, String, String)>,
    ) -> Vec<PlannedToolCall> {
        let mut planned = Vec::with_capacity(tool_calls.len());
        for (id, name, arguments) in tool_calls {
            match self.tools.approval_requirement(&name, &arguments).await {
                Ok(ApprovalRequirement::NotRequired) => planned.push(PlannedToolCall::Ready {
                    id,
                    name,
                    arguments,
                }),
                Ok(ApprovalRequirement::Required { reason }) => {
                    let request = ToolApprovalRequest {
                        request_id: id.clone(),
                        tool_call_id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        reason,
                    };
                    planned.push(PlannedToolCall::NeedsApproval {
                        id,
                        name,
                        arguments,
                        request,
                    });
                }
                Err(err) => planned.push(PlannedToolCall::PolicyError {
                    id,
                    error: format!("{err:#}"),
                }),
            }
        }
        planned
    }
}

enum PlannedToolCall {
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

async fn execute_and_commit<S: SessionStore>(
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

async fn commit_tool_result<S: SessionStore>(
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

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use domain::{ContentBlock, Role};

    use super::*;
    use crate::approval::ToolApprovalDecision;
    use crate::cancel::CancelToken;
    use crate::fake::{FakeLlmProvider, FakeSessionStore, FakeTool, tool_registry_with};
    use crate::{Tool, ToolDef};
    use domain::Usage;

    fn make_runner(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
        tools: ToolRegistry,
    ) -> SessionRunner<FakeLlmProvider, FakeSessionStore> {
        SessionRunner::new(llm, store, tools, String::new())
    }

    struct ApprovalFakeTool {
        name: String,
        output: String,
        calls: Mutex<Vec<String>>,
    }

    impl ApprovalFakeTool {
        fn new(name: &str, output: &str) -> Self {
            Self {
                name: name.into(),
                output: output.into(),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Tool for ApprovalFakeTool {
        fn def(&self) -> ToolDef {
            ToolDef {
                name: self.name.clone(),
                description: "approval fake".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        fn approval_requirement<'a>(
            &'a self,
            _args: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<ApprovalRequirement>> + Send + 'a>> {
            Box::pin(async {
                Ok(ApprovalRequirement::Required {
                    reason: "test approval required".into(),
                })
            })
        }

        fn execute<'a>(
            &'a self,
            args: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(args.to_owned());
                Ok(self.output.clone())
            })
        }
    }

    struct ScriptedApprover {
        decisions: HashMap<String, bool>,
        batches: Mutex<Vec<Vec<ToolApprovalRequest>>>,
    }

    impl ScriptedApprover {
        fn new(decisions: impl IntoIterator<Item = (&'static str, bool)>) -> Self {
            Self {
                decisions: decisions
                    .into_iter()
                    .map(|(id, approved)| (id.to_owned(), approved))
                    .collect(),
                batches: Mutex::new(Vec::new()),
            }
        }

        fn batches(&self) -> Vec<Vec<ToolApprovalRequest>> {
            self.batches.lock().unwrap().clone()
        }
    }

    impl ToolApprover for ScriptedApprover {
        fn approve(
            &self,
            requests: Vec<ToolApprovalRequest>,
            _cancel: CancelToken,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<ToolApprovalDecision>> + Send + '_>>
        {
            self.batches.lock().unwrap().push(requests.clone());
            let decisions: Vec<_> = requests
                .into_iter()
                .map(|request| {
                    Ok(ToolApprovalDecision {
                        approved: self
                            .decisions
                            .get(&request.request_id)
                            .copied()
                            .unwrap_or(false),
                        request_id: request.request_id,
                    })
                })
                .collect();
            Box::pin(futures::stream::iter(decisions))
        }
    }

    // -- start --

    #[tokio::test]
    async fn start_returns_id_and_persists_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();
        let outcome = runner
            .start(id, "/project".into(), "Hello", CancelToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(outcome, TurnOutcome::Completed);

        let saved = runner.store.get(id).expect("session saved");
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.messages[1].role, Role::Assistant);
        assert_eq!(saved.messages[1].text(), "Hello back!");
    }

    #[tokio::test]
    async fn start_invokes_message_appended_for_user_and_assistant() {
        let llm = FakeLlmProvider::new();
        llm.push_text("reply");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();

        let mut appended_texts = Vec::new();
        runner
            .start(id, "/project".into(), "hi", CancelToken::new(), |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_texts.push((m.role.clone(), m.text()));
                }
            })
            .await
            .unwrap();

        assert_eq!(
            appended_texts,
            vec![
                (Role::User, "hi".to_owned()),
                (Role::Assistant, "reply".to_owned()),
            ]
        );
    }

    // -- resume --

    #[tokio::test]
    async fn resume_loads_and_appends_messages() {
        let llm = FakeLlmProvider::new();
        llm.push_text("continued");
        let store = FakeSessionStore::new();

        let id = SessionId::new_v4();
        let mut existing = Session::new(id, "/project".into(), "/project".into());
        existing.push_message(Message::user("turn 1"));
        existing.push_message(Message::assistant(vec![ContentBlock::Text {
            text: "response 1".into(),
        }]));
        store.insert(existing);

        let runner = make_runner(llm, store, ToolRegistry::new());
        runner
            .resume(id, "turn 2", CancelToken::new(), |_| {})
            .await
            .unwrap();

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 4);
        assert_eq!(saved.messages[2].text(), "turn 2");
        assert_eq!(saved.messages[3].text(), "continued");
    }

    #[tokio::test]
    async fn resume_nonexistent_session_returns_error() {
        let llm = FakeLlmProvider::new();
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());

        let result = runner
            .resume(SessionId::new_v4(), "hi", CancelToken::new(), |_| {})
            .await;
        assert!(result.is_err());
    }

    // -- streaming callback --

    #[tokio::test]
    async fn start_surfaces_stream_deltas_in_arrival_order() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::TextDelta {
                delta: "Hello, ".into(),
            },
            StreamEvent::TextDelta {
                delta: "world!".into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();
        let mut deltas: Vec<StreamEvent> = Vec::new();
        runner
            .start(id, "/project".into(), "hi", CancelToken::new(), |evt| {
                if let TurnEvent::StreamDelta(e) = evt {
                    deltas.push(e.clone());
                }
            })
            .await
            .unwrap();

        assert_eq!(deltas.len(), 3);
        assert!(matches!(&deltas[0], StreamEvent::TextDelta { delta: s } if s == "Hello, "));
        assert!(matches!(&deltas[1], StreamEvent::TextDelta { delta: s } if s == "world!"));
        assert!(matches!(&deltas[2], StreamEvent::Finished { .. }));
    }

    // -- connection-time error --

    #[tokio::test]
    async fn connection_time_error_propagates_before_any_stream_deltas() {
        // A failure at `llm.stream(...).await?` must propagate through
        // `run_turn` as an error, and no StreamDelta events should fire.
        // The user message is still committed — that ordering keeps the
        // persisted session consistent even if the caller retries.
        let llm = FakeLlmProvider::new();
        llm.push_error("dns failure");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();

        let mut deltas: Vec<StreamEvent> = Vec::new();
        let result = runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |evt| {
                if let TurnEvent::StreamDelta(e) = evt {
                    deltas.push(e.clone());
                }
            })
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dns failure"));
        assert!(deltas.is_empty());
    }

    // -- incremental persistence --

    #[tokio::test]
    async fn user_message_persisted_even_when_llm_fails() {
        // Before this fix, a connection error meant the session was never
        // saved — the user message was lost entirely. Now save() fires
        // after each push_message, so the user message survives.
        let llm = FakeLlmProvider::new();
        llm.push_error("connection refused");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();

        let result = runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await;
        assert!(result.is_err());

        // The session should be on disk with the user message.
        let saved = runner.store.get(id).expect("session should be persisted");
        assert_eq!(saved.messages.len(), 1);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[0].text(), "hi");
    }

    #[tokio::test]
    async fn partial_tool_loop_persisted_on_error() {
        // If the LLM fails on the second iteration (after one tool call
        // completed), the session should contain everything up to the
        // failure point: user + assistant(tool_call) + tool_result.
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "t", "{}");
        llm.push_error("server error");

        let t = Arc::new(FakeTool::new("t"));
        t.push_ok("tool-output");
        let tools = tool_registry_with(vec![t as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();

        let result = runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await;
        assert!(result.is_err());

        let saved = runner.store.get(id).expect("session should be persisted");
        assert_eq!(saved.messages.len(), 3);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[1].role, Role::Assistant);
        assert_eq!(saved.messages[2].role, Role::Tool);
    }

    // -- mid-stream error --

    #[tokio::test]
    async fn mid_stream_error_aborts_turn_but_preserves_earlier_events() {
        let llm = FakeLlmProvider::new();
        llm.push_error_after(
            vec![StreamEvent::TextDelta {
                delta: "partial ".into(),
            }],
            "connection lost",
        );
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();
        let mut deltas: Vec<StreamEvent> = Vec::new();
        let result = runner
            .start(id, "/project".into(), "hi", CancelToken::new(), |evt| {
                if let TurnEvent::StreamDelta(e) = evt {
                    deltas.push(e.clone());
                }
            })
            .await;

        assert_eq!(deltas.len(), 1);
        assert!(matches!(&deltas[0], StreamEvent::TextDelta { delta: s } if s == "partial "));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("connection lost"));
    }

    // -- tool loop --

    #[tokio::test]
    async fn no_tool_calls_returns_after_one_iteration() {
        // Parity with pre-tool behavior: a text-only response exits the
        // loop after the first iteration.
        let llm = FakeLlmProvider::new();
        llm.push_text("just text");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());

        runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |_| {},
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn single_tool_call_executes_and_loops_for_final_reply() {
        let llm = FakeLlmProvider::new();
        // Iteration 1: model calls a tool.
        llm.push_tool_call("call_1", "echo", r#"{"x":1}"#);
        // Iteration 2: model produces the final text after seeing the tool
        // result.
        llm.push_text("all done");

        let echo = Arc::new(FakeTool::new("echo"));
        echo.push_ok("tool-output");
        let tools = tool_registry_with(vec![echo.clone() as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await
            .unwrap();

        // Session: user + asst(tool_call) + tool_result + asst("all done") = 4.
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 4, "{:?}", saved.messages);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[1].role, Role::Assistant);
        assert_eq!(saved.messages[2].role, Role::Tool);
        assert_eq!(saved.messages[3].role, Role::Assistant);
        assert_eq!(saved.messages[3].text(), "all done");

        // Tool result message carried the tool's output and is_error=false.
        match &saved.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert_eq!(content, "tool-output");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult block"),
        }

        // Tool saw the arguments verbatim.
        assert_eq!(echo.calls(), vec![r#"{"x":1}"#.to_owned()]);
    }

    #[tokio::test]
    async fn parallel_tool_calls_all_execute_in_order() {
        let llm = FakeLlmProvider::new();
        // Iteration 1: two parallel tool calls.
        llm.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "a".into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c2".into(),
                name: "b".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: "{}".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 1,
                delta: "{}".into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        // Iteration 2: final response.
        llm.push_text("finished");

        let a = Arc::new(FakeTool::new("a"));
        a.push_ok("from-a");
        let b = Arc::new(FakeTool::new("b"));
        b.push_ok("from-b");
        let tools =
            tool_registry_with(vec![a.clone() as Arc<dyn Tool>, b.clone() as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await
            .unwrap();

        // user + asst(2 tool calls) + tool_result_a + tool_result_b + asst("finished") = 5.
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 5);
        assert_eq!(saved.messages[2].role, Role::Tool);
        assert_eq!(saved.messages[3].role, Role::Tool);
        match (&saved.messages[2].content[0], &saved.messages[3].content[0]) {
            (
                ContentBlock::ToolResult {
                    content: c1,
                    tool_call_id: id1,
                    ..
                },
                ContentBlock::ToolResult {
                    content: c2,
                    tool_call_id: id2,
                    ..
                },
            ) => {
                assert_eq!(c1, "from-a");
                assert_eq!(id1, "c1");
                assert_eq!(c2, "from-b");
                assert_eq!(id2, "c2");
            }
            _ => panic!("expected two ToolResult blocks"),
        }
    }

    #[tokio::test]
    async fn approval_required_calls_are_batched_and_results_keep_original_order() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "a".into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c2".into(),
                name: "b".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: r#"{"first":true}"#.into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 1,
                delta: r#"{"second":true}"#.into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        llm.push_text("finished");

        let a = Arc::new(ApprovalFakeTool::new("a", "from-a"));
        let b = Arc::new(ApprovalFakeTool::new("b", "from-b"));
        let tools =
            tool_registry_with(vec![a.clone() as Arc<dyn Tool>, b.clone() as Arc<dyn Tool>]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let approver = ScriptedApprover::new([("c1", true), ("c2", true)]);
        let id = SessionId::new_v4();
        let mut approval_events = Vec::new();

        runner
            .start_with_approver(
                id,
                "/p".into(),
                "hi",
                CancelToken::new(),
                &approver,
                |evt| match evt {
                    TurnEvent::ToolApprovalRequested { requests } => {
                        approval_events.push(requests.len())
                    }
                    TurnEvent::ToolApprovalResolved {
                        request_id,
                        approved,
                    } => {
                        assert!(approved, "expected approved decision for {request_id}");
                    }
                    _ => {}
                },
            )
            .await
            .unwrap();

        assert_eq!(approval_events, vec![2]);
        let batches = approver.batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0]
                .iter()
                .map(|r| r.request_id.as_str())
                .collect::<Vec<_>>(),
            vec!["c1", "c2"]
        );

        let saved = runner.store.get(id).unwrap();
        match (&saved.messages[2].content[0], &saved.messages[3].content[0]) {
            (
                ContentBlock::ToolResult {
                    content: c1,
                    tool_call_id: id1,
                    ..
                },
                ContentBlock::ToolResult {
                    content: c2,
                    tool_call_id: id2,
                    ..
                },
            ) => {
                assert_eq!(id1, "c1");
                assert_eq!(c1, "from-a");
                assert_eq!(id2, "c2");
                assert_eq!(c2, "from-b");
            }
            _ => panic!("expected tool results"),
        }
        assert_eq!(a.calls(), vec![r#"{"first":true}"#.to_owned()]);
        assert_eq!(b.calls(), vec![r#"{"second":true}"#.to_owned()]);
    }

    #[tokio::test]
    async fn rejected_approval_appends_error_result_and_turn_continues() {
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "danger", "{}");
        llm.push_text("recovered");

        let danger = Arc::new(ApprovalFakeTool::new("danger", "should not run"));
        let tools = tool_registry_with(vec![danger.clone() as Arc<dyn Tool>]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let approver = ScriptedApprover::new([("c1", false)]);
        let id = SessionId::new_v4();

        runner
            .start_with_approver(id, "/p".into(), "hi", CancelToken::new(), &approver, |_| {})
            .await
            .unwrap();

        assert!(danger.calls().is_empty());
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages[3].text(), "recovered");
        match &saved.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert_eq!(content, TOOL_REJECTED_MESSAGE);
                assert!(is_error);
            }
            _ => panic!("expected rejected tool result"),
        }
    }

    #[tokio::test]
    async fn cancel_while_waiting_for_approval_returns_cancelled_without_executing() {
        use std::sync::Arc;
        use tokio::sync::mpsc;

        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "danger", "{}");

        let danger = Arc::new(ApprovalFakeTool::new("danger", "should not run"));
        let tools = tool_registry_with(vec![danger.clone() as Arc<dyn Tool>]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);

        struct BlockingApprover {
            tx: mpsc::Sender<Vec<ToolApprovalRequest>>,
        }
        impl ToolApprover for BlockingApprover {
            fn approve(
                &self,
                requests: Vec<ToolApprovalRequest>,
                cancel: CancelToken,
            ) -> Pin<Box<dyn futures::Stream<Item = Result<ToolApprovalDecision>> + Send + '_>>
            {
                let tx = self.tx.clone();
                Box::pin(futures::stream::unfold(
                    (tx, cancel, Some(requests)),
                    |(tx, cancel, requests)| async move {
                        if let Some(pending) = requests {
                            let _ = tx.send(pending).await;
                        }
                        loop {
                            if cancel.is_cancelled() {
                                return None;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                    },
                ))
            }
        }

        let (tx, mut rx) = mpsc::channel(1);
        let approver = BlockingApprover { tx };
        let token = CancelToken::new();
        let token_clone = token.clone();
        let id = SessionId::new_v4();

        let handle = tokio::spawn(async move {
            runner
                .start_with_approver(id, "/p".into(), "hi", token_clone, &approver, |_| {})
                .await
        });

        let requests = rx.recv().await.expect("approver should receive requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].request_id, "c1");

        token.cancel();

        let outcome = handle.await.unwrap().unwrap();
        assert_eq!(outcome, TurnOutcome::Cancelled);
        assert!(
            danger.calls().is_empty(),
            "cancelled approval must not execute the tool"
        );
    }

    #[tokio::test]
    async fn mixed_approval_and_not_required_preserves_original_order() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "safe".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: r#"{"x":1}"#.into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c2".into(),
                name: "risky".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 1,
                delta: r#"{"y":2}"#.into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        llm.push_text("done");

        let safe = Arc::new(FakeTool::new("safe"));
        safe.push_ok("safe-result");
        let risky = Arc::new(ApprovalFakeTool::new("risky", "risky-result"));
        let tools = tool_registry_with(vec![
            safe.clone() as Arc<dyn Tool>,
            risky.clone() as Arc<dyn Tool>,
        ]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let approver = ScriptedApprover::new([("c2", true)]);
        let id = SessionId::new_v4();
        let mut approval_event_count = 0;

        runner
            .start_with_approver(
                id,
                "/p".into(),
                "hi",
                CancelToken::new(),
                &approver,
                |evt| {
                    if let TurnEvent::ToolApprovalRequested { requests } = evt {
                        approval_event_count += 1;
                        assert_eq!(requests.len(), 1);
                        assert_eq!(requests[0].request_id, "c2");
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(approval_event_count, 1);

        let saved = runner.store.get(id).unwrap();
        match (&saved.messages[2].content[0], &saved.messages[3].content[0]) {
            (
                ContentBlock::ToolResult {
                    content: c1,
                    tool_call_id: id1,
                    is_error: e1,
                    ..
                },
                ContentBlock::ToolResult {
                    content: c2,
                    tool_call_id: id2,
                    is_error: e2,
                    ..
                },
            ) => {
                assert_eq!(id1, "c1");
                assert_eq!(c1, "safe-result");
                assert!(!e1);
                assert_eq!(id2, "c2");
                assert_eq!(c2, "risky-result");
                assert!(!e2);
            }
            _ => panic!("expected two tool results"),
        }
    }

    #[tokio::test]
    async fn partial_approval_in_batch_executes_approved_and_rejects_rest() {
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "a".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: r#"{"first":true}"#.into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c2".into(),
                name: "b".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 1,
                delta: r#"{"second":true}"#.into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                },
            },
        ]);
        llm.push_text("done");

        let a = Arc::new(ApprovalFakeTool::new("a", "from-a"));
        let b = Arc::new(ApprovalFakeTool::new("b", "from-b"));
        let tools =
            tool_registry_with(vec![a.clone() as Arc<dyn Tool>, b.clone() as Arc<dyn Tool>]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let approver = ScriptedApprover::new([("c1", true), ("c2", false)]);
        let id = SessionId::new_v4();

        runner
            .start_with_approver(id, "/p".into(), "hi", CancelToken::new(), &approver, |_| {})
            .await
            .unwrap();

        assert_eq!(a.calls(), vec![r#"{"first":true}"#.to_owned()]);
        assert!(b.calls().is_empty(), "rejected tool must not execute");

        let saved = runner.store.get(id).unwrap();
        match (&saved.messages[2].content[0], &saved.messages[3].content[0]) {
            (
                ContentBlock::ToolResult {
                    content: c1,
                    tool_call_id: id1,
                    is_error: e1,
                    ..
                },
                ContentBlock::ToolResult {
                    content: c2,
                    tool_call_id: id2,
                    is_error: e2,
                    ..
                },
            ) => {
                assert_eq!(id1, "c1");
                assert_eq!(c1, "from-a");
                assert!(!e1);
                assert_eq!(id2, "c2");
                assert_eq!(c2, TOOL_REJECTED_MESSAGE);
                assert!(e2);
            }
            _ => panic!("expected two tool results in original order"),
        }
    }

    #[tokio::test]
    async fn policy_error_on_approval_requirement_becomes_error_tool_result() {
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "picky", "anything");
        llm.push_text("recovered");

        struct PickyTool;
        impl Tool for PickyTool {
            fn def(&self) -> domain::ToolDef {
                domain::ToolDef {
                    name: "picky".into(),
                    description: "always errors".into(),
                    parameters: serde_json::json!({"type": "object"}),
                }
            }
            fn approval_requirement<'a>(
                &'a self,
                args: &'a str,
            ) -> Pin<Box<dyn Future<Output = Result<ApprovalRequirement>> + Send + 'a>>
            {
                Box::pin(async move { bail!("policy check failed: {args}") })
            }
            fn execute<'a>(
                &'a self,
                _args: &'a str,
            ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
                Box::pin(async { Ok("should not run".into()) })
            }
        }

        let tools = tool_registry_with(vec![Arc::new(PickyTool) as Arc<dyn Tool>]);
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);

        let id = SessionId::new_v4();
        runner
            .start_with_approver(
                id,
                "/p".into(),
                "hi",
                CancelToken::new(),
                &NoApprovalRequired,
                |_| {},
            )
            .await
            .unwrap();

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages[3].text(), "recovered");
        match &saved.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(content.contains("policy check failed"), "got: {content}");
                assert!(is_error);
            }
            _ => panic!("expected policy error tool result"),
        }
    }

    #[tokio::test]
    async fn tool_error_becomes_error_result_and_loop_continues() {
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "bad", "{}");
        llm.push_text("recovered");

        let bad = Arc::new(FakeTool::new("bad"));
        bad.push_err("disk exploded");
        let tools = tool_registry_with(vec![bad.clone() as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await
            .unwrap();

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 4);
        match &saved.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(content.contains("disk exploded"), "got {content}");
                assert!(is_error);
            }
            _ => panic!("expected ToolResult"),
        }
        // Loop continued — final assistant message is present.
        assert_eq!(saved.messages[3].text(), "recovered");
    }

    #[tokio::test]
    async fn unknown_tool_name_becomes_error_result_and_loop_continues() {
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "ghost", "{}");
        llm.push_text("moved on");

        // Register a different tool — 'ghost' isn't in the registry.
        let other = Arc::new(FakeTool::new("real"));
        let tools = tool_registry_with(vec![other as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        runner
            .start(id, "/p".into(), "hi", CancelToken::new(), |_| {})
            .await
            .unwrap();

        let saved = runner.store.get(id).unwrap();
        match &saved.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(content.contains("ghost"), "got {content}");
                assert!(is_error);
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[tokio::test]
    async fn tool_calls_without_registry_returns_error() {
        // If the model tries to call tools but nothing is registered, we
        // can't make progress — returning an error is safer than looping.
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "anything", "{}");

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let result = runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |_| {},
            )
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no tools are registered"),
            "error message should explain the misconfiguration"
        );
    }

    #[tokio::test]
    async fn message_appended_fires_for_every_message_in_order() {
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "t", "{}");
        llm.push_text("final");

        let t = Arc::new(FakeTool::new("t"));
        t.push_ok("ok");
        let tools = tool_registry_with(vec![t as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);

        let mut appended_roles: Vec<Role> = Vec::new();
        runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |evt| {
                    if let TurnEvent::MessageAppended(m) = evt {
                        appended_roles.push(m.role.clone());
                    }
                },
            )
            .await
            .unwrap();

        // user → asst(tool-call) → tool-result → asst(final)
        assert_eq!(
            appended_roles,
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
    }

    #[tokio::test]
    async fn loop_iteration_cap_fires_error() {
        // A model that keeps calling tools forever should hit the iteration
        // cap. Queue exactly `MAX_LOOP_ITERATIONS + 1` tool-call responses
        // so the loop runs the cap and then still has work to do on the
        // next iteration.
        let llm = FakeLlmProvider::new();
        for _ in 0..(super::MAX_LOOP_ITERATIONS + 1) {
            llm.push_tool_call("c", "t", "{}");
        }

        let t = Arc::new(FakeTool::new("t"));
        for _ in 0..(super::MAX_LOOP_ITERATIONS + 1) {
            t.push_ok("ok");
        }
        let tools = tool_registry_with(vec![t as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let result = runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |_| {},
            )
            .await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("iterations"),
            "iteration-cap error should mention iterations"
        );
    }

    // -- system prompt --

    #[tokio::test]
    async fn system_prompt_forwarded_to_llm_provider() {
        let llm = FakeLlmProvider::new();
        llm.push_text("ok");
        let store = FakeSessionStore::new();
        let runner = SessionRunner::new(
            llm,
            store,
            ToolRegistry::new(),
            "You are a coding assistant.".to_owned(),
        );
        runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |_| {},
            )
            .await
            .unwrap();
        assert_eq!(
            runner.llm.system_prompts(),
            vec!["You are a coding assistant."]
        );
    }

    // -- cancellation ---------------------------------------------------------

    #[tokio::test]
    async fn cancel_before_streaming_returns_cancelled_with_no_assistant_message() {
        let llm = FakeLlmProvider::new();
        llm.push_text("should not appear");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();

        // Pre-set the token before the turn starts.
        let token = CancelToken::new();
        token.cancel();

        let mut appended_roles = Vec::new();
        let outcome = runner
            .start(id, "/p".into(), "hi", token, |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_roles.push(m.role.clone());
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, TurnOutcome::Cancelled);
        // Only the user message should be committed — no assistant message.
        assert_eq!(appended_roles, vec![Role::User]);
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 1);
        assert_eq!(saved.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn cancel_mid_stream_commits_partial_assistant_message() {
        use futures::SinkExt;

        let llm = FakeLlmProvider::new();
        let mut tx = llm.push_channel();
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();
        let token = CancelToken::new();
        let token_clone = token.clone();

        let mut appended_roles = Vec::new();
        let handle = tokio::spawn(async move {
            runner
                .start(id, "/p".into(), "hi", token_clone, |evt| {
                    if let TurnEvent::MessageAppended(m) = evt {
                        appended_roles.push(m.role.clone());
                    }
                })
                .await
                .map(|outcome| (outcome, appended_roles, runner))
        });

        // Send one text delta, then cancel.
        tx.send(Ok(StreamEvent::TextDelta {
            delta: "partial".into(),
        }))
        .await
        .unwrap();

        // Small yield to let the runner process the event.
        tokio::task::yield_now().await;
        token.cancel();

        // Send another event so the runner's `next().await` unblocks and
        // sees the cancel flag. The runner checks `is_cancelled()` after
        // each event.
        let _ = tx
            .send(Ok(StreamEvent::TextDelta {
                delta: " ignored".into(),
            }))
            .await;

        let (outcome, appended_roles, runner) = handle.await.unwrap().unwrap();
        assert_eq!(outcome, TurnOutcome::Cancelled);
        // user + partial assistant
        assert_eq!(appended_roles, vec![Role::User, Role::Assistant]);

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[1].role, Role::Assistant);
        // The partial message must contain "partial" (the first delta).
        // It may also contain " ignored" depending on timing, but
        // "partial" must always be present.
        let text = saved.messages[1].text();
        assert!(
            text.contains("partial"),
            "partial assistant message should contain 'partial', got: {text}"
        );
    }

    #[tokio::test]
    async fn cancel_pre_set_skips_streaming_entirely() {
        // Token is set before the turn starts. The check at the top of the
        // loop iteration fires before the LLM stream is opened, so no
        // assistant message is created. This exercises the early exit at
        // the loop-top cancel check — distinct from `cancel_before_streaming`
        // which pre-sets the token before `start()` is even called.
        let llm = FakeLlmProvider::new();
        let token = CancelToken::new();
        token.cancel(); // cancel immediately

        llm.push_text("should not appear");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();

        let mut appended_roles = Vec::new();
        let outcome = runner
            .start(id, "/p".into(), "hi", token, |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_roles.push(m.role.clone());
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, TurnOutcome::Cancelled);
        assert_eq!(appended_roles, vec![Role::User]);
        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 1);
    }

    #[tokio::test]
    async fn cancel_between_tool_calls_skips_remaining_calls() {
        // Two parallel tool calls; cancel after the first executes.
        let llm = FakeLlmProvider::new();
        llm.push_response(vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "t".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 0,
                delta: "{}".into(),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "c2".into(),
                name: "t".into(),
            },
            StreamEvent::ToolCallArgumentDelta {
                index: 1,
                delta: "{}".into(),
            },
            StreamEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                },
            },
        ]);

        let t = Arc::new(FakeTool::new("t"));
        t.push_ok("first-result");
        t.push_ok("should-not-run"); // second call queued but should be skipped
        let tools = tool_registry_with(vec![t.clone() as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        let token = CancelToken::new();
        let token_clone = token.clone();

        let mut appended_roles = Vec::new();
        let outcome = runner
            .start(id, "/p".into(), "hi", token_clone, |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_roles.push(m.role.clone());
                    // Cancel after the first tool result is committed.
                    if m.role == Role::Tool {
                        token.cancel();
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, TurnOutcome::Cancelled);
        // user + assistant(tool calls) + first tool result = 3
        assert_eq!(
            appended_roles,
            vec![Role::User, Role::Assistant, Role::Tool]
        );
        // Only one tool call was executed.
        assert_eq!(t.calls().len(), 1);

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 3);
    }

    #[tokio::test]
    async fn cancel_after_assistant_committed_skips_all_tool_calls() {
        // Edge case: cancel arrives after the assistant message (containing
        // tool calls) is committed but before any tool executes. The cancel
        // check at the top of the tool-execution loop should fire, skipping
        // all tool calls.
        let llm = FakeLlmProvider::new();
        llm.push_tool_call("c1", "t", "{}");

        let t = Arc::new(FakeTool::new("t"));
        t.push_ok("should-not-run");
        let tools = tool_registry_with(vec![t.clone() as Arc<dyn Tool>]);

        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, tools);
        let id = SessionId::new_v4();
        let token = CancelToken::new();
        let token_clone = token.clone();

        let mut appended_roles = Vec::new();
        let outcome = runner
            .start(id, "/p".into(), "hi", token_clone, |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_roles.push(m.role.clone());
                    // Cancel right after the assistant message with tool calls
                    // is committed — before any tool call executes.
                    if m.role == Role::Assistant {
                        token.cancel();
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, TurnOutcome::Cancelled);
        // user + assistant(tool call) = 2, no tool results
        assert_eq!(appended_roles, vec![Role::User, Role::Assistant]);
        // No tool calls were executed.
        assert_eq!(t.calls().len(), 0);

        let saved = runner.store.get(id).unwrap();
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(saved.messages[0].role, Role::User);
        assert_eq!(saved.messages[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn no_cancel_returns_completed() {
        // Regression guard: a turn with no cancellation still returns Completed.
        let llm = FakeLlmProvider::new();
        llm.push_text("done");
        let store = FakeSessionStore::new();
        let runner = make_runner(llm, store, ToolRegistry::new());

        let outcome = runner
            .start(
                SessionId::new_v4(),
                "/p".into(),
                "hi",
                CancelToken::new(),
                |_| {},
            )
            .await
            .unwrap();

        assert_eq!(outcome, TurnOutcome::Completed);
    }
}
