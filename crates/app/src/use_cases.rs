use std::path::PathBuf;

use anyhow::{Result, bail};
use domain::{Message, Session, SessionId, StreamEvent};
use futures::StreamExt;

use crate::ports::{LlmProvider, SessionStore};
use crate::stream::StreamAccumulator;
use crate::tools::{ToolRegistry, extract_tool_calls};

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
}

impl<L: LlmProvider, S: SessionStore> SessionRunner<L, S> {
    pub fn new(llm: L, store: S, tools: ToolRegistry) -> Self {
        Self { llm, store, tools }
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
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<SessionId> {
        let mut session = Session::new(id, workspace_root);
        self.run_turn(&mut session, input, on_event).await?;
        Ok(id)
    }

    /// Load an existing session and run the next turn.
    pub async fn resume(
        &self,
        id: SessionId,
        input: &str,
        on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<()> {
        let mut session = self.store.load(id).await?;
        self.run_turn(&mut session, input, on_event).await?;
        Ok(())
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
        mut on_event: impl FnMut(TurnEvent<'_>) + Send,
    ) -> Result<()> {
        // Commit the user message before any network work — its presence in
        // `session.messages` is what keeps the final persisted session valid
        // if a later error aborts the turn.
        session.push_message(Message::user(input));
        on_event(TurnEvent::MessageAppended(
            session.messages.last().expect("just pushed"),
        ));
        self.store.save(session).await?;

        for _ in 0..MAX_LOOP_ITERATIONS {
            // Expose tool schemas only if we actually have tools registered
            // — keeps the wire payload minimal for tool-free compositions.
            let defs = self.tools.defs();

            let mut event_stream = self.llm.stream(&session.messages, &defs).await?;
            let mut acc = StreamAccumulator::new();
            while let Some(event) = event_stream.next().await {
                let event = event?;
                on_event(TurnEvent::StreamDelta(&event));
                acc.push(event);
            }
            let response = acc.into_message();
            session.push_message(response.clone());
            on_event(TurnEvent::MessageAppended(
                session.messages.last().expect("just pushed"),
            ));
            self.store.save(session).await?;

            let tool_calls = extract_tool_calls(&response);
            if tool_calls.is_empty() {
                return Ok(());
            }

            // If the model is asking for a tool but we have no registry, we
            // can't make progress. Return an error so the caller can surface
            // the misconfiguration rather than silently looping.
            if self.tools.is_empty() {
                bail!("model returned tool calls but no tools are registered");
            }

            for (id, name, arguments) in tool_calls {
                let (content, is_error) = match self.tools.execute(&name, &arguments).await {
                    Ok(out) => (out, false),
                    Err(e) => (format!("{e:#}"), true),
                };
                let tool_msg = Message::tool_result(id, content, is_error);
                session.push_message(tool_msg);
                on_event(TurnEvent::MessageAppended(
                    session.messages.last().expect("just pushed"),
                ));
                self.store.save(session).await?;
            }
        }

        bail!("tool-call loop exceeded {MAX_LOOP_ITERATIONS} iterations — model is not converging");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use domain::{ContentBlock, Role};

    use super::*;
    use crate::Tool;
    use crate::fake::{FakeLlmProvider, FakeSessionStore, FakeTool, tool_registry_with};
    use domain::Usage;

    fn make_runner(
        llm: FakeLlmProvider,
        store: FakeSessionStore,
        tools: ToolRegistry,
    ) -> SessionRunner<FakeLlmProvider, FakeSessionStore> {
        SessionRunner::new(llm, store, tools)
    }

    // -- start --

    #[tokio::test]
    async fn start_returns_id_and_persists_session() {
        let llm = FakeLlmProvider::new();
        llm.push_text("Hello back!");
        let store = FakeSessionStore::new();

        let runner = make_runner(llm, store, ToolRegistry::new());
        let id = SessionId::new_v4();
        let returned_id = runner
            .start(id, "/project".into(), "Hello", |_| {})
            .await
            .unwrap();
        assert_eq!(returned_id, id);

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
            .start(id, "/project".into(), "hi", |evt| {
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
        let mut existing = Session::new(id, "/project".into());
        existing.push_message(Message::user("turn 1"));
        existing.push_message(Message::assistant(vec![ContentBlock::Text {
            text: "response 1".into(),
        }]));
        store.insert(existing);

        let runner = make_runner(llm, store, ToolRegistry::new());
        runner.resume(id, "turn 2", |_| {}).await.unwrap();

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

        let result = runner.resume(SessionId::new_v4(), "hi", |_| {}).await;
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
            .start(id, "/project".into(), "hi", |evt| {
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
            .start(id, "/p".into(), "hi", |evt| {
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

        let result = runner.start(id, "/p".into(), "hi", |_| {}).await;
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

        let result = runner.start(id, "/p".into(), "hi", |_| {}).await;
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
            .start(id, "/project".into(), "hi", |evt| {
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
            .start(SessionId::new_v4(), "/p".into(), "hi", |_| {})
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
        runner.start(id, "/p".into(), "hi", |_| {}).await.unwrap();

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
        runner.start(id, "/p".into(), "hi", |_| {}).await.unwrap();

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
        runner.start(id, "/p".into(), "hi", |_| {}).await.unwrap();

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
        runner.start(id, "/p".into(), "hi", |_| {}).await.unwrap();

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
            .start(SessionId::new_v4(), "/p".into(), "hi", |_| {})
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
            .start(SessionId::new_v4(), "/p".into(), "hi", |evt| {
                if let TurnEvent::MessageAppended(m) = evt {
                    appended_roles.push(m.role.clone());
                }
            })
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
            .start(SessionId::new_v4(), "/p".into(), "hi", |_| {})
            .await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("iterations"),
            "iteration-cap error should mention iterations"
        );
    }
}
