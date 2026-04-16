# Codebase Audit: Hacks, Stubs, and Shortcuts

## Stubs / Unimplemented

1. **`OllamaProvider`** — `crates/adapter-llm/src/ollama.rs:28` — entire `stream()` is `todo!()`. All fields prefixed with `_` to suppress unused warnings.
2. **`BashShell`** — `crates/adapter-fs/src/lib.rs:70-72` — `run()` is `todo!()` with a TODO comment.
3. **No cancel/stop mechanism** — no way to abort an in-progress stream or tool loop. Kill-on-drop of the child process is the only "cancel."
4. **No tool-approval flow** — tools auto-execute; destructive tools have no permission gate.
5. **No session management UI** — sessions can only be resumed via CLI flag, not from within the GUI.

## Data-Loss / Correctness Risks

~~6. **`DiskSessionStore` does blocking I/O on async methods**~~ — _Resolved: `adapter-storage` and `adapter-fs` now use `tokio::fs` throughout; `FileSystem` trait is async._
7. **SSE buffer allocates a full copy on every split** — `crates/adapter-llm/src/openrouter/sse.rs:36-37` — `buffer[..boundary].to_owned()` and `buffer[boundary + 2..].to_owned()` copies the entire remaining buffer on every event. Should drain or use a `VecDeque`/cursor.
8. **`String::from_utf8_lossy` on every SSE chunk** — `sse.rs:29` — produces a `Cow<str>` that then gets `.replace("\r\n", "\n")` and pushed into the buffer. The replacement scans the entire string every time.
~~9. **Tool result message for missing `ToolResult` block silently produces empty strings**~~ — _Resolved: `wire_message` returns `Result`; a `Role::Tool` message without a `ToolResult` block is now a structured error._

## Duplicated / Dead Code

~~10. **`AgentClient::new` and `AgentClient::spawn` duplicate reader/writer setup**~~ — _Resolved in `dee8565`: `spawn` routes through `Self::new`; both share `spawn_reader`/`spawn_writer` helpers._
11. **`DiskSessionStore` created twice in `bin-agent`** — `crates/bin-agent/src/main.rs:109-110` — one `store` for the runner and a separate `history_store` for the driver's preload, both pointing at the same directory. The comment says "stateless beyond its root path" which is true, but it's still a smell — the runner could expose a `store()` accessor instead.
12. **`ToolRegistry::execute` calls `.def()` on every tool on every dispatch** — `crates/app/src/tools/mod.rs:102` — linear scan calling `def()` (which allocates `ToolDef`) for name matching. Should store tools in a `HashMap<String, Arc<dyn Tool>>` or at least compare names without building the full `ToolDef`.

## Performance / Allocation Waste

~~13. **`StreamAccumulator::assemble_blocks` clones everything**~~ — _Resolved: `snapshot()` returns a borrowed `Snapshot<'a>` backed by a lazily built `OnceCell` cache; no per-frame cloning._
14. **Wire role strings allocated fresh every time** — `wire.rs:158,174,185,214` — `"user".to_owned()`, `"assistant".to_owned()`, `"function".to_owned()`, `"tool".to_owned()`. These could be `&'static str` on the wire struct or use `Cow`.
15. **`text()` on `Message` allocates a `Vec` then joins** — `crates/domain/src/message.rs:54-63` — collects into `Vec<&str>` then `.join("")`. Could use `fold` or push to a pre-sized `String`.
~~16. **`list()` deserializes every session file to count messages**~~ — _Resolved: `list()` is now a directory scan; `SessionSummary::message_count` removed._

## Hardcoded / Config Gaps

17. **Model is a CLI default, not configurable at runtime** — `crates/bin-gui/src/main.rs:30` — `default_value = "deepseek/deepseek-r1"`. No in-app model picker.
18. **SSE endpoint URL hardcoded** — `crates/adapter-llm/src/openrouter/mod.rs:40` — `"https://openrouter.ai/api/v1/chat/completions"` baked in. No way to point at a proxy or different base URL.
19. **Channel buffer size 64 is arbitrary** — `crates/adapter-llm/src/openrouter/mod.rs:49` — `tokio::sync::mpsc::channel(64)`. No justification; if the producer outruns the consumer, events silently backpressure.
20. **Reasoning flags always sent** — `wire.rs:145-146` — `reasoning: Some(serde_json::json!({}))` and `include_reasoning: Some(true)` are always included regardless of model. Some models may reject or be confused by these.

## Driver Batching Problem

~~21. **`run_turn` buffers ALL events and flushes after the turn completes**~~ — _Resolved in `dee8565`: driver now uses `mpsc::unbounded_channel` + `tokio::join!` to stream events live. Regression test `stream_deltas_arrive_before_turn_complete` guards this._

## Minor / Style

22. **`eframe::NativeOptions::default()`** — `crates/adapter-egui/src/lib.rs:161` — no window size, title bar config, or icon set. Opens a tiny default window.
23. **`Message` doesn't derive `PartialEq`** — noted in `stream.rs:489` comment as a reason for a hand-rolled comparison function. Adding the derive would simplify tests.
24. **`Usage` uses `u32` for token counts** — `crates/domain/src/stream.rs:35` — fine for single turns but `session.is_over_budget` sums them as `usize`, creating a silent widening on 64-bit platforms.
