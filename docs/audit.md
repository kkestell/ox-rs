# Ox Codebase Audit

Audited 2026-04-19. Covers all 10 library crates and 2 binary crates (~8,000 lines of production Rust).

---

## High Priority

### H1. `ToolRegistry::register` silently corrupts state on duplicate names

**File:** `crates/app/src/tools/mod.rs:110-114`

```rust
pub fn register(&mut self, tool: Arc<dyn Tool>) {
    let def = tool.def();
    self.by_name.insert(def.name.clone(), tool);
    self.defs.push(def);
}
```

If two tools share a name, `by_name` silently replaces the first but `defs` retains *both*. Callers of `defs()` see a duplicate tool advertised while `execute()` dispatches only to the second registrant. Fix: either panic on duplicate names or replace both entries atomically.

---

### H2. `AgentClient` doc claims `Sync` but the type is not `Sync`

**File:** `crates/agent-host/src/client.rs:47-49`

```rust
/// Send-side handle. Thread-safe by construction — `mpsc::UnboundedSender`
/// is `Sync`, so the state guard can hand out `&AgentClient` references
/// across tasks without cloning the client.
pub struct AgentClient {
    cmd_tx: mpsc::UnboundedSender<AgentCommand>,
    _drop_guard: Option<Box<dyn Send>>,   // Send but NOT Sync
}
```

`Box<dyn Send>` is `Send` but not `Sync`, making the entire struct `!Sync`. The comment is factually wrong. Fix: either change the bound to `Box<dyn Send + Sync>` (and update `with_drop_guard`) or correct the comment.

---

### H3. FIFO/LIFO bug in `FakeGit::merge` test double

**File:** `crates/agent-host/src/fake.rs:150,284`

`enqueue_merge_outcome` documents FIFO ordering but consumption uses `Vec::pop()` which is LIFO. A test enqueuing `[A, B]` gets `B` first. Existing tests only push one outcome so the bug is latent. Fix: use `VecDeque` with `push_back`/`pop_front`, or change `pop()` to `remove(0)`.

---

### H4. Blocking `glob::glob()` on the Tokio runtime

**File:** `crates/adapter-fs/src/lib.rs:37-78`

`walk_glob` runs the synchronous `glob::glob()` crate and per-file `is_file()` `stat()` calls directly on the async runtime. On large repos this stalls the runtime for hundreds of milliseconds. Fix: wrap in `tokio::task::spawn_blocking()`.

---

### H5. Non-atomic session file writes (crash = corruption)

**File:** `crates/adapter-storage/src/lib.rs:185-194`

`DiskSessionStore::save` writes session JSON directly via `tokio::fs::write`. A crash mid-write produces a truncated/empty file. The sibling type `DiskLayoutRepository::persist` (same crate, lines 56-77) correctly uses tmp+rename. Apply the same pattern to `save`.

---

### H6. Bash timeout discards all collected output

**File:** `crates/adapter-fs/src/lib.rs:147-180`

On timeout, `tokio::time::timeout` drops the `io_and_wait` future, destroying the stdout/stderr buffers. The comment on line 170 says "recover any partial output" but the code returns empty strings. Fix: use `tokio::select!` with buffers that survive the timeout branch, or move partial output into shared state.

---

### H8. 176-line god-function `run_turn_with_approver`

**File:** `crates/app/src/use_cases/runner.rs:154-330`

This function handles LLM streaming, tool-call extraction, approval planning, approval-stream consumption, tool execution, cancellation at six scattered points, and iteration capping. Decompose into `stream_llm_response`, `plan_and_execute_tools`, and `process_approval_decisions`.

---

### H9. Inconsistent HTTP error response shapes

**Files:** `crates/bin-web/src/routes/messages.rs:22-24`, `routes/sessions.rs:24-28`, `routes/layout.rs:26-30`

Some error responses return plain text (`"input required"`), others return structured JSON (`{"reason":"..."}`). The frontend must handle both formats. Fix: standardize on JSON for all error responses.

---

### H10. Fragile SSE fallback JSON construction

**File:** `crates/bin-web/src/sse.rs:69-74`

```rust
format!(r#"{{"type":"error","message":"sse encode failed: {e}"}}"#)
```

If the error message `e` contains `"` or `\`, the manually-constructed JSON is malformed. Fix: use `serde_json::json!({ "type": "error", "message": format!("...") })`.

---

### H11. `expect("mutex poisoned")` in `Drop` implementations

**Files:** `crates/bin-web/src/session/mod.rs:132`, `lifecycle/close_guard.rs:41`

Panicking during `Drop` aborts the process. A poisoned mutex means another thread already panicked — re-panicking in `drop` makes things worse. Fix: use `.unwrap_or_else(|e| e.into_inner())` in `Drop` impls, or at minimum in the session cleanup path.

---

## Medium Priority

### M1. Default `approval_requirement` silently always-requires approval

**File:** `crates/app/src/tools/mod.rs:63-72`

The default `Tool::approval_requirement` returns `Required`. `BashTool`, `MergeTool`, and `AbandonTool` rely on this default silently — there is no comment acknowledging the choice, and adding a new tool without overriding the method will silently require approval for every call. Fix: document the intent or remove the default body (forcing explicit opt-in).

---

### M2. JSON argument parsing duplicated per tool (`approval_requirement` + `execute`)

**Files:** `crates/app/src/tools/{read_file,write_file,edit_file,glob,grep}.rs`

Every file tool parses its JSON args struct twice — once in `approval_requirement` and once in `execute`. A parsed-args struct computed once and threaded through would eliminate the duplication.

---

### M3. Unbounded channels with no backpressure

**File:** `crates/agent-host/src/client.rs:85-86`

Both command and event channels are `mpsc::unbounded_channel()`. If the agent produces events faster than the host consumes them, memory grows without bound. Fix: use `mpsc::channel` with a bounded capacity (e.g., 256).

---

### M4. No frame size limit on `read_frame` (OOM risk)

**File:** `crates/protocol/src/lib.rs:167-180`

`read_line` reads an arbitrarily large line into memory. A buggy or malicious agent could OOM the host. Fix: add a configurable max frame size.

---

### M5. Public fields on `SessionRuntime` break encapsulation

**File:** `crates/agent-host/src/session_runtime.rs:18-26`

All fields are `pub` but the struct has state-transition methods (`apply_event`, `begin_send`, `begin_close`) that maintain invariants. External code can bypass the state machine. Fix: make fields private and expose read accessors.

---

### M6. No protocol version in `Ready` handshake

**File:** `crates/protocol/src/lib.rs:85-88`

If host and agent are built from different commits, they may speak incompatible protocol versions. `#[non_exhaustive]` prevents panics but not semantic mismatches. Fix: add `protocol_version: u32` to `Ready`.

---

### M7. No graceful shutdown for IPC reader/writer tasks

**File:** `crates/agent-host/src/client.rs:133,169`

`JoinHandle`s from `tokio::spawn` are discarded. Panics are silently swallowed; there is no `shutdown()` method. Fix: store handles and add cancellation support.

---

### M8. `std::sync::RwLock` / `std::sync::Mutex` in async context

**Files:** `crates/bin-web/src/registry/mod.rs:47`, `session/mod.rs:87-90`

Comments say "never held across an `.await`" but this is a social contract, not compiler-enforced. A future `.await` added while holding the lock will block a Tokio thread. Fix: use `tokio::sync::RwLock` / `tokio::sync::Mutex` to make the contract self-enforcing.

---

### M9. Full-history clone for slug extraction

**File:** `crates/bin-web/src/session/pump.rs:176-179`

The entire `Vec<AgentEvent>` history is cloned just to extract the first user message's text. Fix: iterate the locked history directly and return only the needed `String`.

---

### M10. Blocking `Path::exists()` in async context

**File:** `crates/adapter-git/src/lib.rs:310`

```rust
if !worktree_path.exists() {  // blocking stat() syscall
```

Fix: use `tokio::fs::metadata(worktree_path).await.is_ok()`.

---

### M11. `SecretStore::get` is sync while all other ports are async

**File:** `crates/app/src/ports.rs:42-44`

A future secret store (Vault, AWS SM) would need async I/O. The sync signature precludes this without a breaking change.

---

### M12. Inconsistent async signatures between port traits

**File:** `crates/app/src/ports.rs`

`LlmProvider` uses `impl Future` (not dyn-compatible); `SessionStore` uses manual `Pin<Box<dyn Future>>` (dyn-compatible). Same file, no principled reason for the difference.

---

### M13. `eprintln!` for error reporting in slug generator

**File:** `crates/adapter-llm/src/openrouter/slug.rs:43-55`

The rest of the codebase uses `anyhow` for structured error propagation. `eprintln!` here is inconsistent and untestable. Fix: use `tracing::warn!` or propagate the error.

---

### M14. Hardcoded `/bin/bash` — not portable

**File:** `crates/adapter-fs/src/lib.rs:100`

Fails on NixOS, FreeBSD, and minimal Docker images where bash is not at `/bin/bash`. Fix: resolve from `$SHELL` or fall back to `/bin/sh`.

---

### M15. Unused `async-trait` dependency

**File:** `crates/adapter-storage/Cargo.toml:10`

No file in `adapter-storage` uses `#[async_trait]`. All implementations manually return `Pin<Box<dyn Future>>`. Fix: remove the dependency.

---

### M16. `assert_repo` spawns three separate `git` processes

**File:** `crates/adapter-git/src/lib.rs:102-127`

Two logical checks (inside-work-tree + not-detached-head) could be consolidated into fewer git invocations.

---

### M17. 190-line `build_system_prompt` function

**File:** `crates/bin-agent/src/main.rs:32-222`

A single function constructing a static string prompt. Fix: use `include_str!` with placeholder substitution, or break into a dedicated `prompt.rs` module.

---

### M18. Silent event loss for malformed frames mid-turn

**File:** `crates/bin-agent/src/driver/mod.rs:193-195`

Between turns, malformed frames produce an `Error` event. Mid-turn, they are silently discarded. This inconsistency means a client sending garbage mid-turn gets no feedback.

---

### M19. `load_history` pattern duplicated

**Files:** `crates/bin-web/src/registry/restore.rs:50-57`, `registry/snapshot.rs:39-46`

Both files contain identical three-way error-handling logic for loading a layout. Fix: extract a helper method.

---

### M20. Wire errors conflated with application errors in reader task

**File:** `crates/agent-host/src/client.rs:146-155`

Protocol/transport errors are converted into `AgentEvent::Error` and mixed into the same stream as application-level errors. The consumer cannot distinguish wire corruption from model errors. Fix: use a `Result`-like wrapper at the stream level.

---

## Low Priority

### L1. Missing `Eq` derive on `ContentBlock`

**File:** `crates/domain/src/content_block.rs:3`

All fields implement `Eq`. Every surrounding type derives it. One-word fix for consistency.

### L2. Missing `PartialEq` derive on `Message`

**File:** `crates/domain/src/message.rs:14`

All fields implement it. Tests compare fields individually instead of using `assert_eq!` on the whole struct.

### L3. Missing `Debug` derive on `Snapshot`

**File:** `crates/domain/src/stream/snapshot.rs:16`

Both fields implement `Debug`. Every surrounding type derives it.

### L4. `BTreeMap<usize, ToolCallAccum>` for dense sequential keys

**File:** `crates/domain/src/stream/accumulator.rs:30`

Tool call indices are always dense and sequential. `Vec<Option<ToolCallAccum>>` would be O(1) instead of O(log n).

### L5. Misleading `Sync`/`OnceCell` comment

**File:** `crates/domain/src/stream/accumulator.rs:21-24`

`OnceCell<T>` is `Sync` when `T: Sync`. The comment incorrectly blames `OnceCell` for the type not being `Sync`.

### L6. `OnceCell` cache adds complexity for marginal gain

**File:** `crates/domain/src/stream/accumulator.rs:39-40`

Assembly is cheap (a few string clones). A simpler `Option<Vec<ContentBlock>>` would be clearer.

### L7. `assemble_blocks` clones strings that could be moved in the consuming path

**File:** `crates/domain/src/stream/accumulator.rs:139-178`

`into_message` clones data it could take ownership of because `assemble_blocks` is shared with the borrowing `snapshot` path. Fix: add `assemble_blocks_owned(self)`.

### L8. `ToolDef` doc claims wire usage but lacks `Serialize`/`Deserialize`

**File:** `crates/domain/src/stream/mod.rs:59-70`

Adapter-LLM manually converts to `WireTool`. The doc is misleading.

### L9. `Message::text()` intermediate `Vec` allocation

**File:** `crates/domain/src/message.rs:64-73`

Collects into `Vec<&str>` only to join. `itertools::Itertools::join()` or a fold avoids the allocation.

### L10. `push_message` is a trivial `Vec::push` wrapper

**File:** `crates/domain/src/session.rs:65-67`

Adds API surface without validation, event emission, or invariant enforcement.

### L11. `format!` in hot loop for hashline rendering

**File:** `crates/app/src/tools/hashlines.rs:98`

Allocates a temporary `String` per line. `write!` on a pre-allocated `String` avoids the intermediate.

### L12. Unnecessary clone of `approval_requests`

**File:** `crates/app/src/use_cases/runner.rs:281`

`TurnEvent::ToolApprovalRequested` owns a `Vec` while other variants borrow. The event enum could take `&[ToolApprovalRequest]` instead.

### L13. Unnamed `(String, String, String)` tuple in pending map

**File:** `crates/app/src/use_cases/runner.rs:235`

Opaque tuple representing `(id, name, arguments)`. A named struct would be self-documenting.

### L14. Missing `Debug` derives on tool structs and `SessionRunner`

**Files:** All tool structs in `crates/app/src/tools/`, `crates/app/src/use_cases/runner.rs`

### L15. Missing `PartialEq` on `WalkResult` and `CommandOutput`

**File:** `crates/app/src/ports.rs:87-110`

Every other data type in the crate derives it.

### L16. `TurnEvent` inconsistently borrows vs. owns

**File:** `crates/app/src/use_cases/turn_event.rs:28-33`

`StreamDelta` and `MessageAppended` borrow; `ToolApprovalRequested` and `ToolApprovalResolved` own.

### L17. Six scattered `cancel.is_cancelled()` checks in one function

**File:** `crates/app/src/use_cases/runner.rs`

A structured cancellation approach (cancel-guard wrapper or `Cancelled` result type) would reduce scattered conditionals.

### L18. `fake.rs` is a 705-line monolith

**File:** `crates/app/src/fake.rs`

Five unrelated test doubles. Fix: split into a `fake/` directory.

### L19. Duplicated `WALK_MAX_BYTES` constant

**Files:** `tools/glob.rs:23`, `tools/grep.rs:26`

Same magic number defined independently. Fix: extract to `tools/mod.rs` or `ports.rs`.

### L20. Plurality-suffix logic duplicated across three tools

**Files:** `tools/glob.rs:123`, `tools/grep.rs:179-180`, `tools/edit_file.rs:236-237`

Fix: extract a `fn plural(count: usize, singular: &str, plural_form: &str) -> &str`.

### L21. Tool struct boilerplate repeated for every file tool

Each tool repeats: `fs: Arc<F>`, `workspace_root: PathBuf`, `fn new(...)`, path resolution, and approval checking. A shared `FileToolContext<F>` would eliminate the repetition.

### L22. `.unwrap_or(None)` instead of `.flatten()`

**File:** `crates/agent-host/src/fake.rs:375`

Idiomatic Rust uses `.flatten()` to collapse `Option<Option<T>>`.

### L23. `grep.rs` silently swallows file-read errors

**File:** `crates/app/src/tools/grep.rs:142-147`

`Err(_)` discards all diagnostic information. Fix: include the error variant in the "read errors" note.

### L24. `adapter-llm` SSE parser allocates per chunk

**File:** `crates/adapter-llm/src/openrouter/sse.rs:32-37`

`replace()`, `to_owned()` on every SSE chunk. A `Vec<u8>` buffer would reduce allocation pressure.

### L25. Duplicated `reqwest::Client` in OpenRouter

**Files:** `adapter-llm/src/openrouter/mod.rs:27`, `slug.rs:26`

Both `OpenRouterProvider` and `OpenRouterSlugGenerator` construct separate clients despite sharing the same HTTP needs. `reqwest::Client` is designed to be shared (connection pool).

### L26. Hardcoded OpenRouter API URL in two files

**Files:** `adapter-llm/src/openrouter/mod.rs:45`, `slug.rs:80`

Fix: extract to a constant.

### L27. Hardcoded reasoning flags sent for all models

**File:** `adapter-llm/src/openrouter/wire.rs:168-169`

`reasoning` and `include_reasoning` are sent unconditionally. Should be conditional on model support.

### L28. Full frame content in error message

**File:** `crates/protocol/src/lib.rs:178-179`

A 2 MiB frame produces a 2 MiB error string. Fix: truncate the preview to 256 bytes.

### L29. New `String` allocation per frame in `read_frame`

**File:** `crates/protocol/src/lib.rs:167-168`

A caller-owned reusable buffer would avoid per-frame allocation. Low priority for interactive use.

### L30. `normalize_sizes` mutates in-place with length parameter

**File:** `crates/agent-host/src/layout.rs:57-75`

Unusual API. Returning a new `Vec<f32>` would be more idiomatic.

### L31. Missing doc on why `Pin<Box<...>>` is used in trait methods

**Files:** `agent-host/src/{layout,git,close_request_sink,first_turn_sink,slug_generator}.rs`

All use manual `Pin<Box<...>>` for object safety but none explain why. Fix: add a brief doc comment.

### L32. Wildcard match arms on `#[non_exhaustive]` types

**Files:** `routes/messages.rs:53-57`, `registry/dispatch.rs:39-41`

Silently absorb future variants. Fix: match remaining variants explicitly.

### L33. Dead code type alias in `sse.rs`

**File:** `crates/bin-web/src/sse.rs:81-82`

`type _Marker = Box<dyn Stream<...>>` is never used. Remove it.

### L34. `#[allow(dead_code)]` on multiple public methods

**Files:** `session/methods.rs:79,92,100,225`, `registry/mod.rs:100,131,282`

If needed, remove the suppressions by using them. If not, remove the dead code.

### L35. Dead session lookup in `snapshot()`

**File:** `crates/bin-web/src/registry/snapshot.rs:56-57`

```rust
if let Some(session) = sessions.get(id) {
    let _ = session; // Summary doesn't need session-level state today.
```

Does a full `HashMap` lookup + `Arc` clone for nothing. Fix: use `sessions.contains_key(id)`.

### L36. Too many arguments on `restore`

**File:** `crates/bin-web/src/registry/restore.rs:29`

`#[allow(clippy::too_many_arguments)]` suppresses the lint. Fix: introduce a `RestoreConfig` struct.

### L37. Magic number for 16 MB body limit

**File:** `crates/bin-web/src/routes/mod.rs:60`

`16 * 1024 * 1024` should be a named constant.

### L38. Model string cloned per-session per-snapshot

**File:** `crates/bin-web/src/registry/snapshot.rs:60,69`

Identical model string cloned in a loop. Fix: clone once outside the loop or use `Arc<str>`.

### L39. `short_prefix` allocates full UUID to slice 8 chars

**File:** `crates/bin-web/src/lifecycle/mod.rs:621-624`

Allocates a 36-char string only to take the first 8 chars. Direct hex encoding avoids the intermediate.

### L40. `rejection_message` heap-allocates static strings

**File:** `crates/bin-web/src/lifecycle/mod.rs:630-639`

Six of seven variants return `&'static str` data via `.to_owned()`. `Cow<'static, str>` avoids it.

### L41. Async function that doesn't await

**File:** `crates/bin-web/src/registry/dispatch.rs:21`

`send_command` is `async` but performs no `.await`. All operations are synchronous.

### L42. Boolean flag controlling `Drop` behavior

**File:** `crates/bin-web/src/lifecycle/close_guard.rs:21-22,36-40`

A boolean `clear_session_on_drop` controls complex conditional `Drop` logic. An enum or split guard types would be clearer.

### L43. `Ordering::SeqCst` used everywhere

**Files:** `session/pump.rs:166,201`, `session/methods.rs:86,93,102,166,187,213,293`

All atomic operations use the strongest memory ordering. Several could use `Acquire`/`Release`.

### L44. Duplex buffer size magic number in tests

Multiple test files use `duplex(64 * 1024)`. Fix: extract a shared constant.

### L45. `ToolApprovalResolved` echo semantics undocumented

**File:** `crates/protocol/src/lib.rs:98-104`

The agent echoes back every approval resolution. The purpose is unclear. Fix: document the semantics or remove if vestigial.

### L46. `BashTool` relies on default approval without acknowledgment

**File:** `crates/app/src/tools/bash.rs`

Does not override `approval_requirement`. Every bash command requires approval via an implicit default in a different file.

### L47. `envelope` parse fallback silently discards errors

**File:** `adapter-llm/src/openrouter/catalog.rs:133-136`

If the envelope parse fails, the error is discarded before trying bare-array parse. Fix: try envelope first, only fall back if input doesn't look like an object.

### L48. Missing `.with_context()` on `FileSystem::read`

**File:** `crates/adapter-fs/src/lib.rs:14-16`

Other methods in the same impl add filename context. This one produces bare `io::Error` messages.

### L49. `Captured` struct missing `Debug` derive

**File:** `crates/adapter-git/src/lib.rs:43-47`

Every other struct in the crate derives it.

### L50. `spill_path` silently uses epoch 0 on time error

**File:** `crates/app/src/tools/spill.rs:68-71`

`unwrap_or_default()` masks a potentially surprising failure and could cause file-name collisions.

### L51. Agent stderr inherited directly

**File:** `crates/adapter-process/src/lib.rs:34`

Unstructured agent diagnostics (panics, debug prints) interleave with host output. Fix: capture agent stderr for structured logging.

---

## Summary

| Priority | Count |
|----------|-------|
| High     | 11    |
| Medium   | 20    |
| Low      | 51    |

### Themes

1. **Encapsulation gaps.** Public fields on state machines (`SessionRuntime`), silently-corrupting collections (`ToolRegistry`), and implicit default behavior (approval requirement) are the most actionable design issues.

2. **Async/blocking mismatches.** Synchronous filesystem and glob operations run on the Tokio runtime. std sync primitives are used where Tokio equivalents would be safer. These are latent bugs that become real under load.

3. **Inconsistent error handling.** HTTP routes mix plain-text and JSON errors. `eprintln!` coexists with `anyhow`. Silent error discarding in grep/grep and frame parsing makes debugging harder. Mutex poisoning panics in `Drop` can abort the process.

4. **Duplication.** Tool struct boilerplate, `WALK_MAX_BYTES`, `load_history` logic, pluralization, and OpenRouter URL/constants are copy-pasted across files. Shared helpers or a context struct would reduce surface area.

5. **Unidiomatic Rust.** Missing derives (`Debug`, `Eq`, `PartialEq`), `.unwrap_or(None)` instead of `.flatten()`, unnecessary intermediate allocations, and wildcard match arms on `#[non_exhaustive]` enums.
