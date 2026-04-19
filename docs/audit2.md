# Architectural Audit #2

Deep review of the full codebase across all 12 crates (~20K lines of Rust).

---

## 1. Hexagonal Architecture Violations

### 1.1 `agent-host` depends on the application layer ✅ FIXED

`agent-host`'s `Cargo.toml` depends on `app`. In hexagonal architecture the port-defining crate should sit *above* the application layer — it defines the interfaces that adapters implement. Here the dependency is inverted: `agent-host → app → domain`.

The dependency exists because:
- `SessionRuntime` uses `app::StreamAccumulator` directly (`session_runtime.rs:8`)
- `agent-host` re-exports `app::StreamAccumulator` from its own `lib.rs` (`pub use app::StreamAccumulator`)

**Fix:** Either move `StreamAccumulator` to `domain` (it only depends on domain types), or move the state machine to `app` and have `agent-host` define only the port traits. The current placement makes `agent-host` a "port + some application logic" crate, which defeats its stated purpose of being "framework-agnostic."

**Resolution:** Moved `StreamAccumulator`, `Snapshot`, and `ToolDef` to `domain` alongside the `StreamEvent` / `Usage` types they build on. Dropped `app` from `agent-host`'s dependencies.

### 1.2 Two session storage ports for the same concept ✅ FIXED

Two separate traits model session persistence:

| Trait | Location | Style |
|---|---|---|
| `app::SessionStore` | `app/src/ports.rs:23` | RPITIT (`impl Future<Output = ...>`) |
| `agent_host::SessionRecords` | `agent-host/src/session_records.rs:13` | `#[async_trait]` |

`DiskSessionStore` implements both (`adapter-storage/src/lib.rs:133-145`). The `SessionRecords` impl just delegates to `SessionStore` methods. `SessionRecords` exists because `app::SessionStore` uses RPITIT, which can't be turned into a trait object — but the right fix is to make the trait object-safe (use `#[async_trait]` or boxed futures), not to define a second parallel trait.

**Resolution:** Reshaped `app::SessionStore` to be dyn-compatible (RPITIT with explicit `Pin<Box<dyn Future + Send + '_>>` where needed) and renamed `load` to `try_load` returning `Result<Option<Session>>`. Deleted `SessionRecords` and its impl; all call sites now hold `Arc<dyn SessionStore>`.

### 1.3 Two `ToolApprovalRequest` types ✅ FIXED

Identical structs defined independently:

- `app::ToolApprovalRequest` in `app/src/approval.rs:19`
- `protocol::ToolApprovalRequest` in `protocol/src/lib.rs:55`

Both have the same five fields: `request_id`, `tool_call_id`, `name`, `arguments`, `reason`. The `bin-agent` driver manually converts between them field-by-field at `driver.rs:415-421`. This is a classic violation: the same concept exists in two crates because neither depends on the other, so a third definition was introduced instead of deciding where it truly belongs.

**Fix:** Pick one canonical location (probably `protocol` since it's the wire type) and have `app` import or re-export it.

**Resolution:** Made `protocol` the canonical home and had `app` depend on it and re-export the type. The field-by-field conversion in the driver is gone.

### 1.4 `app` re-exports `domain` types, creating dual import paths

`app/src/lib.rs:29` re-exports `StreamEvent` and `Usage` from `domain`:

```rust
pub use domain::{StreamEvent, Usage};
```

This means consumers can `use app::StreamEvent` or `use domain::StreamEvent` interchangeably, blurring the domain boundary. The re-export is documented as a convenience to avoid "chasing the move," but it means the crate boundary is meaningless for these types.

---

## 2. Duplicated Code and Dual Implementations

### 2.1 `async_trait` vs RPITIT inconsistency ✅ FIXED

The codebase uses two different approaches for async trait methods:

- **RPITIT** (return-position `impl Trait`): `app::LlmProvider`, `app::SessionStore`, `app::FileSystem`, `app::Shell` in `app/src/ports.rs`
- **`#[async_trait]`**: `agent_host::Git`, `agent_host::SessionRecords`, `agent_host::SlugGenerator`, `agent_host::AgentSpawner`, `agent_host::CloseRequestSink`, `agent_host::FirstTurnSink`

This is why `SessionRecords` exists alongside `SessionStore` — RPITIT traits can't be object-safe in some patterns, so `async_trait` is used for the host-side traits that need `dyn`. The codebase should pick one strategy. Since edition 2024 + MSRV allows RPITIT in trait objects with `-> impl Future + Send`, or can use explicit boxing, `async_trait` is unnecessary overhead.

**Resolution:** Converted every `agent-host` trait to RPITIT. Traits used behind `Arc<dyn _>` (`CloseRequestSink`, `FirstTurnSink`, `AgentSpawner`, `Git`, `SlugGenerator`) return `Pin<Box<dyn Future + Send + '_>>`; non-dyn traits return `impl Future + Send`. Dropped `async-trait` from `agent-host`'s dependencies.

### 2.2 `normalize_lexical` in `app::approval` is crate-local but broadly useful

`app/src/approval.rs:140` defines `normalize_lexical(path: &Path) -> PathBuf` which does `.`/`..` elimination. This is a general path utility that could live in `domain` or a shared utility crate. It's currently `pub(crate)` and only used in the approval module (and once in `fake.rs:314` through a crate-qualified path).

### 2.3 Two separate slug-generation stack definitions

The `openrouter/slug.rs` wire types (`RequestBody`, `ChatMessage`, `ChatResponse`, etc. at lines 163-211) are completely separate from the `openrouter/wire.rs` types (`RequestBody`, `WireMessage`, etc.), even though they send requests to the same endpoint (`https://openrouter.ai/api/v1/chat/completions`). Two independent request/response type hierarchies for the same API.

### 2.4 Dual `fake` modules

- `app/src/fake.rs` (697 lines): `FakeLlmProvider`, `FakeSessionStore`, `FakeFileSystem`, `FakeTool`, `FakeShell`
- `agent-host/src/fake.rs` (611 lines): `FakeGit`, `FakeSlugGenerator`, `FakeCloseRequestSink`, `FakeFirstTurnSink`

This split is somewhat expected given the crate structure, but tests in `bin-web` must import fakes from both crates simultaneously. The split forces every integration test to juggle two separate fake ecosystems.

---

## 3. Complex / Unusual Code

### 3.1 Mega-files ✅ FIXED

| File | Lines | Contents |
|---|---|---|
| `app/src/use_cases.rs` | 1835 | `SessionRunner`, `TurnEvent`, `TurnOutcome`, tool loop, approval planning, tests |
| `bin-web/src/routes.rs` | 1788 | All HTTP handlers, merge/abandon logic, 1000+ lines of tests |
| `bin-agent/src/driver.rs` | 1494 | IPC loop, `ApprovalBroker`, `TurnRun`, `run_turn`, tests |
| `bin-web/src/session.rs` | 1094 | `ActiveSession`, pump task, startup replay suppression, tests |
| `bin-web/src/registry.rs` | 1161 | `SessionRegistry`, layout persistence, restore logic, tests |
| `bin-web/src/lifecycle.rs` | 943 | `SessionLifecycle`, merge/abandon, slug rename, tests |
| `app/src/stream.rs` | 865 | `StreamAccumulator`, `Snapshot`, 400+ lines of tests |

The first three are the most concerning: they each bundle multiple responsibilities that would benefit from extraction into submodules. `use_cases.rs` in particular mixes the session runner, the tool-call loop, approval planning, and comprehensive tests all in a single flat file.

**Resolution:** Every flagged file was split into per-concern submodules preserving git history via `git mv`:

- `app/use_cases` → `runner.rs`, `tool_loop.rs`, `turn_event.rs`, `mod.rs`
- `bin-web/routes` → `sessions.rs`, `messages.rs`, `layout.rs`, `static_assets.rs`, `mod.rs`
- `bin-agent/driver` → `approval_broker.rs`, `reader.rs`, `turn.rs`, `mod.rs`
- `bin-web/session` → `methods.rs`, `pump.rs`, `mod.rs`
- `bin-web/registry` → `snapshot.rs`, `dispatch.rs`, `restore.rs`, `mod.rs`
- `bin-web/lifecycle` → `close_guard.rs`, `sinks.rs`, `mod.rs`
- `domain/stream` (post-move from 1.1) → `accumulator.rs`, `snapshot.rs`, `mod.rs`

### 3.2 `ApprovalBroker` polls with a 25ms sleep loop

`bin-agent/src/driver.rs:64-88` implements cancellation checking in the approval broker by sleeping for 25ms in a loop:

```rust
loop {
    if cancel.is_cancelled() {
        // ...
    }
    tokio::select! {
        decision = &mut rx => { ... }
        _ = tokio::time::sleep(Duration::from_millis(25)) => {}
    }
}
```

This is unusual. The idiomatic approach would be to use a `tokio::sync::watch` or `CancellationToken` from `tokio_util` that integrates directly with `select!` without a polling interval. The current approach wastes scheduler slots every 25ms per pending approval.

### 3.3 `SessionRuntime` uses free functions instead of methods

`agent-host/src/session_runtime.rs` exposes its API as free functions:

```rust
pub fn apply_event(state: &mut SessionRuntime, event: AgentEvent)
pub fn begin_send(state: &mut SessionRuntime) -> ShouldSend
pub fn begin_close(state: &mut SessionRuntime) -> BeginClose
pub fn clear_closing(state: &mut SessionRuntime)
```

Every caller must pass `&mut SessionRuntime` explicitly. The comment says this is for "ownership model" reasons, but `&mut self` methods would be more natural and idiomatic. The free-function style forces callers to write `begin_send(&mut rt)` instead of `rt.begin_send()`.

### 3.4 Three-phase init cycle between `SessionLifecycle` and `SessionRegistry` ✅ FIXED

`bin-web/main.rs:215-224` documents a three-phase init to break a reference cycle:

1. Build `SessionLifecycle` with an empty `Weak<SessionRegistry>`
2. Build `SessionRegistry`, passing `lifecycle` as `CloseRequestSink`
3. Call `lifecycle.set_registry(Arc::downgrade(&registry))`

The lifecycle accesses the registry through `Weak::upgrade()` at runtime, which can silently return `None` if the registry was dropped. This is architecturally fragile — any lifecycle method that needs the registry must handle the "registry gone" case, and there's no compile-time guarantee that `set_registry` was called before the first use.

**Resolution:** Replaced the `Weak<SessionRegistry>` callback with `tokio::sync::mpsc` channels. Sessions now hold `Arc<dyn CloseRequestSink>` backed by a `ChannelCloseSink` that pushes `CloseRequestMsg`/`FirstTurnMsg` onto unbounded channels; the composition root spawns consumer tasks that drain them and dispatch into `SessionLifecycle`. No more `Weak`, no more `set_registry`, init is single-phase.

### 3.5 `DiskLayoutRepository` uses blocking `std::sync::Mutex` in async context ✅ FIXED

`adapter-storage/src/lib.rs:3` uses `std::sync::Mutex` for the in-memory layout cache and calls `std::fs` (blocking I/O) for persistence. Meanwhile, `DiskSessionStore` in the same file uses `tokio::fs` (async I/O). The layout repo's `persist` method does synchronous file I/O while holding the mutex — this can block a tokio worker thread.

**Resolution:** Swapped `std::sync::Mutex` for `tokio::sync::Mutex` and `std::fs::*` for `tokio::fs::*`. The mutex is held only around the in-memory cache update; disk writes happen outside the critical section so no `.await` straddles a lock.

### 3.6 `ActiveSession` has 13 fields behind 6 different synchronization primitives

`bin-web/src/session.rs:78-120` — `ActiveSession` manages:

- `session_id: OnceLock<SessionId>`
- `agent: Mutex<AgentClient>`
- `history: Arc<Mutex<Vec<AgentEvent>>>`
- `tx: broadcast::Sender<AgentEvent>`
- `runtime: Arc<Mutex<SessionRuntime>>`
- `alive: Arc<AtomicBool>`
- `fresh: Arc<AtomicBool>`
- `ready_notify: Arc<tokio::sync::Notify>`
- `pump: Mutex<tokio::task::AbortHandle>`
- `close_sink: Arc<dyn CloseRequestSink>`
- `first_turn_sink: Arc<dyn FirstTurnSink>`

The locking discipline comment at the top of the file is necessary because the locking order is not enforced by the type system. Any future contributor who gets the lock order wrong will introduce deadlocks or races.

### 3.7 Startup replay suppression in the pump is heuristic

`bin-web/src/session.rs:470-489` implements a `filtering_startup_replay` mode that suppresses replayed `MessageAppended` frames by comparing them against the runtime's existing message list using a `messages_match` equality check. If a resumed agent replays messages that happen to match the existing history, they're silently dropped. If the agent produces a message that *doesn't* match the expected sequence, replay filtering is disabled. This is fragile — any divergence between the agent's replay and the host's history breaks the suppression.

### 3.8 `CloseGuard` carries a `'a` lifetime on `Mutex<HashMap>` that belongs to `SessionLifecycle`

`bin-web/src/lifecycle.rs:462-489` — The `CloseGuard<'a>` borrows `&'a Mutex<HashMap<SessionId, CloseState>>` from the lifecycle's `self.closing` field. This means the guard holds a borrow of the lifecycle through the mutex, not through the lifecycle itself. This works but is an unusual pattern — typically the guard would borrow `&'a SessionLifecycle` directly.

### 3.9 Nine-parameter `spawn_pump` function

`bin-web/src/session.rs:435` — The `spawn_pump` function takes 9 parameters. The comment says "grouping them under a struct just renames the problem," but a `PumpContext` struct would at least give names to the parameter positions and make call sites self-documenting.

---

## 4. Summary of Recommendations

| Priority | Issue | Fix | Status |
|---|---|---|---|
| High | `agent-host → app` dependency inversion | Move `StreamAccumulator` to `domain` or make `agent-host` a pure port crate | ✅ Fixed |
| High | Two `ToolApprovalRequest` types | Unify into one definition in `protocol` | ✅ Fixed |
| High | Two session storage ports | Make `app::SessionStore` object-safe and remove `SessionRecords` | ✅ Fixed |
| Medium | `async_trait` vs RPITIT inconsistency | Pick one strategy project-wide | ✅ Fixed |
| Medium | Mega-files (1835, 1788, 1494 lines) | Extract submodules | ✅ Fixed |
| Medium | Three-phase init cycle | Restructure to avoid the cycle, or use a builder that enforces ordering | ✅ Fixed |
| Medium | Blocking I/O in `DiskLayoutRepository` | Switch to `tokio::fs` | ✅ Fixed |
| Low | Free-function API on `SessionRuntime` | Convert to `&mut self` methods | — |
| Low | 25ms polling in `ApprovalBroker` | Use a proper cancellation token | — |
| Low | Dual import paths for `StreamEvent`/`Usage` | Remove the `app` re-export, import from `domain` directly | — |
| Low | 9-parameter `spawn_pump` | Extract a `PumpContext` struct | — |
