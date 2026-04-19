# Per-Session Context Usage Tracking

## Goal

Show each session's current context usage against the model's context window in the ox web UI — e.g. `100,000 / 200,000 (50%)` — to the left of the Cancel/Merge/Abandon button cluster. The number must update after every LLM turn (including tool-use iterations) and must survive page reloads.

## Desired outcome

- A usage chip renders in `.session-actions`, immediately to the left of the Cancel button, for every session.
- Numerator is the `prompt_tokens` reported by OpenRouter for the **most recent** LLM call (matches the C# `ox` behavior).
- Denominator is the model's `context_length` as reported by OpenRouter's `/models` endpoint.
- Pre-first-turn state shows `0 / 200,000 (0%)`.
- After a page reload the chip is correct immediately (no "flash of empty" until the next turn).
- If OpenRouter's `/models` endpoint is unavailable at startup, `bin-web` refuses to boot with a clear error. This is not graceful-degrade territory — usage tracking is load-bearing.

## How we got here

The user wants parity with the C# `ox` context chip, but rendered in the web UI to the left of the merge/abandon cluster instead of on a TUI status line. Exploration surfaced two important facts:

1. **Usage is already flowing end-to-end.** `StreamEvent::Finished { usage: Usage }` is emitted by the OpenRouter adapter, propagates through `StreamAccumulator`, is forwarded as `AgentEvent::StreamDelta` / `AgentEvent::MessageAppended` to the host, and arrives at the browser over SSE. The frontend has an explicit comment (`app.js:110` — "Usage is not rendered in V1; ignore") where it throws the number away.
2. **The denominator is not known anywhere.** No model-to-context-window map exists in ox-rs today. The cached `experiments/openrouter_models.json` confirms OpenRouter exposes `context_length` per model id (e.g. 16384, 131072, 262144), so the cleanest source is the live `/models` endpoint on startup.

The user confirmed two design decisions during brainstorming:

- **Model metadata source:** fetch `GET /api/v1/models` at `bin-web` startup, cache the `id → context_length` map in memory. If the fetch fails, the host exits with a non-zero status. No offline fallback, no lazy first-turn fetch.
- **Usage storage:** drop the misnamed `Message.token_count: usize` and replace it with `Message.usage: Option<Usage>`. The existing field only tracks completion tokens, which is the wrong half of what we want, and it's been sitting there unused by the UI anyway.

## Summary of approach

Four layers of changes:

1. **Domain** — rename/retype `Message.token_count: usize` → `Message.usage: Option<Usage>`. Update `StreamAccumulator::into_message` to populate it from the already-accumulated `self.usage`.
2. **App** — add a `ModelCatalog` port: `fn context_window(&self, model: &str) -> Option<u32>`. Keep it small and read-only. Inject it into the host wiring (not the agent subprocess — the agent doesn't need it; this is a UI concern).
3. **adapter-llm** — add `OpenRouterCatalog`, an implementation of `ModelCatalog` that fetches `GET /api/v1/models` once at construction time and caches the parsed `id → context_length` map. Hard-fails on any transport, status, or schema error.
4. **bin-web + assets** — fetch and wire the catalog in `main.rs`; include `context_window` in the HTTP session-load responses; stop discarding `usage` in `app.js`; render the chip in `index.html` and style it in `styles.css`.

No change to the protocol crate. No change to the agent subprocess. `AgentEvent` already carries usage via `MessageAppended.message.usage` once the domain change lands. The host looks up `context_window` per session from the in-memory catalog and attaches it to the JSON that bootstraps the browser's view of the session.

## Related code

- `crates/domain/src/stream/mod.rs:52` — `Usage` struct definition; already has `prompt_tokens`, `completion_tokens`, `reasoning_tokens` with serde derives. No shape change needed.
- `crates/domain/src/stream/mod.rs:18` — `StreamEvent` enum; `Finished { usage }` variant is the one that carries usage out of the adapter.
- `crates/domain/src/message.rs:17` — `Message` struct. This is where the `token_count: usize` → `usage: Option<Usage>` swap lands (three constructor sites at lines 17, 25, 33, 49).
- `crates/domain/src/session.rs:67` — `Session::is_over_limit` sums `m.token_count`. Must be reimplemented against `usage` or dropped.
- `crates/domain/src/stream/snapshot.rs:10` — `Snapshot.token_count` field; swap for `usage` or derived accessor.
- `crates/domain/src/stream/accumulator.rs:25` — `StreamAccumulator` already stores `usage: Usage`; `into_message()` (lines 100, 114) must start wiring it into the produced `Message`. Multiple in-crate tests (lines 187, 420, 520, 523, 606, 682, 703) assert on `token_count` and will be updated.
- `crates/agent-host/src/session_runtime.rs:249` — test assertion on `snap.token_count`; update when the snapshot field changes.
- `crates/adapter-llm/src/openrouter/wire.rs:316` — constructs a `Message { .., token_count: 0 }`; update to `usage: None`.
- `crates/bin-web/src/session/pump.rs:216` — message equality check that compares `token_count`; reconcile against new `usage` field (see equality note under Structural considerations).
- `crates/adapter-llm/src/openrouter/` — OpenRouter HTTP/SSE implementation. The new `OpenRouterCatalog` belongs here (sibling module). Reuse the existing HTTP client and auth header setup.
- `crates/adapter-llm/src/openrouter/wire.rs:67` — `SseChunk.usage` is already parsed; no change needed.
- `crates/app/src/` — add a `ports::model_catalog` module (new) next to the other ports. Re-export via `lib.rs`.
- `crates/bin-web/src/main.rs` — startup composition root. This is where `OpenRouterCatalog::new().await?` gets called and the resulting catalog is passed into the registry.
- `crates/bin-web/src/registry.rs` / `session.rs` — per-session state. Store the resolved `context_window: u32` alongside the model string when a session is created or restored.
- `crates/bin-web/src/routes.rs` — session HTTP endpoints; include `context_window` in the JSON bodies.
- `crates/bin-web/assets/index.html:20` — Cancel/Merge/Abandon button cluster; chip markup goes immediately before the Cancel button.
- `crates/bin-web/assets/app.js:110` — the "Usage is not rendered in V1; ignore" site. Replace with actual rendering.
- `crates/bin-web/assets/styles.css` — add `.usage-chip` styles consistent with the existing session header.
- `experiments/openrouter_models.json` — pre-captured `/models` response. Use it as a fixture for unit-testing `OpenRouterCatalog` parsing; also useful as a reference for the JSON shape.
- `~/src/ox/src/Ox/OxApp.cs:662` and `~/src/ox/src/Ox/Views/InputStatusFormatter.cs:19` — C# reference implementations of "compute percent after TurnCompleted" and "render `{percent}% {model}`". The Rust version mirrors the numerator and formula; only the rendering differs.

## Current state

- Relevant existing behavior:
  - `StreamEvent::Finished { usage }` already carries OpenRouter's usage numbers end-to-end.
  - `StreamAccumulator` already accumulates `usage` internally; it just doesn't surface it in `into_message`.
  - `Message.token_count: usize` stores `usage.completion_tokens` (wrong half) and is unused by the UI.
  - SSE delivery of `AgentEvent`s to the browser is already JSON-serialized via serde; no encoding work.
  - Session reload replays the full `Session.messages` history over SSE; whatever we put on `Message` is available on reload for free.
- Existing patterns to follow:
  - Ports-and-adapters: app defines traits, adapter crates implement them. `LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, `Shell` are templates. `ModelCatalog` should follow the same shape.
  - `bin-web` is the sole composition root for the web side. All adapter wiring (including the new catalog fetch) belongs in `main.rs` there.
  - No-build frontend: hand-written JS/HTML/CSS, no bundler. Changes go straight in `assets/`.
- Constraints from the current implementation:
  - The agent subprocess already runs with `--model` and talks only NDJSON frames defined in `protocol`. Keep the catalog out of the agent; it's a UI/host concern.
  - `protocol` depends only on `domain`, not on `app`. The `Message` struct is shared across the wire, so any change to `Message` ripples through both disk and IPC serialization. Greenfield — no migration, but all three sides (agent, host, disk) need recompiling cleanly.

## Structural considerations

- **Hierarchy.** The catalog is a read-only, host-scoped service. It fits at the app/adapter boundary cleanly: trait in `app`, implementation in `adapter-llm`, wired from `bin-web`. No new layer.
- **Abstraction.** `ModelCatalog` has one method (`context_window`). Resist the urge to generalize into a richer "ModelMetadata" provider now — YAGNI. Add fields when a second caller needs them.
- **Modularization.** The catalog fetch lives in a new `openrouter/catalog.rs` module inside `adapter-llm`. Don't stuff it into the existing `openrouter/mod.rs` — it's a distinct concern (one-shot discovery vs. per-turn streaming) with different failure semantics. Same crate, same client credentials, separate module.
- **Encapsulation.** `Message.usage` is a full `Usage` struct, not a single `u32`. Callers who only want `prompt_tokens` do `message.usage.as_ref().map(|u| u.prompt_tokens)`. Keeping the struct whole avoids information loss for the "per-turn chart" follow-ups and keeps the domain boundary clean.
- **Testability.** The catalog takes a response body (string or bytes) and returns a map. Write it as `OpenRouterCatalog::parse(&str) -> Result<HashMap<String, u32>, CatalogError>` so the parse is testable without a live HTTP call; the `fetch_and_parse` function just composes `reqwest` with `parse`. Use `experiments/openrouter_models.json` as the parse fixture.
- **Port placement — `app` vs `agent-host`.** `app` already defines the `LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, and `Shell` ports — a single flat `crates/app/src/ports.rs`. Add `ModelCatalog` there, alongside the others, for consistency even though no current use case consumes it. Rationale: `app` is the home for port traits in this workspace; putting a fresh port in `agent-host` would fork the pattern for future readers. The only consumer today is `bin-web`, and that's fine — ports don't have to be used by use cases to live in `app`. If a second provider (e.g. Ollama) later needs a catalog, the trait is already in the right place.
- **`context_window` on HTTP vs `AgentEvent::Ready`.** `AgentEvent::Ready { session_id, workspace_root }` is a plausible-looking carrier, but the agent subprocess has no need for the context window — it never computes it, never compares against it, never echoes it meaningfully. Threading it through the protocol would require either (a) the host sending `--context-window N` as a CLI arg just so the agent can echo it back, or (b) adding an extra NDJSON frame from host → agent. Both add round-trips for a value the host already knows from the catalog. The HTTP session-load endpoint already returns per-session JSON to the browser, so adding one more field there is the cheapest, most locally scoped change. Keep the protocol stable; keep UI-display metadata on the HTTP side.
- **Message equality at `pump.rs:216`.** The pump currently compares `left.token_count == right.token_count` as part of message equality. With the rename, this becomes `left.usage == right.usage`. `Usage` derives `PartialEq` so this just works, but the semantics shift subtly — `None` vs `Some(Usage::default())` are now distinct. Audit the pump's use of this equality (is it comparing committed state against a broadcast?) and confirm that `None` on both sides is the initial state. If the old code was specifically relying on `0 == 0`, the new equivalent is `None == None`, which holds. If it expected `token_count` to always be populated with a number, the equivalent is `usage.unwrap_or_default() == usage.unwrap_or_default()`. Decide during implementation.

## Refactoring

A single focused refactor lands before the feature:

1. **Rename `Message.token_count` → `Message.usage`.** Drop the `usize` field; add `usage: Option<Usage>`. Concrete call sites (from a workspace-wide grep for `token_count`):
   - `crates/domain/src/message.rs:17, 25, 33, 49` — struct field and three constructor defaults.
   - `crates/domain/src/session.rs:67` — `is_over_limit` currently sums `m.token_count`. Either delete the method (check whether anything calls it) or reimplement as a sum over `m.usage.as_ref().map(|u| u.prompt_tokens + u.completion_tokens + u.reasoning_tokens).unwrap_or(0)`. Preference: delete unless something actually calls it; a stale method on the domain type is noise.
   - `crates/domain/src/stream/snapshot.rs:10` — `Snapshot.token_count: usize`. Replace with `usage: Option<Usage>` (or just `usage: Usage` since the accumulator always has one; decide based on callers).
   - `crates/domain/src/stream/accumulator.rs:100, 114` — `into_message` and `from_snapshot` paths; both set `usage: Some(self.usage)` instead of the old `token_count`.
   - `crates/domain/src/stream/accumulator.rs:187, 420, 520, 523, 606, 682, 703` — in-crate tests; update assertions to read `msg.usage.unwrap().completion_tokens` (or equivalent).
   - `crates/agent-host/src/session_runtime.rs:249` — test assertion on `snap.token_count`; update alongside the `Snapshot` field change.
   - `crates/adapter-llm/src/openrouter/wire.rs:316` — constructs a `Message { .., token_count: 0 }`; change to `usage: None`.
   - `crates/bin-web/src/session/pump.rs:216` — message equality check; see equality discussion in Structural considerations.

No other pre-feature refactors. The existing port/adapter structure already gives a clean slot for `ModelCatalog`.

## Research

### Repo findings

- The `Usage` struct (`domain/src/stream/mod.rs:52`) carries `prompt_tokens`, `completion_tokens`, `reasoning_tokens`. All `u32`, all serde-derived.
- OpenRouter's streaming SSE parser already extracts `usage` from the final chunk (`adapter-llm/src/openrouter/sse.rs:120`). `stream_options: { include_usage: true }` is **not** set and is **not needed** — OpenRouter emits usage in the final chunk unconditionally in the current request shape.
- The browser-side `StreamAccumulator` JS mirror in `app.js:62-147` receives the `Finished` event with usage and explicitly discards it with a comment.
- `Session` on disk is a single JSON per session with the full `messages` vector. Changing `Message` changes the on-disk schema — greenfield, so no migration, but every existing ad-hoc session JSON becomes unreadable. Acceptable per project rules (no users, no backwards compat).
- `AgentEvent::Ready { session_id, workspace_root }` already exists; it does *not* carry the model, and the agent subprocess doesn't need to know the context window. Keep `context_window` out of the protocol — thread it through the host's HTTP layer only.
- `bin-web/assets/index.html:20-22` — the three buttons are siblings inside `.session-actions`. The chip slots in as a new sibling before `<button class="cancel">`.

### External research

- **OpenRouter `/api/v1/models`:** returns a top-level array (no pagination) where each entry has `id` (matches what `OpenRouterProvider` sends as `model`) and `context_length: u32`. `experiments/openrouter_models.json` is a real captured response and contains hundreds of entries; parsing is trivial `serde_json`. Authentication is not required for this endpoint in current OpenRouter behavior, but the adapter already has credentials — send them anyway to avoid a future breaking surprise.
- **C# reference (`~/src/ox`):** `OxApp.cs:662-672` computes `contextPercent = tokens * 100 / contextWindow` on every `TurnCompleted`, using the *last* LLM call's `InputTokenCount`. `UrSession.ExtractLastInputTokens()` walks the JSONL history on session load to restore the number. The Rust port replicates the numerator definition (last assistant message's `prompt_tokens`) and the "restore from history" behavior (read the last assistant `Message.usage.prompt_tokens` on reload).
- **Formatting:** the user's example (`100,000 / 200,000 (50%)`) implies comma-thousands. Use `Intl.NumberFormat('en-US')` in JS; no server-side formatting.

## Test plan

- **Key behaviors to verify (public-interface assertions):**
  - Given an OpenRouter `/models` JSON body, `OpenRouterCatalog::parse` produces a map whose entries match `(id, context_length)` and whose size matches the number of models in the body.
  - Given a known model id, `ModelCatalog::context_window(&id)` returns `Some(n)` with the parsed value.
  - Given an unknown model id, `ModelCatalog::context_window(&id)` returns `None`.
  - `StreamAccumulator::into_message()` produces a `Message` whose `usage` matches the `Finished { usage }` event it saw.
  - A `Session` serialized with `Message.usage = Some(...)` round-trips through disk unchanged.
  - An HTTP session-load response includes the session's `context_window` as an integer, resolved from the model string via the injected catalog.

- **Test levels:**
  - Unit (`adapter-llm`): `OpenRouterCatalog::parse` against the `experiments/openrouter_models.json` fixture. Pure function; no async, no network.
  - Unit (`domain`): `Message` serde round-trip with `Some` and `None` usage. `StreamAccumulator::into_message` usage wiring.
  - Unit (`bin-web` or integration style): route handler returns the expected JSON shape given a stub `ModelCatalog`. Use a hand-rolled fake catalog (a `HashMap` wrapper) — ports make this trivial without any mocking framework.
  - Manual (browser): chip renders in the right spot, updates after a turn, formats with commas, survives a hard reload.

- **Edge cases and failure modes:**
  - `/models` fetch fails (network, 5xx, malformed JSON): `bin-web` exits non-zero with a message identifying the endpoint and the error. Verify with a unit test that calls the parse function with garbage input and asserts the error type.
  - Session references a model id not present in the catalog: session creation/restore fails loudly. Test by constructing a catalog without the model and asserting the registry's session-create path returns an error.
  - Turn with no `usage` in the stream (hypothetical — some models): `StreamAccumulator::into_message` yields `Message { usage: None, .. }`; the chip keeps the previous displayed value (or `0 / X (0%)` if this is the first turn). Covered by the "None" branch of the domain round-trip test.
  - `prompt_tokens > context_window` (model somehow overran, or the catalog is stale): chip renders truthfully, e.g. `210,000 / 200,000 (105%)`. No clamping. Verified manually; no test needed for a cosmetic case.
  - Multi-iteration turn (tool calls → another LLM call): every intermediate assistant message gets its own `Finished` and its own `usage`; the chip naturally reflects the *latest* one because it reads from the latest assistant message. No explicit test — falls out of the per-message storage choice.
  - Pre-first-turn reload (session has no assistant messages yet): chip shows `0 / X (0%)`. Covered by the route handler test with an empty message history.

- **What NOT to test:**
  - Happy-path "did the chip update after a turn?" via a mocked browser. The path has many moving parts (SSE → JS accumulator → DOM); a test would mostly re-implement the code. Manual verification is cheaper.
  - Serde correctness of `Usage` itself — already covered implicitly by existing streaming tests.
  - `reqwest` / Axum framework guarantees.

## Implementation plan

### Refactor first

- [ ] `crates/domain/src/message.rs` — replace `pub token_count: usize` with `pub usage: Option<Usage>`. Update the three constructor defaults (lines ~25, 33, 49) to set `usage: None`. Import `Usage` from `crate::stream`.
- [ ] `crates/domain/src/session.rs:67` — either delete `is_over_limit` (preferred if no external caller) or reimplement against `m.usage`. Grep for callers first; if none, drop it.
- [ ] `crates/domain/src/stream/snapshot.rs` — replace `pub token_count: usize` with `pub usage: Usage` (not `Option` — the accumulator always has one, and callers want direct access for live rendering).
- [ ] `crates/domain/src/stream/accumulator.rs` — update `into_message()` (~line 100) and `from_snapshot()` (~line 114) to set `usage: Some(self.usage)` on the produced `Message`. Update `snapshot()` to carry `usage: self.usage.clone()` (or copy — `Usage` derives `Copy` if possible; check). Update all internal test assertions (lines 187, 420, 520, 523, 606, 682, 703) to read the new fields.
- [ ] `crates/agent-host/src/session_runtime.rs:249` — update `assert_eq!(snap.token_count, 2)` to the new shape, e.g. `assert_eq!(snap.usage.completion_tokens, 2)`.
- [ ] `crates/adapter-llm/src/openrouter/wire.rs:316` — `token_count: 0` → `usage: None`.
- [ ] `crates/bin-web/src/session/pump.rs:216` — update equality from `left.token_count == right.token_count` to `left.usage == right.usage`. Read the surrounding block; confirm `None == None` is the correct initial-state semantic. If the code was relying on `0 == 0`, this is equivalent; if it was relying on "both sides populated", swap to `.unwrap_or_default()` on both sides.
- [ ] `cargo build --workspace` green. `cargo test --workspace` green — any remaining `token_count` assertions fail noisily and get updated or deleted.

### Model catalog port

- [ ] Add the `ModelCatalog` trait to the existing `crates/app/src/ports.rs` (single flat file — do not create a `ports/` subdirectory). Signature: `pub trait ModelCatalog: Send + Sync { fn context_window(&self, model: &str) -> Option<u32>; }`. Re-export from `lib.rs` alongside the other ports.
- [ ] No use case needs the trait yet (it's a UI-side concern); the trait exists so `bin-web` can depend on an abstraction instead of `adapter-llm`'s concrete type, and so a future Ollama/alternate-provider catalog has an obvious slot.

### OpenRouter catalog adapter

- [ ] Create `crates/adapter-llm/src/openrouter/catalog.rs`. Define:
  - A pure `pub fn parse(body: &str) -> Result<HashMap<String, u32>, CatalogError>` that decodes the `/models` array and pulls `id` + `context_length` (skip entries missing `context_length` — log, don't fail).
  - A `pub struct OpenRouterCatalog { entries: HashMap<String, u32> }` implementing `app::ModelCatalog`.
  - `impl OpenRouterCatalog { pub async fn fetch(client: &reqwest::Client, base_url: &str, api_key: &str) -> Result<Self, CatalogError> }`.
- [ ] `CatalogError` enum covers `Transport`, `Status(u16)`, `Parse(serde_json::Error)`, `EmptyCatalog`. `thiserror` if the crate already uses it; otherwise plain enum with `Display`.
- [ ] Unit test `parse` against `experiments/openrouter_models.json` — assert at least one known model (e.g. `meta-llama/llama-3.1-8b-instruct` → `16384`) resolves correctly, assert map size matches the JSON array length.
- [ ] Unit test `parse` against `"{}"` and `"not json"` — assert `Err(_)` of the expected kind.

### Host wiring

- [ ] In `crates/bin-web/src/main.rs`, after resolving config but before starting the Axum server, construct the HTTP client and call `OpenRouterCatalog::fetch(...).await`. On error, `eprintln!` the full error chain and return a non-zero exit. Do not `unwrap` — emit a helpful message naming the endpoint.
- [ ] Pass the catalog (as `Arc<dyn ModelCatalog>`) into the session registry / router state.
- [ ] In the registry or session struct, when a session is created or restored, resolve `context_window = catalog.context_window(&model)`. If `None`, refuse to create/restore the session with a clear error message naming the model.
- [ ] Include `context_window: u32` in every HTTP response body that returns session metadata. Concretely: the response for `GET /api/sessions/{id}`, the response for `POST /api/sessions`, and any list endpoint that returns per-session details. Update the corresponding response structs in `routes.rs`.
- [ ] Do **not** change `protocol::AgentEvent`. Context window is a host/UI concern — the agent subprocess has no reason to know it.

### Frontend rendering

- [ ] `crates/bin-web/assets/index.html` — inside `.session-actions`, add `<span class="usage-chip" hidden>— / — (—)</span>` immediately before the Cancel button. `hidden` by default; JS reveals it once we have numbers.
- [ ] `crates/bin-web/assets/styles.css` — add `.usage-chip` styles: monospace-ish, subdued color, fixed height matching the button cluster, margin-right before the buttons. Match the existing header typography; no new palette.
- [ ] `crates/bin-web/assets/app.js`:
  - Store `contextWindow` on the session-state object at load time (from the HTTP response).
  - Delete the "Usage is not rendered in V1; ignore" comment at `app.js:110` and surrounding no-op.
  - On each `MessageAppended` event, if the message is an assistant message with non-null `usage`, update the session's cached `lastUsage`. Re-render the chip.
  - Formatting: `new Intl.NumberFormat('en-US').format(n)` for both numbers. Percent rounded to integer; `Math.round(used / window * 100)`. Render `${used} / ${window} (${pct}%)`.
  - On session load/reload, initialize `lastUsage` by walking the replayed history once to find the last assistant message with `usage`. If none, render `0 / ${window} (0%)`.
  - Unhide the chip as soon as the session has a `contextWindow` (i.e. immediately after load).

### Validation

- [ ] `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` all green.
- [ ] Run `cargo run -p bin-web`. Confirm: startup logs show the catalog was fetched; chip appears in the UI for a fresh session; chip updates after a turn; formatting is `100,000 / 200,000 (50%)`.
- [ ] Kill the web host, hard-reload the browser tab: chip is empty (no server). Restart the host: chip returns with the correct last-turn numbers (restored from `Session.messages.last().usage`).
- [ ] Simulate `/models` failure: block the endpoint at the network layer (or point `OPENROUTER_BASE_URL` at a dead host) and confirm `bin-web` exits non-zero with a readable message.
- [ ] Create a session with a model id not present in the catalog (temporarily hardcode something bogus). Confirm session creation is rejected with an error identifying the model.

## Impact assessment

- **Code paths affected:** `domain::Message` schema (breaking, greenfield-OK); `StreamAccumulator::into_message`; new port in `app`; new module in `adapter-llm/openrouter`; `bin-web` startup, registry, routes, and assets.
- **Data or schema impact:** `Session` JSON on disk gains `usage` per assistant message; existing on-disk sessions won't deserialize. Acceptable — project rule: no back-compat, no users. Any local `~/.ox/workspaces/*/sessions/*.json` will be invalid and must be deleted.
- **Dependency or API impact:** None new. `reqwest` and `serde_json` are already used. No new workspace dependencies.
- **Protocol wire impact:** None. `AgentEvent` is unchanged. The browser gets `context_window` via HTTP, not via SSE.

## Validation

- Tests: unit tests for `OpenRouterCatalog::parse`, for `Message` serde round-trip with `Some` and `None` usage, for the route handler returning `context_window`.
- Lint/format/typecheck: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Manual verification: fresh-session chip, post-turn chip update, reload restore, forced `/models` failure exits the host, unknown-model session creation is rejected.

## Gaps and follow-up

- Gap: Ollama or other non-OpenRouter providers have no `ModelCatalog` implementation. Fine for now — `adapter-llm` is OpenRouter-only per `AGENTS.md`; Ollama is "intentionally deferred."
- Gap: The catalog is fetched once at startup and never refreshed. If OpenRouter publishes a new model mid-session, restart is required. Acceptable; document in a future Ollama/provider-config follow-up.
- Follow-up: A per-turn usage-over-time chart becomes trivial once `Message.usage` is preserved per-message. Not in scope here.
- Follow-up: Cost estimation. OpenRouter `/models` also returns `pricing.prompt` / `pricing.completion`. A natural extension, but YAGNI for now — the user asked for context usage, not dollars.
