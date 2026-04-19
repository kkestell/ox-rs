# AGENTS.md

Ox is a local web AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture and a two-process design: the `ox` web host (`bin-web`) runs an Axum server, serves a no-build static frontend, spawns one or more headless agent subprocesses (`ox-agent`), and talks to each agent over NDJSON on stdin/stdout. Pluggable backends cover model providers, session persistence, filesystem access, git/worktree operations, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024).
- **Build system:** Cargo (virtual workspace — 10 library crates + 2 binary crates).
- **Web framework:** Axum + Tokio in `bin-web`; static `index.html`, `app.js`, and `styles.css` are embedded into the binary with `include_str!` and served with `Cache-Control: no-store`.
- **Frontend:** Plain browser JavaScript, no build step. The client uses HTTP JSON endpoints for commands/layout and Server-Sent Events (SSE) for live `AgentEvent` updates.
- **IPC:** NDJSON over stdin/stdout between the web host process and each agent subprocess (tokio pipes, process adapter owns `kill_on_drop` child lifetime).

## Codebase Map

- `crates/domain/` — Core types: `Session`, `Message`, `ContentBlock`, `Role`, `SessionId`, `SessionSummary`, `StreamEvent`, `Usage`, `CloseIntent`, `ToolDef`, `StreamAccumulator`, `Snapshot`. All serde-derived so the same shapes serialize to disk *and* cross the host↔agent wire.
- `crates/app/` — Application layer: port traits (`LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, `Shell`, plus helper structs `WalkResult`, `CommandOutput`), use cases (`SessionRunner`, `TurnEvent`, `TurnOutcome`), cancellation (`CancelToken`), approval (`ToolApprover` trait, `ToolApprovalRequest`, `ToolApprovalDecision`, `ApprovalRequirement`, `MissingPathPolicy`, `NoApprovalRequired`), lifecycle (`CloseSignal`, `MergeTool`, `AbandonTool`), tools (`Tool` trait, `ToolRegistry`, `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GlobTool`, `GrepTool`, `BashTool`, `TodoWriteTool`, hashline helpers, spill-to-file utility).
- `crates/protocol/` — Wire protocol between the web host process and `ox-agent`: `AgentCommand`, `AgentEvent`, `ToolApprovalRequest`, and `read_frame`/`write_frame` helpers. Depends only on `domain` — no dep on `app` so the wire types cannot accidentally leak application-layer concerns.
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter streaming chat/SSE plus OpenRouter slug generation; Ollama remains intentionally deferred.
- `crates/adapter-storage/` — User-local storage adapters: `DiskSessionStore` (one JSON file per session) and `DiskLayoutRepository` (`~/.ox/workspaces.json` layout persistence).
- `crates/adapter-fs/` — Filesystem and shell: `LocalFileSystem` (implemented), `BashShell` (implemented — spawns `/bin/bash -c`, concurrent stdout/stderr capture, timeout via `tokio::time::timeout`, kill on timeout, byte-capped output with pipe draining).
- `crates/adapter-git/` — Git adapter: repository validation, current-branch discovery, worktree/branch management, merge/abandon support, and dirty/conflict detection through the git CLI.
- `crates/adapter-process/` — Process adapter implementing `agent-host::AgentSpawner` with `tokio::process::Command` and child lifetime management.
- `crates/adapter-secrets/` — Secret retrieval: `EnvSecretStore` (implemented).
- `crates/agent-host/` — Host library: `AgentClient` / `AgentEventStream` (IPC client over generic async I/O with reader/writer tasks), `AgentSpawnConfig`, `AgentSpawner`, `SessionRuntime` (with `ShouldSend`, `BeginClose`), `LayoutRepository`, `WorkspaceContext`, `Git` trait (`WorktreeStatus`, `MergeOutcome`), `SlugGenerator` trait, lifecycle sink traits (`CloseRequestSink`, `FirstTurnSink`), and layout/path helpers (`normalize_sizes`, `workspace_slug`). Depends on `domain`, `protocol` only — no dep on `app`. Framework-agnostic — `bin-web` owns HTTP, SSE, browser state, and lifecycle orchestration.
- `crates/bin-web/` — `ox` binary: Axum composition root for the local web UI. `main.rs` resolves CLI/env/paths, validates the git workspace, restores saved sessions, wires lifecycle + registry + router, binds `127.0.0.1:<port>`, and optionally opens the browser. `startup.rs` validates the git workspace before the server starts. `routes/` exposes session/message/cancel/layout/merge/abandon/static-asset endpoints, `sse.rs` streams `AgentEvent` history + live updates, `session/` owns per-session pump/runtime/history/broadcast state, `registry/` owns the live session map and layout persistence, `lifecycle/` coordinates merge/abandon/slug-rename/close-guard flows, `state.rs` holds shared Axum state, and `assets/` contains the embedded no-build frontend.
- `crates/bin-agent/` — `ox-agent` binary: composition root for the agent process. Parses CLI (`--workspace-root`, `--model`, `--sessions-dir`, `--resume`, `--session-id`), wires adapters, builds a `SessionRunner`, and hands control to `driver::agent_driver` which drives NDJSON I/O over stdin/stdout. The `driver/` module contains `turn.rs` (bridging sync callbacks to async writes), `reader.rs` (cancellation-safe frame reads), and `approval_broker.rs` (tool approval via oneshot channels).
- `experiments/` — Throwaway scripts for testing provider APIs.
- `docs/` — Research and design notes.

## Commands

- Build (Rust workspace): `cargo build`
- Run the web UI: `cargo run -p bin-web`
- Run the agent headless: `cargo run -p bin-agent -- --workspace-root … --model … --sessions-dir … [--resume <id> | --session-id <id>]`
- Test: `cargo test`
- Test (single crate): `cargo test -p <crate-name>`
- Lint: `cargo clippy --workspace --all-targets`
- Format: `cargo fmt`

## Project Rules

- This is greenfield development. There are no users. There are no backwards compatibility concerns.
- Nothing is pre-existing. All builds and tests are green upstream. If something fails, your work caused it. Investigate and fix — never dismiss a failure as pre-existing.
- Use `cargo add` for third-party dependencies -- never hand-edit `[dependencies]` in Cargo.toml.
- Commits must follow the 7 rules of great commit messages with NO Claude Code attribution.

### Git workspace

Ox only runs inside a git repository. On startup, `bin-web` calls
`git.assert_repo(&workspace_root)` and aborts with a targeted error if the
workspace is not a git repo (or has a detached HEAD). The repo's current
branch at startup is captured as the workspace's **base branch** — the
branch new session branches fork from and the branch `merge` folds them
back into.

Every session is scoped to its own git branch (`ox/<slug>-<short-uuid>`,
slug filled in after the first turn) inside its own worktree at
`~/.ox/workspaces/<workspace-slug>/worktrees/<short-uuid>/`. The agent
subprocess runs with `--workspace-root` pointed at the **worktree**, not
the base workspace, so tool edits are confined to the session's branch.
Closing a session takes one of two paths, both orchestrated by
`bin-web::lifecycle::SessionLifecycle`:

- **Merge** (`POST /api/sessions/{id}/merge`) folds the session branch into
  the base branch, removes the worktree, deletes the branch, and deletes
  the session JSON. Abort-on-conflict: a dirty worktree, a dirty base
  branch, or a conflicting merge leaves everything intact and surfaces a
  409 to the caller.
- **Abandon** (`POST /api/sessions/{id}/abandon`) discards the worktree,
  branch, and session JSON without merging. Dirty worktrees require
  `{"confirm": true}` — a bare POST returns a 409 that the UI turns into a
  native confirmation dialog.

Both paths are reachable from the UI (Merge / Abandon buttons) and from
inside the agent (the `merge` / `abandon` tools in `app::lifecycle`, which
trip an in-process `CloseSignal` that the driver drains at the end of the
turn and relays to the host as a `RequestClose` frame).
