# AGENTS.md

Ox is a desktop AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture and a two-process design: a native GPUI GUI (`ox`) spawns one or more headless agent subprocesses (`ox-agent`) and talks to each over NDJSON on stdin/stdout. Pluggable backends cover model providers, session persistence, filesystem access, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024).
- **Build system:** Cargo (virtual workspace — 8 library crates + 2 binary crates).
- **GUI framework:** GPUI for the native desktop surface.
- **IPC:** NDJSON over stdin/stdout between the GUI process and each agent subprocess (tokio pipes, `kill_on_drop` for child lifetime).

## Codebase Map

- `crates/domain/` — Core types: `Session`, `Message`, `ContentBlock`, `Role`, `SessionId`, `SessionSummary`, `StreamEvent`, `Usage`. All serde-derived so the same shapes serialize to disk *and* cross the GUI↔agent wire.
- `crates/app/` — Application layer: port traits (`LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, `Shell`), use cases (`SessionRunner`, `TurnEvent`, `TurnOutcome`), cancellation (`CancelToken`), streaming (`StreamAccumulator`, `ToolDef`), tools (`Tool` trait, `ToolRegistry`, `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GlobTool`, `GrepTool`, `BashTool`, hashline helpers, spill-to-file utility). Re-exports `StreamEvent`/`Usage` from domain for caller convenience.
- `crates/protocol/` — Wire protocol between the GUI process and `ox-agent`: `AgentCommand`, `AgentEvent`, and `read_frame`/`write_frame` helpers. Depends only on `domain` — no dep on `app` so the wire types cannot accidentally leak application-layer concerns.
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter (streaming via SSE), Ollama (stub).
- `crates/adapter-storage/` — Session persistence: `DiskSessionStore` (one JSON file per session).
- `crates/adapter-fs/` — Filesystem and shell: `LocalFileSystem` (implemented), `BashShell` (implemented — spawns `/bin/bash -c`, concurrent stdout/stderr capture, timeout via `tokio::time::timeout`, kill on timeout, byte-capped output with pipe draining).
- `crates/adapter-secrets/` — Secret retrieval: `EnvSecretStore` (implemented).
- `crates/agent-host/` — Host library: `WorkspaceState` (multi-split workspace), `AgentSplit` (per-agent state), `AgentClient` / `AgentEventStream` (IPC client over stdio with reader/writer tasks), `AgentSpawnConfig`, `WorkspaceLayouts` (layout persistence), `classify_input` / `SplitAction` (slash-command parser). Depends on `app`, `domain`, `protocol`; implements no app ports.
- `crates/bin-gpui/` — `ox` binary: native GPUI composition root for the GUI process. Owns rendering, input, menu actions, modal state, and workspace controller methods; delegates subprocess and split state to `agent-host`.
- `crates/bin-agent/` — `ox-agent` binary: composition root for the agent process. Parses CLI, wires adapters, builds a `SessionRunner`, and hands control to `driver::agent_driver` which drives NDJSON I/O over stdin/stdout.
- `experiments/` — Throwaway scripts for testing provider APIs.
- `docs/` — Research and design notes.

## Commands

- Build (Rust workspace): `cargo build`
- Run the GUI: `cargo run -p bin-gpui`
- Run the agent headless: `cargo run -p bin-agent -- --workspace-root … --model … --sessions-dir …`
- Test: `cargo test`
- Test (single crate): `cargo test -p <crate-name>`
- Lint: `cargo clippy --workspace --all-targets`
- Format: `cargo fmt`

## Project Rules

- This is greenfield development. There are no users. There are no backwards compatibility concerns.
- Nothing is pre-existing. All builds and tests are green upstream. If something fails, your work caused it. Investigate and fix — never dismiss a failure as pre-existing.
- Use `cargo add` for third-party dependencies -- never hand-edit `[dependencies]` in Cargo.toml. 
- Commits must follow the 7 rules of great commit messages with NO Claude Code attribution.
