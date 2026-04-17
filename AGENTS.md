# AGENTS.md

Ox is a desktop AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture and a two-process design: a Tauri GUI (`ox-tauri`, Rust shell + TypeScript webview) spawns one or more headless agent subprocesses (`ox-agent`) and talks to each over NDJSON on stdin/stdout. Pluggable backends cover model providers, session persistence, filesystem access, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024) for the backend; TypeScript for the webview.
- **Build system:** Cargo (virtual workspace — 8 library crates + 2 binary crates); Vite for the frontend bundle.
- **GUI framework:** Tauri 2 (WebKitGTK on Linux) with a hand-rolled TypeScript UI — no UI framework, direct DOM.
- **IPC:** NDJSON over stdin/stdout between the GUI process and each agent subprocess (tokio pipes, `kill_on_drop` for child lifetime); Tauri `invoke` / `emit` between the Rust shell and the webview.

## Codebase Map

- `crates/domain/` — Core types: `Session`, `Message`, `ContentBlock`, `Role`, `SessionId`, `SessionSummary`, `StreamEvent`, `Usage`. All serde-derived so the same shapes serialize to disk *and* cross the GUI↔agent wire.
- `crates/app/` — Application layer: port traits (`LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, `Shell`), use cases (`SessionRunner`, `TurnEvent`, `TurnOutcome`), cancellation (`CancelToken`), streaming (`StreamAccumulator`, `ToolDef`), tools (`Tool` trait, `ToolRegistry`, `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GlobTool`, `GrepTool`, `BashTool`, hashline helpers, spill-to-file utility). Re-exports `StreamEvent`/`Usage` from domain for caller convenience.
- `crates/protocol/` — Wire protocol between the GUI process and `ox-agent`: `AgentCommand`, `AgentEvent`, and `read_frame`/`write_frame` helpers. Depends only on `domain` — no dep on `app` so the wire types cannot accidentally leak application-layer concerns.
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter (streaming via SSE), Ollama (stub).
- `crates/adapter-storage/` — Session persistence: `DiskSessionStore` (one JSON file per session).
- `crates/adapter-fs/` — Filesystem and shell: `LocalFileSystem` (implemented), `BashShell` (implemented — spawns `/bin/bash -c`, concurrent stdout/stderr capture, timeout via `tokio::time::timeout`, kill on timeout, byte-capped output with pipe draining).
- `crates/adapter-secrets/` — Secret retrieval: `EnvSecretStore` (implemented).
- `crates/agent-host/` — Host library: `WorkspaceState` (multi-split workspace), `AgentSplit` (per-agent state), `AgentClient` / `AgentEventStream` (IPC client over stdio with reader/writer tasks), `AgentSpawnConfig`, `WorkspaceLayouts` (layout persistence), `classify_input` / `SplitAction` (slash-command parser). Depends on `app`, `domain`, `protocol`; implements no app ports.
- `crates/bin-tauri/` — `ox-tauri` binary: composition root for the GUI process. Embeds a Tauri webview that loads the TypeScript bundle under `crates/bin-tauri/ui/`, exposes `invoke` commands that call into `agent-host`, forwards `AgentEvent` streams to the webview, and wires the native menu. On `run()` it spawns the initial agent subprocess and (on File > Open... or `/new`) additional ones using a cloneable `AgentSpawnConfig` template.
- `crates/bin-tauri/ui/` — TypeScript/Vite frontend for `ox-tauri`. Hand-rolled DOM — no UI framework. Owns the transcript renderer, split layout, input handling, and a TypeScript mirror of `StreamAccumulator` for live streaming. The webview is the only place `user-select` is enabled, which is the whole reason the egui frontend was retired.
- `crates/bin-agent/` — `ox-agent` binary: composition root for the agent process. Parses CLI, wires adapters, builds a `SessionRunner`, and hands control to `driver::agent_driver` which drives NDJSON I/O over stdin/stdout.
- `experiments/` — Throwaway scripts for testing provider APIs.
- `docs/` — Research and design notes.

## Commands

- Build (Rust workspace): `cargo build`
- Build (frontend bundle, required before `cargo run -p bin-tauri`): `cd crates/bin-tauri/ui && npm run build`
- Run the GUI: `cargo run -p bin-tauri` (or `cargo tauri dev` from `crates/bin-tauri/` for HMR on the webview)
- Run the agent headless: `cargo run -p bin-agent -- --workspace-root … --model … --sessions-dir …`
- Test: `cargo test`
- Test (single crate): `cargo test -p <crate-name>`
- Lint: `cargo clippy --workspace --all-targets`
- Format: `cargo fmt`
- Frontend type-check: `cd crates/bin-tauri/ui && npx tsc --noEmit`

## Project Rules

- This is greenfield development. There are no users. There are no backwards compatibility concerns.
- Nothing is pre-existing. All builds and tests are green upstream. If something fails, your work caused it. Investigate and fix — never dismiss a failure as pre-existing.
- Use `cargo add` for third-party dependencies -- never hand-edit `[dependencies]` in Cargo.toml. 
- Commits must follow the 7 rules of great commit messages with NO Claude Code attribution.

## Architecture

Layered from top (composition) to bottom (pure types):

- **Binaries** (`bin-tauri`, `bin-agent`) — compose adapters and host libraries; `bin-agent` also depends on `protocol` directly. `bin-tauri` additionally hosts the Tauri webview and ships a TypeScript frontend under `crates/bin-tauri/ui/`.
- **Adapters** (`adapter-llm`, `adapter-storage`, `adapter-fs`, `adapter-secrets`) — depend on `app` ports.
- **Host libraries** (`agent-host`) — orchestrate one or more `ox-agent` subprocesses on behalf of a desktop binary. Depend on `app`, `domain`, `protocol`; implement no app ports, so they are not adapters in the ports-and-adapters sense. Separate layer beside adapters.
- **Application** (`app`) and **Wire protocol** (`protocol`) — siblings; both depend on `domain` and neither depends on the other. `protocol` deliberately avoids `app` so wire types can't drag in application behavior.
- **Domain** (`domain`) — no internal deps.

`bin-tauri` doesn't depend on `bin-agent` — the only coupling between the two processes is the `protocol` crate (and the wire-format mirrors in the TypeScript frontend).

(No diagrams. Mermaid can't lay out a 5-node DAG without crossing edges, so prose wins.)
