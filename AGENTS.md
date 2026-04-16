# AGENTS.md

Ox is a desktop AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture and a two-process design: an egui GUI (`ox-gui`) spawns one or more headless agent subprocesses (`ox-agent`) and talks to each over NDJSON on stdin/stdout. Pluggable backends cover model providers, session persistence, filesystem access, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024)
- **Build system:** Cargo (virtual workspace — 8 library crates + 2 binary crates)
- **GUI framework:** egui (via eframe)
- **IPC:** NDJSON over stdin/stdout (tokio pipes, `kill_on_drop` for child lifetime)

## Codebase Map

- `crates/domain/` — Core types: `Session`, `Message`, `ContentBlock`, `Role`, `SessionId`, `SessionSummary`, `StreamEvent`, `Usage`. All serde-derived so the same shapes serialize to disk *and* cross the GUI↔agent wire.
- `crates/app/` — Application layer: port traits (`LlmProvider`, `SessionStore`, `SecretStore`, `FileSystem`, `Shell`), use cases (`SessionRunner`, `TurnEvent`, `TurnOutcome`), cancellation (`CancelToken`), streaming (`StreamAccumulator`, `ToolDef`), tools (`Tool` trait, `ToolRegistry`, `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GlobTool`, `GrepTool`, `BashTool`, hashline helpers, spill-to-file utility). Re-exports `StreamEvent`/`Usage` from domain for caller convenience.
- `crates/protocol/` — Wire protocol between `ox-gui` and `ox-agent`: `AgentCommand`, `AgentEvent`, and `read_frame`/`write_frame` helpers. Depends only on `domain` — no dep on `app` so the wire types cannot accidentally leak application-layer concerns.
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter (streaming via SSE), Ollama (stub).
- `crates/adapter-storage/` — Session persistence: `DiskSessionStore` (one JSON file per session).
- `crates/adapter-egui/` — GUI library: `OxApp` (egui root), `AgentSplit` (per-agent state), `AgentClient` (IPC client over stdio with reader/writer tasks), `AgentSpawnConfig`.
- `crates/adapter-fs/` — Filesystem and shell: `LocalFileSystem` (implemented), `BashShell` (implemented — spawns `/bin/bash -c`, concurrent stdout/stderr capture, timeout via `tokio::time::timeout`, kill on timeout, byte-capped output with pipe draining).
- `crates/adapter-secrets/` — Secret retrieval: `EnvSecretStore` (implemented).
- `crates/bin-gui/` — `ox-gui` binary: composition root for the GUI process. Parses CLI, locates the `ox-agent` binary, spawns the initial agent, passes a cloneable `AgentSpawnConfig` template to `OxApp` for `/new` and File > Open..., passes the package version for Help > About, runs the egui window. Prints `ox-gui --resume <id>` per active session on shutdown.
- `crates/bin-agent/` — `ox-agent` binary: composition root for the agent process. Parses CLI, wires adapters, builds a `SessionRunner`, and hands control to `driver::agent_driver` which drives NDJSON I/O over stdin/stdout.
- `experiments/` — Throwaway scripts for testing provider APIs.
- `docs/` — Research and design notes.

## Commands

- Build: `cargo build`
- Run: `cargo run -p bin-gui`
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

## Architecture

Layered from top (composition) to bottom (pure types):

- **Binaries** (`bin-gui`, `bin-agent`) — compose adapters; `bin-agent` also depends on `protocol` directly.
- **Adapters** (`adapter-egui`, `adapter-llm`, `adapter-storage`, `adapter-fs`, `adapter-secrets`) — depend on `app` ports; `adapter-egui` also depends on `protocol`.
- **Application** (`app`) and **Wire protocol** (`protocol`) — siblings; both depend on `domain` and neither depends on the other. `protocol` deliberately avoids `app` so wire types can't drag in application behavior.
- **Domain** (`domain`) — no internal deps.

`bin-gui` doesn't depend on `bin-agent` — the only coupling between the two processes is the `protocol` crate.

(No diagrams. Mermaid can't lay out a 5-node DAG without crossing edges, so prose wins.)
