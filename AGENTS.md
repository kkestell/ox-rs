# AGENTS.md

Ox is a desktop AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture to stream LLM responses through an egui GUI, with pluggable backends for model providers, session persistence, filesystem access, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024)
- **Build system:** Cargo (workspace with 7 internal crates + root binary)
- **GUI framework:** egui (via eframe)

## Codebase Map

- `src/main.rs` — Binary entry point; composition root wiring adapters, tokio runtime, channels, GUI, and tool registry
- `crates/domain/` — Core types: Session, Message, ContentBlock, Role, SessionId, SessionSummary
- `crates/app/` — Application layer: port traits (LlmProvider, SessionStore, SecretStore, FileSystem, Shell), use cases (SessionRunner, TurnEvent), streaming (StreamEvent, StreamAccumulator, ToolDef), tools (Tool trait, ToolRegistry, ReadFileTool, WriteFileTool, EditFileTool, hashline helpers)
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter (streaming via SSE), Ollama (stub)
- `crates/adapter-storage/` — Session persistence: DiskSessionStore (stub)
- `crates/adapter-egui/` — GUI client: egui/eframe native window with channel-driven backend controller (`backend.rs`)
- `crates/adapter-fs/` — Filesystem and shell: LocalFileSystem (implemented), BashShell (stub)
- `crates/adapter-secrets/` — Secret retrieval: EnvSecretStore (implemented)
- `experiments/` — Throwaway scripts for testing provider APIs
- `docs/` — Research and design notes

## Commands

- Build: `cargo build`
- Run: `cargo run`
- Test: `cargo test`
- Test (single crate): `cargo test -p <crate-name>`
- Lint: `cargo clippy`
- Format: `cargo fmt`

## Project Rules

- This is greenfield development. There are no users. There are no backwards compatibility concerns.
- Nothing is pre-existing. All builds and tests are green upstream. If something fails, your work caused it. Investigate and fix — never dismiss a failure as pre-existing.
- Use `cargo add` for third-party dependencies -- never hand-edit `[dependencies]` in Cargo.toml. 
- Commits must follow the 7 rules of great commit messages with NO Claude Code attribution.

## Architecture

These diagrams must be kept up to date at all times.

Use separate diagrams for separate questions. Do not collapse structure, crate dependencies, and runtime flow into one graph.

### Structural Layers

The stable architectural shape.

```mermaid
flowchart TB
    main["Binary<br/>src/main.rs"]
    adapters["Adapters<br/>adapter-egui · adapter-llm · adapter-storage · adapter-fs · adapter-secrets"]
    app["Application<br/>ports · use_cases · stream"]
    domain["Domain<br/>Session · Message · ContentBlock · Role · SessionId · SessionSummary"]

    main --> adapters
    adapters --> app
    app --> domain
```

Notes:
- `app` depends on `domain`.
- Adapters depend on `app` ports.
- Some adapters also depend directly on `domain` types for translation and persistence.

### Current Runtime Path

What is actually implemented today.

```mermaid
sequenceDiagram
    participant M as main.rs
    participant G as OxApp (egui)
    participant BC as run_backend
    participant SR as SessionRunner
    participant SS as SessionStore
    participant LP as LlmProvider
    participant TR as ToolRegistry
    participant ACC as StreamAccumulator

    M->>SS: load(id) [--resume only]
    M->>G: create channels, spawn backend, launch GUI (with initial messages if resuming)
    G->>BC: BackendCommand::SendMessage (via channel)
    BC->>SR: start(id, workspace_root, input, on_event) or resume(id, input, on_event)
    SR->>SS: load session [resume] / create new [start]
    SR->>SR: append user message
    BC-->>G: BackendEvent::MessageAppended(user)

    loop until no tool calls or iteration cap
        SR->>LP: stream(messages, tool_defs)
        loop each streamed event
            SR->>BC: on_event(TurnEvent::StreamDelta)
            BC-->>G: BackendEvent::StreamDelta
            SR->>ACC: push(event)
        end
        ACC-->>SR: completed Message
        SR->>SR: append assistant message
        BC-->>G: BackendEvent::MessageAppended(assistant)
        alt assistant emitted tool calls
            loop each tool call
                SR->>TR: execute(name, args)
                TR-->>SR: Result<String>
                SR->>SR: append tool-result message
                BC-->>G: BackendEvent::MessageAppended(tool)
            end
        else no tool calls
            Note over SR: exit loop
        end
    end

    SR->>SS: save(updated session)
    BC-->>G: BackendEvent::TurnComplete
    Note over M: after GUI exits, print "ox --resume <id>" to stderr
```

Current status:
- `src/main.rs`: composition root with CLI parsing (`--resume <id>`), session pre-loading, adapter wiring, tool registration (read_file, write_file, edit_file), and resume-command output on exit.
- `adapter-egui`: channel-driven GUI with message display, text input, send button, event polling, and incremental streaming display for text, reasoning, and tool-call arguments. Accepts initial messages for session resume. `backend.rs` contains the `run_backend` controller and channel protocol types (`BackendCommand`; `BackendEvent::{StreamDelta, MessageAppended, TurnComplete, Error}`). Backend accepts an optional initial session ID and returns the final session ID.
- `app::tools`: file-editing tool suite — `read_file` (hashlined output with offset/limit), `write_file` (creates parent dirs), `edit_file` (replace/insert_after operations anchored by hashlines with mismatch detection). The `Tool` trait + `ToolRegistry` form the in-process execution contract.
- `adapter-llm/OpenRouterProvider`: implemented streaming path.
- `adapter-llm/OllamaProvider`: stub.
- `adapter-storage/DiskSessionStore`: implemented (load, save, list).
- `adapter-fs/LocalFileSystem`: implemented (now creates parent dirs on write).
- `adapter-fs/BashShell`: stub.
- `adapter-secrets/EnvSecretStore`: implemented.

Not yet implemented:
- Model/config selection (model is hardcoded).
- Session management UI (sessions can be resumed via CLI, but no in-app session browser/switcher).
- Error recovery UI (errors displayed but no retry/dismiss).
- Cancel/stop generation (no way to abort an in-progress stream or a long tool-call loop).
- Tool-approval flow (tools auto-execute; destructive tools have no permission gate).
- Bash tool (trait and registry are ready; `BashShell` is still a stub).
