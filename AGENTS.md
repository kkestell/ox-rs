# AGENTS.md

Ox is a terminal-based AI coding assistant built in Rust. It uses a hexagonal (ports-and-adapters) architecture to stream LLM responses through a TUI, with pluggable backends for model providers, session persistence, filesystem access, and secret management.

## Tech Stack

- **Language:** Rust (edition 2024)
- **Build system:** Cargo (workspace with 7 internal crates + root binary)

## Codebase Map

- `src/main.rs` — Binary entry point; composition root (stub — not yet wiring adapters)
- `crates/domain/` — Core types: Session, Message, ContentBlock, Role, SessionId, SessionSummary
- `crates/app/` — Application layer: port traits (LlmProvider, SessionStore, SecretStore, FileSystem, Shell), use cases (ContinueSession), streaming (StreamEvent, StreamAccumulator, ToolDef)
- `crates/adapter-llm/` — LLM provider implementations: OpenRouter (streaming via SSE), Ollama (stub)
- `crates/adapter-storage/` — Session persistence: DiskSessionStore (stub)
- `crates/adapter-tui/` — Terminal UI (stub)
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
    adapters["Adapters<br/>adapter-tui · adapter-llm · adapter-storage · adapter-fs · adapter-secrets"]
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
    participant UC as ContinueSession
    participant SS as SessionStore
    participant LP as LlmProvider
    participant OR as OpenRouterProvider
    participant SSE as SSE parser
    participant ACC as StreamAccumulator

    UC->>SS: load(session_id)
    SS-->>UC: Session
    UC->>LP: stream(messages, tools=[])
    LP->>OR: provider call
    OR->>SSE: parse response body
    SSE-->>UC: StreamEvent

    loop each streamed event
        UC->>ACC: push(event)
    end

    ACC-->>UC: completed Message
    UC->>SS: save(updated session)
```

Current status:
- `src/main.rs`: stub binary, no composition root yet.
- `adapter-tui`: stub, not wired to `ContinueSession`.
- `adapter-llm/OpenRouterProvider`: implemented streaming path.
- `adapter-llm/OllamaProvider`: stub.
- `adapter-storage/DiskSessionStore`: constructor only, load/save/list are stubs.
- `adapter-fs/LocalFileSystem`: implemented.
- `adapter-fs/BashShell`: stub.
- `adapter-secrets/EnvSecretStore`: implemented.

### Target Runtime Path

What a full end-to-end turn should look like once wiring is complete.

```mermaid
sequenceDiagram
    participant M as main.rs
    participant T as TuiApp
    participant UC as ContinueSession
    participant SS as SessionStore
    participant LP as LlmProvider
    participant ACC as StreamAccumulator

    M->>T: wire adapters and start app
    T->>UC: submit user input
    UC->>SS: load session
    UC->>LP: stream model response

    loop each streamed event
        UC->>ACC: push(event)
        ACC-->>T: incremental UI state
    end

    UC->>SS: save updated session
    UC-->>T: completed assistant message
```
