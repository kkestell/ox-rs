// Hand-written TypeScript mirrors of the Rust types we receive over Tauri
// IPC. The shapes track the serde attributes in `crates/domain`,
// `crates/protocol`, and `crates/agent-host`. Fields that are optional on
// the Rust side (`Option<T>` with `skip_serializing_if = "Option::is_none"`)
// become `field?: T` here so we can read snapshots that omit them.
//
// Adding a variant to an internally-tagged enum in Rust must be mirrored
// here. The `assertNever` helper in `state.ts` turns missed variants into a
// TypeScript compile error instead of a silent no-op, which is the point.

// ---- Identifiers ---------------------------------------------------------

// `SessionId` (domain) and `SplitId` (agent-host) are both `Uuid` newtypes.
// `SessionId` uses `#[serde(transparent)]`; `SplitId` is a single-field
// tuple struct which serde serializes as the inner value by default. Both
// cross the wire as plain UUID strings.
export type SessionId = string;
export type SplitId = string;

// ---- Domain types --------------------------------------------------------

export type Role = "user" | "assistant" | "tool";

export type ContentBlock =
  | { type: "text"; text: string }
  | {
      type: "reasoning";
      content: string;
      signature?: string;
      encrypted?: string;
      format?: string;
    }
  | { type: "tool_call"; id: string; name: string; arguments: string }
  | {
      type: "tool_result";
      tool_call_id: string;
      content: string;
      is_error: boolean;
    };

export interface Message {
  role: Role;
  content: ContentBlock[];
  token_count: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number;
}

export type StreamEvent =
  | { type: "text_delta"; delta: string }
  | { type: "reasoning_delta"; delta: string }
  | { type: "encrypted_reasoning"; data: string; format: string }
  | { type: "tool_call_start"; index: number; id: string; name: string }
  | { type: "tool_call_argument_delta"; index: number; delta: string }
  | { type: "reasoning_signature"; signature: string }
  | { type: "finished"; usage: Usage };

// ---- Wire protocol types (protocol crate) --------------------------------

export type AgentEvent =
  | { type: "ready"; session_id: SessionId; workspace_root: string }
  | { type: "stream_delta"; event: StreamEvent }
  | { type: "message_appended"; message: Message }
  | { type: "turn_complete" }
  | { type: "turn_cancelled" }
  | { type: "error"; message: string };

// ---- Snapshot types (agent-host crate) -----------------------------------

// The authoritative view of a single split emitted by
// `WorkspaceState::snapshot`. `streaming` is the in-flight accumulator
// *flattened into `ContentBlock`s*, not the raw accumulator — the Rust side
// already assembled the blocks for us.
export interface SplitSnapshot {
  id: SplitId;
  session_id: SessionId | null;
  messages: Message[];
  streaming: ContentBlock[] | null;
  waiting: boolean;
  error: string | null;
  cancelled: boolean;
}

export interface Snapshot {
  splits: SplitSnapshot[];
  split_fracs: number[];
  focused: SplitId | null;
  workspace_root: string;
}

// ---- Command return types (bin-tauri crate) ------------------------------

// `SubmitOutcome` is `#[serde(tag = "kind", rename_all = "snake_case")]` on
// the Rust side, so the discriminator is `kind`, not `type`. Variants with
// no fields still carry `kind` so all shapes are uniform.
export type SubmitOutcome =
  | { kind: "sent_message" }
  | { kind: "new_split"; split_id: SplitId }
  | { kind: "closed_split" }
  | { kind: "cancel_requested" }
  | { kind: "quit_requested" };

export interface AppInfo {
  name: string;
  version: string;
}

// ---- Tauri event payload types -------------------------------------------

export interface AgentEventPayload {
  split_id: SplitId;
  event: AgentEvent;
}

export interface SplitAddedPayload {
  split_id: SplitId;
  snapshot: Snapshot;
}

export interface SplitClosedPayload {
  split_id: SplitId;
  snapshot: Snapshot;
}

export interface WorkspaceReplacedPayload {
  snapshot: Snapshot;
}

// ---- Exhaustiveness helper -----------------------------------------------

// Call in the `default` arm of a discriminated-union switch. If the compiler
// ever infers `never`, every variant is handled. If not, adding a variant
// without a matching case becomes a type error here, which is exactly where
// we want the drift between Rust and TS to surface.
export function assertNever(x: never): never {
  throw new Error(`unreachable: unexpected variant ${JSON.stringify(x)}`);
}
