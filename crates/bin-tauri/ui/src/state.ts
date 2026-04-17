// Client-side workspace store.
//
// Responsibilities:
// - Hold the authoritative view of each split (messages, streaming content,
//   waiting/error/cancelled flags) that the transcript renderer reads.
// - Mirror `AgentSplit::handle_event` in `applyAgentEvent` so incremental
//   `agent_event`s keep the UI in step with backend state.
// - Mirror `StreamAccumulator::assemble_blocks` in the `StreamAccumulator`
//   TS class so live-streaming renders match what the final committed
//   message will look like. The backend is authoritative; this is a
//   rendering-only duplication.
// - Broadcast a `changed(splitId)` signal so the renderer can re-draw just
//   the affected split instead of re-rendering the entire workspace on
//   every delta.

import type {
  AgentEvent,
  ContentBlock,
  Message,
  SessionId,
  Snapshot,
  SplitId,
  SplitSnapshot,
  StreamEvent,
} from "./types";
import { assertNever } from "./types";

// ---- StreamAccumulator TS mirror -----------------------------------------

// Mirrors `crates/app/src/stream.rs::StreamAccumulator`. The ordering rules
// in `assembleBlocks` (reasoning → text → tool calls by index) must stay in
// lockstep with the Rust implementation — if the two disagree, the live
// view flickers when the backend's committed `MessageAppended` replaces
// our live blocks.
//
// Drift risk is minimized by pinning the tests on the Rust side; the
// snapshot path in `SplitSnapshot::streaming` carries already-assembled
// blocks, so this accumulator is only exercised when we receive raw
// `StreamDelta` events. If bugs ever surface, the fix is to call
// `get_snapshot()` and rebuild; it's cheap.
export class StreamAccumulator {
  private text = "";
  private reasoning = "";
  private signature?: string;
  private encrypted?: { data: string; format: string };
  private toolCalls = new Map<
    number,
    { id: string; name: string; arguments: string }
  >();

  push(event: StreamEvent): void {
    switch (event.type) {
      case "text_delta":
        this.text += event.delta;
        break;
      case "reasoning_delta":
        this.reasoning += event.delta;
        break;
      case "encrypted_reasoning":
        this.encrypted = { data: event.data, format: event.format };
        break;
      case "tool_call_start":
        this.toolCalls.set(event.index, {
          id: event.id,
          name: event.name,
          arguments: "",
        });
        break;
      case "tool_call_argument_delta": {
        const tc = this.toolCalls.get(event.index);
        if (tc) tc.arguments += event.delta;
        // Orphan deltas for an index that was never started are silently
        // dropped — same as the Rust side.
        break;
      }
      case "reasoning_signature":
        this.signature = event.signature;
        break;
      case "finished":
        // Usage is informational for the transcript; we don't use it in the
        // live-rendering path. The committed `MessageAppended` frame carries
        // the authoritative token count.
        break;
      default:
        assertNever(event);
    }
  }

  // Build the ordered `ContentBlock[]` the renderer consumes. Mirrors
  // `StreamAccumulator::assemble_blocks`. Rebuilt on every call — this is
  // called at most once per incoming event, so the copy cost is negligible
  // compared to what a DOM mutation would run.
  assembleBlocks(): ContentBlock[] {
    const blocks: ContentBlock[] = [];

    // Reasoning block: readable wins over encrypted when both arrived
    // (shouldn't happen in practice; the preference is documented in the
    // Rust side).
    if (this.reasoning.length > 0) {
      blocks.push({
        type: "reasoning",
        content: this.reasoning,
        ...(this.signature !== undefined ? { signature: this.signature } : {}),
      });
    } else if (this.encrypted) {
      blocks.push({
        type: "reasoning",
        content: "",
        ...(this.signature !== undefined ? { signature: this.signature } : {}),
        encrypted: this.encrypted.data,
        format: this.encrypted.format,
      });
    }

    if (this.text.length > 0) {
      blocks.push({ type: "text", text: this.text });
    }

    // BTreeMap ordering on the Rust side; here we sort keys explicitly
    // since `Map` preserves insertion order, not numeric order, and tool
    // calls can start out of order (rare, but the protocol allows it).
    const indices = Array.from(this.toolCalls.keys()).sort((a, b) => a - b);
    for (const idx of indices) {
      const tc = this.toolCalls.get(idx)!;
      blocks.push({
        type: "tool_call",
        id: tc.id,
        name: tc.name,
        arguments: tc.arguments,
      });
    }

    return blocks;
  }

  // Convert the accumulator into a committed assistant message. Mirrors
  // `StreamAccumulator::into_message`. Used by `TurnCancelled` to preserve
  // partial content when no `MessageAppended` arrived first.
  intoMessage(): Message {
    return {
      role: "assistant",
      content: this.assembleBlocks(),
      token_count: 0,
    };
  }
}

// ---- Split state ---------------------------------------------------------

export interface SplitState {
  id: SplitId;
  sessionId: SessionId | null;
  messages: Message[];
  // Holds either the assembled blocks from an initial snapshot (set by
  // `rebuildFromSnapshot`) or `null` when a streaming accumulator is in
  // flight. The accumulator itself is tracked separately so we only
  // re-assemble blocks when the accumulator has actually advanced.
  accumulator: StreamAccumulator | null;
  // Cache of assembled blocks. Populated lazily: either from a snapshot's
  // `streaming` field, or from the accumulator after each push. Null when
  // no turn is in progress.
  streamingBlocks: ContentBlock[] | null;
  waiting: boolean;
  error: string | null;
  cancelled: boolean;
}

function emptySplitState(id: SplitId): SplitState {
  return {
    id,
    sessionId: null,
    messages: [],
    accumulator: null,
    streamingBlocks: null,
    waiting: false,
    error: null,
    cancelled: false,
  };
}

function splitFromSnapshot(snap: SplitSnapshot): SplitState {
  return {
    id: snap.id,
    sessionId: snap.session_id,
    messages: snap.messages,
    // The snapshot flattens the accumulator into assembled blocks for us.
    // We *don't* reconstruct a `StreamAccumulator` from those blocks —
    // there's no way to split a flattened block list back into deltas —
    // so the next `StreamDelta` will start a fresh accumulator at our end
    // while the backend continues from where it was. In practice a
    // snapshot is only taken between turns or on first load, so this
    // mismatch doesn't surface. If it ever does, `get_snapshot()` again.
    accumulator: null,
    streamingBlocks: snap.streaming,
    waiting: snap.waiting,
    error: snap.error,
    cancelled: snap.cancelled,
  };
}

// ---- Workspace store -----------------------------------------------------

// Subscriber callbacks are keyed by split id so the transcript renderer
// can subscribe to one split at a time and skip work for the others.
// Separator drag and layout changes fire a separate `onLayoutChanged`
// callback so the grid template only rebuilds when it actually needs to.
type SplitListener = (split: SplitState) => void;
type LayoutListener = (ws: WorkspaceState) => void;

export interface WorkspaceState {
  splits: Map<SplitId, SplitState>;
  order: SplitId[];
  focused: SplitId | null;
  splitFracs: number[];
  workspaceRoot: string;
}

export class WorkspaceStore {
  private state: WorkspaceState;
  private splitListeners = new Set<SplitListener>();
  private layoutListeners = new Set<LayoutListener>();

  constructor(initial: Snapshot) {
    this.state = fromSnapshot(initial);
  }

  get(): WorkspaceState {
    return this.state;
  }

  getSplit(id: SplitId): SplitState | undefined {
    return this.state.splits.get(id);
  }

  // Replace the whole store from a fresh authoritative snapshot. Used on
  // initial load, after `workspace_replaced`, and as a recovery path.
  rebuildFromSnapshot(snapshot: Snapshot): void {
    this.state = fromSnapshot(snapshot);
    this.notifyLayout();
    for (const split of this.state.splits.values()) this.notifySplit(split);
  }

  // Incrementally add/remove/update on `split_added` / `split_closed`.
  // Both events carry a fresh snapshot so we can just rebuild — keeps the
  // focused/fractions/order fields in lockstep with backend state without
  // manually patching each field.
  applySnapshot(snapshot: Snapshot): void {
    this.rebuildFromSnapshot(snapshot);
  }

  // The transcript renderer calls this per `agent_event` payload. Logic
  // mirrors `AgentSplit::handle_event`:
  //   - Ready:            record session_id, leave everything else alone
  //   - StreamDelta:      seed/advance the accumulator
  //   - MessageAppended:  append and drop the accumulator
  //   - TurnComplete:     clear waiting, streaming, error
  //   - TurnCancelled:    commit partial to messages if non-empty, flip
  //                       cancelled, clear waiting/streaming
  //   - Error:            clear streaming, set error, clear waiting
  applyAgentEvent(splitId: SplitId, event: AgentEvent): void {
    const split = this.state.splits.get(splitId);
    if (!split) return; // racing with a close; ignore
    switch (event.type) {
      case "ready":
        split.sessionId = event.session_id;
        break;
      case "stream_delta": {
        if (!split.accumulator) {
          split.accumulator = new StreamAccumulator();
        }
        split.accumulator.push(event.event);
        split.streamingBlocks = split.accumulator.assembleBlocks();
        break;
      }
      case "message_appended":
        split.messages = [...split.messages, event.message];
        split.accumulator = null;
        split.streamingBlocks = null;
        split.error = null;
        break;
      case "turn_complete":
        split.waiting = false;
        split.accumulator = null;
        split.streamingBlocks = null;
        split.error = null;
        break;
      case "turn_cancelled": {
        split.waiting = false;
        if (split.accumulator) {
          const msg = split.accumulator.intoMessage();
          if (msg.content.length > 0) {
            split.messages = [...split.messages, msg];
          }
          split.accumulator = null;
          split.streamingBlocks = null;
        }
        split.cancelled = true;
        split.error = null;
        break;
      }
      case "error":
        split.accumulator = null;
        split.streamingBlocks = null;
        split.error = event.message;
        split.waiting = false;
        break;
      default:
        // The Rust side is `#[non_exhaustive]`; future variants flow
        // through as a compile error here until the TS mirror catches up.
        assertNever(event);
    }
    this.notifySplit(split);
  }

  // Called synchronously after a successful `submit` whose classification
  // was `Send` — the backend flips `waiting = true` but doesn't emit an
  // event for it, so we mirror that locally. Mirrors `WorkspaceState::send`.
  markSent(splitId: SplitId): void {
    const split = this.state.splits.get(splitId);
    if (!split || split.waiting) return;
    split.waiting = true;
    split.error = null;
    split.cancelled = false;
    this.notifySplit(split);
  }

  setFocused(id: SplitId): void {
    if (this.state.focused === id) return;
    this.state.focused = id;
    this.notifyLayout();
  }

  setFractions(fracs: number[]): void {
    // Normalize length mismatches defensively; the backend's response to
    // `setSplitFractions` is the source of truth, but we may update from
    // either side (live drag paints before the command completes).
    if (fracs.length !== this.state.order.length) return;
    this.state.splitFracs = fracs.slice();
    this.notifyLayout();
  }

  onSplitChanged(cb: SplitListener): () => void {
    this.splitListeners.add(cb);
    return () => this.splitListeners.delete(cb);
  }

  onLayoutChanged(cb: LayoutListener): () => void {
    this.layoutListeners.add(cb);
    return () => this.layoutListeners.delete(cb);
  }

  private notifySplit(split: SplitState): void {
    for (const cb of this.splitListeners) cb(split);
  }

  private notifyLayout(): void {
    for (const cb of this.layoutListeners) cb(this.state);
  }
}

function fromSnapshot(snap: Snapshot): WorkspaceState {
  const splits = new Map<SplitId, SplitState>();
  const order: SplitId[] = [];
  for (const s of snap.splits) {
    splits.set(s.id, splitFromSnapshot(s));
    order.push(s.id);
  }
  return {
    splits,
    order,
    focused: snap.focused,
    splitFracs: snap.split_fracs,
    workspaceRoot: snap.workspace_root,
  };
}

// Re-export for callers that want to build a fresh empty split state in
// tests or optimistic UI paths.
export { emptySplitState };
