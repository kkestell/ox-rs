// Per-split transcript renderer.
//
// The transcript is a scrolling `<section>` that mounts once per split and
// updates in place when the store announces a change. Messages are DOM
// nodes appended as they arrive; streaming content lives in a single
// `.streaming` container that gets rebuilt on each delta. Tool results
// merge into their matching `<details>` tool call so the user doesn't see
// them as a separate message.
//
// Text selection: `user-select: text` on `.transcript` and its
// descendants (via `main.css`) gives the webview native click-drag
// selection. `selection.ts` adds `Ctrl+A` scoping on top.

import type { ContentBlock, Message, SplitId } from "./types";
import { attachTranscriptSelection } from "./selection";
import type { SplitState, WorkspaceStore } from "./state";

export function mountTranscript(
  container: HTMLElement,
  splitId: SplitId,
  store: WorkspaceStore,
): () => void {
  container.classList.add("transcript");
  // Tabindex so the element can become focus target for the `Ctrl+A`
  // scoping handler; also lets the user click into the transcript to
  // select it as the keyboard selection anchor.
  container.tabIndex = 0;

  const detachSelection = attachTranscriptSelection(container);

  const render = () => {
    const split = store.getSplit(splitId);
    if (!split) return;
    paint(container, split);
  };

  const unsub = store.onSplitChanged((split) => {
    if (split.id === splitId) render();
  });

  // Initial paint.
  render();

  return () => {
    unsub();
    detachSelection();
  };
}

// Full repaint. The transcript is cheap to rebuild — a few hundred DOM
// nodes for a long session — and this sidesteps every reconciliation bug
// a manual diff could introduce. If profiling ever shows this as the
// bottleneck, the escape hatch is to append only the tail and keep a
// cursor into the store's message list.
function paint(container: HTMLElement, split: SplitState): void {
  const atBottom = isScrolledToBottom(container);
  container.replaceChildren();

  // Index tool results by their tool_call_id so we can fold them into
  // the matching `<details>` element when we encounter it. Tool-result
  // messages themselves don't get their own `.message` node; they're
  // inlined under their call. The Rust side guarantees one result per
  // call, but we use a `Map<string, ContentBlock[]>` defensively so
  // multiple results (should they ever arrive) all show up.
  const toolResults = collectToolResults(split.messages);

  for (const msg of split.messages) {
    // Tool-result-only messages are rendered under their matching call,
    // not as a standalone message entry.
    if (isToolResultOnlyMessage(msg)) continue;
    container.appendChild(renderMessage(msg, toolResults));
  }

  // Streaming content goes at the bottom in its own frame so we can keep
  // a stable reference and update it in place. At `TurnComplete` /
  // `TurnCancelled` the backend clears `streaming`, which makes the block
  // drop out of the DOM on the next paint.
  if (split.streamingBlocks && split.streamingBlocks.length > 0) {
    const frame = document.createElement("div");
    frame.className = "message role-assistant streaming";
    appendBlocks(frame, split.streamingBlocks, toolResults);
    container.appendChild(frame);
  }

  if (split.waiting && !split.streamingBlocks) {
    container.appendChild(statusLine("…", "waiting"));
  }
  if (split.cancelled) {
    container.appendChild(statusLine("cancelled", "cancelled"));
  }
  if (split.error) {
    container.appendChild(statusLine(split.error, "error"));
  }

  if (atBottom) scrollToBottom(container);
}

function renderMessage(
  msg: Message,
  toolResults: Map<string, ContentBlock[]>,
): HTMLElement {
  const el = document.createElement("div");
  el.className = `message role-${msg.role}`;
  appendBlocks(el, msg.content, toolResults);
  return el;
}

function appendBlocks(
  parent: HTMLElement,
  blocks: ContentBlock[],
  toolResults: Map<string, ContentBlock[]>,
): void {
  for (const block of blocks) {
    switch (block.type) {
      case "text": {
        const n = document.createElement("div");
        n.className = "block-text";
        n.textContent = block.text;
        parent.appendChild(n);
        break;
      }
      case "reasoning": {
        // Visible reasoning renders inline; an encrypted-only blob renders
        // as a short placeholder so the user knows the turn produced
        // private reasoning that can't be shown.
        if (block.content.length > 0) {
          const n = document.createElement("div");
          n.className = "block-reasoning";
          n.textContent = block.content;
          parent.appendChild(n);
        } else if (block.encrypted) {
          const n = document.createElement("div");
          n.className = "block-reasoning encrypted";
          n.textContent = "[encrypted reasoning]";
          parent.appendChild(n);
        }
        break;
      }
      case "tool_call": {
        const details = document.createElement("details");
        details.className = "block-tool-call";

        const summary = document.createElement("summary");
        summary.textContent = `${block.name}(${block.arguments})`;
        details.appendChild(summary);

        // Inline the paired tool result(s). If the result hasn't arrived
        // yet (mid-turn), the body is empty and fills in on the next
        // paint when the `MessageAppended` for the tool role lands.
        const results = toolResults.get(block.id) ?? [];
        for (const result of results) {
          if (result.type !== "tool_result") continue;
          const body = document.createElement("pre");
          body.className = result.is_error
            ? "tool-result error"
            : "tool-result";
          body.textContent = result.content;
          details.appendChild(body);
        }

        parent.appendChild(details);
        break;
      }
      case "tool_result":
        // Tool results are inlined under their call and not rendered
        // standalone. This branch is only taken when a tool-result block
        // appears under a non-tool-role message, which shouldn't happen
        // in practice but would surface here if it did.
        break;
      default: {
        const exhaustive: never = block;
        throw new Error(
          `unexpected content block: ${JSON.stringify(exhaustive)}`,
        );
      }
    }
  }
}

function isToolResultOnlyMessage(msg: Message): boolean {
  return (
    msg.role === "tool" &&
    msg.content.length > 0 &&
    msg.content.every((b) => b.type === "tool_result")
  );
}

function collectToolResults(
  messages: Message[],
): Map<string, ContentBlock[]> {
  const map = new Map<string, ContentBlock[]>();
  for (const msg of messages) {
    if (msg.role !== "tool") continue;
    for (const block of msg.content) {
      if (block.type !== "tool_result") continue;
      const bucket = map.get(block.tool_call_id);
      if (bucket) bucket.push(block);
      else map.set(block.tool_call_id, [block]);
    }
  }
  return map;
}

function statusLine(text: string, kind: string): HTMLElement {
  const n = document.createElement("div");
  n.className = `status ${kind}`;
  n.textContent = text;
  return n;
}

// Autoscroll policy: if the user was at the bottom before the paint,
// stick to the bottom after. If they had scrolled up to read, don't
// yank them back. A 4px slop accounts for fractional scroll positions.
function isScrolledToBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 4;
}

function scrollToBottom(el: HTMLElement): void {
  el.scrollTop = el.scrollHeight;
}
