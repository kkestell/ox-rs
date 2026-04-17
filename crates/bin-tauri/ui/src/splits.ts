// Renders the splits workspace as a CSS grid and handles separator drag.
//
// Layout: `grid-template-columns: fr₁ 6px fr₂ 6px fr₃ …` — the fractions
// are kept on the `--split-fracs` side-channel and used to set `fr` values.
// A single `SplitContainer` mounts once and updates in place when splits
// are added, removed, or resized.

import { setSplitFractions } from "./bridge";
import { mountInput } from "./input";
import { mountTranscript } from "./transcript";
import type { SplitId } from "./types";
import type { WorkspaceStore, SplitState } from "./state";

// Per-split DOM handle. The renderer keeps these alive across re-renders
// so we can update in place rather than rebuilding each split's transcript
// on every event.
interface SplitView {
  root: HTMLElement;
  unmountTranscript: () => void;
  unmountInput: () => void;
}

// The separator is 6px wide. Agree with CSS in `main.css`.
const SEPARATOR_PX = 6;

export function mountSplits(
  container: HTMLElement,
  store: WorkspaceStore,
): void {
  container.classList.add("splits");

  const views = new Map<SplitId, SplitView>();

  function render(): void {
    const ws = store.get();
    const order = ws.order;

    // Tear down views for splits that went away.
    for (const [id, view] of views) {
      if (!ws.splits.has(id)) {
        view.unmountTranscript();
        view.unmountInput();
        view.root.remove();
        views.delete(id);
      }
    }

    // Build new views for splits we haven't seen yet.
    for (const id of order) {
      if (views.has(id)) continue;
      const split = ws.splits.get(id)!;
      const view = buildSplitView(split, store);
      views.set(id, view);
    }

    // Re-assemble the grid with splits in backend order, inserting
    // separators between every pair of consecutive splits. Re-insertion
    // is cheap and guarantees DOM order matches `order` even after
    // add/close events that don't touch existing views.
    container.replaceChildren();
    for (let i = 0; i < order.length; i++) {
      const id = order[i];
      if (id === undefined) continue;
      const view = views.get(id);
      if (!view) continue;
      if (i > 0) container.appendChild(buildSeparator(i - 1, store));
      container.appendChild(view.root);
      view.root.classList.toggle("focused", ws.focused === id);
    }

    // Grid template: fr₁ 6px fr₂ 6px fr₃ … The fractions come from the
    // backend; invalid input was normalized before it reached us.
    const parts: string[] = [];
    for (let i = 0; i < ws.splitFracs.length; i++) {
      if (i > 0) parts.push(`${SEPARATOR_PX}px`);
      parts.push(`${ws.splitFracs[i]}fr`);
    }
    container.style.gridTemplateColumns = parts.join(" ");
  }

  render();

  // Full re-render on layout changes (add/close/replace_workspace/focus).
  // These are rare enough that a full pass beats the complexity of
  // diffing. Per-split updates for streaming/messages are handled by the
  // transcript module, which subscribes per-split.
  store.onLayoutChanged(() => render());
}

function buildSplitView(split: SplitState, store: WorkspaceStore): SplitView {
  const root = document.createElement("section");
  root.className = "split";
  root.dataset["splitId"] = split.id;

  // Click anywhere inside the split updates focus. We stash the id on the
  // root so the click handler doesn't need a closure capture per view.
  root.addEventListener("mousedown", () => store.setFocused(split.id));

  const transcriptEl = document.createElement("div");
  transcriptEl.className = "split-transcript";
  const inputEl = document.createElement("div");
  inputEl.className = "split-input";

  root.appendChild(transcriptEl);
  root.appendChild(inputEl);

  const unmountTranscript = mountTranscript(transcriptEl, split.id, store);
  const unmountInput = mountInput(inputEl, split.id, store);

  return { root, unmountTranscript, unmountInput };
}

// Build one separator between the splits at `leftIdx` and `leftIdx + 1`.
// The separator captures drag events and adjusts the two neighboring
// fractions; nothing else in the grid moves.
function buildSeparator(leftIdx: number, store: WorkspaceStore): HTMLElement {
  const sep = document.createElement("div");
  sep.className = "separator";
  sep.setAttribute("role", "separator");
  sep.setAttribute("aria-orientation", "vertical");

  sep.addEventListener("mousedown", (evt) => startDrag(evt, leftIdx, store));

  return sep;
}

// Drag state is tracked on the window via capturing listeners. Using the
// window ensures we keep receiving move events even if the cursor leaves
// the separator (which is only 6px wide and easy to fly past).
function startDrag(
  evt: MouseEvent,
  leftIdx: number,
  store: WorkspaceStore,
): void {
  evt.preventDefault();
  const ws = store.get();
  const container = (evt.currentTarget as HTMLElement).parentElement;
  if (!container) return;

  const totalPx = container.clientWidth - SEPARATOR_PX * (ws.order.length - 1);
  // Convert the current fractions to pixel widths so the drag math is in
  // pixel space; we re-normalize to fractions at the end.
  const fracSum = ws.splitFracs.reduce((a, b) => a + b, 0) || 1;
  const startWidths = ws.splitFracs.map((f) => (f / fracSum) * totalPx);
  const startX = evt.clientX;
  const leftStart = startWidths[leftIdx] ?? 0;
  const rightStart = startWidths[leftIdx + 1] ?? 0;
  const pairTotal = leftStart + rightStart;

  // Arbitrary minimum width — stop the user from collapsing a split
  // completely. 40px gives enough room for the scrollbar to stay visible.
  const MIN = 40;

  const onMove = (e: MouseEvent) => {
    const delta = e.clientX - startX;
    let newLeft = leftStart + delta;
    newLeft = Math.max(MIN, Math.min(pairTotal - MIN, newLeft));
    const newRight = pairTotal - newLeft;

    const widths = startWidths.slice();
    widths[leftIdx] = newLeft;
    widths[leftIdx + 1] = newRight;

    const sum = widths.reduce((a, b) => a + b, 0) || 1;
    const fracs = widths.map((w) => w / sum);
    store.setFractions(fracs);
  };

  const onUp = () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    document.body.style.removeProperty("cursor");
    document.body.style.removeProperty("user-select");
    // Persist the final fractions to the backend. The returned normalized
    // vector is authoritative; snap to it in case the backend clamped.
    setSplitFractions(store.get().splitFracs).then((normalized) =>
      store.setFractions(normalized),
    );
  };

  // While dragging, disable selection globally and force the resize cursor
  // so the UX doesn't change mid-drag when the cursor leaves the 6px strip.
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}
