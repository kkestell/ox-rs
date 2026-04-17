// Per-split `<textarea>` input. Enter submits, Shift+Enter inserts a
// newline, Escape cancels an in-flight turn. The textarea auto-grows as
// the user types up to a soft cap so a long draft doesn't push the
// transcript off-screen.

import { cancelSplit, submit } from "./bridge";
import type { SplitId, SubmitOutcome } from "./types";
import { assertNever } from "./types";
import type { WorkspaceStore } from "./state";

const MAX_ROWS = 8;

// Mounts a textarea into `container` and wires it to the store for the
// given split. Returns an unmount function.
export function mountInput(
  container: HTMLElement,
  splitId: SplitId,
  store: WorkspaceStore,
): () => void {
  const ta = document.createElement("textarea");
  ta.className = "input";
  ta.rows = 1;
  ta.spellcheck = false;
  ta.placeholder = "Send a message…";
  container.appendChild(ta);

  const autogrow = () => {
    ta.style.height = "auto";
    // Line height comes from CSS; cap at MAX_ROWS × line-height to avoid
    // the textarea eating the transcript on a multi-paragraph draft.
    const style = window.getComputedStyle(ta);
    const line = parseFloat(style.lineHeight) || 18;
    const padding =
      parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const cap = line * MAX_ROWS + padding;
    ta.style.height = `${Math.min(ta.scrollHeight, cap)}px`;
    ta.style.overflowY = ta.scrollHeight > cap ? "auto" : "hidden";
  };

  ta.addEventListener("input", autogrow);

  const onKey = (evt: KeyboardEvent) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      const text = ta.value;
      // Empty or whitespace-only input is a no-op. Matches the egui
      // behavior — `classify_input` in agent-host also skips it.
      if (text.trim().length === 0) return;
      ta.value = "";
      autogrow();
      void submit(splitId, text).then((outcome) =>
        handleSubmitOutcome(outcome, splitId, store),
      );
      return;
    }
    if (evt.key === "Escape") {
      evt.preventDefault();
      void cancelSplit(splitId);
    }
  };
  ta.addEventListener("keydown", onKey);

  // Focus on mount so the user can start typing immediately. The first
  // paint calls this while the DOM is settling, so defer one tick.
  requestAnimationFrame(() => {
    if (store.get().focused === splitId) ta.focus();
  });

  // When the store's focused split changes to this one, grab focus. The
  // transcript click handler updates focus too, so clicking into a
  // transcript and pressing a key goes to the right input.
  const unsubFocus = store.onLayoutChanged((ws) => {
    if (ws.focused === splitId && document.activeElement !== ta) {
      ta.focus();
    }
  });

  autogrow();

  return () => {
    ta.removeEventListener("input", autogrow);
    ta.removeEventListener("keydown", onKey);
    unsubFocus();
    ta.remove();
  };
}

// Map the classification result into frontend-visible side effects. The
// backend already performed the state mutation — our job is to mirror the
// waiting flag (for `sent_message`) and update focus (for `new_split`).
function handleSubmitOutcome(
  outcome: SubmitOutcome,
  splitId: SplitId,
  store: WorkspaceStore,
): void {
  switch (outcome.kind) {
    case "sent_message":
      store.markSent(splitId);
      return;
    case "new_split":
      // `split_added` event will rebuild layout; just shift focus.
      store.setFocused(outcome.split_id);
      return;
    case "closed_split":
      // `split_closed` event will rebuild layout.
      return;
    case "cancel_requested":
      // Backend will emit `TurnCancelled` when the agent acks.
      return;
    case "quit_requested":
      // Backend either closes the window directly or emits
      // `confirm_quit_requested` for the modal.
      return;
    default:
      assertNever(outcome);
  }
}
