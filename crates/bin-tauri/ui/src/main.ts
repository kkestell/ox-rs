// Frontend entry point.
//
// Wire the store, DOM, event subscriptions, and dialog modals together.
// Everything that runs before `DOMContentLoaded` goes here; per-split
// mounts happen inside `splits.ts`.

import "./styles/main.css";

import {
  getSnapshot,
  onAgentEvent,
  onConfirmQuitRequested,
  onShowAbout,
  onSplitAdded,
  onSplitClosed,
  onWorkspaceReplaced,
} from "./bridge";
import { showAboutDialog, showQuitConfirmDialog } from "./dialogs";
import { mountSplits } from "./splits";
import { WorkspaceStore } from "./state";

async function main(): Promise<void> {
  const initial = await getSnapshot();
  const store = new WorkspaceStore(initial);

  const appEl = document.getElementById("app");
  if (!appEl) throw new Error("missing #app in index.html");
  mountSplits(appEl, store);

  // Per-event wiring: streaming deltas feed the reducer; workspace-level
  // events replace the whole snapshot so fractions/order/focus stay in
  // lockstep with Rust state.
  await onAgentEvent(({ split_id, event }) => {
    store.applyAgentEvent(split_id, event);
  });
  await onSplitAdded(({ snapshot }) => {
    store.applySnapshot(snapshot);
  });
  await onSplitClosed(({ snapshot }) => {
    store.applySnapshot(snapshot);
  });
  await onWorkspaceReplaced(({ snapshot }) => {
    store.applySnapshot(snapshot);
  });

  // Menu-triggered flows. The Rust side emits these after a native menu
  // item click or a close-window attempt while a turn is in flight.
  await onConfirmQuitRequested(() => showQuitConfirmDialog());
  await onShowAbout(() => void showAboutDialog());
}

// The Tauri webview loads `main.ts` as a module, so the script runs
// after parse. `DOMContentLoaded` may have already fired by then, so
// check `readyState` and either run immediately or wait.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void main());
} else {
  void main();
}
