// Typed wrappers around `invoke` (command) and `listen` (event) from
// `@tauri-apps/api`. Everything the frontend calls on the backend flows
// through here, so type drift between the Rust commands and the JS side
// surfaces in this one file instead of being scattered across call sites.
//
// Command arguments use camelCase on the JS side; Tauri auto-converts them
// to the snake_case parameters in Rust. Event *payload* fields stay
// snake_case because they're plain serde-derived structs crossing the
// boundary, not invoke arguments.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type {
  AgentEventPayload,
  AppInfo,
  Snapshot,
  SplitAddedPayload,
  SplitClosedPayload,
  SplitId,
  SubmitOutcome,
  WorkspaceReplacedPayload,
} from "./types";

// ---- Commands ------------------------------------------------------------

export function getSnapshot(): Promise<Snapshot> {
  return invoke<Snapshot>("get_snapshot");
}

export function submit(splitId: SplitId, input: string): Promise<SubmitOutcome> {
  return invoke<SubmitOutcome>("submit", { splitId, input });
}

export function cancelSplit(splitId: SplitId): Promise<void> {
  return invoke<void>("cancel_split", { splitId });
}

export function closeSplit(splitId: SplitId): Promise<void> {
  return invoke<void>("close_split", { splitId });
}

export function newSplit(): Promise<SplitId> {
  return invoke<SplitId>("new_split");
}

export function openWorkspace(path: string): Promise<void> {
  return invoke<void>("open_workspace", { path });
}

// Returns the *normalized* fractions the backend actually stored, which
// may differ from the input if it was out-of-shape or non-finite. Callers
// should snap their grid to the returned values rather than trusting their
// own local copy.
export function setSplitFractions(fractions: number[]): Promise<number[]> {
  return invoke<number[]>("set_split_fractions", { fractions });
}

export function requestQuit(): Promise<void> {
  return invoke<void>("request_quit");
}

export function confirmQuit(): Promise<void> {
  return invoke<void>("confirm_quit");
}

export function cancelQuit(): Promise<void> {
  return invoke<void>("cancel_quit");
}

export function getAppInfo(): Promise<AppInfo> {
  return invoke<AppInfo>("get_app_info");
}

// ---- Events --------------------------------------------------------------

// Event name constants mirror the ones in `bin-tauri/src/events.rs`. Using
// named exports here catches typos in the subscription-site code instead of
// letting a silent no-op slip through.

export const EVT_AGENT_EVENT = "agent_event";
export const EVT_SPLIT_ADDED = "split_added";
export const EVT_SPLIT_CLOSED = "split_closed";
export const EVT_WORKSPACE_REPLACED = "workspace_replaced";
export const EVT_CONFIRM_QUIT_REQUESTED = "confirm_quit_requested";
export const EVT_SHOW_ABOUT = "show_about";

export function onAgentEvent(
  cb: (payload: AgentEventPayload) => void,
): Promise<UnlistenFn> {
  return listen<AgentEventPayload>(EVT_AGENT_EVENT, (e) => cb(e.payload));
}

export function onSplitAdded(
  cb: (payload: SplitAddedPayload) => void,
): Promise<UnlistenFn> {
  return listen<SplitAddedPayload>(EVT_SPLIT_ADDED, (e) => cb(e.payload));
}

export function onSplitClosed(
  cb: (payload: SplitClosedPayload) => void,
): Promise<UnlistenFn> {
  return listen<SplitClosedPayload>(EVT_SPLIT_CLOSED, (e) => cb(e.payload));
}

export function onWorkspaceReplaced(
  cb: (payload: WorkspaceReplacedPayload) => void,
): Promise<UnlistenFn> {
  return listen<WorkspaceReplacedPayload>(EVT_WORKSPACE_REPLACED, (e) =>
    cb(e.payload),
  );
}

export function onConfirmQuitRequested(cb: () => void): Promise<UnlistenFn> {
  return listen<unknown>(EVT_CONFIRM_QUIT_REQUESTED, () => cb());
}

export function onShowAbout(cb: () => void): Promise<UnlistenFn> {
  return listen<unknown>(EVT_SHOW_ABOUT, () => cb());
}
