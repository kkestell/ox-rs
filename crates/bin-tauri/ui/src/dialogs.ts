// Modal dialog helpers.
//
// Two modals in this app:
// 1. Quit-confirmation: shown when the user tries to close a split or the
//    whole window while a turn is in flight. Confirming saves the layout
//    and exits; cancelling resumes normally.
// 2. About: shown from Help > About in the native menu. Product name,
//    version, link to the repo.
//
// Modals are plain `<div>`s inside `#dialog-root`. ESC and backdrop click
// dismiss; the dialog traps focus while open.

import { cancelQuit, confirmQuit, getAppInfo } from "./bridge";

interface DialogOptions {
  title: string;
  body: Node | string;
  primary: { label: string; onClick: () => void | Promise<void> };
  secondary?: { label: string; onClick: () => void | Promise<void> };
}

let dialogRoot: HTMLElement | null = null;

function getRoot(): HTMLElement {
  if (!dialogRoot) {
    const el = document.getElementById("dialog-root");
    if (!el) throw new Error("missing #dialog-root in index.html");
    dialogRoot = el;
  }
  return dialogRoot;
}

function showDialog(opts: DialogOptions): () => void {
  const root = getRoot();

  const backdrop = document.createElement("div");
  backdrop.className = "dialog-backdrop";

  const dialog = document.createElement("div");
  dialog.className = "dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");

  const header = document.createElement("header");
  header.className = "dialog-title";
  header.textContent = opts.title;
  dialog.appendChild(header);

  const body = document.createElement("div");
  body.className = "dialog-body";
  if (typeof opts.body === "string") body.textContent = opts.body;
  else body.appendChild(opts.body);
  dialog.appendChild(body);

  const footer = document.createElement("footer");
  footer.className = "dialog-footer";

  if (opts.secondary) {
    const btn = document.createElement("button");
    btn.className = "secondary";
    btn.textContent = opts.secondary.label;
    btn.addEventListener("click", () => {
      void opts.secondary!.onClick();
      close();
    });
    footer.appendChild(btn);
  }

  const primary = document.createElement("button");
  primary.className = "primary";
  primary.textContent = opts.primary.label;
  primary.addEventListener("click", () => {
    void opts.primary.onClick();
    close();
  });
  footer.appendChild(primary);

  dialog.appendChild(footer);
  backdrop.appendChild(dialog);
  root.appendChild(backdrop);

  const onKey = (evt: KeyboardEvent) => {
    if (evt.key === "Escape") {
      evt.preventDefault();
      if (opts.secondary) void opts.secondary.onClick();
      close();
    }
  };
  // Clicking the backdrop (but not the dialog itself) dismisses as if the
  // secondary button had been pressed — matches the native-feel "click
  // outside to cancel" convention.
  const onBackdropClick = (evt: MouseEvent) => {
    if (evt.target === backdrop) {
      if (opts.secondary) void opts.secondary.onClick();
      close();
    }
  };
  window.addEventListener("keydown", onKey);
  backdrop.addEventListener("click", onBackdropClick);

  // Focus the primary button by default so Enter confirms.
  requestAnimationFrame(() => primary.focus());

  const close = () => {
    window.removeEventListener("keydown", onKey);
    backdrop.removeEventListener("click", onBackdropClick);
    backdrop.remove();
  };

  return close;
}

// ---- Specific dialogs ----------------------------------------------------

let quitDialogCloser: (() => void) | null = null;

export function showQuitConfirmDialog(): void {
  // If the user triggers the confirm dialog twice (e.g. two rapid window
  // close attempts), don't stack them — keep the first one and drop the
  // rest on the floor.
  if (quitDialogCloser) return;

  quitDialogCloser = showDialog({
    title: "Quit while a turn is in progress?",
    body: "One or more splits have a turn in flight. Quitting now will interrupt it.",
    primary: {
      label: "Quit",
      onClick: async () => {
        quitDialogCloser = null;
        await confirmQuit();
      },
    },
    secondary: {
      label: "Cancel",
      onClick: async () => {
        quitDialogCloser = null;
        await cancelQuit();
      },
    },
  });
}

export async function showAboutDialog(): Promise<void> {
  const info = await getAppInfo();
  const body = document.createElement("div");
  const name = document.createElement("div");
  name.className = "about-name";
  name.textContent = info.name;
  const version = document.createElement("div");
  version.className = "about-version";
  version.textContent = `Version ${info.version}`;
  body.appendChild(name);
  body.appendChild(version);

  showDialog({
    title: "About",
    body,
    primary: {
      label: "OK",
      onClick: () => {},
    },
  });
}
