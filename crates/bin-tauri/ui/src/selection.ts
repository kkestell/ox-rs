// Transcript-scoped `Ctrl+A`.
//
// The browser's default `Ctrl+A` selects everything inside the focused
// editable (or the document root if nothing is focused). That's not what
// we want inside a transcript: we want Select-All to mean "the content
// of *this* split's transcript", not the entire window including inputs
// and the other splits. When `Ctrl+A` fires inside an `<input>` or
// `<textarea>`, we let the browser handle it — that's the normal input
// select-all.

export function attachTranscriptSelection(container: HTMLElement): () => void {
  const onKey = (evt: KeyboardEvent) => {
    const isSelectAll =
      (evt.ctrlKey || evt.metaKey) &&
      !evt.shiftKey &&
      !evt.altKey &&
      evt.key.toLowerCase() === "a";
    if (!isSelectAll) return;

    const target = evt.target as Element | null;
    // If the keystroke is happening inside a form control, leave it
    // alone — the browser's native select-all is correct there.
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement
    ) {
      return;
    }

    const selection = window.getSelection();
    if (!selection) return;
    evt.preventDefault();
    selection.selectAllChildren(container);
  };

  // Attach on the transcript container itself and bubble up — any
  // keyboard event originating inside the transcript gets the scoped
  // select-all. Events from outside never reach us.
  container.addEventListener("keydown", onKey);

  return () => container.removeEventListener("keydown", onKey);
}
