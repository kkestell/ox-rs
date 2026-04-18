// ox web client — single ES module, no build step.
//
// Architecture:
//
// - `state.sessions` maps a SessionId to a `SessionView`, which owns a pane's
//   DOM references, its `EventSource`, an optional streaming accumulator, and
//   flags (`turning`, `ended`).
// - Events flow one way: SSE frames → `applyEvent` → DOM mutations. User
//   actions flow the other way: click/keypress → `fetch` → server mutates
//   state → change propagates back over SSE.
// - The server is the source of truth. We never render a user's typed input
//   optimistically; we wait for the `MessageAppended(user)` frame so two
//   tabs watching the same session see the same transcript. That latency is
//   bounded by a single stdio write on the agent side.

const state = {
  workspaceRoot: "",
  sessions: new Map(),
};

// ---------------------------------------------------------------------------
// StreamAccumulator — JS port of `crates/app/src/stream.rs::StreamAccumulator`
//
// The Rust type decides block order when baking a `Message` from a stream of
// `StreamEvent`s. We mirror its rules here so that the live partial render
// matches the eventual committed `MessageAppended(assistant)` frame — any
// drift would cause the transcript to flicker when the accumulator is swapped
// out for the committed message.
//
// Order: reasoning (readable beats encrypted), then text, then tool calls
// sorted by `index`.
// ---------------------------------------------------------------------------

class StreamAccumulator {
  constructor() {
    this.text = "";
    this.reasoning = "";
    this.signature = null;
    this.encrypted = null; // { data, format } | null
    this.toolCalls = new Map(); // index → { id, name, arguments }
  }

  push(event) {
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
        break;
      }
      case "reasoning_signature":
        this.signature = event.signature;
        break;
      case "finished":
        // Usage is not rendered in V1; ignore.
        break;
    }
  }

  blocks() {
    const out = [];
    if (this.reasoning) {
      out.push({
        type: "reasoning",
        content: this.reasoning,
        signature: this.signature,
      });
    } else if (this.encrypted) {
      out.push({
        type: "reasoning",
        content: "",
        signature: this.signature,
        encrypted: this.encrypted.data,
        format: this.encrypted.format,
      });
    }
    if (this.text) {
      out.push({ type: "text", text: this.text });
    }
    const keys = [...this.toolCalls.keys()].sort((a, b) => a - b);
    for (const k of keys) {
      const tc = this.toolCalls.get(k);
      out.push({
        type: "tool_call",
        id: tc.id,
        name: tc.name,
        arguments: tc.arguments,
      });
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function main() {
  let snapshot;
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    snapshot = await res.json();
  } catch (err) {
    const bar = document.createElement("div");
    bar.className = "banner";
    bar.textContent = `Failed to load sessions: ${err}`;
    document.body.prepend(bar);
    return;
  }

  state.workspaceRoot = snapshot.workspace_root;
  document.getElementById("workspace-path").textContent = snapshot.workspace_root;

  // Prefer layout order for display, but tolerate a layout blob that lists
  // ids the server no longer knows about, and live sessions the layout
  // doesn't (e.g. a restore-fallback session created before any client
  // PUT landed).
  const byId = new Map(snapshot.sessions.map((s) => [s.session_id, s]));
  const order = [];
  for (const id of snapshot.layout.order) {
    if (byId.has(id)) order.push(id);
  }
  for (const s of snapshot.sessions) {
    if (!order.includes(s.session_id)) order.push(s.session_id);
  }

  for (let i = 0; i < order.length; i++) {
    const id = order[i];
    const summary = byId.get(id);
    const size = snapshot.layout.sizes[i];
    mountSession(id, summary ? summary.model : "", size);
  }
  renderGutters();

  document.getElementById("new-session").addEventListener("click", onNewSession);
}

// ---------------------------------------------------------------------------
// Per-session mount
// ---------------------------------------------------------------------------

function mountSession(id, model, size) {
  const tpl = document.getElementById("session-template");
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.dataset.sessionId = id;
  // Use flex-grow as the proportional weight. `flex: <g> 1 0` makes every
  // pane share the row by the ratio of its flex-grow, which matches the
  // normalized fraction the server persists.
  if (typeof size === "number" && isFinite(size) && size > 0) {
    node.style.flex = `${size} 1 0`;
  }

  const sess = {
    id,
    model,
    root: node,
    header: node.querySelector(".session-id"),
    transcript: node.querySelector(".transcript"),
    banner: node.querySelector(".banner"),
    composer: node.querySelector(".composer"),
    textarea: node.querySelector("textarea"),
    sendButton: node.querySelector(".composer button[type=submit]"),
    cancelButton: node.querySelector(".cancel"),
    closeButton: node.querySelector(".close"),
    eventSource: null,
    accumulator: null,
    streamingEl: null,
    turning: false,
    ended: false,
  };
  sess.header.textContent = id;

  sess.composer.addEventListener("submit", (ev) => {
    ev.preventDefault();
    sendMessage(sess);
  });
  sess.textarea.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      sendMessage(sess);
    }
  });
  sess.cancelButton.addEventListener("click", () => cancelTurn(sess));
  sess.closeButton.addEventListener("click", () => closeSession(sess));

  document.getElementById("sessions").appendChild(node);
  state.sessions.set(id, sess);

  openStream(sess);
  return sess;
}

function openStream(sess) {
  const es = new EventSource(`/api/sessions/${sess.id}/events`);
  sess.eventSource = es;
  es.onmessage = (ev) => {
    try {
      const evt = JSON.parse(ev.data);
      applyEvent(sess, evt);
    } catch (err) {
      console.error("ox: bad SSE frame:", err, ev.data);
    }
  };
  es.onerror = () => {
    // The EventSource auto-reconnects while readyState is CONNECTING. CLOSED
    // means the browser gave up (fatal HTTP error, 404 after DELETE, or the
    // stream cleanly ended after the session was removed server-side).
    if (es.readyState === EventSource.CLOSED) {
      markSessionEnded(sess);
    }
  };
}

// ---------------------------------------------------------------------------
// Event application
// ---------------------------------------------------------------------------

function applyEvent(sess, event) {
  switch (event.type) {
    case "ready":
      sess.header.textContent = event.session_id;
      break;
    case "message_appended":
      // A committed message supersedes the in-progress draft for this turn:
      // drop the draft, then render the authoritative content.
      finishStreaming(sess);
      appendMessage(sess, event.message);
      break;
    case "stream_delta":
      if (!sess.accumulator) beginStreaming(sess);
      sess.accumulator.push(event.event);
      renderStreaming(sess);
      break;
    case "turn_complete":
    case "turn_cancelled":
      finishStreaming(sess);
      clearBanner(sess);
      sess.turning = false;
      setComposerEnabled(sess, true);
      break;
    case "error":
      finishStreaming(sess);
      showBanner(sess, event.message);
      sess.turning = false;
      setComposerEnabled(sess, true);
      break;
  }
}

function beginStreaming(sess) {
  sess.accumulator = new StreamAccumulator();
  // The wrapper uses `display: contents` so its child blocks participate in
  // the transcript's flex gap as if the wrapper weren't there — we keep the
  // wrapper purely to own the draft for atomic replace/remove.
  sess.streamingEl = document.createElement("div");
  sess.streamingEl.className = "streaming";
  sess.transcript.appendChild(sess.streamingEl);
}

function renderStreaming(sess) {
  if (!sess.streamingEl || !sess.accumulator) return;
  sess.streamingEl.textContent = "";
  for (const block of sess.accumulator.blocks()) {
    sess.streamingEl.appendChild(renderBlock(block, "assistant"));
  }
  scrollToBottom(sess);
}

function finishStreaming(sess) {
  if (sess.streamingEl) {
    sess.streamingEl.remove();
    sess.streamingEl = null;
  }
  sess.accumulator = null;
}

function appendMessage(sess, message) {
  if (!message.content || message.content.length === 0) return;
  for (const block of message.content) {
    sess.transcript.appendChild(renderBlock(block, message.role));
  }
  scrollToBottom(sess);
}

function renderBlock(block, role) {
  switch (block.type) {
    case "text": {
      const div = document.createElement("div");
      div.className = `block ${role}`;
      div.textContent = block.text;
      return div;
    }
    case "reasoning": {
      const wrap = document.createElement("div");
      wrap.className = "block reasoning";
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "reasoning";
      details.appendChild(summary);
      const body = document.createElement("div");
      if (block.content) {
        body.textContent = block.content;
      } else if (block.encrypted) {
        body.textContent = `(encrypted, format=${block.format || "unknown"})`;
      }
      details.appendChild(body);
      wrap.appendChild(details);
      return wrap;
    }
    case "tool_call": {
      const wrap = document.createElement("div");
      wrap.className = "block tool";
      const label = document.createElement("span");
      label.className = "tool-label";
      label.textContent = `→ ${block.name}`;
      wrap.appendChild(label);
      const pre = document.createElement("pre");
      pre.textContent = prettyJson(block.arguments);
      wrap.appendChild(pre);
      return wrap;
    }
    case "tool_result": {
      const wrap = document.createElement("div");
      wrap.className = `block ${block.is_error ? "error" : "tool"}`;
      const label = document.createElement("span");
      label.className = "tool-label";
      label.textContent = block.is_error ? "← error" : "← result";
      wrap.appendChild(label);
      const pre = document.createElement("pre");
      pre.textContent = block.content;
      wrap.appendChild(pre);
      return wrap;
    }
    default: {
      const div = document.createElement("div");
      div.className = "block";
      div.textContent = JSON.stringify(block);
      return div;
    }
  }
}

// Best-effort JSON prettify for tool arguments. Arguments may be unparseable
// mid-stream, so we fall back to the raw string — the partial render must
// never throw.
function prettyJson(s) {
  if (!s) return "";
  try {
    return JSON.stringify(JSON.parse(s), null, 2);
  } catch {
    return s;
  }
}

function scrollToBottom(sess) {
  sess.transcript.scrollTop = sess.transcript.scrollHeight;
}

// ---------------------------------------------------------------------------
// Composer actions
// ---------------------------------------------------------------------------

async function sendMessage(sess) {
  if (sess.turning || sess.ended) return;
  const input = sess.textarea.value;
  if (!input.trim()) return;
  sess.turning = true;
  setComposerEnabled(sess, false);
  clearBanner(sess);
  let res;
  try {
    res = await fetch(`/api/sessions/${sess.id}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input }),
    });
  } catch (err) {
    sess.turning = false;
    setComposerEnabled(sess, true);
    showBanner(sess, `send failed: ${err}`);
    return;
  }
  if (res.status === 204) {
    sess.textarea.value = "";
    // Server confirmed receipt. The composer stays disabled until
    // turn_complete / turn_cancelled / error flips `turning` back.
    return;
  }
  if (res.status === 409) {
    // The server rejected the send because begin_send returned Skip — a
    // turn is already in flight. Rare race (user hammered Send during a
    // slow tool call); flash and let the real turn finish.
    flashComposer(sess);
    sess.turning = false;
    setComposerEnabled(sess, true);
    return;
  }
  if (res.status === 410) {
    markSessionEnded(sess);
    showBanner(sess, "agent exited");
    return;
  }
  if (res.status === 404) {
    markSessionEnded(sess);
    showBanner(sess, "session not found");
    return;
  }
  // 400 empty input (shouldn't happen — we check above) and any 5xx fall
  // through to a generic banner.
  sess.turning = false;
  setComposerEnabled(sess, true);
  showBanner(sess, `send failed: HTTP ${res.status}`);
}

async function cancelTurn(sess) {
  try {
    await fetch(`/api/sessions/${sess.id}/cancel`, { method: "POST" });
  } catch (err) {
    console.error("ox: cancel failed", err);
  }
}

async function closeSession(sess) {
  try {
    await fetch(`/api/sessions/${sess.id}`, { method: "DELETE" });
  } catch (err) {
    console.error("ox: close failed", err);
  }
  if (sess.eventSource) sess.eventSource.close();
  sess.root.remove();
  state.sessions.delete(sess.id);
  renderGutters();
  putLayout();
}

async function onNewSession() {
  let res;
  try {
    res = await fetch("/api/sessions", { method: "POST" });
  } catch (err) {
    console.error("ox: new session failed", err);
    return;
  }
  if (!res.ok) {
    console.error(`ox: new session failed: HTTP ${res.status}`);
    return;
  }
  const body = await res.json();
  // The new pane joins the row with an equal share so it's visible. The
  // user can drag gutters afterwards; drag-end will overwrite the layout.
  distributeEvenly();
  mountSession(body.session_id, "", 1);
  renderGutters();
  putLayout();
}

// ---------------------------------------------------------------------------
// Session-state helpers
// ---------------------------------------------------------------------------

function setComposerEnabled(sess, enabled) {
  if (sess.ended) return;
  sess.textarea.disabled = !enabled;
  sess.sendButton.disabled = !enabled;
  sess.cancelButton.hidden = enabled;
}

function flashComposer(sess) {
  sess.composer.classList.remove("flash");
  // Force a style recalculation so the animation re-runs on repeated 409s.
  void sess.composer.offsetWidth;
  sess.composer.classList.add("flash");
}

function showBanner(sess, message) {
  sess.banner.textContent = message;
  sess.banner.hidden = false;
}

function clearBanner(sess) {
  sess.banner.textContent = "";
  sess.banner.hidden = true;
}

function markSessionEnded(sess) {
  if (sess.ended) return;
  sess.ended = true;
  sess.turning = false;
  if (sess.eventSource) {
    sess.eventSource.close();
    sess.eventSource = null;
  }
  sess.root.classList.add("ended");
  sess.textarea.disabled = true;
  sess.sendButton.disabled = true;
  sess.cancelButton.hidden = true;
  // `closeButton` stays enabled — the user needs a way to dismiss the pane.
}

// ---------------------------------------------------------------------------
// Layout and gutters
// ---------------------------------------------------------------------------

function currentLayout() {
  const order = [];
  const sizes = [];
  for (const node of document.querySelectorAll(".session")) {
    const id = node.dataset.sessionId;
    if (!id) continue;
    order.push(id);
    const grow = parseFloat(node.style.flexGrow || "1");
    sizes.push(isFinite(grow) && grow > 0 ? grow : 1);
  }
  const sum = sizes.reduce((a, b) => a + b, 0) || 1;
  return { order, sizes: sizes.map((s) => s / sum) };
}

function putLayout() {
  const body = currentLayout();
  return fetch("/api/layout", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }).catch((err) => {
    console.error("ox: put layout failed", err);
  });
}

function distributeEvenly() {
  const panes = document.querySelectorAll(".session");
  const n = panes.length + 1; // include the pane about to be mounted
  const share = 1 / n;
  for (const p of panes) p.style.flex = `${share} 1 0`;
}

function renderGutters() {
  for (const g of document.querySelectorAll(".gutter")) g.remove();
  const panes = [...document.querySelectorAll(".session")];
  for (let i = 0; i < panes.length - 1; i++) {
    const g = document.createElement("div");
    g.className = "gutter";
    g.addEventListener("pointerdown", onGutterDown);
    panes[i].after(g);
  }
}

// A pointer gesture is by nature global — only one can be active at a time.
let drag = null;

function onGutterDown(ev) {
  ev.preventDefault();
  const gutter = ev.currentTarget;
  const left = gutter.previousElementSibling;
  const right = gutter.nextElementSibling;
  if (!left || !right) return;
  const leftRect = left.getBoundingClientRect();
  const rightRect = right.getBoundingClientRect();
  drag = {
    gutter,
    left,
    right,
    startX: ev.clientX,
    leftStart: leftRect.width,
    rightStart: rightRect.width,
    totalFlex:
      (parseFloat(left.style.flexGrow) || 1) +
      (parseFloat(right.style.flexGrow) || 1),
  };
  gutter.classList.add("dragging");
  gutter.setPointerCapture(ev.pointerId);
  gutter.addEventListener("pointermove", onGutterMove);
  gutter.addEventListener("pointerup", onGutterUp);
  gutter.addEventListener("pointercancel", onGutterUp);
}

function onGutterMove(ev) {
  if (!drag) return;
  const dx = ev.clientX - drag.startX;
  // Keep both panes above a minimum width so the flex ratios we emit don't
  // go pathological. The browser would clamp visually anyway, but the
  // persisted layout then wouldn't round-trip cleanly.
  const MIN = 120;
  const leftW = Math.max(MIN, drag.leftStart + dx);
  const rightW = Math.max(MIN, drag.rightStart - dx);
  const combined = leftW + rightW;
  const leftShare = (leftW / combined) * drag.totalFlex;
  const rightShare = (rightW / combined) * drag.totalFlex;
  drag.left.style.flex = `${leftShare} 1 0`;
  drag.right.style.flex = `${rightShare} 1 0`;
}

function onGutterUp(ev) {
  if (!drag) return;
  const g = drag.gutter;
  g.classList.remove("dragging");
  g.removeEventListener("pointermove", onGutterMove);
  g.removeEventListener("pointerup", onGutterUp);
  g.removeEventListener("pointercancel", onGutterUp);
  try {
    g.releasePointerCapture(ev.pointerId);
  } catch {
    // The pointer may already have been released by the browser (e.g.,
    // pointercancel); `releasePointerCapture` throws in that case.
  }
  drag = null;
  putLayout();
}

main();
