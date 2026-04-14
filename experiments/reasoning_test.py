# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""
Reasoning streaming experiment. Loads the OpenRouter model catalog, filters
to models that support reasoning parameters, picks the cheapest 5, and
streams a thinking-heavy prompt at each one.  Logs every SSE chunk so we
can see exactly how reasoning data arrives over the wire.

Usage:
    uv run --script experiments/reasoning_test.py
    uv run --script experiments/reasoning_test.py deepseek/deepseek-r1 qwen/qwen3-8b

Results (2026-04-14)
====================

Three distinct behaviors observed across providers:

1. OPEN REASONING (delta.reasoning + delta.reasoning_details type=reasoning.text)
   Models stream readable chain-of-thought in both the legacy `reasoning` field
   and structured `reasoning_details` array.  The two are redundant — same text,
   chunk by chunk.  `format` is always "unknown" for these.

   Model                        Provider    Chunks  Reasoning chars  Content chars
   -------------------------    ---------   ------  ---------------  -------------
   openai/gpt-oss-20b          Novita       2047   7441             0 (hit length)
   openai/gpt-oss-120b         Parasail      928   2920             426
   nvidia/nemotron-nano-9b-v2  DeepInfra     513   1429             348
   arcee-ai/trinity-mini       Clarifai       54   1341             443
   qwen/qwen3.5-9b             Venice       2042   6674             255 (hit length)

   Chunk granularity varies by provider: Clarifai sends sentence-level chunks
   (38 reasoning deltas for 1341 chars) while Novita/Venice stream token-level
   (~2000 deltas).

2. ENCRYPTED REASONING (delta.reasoning_details type=reasoning.encrypted)
   Models think internally but return opaque encrypted blobs instead of readable
   text.  The legacy `reasoning` field is absent or null.

   Model                           Provider  Format            Reasoning tokens
   ----------------------------    --------  ----------------  ----------------
   google/gemini-3-flash-preview   Google    google-gemini-v1  0 (not reported)
   openai/gpt-5-mini               OpenAI    (base64 blob)     512

   GPT-5 Mini explicitly refuses to share reasoning: "Sorry—I can't share my
   step-by-step internal chain-of-thought."  Gemini returns a short encrypted
   payload in a single delta.

3. OPEN REASONING — REQUIRES EFFORT (Anthropic)
   Sending `reasoning: {}` alone produces NO reasoning output.  Must send
   `reasoning: {"effort": "high"}` (or max_tokens) to activate extended
   thinking.  Once activated, streams readable CoT like the OSS models.

   Model                       Provider        Chunks  Reasoning chars  Content chars
   -------------------------   --------------  ------  ---------------  -------------
   anthropic/claude-haiku-4.5  Amazon Bedrock      84  477              367

   Differences from OSS models:
   - `format` is "anthropic-claude-v1" (not "unknown")
   - Final reasoning_details delta includes a `signature` field (base64 blob
     for verifying reasoning integrity — unique to Anthropic)
   - 130 reasoning_tokens billed

   Without effort: 0 reasoning chunks, 0 reasoning_tokens, just content.

Usage & billing:
  - `completion_tokens_details.reasoning_tokens` tracks thinking tokens
    separately from content tokens in all cases.
  - Open-reasoning models bill reasoning tokens at the completion rate.
  - GPT-5 Mini billed 512 reasoning tokens despite encrypting the output.
  - Gemini reported 0 reasoning tokens.
  - Anthropic reported 130 reasoning tokens (with effort=high).

Key takeaway for ox:
  - For open-source / open-weight models: just set `reasoning: {}` +
    `include_reasoning: true` and read `delta.reasoning`.
  - For Gemini/OpenAI: reasoning is encrypted or refused — these providers
    keep chain-of-thought private.  Still billed for it.
  - For Anthropic: must pass `reasoning: {"effort": "..."}` or
    `reasoning: {"max_tokens": N}` to activate.  Once on, same shape as
    OSS models but with format="anthropic-claude-v1" and a signature.
"""

import httpx
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_FILE = Path(__file__).parent / "openrouter_models.json"

PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep does the farmer have left? "
    "Think through this carefully step by step before answering."
)


def load_reasoning_models() -> list[dict]:
    """Return models that support 'reasoning' and/or 'include_reasoning',
    sorted by prompt price (cheapest first)."""
    catalog = json.loads(MODELS_FILE.read_text())
    out = []
    for m in catalog:
        params = set(m.get("supported_parameters", []))
        has_reasoning = "reasoning" in params
        has_include = "include_reasoning" in params
        if has_reasoning or has_include:
            out.append({
                "id": m["id"],
                "name": m["name"],
                "has_reasoning": has_reasoning,
                "has_include_reasoning": has_include,
                "has_reasoning_effort": "reasoning_effort" in params,
                "prompt_price": m["pricing"]["prompt"],
            })
    out.sort(key=lambda m: float(m["prompt_price"]))
    return out


def build_extra_body(model_info: dict) -> dict:
    """Build the reasoning request params based on what the model supports."""
    body: dict = {}
    if model_info["has_reasoning"]:
        reasoning_obj: dict = {}
        if model_info["has_reasoning_effort"]:
            reasoning_obj["effort"] = "high"
        body["reasoning"] = reasoning_obj
    if model_info["has_include_reasoning"]:
        body["include_reasoning"] = True
    return body


def stream_model(model_id: str, extra_body: dict):
    """Stream a chat completion. Yields (event_type, detail) tuples.

    Event types:
      "reasoning_delta"   — a chunk of reasoning text
      "reasoning_detail"  — a structured reasoning_details delta
      "content_delta"     — a chunk of content text
      "finish"            — finish_reason string
      "usage"             — usage dict
      "unknown_delta_key" — any delta key we didn't expect
      "error"             — error from the API
      "raw_chunk"         — the full parsed chunk (emitted for every SSE line)
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 2048,
        "stream": True,
        **extra_body,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
    }

    with httpx.stream("POST", BASE_URL, json=payload, headers=headers, timeout=120.0) as r:
        if r.status_code != 200:
            body = "".join(r.iter_lines())
            yield ("error", f"HTTP {r.status_code}: {body[:500]}")
            return

        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            yield ("raw_chunk", chunk)

            # Usage often arrives in the final chunk at top level
            if "usage" in chunk:
                yield ("usage", chunk["usage"])

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                fr = choice.get("finish_reason")

                if fr:
                    yield ("finish", fr)

                # Reasoning text (legacy field on delta)
                r_text = delta.get("reasoning")
                if r_text is not None and r_text != "":
                    yield ("reasoning_delta", r_text)

                # Reasoning details (structured array on delta)
                r_details = delta.get("reasoning_details")
                if r_details:
                    for d in r_details:
                        yield ("reasoning_detail", d)

                # Content text
                c = delta.get("content")
                if c is not None and c != "":
                    yield ("content_delta", c)

                # Flag anything unexpected in the delta
                known = {"role", "content", "reasoning", "reasoning_details",
                         "tool_calls", "refusal"}
                for k in delta:
                    if k not in known:
                        yield ("unknown_delta_key", {k: delta[k]})


def run_model(model_info: dict):
    """Stream one model and print a structured report."""
    extra = build_extra_body(model_info)

    print(f"\n{'='*70}")
    print(f"  {model_info['name']}  ({model_info['id']})")
    print(f"  params sent: {json.dumps(extra)}")
    print(f"{'='*70}")

    reasoning_chunks: list[str] = []
    detail_items: list[dict] = []
    content_chunks: list[str] = []
    unknown_keys: dict[str, list] = {}
    finish_reason = None
    usage = None
    chunk_count = 0
    first_raw: dict | None = None

    for etype, val in stream_model(model_info["id"], extra):
        if etype == "raw_chunk":
            chunk_count += 1
            if first_raw is None:
                first_raw = val
            continue
        if etype == "error":
            print(f"  ERROR: {val}")
            return
        if etype == "reasoning_delta":
            reasoning_chunks.append(val)
        elif etype == "reasoning_detail":
            detail_items.append(val)
        elif etype == "content_delta":
            content_chunks.append(val)
        elif etype == "finish":
            finish_reason = val
        elif etype == "usage":
            usage = val
        elif etype == "unknown_delta_key":
            for k, v in val.items():
                unknown_keys.setdefault(k, []).append(v)

    # -- First raw chunk (to see the shape) --
    print(f"\n  first chunk:")
    print(f"    {json.dumps(first_raw, indent=2)[:500]}")

    # -- Reasoning (legacy delta.reasoning) --
    reasoning_text = "".join(reasoning_chunks)
    if reasoning_text:
        preview = reasoning_text[:400].replace("\n", "\n    ")
        print(f"\n  [reasoning] {len(reasoning_chunks)} chunks, {len(reasoning_text)} chars:")
        print(f"    {preview}")
        if len(reasoning_text) > 400:
            print(f"    ... ({len(reasoning_text) - 400} more chars)")
    else:
        print(f"\n  [reasoning] (absent from delta — {len(reasoning_chunks)} chunks)")

    # -- Reasoning details (structured delta.reasoning_details) --
    if detail_items:
        types_seen = {}
        for d in detail_items:
            t = d.get("type", "?")
            types_seen[t] = types_seen.get(t, 0) + 1
        print(f"\n  [reasoning_details] {len(detail_items)} deltas, types: {types_seen}")
        # Show first and last detail to see the shape
        print(f"    first: {json.dumps(detail_items[0])[:300]}")
        if len(detail_items) > 1:
            print(f"    last:  {json.dumps(detail_items[-1])[:300]}")
    else:
        print(f"  [reasoning_details] (absent from delta)")

    # -- Content --
    content_text = "".join(content_chunks)
    if content_text:
        preview = content_text[:300].replace("\n", "\n    ")
        print(f"\n  [content] {len(content_chunks)} chunks, {len(content_text)} chars (finish={finish_reason}):")
        print(f"    {preview}")
    else:
        print(f"\n  [content] (empty, finish={finish_reason})")

    # -- Usage --
    if usage:
        print(f"\n  [usage] {json.dumps(usage)}")

    # -- Unknown keys --
    if unknown_keys:
        print(f"\n  [unknown delta keys]")
        for k, vals in unknown_keys.items():
            print(f"    {k}: {len(vals)} occurrences, sample={json.dumps(vals[0])[:200]}")

    # -- Summary --
    print(f"\n  [stream summary] {chunk_count} SSE chunks total")
    print()


def main():
    all_models = load_reasoning_models()

    # --effort forces reasoning.effort on all selected models
    # --no-include disables include_reasoning (test: billed but hidden?)
    args = sys.argv[1:]
    force_effort = False
    no_include = False
    for flag in ("--effort", "--no-include"):
        if flag in args:
            if flag == "--effort":
                force_effort = True
            elif flag == "--no-include":
                no_include = True
            args = [a for a in args if a != flag]

    if args:
        by_id = {m["id"]: m for m in all_models}
        selected = []
        for mid in args:
            if mid in by_id:
                selected.append(by_id[mid])
            else:
                selected.append({
                    "id": mid, "name": mid,
                    "has_reasoning": True,
                    "has_include_reasoning": True,
                    "has_reasoning_effort": False,
                    "prompt_price": "?",
                })
    else:
        selected = all_models[:5]

    if force_effort:
        for m in selected:
            m["has_reasoning_effort"] = True
    if no_include:
        for m in selected:
            m["has_include_reasoning"] = False

    print(f"Prompt: {PROMPT!r}\n")
    print(f"Testing {len(selected)} models (streaming):")
    for m in selected:
        extra = build_extra_body(m)
        print(f"  - {m['id']}  extra={json.dumps(extra)}")

    for m in selected:
        try:
            run_model(m)
        except Exception as e:
            print(f"\n  EXCEPTION for {m['id']}: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
