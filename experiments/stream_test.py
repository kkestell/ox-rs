# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""
Streaming tool-call experiment. Tests models through a full agent loop
(user msg -> tool calls -> tool results -> ... -> stop) and logs every
SSE chunk so we can see exactly how tool calls arrive over the wire.

Usage:
    uv run --script experiments/stream_test.py
    uv run --script experiments/stream_test.py deepseek/deepseek-v3.2

Results (2026-04-14, 2 runs per model)
=======================================

Tested 14 models across major providers. Each model was run twice to check
for non-determinism. Key findings:

1. TOOL CALL STREAM SHAPE
   All providers use the same OpenAI-compatible delta shape for tool calls:
   - `delta.tool_calls[i].id` + `delta.tool_calls[i].function.name` in first delta
   - `delta.tool_calls[i].function.arguments` streamed in subsequent deltas
   - `finish_reason: "tool_calls"` signals the model wants tool results

2. PARALLEL vs SEQUENTIAL TOOL CALLS
   Most models emit both tool calls in a single iteration (parallel), but this
   is NON-DETERMINISTIC for several models. Across 2 runs:

   Consistently parallel: mistral-nemo, qwen-turbo, minimax-m2.7,
     claude-haiku-4.5, gpt-5-mini, gemini-3-flash, nemotron-nano-9b,
     amazon nova-micro
   Consistently sequential: gpt-oss-20b
   Non-deterministic: deepseek-v3.2 (parallel then sequential),
     qwen3.6-plus (sequential then parallel), llama-3.1-8b (no-tools then
     sequential), llama-3-8b (no-tools then parallel)

3. TEXT + TOOL INTERLEAVING
   Whether text precedes tool calls in the same iteration is also
   non-deterministic for some models (e.g. gpt-5-nano: tools-only in run 1,
   text+tools in run 2). Only the following were stable across both runs:
   - Always tools-only: mistral-nemo, gemini-3-flash, gpt-5-mini
   - Always text-then-tools: qwen-turbo, claude-haiku-4.5, amazon nova-micro
   Others varied between runs.

4. UNRELIABLE TOOL CALLING
   - meta-llama/llama-3.1-8b-instruct (Cerebras): ignored tools in run 1,
     used them in run 2. Unreliable.
   - meta-llama/llama-3-8b-instruct (DeepInfra): same — ignored tools in
     run 1, used them in run 2.
   - openai/gpt-oss-20b (DeepInfra): corrupted tool name with <|channel|>
     token in run 1, clean in run 2. Unreliable.

5. CHUNK GRANULARITY
   Varies wildly by provider. Google Gemini sends 4-7 chunks total.
   OpenAI reasoning models send 10-250 chunks. Most others 10-75.

6. FINISH REASON
   Consistent: "tool_calls" when tools are requested, "stop" for final response.

7. REASONING + TOOLS
   Models with reasoning support stream reasoning BEFORE tool calls in the
   same iteration. Reasoning tokens are billed. This means the first delta
   often contains both `delta.reasoning` and empty `delta.content`.
   Models observed: gpt-oss-20b, gpt-5-mini, gpt-5-nano, nemotron-nano,
   minimax-m2.7, qwen3.6-plus, google gemini (encrypted).
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

FAKE_RESULTS = {
    "get_weather": {"temp": "22C", "condition": "partly cloudy", "humidity": "65%"},
    "get_population": {"population": "13,960,000", "year": 2023},
}

PROMPT = (
    "What's the weather AND population of Tokyo? "
    "Use both tools. Also tell me briefly what you're doing."
)


def load_tool_models() -> list[dict]:
    """Return models that support 'tools', sorted by prompt price (cheapest first)."""
    catalog = json.loads(MODELS_FILE.read_text())
    out = []
    for m in catalog:
        params = set(m.get("supported_parameters", []))
        if "tools" not in params:
            continue
        out.append({
            "id": m["id"],
            "name": m["name"],
            "has_tool_choice": "tool_choice" in params,
            "has_parallel_tool_calls": "parallel_tool_calls" in params,
            "has_reasoning": "reasoning" in params,
            "prompt_price": m["pricing"]["prompt"],
        })
    out.sort(key=lambda m: float(m["prompt_price"]))
    return out


def stream_completion(model, messages):
    """Stream one LLM call. Yields (event_type, detail) tuples.

    Event types:
      "text"          — a chunk of content text
      "tool_start"    — tool_calls delta with id/name (beginning of a call)
      "tool_args"     — tool_calls delta with argument fragment
      "finish"        — finish_reason string
      "usage"         — usage dict
      "raw_chunk"     — the full parsed chunk (emitted for every SSE line)
      "error"         — error from the API
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
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

            if "usage" in chunk:
                yield ("usage", chunk["usage"])

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                fr = choice.get("finish_reason")

                if fr:
                    yield ("finish", fr)

                # Text content
                c = delta.get("content")
                if c is not None and c != "":
                    yield ("text", c)

                # Tool calls
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    fn = tc.get("function", {})
                    if tc.get("id") or fn.get("name"):
                        yield ("tool_start", {"index": idx, "id": tc.get("id", ""), "name": fn.get("name", "")})
                    args = fn.get("arguments", "")
                    if args:
                        yield ("tool_args", {"index": idx, "args": args})


def run_agent_loop(model_id: str):
    """Run a full agent loop (request -> tool calls -> tool results -> ... -> stop).
    Returns a list of iteration reports."""
    messages = [{"role": "user", "content": PROMPT}]
    iterations = []

    for iteration_num in range(1, 8):
        text_parts = []
        tool_calls = {}  # index -> {id, name, arguments}
        finish_reason = None
        usage = None
        chunk_count = 0
        first_raw = None
        event_log = []  # (type, count) groups for shape display

        current_event_type = None
        current_event_count = 0

        for etype, val in stream_completion(model_id, messages):
            if etype == "raw_chunk":
                chunk_count += 1
                if first_raw is None:
                    first_raw = val
                continue
            if etype == "error":
                iterations.append({"error": val, "iteration": iteration_num})
                return iterations

            # Track event shape
            if etype == current_event_type:
                current_event_count += 1
            else:
                if current_event_type is not None:
                    event_log.append((current_event_type, current_event_count))
                current_event_type = etype
                current_event_count = 1

            if etype == "text":
                text_parts.append(val)
            elif etype == "tool_start":
                idx = val["index"]
                tool_calls[idx] = {"id": val["id"], "name": val["name"], "arguments": ""}
            elif etype == "tool_args":
                idx = val["index"]
                if idx in tool_calls:
                    tool_calls[idx]["arguments"] += val["args"]
            elif etype == "finish":
                finish_reason = val
            elif etype == "usage":
                usage = val

        if current_event_type is not None:
            event_log.append((current_event_type, current_event_count))

        text = "".join(text_parts)
        tools = list(tool_calls.values())

        iterations.append({
            "iteration": iteration_num,
            "finish_reason": finish_reason,
            "text": text,
            "tool_calls": tools,
            "usage": usage,
            "chunk_count": chunk_count,
            "first_raw": first_raw,
            "event_log": event_log,
        })

        if finish_reason == "stop" or not tools:
            break

        # Append assistant message + tool results for next iteration
        assistant_msg = {"role": "assistant", "content": text or None}
        assistant_msg["tool_calls"] = [
            {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in tools
        ]
        messages.append(assistant_msg)

        for t in tools:
            messages.append({
                "role": "tool",
                "tool_call_id": t["id"],
                "content": json.dumps(FAKE_RESULTS.get(t["name"], {"error": "unknown tool"})),
            })

    return iterations


def print_event_shape(event_log):
    """Print a compact representation of the event stream shape."""
    for etype, count in event_log:
        if etype == "text":
            print(f"      text          x{count}")
        elif etype == "tool_start":
            print(f"      tool_start    x{count}")
        elif etype == "tool_args":
            print(f"      tool_args     x{count}")
        elif etype == "finish":
            print(f"      finish        x{count}")
        elif etype == "usage":
            print(f"      usage         x{count}")


def run_model(model_info: dict):
    """Run the full agent loop for one model and print a structured report."""
    print(f"\n{'='*70}")
    print(f"  {model_info['name']}  ({model_info['id']})")
    print(f"  tool_choice={model_info['has_tool_choice']}  parallel={model_info['has_parallel_tool_calls']}")
    print(f"{'='*70}")

    iterations = run_agent_loop(model_info["id"])

    for it in iterations:
        if "error" in it:
            print(f"\n  iteration {it['iteration']}  ERROR: {it['error']}")
            return

        print(f"\n  iteration {it['iteration']}  (finish={it['finish_reason']}, chunks={it['chunk_count']})")

        # Event shape
        print(f"    stream shape:")
        print_event_shape(it["event_log"])

        # First raw chunk
        if it["first_raw"] and it["iteration"] == 1:
            print(f"    first chunk: {json.dumps(it['first_raw'], indent=2)[:400]}")

        # Text
        if it["text"]:
            preview = it["text"][:120].replace("\n", "\\n")
            print(f"    text: {preview!r}{'...' if len(it['text']) > 120 else ''}")

        # Tool calls
        if it["tool_calls"]:
            for t in it["tool_calls"]:
                print(f"    call: {t['name']}({t['arguments']})  id={t['id']}")

        # Usage
        if it["usage"]:
            print(f"    usage: {json.dumps(it['usage'])}")

    # Summary
    total_iterations = len(iterations)
    last = iterations[-1] if iterations else {}
    text_iterations = sum(1 for it in iterations if it.get("text"))
    tool_iterations = sum(1 for it in iterations if it.get("tool_calls"))

    print(f"\n  SUMMARY: {total_iterations} iterations, {tool_iterations} with tool calls, "
          f"final finish={last.get('finish_reason')}")

    # Check: did it call both tools?
    all_tool_names = set()
    for it in iterations:
        for tc in it.get("tool_calls", []):
            all_tool_names.add(tc["name"])
    if all_tool_names == {"get_weather", "get_population"}:
        print(f"  -> Called BOTH tools")
    elif all_tool_names:
        print(f"  -> Called: {all_tool_names} (MISSING some)")
    else:
        print(f"  -> Called NO tools")

    # Check: parallel vs sequential tool calls
    for it in iterations:
        tcs = it.get("tool_calls", [])
        if len(tcs) >= 2:
            print(f"  -> PARALLEL tool calls in iteration {it['iteration']} ({len(tcs)} calls)")
            break
    else:
        tool_iters = [it for it in iterations if it.get("tool_calls")]
        if len(tool_iters) >= 2:
            print(f"  -> SEQUENTIAL tool calls across {len(tool_iters)} iterations")

    print()


def main():
    all_models = load_tool_models()

    args = sys.argv[1:]
    if args:
        by_id = {m["id"]: m for m in all_models}
        selected = []
        for mid in args:
            if mid in by_id:
                selected.append(by_id[mid])
            else:
                selected.append({
                    "id": mid, "name": mid,
                    "has_tool_choice": False,
                    "has_parallel_tool_calls": False,
                    "has_reasoning": False,
                    "prompt_price": "?",
                })
    else:
        selected = all_models[:5]

    print(f"Prompt: {PROMPT!r}\n")
    print(f"Testing {len(selected)} models (streaming tool calls):")
    for m in selected:
        print(f"  - {m['id']}  (tool_choice={m['has_tool_choice']}, parallel={m['has_parallel_tool_calls']})")

    for m in selected:
        try:
            run_model(m)
        except Exception as e:
            print(f"\n  EXCEPTION for {m['id']}: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
