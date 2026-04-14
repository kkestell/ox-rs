# Streaming Tool Calls (via OpenRouter)

All models are accessed through the OpenRouter `/api/v1/chat/completions`
endpoint with `stream: true`. This document captures how tool calls arrive
over the wire. See `experiments/stream_test.py`.

Each model was tested twice to distinguish stable behavior from
non-deterministic variance. Several conclusions that looked solid after one
run turned out to be unreliable.

## Request Shape

Standard OpenAI-compatible tool-calling request:
```json
{
  "model": "...",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [{"type": "function", "function": {"name": "...", ...}}],
  "stream": true
}
```

## Streaming Delta Shape

All providers use the same wire format for streaming tool calls. A tool call
arrives as a sequence of SSE chunks:

**First delta** — carries the tool call ID and function name:
```json
{
  "delta": {
    "role": "assistant",
    "tool_calls": [
      {
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "get_weather", "arguments": "" }
      }
    ]
  }
}
```

**Subsequent deltas** — stream the arguments as string fragments:
```json
{
  "delta": {
    "tool_calls": [
      {
        "index": 0,
        "function": { "arguments": "{\"city\":" }
      }
    ]
  }
}
```

**Completion** — `finish_reason` signals intent:
```json
{
  "choices": [{ "finish_reason": "tool_calls" }]
}
```

The client accumulates `function.arguments` fragments and JSON-parses the
final concatenated string. The `index` field identifies which tool call
a fragment belongs to when multiple calls are in flight.

## Parallel vs Sequential Tool Calls

When asked to use two tools, models may emit both in a single iteration
(parallel) or one per iteration (sequential). This behavior is
**non-deterministic for several models** — the same model can go either way
across runs.

### Consistently parallel (2/2 runs)

Mistral Nemo, Qwen Turbo, MiniMax M2.7, Anthropic Claude Haiku 4.5, GPT-5
Mini, GPT-5 Nano, Gemini 3 Flash, Nemotron Nano 9B V2, Amazon Nova Micro.

### Consistently sequential (2/2 runs)

GPT-OSS-20B (always one tool per iteration).

### Non-deterministic

| Model | Run 1 | Run 2 |
|-------|-------|-------|
| DeepSeek V3.2 | Parallel | Sequential |
| Qwen 3.6 Plus | Sequential | Parallel |
| Llama 3.1 8B | No tools at all | Sequential |
| Llama 3 8B | No tools at all | Parallel |

The `parallel_tool_calls` parameter in the OpenRouter catalog is `false` for
all tested models, yet parallel behavior is the norm. This parameter appears
to reflect whether the model *advertises* parallel support, not whether it
actually does it.

### Implication for ox

The agent loop must handle both parallel and sequential tool calling from any
model. Never assume a model will be parallel just because it was last time.

## Text + Tool Interleaving

Whether the model streams explanatory text before tool calls in the same
iteration is **also non-deterministic** for some models. Only a subset
showed stable behavior across both runs:

### Consistently tools-only (no text in tool iteration)

Mistral Nemo, GPT-5 Mini, Google Gemini 3 Flash.

These models emit tool calls immediately, saving all commentary for the
response after tool results arrive.

```
tool_start[0] -> tool_args[0] -> tool_start[1] -> tool_args[1] -> finish
```

### Consistently text-then-tools

Qwen Turbo, Anthropic Claude Haiku 4.5, Amazon Nova Micro.

These models always stream explanatory text before tool calls.

```
text x14 -> tool_start[0] -> tool_args[0] -> tool_start[1] -> tool_args[1] -> finish
```

### Non-deterministic

| Model | Run 1 | Run 2 |
|-------|-------|-------|
| GPT-5 Nano | Tools-only | Text + tools |
| DeepSeek V3.2 | Text + tools | Text + tools (but sequential) |
| MiniMax M2.7 | Minimal text + tools | Text + tools |
| Nemotron Nano 9B | Tools-only | Tools-only |

One notable case: Llama 3 8B (DeepInfra) streamed text *after* tool calls
in run 2 (`text -> tool_start -> tool_args -> tool_start -> tool_args ->
text -> finish`), with 14 text chunks trailing the tool call deltas. This
means text can appear on *both sides* of tool calls.

### Implication for ox

The TUI must handle text arriving before, after, or interleaved with tool
calls. A simplistic "if `finish_reason` is `tool_calls`, there's no text"
assumption would drop content from most models.

## Tool Results and the Agent Loop

After receiving `finish_reason: "tool_calls"`, the client must:

1. Build the assistant message with both `content` (if any) and `tool_calls`
2. Append one `role: "tool"` message per tool call, keyed by `tool_call_id`
3. Send the extended conversation back for the next iteration

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "I'll look that up.", "tool_calls": [
    {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}
  ]},
  {"role": "tool", "tool_call_id": "call_abc", "content": "{\"temp\":\"22C\"}"}
]
```

All tested models correctly consumed tool results and produced a final
`finish_reason: "stop"` response incorporating the data (when they used
tools at all).

## Reasoning + Tool Calls

Models with reasoning support stream reasoning *before* tool calls in the
same iteration. The first SSE chunk typically contains both
`delta.reasoning` (or `delta.reasoning_details`) and empty `delta.content`:

```
reasoning x96 -> tool_start[0] -> tool_args[0] -> tool_start[1] -> ... -> finish
```

| Model | Reasoning tokens (tool iter) | Reasoning type |
|-------|------------------------------|----------------|
| GPT-OSS-20B | 58-106 | Open (`reasoning.text`) |
| Nemotron Nano 9B V2 | 183 (stable) | Open (`reasoning.text`) |
| MiniMax M2.7 | 30-44 | Open (`reasoning.text`) |
| Qwen 3.6 Plus | 306-715 | Open (`reasoning.text`) |
| GPT-5 Mini | 64-128 | Encrypted (`reasoning.encrypted`) |
| GPT-5 Nano | 256-384 | Encrypted (`reasoning.encrypted`) |
| Gemini 3 Flash | 0 (not reported) | Encrypted (`reasoning.encrypted`) |

Reasoning tokens are billed even when the model is just deciding which tools
to call. Token counts varied across runs, sometimes by 2x or more (Qwen 3.6
Plus: 306 vs 715).

## Unreliable Models

| Model | Provider | Issue |
|-------|----------|-------|
| meta-llama/llama-3.1-8b-instruct | Cerebras | Ignored tools in run 1, used them in run 2 |
| meta-llama/llama-3-8b-instruct | DeepInfra | Ignored tools in run 1, used them in run 2 |
| openai/gpt-oss-20b | DeepInfra | Corrupted tool name with `<\|channel\|>` token in run 1, clean in run 2 |

The Llama 8B models are unreliable for tool calling despite being listed in
the catalog with `tools` support. Whether they use tools at all appears to
be a coin flip. Larger Llama variants were not tested.

GPT-OSS-20B leaked a special token into a function name in one run, causing
tool result matching to fail. This is likely insufficient output filtering on
DeepInfra's end. It also consistently uses sequential tool calls, requiring
extra round trips.

## Chunk Granularity

SSE chunk count varies dramatically by provider, and is somewhat variable
across runs. Ranges shown from both runs:

| Provider | Chunks (tool iter) | Chunks (text iter) | Notes |
|----------|-------------------|-------------------|-------|
| Google | 7 | 4-7 | Very coarse, sentence-level |
| Mistral | 6 | 22-33 | Moderate |
| MiniMax | 10 | 2-5 | Coarse |
| Alibaba (Qwen Turbo) | 22 | 15-17 | Moderate |
| Amazon Bedrock | 19-86 | 26-125 | Variable |
| AtlasCloud (DeepSeek) | 9-22 | 66-70 | Moderate to fine |
| DeepInfra | 48-156 | 93-435 | Token-level |
| OpenAI | 9-182 | 58-366 | Highly variable (reasoning-dependent) |

For the TUI, chunk granularity affects perceived responsiveness. Token-level
streamers (DeepInfra, OpenAI) give smooth character-by-character display.
Sentence-level streamers (Google, MiniMax) produce jumpier updates.

## Token Accounting

All providers include a `usage` object in the final SSE chunk. Tool-calling
tokens count as `completion_tokens`:

```json
{
  "usage": {
    "prompt_tokens": 635,
    "completion_tokens": 111,
    "total_tokens": 746,
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

Multi-iteration agent loops accumulate costs across iterations. Prompt
tokens grow as the conversation history expands with tool results.

## Practical Implications for ox

1. **Accumulate tool calls by index.** Fragments arrive across multiple SSE
   chunks. Buffer `id`, `name`, and `arguments` per index, JSON-parse
   arguments only after `finish_reason: "tool_calls"`.

2. **Handle text anywhere in the stream.** Text can appear before, after, or
   interleaved with tool calls. It can also be absent entirely. Don't assume
   a fixed ordering.

3. **Support both parallel and sequential.** The agent loop must handle N
   tool calls in one iteration or one-at-a-time across iterations. Any model
   may switch between these across requests.

4. **Reasoning streams before tools.** When reasoning is enabled, reasoning
   deltas precede tool call deltas. The display should show "thinking..."
   then transition to "calling tool..." within the same response.

5. **Budget for variance.** Reasoning token counts vary significantly across
   runs (2x for some models). Sequential tool callers add extra iterations.
   Track `usage` per iteration for accurate cost accounting.

6. **Avoid Llama 8B for tool calling.** Despite catalog support, these
   models don't reliably emit tool calls. GPT-OSS-20B is also flaky
   (token corruption, always sequential).
