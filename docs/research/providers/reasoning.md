# LLM Provider Notes (via OpenRouter)

All models are accessed through the OpenRouter `/api/v1/chat/completions`
endpoint. This document captures provider-specific behaviors we discovered
through experimentation (see `experiments/reasoning_test.py`).

## Reasoning Support

Out of 233 models in the OpenRouter catalog, 131 advertise support for
`reasoning` and/or `include_reasoning` parameters. Actual behavior varies
significantly by provider.

### Request Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `reasoning` | object | Structured reasoning control. `{}` enables with defaults. |
| `reasoning.effort` | string | `"high"`, `"medium"`, `"low"`, etc. Required for Anthropic. |
| `reasoning.max_tokens` | integer | Explicit reasoning token budget. Mutually exclusive with `effort`. |
| `reasoning.exclude` | bool | If true, model reasons internally but output is suppressed. |
| `include_reasoning` | bool | Legacy toggle. Equivalent to `reasoning: {}`. |

### Provider Behaviors

There are three distinct tiers of reasoning support:

#### 1. Open Reasoning (OSS / open-weight models)

Send `reasoning: {}` with `include_reasoning: true`. Readable chain-of-thought
streams back in real time.

Tested models:
- `openai/gpt-oss-20b` (Novita)
- `openai/gpt-oss-120b` (Parasail)
- `nvidia/nemotron-nano-9b-v2` (DeepInfra)
- `arcee-ai/trinity-mini` (Clarifai)
- `qwen/qwen3.5-9b` (Venice)

Request:
```json
{
  "reasoning": {},
  "include_reasoning": true
}
```

Streaming delta shape:
```json
{
  "delta": {
    "reasoning": "Let me think...",
    "reasoning_details": [
      {
        "type": "reasoning.text",
        "text": "Let me think...",
        "format": "unknown",
        "index": 0
      }
    ],
    "content": ""
  }
}
```

Notes:
- `delta.reasoning` (legacy string) and `delta.reasoning_details` (structured
  array) are redundant — same text, chunk by chunk.
- `format` is always `"unknown"` for these models.
- Chunk granularity varies by provider infrastructure, not model. Clarifai
  sends sentence-level chunks (~38 for 1300 chars) while Novita/Venice stream
  token-by-token (~2000 for similar length).

#### 2. Anthropic (requires explicit effort)

Sending `reasoning: {}` alone produces **no reasoning output**. You must
include `effort` (or `max_tokens`) to activate extended thinking, despite the
catalog not listing `reasoning_effort` as a supported parameter for these
models.

Tested models:
- `anthropic/claude-haiku-4.5` (Amazon Bedrock)

Request:
```json
{
  "reasoning": { "effort": "high" },
  "include_reasoning": true
}
```

Streaming delta shape:
```json
{
  "delta": {
    "reasoning": "Let me work through this...",
    "reasoning_details": [
      {
        "type": "reasoning.text",
        "text": "Let me work through this...",
        "format": "anthropic-claude-v1",
        "index": 0
      }
    ]
  }
}
```

The final `reasoning_details` delta includes a `signature` field — a base64
blob Anthropic uses to verify reasoning integrity:
```json
{
  "type": "reasoning.text",
  "signature": "EoIFCkgIDBABGAIqQM0eNf2/4E+UNudb...",
  "format": "anthropic-claude-v1",
  "index": 0
}
```

Differences from OSS models:
- `format` is `"anthropic-claude-v1"` (not `"unknown"`)
- Final delta carries a `signature` field
- `effort` is **required** — without it, zero reasoning output

#### 3. Encrypted Reasoning (OpenAI, Google)

These providers run reasoning internally but return opaque encrypted blobs
instead of readable text. The legacy `delta.reasoning` field is absent or null.

Tested models:
- `openai/gpt-5-mini` (OpenAI)
- `google/gemini-3-flash-preview` (Google)

Request:
```json
{
  "reasoning": {},
  "include_reasoning": true
}
```

Streaming delta shape:
```json
{
  "delta": {
    "reasoning_details": [
      {
        "type": "reasoning.encrypted",
        "data": "gAAAAABp3nWY4gabBRg5...",
        "format": "google-gemini-v1"
      }
    ]
  }
}
```

Notes:
- No readable reasoning is available. GPT-5 Mini explicitly states: "Sorry — I
  can't share my step-by-step internal chain-of-thought."
- Encrypted reasoning blobs should be passed back unmodified in multi-turn
  conversations so the model can maintain its chain of thought.
- You are still billed for reasoning tokens (GPT-5 Mini: 512 reasoning tokens).
- Gemini reports 0 reasoning tokens despite returning encrypted data — appears
  to be a reporting gap.

### Token Accounting

All providers include `completion_tokens_details.reasoning_tokens` in the usage
object, allowing reasoning and content tokens to be tracked separately:

```json
{
  "usage": {
    "prompt_tokens": 72,
    "completion_tokens": 289,
    "total_tokens": 361,
    "completion_tokens_details": {
      "reasoning_tokens": 130
    }
  }
}
```

| Provider | Reasoning tokens reported | Billed |
|----------|--------------------------|--------|
| OSS models | Yes, accurate | Yes, at completion rate |
| Anthropic | Yes, accurate | Yes, at completion rate |
| OpenAI | Yes (512 for GPT-5 Mini) | Yes, despite encryption |
| Google | 0 (appears underreported) | Unclear |

### Practical Implications for ox

When making a request, determine the reasoning strategy from the model ID:

1. **Anthropic models** (`anthropic/*`): Must send `reasoning: {"effort": "high"}`
   to activate thinking. Without effort, reasoning is silently disabled.

2. **OpenAI proprietary / Google models** (`openai/gpt-5*`, `google/*`):
   Reasoning is encrypted. Still send `reasoning: {}` + `include_reasoning: true`
   to enable it for quality, but there is nothing to display to the user. Pass
   encrypted blobs back in multi-turn.

3. **Everything else** (OSS/open-weight): Send `reasoning: {}` +
   `include_reasoning: true`. Read `delta.reasoning` for displayable
   chain-of-thought.

To read reasoning from the stream, consume `delta.reasoning` (simple string)
rather than `delta.reasoning_details` — the two are redundant for readable
models, and `delta.reasoning` is absent for encrypted models, making it a
natural discriminator.
