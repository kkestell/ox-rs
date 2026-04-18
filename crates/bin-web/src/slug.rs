//! Production [`SlugGenerator`] backed by a direct OpenRouter
//! chat/completions call.
//!
//! The slug rename flow (see `SessionLifecycle::on_first_turn_complete`)
//! asks the LLM for a short kebab-case label derived from the user's
//! first message, then renames the session's git branch and worktree
//! directory from `ox/<short-uuid>` to `ox/<slug>-<short-uuid>`. The
//! whole flow is best-effort: every failure mode here — network error,
//! timeout, malformed response, non-conforming slug — returns `None`
//! and lets the coordinator keep the original short-UUID name.
//!
//! Implementation notes:
//!
//! - Uses `reqwest` directly rather than routing through
//!   [`adapter_llm::OpenRouterProvider`]. That provider is built around
//!   streaming chat turns with tool calls — overkill for a 1-turn,
//!   non-streaming, JSON-response call. Keeping this path self-contained
//!   means a change to the slug format doesn't drag the full chat
//!   pipeline along.
//! - 10-second timeout (`SLUG_TIMEOUT`) via `tokio::time::timeout`. If
//!   the generator hangs the session stays on its short-UUID name —
//!   never blocks session progress.
//! - `temperature: 0.0` keeps the slug deterministic for identical first
//!   messages (useful when debugging: re-sending the same first line on
//!   a new session gets the same slug).
//! - The prompt asks for JSON with a single `slug` field. We parse it
//!   with `serde_json` and reject any shape mismatch as `None`.
//! - The validator enforces `^[a-z0-9]+(-[a-z0-9]+){0,4}$` — up to five
//!   lowercase-alphanum segments separated by hyphens. The LLM
//!   occasionally returns an empty string, leading/trailing hyphens, or
//!   mixed case; all of those are rejected.

use std::time::Duration;

use agent_host::SlugGenerator;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

const SLUG_TIMEOUT: Duration = Duration::from_secs(10);

/// System prompt that pins the model to the slug contract. Short and
/// machine-readable — the model's only job here is to emit a single
/// JSON object, so we leave all the "why" context out and just tell it
/// what to return.
const SLUG_SYSTEM_PROMPT: &str = "You generate short kebab-case slugs that summarize a software task request. Return ONLY a JSON object of the form {\"slug\": \"...\"} with a slug that is 1-5 lowercase-alphanumeric segments separated by single hyphens, 3-40 characters total. No leading or trailing hyphens. No punctuation. No quotes. No commentary.";

pub struct CliSlugGenerator {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl CliSlugGenerator {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl SlugGenerator for CliSlugGenerator {
    async fn generate(&self, first_message: &str) -> Option<String> {
        match tokio::time::timeout(SLUG_TIMEOUT, self.request(first_message)).await {
            Ok(Ok(Some(slug))) => Some(slug),
            Ok(Ok(None)) => None,
            Ok(Err(err)) => {
                eprintln!("ox: slug generator request failed: {err:#}");
                None
            }
            Err(_) => {
                eprintln!(
                    "ox: slug generator timed out after {}s",
                    SLUG_TIMEOUT.as_secs()
                );
                None
            }
        }
    }
}

impl CliSlugGenerator {
    /// Issue the OpenRouter request and pull a validated slug out of the
    /// response. Any shape mismatch returns `Ok(None)` (not `Err`) so
    /// the outer `generate` logs a different message for transport-level
    /// errors vs. content-level rejects.
    async fn request(&self, first_message: &str) -> Result<Option<String>, reqwest::Error> {
        let body = RequestBody {
            model: &self.model,
            temperature: 0.0,
            response_format: ResponseFormat {
                kind: "json_object",
            },
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: SLUG_SYSTEM_PROMPT,
                },
                ChatMessage {
                    role: "user",
                    content: first_message,
                },
            ],
        };

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let parsed: ChatResponse = response.json().await?;
        let Some(choice) = parsed.choices.into_iter().next() else {
            return Ok(None);
        };
        let text = choice.message.content;

        // The model occasionally wraps the JSON in Markdown code fences
        // despite the `response_format` hint. Tolerate a leading/trailing
        // ``` fence by stripping it before deserializing.
        let trimmed = strip_code_fence(text.trim());

        let envelope: SlugEnvelope = match serde_json::from_str(trimmed) {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        Ok(validate_slug(&envelope.slug))
    }
}

/// Return `Some(slug)` if it matches `^[a-z0-9]+(-[a-z0-9]+){0,4}$` and
/// is within the length budget, else `None`. Hand-rolled rather than
/// pulling in the `regex` crate — the rule is simple enough that an
/// explicit walk is both clearer and cheaper.
fn validate_slug(slug: &str) -> Option<String> {
    if slug.is_empty() || slug.len() > 40 {
        return None;
    }
    let mut segment_count = 0u8;
    let mut segment_len = 0usize;
    for ch in slug.chars() {
        if ch == '-' {
            // Leading hyphen, or two hyphens in a row → reject.
            if segment_len == 0 {
                return None;
            }
            segment_count += 1;
            if segment_count >= 5 {
                return None;
            }
            segment_len = 0;
        } else if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            segment_len += 1;
        } else {
            // Uppercase, punctuation, whitespace, non-ASCII: reject.
            return None;
        }
    }
    // Trailing hyphen → final segment has length 0.
    if segment_len == 0 {
        return None;
    }
    Some(slug.to_owned())
}

/// Strip a leading ```[lang]\n … \n``` fence if present. The model is
/// asked for raw JSON but sometimes inserts a fence anyway.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }
    // Drop the first line (the opening fence, with optional language tag).
    let after_open = match trimmed.find('\n') {
        Some(idx) => &trimmed[idx + 1..],
        None => return trimmed,
    };
    match after_open.rfind("```") {
        Some(idx) => after_open[..idx].trim(),
        None => after_open.trim(),
    }
}

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    temperature: f32,
    response_format: ResponseFormat,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

#[derive(Deserialize)]
struct SlugEnvelope {
    slug: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_single_segment() {
        assert_eq!(validate_slug("login"), Some("login".to_owned()));
    }

    #[test]
    fn validate_accepts_five_segments() {
        assert_eq!(
            validate_slug("fix-the-login-form-bug"),
            Some("fix-the-login-form-bug".to_owned())
        );
    }

    #[test]
    fn validate_rejects_six_segments() {
        // Six hyphen-separated segments exceeds the cap of five, which
        // keeps the final directory name from growing unbounded.
        assert_eq!(validate_slug("a-b-c-d-e-f"), None);
    }

    #[test]
    fn validate_rejects_uppercase() {
        assert_eq!(validate_slug("Fix-Login"), None);
    }

    #[test]
    fn validate_rejects_leading_hyphen() {
        assert_eq!(validate_slug("-fix"), None);
    }

    #[test]
    fn validate_rejects_trailing_hyphen() {
        assert_eq!(validate_slug("fix-"), None);
    }

    #[test]
    fn validate_rejects_double_hyphen() {
        assert_eq!(validate_slug("fix--login"), None);
    }

    #[test]
    fn validate_rejects_empty() {
        assert_eq!(validate_slug(""), None);
    }

    #[test]
    fn validate_rejects_whitespace() {
        assert_eq!(validate_slug("fix login"), None);
    }

    #[test]
    fn validate_rejects_punctuation() {
        assert_eq!(validate_slug("fix_login"), None);
        assert_eq!(validate_slug("fix.login"), None);
    }

    #[test]
    fn validate_rejects_overlong() {
        // 41 chars, just over the cap. Catches runaway model output
        // before it becomes a surprising directory name.
        let s = "a".repeat(41);
        assert_eq!(validate_slug(&s), None);
    }

    #[test]
    fn validate_accepts_digits() {
        assert_eq!(validate_slug("fix-bug-42"), Some("fix-bug-42".to_owned()));
    }

    #[test]
    fn strip_fence_removes_triple_backticks() {
        assert_eq!(
            strip_code_fence("```json\n{\"slug\": \"login\"}\n```"),
            "{\"slug\": \"login\"}"
        );
    }

    #[test]
    fn strip_fence_returns_input_unchanged_when_no_fence() {
        assert_eq!(
            strip_code_fence("{\"slug\": \"login\"}"),
            "{\"slug\": \"login\"}"
        );
    }

    #[test]
    fn strip_fence_handles_bare_backticks_without_language_tag() {
        assert_eq!(
            strip_code_fence("```\n{\"slug\": \"fix\"}\n```"),
            "{\"slug\": \"fix\"}"
        );
    }
}
