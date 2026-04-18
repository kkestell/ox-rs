//! OpenRouter-backed [`SlugGenerator`].
//!
//! The host asks for a short kebab-case label after the first completed turn.
//! This is provider-specific infrastructure, but intentionally separate from
//! the streaming chat provider because slugging is a non-streaming JSON call.

use std::time::Duration;

use agent_host::SlugGenerator;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

const SLUG_TIMEOUT: Duration = Duration::from_secs(10);
const SLUG_SYSTEM_PROMPT: &str = "You generate short kebab-case slugs that summarize a software task request. Return ONLY a JSON object of the form {\"slug\": \"...\"} with a slug that is 1-5 lowercase-alphanumeric segments separated by single hyphens, 3-40 characters total. No leading or trailing hyphens. No punctuation. No quotes. No commentary.";

pub struct OpenRouterSlugGenerator {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenRouterSlugGenerator {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl SlugGenerator for OpenRouterSlugGenerator {
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

impl OpenRouterSlugGenerator {
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
        let trimmed = strip_code_fence(choice.message.content.trim());
        let envelope: SlugEnvelope = match serde_json::from_str(trimmed) {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        Ok(validate_slug(&envelope.slug))
    }
}

fn validate_slug(slug: &str) -> Option<String> {
    if slug.is_empty() || slug.len() > 40 {
        return None;
    }
    let mut segment_count = 0u8;
    let mut segment_len = 0usize;
    for ch in slug.chars() {
        if ch == '-' {
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
            return None;
        }
    }
    if segment_len == 0 {
        return None;
    }
    Some(slug.to_owned())
}

fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }
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
