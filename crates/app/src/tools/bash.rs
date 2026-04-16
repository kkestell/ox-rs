//! `bash` tool.
//!
//! Executes a shell command in the workspace directory via `/bin/bash -c`.
//! Captures stdout and stderr, enforces a configurable timeout, and returns
//! a formatted result with labeled sections. Output is truncated at 2000
//! lines / 100KB to keep context-window cost bounded.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{Tool, require_non_empty};
use crate::ports::Shell;
use crate::stream::ToolDef;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_OUTPUT_LINES: usize = 2_000;
const MAX_OUTPUT_BYTES: usize = 100 * 1024; // 100 KB

pub struct BashTool<S> {
    shell: Arc<S>,
}

impl<S> BashTool<S> {
    pub fn new(shell: Arc<S>) -> Self {
        Self { shell }
    }
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    timeout_ms: Option<u64>,
}

impl<S: Shell + 'static> Tool for BashTool<S> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "bash".into(),
            description: "Execute a shell command in the workspace directory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds. Defaults to 120000 (2 minutes)."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: BashArgs =
                serde_json::from_str(args).context("bash: invalid JSON arguments")?;
            require_non_empty("command", &parsed.command)?;

            let timeout_ms = parsed.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
            let timeout = std::time::Duration::from_millis(timeout_ms);

            let output = self
                .shell
                .run(&parsed.command, timeout)
                .await
                .with_context(|| format!("bash: failed to run command: {}", parsed.command))?;

            Ok(format_result(&output))
        })
    }
}

/// Format the command output into labeled sections matching the ox reference
/// format. Includes timeout marker, exit code, and truncated stdout/stderr.
fn format_result(output: &crate::ports::CommandOutput) -> String {
    let mut parts = Vec::new();

    if output.timed_out {
        parts.push("[command timed out]".to_owned());
    }

    parts.push(format!("Exit code: {}", output.exit_code));

    if !output.stdout.is_empty() {
        parts.push("--- stdout ---".to_owned());
        parts.push(truncate_output(&output.stdout));
    }

    if !output.stderr.is_empty() {
        parts.push("--- stderr ---".to_owned());
        parts.push(truncate_output(&output.stderr));
    }

    // Explicit "no output" so the LLM doesn't think it missed something.
    if output.stdout.is_empty() && output.stderr.is_empty() && !output.timed_out {
        parts.push("(no output)".to_owned());
    }

    parts.join("\n")
}

/// Cap output at MAX_OUTPUT_LINES lines and MAX_OUTPUT_BYTES bytes, appending
/// a `[truncated]` marker if either limit is hit.
fn truncate_output(s: &str) -> String {
    // Use .lines() rather than .split('\n') so a trailing newline doesn't
    // inflate the count by one (split produces an extra empty element).
    let line_count = s.lines().count();
    if line_count <= MAX_OUTPUT_LINES && s.len() <= MAX_OUTPUT_BYTES {
        return s.trim_end().to_owned();
    }

    let mut result = String::new();
    for (count, line) in s.lines().enumerate() {
        if count >= MAX_OUTPUT_LINES || result.len() >= MAX_OUTPUT_BYTES {
            break;
        }
        // If appending this line would exceed the byte limit, include only
        // the portion that fits. Without this, a single line longer than
        // MAX_OUTPUT_BYTES would be included in full.
        let remaining = MAX_OUTPUT_BYTES.saturating_sub(result.len());
        if line.len() > remaining {
            // Truncate at a char boundary to avoid splitting a multi-byte char.
            let truncated = &line[..line.floor_char_boundary(remaining)];
            result.push_str(truncated);
            result.push('\n');
            break;
        }
        result.push_str(line);
        result.push('\n');
    }

    result.truncate(result.trim_end().len());
    result.push_str("\n[truncated]");
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::fake::FakeShell;
    use crate::ports::CommandOutput;

    fn tool(shell: Arc<FakeShell>) -> BashTool<FakeShell> {
        BashTool::new(shell)
    }

    // -- Successful command --

    #[tokio::test]
    async fn successful_command_shows_exit_code_and_output() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: "hello world\n".into(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t
            .execute(r#"{"command":"echo hello world"}"#)
            .await
            .unwrap();
        assert!(result.contains("Exit code: 0"), "got: {result}");
        assert!(result.contains("--- stdout ---"), "got: {result}");
        assert!(result.contains("hello world"), "got: {result}");
    }

    // -- Stderr only --

    #[tokio::test]
    async fn stderr_only_shows_stderr_section() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: "error msg\n".into(),
            exit_code: 1,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"bad cmd"}"#).await.unwrap();
        assert!(result.contains("--- stderr ---"), "got: {result}");
        assert!(result.contains("error msg"), "got: {result}");
        assert!(!result.contains("--- stdout ---"), "got: {result}");
    }

    // -- No output --

    #[tokio::test]
    async fn no_output_shows_marker() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"true"}"#).await.unwrap();
        assert!(result.contains("(no output)"), "got: {result}");
    }

    // -- Timeout --

    #[tokio::test]
    async fn timeout_shows_marker() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: -1,
            timed_out: true,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"sleep 999"}"#).await.unwrap();
        assert!(result.contains("[command timed out]"), "got: {result}");
        assert!(!result.contains("(no output)"), "got: {result}");
    }

    // -- Non-zero exit code --

    #[tokio::test]
    async fn nonzero_exit_code_reported() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 42,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"exit 42"}"#).await.unwrap();
        assert!(result.contains("Exit code: 42"), "got: {result}");
    }

    // -- Custom timeout_ms forwarded --

    #[tokio::test]
    async fn custom_timeout_forwarded_to_shell() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell.clone());
        t.execute(r#"{"command":"x","timeout_ms":5000}"#)
            .await
            .unwrap();

        let calls = shell.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "x");
        assert_eq!(calls[0].1, std::time::Duration::from_millis(5000));
    }

    // -- Default timeout --

    #[tokio::test]
    async fn default_timeout_is_120_seconds() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell.clone());
        t.execute(r#"{"command":"x"}"#).await.unwrap();

        let calls = shell.calls();
        assert_eq!(calls[0].1, std::time::Duration::from_millis(120_000));
    }

    // -- Truncation: line limit --

    #[tokio::test]
    async fn output_truncated_at_line_limit() {
        let big_stdout: String = (0..2500).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: big_stdout,
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(result.contains("[truncated]"), "got: {result}");
        // Should contain early lines but not the last ones.
        assert!(result.contains("line 0"), "got: {result}");
        assert!(!result.contains("line 2499"), "got: {result}");
    }

    // -- Truncation: byte limit --

    #[tokio::test]
    async fn output_truncated_at_byte_limit() {
        // A single long line exceeding 100KB.
        let big_stdout = "x".repeat(150 * 1024);
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: big_stdout,
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(result.contains("[truncated]"), "got: {result}");
        // The result must actually be bounded — a single huge line must not
        // pass through untruncated.
        assert!(
            result.len() < MAX_OUTPUT_BYTES + 1024,
            "result should be bounded near MAX_OUTPUT_BYTES, got {} bytes",
            result.len()
        );
    }

    // -- Truncation: exactly at line limit (no truncation) --

    #[tokio::test]
    async fn exactly_at_line_limit_is_not_truncated() {
        let stdout: String = (0..2000).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout,
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(!result.contains("[truncated]"), "got: {result}");
        assert!(result.contains("line 1999"), "got: {result}");
    }

    // -- Truncation: one over line limit --

    #[tokio::test]
    async fn one_over_line_limit_is_truncated() {
        let stdout: String = (0..2001).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout,
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(result.contains("[truncated]"), "got: {result}");
        assert!(!result.contains("line 2000"), "got: {result}");
    }

    // -- Shell error propagates --

    #[tokio::test]
    async fn shell_error_propagates() {
        let shell = Arc::new(FakeShell::new());
        shell.push_err("spawn failed");
        let t = tool(shell);
        let err = t.execute(r#"{"command":"echo hi"}"#).await.unwrap_err();
        // The error chain includes both the with_context wrapper and the
        // original error; format with {:#} to see the full chain.
        let msg = format!("{err:#}");
        assert!(msg.contains("spawn failed"), "got: {msg}");
    }

    // -- Both stdout and stderr present --

    #[tokio::test]
    async fn both_stdout_and_stderr_shows_both_sections() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: "out line\n".into(),
            stderr: "err line\n".into(),
            exit_code: 0,
            timed_out: false,
        });
        let t = tool(shell);
        let result = t.execute(r#"{"command":"both"}"#).await.unwrap();
        assert!(result.contains("--- stdout ---"), "got: {result}");
        assert!(result.contains("out line"), "got: {result}");
        assert!(result.contains("--- stderr ---"), "got: {result}");
        assert!(result.contains("err line"), "got: {result}");
        assert!(!result.contains("(no output)"), "got: {result}");
    }

    // -- Missing command arg --

    #[tokio::test]
    async fn missing_command_is_error() {
        let shell = Arc::new(FakeShell::new());
        let t = tool(shell);
        assert!(t.execute(r#"{}"#).await.is_err());
    }

    // -- Empty command string --

    #[tokio::test]
    async fn empty_command_is_error() {
        let shell = Arc::new(FakeShell::new());
        let t = tool(shell);
        assert!(t.execute(r#"{"command":""}"#).await.is_err());
    }

    // -- Invalid JSON --

    #[tokio::test]
    async fn invalid_json_is_error() {
        let shell = Arc::new(FakeShell::new());
        let t = tool(shell);
        assert!(t.execute("not json").await.is_err());
    }
}
