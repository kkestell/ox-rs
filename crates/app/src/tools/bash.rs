//! `bash` tool.
//!
//! Executes a shell command in the workspace directory via `/bin/bash -c`.
//! Captures stdout and stderr, enforces a configurable timeout, and returns
//! a formatted result with labeled sections. Large output is spilled to a
//! temp file under `.ox/tmp/` with a preview shown inline; the agent can
//! then use `read_file` or `grep` to explore the full result.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::spill::{self, PREVIEW_LINES};
use super::{Tool, require_non_empty};
use crate::ports::{FileSystem, Shell};
use crate::stream::ToolDef;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// Byte cap passed to Shell::run to prevent unbounded memory consumption.
const SHELL_MAX_BYTES: usize = 10 * 1024 * 1024; // 10 MB

pub struct BashTool<S, F> {
    shell: Arc<S>,
    fs: Arc<F>,
    workspace_root: PathBuf,
}

impl<S, F> BashTool<S, F> {
    pub fn new(shell: Arc<S>, fs: Arc<F>, workspace_root: PathBuf) -> Self {
        Self {
            shell,
            fs,
            workspace_root,
        }
    }
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    timeout_ms: Option<u64>,
}

impl<S: Shell + 'static, F: FileSystem + 'static> Tool for BashTool<S, F> {
    fn def(&self) -> ToolDef {
        ToolDef {
            name: "bash".into(),
            description: "Execute a shell command in the workspace directory. \
                Long output is automatically truncated and spilled to a file — \
                never pipe through head, tail, or grep to manage output length. \
                Do not use this for tasks that have a dedicated tool: use \
                `read_file` instead of cat/head/tail, `glob` instead of find/ls, \
                `grep` instead of grep/rg. Reserve bash for build commands, test \
                runners, git operations, and other tasks with no dedicated tool."
                .into(),
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
                .run(&parsed.command, timeout, SHELL_MAX_BYTES)
                .await
                .with_context(|| format!("bash: failed to run command: {}", parsed.command))?;

            self.format_result(&output).await
        })
    }
}

impl<S, F: FileSystem> BashTool<S, F> {
    /// Format the command output into labeled sections. Large streams are
    /// spilled to temp files with a preview shown inline.
    async fn format_result(&self, output: &crate::ports::CommandOutput) -> Result<String> {
        let mut parts = Vec::new();

        if output.timed_out {
            parts.push("[command timed out]".to_owned());
        }

        parts.push(format!("Exit code: {}", output.exit_code));

        if !output.stdout.is_empty() {
            let section = self
                .format_stream("stdout", &output.stdout, output.truncated)
                .await?;
            parts.push(section);
        }

        if !output.stderr.is_empty() {
            let section = self
                .format_stream("stderr", &output.stderr, output.truncated)
                .await?;
            parts.push(section);
        }

        // Explicit "no output" so the LLM doesn't think it missed something.
        if output.stdout.is_empty() && output.stderr.is_empty() && !output.timed_out {
            parts.push("(no output)".to_owned());
        }

        Ok(parts.join("\n"))
    }

    /// Format a single output stream (stdout or stderr). If the content
    /// exceeds the inline threshold, spill to a temp file and show a preview.
    async fn format_stream(
        &self,
        stream_name: &str,
        content: &str,
        was_truncated: bool,
    ) -> Result<String> {
        let trimmed = content.trim_end();

        if spill::needs_spill(trimmed) {
            let info = spill::spill(
                self.fs.as_ref(),
                &self.workspace_root,
                trimmed,
                &format!("bash-{stream_name}"),
            )
            .await?;

            let preview = spill::preview(trimmed, PREVIEW_LINES);
            let mut section = format!(
                "--- {stream_name} ({} lines, {} KB) ---\n{}\n[full output: {}]",
                info.total_lines,
                info.total_bytes / 1024,
                preview.trim_end(),
                info.display_path,
            );
            if was_truncated {
                section.push_str("\n[output was capped at the byte limit]");
            }
            Ok(section)
        } else {
            let mut section = format!("--- {stream_name} ---\n{trimmed}");
            if was_truncated {
                section.push_str("\n[output was capped at the byte limit]");
            }
            Ok(section)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::fake::{FakeFileSystem, FakeShell};
    use crate::ports::CommandOutput;

    fn tool(shell: Arc<FakeShell>, fs: Arc<FakeFileSystem>) -> BashTool<FakeShell, FakeFileSystem> {
        BashTool::new(shell, fs, PathBuf::from("/ws"))
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell.clone(), fs);
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell.clone(), fs);
        t.execute(r#"{"command":"x"}"#).await.unwrap();

        let calls = shell.calls();
        assert_eq!(calls[0].1, std::time::Duration::from_millis(120_000));
    }

    // -- Small output stays inline --

    #[tokio::test]
    async fn small_output_stays_inline() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: "hello\nworld\n".into(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs.clone());
        let result = t.execute(r#"{"command":"echo hello"}"#).await.unwrap();
        assert!(result.contains("--- stdout ---"), "got: {result}");
        assert!(result.contains("hello\nworld"), "got: {result}");
        // No spill file should be written.
        assert!(!result.contains(".ox/tmp/"), "got: {result}");
    }

    // -- Large stdout spills --

    #[tokio::test]
    async fn large_stdout_spills_to_file() {
        let big_stdout: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: big_stdout.clone(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs.clone());
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();

        // Should contain a preview (first 50 lines) and a file path.
        assert!(result.contains("line 0"), "got: {result}");
        assert!(result.contains("line 49"), "got: {result}");
        assert!(result.contains("[full output:"), "got: {result}");
        assert!(result.contains(".ox/tmp/bash-stdout-"), "got: {result}");
        // Metadata in the header.
        assert!(result.contains("500 lines"), "got: {result}");

        // The last lines should NOT be in the inline output.
        assert!(!result.contains("line 499"), "got: {result}");
    }

    // -- Large stderr spills independently --

    #[tokio::test]
    async fn large_stderr_spills_independently() {
        let big_stderr: String = (0..300).map(|i| format!("err {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: "small\n".into(),
            stderr: big_stderr,
            exit_code: 0,
            timed_out: false,
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs.clone());
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();

        // Stdout should be inline, stderr should spill.
        assert!(result.contains("--- stdout ---\nsmall"), "got: {result}");
        assert!(result.contains(".ox/tmp/bash-stderr-"), "got: {result}");
    }

    // -- Both streams spill --

    #[tokio::test]
    async fn both_streams_can_spill() {
        let big: String = (0..300).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: big.clone(),
            stderr: big,
            exit_code: 0,
            timed_out: false,
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs.clone());
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();

        assert!(result.contains(".ox/tmp/bash-stdout-"), "got: {result}");
        assert!(result.contains(".ox/tmp/bash-stderr-"), "got: {result}");
    }

    // -- Truncated output shows notice --

    #[tokio::test]
    async fn truncated_output_shows_cap_notice() {
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: "some output\n".into(),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
            truncated: true,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(
            result.contains("[output was capped at the byte limit]"),
            "got: {result}"
        );
    }

    // -- Exit code and timeout markers appear regardless of spilling --

    #[tokio::test]
    async fn exit_code_and_timeout_appear_with_spill() {
        let big_stdout: String = (0..300).map(|i| format!("line {i}\n")).collect();
        let shell = Arc::new(FakeShell::new());
        shell.push_output(CommandOutput {
            stdout: big_stdout,
            stderr: String::new(),
            exit_code: 1,
            timed_out: true,
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        let result = t.execute(r#"{"command":"gen"}"#).await.unwrap();
        assert!(result.contains("[command timed out]"), "got: {result}");
        assert!(result.contains("Exit code: 1"), "got: {result}");
    }

    // -- Shell error propagates --

    #[tokio::test]
    async fn shell_error_propagates() {
        let shell = Arc::new(FakeShell::new());
        shell.push_err("spawn failed");
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        let err = t.execute(r#"{"command":"echo hi"}"#).await.unwrap_err();
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
            truncated: false,
        });
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
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
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        assert!(t.execute(r#"{}"#).await.is_err());
    }

    // -- Empty command string --

    #[tokio::test]
    async fn empty_command_is_error() {
        let shell = Arc::new(FakeShell::new());
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        assert!(t.execute(r#"{"command":""}"#).await.is_err());
    }

    // -- Invalid JSON --

    #[tokio::test]
    async fn invalid_json_is_error() {
        let shell = Arc::new(FakeShell::new());
        let fs = Arc::new(FakeFileSystem::new());
        let t = tool(shell, fs);
        assert!(t.execute("not json").await.is_err());
    }
}
