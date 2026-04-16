//! Hashline tagging for file contents.
//!
//! A "hashline" is a short tag prepended to each line of a file so the LLM can
//! reference a specific line by a stable identifier instead of having to
//! reproduce the line verbatim. The format is:
//!
//! ```text
//! {N}:{hash}| <original line>
//! ```
//!
//! where `N` is the 1-indexed line number and `hash` is a 3-character base-36
//! digest of the line's content (excluding any trailing newline). Hashlines
//! exist purely for round-tripping between `read_file` and `edit_file`: the
//! LLM reads a file with hashlines, copies the tag of the line it wants to
//! target, and hands that tag back as an anchor in an edit. Anchors hard-fail
//! on mismatch, which catches stale reads before they corrupt the file.
//!
//! 3 base-36 characters give ~46k slots; within a single file, line numbers
//! disambiguate collisions. That's enough: the hash exists to detect *drift*,
//! not to globally identify lines.

use anyhow::{bail, Context, Result};

/// Encodes a `u32` as a left-padded 3-character base-36 string.
///
/// Exactly 3 chars because the callers want a fixed-width tag. The input is
/// truncated to 3 digits of base 36 (15 bits of the input), which is fine —
/// the hash is already not collision-resistant; it's a drift detector.
fn base36_3(mut n: u32) -> String {
    const CHARSET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = [b'0'; 3];
    for slot in out.iter_mut().rev() {
        *slot = CHARSET[(n % 36) as usize];
        n /= 36;
    }
    // Safety: CHARSET is ASCII, so out is valid UTF-8.
    String::from_utf8(out.to_vec()).expect("CHARSET is ASCII")
}

/// 3-character base-36 digest of `line`'s bytes.
///
/// Uses `blake3` for the underlying hash — cryptographically strong would be
/// overkill here, but blake3 is already a small, fast dep we're pulling in,
/// and using it means we get a uniform-looking tag without rolling our own
/// mixer. We take the first 4 bytes of the digest as a big-endian `u32` and
/// fold them into 3 base-36 digits.
pub fn hash_line(line: &str) -> String {
    let digest = blake3::hash(line.as_bytes());
    let bytes = digest.as_bytes();
    let n = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    base36_3(n)
}

/// Split `text` into lines the same way `edit_file` later splits them, so
/// `render_with_hashlines` and edit-anchor verification agree on line count
/// and content.
///
/// The rule: every `\n` ends a line. A trailing `\n` means the final line is
/// empty but counted. This matches `str::split_terminator('\n')` when there
/// IS a trailing newline, and `str::split('\n')` when there isn't — we unify
/// by always using `split('\n')` and stripping the implicit empty final line
/// when a trailing newline is present.
///
/// `\r` inside a line is treated as part of the line content: CRLF files will
/// have different hashes than their LF counterparts, but round-trips stay
/// consistent within a single file.
pub fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut parts: Vec<&str> = text.split('\n').collect();
    // split('\n') on "a\nb\n" yields ["a", "b", ""]. Drop that trailing ""
    // so "N lines" means N content lines. When we render, we'll re-attach
    // the trailing newline if the original text had one.
    if text.ends_with('\n') {
        parts.pop();
    }
    parts
}

/// Render `text` with `{N}:{hash}| ` prefixes, where `N` starts at
/// `start_line_1_indexed` and increments per line.
///
/// Preserves the trailing-newline shape of the input: if `text` ended with a
/// newline, so does the output. This makes the output safe to feed back into
/// `split_lines` and get the same line count.
///
/// Empty input returns empty output — no hashlines, no trailing newline.
pub fn render_with_hashlines(text: &str, start_line_1_indexed: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    let trailing_newline = text.ends_with('\n');
    let lines = split_lines(text);
    let mut out = String::with_capacity(text.len() + lines.len() * 8);
    for (i, line) in lines.iter().enumerate() {
        let n = start_line_1_indexed + i;
        out.push_str(&format!("{n:04}:{}| {line}", hash_line(line)));
        // Every rendered line gets its own newline except optionally the last
        // one, which matches the input's trailing shape.
        if i + 1 < lines.len() || trailing_newline {
            out.push('\n');
        }
    }
    out
}

/// Parsed anchor referring to a position in a file.
///
/// `Start` and `End` are sentinels used by `insert_after` for "prepend" and
/// "append" operations. `Line { line, hash }` references a concrete line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anchor {
    Start,
    End,
    Line { line: usize, hash: String },
}

/// Parse an anchor string. Accepts:
///
/// - `"start"` and `"end"` (case-insensitive) → sentinels.
/// - `"{N}:{hash}"` where `N` is a positive integer and `hash` is non-empty
///   — the trailing `|` and content are not included here; anchors are just
///   the `{N}:{hash}` prefix the LLM extracts from a hashline.
///
/// Malformed anchors fail with an error the LLM can read and correct.
pub fn parse_anchor(s: &str) -> Result<Anchor> {
    let trimmed = s.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "start" {
        return Ok(Anchor::Start);
    }
    if lower == "end" {
        return Ok(Anchor::End);
    }
    let (line_str, hash) = trimmed.split_once(':').with_context(|| {
        format!("anchor must be 'start', 'end', or '{{line}}:{{hash}}', got {trimmed:?}")
    })?;
    let line: usize = line_str.parse().with_context(|| {
        format!("anchor line number must be a positive integer, got {line_str:?}")
    })?;
    if line == 0 {
        bail!("anchor line numbers are 1-indexed; got 0");
    }
    if hash.is_empty() {
        bail!("anchor hash must not be empty");
    }
    Ok(Anchor::Line {
        line,
        hash: hash.to_owned(),
    })
}

/// Verify that an anchor matches the current state of `lines` and return the
/// 0-indexed position it refers to.
///
/// For `Start`, returns `0`. For `End`, returns `lines.len()` — callers that
/// need "insert after end" should treat this as "at index lines.len()". For
/// `Line { line, hash }`, returns `line - 1` if the hash matches the line's
/// current content, and an error otherwise.
///
/// Errors name the bad anchor and include the current hash so the LLM can
/// self-correct without a second `read_file`.
pub fn verify_anchor(lines: &[&str], anchor: &Anchor) -> Result<usize> {
    match anchor {
        Anchor::Start => Ok(0),
        Anchor::End => Ok(lines.len()),
        Anchor::Line { line, hash } => {
            if *line > lines.len() {
                bail!(
                    "anchor line {line} is out of range (file has {} lines)",
                    lines.len()
                );
            }
            let actual_line = lines[line - 1];
            let actual_hash = hash_line(actual_line);
            if &actual_hash != hash {
                bail!("hash mismatch at line {line}: anchor says {hash}, current is {actual_hash}");
            }
            Ok(line - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- hash_line --

    #[test]
    fn hash_is_deterministic() {
        let a = hash_line("hello world");
        let b = hash_line("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_is_three_base36_chars() {
        for sample in ["", "a", "hello", "x".repeat(4096).as_str(), "\tabc\n"] {
            let h = hash_line(sample);
            assert_eq!(h.len(), 3, "hash for {sample:?} was {h:?}");
            assert!(
                h.chars()
                    .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()),
                "hash {h:?} contains non-base36 chars"
            );
        }
    }

    #[test]
    fn different_lines_usually_differ() {
        // Not a guarantee (3 base36 chars can collide), but in practice on
        // these trivially-different inputs they shouldn't.
        assert_ne!(hash_line("alpha"), hash_line("beta"));
    }

    // -- render_with_hashlines --

    #[test]
    fn render_empty_returns_empty() {
        assert_eq!(render_with_hashlines("", 1), "");
    }

    #[test]
    fn render_preserves_trailing_newline() {
        let input = "alpha\nbeta\n";
        let out = render_with_hashlines(input, 1);
        assert!(out.ends_with('\n'), "output {out:?} lost trailing newline");
    }

    #[test]
    fn render_no_trailing_newline_is_preserved() {
        let input = "alpha\nbeta";
        let out = render_with_hashlines(input, 1);
        assert!(
            !out.ends_with('\n'),
            "output {out:?} grew a trailing newline"
        );
    }

    #[test]
    fn render_uses_provided_start_line() {
        let out = render_with_hashlines("x\ny\n", 10);
        // First line is numbered 10, second 11.
        assert!(out.starts_with("0010:"), "got {out:?}");
        let second_line = out.lines().nth(1).unwrap();
        assert!(second_line.starts_with("0011:"), "got {second_line:?}");
    }

    #[test]
    fn render_single_line_no_newline() {
        let out = render_with_hashlines("solo", 1);
        assert!(out.starts_with("0001:"));
        assert!(out.contains("| solo"));
        assert!(!out.ends_with('\n'));
    }

    #[test]
    fn render_line_count_matches_input_line_count() {
        let input = "a\nb\nc\n";
        let out = render_with_hashlines(input, 1);
        // 3 content lines in, 3 rendered lines out.
        assert_eq!(out.lines().count(), 3);
    }

    // -- parse_anchor --

    #[test]
    fn parse_start_and_end() {
        assert_eq!(parse_anchor("start").unwrap(), Anchor::Start);
        assert_eq!(parse_anchor("END").unwrap(), Anchor::End);
        assert_eq!(parse_anchor("  Start  ").unwrap(), Anchor::Start);
    }

    #[test]
    fn parse_numeric_anchor() {
        let a = parse_anchor("42:a3z").unwrap();
        assert_eq!(
            a,
            Anchor::Line {
                line: 42,
                hash: "a3z".into(),
            }
        );
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(parse_anchor("").is_err());
        assert!(parse_anchor("notanumber:hash").is_err());
        assert!(parse_anchor("10").is_err());
        assert!(parse_anchor("10:").is_err());
        assert!(parse_anchor("0:abc").is_err()); // 1-indexed
    }

    // -- verify_anchor --

    #[test]
    fn verify_start_returns_zero() {
        let lines: Vec<&str> = vec!["a", "b"];
        assert_eq!(verify_anchor(&lines, &Anchor::Start).unwrap(), 0);
    }

    #[test]
    fn verify_end_returns_len() {
        let lines: Vec<&str> = vec!["a", "b", "c"];
        assert_eq!(verify_anchor(&lines, &Anchor::End).unwrap(), 3);
    }

    #[test]
    fn verify_line_returns_zero_indexed_position() {
        let lines: Vec<&str> = vec!["alpha", "beta", "gamma"];
        let hash = hash_line("beta");
        let anchor = Anchor::Line { line: 2, hash };
        assert_eq!(verify_anchor(&lines, &anchor).unwrap(), 1);
    }

    #[test]
    fn verify_hash_mismatch_reports_both_hashes() {
        let lines: Vec<&str> = vec!["alpha", "beta"];
        let anchor = Anchor::Line {
            line: 1,
            hash: "xxx".into(),
        };
        let err = verify_anchor(&lines, &anchor).unwrap_err().to_string();
        // Mentions the bad anchor and the current state so the LLM can fix it.
        assert!(err.contains("line 1"), "got {err}");
        assert!(err.contains("xxx"), "got {err}");
        assert!(err.contains(&hash_line("alpha")), "got {err}");
    }

    #[test]
    fn verify_line_out_of_range_is_an_error() {
        let lines: Vec<&str> = vec!["alpha"];
        let anchor = Anchor::Line {
            line: 5,
            hash: "xxx".into(),
        };
        assert!(verify_anchor(&lines, &anchor).is_err());
    }

    #[test]
    fn duplicate_lines_disambiguated_by_line_number() {
        // Two lines with identical content → identical hashes. The line
        // number is what disambiguates them at verify time.
        let lines: Vec<&str> = vec!["same", "other", "same"];
        let h = hash_line("same");
        let first = Anchor::Line {
            line: 1,
            hash: h.clone(),
        };
        let third = Anchor::Line { line: 3, hash: h };
        assert_eq!(verify_anchor(&lines, &first).unwrap(), 0);
        assert_eq!(verify_anchor(&lines, &third).unwrap(), 2);
    }
}
