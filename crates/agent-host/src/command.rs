//! Slash-command classification for user input.
//!
//! `classify_input` decides whether a line of user input is a built-in
//! command (`/new`, `/quit`, `/close`) that the UI should handle directly,
//! or a regular message that should be forwarded to the agent. Extracted
//! from the egui render loop so it can be unit-tested — and reused — without
//! any UI context.
//!
//! `SplitAction` variants are intentionally unit-only: callers pair the
//! returned action with the target split's `SplitId` themselves. That keeps
//! this module free of any notion of "current split index" and lets both the
//! egui and Tauri frontends reuse it unchanged.

/// Classify user input as a built-in slash command or a regular message.
///
/// Only `/new`, `/quit`, and `/close` are intercepted. Any other slash
/// prefix (e.g. `/help`) is treated as a regular message and forwarded to
/// the agent — matching the behavior users expect from chat interfaces that
/// pass unknown slashes through.
pub fn classify_input(text: &str) -> SplitAction {
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("/new") {
        SplitAction::New
    } else if trimmed.eq_ignore_ascii_case("/quit") {
        SplitAction::QuitApp
    } else if trimmed.eq_ignore_ascii_case("/close") {
        SplitAction::CloseSplit
    } else {
        SplitAction::Send
    }
}

/// Actions the UI can take in response to a line of user input.
///
/// Variants carry no payload — the caller already knows which split the
/// input came from and pairs the action with the appropriate `SplitId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitAction {
    /// Send the input as a user message to the agent on the active split.
    Send,
    /// Spawn a new agent and append a split to the workspace.
    New,
    /// Quit the whole app.
    QuitApp,
    /// Close the active split. Drops its agent (`kill_on_drop`).
    CloseSplit,
    /// Cancel the in-progress turn on the active split.
    Cancel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_input_recognizes_new() {
        assert_eq!(classify_input("/new"), SplitAction::New);
        assert_eq!(classify_input("  /new  "), SplitAction::New);
        assert_eq!(classify_input("/NEW"), SplitAction::New);
    }

    #[test]
    fn classify_input_recognizes_quit() {
        assert_eq!(classify_input("/quit"), SplitAction::QuitApp);
        assert_eq!(classify_input("  /QUIT  "), SplitAction::QuitApp);
    }

    #[test]
    fn classify_input_recognizes_close() {
        assert_eq!(classify_input("/close"), SplitAction::CloseSplit);
        assert_eq!(classify_input("/CLOSE"), SplitAction::CloseSplit);
    }

    #[test]
    fn classify_input_passes_unknown_slash_commands_as_messages() {
        // `/help`, `/foo`, etc. are NOT intercepted — they should be sent
        // to the agent as regular messages.
        assert_eq!(classify_input("/help"), SplitAction::Send);
        assert_eq!(classify_input("/foo bar"), SplitAction::Send);
    }

    #[test]
    fn classify_input_passes_plain_text_as_messages() {
        assert_eq!(classify_input("hello world"), SplitAction::Send);
        assert_eq!(classify_input(""), SplitAction::Send);
        assert_eq!(classify_input("   "), SplitAction::Send);
    }

    #[test]
    fn classify_input_does_not_match_partial_slash_commands() {
        // Slash-prefixed text that isn't exactly one of the known commands
        // must not be intercepted — even if it looks close.
        assert_eq!(classify_input("/news"), SplitAction::Send);
        assert_eq!(classify_input("/new tab"), SplitAction::Send);
        assert_eq!(classify_input("//new"), SplitAction::Send);
    }
}
