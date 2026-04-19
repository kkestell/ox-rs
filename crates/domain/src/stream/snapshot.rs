use crate::ContentBlock;

/// Non-consuming, borrowed view of a `StreamAccumulator`'s current state.
///
/// Role is always `Role::Assistant`. The content slice borrows the
/// accumulator's internal cache and is only rebuilt when new events arrive.
/// This avoids per-frame cloning during live streaming in the GUI.
pub struct Snapshot<'a> {
    pub content: &'a [ContentBlock],
    pub token_count: usize,
}
