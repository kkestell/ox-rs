use crate::ContentBlock;

use super::Usage;

/// Non-consuming, borrowed view of a `StreamAccumulator`'s current state.
///
/// Role is always `Role::Assistant`. The content slice borrows the
/// accumulator's internal cache and is only rebuilt when new events arrive.
/// This avoids per-frame cloning during live streaming in the GUI.
///
/// `usage` mirrors what the accumulator has seen so far. Before a
/// `Finished` event arrives it is the `Usage::default()` zero-struct;
/// after `Finished` it reflects the provider-reported counts. Keeping
/// it as a value (not `Option`) matches the accumulator's internal
/// representation and spares every live-render caller an unwrap.
pub struct Snapshot<'a> {
    pub content: &'a [ContentBlock],
    pub usage: Usage,
}
