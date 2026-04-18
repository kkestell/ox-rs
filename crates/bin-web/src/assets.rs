//! Static assets embedded at compile time.
//!
//! Three files ship in the binary — `index.html`, `app.js`, and
//! `styles.css`. Embedding via `include_str!` keeps `ox` a single
//! self-contained executable and avoids any questions about "where
//! do the assets live relative to the binary?" on a user's machine.

pub const INDEX_HTML: &str = include_str!("../assets/index.html");
pub const APP_JS: &str = include_str!("../assets/app.js");
pub const STYLES_CSS: &str = include_str!("../assets/styles.css");
