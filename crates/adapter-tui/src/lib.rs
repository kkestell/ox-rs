use anyhow::Result;

pub struct TuiApp {
    // TODO: hold use-case references
}

impl Default for TuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiApp {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self) -> Result<()> {
        // TODO: implement TUI event loop
        todo!()
    }
}
