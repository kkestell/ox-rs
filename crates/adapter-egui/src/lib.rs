use anyhow::Result;
use eframe::egui;

pub struct OxApp {
    // TODO: hold use-case references (SessionRunner, etc.)
}

impl Default for OxApp {
    fn default() -> Self {
        Self::new()
    }
}

impl OxApp {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(self) -> Result<()> {
        let options = eframe::NativeOptions::default();
        eframe::run_native("Ox", options, Box::new(|_cc| Ok(Box::new(self))))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }
}

impl eframe::App for OxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Ox");
            // TODO: wire up session UI
        });
    }
}
