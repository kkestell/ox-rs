use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::ProvidersConfig;

pub const SHIPPED_SETTINGS_JSON: &str = include_str!("settings.default.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    pub default_model: String,
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read_to_string(path)
            .with_context(|| format!("reading settings config {}", path.display()))?;
        serde_json::from_str(&bytes)
            .with_context(|| format!("parsing settings config {}", path.display()))
    }

    pub fn shipped_default() -> Result<Self> {
        serde_json::from_str(SHIPPED_SETTINGS_JSON).context("parsing shipped settings config")
    }

    pub fn write_shipped_default_if_missing(path: &Path) -> Result<bool> {
        if path.exists() {
            return Ok(false);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config directory {}", parent.display()))?;
        }
        std::fs::write(path, SHIPPED_SETTINGS_JSON)
            .with_context(|| format!("writing shipped settings config {}", path.display()))?;
        Ok(true)
    }

    pub fn validate(&self, providers: &ProvidersConfig) -> Result<()> {
        providers.ensure_model_exists(&self.default_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ox-settings-test-{name}-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn shipped_default_parses() {
        let settings = Settings::shipped_default().unwrap();
        assert_eq!(settings.default_model, "deepseek/deepseek-v3.2");
    }

    #[test]
    fn validate_rejects_unknown_default_model() {
        let providers = ProvidersConfig::shipped_default().unwrap();
        let settings = Settings {
            default_model: "missing/model".into(),
        };
        let err = settings.validate(&providers).unwrap_err();
        assert!(format!("{err:#}").contains("missing/model"));
    }

    #[test]
    fn write_shipped_default_only_when_missing() {
        let path = tmp_file("write-default");
        assert!(Settings::write_shipped_default_if_missing(&path).unwrap());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            SHIPPED_SETTINGS_JSON
        );
        assert!(!Settings::write_shipped_default_if_missing(&path).unwrap());
        let _ = std::fs::remove_file(path);
    }
}
