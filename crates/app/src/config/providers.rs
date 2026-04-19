use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

pub const SHIPPED_PROVIDERS_JSON: &str = include_str!("providers.default.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvidersConfig {
    pub providers: Vec<Provider>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    pub base_url: Option<String>,
    pub models: Vec<Model>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    OpenRouter,
    Ollama,
    OpenAi,
    Google,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub context_in: u32,
}

impl ProvidersConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read_to_string(path)
            .with_context(|| format!("reading providers config {}", path.display()))?;
        let config: Self = serde_json::from_str(&bytes)
            .with_context(|| format!("parsing providers config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn shipped_default() -> Result<Self> {
        let config: Self = serde_json::from_str(SHIPPED_PROVIDERS_JSON)
            .context("parsing shipped providers config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn write_shipped_default_if_missing(path: &Path) -> Result<bool> {
        if path.exists() {
            return Ok(false);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config directory {}", parent.display()))?;
        }
        std::fs::write(path, SHIPPED_PROVIDERS_JSON)
            .with_context(|| format!("writing shipped providers config {}", path.display()))?;
        Ok(true)
    }

    pub fn validate(&self) -> Result<()> {
        if self.providers.is_empty() {
            bail!("providers config must contain at least one provider");
        }

        let mut provider_ids = HashSet::new();
        let mut model_ids = HashSet::new();
        for provider in &self.providers {
            if provider.id.trim().is_empty() {
                bail!("provider id must not be empty");
            }
            if !provider_ids.insert(provider.id.clone()) {
                bail!("duplicate provider id {:?}", provider.id);
            }
            if provider.name.trim().is_empty() {
                bail!("provider {:?} name must not be empty", provider.id);
            }
            if provider.models.is_empty() {
                bail!("provider {:?} must contain at least one model", provider.id);
            }
            for model in &provider.models {
                if model.id.trim().is_empty() {
                    bail!("model id must not be empty for provider {:?}", provider.id);
                }
                if !model_ids.insert(model.id.clone()) {
                    bail!("duplicate model id {:?}", model.id);
                }
                if model.name.trim().is_empty() {
                    bail!("model {:?} name must not be empty", model.id);
                }
                if model.context_in == 0 {
                    bail!("model {:?} context_in must be greater than zero", model.id);
                }
            }
        }
        Ok(())
    }

    pub fn model(&self, model_id: &str) -> Option<(&Provider, &Model)> {
        self.providers.iter().find_map(|provider| {
            provider
                .models
                .iter()
                .find(|model| model.id == model_id)
                .map(|model| (provider, model))
        })
    }

    pub fn ensure_model_exists(&self, model_id: &str) -> Result<()> {
        self.model(model_id)
            .map(|_| ())
            .ok_or_else(|| anyhow!("model {:?} is not present in providers config", model_id))
    }

    pub fn is_wired_model(&self, model_id: &str) -> bool {
        self.model(model_id)
            .is_some_and(|(provider, _)| provider.provider_type == ProviderType::OpenRouter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ox-providers-test-{name}-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn shipped_default_parses_and_validates() {
        let config = ProvidersConfig::shipped_default().unwrap();
        assert!(config.model("deepseek/deepseek-v3.2").is_some());
    }

    #[test]
    fn load_rejects_malformed_json() {
        let path = tmp_file("malformed");
        std::fs::write(&path, "{ nope").unwrap();
        let err = ProvidersConfig::load(&path).unwrap_err();
        assert!(format!("{err:#}").contains("parsing providers config"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_unknown_provider_type() {
        let path = tmp_file("unknown-type");
        std::fs::write(
            &path,
            r#"{"providers":[{"id":"x","name":"X","type":"mystery","base_url":null,"models":[]}]}"#,
        )
        .unwrap();
        let err = ProvidersConfig::load(&path).unwrap_err();
        assert!(format!("{err:#}").contains("unknown variant"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_shipped_default_only_when_missing() {
        let path = tmp_file("write-default");
        assert!(ProvidersConfig::write_shipped_default_if_missing(&path).unwrap());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            SHIPPED_PROVIDERS_JSON
        );
        assert!(!ProvidersConfig::write_shipped_default_if_missing(&path).unwrap());
        let _ = std::fs::remove_file(path);
    }
}
