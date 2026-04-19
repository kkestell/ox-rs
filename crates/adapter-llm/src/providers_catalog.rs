use std::sync::Arc;

use app::ModelCatalog;
use app::config::ProvidersConfig;

pub struct ProvidersCatalog {
    providers: Arc<ProvidersConfig>,
}

impl ProvidersCatalog {
    pub fn new(providers: Arc<ProvidersConfig>) -> Self {
        Self { providers }
    }
}

impl ModelCatalog for ProvidersCatalog {
    fn context_window(&self, model: &str) -> Option<u32> {
        self.providers
            .model(model)
            .map(|(_, model)| model.context_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_window_comes_from_providers_config() {
        let config = Arc::new(ProvidersConfig::shipped_default().unwrap());
        let catalog = ProvidersCatalog::new(config);
        assert_eq!(
            catalog.context_window("deepseek/deepseek-v3.2"),
            Some(163840)
        );
        assert_eq!(catalog.context_window("missing/model"), None);
    }
}
