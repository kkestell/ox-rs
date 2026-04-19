mod ollama;
mod openrouter;
mod providers_catalog;

pub use ollama::OllamaProvider;
pub use openrouter::{OpenRouterProvider, OpenRouterSlugGenerator};
pub use providers_catalog::ProvidersCatalog;
