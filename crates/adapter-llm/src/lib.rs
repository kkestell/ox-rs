mod ollama;
mod openrouter;

pub use ollama::OllamaProvider;
pub use openrouter::{CatalogError, OpenRouterCatalog, OpenRouterProvider, OpenRouterSlugGenerator};
