use anyhow::Result;

pub struct EnvSecretStore;

impl app::SecretStore for EnvSecretStore {
    fn get(&self, key: &str) -> Result<Option<String>> {
        match std::env::var(key) {
            Ok(val) => Ok(Some(val)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
