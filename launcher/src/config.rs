use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug)]
pub struct LauncherConfig {
    pub data_dir: PathBuf, // ~/.local/share/agentic-concierge (or CONCIERGE_DATA_DIR)
    pub venv_dir: PathBuf, // data_dir/venv
    pub uv_path: PathBuf,  // data_dir/uv
    pub version_file: PathBuf, // data_dir/installed_version
    pub extras_file: PathBuf, // data_dir/installed_extras
    /// The directory containing the installed launcher binary (e.g. `~/.local/bin`).
    /// Stored for Phase 14+ use; not yet read by any hot path.
    #[allow(dead_code)]
    pub bin_dir: PathBuf, // ~/.local/bin
    pub installed_bin: PathBuf, // bin_dir/concierge (self-path for atomic replace)
    pub skip_update: bool, // CONCIERGE_NO_UPDATE_CHECK=1
    pub package_name: String, // "agentic-concierge"
    pub pypi_extra: Option<String>, // CONCIERGE_EXTRA env var (e.g. "mcp,otel")
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("could not determine home directory")]
    NoHomeDir,
}

pub fn launcher_config() -> Result<LauncherConfig, ConfigError> {
    let data_dir = if let Ok(v) = std::env::var("CONCIERGE_DATA_DIR") {
        PathBuf::from(v)
    } else {
        dirs::data_local_dir()
            .ok_or(ConfigError::NoHomeDir)?
            .join("agentic-concierge")
    };

    let venv_dir = data_dir.join("venv");
    let uv_path = data_dir.join("uv");
    let version_file = data_dir.join("installed_version");
    let extras_file = data_dir.join("installed_extras");

    let bin_dir = dirs::executable_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("bin")))
        .ok_or(ConfigError::NoHomeDir)?;

    let installed_bin = bin_dir.join("concierge");

    let skip_update = std::env::var("CONCIERGE_NO_UPDATE_CHECK")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Default to "all" extras — install everything the hardware can support.
    // Users can override with CONCIERGE_EXTRA="" for bare-bones or a specific subset.
    let pypi_extra = match std::env::var("CONCIERGE_EXTRA") {
        Ok(v) if v.is_empty() => None,          // explicit "" = no extras
        Ok(v) => Some(v),                        // user-specified extras
        Err(_) => Some("all".to_string()),       // default = all
    };

    Ok(LauncherConfig {
        data_dir,
        venv_dir,
        uv_path,
        version_file,
        extras_file,
        bin_dir,
        installed_bin,
        skip_update,
        package_name: "agentic-concierge".to_string(),
        pypi_extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-var tests to prevent interference between parallel test threads.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn skip_update_true() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CONCIERGE_NO_UPDATE_CHECK", "1");
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-su1");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_NO_UPDATE_CHECK");
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert!(config.skip_update);
    }

    #[test]
    fn skip_update_false_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("CONCIERGE_NO_UPDATE_CHECK");
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-su0");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert!(!config.skip_update);
    }

    #[test]
    fn data_dir_under_home() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("CONCIERGE_DATA_DIR");
        std::env::remove_var("CONCIERGE_NO_UPDATE_CHECK");
        let config = launcher_config().unwrap();
        let local_share = dirs::data_local_dir().unwrap();
        assert!(config.data_dir.starts_with(&local_share));
        assert!(config.data_dir.ends_with("agentic-concierge"));
    }

    #[test]
    fn venv_dir_under_data_dir() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-venv");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert_eq!(config.venv_dir, config.data_dir.join("venv"));
    }

    #[test]
    fn env_override_respected() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-override-12345");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert_eq!(config.data_dir, PathBuf::from("/tmp/test-override-12345"));
    }

    #[test]
    fn pypi_extra_defaults_to_all() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("CONCIERGE_EXTRA");
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-extra-default");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert_eq!(config.pypi_extra, Some("all".to_string()));
    }

    #[test]
    fn pypi_extra_empty_means_none() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CONCIERGE_EXTRA", "");
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-extra-empty");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_EXTRA");
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert_eq!(config.pypi_extra, None);
    }

    #[test]
    fn pypi_extra_custom_value() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CONCIERGE_EXTRA", "mcp,otel");
        std::env::set_var("CONCIERGE_DATA_DIR", "/tmp/test-concierge-extra-custom");
        let config = launcher_config().unwrap();
        std::env::remove_var("CONCIERGE_EXTRA");
        std::env::remove_var("CONCIERGE_DATA_DIR");
        assert_eq!(config.pypi_extra, Some("mcp,otel".to_string()));
    }
}
