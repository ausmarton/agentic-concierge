use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::config::LauncherConfig;

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("no suitable Python >= 3.10 found in PATH and uv installation failed")]
    NoPython,
    #[error("failed to create virtual environment: {0}")]
    VenvCreation(String),
    #[error("failed to install package (exit code {code}): {stderr}")]
    PackageInstall { code: i32, stderr: String },
    #[error("uv binary is not executable after download")]
    UvNotExecutable,
}

/// Ensure managed venv exists with agentic-concierge installed.
/// Returns path to the venv's concierge-py binary.
///
/// Fast path: if `venv_dir/bin/concierge-py` already exists, install any
/// new extras requested via `CONCIERGE_EXTRA` and return.
/// First-time path: detect system Python >= 3.10 or download uv, create venv, pip install.
pub fn ensure_environment(config: &LauncherConfig) -> anyhow::Result<PathBuf> {
    let concierge_bin = config.venv_dir.join("bin").join("concierge-py");
    if concierge_bin.exists() {
        ensure_extras(config)?;
        return Ok(concierge_bin);
    }

    // Migration: venv exists with old entry point name "concierge" → upgrade in place.
    let old_bin = config.venv_dir.join("bin").join("concierge");
    let pip = config.venv_dir.join("bin").join("pip");
    if old_bin.exists() && pip.exists() {
        eprintln!("[concierge] migrating venv entry point...");
        let package_spec = match &config.pypi_extra {
            Some(extra) => format!("{}[{}]", config.package_name, extra),
            None => config.package_name.clone(),
        };
        let output = std::process::Command::new(&pip)
            .args(["install", "--upgrade", &package_spec])
            .output();
        if let Ok(out) = output {
            if out.status.success() && concierge_bin.exists() {
                ensure_extras(config)?;
                return Ok(concierge_bin);
            }
        }
        // If migration failed, fall through to full setup.
    }

    // First-time setup
    std::fs::create_dir_all(&config.data_dir)?;

    let python = try_system_python();

    if python.is_none() {
        ensure_uv(config).map_err(|e| {
            eprintln!("[concierge] could not install uv: {}", e);
            SetupError::NoPython
        })?;
    }

    // Create venv
    match &python {
        Some(py_path) => {
            let status = std::process::Command::new(py_path)
                .args(["-m", "venv"])
                .arg(&config.venv_dir)
                .status()
                .map_err(|e| SetupError::VenvCreation(e.to_string()))?;
            if !status.success() {
                return Err(SetupError::VenvCreation("venv creation failed".to_string()).into());
            }
        }
        None => {
            // Use uv
            let status = std::process::Command::new(&config.uv_path)
                .args(["venv", "--python", "3.12"])
                .arg(&config.venv_dir)
                .status()
                .map_err(|e| SetupError::VenvCreation(e.to_string()))?;
            if !status.success() {
                return Err(SetupError::VenvCreation("uv venv creation failed".to_string()).into());
            }
        }
    }

    // pip install
    let pip = config.venv_dir.join("bin").join("pip");
    let package_spec = match &config.pypi_extra {
        Some(extra) => format!("{}[{}]", config.package_name, extra),
        None => config.package_name.clone(),
    };
    let output = std::process::Command::new(&pip)
        .args(["install", "--upgrade", &package_spec])
        .output()
        .map_err(|e| SetupError::PackageInstall {
            code: -1,
            stderr: e.to_string(),
        })?;
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(SetupError::PackageInstall { code, stderr }.into());
    }

    // Write version file
    std::fs::write(&config.version_file, env!("CARGO_PKG_VERSION"))?;

    // Write extras marker so ensure_extras can detect changes later.
    let extras_val = config.pypi_extra.as_deref().unwrap_or("");
    std::fs::write(&config.extras_file, extras_val)?;

    Ok(concierge_bin)
}

/// Install extras into an existing venv when `CONCIERGE_EXTRA` changes.
///
/// Compares the current `CONCIERGE_EXTRA` value against the marker file
/// `data_dir/installed_extras`. If they differ, runs `pip install` with the
/// new extras spec and updates the marker. Skipped when `CONCIERGE_EXTRA`
/// is unset (no extras requested).
fn ensure_extras(config: &LauncherConfig) -> anyhow::Result<()> {
    let requested = match &config.pypi_extra {
        Some(extra) if !extra.is_empty() => extra.as_str(),
        _ => return Ok(()), // no extras requested — nothing to do
    };

    // Read what was previously installed.
    let installed = std::fs::read_to_string(&config.extras_file).unwrap_or_default();
    if installed.trim() == requested {
        return Ok(()); // already up-to-date
    }

    eprintln!("[concierge] installing extras: {}", requested);
    let pip = config.venv_dir.join("bin").join("pip");
    let package_spec = format!("{}[{}]", config.package_name, requested);
    let output = std::process::Command::new(&pip)
        .args(["install", "--upgrade", &package_spec])
        .output()
        .map_err(|e| SetupError::PackageInstall {
            code: -1,
            stderr: e.to_string(),
        })?;
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(SetupError::PackageInstall { code, stderr }.into());
    }

    // Post-install: download Playwright browsers when browser extra is included.
    if requested.contains("browser") || requested == "all" {
        ensure_playwright_browsers(config);
    }

    // Update marker so we don't re-install next time.
    std::fs::write(&config.extras_file, requested)?;
    Ok(())
}

/// Download Playwright browser binaries if the browser extra is installed.
///
/// Runs `python -m playwright install chromium` inside the managed venv.
/// Failure is non-fatal — browser tools will simply be unavailable.
fn ensure_playwright_browsers(config: &LauncherConfig) {
    let python = config.venv_dir.join("bin").join("python");
    if !python.exists() {
        return;
    }
    eprintln!("[concierge] downloading Playwright browser binaries...");
    match std::process::Command::new(&python)
        .args(["-m", "playwright", "install", "chromium"])
        .output()
    {
        Ok(output) if output.status.success() => {
            eprintln!("[concierge] Playwright browsers installed");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "[concierge] WARNING: Playwright browser install failed (exit {}): {}",
                output.status.code().unwrap_or(-1),
                stderr.lines().next().unwrap_or("unknown error"),
            );
        }
        Err(e) => {
            eprintln!("[concierge] WARNING: could not run playwright install: {}", e);
        }
    }
}

/// Upgrade the installed package to a specific version (called after self-update).
///
/// Includes extras from `CONCIERGE_EXTRA` so that `--self-update` preserves
/// previously requested optional dependencies (mcp, otel, browser, embed, all).
pub fn upgrade_package(config: &LauncherConfig, version: &str) -> anyhow::Result<()> {
    let pip = config.venv_dir.join("bin").join("pip");
    let package_spec = match &config.pypi_extra {
        Some(extra) => format!("{}[{}]=={}", config.package_name, extra, version),
        None => format!("{}=={}", config.package_name, version),
    };
    let output = std::process::Command::new(&pip)
        .args(["install", "--upgrade", &package_spec])
        .output()
        .map_err(|e| SetupError::PackageInstall {
            code: -1,
            stderr: e.to_string(),
        })?;
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(SetupError::PackageInstall { code, stderr }.into());
    }
    std::fs::write(&config.version_file, version)?;

    // Update extras marker so ensure_extras stays in sync after upgrade.
    let extras_val = config.pypi_extra.as_deref().unwrap_or("");
    std::fs::write(&config.extras_file, extras_val)?;

    // Post-upgrade: ensure Playwright browsers if browser extra is included.
    if let Some(extra) = &config.pypi_extra {
        if extra.contains("browser") || extra == "all" {
            ensure_playwright_browsers(config);
        }
    }

    Ok(())
}

/// Read installed package version from version_file; None if file absent.
///
/// Not yet called from main — kept as public API for future status/health display.
#[allow(dead_code)]
pub fn installed_version(config: &LauncherConfig) -> anyhow::Result<Option<String>> {
    if !config.version_file.exists() {
        return Ok(None);
    }
    let version = std::fs::read_to_string(&config.version_file)?;
    Ok(Some(version.trim().to_string()))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Try ["python3", "python"] in PATH. Return Some(path) if >= 3.10, else None.
fn try_system_python() -> Option<PathBuf> {
    for name in &["python3", "python"] {
        if let Ok(output) = std::process::Command::new(name).arg("--version").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let version_str = if stdout.contains("Python") {
                    &*stdout
                } else {
                    &*stderr
                };
                if let Some(version) = parse_python_version(version_str) {
                    if version >= (3, 10) {
                        if let Ok(path) = which_bin(name) {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }
    None
}

fn parse_python_version(s: &str) -> Option<(u32, u32)> {
    let s = s.trim().strip_prefix("Python ")?.trim();
    let mut parts = s.splitn(3, '.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn which_bin(name: &str) -> anyhow::Result<PathBuf> {
    let output = std::process::Command::new("which").arg(name).output()?;
    if output.status.success() {
        let path = String::from_utf8(output.stdout)?.trim().to_string();
        Ok(PathBuf::from(path))
    } else {
        anyhow::bail!("which {} failed", name)
    }
}

/// Ensure uv binary exists at config.uv_path. Downloads from GitHub if absent.
///
/// Uses pure-Rust gzip + tar extraction (flate2 + tar crates) — no system
/// `tar` dependency required.
fn ensure_uv(config: &LauncherConfig) -> anyhow::Result<()> {
    if config.uv_path.exists() {
        return Ok(());
    }

    let arch = std::env::consts::ARCH;
    let url = format!(
        "https://github.com/astral-sh/uv/releases/latest/download/uv-{}-unknown-linux-musl.tar.gz",
        arch
    );

    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("concierge-launcher/{}", env!("CARGO_PKG_VERSION")))
        .build()?;

    let response = client.get(&url).send()?.error_for_status()?;
    let bytes = response.bytes()?;

    // Write tarball to a temp location, then extract with pure-Rust code.
    let extract_dir = config.data_dir.join(".uv-extract");
    std::fs::create_dir_all(&extract_dir)?;
    let tarball = extract_dir.join("uv.tar.gz");
    std::fs::write(&tarball, &bytes)?;

    let uv_bin = extract_uv(&tarball, &extract_dir).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&extract_dir);
    })?;

    std::fs::copy(&uv_bin, &config.uv_path)?;
    let _ = std::fs::remove_dir_all(&extract_dir);

    if !config.uv_path.exists() {
        return Err(SetupError::UvNotExecutable.into());
    }

    Ok(())
}

/// Extract the `uv` binary from a `.tar.gz` archive using pure Rust.
///
/// Iterates archive entries; returns the path of the extracted binary on
/// success, or an error if the archive contains no file named `uv`.
fn extract_uv(archive_path: &Path, dest_dir: &Path) -> anyhow::Result<PathBuf> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let f = std::fs::File::open(archive_path)?;
    let gz = GzDecoder::new(f);
    let mut archive = Archive::new(gz);

    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.path()?.file_name().is_some_and(|n| n == "uv") {
            let out = dest_dir.join("uv");
            entry.unpack(&out)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o755))?;
            }
            return Ok(out);
        }
    }
    Err(anyhow::anyhow!("uv binary not found in archive"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_config(data_dir: &Path) -> LauncherConfig {
        LauncherConfig {
            data_dir: data_dir.to_path_buf(),
            venv_dir: data_dir.join("venv"),
            uv_path: data_dir.join("uv"),
            version_file: data_dir.join("installed_version"),
            extras_file: data_dir.join("installed_extras"),
            bin_dir: data_dir.join("bin"),
            installed_bin: data_dir.join("bin").join("concierge"),
            skip_update: false,
            package_name: "agentic-concierge".to_string(),
            pypi_extra: None,
        }
    }

    #[test]
    fn installed_version_returns_none_when_no_file() {
        let dir = tempdir().unwrap();
        let config = make_config(dir.path());
        let v = installed_version(&config).unwrap();
        assert!(v.is_none());
    }

    #[test]
    fn installed_version_reads_file_contents() {
        let dir = tempdir().unwrap();
        let config = make_config(dir.path());
        std::fs::write(&config.version_file, "0.2.0\n").unwrap();
        let v = installed_version(&config).unwrap();
        assert_eq!(v, Some("0.2.0".to_string()));
    }

    #[test]
    fn ensure_environment_fast_path_returns_existing_binary() {
        let dir = tempdir().unwrap();
        let config = make_config(dir.path());
        std::fs::create_dir_all(config.venv_dir.join("bin")).unwrap();
        let bin = config.venv_dir.join("bin").join("concierge-py");
        std::fs::write(&bin, "#!/bin/sh\necho fake").unwrap();
        let result = ensure_environment(&config).unwrap();
        assert_eq!(result, bin);
    }

    // ── ensure_extras tests ────────────────────────────────────────────────────

    #[test]
    fn ensure_extras_noop_when_no_extra_requested() {
        let dir = tempdir().unwrap();
        let config = make_config(dir.path());
        // pypi_extra is None → should return Ok immediately
        let result = ensure_extras(&config);
        assert!(result.is_ok());
        assert!(!config.extras_file.exists());
    }

    #[test]
    fn ensure_extras_noop_when_already_installed() {
        let dir = tempdir().unwrap();
        let mut config = make_config(dir.path());
        config.pypi_extra = Some("mcp,otel".to_string());
        std::fs::write(&config.extras_file, "mcp,otel").unwrap();
        // Same extras already recorded → should return Ok without running pip
        let result = ensure_extras(&config);
        assert!(result.is_ok());
    }

    // ── extract_uv tests ──────────────────────────────────────────────────────

    /// Build an in-memory .tar.gz containing a single file named `filename`
    /// with `content` as its bytes.
    fn make_tar_gz(filename: &str, content: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use tar::Builder;

        let gz_buf = Vec::new();
        let enc = GzEncoder::new(gz_buf, Compression::default());
        let mut archive = Builder::new(enc);

        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();

        archive.append_data(&mut header, filename, content).unwrap();
        let enc = archive.into_inner().unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_extract_uv_from_synthetic_archive() {
        let dir = tempdir().unwrap();
        let fake_uv_content = b"#!/bin/sh\necho uv fake";

        let tar_gz_bytes = make_tar_gz("uv", fake_uv_content);
        let archive_path = dir.path().join("uv.tar.gz");
        std::fs::write(&archive_path, &tar_gz_bytes).unwrap();

        let result = extract_uv(&archive_path, dir.path());
        assert!(
            result.is_ok(),
            "extract_uv should succeed: {:?}",
            result.err()
        );

        let extracted_path = result.unwrap();
        assert_eq!(extracted_path, dir.path().join("uv"));
        assert_eq!(std::fs::read(&extracted_path).unwrap(), fake_uv_content);
    }

    #[test]
    fn test_extract_uv_missing_binary() {
        let dir = tempdir().unwrap();

        // Archive contains a file named "not-uv", not "uv"
        let tar_gz_bytes = make_tar_gz("not-uv", b"wrong binary");
        let archive_path = dir.path().join("uv.tar.gz");
        std::fs::write(&archive_path, &tar_gz_bytes).unwrap();

        let result = extract_uv(&archive_path, dir.path());
        assert!(
            result.is_err(),
            "should fail when archive has no 'uv' entry"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("uv binary not found"), "error message: {msg}");
    }
}
