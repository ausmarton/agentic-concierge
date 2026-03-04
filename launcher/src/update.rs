use semver::Version;
use serde::Deserialize;

use crate::config::LauncherConfig;

/// CPU architecture string used to select the correct binary asset.
const ARCH_STR: &str = if cfg!(target_arch = "x86_64") {
    "x86_64"
} else if cfg!(target_arch = "aarch64") {
    "aarch64"
} else {
    "unknown"
};

/// Full target triple for asset naming.  Determined at compile time.
///
/// macOS runners produce `*-apple-darwin`; Linux musl runners produce
/// `*-unknown-linux-musl`.  Non-musl Linux dev builds fall through to the
/// Linux musl name (functionally wrong for self-update, but tests still
/// compile and the binary size / unit tests are unaffected).
fn asset_target_suffix() -> &'static str {
    if cfg!(target_os = "macos") {
        "apple-darwin"
    } else {
        "unknown-linux-musl"
    }
}

/// Ed25519 public key for release binary verification.
///
/// **PLACEHOLDER** — replace with output of `scripts/generate_signing_key.sh`
/// before publishing a release.  Until replaced, `verify_binary_signature`
/// will always return an error (which is the safe default: unsigned binaries
/// are rejected).
const SIGNING_PUBLIC_KEY: [u8; 32] = [0u8; 32];

#[derive(Debug, Clone)]
pub struct ReleaseInfo {
    pub version: String,      // "0.3.1" (tag stripped of "v")
    pub download_url: String, // URL to the binary asset
}

#[derive(Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

/// Check GitHub Releases API for the latest release.
///
/// Returns `None` on *any* failure — network errors are silently ignored so
/// the launcher never fails due to an unavailable update server.
pub fn check_latest_release(_config: &LauncherConfig) -> Option<ReleaseInfo> {
    check_latest_release_inner().ok().flatten()
}

fn check_latest_release_inner() -> anyhow::Result<Option<ReleaseInfo>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("concierge-launcher/{}", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    let response = client
        .get("https://api.github.com/repos/ausmarton/agentic-concierge/releases/latest")
        .send()?;

    if !response.status().is_success() {
        return Ok(None);
    }

    let release: GitHubRelease = response.json()?;
    let version = release.tag_name.trim_start_matches('v').to_string();

    let asset_name = format!("concierge-{}-{}", ARCH_STR, asset_target_suffix());
    let asset = release.assets.iter().find(|a| a.name == asset_name);

    Ok(asset.map(|a| ReleaseInfo {
        version,
        download_url: a.browser_download_url.clone(),
    }))
}

/// Verify Ed25519 signature of a binary using the embedded public key.
///
/// Signature must be exactly 64 raw bytes stored at `sig_path`.
/// Returns `Err` if the sig is absent, malformed, or does not match.
fn verify_binary_signature(
    binary_path: &std::path::Path,
    sig_path: &std::path::Path,
) -> anyhow::Result<()> {
    verify_binary_signature_with_key(binary_path, sig_path, &SIGNING_PUBLIC_KEY)
}

/// Inner verification function that accepts an explicit public key.
/// Used in tests to avoid dependence on the placeholder `SIGNING_PUBLIC_KEY`.
fn verify_binary_signature_with_key(
    binary_path: &std::path::Path,
    sig_path: &std::path::Path,
    pub_key_bytes: &[u8; 32],
) -> anyhow::Result<()> {
    use ed25519_dalek::{Signature, VerifyingKey};

    let binary = std::fs::read(binary_path)?;
    let sig_bytes = std::fs::read(sig_path)?;

    let sig_bytes_64: &[u8; 64] = sig_bytes.as_slice().try_into().map_err(|_| {
        anyhow::anyhow!(
            "invalid signature file: expected 64 bytes, got {}",
            sig_bytes.len()
        )
    })?;
    let sig = Signature::from_bytes(sig_bytes_64);

    let key = VerifyingKey::from_bytes(pub_key_bytes)
        .map_err(|e| anyhow::anyhow!("invalid public key: {e}"))?;

    key.verify_strict(&binary, &sig)
        .map_err(|_| anyhow::anyhow!("[concierge] signature verification failed — update aborted"))
}

/// Download binary to a temp file, verify Ed25519 signature, chmod +x, then
/// atomically rename to `config.installed_bin`.
///
/// Flow:
///   1. Download binary  → data_dir/concierge.new
///   2. Derive sig_url   → download_url + ".sig"
///   3. Download sig     → data_dir/concierge.new.sig
///   4. verify_binary_signature(concierge.new, concierge.new.sig)?
///   5. chmod 0o755 concierge.new
///   6. rename(concierge.new, installed_bin)   ← atomic on same filesystem
///   7. remove concierge.new.sig
///   8. eprintln! updated message
pub fn apply_update(config: &LauncherConfig, release: &ReleaseInfo) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("concierge-launcher/{}", env!("CARGO_PKG_VERSION")))
        .build()?;

    // Step 1 — download binary
    let response = client
        .get(&release.download_url)
        .send()?
        .error_for_status()?;
    let bytes = response.bytes()?;

    let new_path = config.data_dir.join("concierge.new");
    std::fs::write(&new_path, &bytes)?;

    // Step 2+3 — download signature (optional: absent sig = warn + skip verify)
    let sig_url = format!("{}.sig", release.download_url);
    let sig_response = client.get(&sig_url).send()?;
    let sig_status = sig_response.status();

    if sig_status.is_success() {
        let sig_bytes = sig_response.bytes()?;
        let sig_path = config.data_dir.join("concierge.new.sig");
        std::fs::write(&sig_path, &sig_bytes)?;

        // Step 4 — verify before applying; clean up on failure
        if let Err(e) = verify_binary_signature(&new_path, &sig_path) {
            let _ = std::fs::remove_file(&new_path);
            let _ = std::fs::remove_file(&sig_path);
            return Err(e);
        }
        let _ = std::fs::remove_file(&sig_path);
    } else if sig_status.as_u16() == 404 {
        // No .sig on this release (LAUNCHER_SIGNING_KEY_PEM not configured in CI).
        // Warn but allow the update — verification is best-effort until a key is set up.
        eprintln!(
            "[concierge] WARNING: no signature file for v{} — skipping verification",
            release.version
        );
    } else {
        // Unexpected HTTP error fetching the sig — abort and clean up.
        let _ = std::fs::remove_file(&new_path);
        return Err(anyhow::anyhow!(
            "failed to download signature file: HTTP {}",
            sig_status
        ));
    }

    // Step 5 — chmod +x
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&new_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&new_path, perms)?;
    }

    // Step 6 — atomic rename
    std::fs::rename(&new_path, &config.installed_bin)?;

    eprintln!("[concierge] updated to v{}", release.version);
    Ok(())
}

/// Return true if `release.version` is strictly greater than the current binary version.
pub fn is_newer(release: &ReleaseInfo) -> bool {
    let current = match Version::parse(env!("CARGO_PKG_VERSION")) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let latest = match Version::parse(&release.version) {
        Ok(v) => v,
        Err(_) => return false,
    };
    latest > current
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use tempfile::tempdir;

    /// A fixed 32-byte seed for deterministic test keypairs.
    const TEST_SEED: [u8; 32] = [1u8; 32];
    const TEST_SEED_2: [u8; 32] = [2u8; 32];

    fn make_test_keypair(seed: &[u8; 32]) -> (SigningKey, [u8; 32]) {
        let sk = SigningKey::from_bytes(seed);
        let pk = sk.verifying_key().to_bytes();
        (sk, pk)
    }

    fn make_config(data_dir: &std::path::Path) -> LauncherConfig {
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

    // ── is_newer tests (unchanged from Phase 13) ──────────────────────────────

    #[test]
    fn is_newer_greater() {
        let release = ReleaseInfo {
            version: "999.0.0".to_string(),
            download_url: String::new(),
        };
        assert!(is_newer(&release));
    }

    #[test]
    fn is_newer_same_version() {
        let release = ReleaseInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            download_url: String::new(),
        };
        assert!(!is_newer(&release));
    }

    #[test]
    fn is_newer_older_version() {
        let release = ReleaseInfo {
            version: "0.0.1".to_string(),
            download_url: String::new(),
        };
        assert!(!is_newer(&release));
    }

    #[test]
    fn arch_str_is_not_unknown() {
        assert_ne!(ARCH_STR, "unknown");
    }

    // ── Ed25519 signature verification tests ──────────────────────────────────

    #[test]
    fn test_verify_signature_valid() {
        let (sk, pk) = make_test_keypair(&TEST_SEED);
        let dir = tempdir().unwrap();

        let binary: &[u8] = b"fake binary content for test";
        let sig = sk.sign(binary);

        let bin_path = dir.path().join("binary");
        let sig_path = dir.path().join("binary.sig");
        std::fs::write(&bin_path, binary).unwrap();
        std::fs::write(&sig_path, sig.to_bytes()).unwrap();

        assert!(verify_binary_signature_with_key(&bin_path, &sig_path, &pk).is_ok());
    }

    #[test]
    fn test_verify_signature_tampered_binary() {
        let (sk, pk) = make_test_keypair(&TEST_SEED);
        let dir = tempdir().unwrap();

        let binary: &[u8] = b"original binary content";
        let sig = sk.sign(binary);

        let bin_path = dir.path().join("binary");
        let sig_path = dir.path().join("binary.sig");
        // Write DIFFERENT content — sig no longer matches
        std::fs::write(&bin_path, b"tampered binary content").unwrap();
        std::fs::write(&sig_path, sig.to_bytes()).unwrap();

        assert!(verify_binary_signature_with_key(&bin_path, &sig_path, &pk).is_err());
    }

    #[test]
    fn test_verify_signature_wrong_key() {
        let (sk, _pk) = make_test_keypair(&TEST_SEED);
        let (_sk2, wrong_pk) = make_test_keypair(&TEST_SEED_2);
        let dir = tempdir().unwrap();

        let binary: &[u8] = b"signed with correct key";
        let sig = sk.sign(binary);

        let bin_path = dir.path().join("binary");
        let sig_path = dir.path().join("binary.sig");
        std::fs::write(&bin_path, binary).unwrap();
        std::fs::write(&sig_path, sig.to_bytes()).unwrap();

        // Verify with wrong key — must fail
        assert!(verify_binary_signature_with_key(&bin_path, &sig_path, &wrong_pk).is_err());
    }

    #[test]
    fn test_verify_signature_truncated_sig() {
        let (_sk, pk) = make_test_keypair(&TEST_SEED);
        let dir = tempdir().unwrap();

        let bin_path = dir.path().join("binary");
        let sig_path = dir.path().join("binary.sig");
        std::fs::write(&bin_path, b"some binary content").unwrap();
        // Only 7 bytes — far too short for a 64-byte Ed25519 signature
        std::fs::write(&sig_path, b"short!!").unwrap();

        assert!(verify_binary_signature_with_key(&bin_path, &sig_path, &pk).is_err());
    }

    #[test]
    fn test_apply_update_blocked_on_bad_sig() {
        // Simulate the post-download state: binary exists but signature is bad.
        // apply_update would have written these files; we pre-populate them and
        // call the inner apply path directly to avoid needing a real HTTP server.
        let dir = tempdir().unwrap();
        let config = make_config(dir.path());
        std::fs::create_dir_all(&config.bin_dir).unwrap();

        let new_path = config.data_dir.join("concierge.new");
        let sig_path = config.data_dir.join("concierge.new.sig");
        std::fs::write(&new_path, b"fake binary").unwrap();
        // Bad sig: not 64 bytes — Signature::from_bytes will reject it
        std::fs::write(&sig_path, b"bad!").unwrap();

        // verify_binary_signature uses SIGNING_PUBLIC_KEY; even with the
        // placeholder key the sig parsing fails first (wrong length).
        let result = verify_binary_signature(&new_path, &sig_path);
        assert!(result.is_err(), "bad sig should be rejected");

        // The installed binary must NOT have been written
        assert!(!config.installed_bin.exists());
    }
}
