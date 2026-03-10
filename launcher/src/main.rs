// main.rs — arg parsing and orchestration only.
// All I/O lives in the four modules below; main.rs is the only file that imports from them.
mod config;
mod exec;
mod setup;
mod update;

use config::launcher_config;
use exec::exec_python_concierge;
use setup::{ensure_environment, upgrade_package};
use update::{apply_update, check_latest_release, is_newer, UpdateCheckResult};

/// Return true if `--self-update` appears anywhere in the CLI args.
/// (No `clap` — keeps the binary small.)
fn parse_launcher_args() -> bool {
    std::env::args().skip(1).any(|a| a == "--self-update")
}

fn main() -> anyhow::Result<()> {
    let self_update = parse_launcher_args();
    let config = launcher_config()?;

    if self_update {
        // --self-update: always try GitHub regardless of skip_update.
        match check_latest_release(&config) {
            UpdateCheckResult::Found(r) => {
                apply_update(&config, &r)?;
                upgrade_package(&config, &r.version)?;
                eprintln!("[concierge] restart to use the updated version");
                std::process::exit(0);
            }
            UpdateCheckResult::AlreadyUpToDate => {
                eprintln!("[concierge] already at latest version (v{})", env!("CARGO_PKG_VERSION"));
                // Ensure the Python package is in sync even when the launcher
                // binary is already current (recovers from failed prior upgrades).
                let concierge_bin = config.venv_dir.join("bin").join("concierge-py");
                if concierge_bin.exists() {
                    if let Err(e) = ensure_environment(&config) {
                        eprintln!("[concierge] WARNING: package sync failed: {}", e);
                    }
                }
                std::process::exit(0);
            }
            UpdateCheckResult::NoBinaryForPlatform { version } => {
                eprintln!(
                    "[concierge] v{} is available but no binary for this platform ({}-{})",
                    version,
                    std::env::consts::ARCH,
                    std::env::consts::OS,
                );
                eprintln!("[concierge] the release build may still be in progress — try again shortly");
                // Still upgrade the Python package even without a launcher binary
                upgrade_package(&config, &version)?;
                std::process::exit(0);
            }
            UpdateCheckResult::Unreachable => {
                eprintln!("[concierge] could not reach GitHub — check your network connection");
                std::process::exit(1);
            }
        }
    } else if !config.skip_update {
        // Passive check: advisory only — never blocks startup.
        if let UpdateCheckResult::Found(r) = check_latest_release(&config) {
            if is_newer(&r) {
                eprintln!(
                    "[concierge] update available: v{} \u{2014} run --self-update",
                    r.version
                );
            }
        }
    }

    let concierge_bin = ensure_environment(&config)?;
    exec_python_concierge(&concierge_bin)
}
