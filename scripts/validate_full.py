#!/usr/bin/env python3
"""
Full validation: ensure a real LLM is available, then run the full test suite.
At least a couple of E2E tests must run against a real LLM to ensure everything
is integrated and working as expected; this script enforces that (no skips).

- If the configured LLM is not reachable, we try to start it (e.g. ollama serve).
- If we cannot reach or start an LLM, we exit with code 1 and do not run tests.
- We run pytest without CONCIERGE_SKIP_REAL_LLM so all tests run (including real-LLM E2E).
- Exit code is pytest's exit code (0 = all passed; non-zero = failure).

Usage (from repo root):
  python scripts/validate_full.py
  python scripts/validate_full.py --no-ensure   # Skip ensure_llm_available; fail if unreachable
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def main():
    no_ensure = "--no-ensure" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-ensure"]

    cfg = __import__("agentic_concierge.config", fromlist=["load_config"]).load_config()
    model_cfg = cfg.models.get("quality") or cfg.models.get("fast")
    if not model_cfg:
        print("ERROR: No model config (quality/fast). Check CONCIERGE_CONFIG_PATH or defaults.")
        return 1

    if not no_ensure and cfg.local_llm_ensure_available and cfg.local_llm_start_cmd:
        from agentic_concierge.infrastructure.llm_bootstrap import ensure_llm_available
        try:
            ok = ensure_llm_available(
                model_cfg.base_url,
                start_cmd=cfg.local_llm_start_cmd,
                timeout_s=cfg.local_llm_start_timeout_s,
            )
            if not ok:
                print("ERROR: LLM at", model_cfg.base_url, "is not reachable and no start command was run.")
                return 1
        except (TimeoutError, FileNotFoundError) as e:
            print("ERROR: Could not ensure LLM is available:", e)
            print("Full validation requires a running LLM. Start Ollama (ollama serve, ollama pull <model>) or set local_llm_ensure_available: false and run with a server.")
            return 1
    else:
        from agentic_concierge.infrastructure.llm_bootstrap import _check_reachable
        if not _check_reachable(model_cfg.base_url, timeout_s=5.0):
            print("ERROR: LLM at", model_cfg.base_url, "is not reachable.")
            print("Full validation requires a running LLM. Start the server or use --no-ensure after starting it.")
            return 1

    env = os.environ.copy()
    env.pop("CONCIERGE_SKIP_REAL_LLM", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"] + args,
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        return result.returncode
    # Full validation requires that no tests were skipped (all run, including real-LLM E2E).
    out = result.stdout + result.stderr
    if " skipped" in out or " skip" in out.lower():
        import re
        m = re.search(r"(\d+) skipped", out)
        n = int(m.group(1)) if m else 1
        print("ERROR: Full validation requires all tests to run (no skips).", n, "test(s) were skipped.", file=sys.stderr)
        print("Real-LLM E2E tests were skipped. Ensure the configured model is available (e.g. ollama pull <model>).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
