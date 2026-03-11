"""Download models for non-Ollama backends.

Supports GGUF downloads for llama_cpp from HuggingFace.
Model source info is loaded from model_sources.yaml.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_sources_cache: Optional[Dict[str, Any]] = None


def _load_model_sources() -> Dict[str, Any]:
    """Load model source mapping from model_sources.yaml."""
    global _sources_cache
    if _sources_cache is not None:
        return _sources_cache
    from agentic_concierge.config.external import load_yaml_config
    _sources_cache = load_yaml_config("model_sources.yaml")
    return _sources_cache


def invalidate_sources_cache() -> None:
    """Clear the cached model sources (useful for testing)."""
    global _sources_cache
    _sources_cache = None


def get_model_source(ollama_name: str) -> Optional[Dict[str, Any]]:
    """Look up source info for an Ollama-style model name.

    Returns dict with keys: gguf_repo, gguf_file, hf_model_id, size_mb
    or None if not found.
    """
    sources = _load_model_sources()
    models = sources.get("models", {})
    return models.get(ollama_name)


def get_hf_model_id(ollama_name: str) -> Optional[str]:
    """Get the HuggingFace model ID for a given Ollama model name."""
    source = get_model_source(ollama_name)
    if source:
        return source.get("hf_model_id")
    return None


def get_gguf_url(ollama_name: str) -> Optional[Tuple[str, str]]:
    """Get the GGUF download URL and filename for an Ollama model name.

    Returns (url, filename) or None if not found.
    """
    source = get_model_source(ollama_name)
    if not source:
        return None
    repo = source.get("gguf_repo")
    filename = source.get("gguf_file")
    if not repo or not filename:
        return None
    sources = _load_model_sources()
    template = sources.get(
        "gguf_url_template",
        "https://huggingface.co/{repo}/resolve/main/{file}",
    )
    url = template.format(repo=repo, file=filename)
    return url, filename


def default_model_dir() -> str:
    """Return the default model directory for GGUF files.

    Uses platformdirs if available, falls back to
    ~/.local/share/agentic-concierge/models/gguf.
    """
    try:
        from platformdirs import user_data_path
        return str(user_data_path("agentic-concierge") / "models" / "gguf")
    except ImportError:
        return str(
            Path.home() / ".local" / "share" / "agentic-concierge" / "models" / "gguf"
        )


# ---------------------------------------------------------------------------
# Model alias index (maps ollama names -> GGUF filenames)
# ---------------------------------------------------------------------------

_ALIAS_FILE = "model_aliases.json"


def _load_aliases(model_dir: str) -> Dict[str, str]:
    """Load model_aliases.json from model_dir."""
    path = os.path.join(model_dir, _ALIAS_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_aliases(model_dir: str, aliases: Dict[str, str]) -> None:
    """Save model_aliases.json to model_dir."""
    path = os.path.join(model_dir, _ALIAS_FILE)
    with open(path, "w") as f:
        json.dump(aliases, f, indent=2)


def resolve_alias(model_dir: str, model_id: str) -> Optional[str]:
    """Resolve an Ollama-style model name to a GGUF filename via aliases.

    Returns the GGUF filename if aliased, None otherwise.
    """
    aliases = _load_aliases(model_dir)
    return aliases.get(model_id)


def list_aliased_models(model_dir: str) -> List[str]:
    """Return all aliased (Ollama-style) model names available in model_dir."""
    aliases = _load_aliases(model_dir)
    # Only return aliases whose GGUF files actually exist
    result = []
    for name, gguf_file in aliases.items():
        if os.path.isfile(os.path.join(model_dir, gguf_file)):
            result.append(name)
    return sorted(result)


# ---------------------------------------------------------------------------
# GGUF download
# ---------------------------------------------------------------------------


async def download_gguf(
    ollama_name: str,
    model_dir: str,
    *,
    timeout_s: int = 1800,
    progress_callback=None,
) -> Optional[str]:
    """Download a GGUF file for the given Ollama model name.

    Args:
        ollama_name: Ollama-style model name (e.g. "qwen2.5:7b")
        model_dir: Directory to save the GGUF file
        timeout_s: Download timeout in seconds (default 30 min)
        progress_callback: Optional callable(downloaded_mb, total_mb) for progress

    Returns:
        The GGUF filename on success, None on failure.
    """
    info = get_gguf_url(ollama_name)
    if info is None:
        logger.warning("No GGUF source info for model %r", ollama_name)
        return None

    url, filename = info
    dest_path = os.path.join(model_dir, filename)

    # Skip if already downloaded
    if os.path.isfile(dest_path):
        source = get_model_source(ollama_name)
        expected_mb = source.get("size_mb", 0) if source else 0
        actual_mb = os.path.getsize(dest_path) / (1024 * 1024)
        # Accept if within 10% of expected size (or no expected size known)
        if expected_mb == 0 or abs(actual_mb - expected_mb) / expected_mb < 0.1:
            logger.info("GGUF %s already exists at %s", filename, dest_path)
            # Ensure alias is registered
            aliases = _load_aliases(model_dir)
            if ollama_name not in aliases:
                aliases[ollama_name] = filename
                _save_aliases(model_dir, aliases)
            return filename

    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Downloading GGUF %s from %s", filename, url)

    import httpx

    tmp_path = dest_path + ".download"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        ) as client:
            async with client.stream("GET", url, follow_redirects=True) as resp:
                if resp.status_code != 200:
                    logger.warning(
                        "GGUF download failed: HTTP %d for %s",
                        resp.status_code, url,
                    )
                    return None

                total = int(resp.headers.get("content-length", 0))
                total_mb = total / (1024 * 1024)
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total:
                            progress_callback(
                                downloaded / (1024 * 1024),
                                total_mb,
                            )

        # Atomic rename
        os.replace(tmp_path, dest_path)
        logger.info(
            "Downloaded GGUF %s (%d MB)",
            filename,
            os.path.getsize(dest_path) // (1024 * 1024),
        )

        # Register alias
        aliases = _load_aliases(model_dir)
        aliases[ollama_name] = filename
        _save_aliases(model_dir, aliases)

        return filename

    except Exception as e:
        logger.warning("GGUF download failed for %s: %s", ollama_name, e)
        # Clean up partial download
        if os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return None
