"""First-run bootstrap: probe system, advise profile, install and pull models.

The bootstrap sequence:
1. Check ``detected.json`` — return cached profile if it exists (unless forced).
2. Probe system resources concurrently with any pre-loading.
3. ``advise_profile()`` from the probe snapshot.
4. If interactive, show a Rich summary panel.
5. If ollama feature is enabled, ``ensure_ollama()`` (start if not running).
6. Pull recommended models with a Rich progress display.
7. ``save_detected(profile)`` to disk.
8. Return ``SystemProfile``.

Call ``concierge bootstrap`` CLI command to run this interactively.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agentic_concierge.bootstrap.system_probe import probe_system  # importable for mocking

if TYPE_CHECKING:
    from agentic_concierge.bootstrap.model_advisor import SystemProfile
    from agentic_concierge.bootstrap.system_probe import SystemProbe

logger = logging.getLogger(__name__)


async def run(
    interactive: bool = True,
    force_profile: Optional[str] = None,
    detected_override: Optional[Path] = None,
) -> "SystemProfile":  # type: ignore[name-defined]
    """Run the first-run bootstrap sequence.

    Args:
        interactive: Show Rich progress panels and prompts when ``True``.
        force_profile: Override profile tier (e.g. ``"medium"``). ``None``
            means auto-detect.
        detected_override: Override the path to ``detected.json`` (for tests).

    Returns:
        The detected (or forced) ``SystemProfile``.
    """
    from agentic_concierge.bootstrap.detected import (
        is_first_run,
        load_detected,
        save_detected,
    )
    from agentic_concierge.bootstrap.model_advisor import SystemProfile, advise_profile
    from agentic_concierge.config.features import ProfileTier

    # 1. Return cached profile if detected.json exists and not forced
    if not force_profile and not is_first_run(path=detected_override):
        cached = load_detected(path=detected_override)
        if cached is not None:
            logger.debug("Bootstrap: cached profile %s loaded.", cached.tier.value)
            return cached

    # 2. Probe system
    if interactive:
        _print_status("Probing system resources…")
    probe = await probe_system()

    # 3. Advise profile (or use forced override)
    if force_profile:
        try:
            tier = ProfileTier(force_profile.lower())
        except ValueError:
            valid = ", ".join(t.value for t in ProfileTier)
            raise ValueError(
                f"Unknown profile {force_profile!r}. Valid values: {valid}"
            ) from None
        # Build a synthetic profile from the forced tier
        from agentic_concierge.bootstrap.model_advisor import _loaded_model_table, _loaded_model_ctx_mb
        import math
        models = _loaded_model_table()[tier]
        model_ctx_mb = _loaded_model_ctx_mb()
        overhead_mb = max(2048, probe.ram_total_mb * 0.15) + 512
        usable_mb = probe.ram_available_mb - overhead_mb
        max_concurrent = max(1, min(math.floor(usable_mb / model_ctx_mb), probe.cpu_cores - 1))
        profile = SystemProfile(
            tier=tier,
            routing_model=models["routing"],
            fast_model=models["fast"],
            quality_model=models["quality"],
            max_concurrent_agents=max_concurrent,
            ram_total_mb=probe.ram_total_mb,
            ram_available_mb=probe.ram_available_mb,
            total_vram_mb=probe.total_vram_mb,
            cpu_cores=probe.cpu_cores,
            cpu_arch=probe.cpu_arch,
            gpu_count=len(probe.gpu_devices),
        )
    else:
        profile = advise_profile(probe)

    # 4. Show summary
    if interactive:
        _print_profile_panel(probe, profile)

    # 5. Ensure Ollama if enabled for this profile
    from agentic_concierge.config.features import PROFILE_FEATURES, Feature
    profile_features = PROFILE_FEATURES.get(profile.tier, frozenset())
    if Feature.OLLAMA in profile_features and probe.ollama_installed:
        if interactive:
            _print_status("Ensuring Ollama is running…")
        from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus
        mgr = BackendManager()
        from agentic_concierge.config import load_config
        try:
            cfg = load_config()
            health = await mgr.ensure_ollama(cfg)
            if health.status != BackendStatus.HEALTHY:
                logger.warning("Ollama not healthy after bootstrap attempt: %s", health.error)
                if interactive:
                    _print_status(
                        f"[yellow]⚠ Ollama not healthy: {health.hint or health.error}[/yellow]"
                    )
        except Exception as e:
            logger.warning("Ollama ensure failed: %s", e)

    # 6. Pull recommended models (non-interactive: skip)
    if interactive and probe.ollama_reachable:
        models_to_pull = [profile.routing_model, profile.fast_model]
        if profile.fast_model != profile.quality_model:
            models_to_pull.append(profile.quality_model)
        await _pull_models(models_to_pull)

    # 6b. Download nano GGUF if inprocess is enabled and model is missing
    if Feature.INPROCESS in profile_features:
        from agentic_concierge.infrastructure.chat.inprocess import is_available
        if is_available():
            await _ensure_nano_model(interactive)
        elif interactive:
            _print_status("[dim]Skipping nano model (mistralrs not installed)[/dim]")

    # 7. Save detected profile
    save_detected(profile, path=detected_override)
    logger.info("Bootstrap complete: tier=%s saved to detected.json.", profile.tier.value)

    return profile


def _print_status(msg: str) -> None:
    """Print a status line using Rich if available, otherwise plain print."""
    try:
        from rich import print as rprint
        rprint(msg)
    except ImportError:
        print(msg)


def _print_profile_panel(probe: "SystemProbe", profile: "SystemProfile") -> None:  # type: ignore[name-defined]
    """Print a Rich panel summarising the detected hardware and recommended profile."""
    try:
        from rich import print as rprint
        from rich.panel import Panel

        gpu_info = (
            ", ".join(f"{d.name} ({d.vram_mb} MB VRAM)" for d in probe.gpu_devices)
            or "No GPU detected"
        )
        body = (
            f"[bold]Profile:[/bold]          {profile.tier.value.upper()}\n"
            f"[bold]CPU:[/bold]              {probe.cpu_cores} cores ({probe.cpu_arch})\n"
            f"[bold]RAM:[/bold]              {profile.ram_total_mb // 1024} GB total, "
            f"{profile.ram_available_mb // 1024} GB available\n"
            f"[bold]GPU:[/bold]              {gpu_info}\n"
            f"[bold]Routing model:[/bold]    {profile.routing_model}\n"
            f"[bold]Fast model:[/bold]       {profile.fast_model}\n"
            f"[bold]Quality model:[/bold]    {profile.quality_model}\n"
            f"[bold]Max agents:[/bold]       {profile.max_concurrent_agents}\n"
            f"[bold]Internet:[/bold]         {'✓' if probe.internet_reachable else '✗'}\n"
            f"[bold]Ollama:[/bold]           {'installed' if probe.ollama_installed else 'not found'}"
            f" ({'reachable' if probe.ollama_reachable else 'unreachable'})\n"
        )
        rprint(Panel.fit(body, title="[bold cyan]agentic-concierge bootstrap[/bold cyan]"))
    except Exception:
        pass


async def _ensure_nano_model(interactive: bool) -> None:
    """Download the nano GGUF model if not already present."""
    from agentic_concierge.config.constants import nano_gguf_filename, nano_gguf_url

    gguf_filename = nano_gguf_filename()
    gguf_download_url = nano_gguf_url()

    try:
        from platformdirs import user_data_path
        models_dir = user_data_path("agentic-concierge") / "models"
    except Exception:
        import os
        models_dir = Path(os.path.expanduser("~")) / ".agentic-concierge" / "models"

    gguf_path = models_dir / gguf_filename
    if gguf_path.exists():
        if interactive:
            _print_status(f"[dim]Nano model already present: {gguf_path}[/dim]")
        return

    if interactive:
        _print_status(f"[bold]Downloading nano model: {gguf_filename}[/bold]")

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = gguf_path.with_suffix(".download")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            async with client.stream("GET", gguf_download_url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                if interactive and total > 0:
                    from rich.progress import (
                        BarColumn,
                        DownloadColumn,
                        Progress,
                        TransferSpeedColumn,
                        TimeRemainingColumn,
                    )
                    progress = Progress(
                        "[progress.description]{task.description}",
                        BarColumn(),
                        DownloadColumn(),
                        TransferSpeedColumn(),
                        TimeRemainingColumn(),
                    )
                    task_id = progress.add_task(gguf_filename, total=total)
                    with progress, open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
                else:
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                            f.write(chunk)
        tmp_path.rename(gguf_path)
        if interactive:
            _print_status(f"[green]Nano model ready: {gguf_path}[/green]")
    except Exception as e:
        logger.warning("Failed to download nano GGUF: %s", e)
        if interactive:
            _print_status(f"[yellow]Could not download nano model: {e}[/yellow]")
        if tmp_path.exists():
            tmp_path.unlink()


async def _pull_models(models: list[str]) -> None:
    """Pull Ollama models with a Rich progress display."""
    try:
        import httpx
        from rich.console import Console
        console = Console()
        for model in models:
            console.print(f"[dim]Pulling {model}…[/dim]")
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    async with client.stream(
                        "POST",
                        "http://localhost:11434/api/pull",
                        json={"name": model, "stream": True},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                import json
                                try:
                                    evt = json.loads(line)
                                    status = evt.get("status", "")
                                    if status == "success":
                                        console.print(f"[green]✓ {model} ready[/green]")
                                        break
                                    if evt.get("error"):
                                        console.print(
                                            f"[yellow]⚠ {model}: {evt['error']}[/yellow]"
                                        )
                                        break
                                except Exception:
                                    pass
            except Exception as e:
                logger.warning("Failed to pull %s: %s", model, e)
                console.print(f"[yellow]⚠ Could not pull {model}: {e}[/yellow]")
    except ImportError:
        pass
