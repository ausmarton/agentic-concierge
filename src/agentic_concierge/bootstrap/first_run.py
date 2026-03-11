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

    # 1. Check for cached profile
    cached_profile: Optional["SystemProfile"] = None
    if not force_profile and not is_first_run(path=detected_override):
        cached_profile = load_detected(path=detected_override)
        if cached_profile is not None:
            logger.debug("Bootstrap: cached profile %s loaded.", cached_profile.tier.value)

    # 2. Probe system (always needed for backend ensure steps)
    if interactive and cached_profile is None:
        _print_status("Probing system resources…")
    probe = await probe_system()

    # 3. Advise profile (or use forced override, or cached)
    if cached_profile is not None:
        profile = cached_profile
    elif force_profile:
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

    # 4. Show summary (skip for cached profiles)
    if interactive and cached_profile is None:
        _print_profile_panel(probe, profile)

    # 5. Ensure backends enabled for this profile (ALWAYS runs, even for cached profiles)
    from agentic_concierge.config.features import PROFILE_FEATURES, Feature, FeatureSet
    profile_features = PROFILE_FEATURES.get(profile.tier, frozenset())
    feature_set = FeatureSet(profile_features)

    from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus
    mgr = BackendManager()
    from agentic_concierge.config import load_config
    try:
        cfg = load_config()
    except Exception:
        cfg = None

    # 5a. Ensure Ollama
    if Feature.OLLAMA in profile_features and probe.ollama_installed:
        if interactive:
            _print_status("Ensuring Ollama is running…")
        if cfg is not None:
            try:
                health = await mgr.ensure_ollama(cfg)
                if health.status != BackendStatus.HEALTHY:
                    logger.warning("Ollama not healthy after bootstrap attempt: %s", health.error)
                    if interactive:
                        _print_status(
                            f"[yellow]⚠ Ollama not healthy: {health.hint or health.error}[/yellow]"
                        )
            except Exception as e:
                logger.warning("Ollama ensure failed: %s", e)

    # 5b. Ensure llama-server if llama_cpp is a preferred backend
    from agentic_concierge.config.platform import detect_platform
    plat = detect_platform()
    if Feature.LLAMA_CPP in profile_features and "llama_cpp" in plat.preferred_backends:
        if not probe.llama_server_installed:
            if interactive:
                _print_status("Installing llama-server (llama-cpp-python[server])…")
            try:
                health = await mgr.ensure_llama_server()
                if health.status == BackendStatus.HEALTHY:
                    if interactive:
                        _print_status("[green]✓ llama-server installed[/green]")
                else:
                    if interactive:
                        _print_status(
                            f"[yellow]⚠ llama-server: {health.hint or health.error}[/yellow]"
                        )
            except Exception as e:
                logger.warning("llama-server install failed: %s", e)

    # 5c. Ensure vLLM if enabled and preferred
    if Feature.VLLM in profile_features and "vllm" in plat.preferred_backends:
        if interactive:
            _print_status("Ensuring vLLM is available…")
        if cfg is not None:
            try:
                health = await mgr.ensure_vllm(cfg, feature_set, install_if_missing=True)
                if health.status == BackendStatus.HEALTHY:
                    if interactive:
                        msg = health.hint or "vLLM is running"
                        _print_status(f"[green]✓ {msg}[/green]")
                elif health.status != BackendStatus.DISABLED:
                    logger.info("vLLM not available: %s", health.hint or health.error)
                    if interactive:
                        _print_status(
                            f"[dim]vLLM: {health.hint or health.error}[/dim]"
                        )
            except Exception as e:
                logger.warning("vLLM ensure failed: %s", e)

    # 5d. Provision models on all healthy backends
    await mgr.probe_all(feature_set)
    healthy = mgr.get_healthy_backends()
    if healthy:
        models_to_provision = [profile.routing_model, profile.fast_model]
        if profile.fast_model != profile.quality_model:
            models_to_provision.append(profile.quality_model)

        if interactive:
            _print_status("Provisioning models on healthy backends...")

        try:
            provisioned = await mgr.ensure_backend_models(
                models_to_provision, healthy, interactive=interactive,
            )
            for backend, models in provisioned.items():
                if models and interactive:
                    _print_status(
                        f"[green]{backend}: {len(models)} model(s) ready[/green]"
                    )
        except Exception as e:
            logger.warning("Backend model provisioning failed: %s", e)

    # 6. Pull recommended models (non-interactive: skip)
    if interactive and probe.ollama_reachable:
        models_to_pull = [profile.routing_model, profile.fast_model]
        if profile.fast_model != profile.quality_model:
            models_to_pull.append(profile.quality_model)
        await _pull_models(models_to_pull)

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
