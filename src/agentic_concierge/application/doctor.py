"""System diagnostics — validates config, probes backends, checks readiness.

Provides ``run_diagnostics()`` which produces a structured ``DiagnosticReport``
with actionable suggestions.  Used by the ``concierge doctor`` CLI command.

See DESIGN_V2.md §12.2 for design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    passed: bool
    message: str
    suggestion: str = ""  # actionable fix if not passed


@dataclass
class DiagnosticReport:
    """Complete diagnostics report."""

    checks: List[CheckResult] = field(default_factory=list)
    backend_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_inventory: List[Dict[str, Any]] = field(default_factory=list)
    runtime_status: Optional[Dict[str, Any]] = None
    memory_budget_mb: int = 0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        return f"{passed}/{total} checks passed"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_config_loadable() -> CheckResult:
    """Verify that the config files load without errors."""
    try:
        from agentic_concierge.config import load_config
        load_config()
        return CheckResult(
            name="config_loadable",
            passed=True,
            message="Configuration loaded successfully",
        )
    except Exception as exc:
        return CheckResult(
            name="config_loadable",
            passed=False,
            message=f"Configuration failed to load: {exc}",
            suggestion="Check ~/.config/concierge/ for invalid YAML files",
        )


def check_backends_yaml() -> CheckResult:
    """Verify that backends.yaml is valid."""
    try:
        from agentic_concierge.config.external import load_yaml_config
        cfg = load_yaml_config("backends.yaml")
        if not cfg.get("backends"):
            return CheckResult(
                name="backends_yaml",
                passed=False,
                message="backends.yaml missing 'backends' section",
                suggestion="Ensure backends.yaml has at least an 'ollama' entry",
            )
        return CheckResult(
            name="backends_yaml",
            passed=True,
            message=f"backends.yaml valid ({len(cfg['backends'])} backends configured)",
        )
    except Exception as exc:
        return CheckResult(
            name="backends_yaml",
            passed=False,
            message=f"backends.yaml error: {exc}",
            suggestion="Check backends.yaml syntax",
        )


def check_memory_budget() -> CheckResult:
    """Verify system memory is detectable and sufficient."""
    from agentic_concierge.infrastructure.backends.model_runtime import detect_memory_budget_mb

    budget = detect_memory_budget_mb()
    if budget == 0:
        return CheckResult(
            name="memory_budget",
            passed=False,
            message="Could not detect system memory",
            suggestion="Ensure /proc/meminfo is readable (Linux) or sysctl works (macOS)",
        )
    if budget < 4000:
        return CheckResult(
            name="memory_budget",
            passed=False,
            message=f"Memory budget too low: {budget} MB",
            suggestion="At least 8 GB RAM recommended for running local models",
        )
    return CheckResult(
        name="memory_budget",
        passed=True,
        message=f"Memory budget: {budget} MB ({budget // 1024} GB)",
    )


async def check_backend_registry() -> CheckResult:
    """Probe backends via BackendRegistry and check health."""
    try:
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry

        registry = BackendRegistry.from_config()
        healthy = await registry.discover()

        if not healthy:
            status = registry.status()
            errors = [
                f"{name}: {info['last_error']}"
                for name, info in status.items()
                if info.get("last_error")
            ]
            error_str = "; ".join(errors) if errors else "all backends unreachable"
            return CheckResult(
                name="backend_registry",
                passed=False,
                message=f"No healthy backends: {error_str}",
                suggestion="Start Ollama with: ollama serve",
            )
        return CheckResult(
            name="backend_registry",
            passed=True,
            message=f"Healthy backends: {', '.join(healthy)}",
        )
    except Exception as exc:
        return CheckResult(
            name="backend_registry",
            passed=False,
            message=f"Backend registry error: {exc}",
            suggestion="Check backend configuration in backends.yaml",
        )


async def check_model_availability(registry: Any = None) -> CheckResult:
    """Check if any models are available for loading."""
    try:
        if registry is None:
            from agentic_concierge.infrastructure.backends.registry import BackendRegistry
            registry = BackendRegistry.from_config()
            await registry.discover()

        models: List[str] = []
        for backend in registry.healthy_backends():
            try:
                available = await backend.list_available()
                models.extend(available)
            except Exception:
                pass

        if not models:
            return CheckResult(
                name="model_availability",
                passed=False,
                message="No models available on any backend",
                suggestion="Pull a model: ollama pull qwen2.5:7b",
            )
        return CheckResult(
            name="model_availability",
            passed=True,
            message=f"{len(models)} model(s) available: {', '.join(models[:5])}"
            + (f" (+{len(models)-5} more)" if len(models) > 5 else ""),
        )
    except Exception as exc:
        return CheckResult(
            name="model_availability",
            passed=False,
            message=f"Model check failed: {exc}",
            suggestion="Ensure at least one backend is healthy",
        )


def check_agent_roles() -> CheckResult:
    """Verify all agent roles are loadable."""
    try:
        from agentic_concierge.application.agent_roles import list_roles, get_role
        roles = list_roles()
        for name in roles:
            get_role(name)
        return CheckResult(
            name="agent_roles",
            passed=True,
            message=f"{len(roles)} agent roles defined: {', '.join(roles)}",
        )
    except Exception as exc:
        return CheckResult(
            name="agent_roles",
            passed=False,
            message=f"Agent roles error: {exc}",
        )


# ---------------------------------------------------------------------------
# Full diagnostics
# ---------------------------------------------------------------------------


async def run_diagnostics() -> DiagnosticReport:
    """Run all diagnostic checks and return a report.

    Checks:
    1. Config loadable
    2. backends.yaml valid
    3. Memory budget detection
    4. Backend registry health
    5. Model availability
    6. Agent roles
    """
    from agentic_concierge.infrastructure.backends.model_runtime import detect_memory_budget_mb

    report = DiagnosticReport()

    # Sync checks
    report.checks.append(check_config_loadable())
    report.checks.append(check_backends_yaml())

    mem_check = check_memory_budget()
    report.checks.append(mem_check)
    report.memory_budget_mb = detect_memory_budget_mb()

    # Async checks
    registry_check = await check_backend_registry()
    report.checks.append(registry_check)

    if registry_check.passed:
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry
        registry = BackendRegistry.from_config()
        await registry.discover()
        report.backend_status = registry.status()

        model_check = await check_model_availability(registry)
        report.checks.append(model_check)
    else:
        report.checks.append(CheckResult(
            name="model_availability",
            passed=False,
            message="Skipped — no healthy backends",
            suggestion="Fix backend issues first",
        ))

    report.checks.append(check_agent_roles())

    return report
