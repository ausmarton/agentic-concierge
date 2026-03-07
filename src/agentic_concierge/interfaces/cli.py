"""CLI: Typer app wired to execute_task."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import httpx
import typer
from rich import print as rprint
from rich.panel import Panel

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.config import load_config
from agentic_concierge.domain import Task, build_task
from agentic_concierge.infrastructure.chat import build_chat_client
from agentic_concierge.infrastructure.llm_discovery import resolve_llm
from agentic_concierge.infrastructure.telemetry import setup_telemetry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry

app = typer.Typer(help="agentic-concierge: on-demand specialist packs (Ollama by default).")


def _workspace_root() -> str:
    return os.environ.get("CONCIERGE_WORKSPACE", ".concierge")


def _render_stream_event(console, event: dict) -> None:  # noqa: ARG001
    """Render a single run event to the terminal during streaming."""
    kind = event.get("kind", "")
    data = event.get("data", {})
    step = event.get("step")
    prefix = f"[dim]{step}[/dim] " if step else ""

    if kind == "model_pull":
        model = data.get("model", "?")
        console.print(f"{prefix}[bold yellow]↓ pulling {model}[/bold yellow] (first time only, may take a minute)...")

    elif kind == "recruitment":
        specialists = data.get("specialist_ids") or [data.get("specialist_id", "?")]
        caps = data.get("required_capabilities", [])
        caps_str = f"  [dim]capabilities: {', '.join(caps)}[/dim]" if caps else ""
        console.print(f"{prefix}[cyan]→ routing[/cyan] {', '.join(specialists)}{caps_str}")

    elif kind == "llm_request":
        step_n = data.get("step", "?")
        msg_count = data.get("message_count", "?")
        console.print(f"{prefix}[dim]◆ step {step_n} — {msg_count} messages[/dim]")

    elif kind == "tool_call":
        tool = data.get("tool", "?")
        args = data.get("args", {})
        # Show a compact one-line summary of the args
        args_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3])
        console.print(f"{prefix}[yellow]⚙ {tool}[/yellow]({args_str})")

    elif kind == "tool_result":
        tool = data.get("tool", "?")
        result = data.get("result", {})
        if isinstance(result, dict) and "error" in result:
            console.print(f"{prefix}[red]✗ {tool}[/red]: {result.get('message', result['error'])}")
        else:
            # Show a short summary of the result
            summary = _result_summary(result)
            console.print(f"{prefix}[green]✓ {tool}[/green]: {summary}")

    elif kind == "tool_error":
        tool = data.get("tool", "?")
        console.print(f"{prefix}[red]✗ {tool}[/red] ({data.get('error_type','error')}): {data.get('error_message','')}")

    elif kind == "security_event":
        console.print(f"{prefix}[bold red]⚠ sandbox violation[/bold red]: {data.get('error_message','')}")

    elif kind == "corrective_reprompt":
        attempt = data.get("attempt", "?")
        max_r = data.get("max_retries", "?")
        console.print(f"{prefix}[yellow]↺ re-prompt ({attempt}/{max_r}): LLM returned text; nudging to use a tool[/yellow]")

    elif kind == "cloud_fallback":
        console.print(f"{prefix}[magenta]☁ cloud fallback[/magenta]: {data.get('reason','?')} → {data.get('cloud_model','?')}")

    elif kind == "approval_requested":
        console.print(
            f"{prefix}[bold yellow]⏸ APPROVAL REQUESTED[/bold yellow]: "
            f"{data.get('action','?')} — {data.get('reason','')}"
        )

    elif kind == "approval_granted":
        console.print(f"{prefix}[green]✓ APPROVED[/green]: {data.get('comment','')}")

    elif kind == "approval_denied":
        console.print(f"{prefix}[red]✗ DENIED[/red]: {data.get('comment','')}")

    elif kind == "delegation_start":
        console.print(
            f"{prefix}[cyan]↳ delegating to {data.get('sub_specialist','?')}[/cyan]: "
            f"{data.get('sub_task','')[:80]}"
        )

    elif kind == "delegation_complete":
        console.print(
            f"{prefix}[cyan]↲ delegation to {data.get('sub_specialist','?')} complete[/cyan]"
        )

    elif kind == "orchestration_plan":
        assignments = data.get("assignments", [])
        model_tier = data.get("model_tier", "")
        tier_str = f" [dim]model_tier={model_tier}[/dim]" if model_tier else ""
        specs = ", ".join(a.get("specialist_id", "?") for a in assignments)
        console.print(f"{prefix}[cyan]◇ plan[/cyan] {specs}{tier_str}")

    elif kind == "pack_start":
        model = data.get("model", "")
        model_str = f" [dim]({model})[/dim]" if model else ""
        console.print(f"{prefix}[bold cyan]◈ pack {data.get('specialist_id','?')}{model_str} starting[/bold cyan]")

    elif kind == "run_complete":
        pass  # Final panel is printed after the loop

    elif kind == "_run_error_":
        console.print(f"[bold red]Run failed[/bold red]: {data.get('error', '?')}")


def _result_summary(result) -> str:
    """Build a short human-readable summary of a tool result dict."""
    if not isinstance(result, dict):
        return repr(result)[:60]
    if "content" in result:
        return f"{len(result['content'])} chars"
    if "files" in result:
        return f"{result.get('count', '?')} files"
    if "returncode" in result:
        rc = result["returncode"]
        out = (result.get("stdout") or "").strip()[:60]
        return f"rc={rc}" + (f" {out!r}" if out else "")
    if "bytes" in result:
        return f"{result['bytes']} bytes → {result.get('path','?')}"
    if "path" in result:
        return result["path"]
    return repr(result)[:60]


async def _run_with_streaming(
    task, chat_client, run_repository, specialist_registry, config, resolved_model_cfg,
    resolved_llm=None, approval_channel=None,
):
    """Run execute_task with an event_queue and render events to the terminal in real-time."""
    from rich.console import Console
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    console = Console()

    bg = asyncio.create_task(
        execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            resolved_model_cfg=resolved_model_cfg,
            resolved_llm=resolved_llm,
            max_steps=40,
            event_queue=event_queue,
            approval_channel=approval_channel,
        )
    )

    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=600.0)
        except asyncio.TimeoutError:
            console.print("[red]Stream timeout — no event for 600 s.[/red]")
            break
        _render_stream_event(console, event)
        if event.get("kind") in ("run_complete", "_run_error_"):
            break

    return await bg


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What you want the fabric to do."),
    pack: str = typer.Option("", help="Force a pack (engineering|research). Leave empty for auto-routing."),
    model_key: str = typer.Option("quality", help="Which model profile to use (quality|fast)."),
    network_allowed: bool = typer.Option(True, help="Allow network tools (web_search, fetch_url)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose (DEBUG) logging to stderr."),
    stream: bool = typer.Option(True, "--stream/--no-stream", "-s", help="Stream events as they happen (default: on)."),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-approve all approval requests (skip interactive prompts)."),
) -> None:
    """Run a task end-to-end and print result + run directory."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = load_config()
    setup_telemetry(config)
    try:
        resolved = resolve_llm(config, model_key)
    except RuntimeError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    for w in resolved.warnings:
        rprint(f"[yellow]Warning: {w}[/yellow]")
    rprint(f"[dim]Using model: {resolved.model} at {resolved.base_url}[/dim]")

    chat_client = build_chat_client(resolved.model_config)
    run_repository = FileSystemRunRepository(workspace_root=_workspace_root())
    specialist_registry = ConfigSpecialistRegistry(config)
    task = build_task(prompt, pack, model_key, network_allowed)

    # Approval channel: interactive CLI prompt unless --auto-approve is set.
    if auto_approve:
        from agentic_concierge.infrastructure.approval import AutoApprovalChannel
        approval_channel = AutoApprovalChannel()
    else:
        from agentic_concierge.infrastructure.approval import CliApprovalChannel
        approval_channel = CliApprovalChannel(
            timeout_s=getattr(config, "approval_timeout_s", 600.0),
        )

    try:
        if stream:
            result = asyncio.run(
                _run_with_streaming(
                    task, chat_client, run_repository, specialist_registry,
                    config, resolved.model_config, resolved_llm=resolved,
                    approval_channel=approval_channel,
                )
            )
        else:
            rprint("[dim]Running task...[/dim]")
            result = asyncio.run(
                execute_task(
                    task,
                    chat_client=chat_client,
                    run_repository=run_repository,
                    specialist_registry=specialist_registry,
                    config=config,
                    resolved_model_cfg=resolved.model_config,
                    resolved_llm=resolved,
                    max_steps=40,
                    approval_channel=approval_channel,
                )
            )
    except httpx.ConnectError as e:
        rprint(
            f"[red]LLM server unreachable.[/red]\n"
            f"  URL: {resolved.base_url}\n  Error: {e}\n"
            "  Install/start your backend (e.g. Ollama: ollama serve) or set local_llm_ensure_available: false."
        )
        sys.exit(1)
    except httpx.ReadTimeout:
        rprint(
            f"[red]LLM read timeout.[/red] The model ({resolved.model}) took too long to respond.\n"
            f"  Use a smaller/faster model or increase timeout in config (models.*.timeout_s)."
        )
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            rprint(
                f"[red]Model not found (404).[/red]\n"
                f"  URL: {resolved.base_url}\n  Model: {resolved.model}\n"
                f"  Pull with: [bold]ollama pull {resolved.model}[/bold] or set CONCIERGE_CONFIG_PATH."
            )
        elif e.response.status_code == 400:
            try:
                err_body = e.response.json()
                detail = (
                    err_body.get("error", {}).get("message")
                    if isinstance(err_body.get("error"), dict)
                    else err_body.get("message", str(err_body))
                )
            except Exception:
                detail = e.response.text or str(e)
            rprint(
                f"[red]LLM server returned 400 Bad Request.[/red]\n"
                f"  URL: {resolved.base_url}\n  Model: {resolved.model}\n  Detail: {detail}"
            )
        else:
            rprint(
                f"[red]LLM server error.[/red]\n"
                f"  URL: {resolved.base_url}\n  Status: {e.response.status_code}\n  {e}"
            )
        sys.exit(1)

    rprint(
        Panel.fit(
            f"[bold]Pack:[/bold] {result.specialist_id}\n"
            f"[bold]Run dir:[/bold] {result.run_dir}\n"
            f"[bold]Workspace:[/bold] {result.workspace_path}\n"
            f"[bold]Model:[/bold] {result.model_name}"
        )
    )
    rprint(json.dumps(result.payload, indent=2, ensure_ascii=False))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the HTTP API (FastAPI + uvicorn)."""
    import uvicorn
    uvicorn.run("agentic_concierge.interfaces.http_api:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# fabric logs subcommands
# ---------------------------------------------------------------------------

logs_app = typer.Typer(help="Inspect past runs.")
app.add_typer(logs_app, name="logs")


@logs_app.command("list")
def logs_list(
    workspace: str = typer.Option(
        ".concierge", "--workspace", "-w", help="Workspace root (default: .fabric)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of runs to show."),
) -> None:
    """List recent runs (most recent first)."""
    import datetime
    from agentic_concierge.infrastructure.workspace.run_reader import list_runs

    summaries = list_runs(workspace, limit=limit)
    if not summaries:
        rprint(f"[dim]No runs found in {workspace}/runs/[/dim]")
        return

    from agentic_concierge.infrastructure.workspace.run_checkpoint import find_resumable_runs
    resumable_ids = set(find_resumable_runs(workspace))

    from rich.table import Table
    table = Table(title=f"Recent runs ({workspace})", show_header=True, header_style="bold")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Started", style="dim")
    table.add_column("Specialists", style="green")
    table.add_column("Events", justify="right")
    table.add_column("Summary", overflow="fold")

    for s in summaries:
        started = (
            datetime.datetime.fromtimestamp(s.first_event_ts).strftime("%Y-%m-%d %H:%M:%S")
            if s.first_event_ts is not None
            else "—"
        )
        specialists = ", ".join(s.specialist_ids) if s.specialist_ids else (s.specialist_id or "—")
        summary = (s.payload_summary or "")[:80]
        run_id_display = s.run_id
        if s.run_id in resumable_ids:
            run_id_display = f"{s.run_id} [dim](resumable)[/dim]"
        table.add_row(run_id_display, started, specialists, str(s.event_count), summary)

    from rich.console import Console
    Console().print(table)


@logs_app.command("show")
def logs_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect."),
    workspace: str = typer.Option(
        ".concierge", "--workspace", "-w", help="Workspace root (default: .fabric)."
    ),
    kinds: str = typer.Option(
        "", "--kinds", "-k",
        help="Comma-separated list of event kinds to show (e.g. 'llm_request,tool_call'). Shows all if empty.",
    ),
) -> None:
    """Show the runlog for a specific run (pretty-printed JSON)."""
    from agentic_concierge.infrastructure.workspace.run_reader import read_run_events
    from rich.syntax import Syntax
    from rich.console import Console

    try:
        events = read_run_events(run_id, workspace)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    filter_kinds = {k.strip() for k in kinds.split(",") if k.strip()} if kinds else None
    shown = [e for e in events if filter_kinds is None or e.get("kind") in filter_kinds]

    console = Console()
    rprint(f"[bold]Run:[/bold] {run_id}  [dim]({len(shown)}/{len(events)} events)[/dim]")
    for ev in shown:
        console.print(Syntax(json.dumps(ev, indent=2, ensure_ascii=False), "json", theme="monokai"))


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full error details."),
) -> None:
    """Check system health: hardware, profile tier, and backend status."""
    import asyncio as _asyncio
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Probe system
    from agentic_concierge.bootstrap.system_probe import probe_system
    from agentic_concierge.bootstrap.model_advisor import advise_profile
    from agentic_concierge.bootstrap.detected import load_detected
    from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus
    from agentic_concierge.config.features import Feature, FeatureSet, PROFILE_FEATURES

    console.print("[dim]Probing system…[/dim]")
    probe = _asyncio.run(probe_system())
    profile = load_detected()
    if profile is None:
        profile = advise_profile(probe)

    # Hardware summary
    gpu_info = (
        ", ".join(f"{d.name} ({d.vram_mb} MB)" for d in probe.gpu_devices)
        or "None detected"
    )
    console.print(Panel.fit(
        f"[bold]Profile:[/bold]   {profile.tier.value.upper()}\n"
        f"[bold]CPU:[/bold]       {probe.cpu_cores} cores ({probe.cpu_arch})\n"
        f"[bold]RAM:[/bold]       {probe.ram_total_mb // 1024} GB total, "
        f"{probe.ram_available_mb // 1024} GB available\n"
        f"[bold]GPU:[/bold]       {gpu_info}\n"
        f"[bold]Disk free:[/bold] {probe.disk_free_mb // 1024} GB\n"
        f"[bold]Internet:[/bold]  {'✓ reachable' if probe.internet_reachable else '✗ unreachable'}",
        title="[bold]System[/bold]",
    ))

    # Feature set
    cfg = load_config()
    feature_set = FeatureSet.from_profile(profile.tier, cfg.features)
    feat_table = Table(show_header=True, header_style="bold")
    feat_table.add_column("Feature")
    feat_table.add_column("Status")
    for f in Feature:
        status = "[green]✓ enabled[/green]" if feature_set.is_enabled(f) else "[dim]✗ disabled[/dim]"
        feat_table.add_row(f.value, status)
    console.print(feat_table)

    # Backend health
    mgr = BackendManager()
    health_map = _asyncio.run(mgr.probe_all(feature_set))
    backend_table = Table(show_header=True, header_style="bold")
    backend_table.add_column("Backend")
    backend_table.add_column("Status")
    backend_table.add_column("Models")
    backend_table.add_column("Hint", overflow="fold")
    for name, h in health_map.items():
        if h.status == BackendStatus.HEALTHY:
            status_str = "[green]● healthy[/green]"
        elif h.status == BackendStatus.DISABLED:
            status_str = "[dim]○ disabled[/dim]"
        else:
            err = f" ({h.error})" if verbose and h.error else ""
            status_str = f"[red]✗ {h.status.value}[/red]{err}"
        models_str = ", ".join(h.models[:3]) or "—"
        hint = h.hint or "—"
        backend_table.add_row(name, status_str, models_str, hint)
    console.print(backend_table)

    # Tools & extras: browser (Playwright) and vector store (ChromaDB)
    import importlib.util as _ilu
    extras_table = Table(show_header=True, header_style="bold")
    extras_table.add_column("Extra")
    extras_table.add_column("Status")
    extras_table.add_column("Install hint", overflow="fold")

    # Detect launcher venv pip for install hints
    from pathlib import Path as _Path
    _launcher_pip = _Path.home() / ".local/share/agentic-concierge/venv/bin/pip"
    _pip_cmd = str(_launcher_pip) if _launcher_pip.exists() else "pip"

    # Browser (Playwright)
    if _ilu.find_spec("playwright") is not None:
        try:
            import playwright as _pw
            pw_ver = getattr(_pw, "__version__", "installed")
        except Exception:
            pw_ver = "installed"
        browser_status = f"[green]✓ available (playwright {pw_ver})[/green]"
        browser_hint = "—"
    else:
        browser_status = "[dim]✗ not installed[/dim]"
        browser_hint = f"{_pip_cmd} install 'agentic-concierge[browser]'"
    extras_table.add_row("browser", browser_status, browser_hint)

    # ChromaDB
    if _ilu.find_spec("chromadb") is not None:
        try:
            import chromadb as _chromadb
            chroma_ver = getattr(_chromadb, "__version__", "installed")
        except Exception:
            chroma_ver = "installed"
        chroma_status = f"[green]✓ available (chromadb {chroma_ver})[/green]"
        chroma_hint = "—"
    else:
        chroma_status = "[dim]✗ not installed[/dim]"
        chroma_hint = f"{_pip_cmd} install 'agentic-concierge[embed]'"
    extras_table.add_row("chromadb", chroma_status, chroma_hint)

    console.print(extras_table)


@app.command("plan")
def plan_cmd(
    prompt: str = typer.Argument(..., help="Task prompt to plan."),
    model_key: str = typer.Option("quality", help="Which model profile to use (quality|fast)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging to stderr."),
) -> None:
    """Show the orchestration plan for a task (does not run it)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    from rich.console import Console
    from rich.panel import Panel
    from agentic_concierge.application.orchestrator import orchestrate_task

    config = load_config()
    try:
        resolved = resolve_llm(config, model_key)
    except RuntimeError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    for w in resolved.warnings:
        rprint(f"[yellow]Warning: {w}[/yellow]")

    chat_client = build_chat_client(resolved.model_config)
    routing_cfg = config.models.get(config.routing_model_key) or resolved.model_config

    try:
        plan = asyncio.run(
            orchestrate_task(
                prompt,
                config,
                chat_client=chat_client,
                model=routing_cfg.model,
            )
        )
    except Exception as e:
        rprint(f"[red]Planning failed: {e}[/red]")
        sys.exit(1)

    console = Console()
    synthesis_str = "yes" if plan.synthesis_required else "no"
    assignments_lines = ""
    for i, assignment in enumerate(plan.specialist_assignments, 1):
        assignments_lines += f"\n  {i}. [bold]{assignment.specialist_id}[/bold]"
        if assignment.brief:
            assignments_lines += f"\n     {assignment.brief}"

    console.print(Panel(
        f"[bold]Mode:[/bold]      {plan.mode}\n"
        f"[bold]Synthesis:[/bold] {synthesis_str}"
        + assignments_lines,
        title="[bold]Orchestration plan[/bold]",
    ))


@app.command("resume")
def resume_cmd(
    run_id_arg: str = typer.Argument(..., help="Run ID to resume.", metavar="RUN_ID"),
    workspace: str = typer.Option(
        ".concierge", "--workspace", "-w", help="Workspace root (default: .concierge)."
    ),
    model_key: str = typer.Option("quality", help="Model profile."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Resume an interrupted run from its checkpoint."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    from pathlib import Path
    from agentic_concierge.infrastructure.workspace.run_checkpoint import load_checkpoint
    from agentic_concierge.application.execute_task import resume_execute_task

    run_dir = str(Path(workspace) / "runs" / run_id_arg)
    checkpoint = load_checkpoint(run_dir)

    if checkpoint is None:
        rprint(f"[red]No checkpoint found for run {run_id_arg!r} in {workspace}.[/red]")
        sys.exit(1)

    completed = checkpoint.completed_specialists
    total = checkpoint.specialist_ids
    remaining = [s for s in total if s not in completed]

    if not remaining:
        rprint(f"[yellow]Run {run_id_arg!r} is already complete (all specialists finished).[/yellow]")
        sys.exit(0)

    rprint(
        f"[cyan]Resuming run {run_id_arg}[/cyan] — "
        f"{len(completed)}/{len(total)} specialists complete, "
        f"continuing from [bold]{remaining[0]}[/bold]..."
    )

    config = load_config()
    setup_telemetry(config)
    try:
        resolved = resolve_llm(config, checkpoint.model_key)
    except RuntimeError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    for w in resolved.warnings:
        rprint(f"[yellow]Warning: {w}[/yellow]")

    chat_client = build_chat_client(resolved.model_config)
    run_repository = FileSystemRunRepository(workspace_root=workspace)
    specialist_registry = ConfigSpecialistRegistry(config)

    try:
        result = asyncio.run(
            resume_execute_task(
                run_id_arg,
                workspace,
                chat_client=chat_client,
                run_repository=run_repository,
                specialist_registry=specialist_registry,
                config=config,
                resolved_model_cfg=resolved.model_config,
                resolved_llm=resolved,
                max_steps=40,
            )
        )
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    rprint(
        Panel.fit(
            f"[bold]Pack:[/bold] {result.specialist_id}\n"
            f"[bold]Run dir:[/bold] {result.run_dir}\n"
            f"[bold]Workspace:[/bold] {result.workspace_path}\n"
            f"[bold]Model:[/bold] {result.model_name}"
        )
    )
    rprint(json.dumps(result.payload, indent=2, ensure_ascii=False))


@app.command("bootstrap")
def bootstrap_cmd(
    profile: str = typer.Option(
        "", "--profile", "-p",
        help="Override detected profile tier (nano|small|medium|large|server).",
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Skip prompts and Rich progress panels."
    ),
) -> None:
    """Detect hardware, select profile, and pull recommended models."""
    import asyncio as _asyncio
    from agentic_concierge.bootstrap import first_run

    force = profile.strip() or None
    try:
        sys_profile = _asyncio.run(
            first_run.run(interactive=not non_interactive, force_profile=force)
        )
        rprint(f"[green]Bootstrap complete.[/green] Profile: [bold]{sys_profile.tier.value}[/bold]")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)


@logs_app.command("search")
def logs_search(
    query: str = typer.Argument(..., help="Search query for past run prompts and summaries."),
    workspace: str = typer.Option(
        ".concierge", "--workspace", "-w", help="Workspace root (default: .fabric)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search past runs by keyword or semantic similarity.

    Uses semantic search (cosine similarity via Ollama embeddings) when
    ``run_index.embedding_model`` is set in config and at least some index
    entries have been embedded.  Falls back to keyword/substring search
    automatically when embeddings are unavailable.
    """
    import datetime
    from agentic_concierge.infrastructure.workspace.run_index import (
        search_index,
        semantic_search_index,
    )
    from rich.table import Table
    from rich.console import Console

    cfg = load_config()
    ri_cfg = cfg.run_index

    if ri_cfg.embedding_model:
        # Use the configured embedding base URL, or derive from the first model.
        embed_base = ri_cfg.embedding_base_url or next(iter(cfg.models.values())).base_url
        results = asyncio.run(
            semantic_search_index(
                workspace, query,
                embedding_model=ri_cfg.embedding_model,
                embedding_base_url=embed_base,
                top_k=limit,
                run_index_config=ri_cfg,
            )
        )
    else:
        results = search_index(workspace, query, limit=limit)

    if not results:
        rprint(f"[dim]No runs matching '{query}' found in {workspace}/run_index.jsonl[/dim]")
        return

    table = Table(
        title=f"Runs matching '{query}' ({workspace})",
        show_header=True, header_style="bold",
    )
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Date", style="dim")
    table.add_column("Specialists", style="green")
    table.add_column("Prompt", overflow="fold")
    table.add_column("Summary", overflow="fold")

    for entry in results:
        date = (
            datetime.datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            if entry.timestamp else "—"
        )
        specialists = ", ".join(entry.specialist_ids) if entry.specialist_ids else "—"
        table.add_row(
            entry.run_id,
            date,
            specialists,
            entry.prompt_prefix[:60],
            (entry.summary or "")[:60],
        )

    Console().print(table)
