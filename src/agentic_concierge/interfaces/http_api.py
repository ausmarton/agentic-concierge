"""HTTP API: FastAPI app wired to execute_task."""

from __future__ import annotations

import asyncio
import collections
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agentic_concierge.application.execute_task import execute_task, resume_execute_task
from agentic_concierge.config import load_config
from agentic_concierge.domain import Task, build_task
from agentic_concierge.infrastructure.chat import build_chat_client
from agentic_concierge.infrastructure.llm_discovery import resolve_llm
from agentic_concierge.infrastructure.telemetry import setup_telemetry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry

from agentic_concierge.infrastructure.approval import HttpApprovalChannel
from agentic_concierge.application.approval import ApprovalDecision

logger = logging.getLogger(__name__)

# Active approval channels keyed by run_id.  Cleaned up via TTL.
_active_channels: dict[str, HttpApprovalChannel] = {}


def _cleanup_stale_channels() -> None:
    """Remove channels that have exceeded their TTL."""
    stale = [rid for rid, ch in _active_channels.items() if ch.is_stale]
    for rid in stale:
        _active_channels.pop(rid, None)


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    config = load_config()
    setup_telemetry(config)
    yield
    _active_channels.clear()


app = FastAPI(title="agentic-concierge", lifespan=_lifespan)


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Optional bearer-token authentication.

    Active only when the ``CONCIERGE_API_KEY`` environment variable is set.
    When active, every endpoint except ``GET /health`` requires an
    ``Authorization: Bearer <key>`` header.  Uses constant-time comparison
    to prevent timing attacks.
    """
    api_key = os.environ.get("CONCIERGE_API_KEY", "").strip()
    if api_key and request.url.path != "/health":
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Authorization: Bearer <key> header required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header[len("Bearer "):]
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Per-IP sliding-window rate limiter
# ---------------------------------------------------------------------------

# Deques of request timestamps (float) keyed by client IP.
_rate_limit_windows: dict = collections.defaultdict(collections.deque)

# Endpoints exempt from rate limiting (health check).
_RATE_LIMIT_EXEMPT = {"/health"}


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    """Optional per-IP sliding-window rate limiting.

    Active when ``CONCIERGE_RATE_LIMIT`` is set to a positive integer (requests
    per minute).  ``GET /health`` is always exempt.  When a client exceeds the
    limit a ``429 Too Many Requests`` response is returned with a
    ``Retry-After`` header.
    """
    limit_str = os.environ.get("CONCIERGE_RATE_LIMIT", "").strip()
    if not limit_str or request.url.path in _RATE_LIMIT_EXEMPT:
        return await call_next(request)

    try:
        limit = int(limit_str)
    except ValueError:
        return await call_next(request)

    if limit <= 0:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    window = _rate_limit_windows[client_ip]
    now = time.monotonic()
    cutoff = now - 60.0  # 1-minute window

    # Remove timestamps older than the window.
    while window and window[0] < cutoff:
        window.popleft()

    if len(window) >= limit:
        retry_after = int(60 - (now - window[0])) + 1
        return JSONResponse(
            status_code=429,
            content={"error": "Too Many Requests", "detail": f"Rate limit: {limit} req/min"},
            headers={"Retry-After": str(retry_after)},
        )

    window.append(now)
    return await call_next(request)


def _workspace_root() -> str:
    return os.environ.get("CONCIERGE_WORKSPACE", ".concierge")


class RunRequest(BaseModel):
    prompt: str
    pack: Optional[str] = None
    model_key: str = "quality"
    network_allowed: bool = True


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run")
async def run(req: RunRequest):
    logger.info(
        "POST /run prompt=%r pack=%s model=%s network=%s",
        req.prompt[:80], req.pack, req.model_key, req.network_allowed,
    )
    config = load_config()

    # resolve_llm performs blocking I/O (HTTP probes, optional subprocess) so we
    # run it on a thread-pool worker to avoid blocking the event loop.
    try:
        resolved = await asyncio.to_thread(resolve_llm, config, req.model_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    chat_client = build_chat_client(resolved.model_config)
    run_repository = FileSystemRunRepository(workspace_root=_workspace_root())
    specialist_registry = ConfigSpecialistRegistry(config)
    specialist_registry.set_runtime_models(resolved.all_chat_models)

    task = build_task(req.prompt, req.pack, req.model_key, req.network_allowed)
    try:
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            resolved_model_cfg=resolved.model_config,
            resolved_llm=resolved,
            max_steps=40,
        )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM server unreachable ({resolved.base_url}): {e}. "
                "Install/start your backend or set local_llm_ensure_available: false."
            ),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=f"Model not found (404). Pull: ollama pull {resolved.model} or set CONCIERGE_CONFIG_PATH.",
            )
        raise HTTPException(
            status_code=503,
            detail=f"LLM server error ({resolved.base_url}): {e.response.status_code}.",
        )

    logger.info(
        "POST /run completed run_id=%s pack=%s model=%s",
        result.run_id.value, result.specialist_id, result.model_name,
    )
    out = dict(result.payload)
    out["_meta"] = {
        "pack": result.specialist_id,
        "specialist_ids": result.specialist_ids,
        "is_task_force": result.is_task_force,
        "run_dir": result.run_dir,
        "workspace": result.workspace_path,
        "model": result.model_name,
        "run_id": result.run_id.value,
        "required_capabilities": result.required_capabilities,
        "warnings": resolved.warnings,
        "fallback_used": resolved.fallback_used,
        "resolved_backend": resolved.resolved_backend,
    }
    return out


# ---------------------------------------------------------------------------
# P8-2: SSE streaming endpoint
# ---------------------------------------------------------------------------

async def _sse_event_generator(
    req: RunRequest,
    event_queue: asyncio.Queue,
    approval_channel: Optional[HttpApprovalChannel] = None,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events from the queue until the run-done sentinel."""
    while True:
        try:
            # Poll with a short timeout so we don't block indefinitely if the
            # background task dies without putting the sentinel.
            event = await asyncio.wait_for(event_queue.get(), timeout=600.0)
        except asyncio.TimeoutError:
            # Yield a keep-alive comment and stop — the run took too long.
            yield ": keep-alive timeout\n\n"
            break

        # Register channel with run_id on first event that has it.
        if approval_channel is not None:
            run_id = (event.get("data") or {}).get("run_id")
            if run_id and run_id not in _active_channels:
                _active_channels[run_id] = approval_channel

        payload = json.dumps(event, ensure_ascii=False)
        yield f"data: {payload}\n\n"

        if event.get("kind") in ("_run_done_", "_run_error_"):
            # Cleanup channel for this run.
            run_id = (event.get("data") or {}).get("run_id")
            if run_id:
                _active_channels.pop(run_id, None)
            break


@app.post("/run/stream")
async def run_stream(req: RunRequest):
    """Stream run events as Server-Sent Events (text/event-stream).

    Each event is a JSON-encoded line in SSE format::

        data: {"kind": "recruitment", "data": {...}, "step": null}\\n\\n
        data: {"kind": "llm_request", ...}\\n\\n
        ...
        data: {"kind": "_run_done_", "data": {"run_id": "...", "ok": true}}\\n\\n

    The stream ends when ``_run_done_`` (success) or ``_run_error_`` (error)
    is received.
    """
    logger.info(
        "POST /run/stream prompt=%r pack=%s model=%s network=%s",
        req.prompt[:80], req.pack, req.model_key, req.network_allowed,
    )
    config = load_config()

    try:
        resolved = await asyncio.to_thread(resolve_llm, config, req.model_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    chat_client = build_chat_client(resolved.model_config)
    run_repository = FileSystemRunRepository(workspace_root=_workspace_root())
    specialist_registry = ConfigSpecialistRegistry(config)
    specialist_registry.set_runtime_models(resolved.all_chat_models)
    task = build_task(req.prompt, req.pack, req.model_key, req.network_allowed)

    # Bounded queue — prevents unbounded memory accumulation if the client reads slowly.
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    # Create HTTP approval channel for this streaming run.
    _cleanup_stale_channels()
    approval_channel = HttpApprovalChannel(
        timeout_s=getattr(config, "approval_timeout_s", 600.0),
    )

    async def _run_task_background() -> None:
        # Register channel once we know the run_id (after execute_task creates it).
        # We pre-register with a placeholder and update in the SSE generator.
        try:
            result = await execute_task(
                task,
                chat_client=chat_client,
                run_repository=run_repository,
                specialist_registry=specialist_registry,
                config=config,
                resolved_model_cfg=resolved.model_config,
                resolved_llm=resolved,
                max_steps=40,
                event_queue=event_queue,
                approval_channel=approval_channel,
            )
            # Register channel with actual run_id for /approve endpoint lookup.
            _active_channels[result.run_id.value] = approval_channel
        except Exception as exc:  # noqa: BLE001
            # Put an error sentinel so the SSE generator terminates cleanly.
            try:
                event_queue.put_nowait({
                    "kind": "_run_error_",
                    "data": {"error": str(exc), "error_type": type(exc).__name__},
                    "step": None,
                })
            except asyncio.QueueFull:
                pass

    asyncio.create_task(_run_task_background())

    return StreamingResponse(
        _sse_event_generator(req, event_queue, approval_channel=approval_channel),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# P8-3: Run status endpoint
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}/status")
async def run_status(run_id: str):
    """Return the status of a run by run_id.

    Response:
    - ``{"status": "completed", "run_id": "...", "specialist_ids": [...]}``
      when a ``run_complete`` event is found in the runlog.
    - ``{"status": "running", "run_id": "..."}``
      when events exist but no ``run_complete`` event yet.
    - 404 when the run_id is not found.
    """
    from agentic_concierge.infrastructure.workspace.run_reader import read_run_events
    from pathlib import Path

    workspace_root = _workspace_root()
    run_dir = str(Path(workspace_root) / "runs" / run_id)

    if not Path(run_dir).is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    try:
        events = list(read_run_events(run_id, workspace_root))
    except FileNotFoundError:
        # run dir exists but no runlog yet — run may still be initializing
        return {"status": "running", "run_id": run_id}

    if not events:
        return {"status": "running", "run_id": run_id}

    for ev in events:
        if ev.get("kind") == "run_complete":
            return {
                "status": "completed",
                "run_id": run_id,
                "specialist_ids": ev.get("data", {}).get("specialist_ids", []),
                "task_force_mode": ev.get("data", {}).get("task_force_mode", "sequential"),
            }

    return {"status": "running", "run_id": run_id}


# ---------------------------------------------------------------------------
# Resume endpoint
# ---------------------------------------------------------------------------


class ResumeRequest(BaseModel):
    run_id: str
    model_key: str = "quality"


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, req: ResumeRequest):
    """Resume a previously interrupted run from its checkpoint.

    Returns the final payload on success, 404 if run/checkpoint not found,
    503 on LLM errors.
    """
    from pathlib import Path
    from agentic_concierge.domain import RunId as _RunId

    logger.info("POST /runs/%s/resume model=%s", run_id, req.model_key)
    config = load_config()
    workspace_root = _workspace_root()
    run_dir = str(Path(workspace_root).resolve() / "runs" / run_id)

    if not Path(run_dir).is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    from agentic_concierge.infrastructure.workspace.graph_checkpoint import load_checkpoint
    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint found for run {run_id!r}. Only interrupted runs can be resumed.",
        )

    try:
        resolved = await asyncio.to_thread(resolve_llm, config, req.model_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    chat_client = build_chat_client(resolved.model_config)
    run_repository = FileSystemRunRepository(workspace_root=workspace_root)
    specialist_registry = ConfigSpecialistRegistry(config)
    specialist_registry.set_runtime_models(resolved.all_chat_models)

    try:
        result = await resume_execute_task(
            _RunId(run_id),
            run_dir,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            resolved_model_cfg=resolved.model_config,
            resolved_llm=resolved,
            max_steps=40,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM server unreachable: {e}",
        )

    out = dict(result.payload)
    out["_meta"] = {
        "pack": result.specialist_id,
        "specialist_ids": result.specialist_ids,
        "run_dir": result.run_dir,
        "model": result.model_name,
        "run_id": result.run_id.value,
        "resumed": True,
    }
    return out


# ---------------------------------------------------------------------------
# Resumable runs list
# ---------------------------------------------------------------------------


@app.get("/runs/resumable")
async def list_resumable_runs():
    """List runs that have checkpoints and can be resumed."""
    from agentic_concierge.infrastructure.workspace.graph_checkpoint import find_resumable_runs
    workspace_root = _workspace_root()
    runs = find_resumable_runs(workspace_root)
    return [
        {
            "run_id": r.run_id,
            "prompt": r.prompt[:200],
            "model_name": r.model_name,
            "specialist_ids": r.specialist_ids,
            "routing_method": r.routing_method,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "total_nodes": r.total_nodes,
            "done_nodes": r.done_nodes,
            "failed_nodes": r.failed_nodes,
            "pending_nodes": r.pending_nodes,
        }
        for r in runs
    ]


# ---------------------------------------------------------------------------
# ADR-021: Human approval endpoint
# ---------------------------------------------------------------------------

class ApproveRequest(BaseModel):
    request_id: str
    approved: bool
    comment: str = ""


@app.post("/runs/{run_id}/approve")
async def approve(run_id: str, req: ApproveRequest):
    """Resolve a pending approval request for a streaming run.

    Returns 200 on success, 404 if no pending approval is found for this
    run_id and request_id combination.
    """
    channel = _active_channels.get(run_id)
    if channel is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active approval channel for run {run_id!r}",
        )
    decision = ApprovalDecision(approved=req.approved, comment=req.comment)
    found = channel.resolve(req.request_id, decision)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No pending approval with request_id={req.request_id!r}",
        )
    return {"ok": True, "approved": req.approved}
