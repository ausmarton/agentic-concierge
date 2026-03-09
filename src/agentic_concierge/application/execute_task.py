"""Execute task use case.

Flow: orchestrate specialist(s) → create run → for each specialist, run a
tool loop until finish_task or max_steps; context from earlier packs is
forwarded to later packs so the task force shares progress (sequential mode),
or all packs run concurrently with the same initial prompt (parallel mode).

The orchestrator can assign template packs (engineering, research,
enterprise_research) or compose dynamic packs by selecting tools and roles.

Dependencies are injected (ports only); this module never imports from
``interfaces``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from agentic_concierge.config import ConciergeConfig, ModelConfig
from agentic_concierge.config.constants import MAX_LLM_CONTENT_IN_RUNLOG_CHARS
from agentic_concierge.domain import RecruitError, RunId, RunResult, Task
from agentic_concierge.application.ports import ChatClient, RunRepository, SpecialistRegistry
from agentic_concierge.application.orchestrator import orchestrate_task, OrchestrationPlan, SpecialistBrief
from agentic_concierge.infrastructure.telemetry import get_tracer

if TYPE_CHECKING:
    from agentic_concierge.infrastructure.llm_discovery import ResolvedLLM

logger = logging.getLogger(__name__)


_FINISH_TOOL_RESULT_CONTENT = json.dumps({"ok": True, "status": "task_completed"})

# How many consecutive plain-text (no-tool-call) responses are allowed before
# we give up and treat the text as the final payload.  On each occurrence below
# the limit we inject a corrective re-prompt nudging the LLM back on track.
_MAX_PLAIN_TEXT_RETRIES = 2

# Repetition / loop detection: if the same (tool, args) signature appears this
# many times within the last _LOOP_DETECT_WINDOW calls, inject a loop-break
# warning so the LLM is forced to re-read the error and try something different.
_LOOP_DETECT_WINDOW: int = 8
_LOOP_DETECT_THRESHOLD: int = 2
# Secondary detector: same tool *name* (ignoring args) repeated this many times
# in the last _TOOL_NAME_LOOP_WINDOW calls → semantic loop (e.g. web_search with
# slightly different queries that all return similar results).
_TOOL_NAME_LOOP_WINDOW: int = 12
_TOOL_NAME_LOOP_THRESHOLD: int = 8

# Maximum times a reviewer can reject before the work is accepted with a warning.
_MAX_REVIEW_ITERATIONS: int = 2

# Maximum model escalations per pack loop (0 = disabled).
_MAX_ESCALATIONS: int = 2

# Delegation depth and step limits.
_MAX_DELEGATION_DEPTH: int = 1
_MAX_DELEGATION_STEPS: int = 15
_MIN_DELEGATION_STEPS: int = 3

# Read-only tools the reviewer is allowed to use from the doer's pack.
_SAFE_REVIEWER_TOOLS: frozenset = frozenset({"read_file", "list_files", "run_tests"})

_APPROVE_WORK_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "approve_work",
        "description": (
            "Approve the specialist's work. Call this when the output is "
            "correct and complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "comment": {
                    "type": "string",
                    "description": "Optional comment on the work quality.",
                },
            },
        },
    },
}

_REQUEST_REVISION_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_revision",
        "description": (
            "Reject the specialist's work and request revisions. "
            "Provide specific, actionable critique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "critique": {
                    "type": "string",
                    "description": (
                        "Specific, actionable critique explaining what "
                        "needs to be fixed."
                    ),
                },
            },
            "required": ["critique"],
        },
    },
}

_REQUEST_APPROVAL_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_approval",
        "description": (
            "Request human approval before performing a dangerous or irreversible "
            "action (deploy, push, write to external systems, etc.). The loop will "
            "pause until the human responds. You will receive an approved/denied "
            "result and should proceed or abort accordingly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Short label for the action (e.g. 'deploy to prod').",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you want to perform this action.",
                },
                "command": {
                    "type": "string",
                    "description": "Exact command or API call to execute.",
                },
            },
            "required": ["action", "reason", "command"],
        },
    },
}

_DELEGATE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delegate_to_specialist",
        "description": (
            "Delegate a sub-task to another specialist agent. The sub-specialist "
            "runs in the same workspace and its finish payload is returned to you "
            "as the tool result. Use this when you need capabilities outside your "
            "own tool set (e.g. an engineering agent delegating research)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sub_task": {
                    "type": "string",
                    "description": "Clear description of the sub-task to delegate.",
                },
                "specialist_type": {
                    "type": "string",
                    "description": (
                        "Type of specialist to delegate to (e.g. 'research', "
                        "'engineering', 'enterprise_research', or 'dynamic')."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool names for a dynamic sub-specialist (only used when "
                        "specialist_type='dynamic')."
                    ),
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Role description for a dynamic sub-specialist (only used "
                        "when specialist_type='dynamic')."
                    ),
                },
            },
            "required": ["sub_task", "specialist_type"],
        },
    },
}


def _select_specialist_model(
    assignment: Optional[SpecialistBrief],
    available_models: List[str],
    base_model_cfg: "ModelConfig",
    chat_client: "ChatClient",
) -> tuple["ModelConfig", "ChatClient"]:
    """Select the best model for a specialist based on its task capabilities.

    Returns the (possibly updated) model_cfg and chat_client.  When no better
    model is found or only one model is available, returns the originals unchanged.
    """
    if not available_models or len(available_models) <= 1 or assignment is None:
        return base_model_cfg, chat_client

    from agentic_concierge.infrastructure.model_profiles import (
        infer_task_capabilities,
        match_models,
    )

    # Use explicit capabilities from orchestrator when available;
    # otherwise infer from template/tools.
    if assignment.required_capabilities:
        from agentic_concierge.infrastructure.model_profiles import KNOWN_CAPABILITIES
        valid_caps = [c for c in assignment.required_capabilities if c in KNOWN_CAPABILITIES]
        caps = {c: 0.6 for c in valid_caps} if valid_caps else {}
    else:
        caps = infer_task_capabilities(
            template_id=assignment.specialist_id if assignment.specialist_id != "dynamic" else None,
            tool_names=assignment.tools,
        )
    if not caps:
        return base_model_cfg, chat_client

    best = match_models(available_models, required_capabilities=caps)
    if not best or best == base_model_cfg.model:
        return base_model_cfg, chat_client

    new_cfg = ModelConfig(
        base_url=base_model_cfg.base_url,
        model=best,
        backend=base_model_cfg.backend,
        api_key=base_model_cfg.api_key,
        timeout_s=base_model_cfg.timeout_s,
    )
    new_client = _rebuild_chat_client(new_cfg, chat_client)
    logger.info(
        "Per-specialist model: %s → %s (caps=%s)",
        base_model_cfg.model, best, caps,
    )
    return new_cfg, new_client


def _emit(
    queue: Optional[asyncio.Queue],
    kind: str,
    data: Dict[str, Any],
    step: Optional[str] = None,
) -> None:
    """Put a run event to the streaming queue (no-op when queue is None or full)."""
    if queue is None:
        return
    try:
        queue.put_nowait({"kind": kind, "data": data, "step": step})
    except asyncio.QueueFull:
        logger.debug("event_queue full; dropping event kind=%s", kind)


def _get_brief(plan: Optional[OrchestrationPlan], specialist_id: str) -> str:
    """Return the orchestrator's brief for this specialist, or '' if not available."""
    if plan is None:
        return ""
    for b in plan.specialist_assignments:
        if b.specialist_id == specialist_id:
            return b.brief or ""
    return ""


def _get_assignment(plan: Optional[OrchestrationPlan], specialist_id: str) -> Optional[SpecialistBrief]:
    """Return the full SpecialistBrief for this specialist, or None."""
    if plan is None:
        return None
    for b in plan.specialist_assignments:
        if b.specialist_id == specialist_id:
            return b
    return None


async def execute_task(
    task: Task,
    *,
    chat_client: ChatClient,
    run_repository: RunRepository,
    specialist_registry: SpecialistRegistry,
    config: ConciergeConfig,
    resolved_model_cfg: Optional[ModelConfig] = None,
    resolved_llm: Optional["ResolvedLLM"] = None,
    max_steps: int = 40,
    event_queue: Optional[asyncio.Queue] = None,
    max_review_iterations: int = _MAX_REVIEW_ITERATIONS,
    max_escalations: int = _MAX_ESCALATIONS,
    approval_channel: Optional[Any] = None,
) -> RunResult:
    """Execute a task end-to-end.

    1. Orchestrate specialist(s) via orchestrate_task (falls back to template).
    2. Create a run directory + workspace.
    3. For each specialist, run a tool loop until ``finish_task`` or
       ``max_steps`` is reached.  When multiple specialists are recruited
       (a task force), the finish payload from each pack is forwarded as
       context to the next pack.
    4. Optionally synthesise multi-specialist outputs.
    5. Return a ``RunResult`` with the run id, paths, and final payload.

    Args:
        task: Prompt, optional specialist override, model key, and network flag.
        chat_client: LLM interface (``ChatClient`` port).
        run_repository: Creates run dirs and appends log events (``RunRepository`` port).
        specialist_registry: Resolves pack by id (``SpecialistRegistry`` port).
        config: Fabric configuration (models, specialists, flags).
        resolved_model_cfg: Pre-resolved model config (e.g. from ``resolve_llm``).
            Falls back to ``config.models[task.model_key]`` when not provided.
        max_steps: Maximum LLM turns *per specialist* before aborting.
        event_queue: Optional asyncio.Queue for real-time event streaming (P8-2).
        max_review_iterations: Max reviewer rejections before accepting (0 = disabled).
        max_escalations: Max model escalations per pack loop (0 = disabled).

    Returns:
        ``RunResult`` with payload set to the final pack's ``finish_task`` args
        (or a synthesised result for multi-specialist tasks with synthesis_required).

    Raises:
        RecruitError: When a specialist id is not found in config.
    """
    # --- cloud fallback wrapping -------------------------------------------------
    if config.cloud_fallback:
        cloud_cfg = config.models.get(config.cloud_fallback.model_key)
        if cloud_cfg is None:
            logger.warning(
                "cloud_fallback.model_key %r not found in config.models; cloud fallback disabled",
                config.cloud_fallback.model_key,
            )
        else:
            from agentic_concierge.infrastructure.chat import build_chat_client
            from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient, FallbackPolicy
            cloud_client = build_chat_client(cloud_cfg)
            policy = FallbackPolicy(config.cloud_fallback.policy)
            chat_client = FallbackChatClient(chat_client, cloud_client, cloud_cfg.model, policy)
            logger.debug(
                "Cloud fallback enabled: policy=%s cloud_model=%s",
                config.cloud_fallback.policy, cloud_cfg.model,
            )

    # --- recruit -----------------------------------------------------------------
    model_cfg = resolved_model_cfg or config.models.get(task.model_key) or config.models["quality"]
    available_models: List[str] = resolved_llm.available_models if resolved_llm else []
    plan: Optional[OrchestrationPlan] = None

    if task.specialist_id:
        specialist_ids: List[str] = [task.specialist_id]
        required_capabilities: List[str] = []
        routing_method = "explicit"
    else:
        if resolved_llm is not None:
            from agentic_concierge.infrastructure.llm_discovery import resolve_routing_model
            routing_model = resolve_routing_model(resolved_llm, config)
        else:
            routing_cfg = config.models.get(config.routing_model_key)
            if routing_cfg is None:
                logger.warning(
                    "routing_model_key %r not in config.models; using task model %r for routing",
                    config.routing_model_key, model_cfg.model,
                )
                routing_cfg = model_cfg
            routing_model = routing_cfg.model
        plan = await orchestrate_task(
            task.prompt, config,
            chat_client=chat_client,
            model=routing_model,
        )
        specialist_ids = [a.specialist_id for a in plan.specialist_assignments]
        required_capabilities = plan.required_capabilities
        routing_method = plan.routing_method

        # --- model tier selection (ADR-023) ---
        # The orchestrator recommends "fast" or "quality".  When "fast" is
        # recommended and we have a fast model config, resolve the best
        # available model for that tier.  If the fast model isn't available
        # and auto_pull is enabled, pull it.  Adaptive escalation (ADR-020)
        # will upgrade to a larger model if the fast one can't handle the task.
        if plan.recommended_model_key != task.model_key:
            tier_cfg = config.models.get(plan.recommended_model_key)
            if tier_cfg is not None:
                preferred = tier_cfg.model
                if preferred in available_models:
                    tier_model = preferred
                elif config.auto_pull_if_missing and tier_cfg.backend == "ollama":
                    # The fast model isn't available — pull it.
                    from agentic_concierge.infrastructure.llm_discovery import (
                        _ollama_pull, _ollama_root, discover_ollama_models,
                        _is_ollama_chat_capable, select_model as _select_model,
                    )
                    logger.info(
                        "Fast model %s not available; pulling it now...", preferred,
                    )
                    _emit(event_queue, "model_pull", {"model": preferred})
                    ollama_root = _ollama_root(tier_cfg.base_url)
                    if _ollama_pull(preferred, ollama_root, timeout_s=600):
                        # Re-discover after pull
                        fresh = discover_ollama_models(tier_cfg.base_url, timeout_s=15.0)
                        fresh = [m for m in (fresh or []) if _is_ollama_chat_capable(m)]
                        tier_model = _select_model(preferred, fresh, is_ollama=True)
                        if tier_model:
                            available_models.append(tier_model)
                        else:
                            tier_model = None
                    else:
                        logger.warning("Failed to pull %s; keeping %s", preferred, model_cfg.model)
                        tier_model = None
                else:
                    tier_model = None

                if tier_model is not None and tier_model != model_cfg.model:
                    old_model = model_cfg.model
                    model_cfg = tier_cfg.model_copy(update={"model": tier_model})
                    from agentic_concierge.infrastructure.chat import build_chat_client
                    chat_client = build_chat_client(model_cfg)
                    logger.info(
                        "Model tier: using %s for this task (was %s)",
                        tier_model, old_model,
                    )
                elif tier_model is None:
                    logger.info(
                        "Model tier %r: no suitable model; keeping %s",
                        plan.recommended_model_key, model_cfg.model,
                    )

    for sid in specialist_ids:
        if sid != "dynamic" and sid not in config.specialists:
            raise RecruitError(f"Unknown specialist: {sid!r}")

    # --- setup -------------------------------------------------------------------
    run_id, run_dir, workspace_path = run_repository.create_run()
    is_task_force = len(specialist_ids) > 1
    logger.info(
        "Task started: specialists=%s run_id=%s task_force=%s",
        specialist_ids, run_id.value, is_task_force,
    )

    task_force_mode = config.task_force_mode if is_task_force else "sequential"
    # Allow orchestrator to override task_force_mode for multi-specialist runs
    if plan is not None and is_task_force and plan.mode in ("sequential", "parallel"):
        task_force_mode = plan.mode

    # Log recruitment event
    _recruitment_event = {
        "specialist_id": specialist_ids[0],
        "specialist_ids": specialist_ids,
        "required_capabilities": required_capabilities,
        "routing_method": routing_method,
        "is_task_force": is_task_force,
        "model": model_cfg.model,
        "model_tier": plan.recommended_model_key if plan else task.model_key,
    }
    run_repository.append_event(run_id, "recruitment", _recruitment_event, step=None)
    _emit(event_queue, "recruitment", _recruitment_event)

    # Emit orchestration_plan event when orchestrator was used
    if plan is not None and plan.routing_method == "orchestrator":
        _orch_plan_ev = {
            "assignments": [
                {"specialist_id": b.specialist_id, "brief": b.brief}
                for b in plan.specialist_assignments
            ],
            "mode": plan.mode,
            "synthesis_required": plan.synthesis_required,
            "reasoning": plan.reasoning,
            "model_tier": plan.recommended_model_key,
        }
        run_repository.append_event(run_id, "orchestration_plan", _orch_plan_ev, step=None)
        _emit(event_queue, "orchestration_plan", _orch_plan_ev)

    # Create initial checkpoint (non-fatal; failure logs a warning)
    _checkpoint = _create_initial_checkpoint(
        run_id=run_id.value,
        run_dir=run_dir,
        workspace_path=workspace_path,
        task=task,
        specialist_ids=specialist_ids,
        task_force_mode=task_force_mode,
        model_cfg=model_cfg,
        routing_method=routing_method,
        required_capabilities=required_capabilities,
        plan=plan,
    )

    # --- pack loop ---------------------------------------------------------------
    tracer = get_tracer()
    final_payload: Dict[str, Any] = {}

    with tracer.start_as_current_span("concierge.execute_task") as root_span:
        root_span.set_attribute("run_id", run_id.value)
        root_span.set_attribute("specialist_ids", ",".join(specialist_ids))
        root_span.set_attribute("is_task_force", is_task_force)
        root_span.set_attribute("routing_method", routing_method)
        root_span.set_attribute("task_force_mode", task_force_mode)

        if task_force_mode == "parallel" and len(specialist_ids) > 1:
            # Parallel: all packs run concurrently; each gets the original prompt.
            tf_event = {"specialist_ids": specialist_ids, "mode": "parallel"}
            run_repository.append_event(run_id, "task_force_parallel", tf_event, step=None)
            _emit(event_queue, "task_force_parallel", tf_event)
            logger.info(
                "Task force PARALLEL: specialists=%s run_id=%s",
                specialist_ids, run_id.value,
            )
            final_payload = await _run_task_force_parallel(
                specialist_ids=specialist_ids,
                task=task,
                workspace_path=workspace_path,
                model_cfg=model_cfg,
                chat_client=chat_client,
                run_repository=run_repository,
                run_id=run_id,
                max_steps=max_steps,
                tracer=tracer,
                event_queue=event_queue,
                specialist_registry=specialist_registry,
                plan=plan,
                available_models=available_models,
                max_review_iterations=max_review_iterations,
                max_escalations=max_escalations,
                approval_channel=approval_channel,
                config=config,
            )

            # Update checkpoint after all parallel specialists complete
            _update_checkpoint(
                _checkpoint, run_dir,
                completed=specialist_ids,
                payloads=final_payload.get("pack_results", {}),
            )

            # Synthesis step for parallel mode
            if plan is not None and plan.synthesis_required:
                try:
                    final_payload = await _synthesise_results(
                        original_prompt=task.prompt,
                        specialist_payloads=final_payload.get("pack_results", {}),
                        chat_client=chat_client,
                        model_cfg=model_cfg,
                        run_repository=run_repository,
                        run_id=run_id,
                        event_queue=event_queue,
                    )
                except Exception as exc:
                    logger.warning("Synthesis failed (%s); using merged parallel result", exc)

        else:
            # Sequential: each pack receives context from the previous pack.
            prev_finish_payload: Optional[Dict[str, Any]] = None
            all_specialist_payloads: Dict[str, Any] = {}

            for pack_idx, specialist_id in enumerate(specialist_ids):
                assignment = _get_assignment(plan, specialist_id)
                pack = specialist_registry.get_pack(
                    specialist_id, workspace_path, task.network_allowed,
                    tools=assignment.tools if assignment else None,
                    role=assignment.role if assignment else None,
                    finish_schema_key=assignment.finish_schema if assignment else None,
                )

                # Per-specialist model selection (ADR-024)
                spec_model_cfg, spec_chat_client = _select_specialist_model(
                    assignment, available_models, model_cfg, chat_client,
                )

                if is_task_force:
                    pack_start_ev = {"specialist_id": specialist_id, "pack_index": pack_idx}
                    run_repository.append_event(run_id, "pack_start", pack_start_ev, step=None)
                    _emit(event_queue, "pack_start", pack_start_ev)
                    logger.info(
                        "Task force SEQUENTIAL: starting pack %d/%d specialist=%s run_id=%s",
                        pack_idx + 1, len(specialist_ids), specialist_id, run_id.value,
                    )

                # Build initial messages for this pack.
                if prev_finish_payload is None:
                    user_content = f"Task:\n{task.prompt}"
                else:
                    prev_specialist_id = specialist_ids[pack_idx - 1]
                    context_block = json.dumps(prev_finish_payload, indent=2, ensure_ascii=False)
                    user_content = (
                        f"Task:\n{task.prompt}\n\n"
                        f"Context from '{prev_specialist_id}' specialist "
                        f"(prior task-force member):\n{context_block}"
                    )

                # Inject orchestrator brief if available
                brief_text = _get_brief(plan, specialist_id)
                if brief_text:
                    user_content += f"\n\nYour specific assignment:\n{brief_text}"

                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": pack.system_prompt},
                    {"role": "user", "content": user_content},
                ]

                step_prefix = f"{specialist_id}_" if is_task_force else ""

                pack_payload = await _execute_pack_loop(
                    pack=pack,
                    messages=messages,
                    model_cfg=spec_model_cfg,
                    chat_client=spec_chat_client,
                    run_repository=run_repository,
                    run_id=run_id,
                    step_prefix=step_prefix,
                    max_steps=max_steps,
                    tracer=tracer,
                    specialist_id=specialist_id,
                    event_queue=event_queue,
                    available_models=available_models,
                    max_review_iterations=max_review_iterations,
                    max_escalations=max_escalations,
                    approval_channel=approval_channel,
                    delegation_depth=_MAX_DELEGATION_DEPTH,
                    specialist_registry=specialist_registry,
                    workspace_path=workspace_path,
                    config=config,
                    finish_schema_key=assignment.finish_schema if assignment else None,
                )

                all_specialist_payloads[specialist_id] = pack_payload
                prev_finish_payload = pack_payload
                final_payload = pack_payload

                # Update checkpoint after each specialist completes
                _update_checkpoint(
                    _checkpoint, run_dir,
                    completed=list(all_specialist_payloads.keys()),
                    payloads=dict(all_specialist_payloads),
                )

            # Synthesis step for sequential mode
            if plan is not None and plan.synthesis_required and len(all_specialist_payloads) > 1:
                try:
                    final_payload = await _synthesise_results(
                        original_prompt=task.prompt,
                        specialist_payloads=all_specialist_payloads,
                        chat_client=chat_client,
                        model_cfg=model_cfg,
                        run_repository=run_repository,
                        run_id=run_id,
                        event_queue=event_queue,
                    )
                except Exception as exc:
                    logger.warning("Synthesis failed (%s); using last specialist result", exc)

    logger.info(
        "Task completed: run_id=%s specialists=%s is_task_force=%s mode=%s",
        run_id.value, specialist_ids, is_task_force, task_force_mode,
    )

    # Write a "run_complete" event
    _run_complete_ev = {
        "run_id": run_id.value,
        "specialist_ids": specialist_ids,
        "task_force_mode": task_force_mode,
    }
    run_repository.append_event(run_id, "run_complete", _run_complete_ev, step=None)
    _emit(event_queue, "run_complete", _run_complete_ev)

    # Delete checkpoint on successful completion
    _delete_run_checkpoint(run_dir)

    result = RunResult(
        run_id=run_id,
        run_dir=run_dir,
        workspace_path=workspace_path,
        specialist_id=specialist_ids[0],
        model_name=model_cfg.model,
        payload=final_payload,
        required_capabilities=required_capabilities,
        specialist_ids=specialist_ids,
    )

    # Append to the cross-run index (failure is non-fatal).
    try:
        from agentic_concierge.infrastructure.workspace.run_index import (
            RunIndexEntry,
            append_to_index,
            embed_text,
        )
        import time
        workspace_root = str(_run_dir_to_workspace_root(run_dir))
        summary_text = (
            final_payload.get("summary")
            or final_payload.get("executive_summary")
            or ""
        )
        entry = RunIndexEntry(
            run_id=run_id.value,
            timestamp=time.time(),
            specialist_ids=specialist_ids,
            prompt_prefix=task.prompt[:200],
            summary=summary_text,
            workspace_path=workspace_path,
            run_dir=run_dir,
            routing_method=routing_method,
            model_name=model_cfg.model,
        )

        run_index_cfg = config.run_index
        if run_index_cfg.embedding_model:
            embed_base = run_index_cfg.embedding_base_url or model_cfg.base_url
            try:
                embed_input = f"{task.prompt[:200]} {summary_text}".strip()
                entry.embedding = await embed_text(
                    embed_input,
                    run_index_cfg.embedding_model,
                    embed_base,
                )
                logger.debug(
                    "RunIndex: embedded entry for run %s (model=%s dims=%d)",
                    run_id.value, run_index_cfg.embedding_model, len(entry.embedding),
                )
            except Exception as exc:
                logger.warning(
                    "RunIndex: embedding failed for run %s (%s); index entry written without embedding",
                    run_id.value, exc,
                )

        append_to_index(workspace_root, entry, run_index_config=config.run_index)
    except Exception as exc:
        logger.warning("Failed to append to run index: %s", exc)

    _emit(event_queue, "_run_done_", {"run_id": result.run_id.value, "ok": True})

    return result


# ---------------------------------------------------------------------------
# Session continuation: resume an interrupted run
# ---------------------------------------------------------------------------

async def resume_execute_task(
    run_id: str,
    workspace_root: str,
    *,
    chat_client: ChatClient,
    run_repository: RunRepository,
    specialist_registry: SpecialistRegistry,
    config: ConciergeConfig,
    resolved_model_cfg: Optional[ModelConfig] = None,
    resolved_llm: Optional["ResolvedLLM"] = None,
    max_steps: int = 40,
    event_queue: Optional[asyncio.Queue] = None,
    max_review_iterations: int = _MAX_REVIEW_ITERATIONS,
    max_escalations: int = _MAX_ESCALATIONS,
    approval_channel: Optional[Any] = None,
) -> RunResult:
    """Resume an interrupted run from its checkpoint.

    Loads ``{workspace_root}/runs/{run_id}/checkpoint.json``, skips already-completed
    specialists, seeds ``prev_finish_payload`` from the last completed specialist's
    payload, and runs the remaining specialists through the existing sequential loop.

    Args:
        run_id: The run ID to resume.
        workspace_root: Root of the workspace (contains ``runs/``).
        chat_client: LLM interface.
        run_repository: Run log and event repository.
        specialist_registry: Resolves packs by specialist ID.
        config: Concierge configuration.
        resolved_model_cfg: Pre-resolved model config; falls back to checkpoint's model_key.
        max_steps: Maximum LLM turns per specialist.
        event_queue: Optional asyncio.Queue for streaming.
        max_escalations: Max model escalations per pack loop (0 = disabled).

    Returns:
        ``RunResult`` with the final payload from the resumed run.

    Raises:
        ValueError: When no checkpoint is found or the run is already complete.
    """
    from pathlib import Path
    from agentic_concierge.infrastructure.workspace.run_checkpoint import (
        load_checkpoint,
        save_checkpoint,
        delete_checkpoint,
    )
    from agentic_concierge.domain import RunId as _RunId

    run_dir = str(Path(workspace_root) / "runs" / run_id)
    checkpoint = load_checkpoint(run_dir)
    if checkpoint is None:
        raise ValueError(f"No checkpoint found for run {run_id!r}")

    remaining_specialists = [
        s for s in checkpoint.specialist_ids
        if s not in checkpoint.completed_specialists
    ]
    if not remaining_specialists:
        raise ValueError(f"Run {run_id!r} is already complete (all specialists finished)")

    run_id_obj = _RunId(value=checkpoint.run_id)
    specialist_ids = checkpoint.specialist_ids
    model_cfg = (
        resolved_model_cfg
        or config.models.get(checkpoint.model_key)
        or config.models["quality"]
    )

    # Reconstruct orchestration plan from checkpoint if available
    plan: Optional[OrchestrationPlan] = None
    if checkpoint.orchestration_plan is not None:
        from agentic_concierge.application.orchestrator import SpecialistBrief
        assignments = [
            SpecialistBrief(a["specialist_id"], a.get("brief", ""))
            for a in checkpoint.orchestration_plan.get("assignments", [])
        ]
        plan = OrchestrationPlan(
            specialist_assignments=assignments,
            mode=checkpoint.orchestration_plan.get("mode", "sequential"),
            synthesis_required=checkpoint.orchestration_plan.get("synthesis_required", False),
            reasoning=checkpoint.orchestration_plan.get("reasoning", ""),
            routing_method=checkpoint.routing_method,
            required_capabilities=checkpoint.required_capabilities,
        )

    tracer = get_tracer()
    final_payload: Dict[str, Any] = {}
    all_specialist_payloads: Dict[str, Any] = dict(checkpoint.payloads)

    # Seed prev_finish_payload from last completed specialist
    prev_finish_payload: Optional[Dict[str, Any]] = None
    if checkpoint.completed_specialists:
        last_completed = checkpoint.completed_specialists[-1]
        prev_finish_payload = checkpoint.payloads.get(last_completed)

    is_task_force = len(specialist_ids) > 1

    with tracer.start_as_current_span("concierge.resume_execute_task") as root_span:
        root_span.set_attribute("run_id", run_id)
        root_span.set_attribute("remaining_specialists", ",".join(remaining_specialists))

        for pack_idx, specialist_id in enumerate(specialist_ids):
            if specialist_id in checkpoint.completed_specialists:
                # Skip already-completed specialists
                prev_finish_payload = checkpoint.payloads.get(specialist_id, prev_finish_payload)
                continue

            pack = specialist_registry.get_pack(specialist_id, checkpoint.workspace_path, True)

            if is_task_force:
                pack_start_ev = {
                    "specialist_id": specialist_id,
                    "pack_index": pack_idx,
                    "resumed": True,
                }
                run_repository.append_event(run_id_obj, "pack_start", pack_start_ev, step=None)
                _emit(event_queue, "pack_start", pack_start_ev)

            # Build messages
            if prev_finish_payload is None:
                user_content = f"Task:\n{checkpoint.task_prompt}"
            else:
                prev_idx = pack_idx - 1
                prev_sid = specialist_ids[prev_idx] if prev_idx >= 0 else "previous"
                context_block = json.dumps(prev_finish_payload, indent=2, ensure_ascii=False)
                user_content = (
                    f"Task:\n{checkpoint.task_prompt}\n\n"
                    f"Context from '{prev_sid}' specialist "
                    f"(prior task-force member):\n{context_block}"
                )

            brief_text = _get_brief(plan, specialist_id)
            if brief_text:
                user_content += f"\n\nYour specific assignment:\n{brief_text}"

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": pack.system_prompt},
                {"role": "user", "content": user_content},
            ]

            step_prefix = f"{specialist_id}_" if is_task_force else ""

            # Resume path: finish_schema_key not available from checkpoint.
            pack_payload = await _execute_pack_loop(
                pack=pack,
                messages=messages,
                model_cfg=model_cfg,
                chat_client=chat_client,
                run_repository=run_repository,
                run_id=run_id_obj,
                step_prefix=step_prefix,
                max_steps=max_steps,
                tracer=tracer,
                specialist_id=specialist_id,
                event_queue=event_queue,
                available_models=resolved_llm.available_models if resolved_llm else [],
                max_review_iterations=max_review_iterations,
                max_escalations=max_escalations,
                approval_channel=approval_channel,
                delegation_depth=_MAX_DELEGATION_DEPTH,
                specialist_registry=specialist_registry,
                workspace_path=checkpoint.workspace_path,
                config=config,
            )

            all_specialist_payloads[specialist_id] = pack_payload
            prev_finish_payload = pack_payload
            final_payload = pack_payload

            # Update checkpoint
            try:
                import time
                checkpoint.completed_specialists = checkpoint.completed_specialists + [specialist_id]
                checkpoint.payloads = dict(all_specialist_payloads)
                checkpoint.updated_at = time.time()
                save_checkpoint(run_dir, checkpoint)
            except Exception as exc:
                logger.warning("Failed to update checkpoint after specialist %s: %s", specialist_id, exc)

    # Synthesis step
    if plan is not None and plan.synthesis_required and len(all_specialist_payloads) > 1:
        try:
            final_payload = await _synthesise_results(
                original_prompt=checkpoint.task_prompt,
                specialist_payloads=all_specialist_payloads,
                chat_client=chat_client,
                model_cfg=model_cfg,
                run_repository=run_repository,
                run_id=run_id_obj,
                event_queue=event_queue,
            )
        except Exception as exc:
            logger.warning("Synthesis failed during resume (%s); using last specialist result", exc)

    # Emit run_complete
    _run_complete_ev = {
        "run_id": run_id,
        "specialist_ids": specialist_ids,
        "task_force_mode": checkpoint.task_force_mode,
        "resumed": True,
    }
    run_repository.append_event(run_id_obj, "run_complete", _run_complete_ev, step=None)
    _emit(event_queue, "run_complete", _run_complete_ev)

    # Delete checkpoint
    try:
        delete_checkpoint(run_dir)
    except Exception as exc:
        logger.warning("Failed to delete checkpoint after resume: %s", exc)

    result = RunResult(
        run_id=run_id_obj,
        run_dir=checkpoint.run_dir,
        workspace_path=checkpoint.workspace_path,
        specialist_id=specialist_ids[0],
        model_name=model_cfg.model,
        payload=final_payload,
        required_capabilities=checkpoint.required_capabilities,
        specialist_ids=specialist_ids,
    )

    _emit(event_queue, "_run_done_", {"run_id": result.run_id.value, "ok": True})
    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers (non-fatal; failure logs a warning)
# ---------------------------------------------------------------------------

def _create_initial_checkpoint(
    *,
    run_id: str,
    run_dir: str,
    workspace_path: str,
    task: Any,
    specialist_ids: List[str],
    task_force_mode: str,
    model_cfg: ModelConfig,
    routing_method: str,
    required_capabilities: List[str],
    plan: Optional[OrchestrationPlan],
) -> Optional[Any]:
    """Create and save the initial checkpoint. Returns the checkpoint object or None on error."""
    try:
        import time
        from agentic_concierge.infrastructure.workspace.run_checkpoint import (
            RunCheckpoint,
            save_checkpoint,
        )

        orch_plan_dict = None
        if plan is not None and plan.routing_method == "orchestrator":
            orch_plan_dict = {
                "assignments": [
                    {"specialist_id": b.specialist_id, "brief": b.brief}
                    for b in plan.specialist_assignments
                ],
                "mode": plan.mode,
                "synthesis_required": plan.synthesis_required,
                "reasoning": plan.reasoning,
            }

        checkpoint = RunCheckpoint(
            run_id=run_id,
            run_dir=run_dir,
            workspace_path=workspace_path,
            task_prompt=task.prompt,
            specialist_ids=specialist_ids,
            completed_specialists=[],
            payloads={},
            task_force_mode=task_force_mode,
            model_key=task.model_key,
            routing_method=routing_method,
            required_capabilities=required_capabilities,
            orchestration_plan=orch_plan_dict,
            created_at=time.time(),
            updated_at=time.time(),
        )
        save_checkpoint(run_dir, checkpoint)
        return checkpoint
    except Exception as exc:
        logger.warning("Failed to create initial checkpoint: %s", exc)
        return None


def _update_checkpoint(
    checkpoint: Optional[Any],
    run_dir: str,
    *,
    completed: List[str],
    payloads: Dict[str, Any],
) -> None:
    """Update the checkpoint's completed_specialists and payloads (non-fatal)."""
    if checkpoint is None:
        return
    try:
        import time
        from agentic_concierge.infrastructure.workspace.run_checkpoint import save_checkpoint

        checkpoint.completed_specialists = list(completed)
        checkpoint.payloads = dict(payloads)
        checkpoint.updated_at = time.time()
        save_checkpoint(run_dir, checkpoint)
    except Exception as exc:
        logger.warning("Failed to update checkpoint: %s", exc)


def _delete_run_checkpoint(run_dir: str) -> None:
    """Delete the run checkpoint (non-fatal)."""
    try:
        from agentic_concierge.infrastructure.workspace.run_checkpoint import delete_checkpoint
        delete_checkpoint(run_dir)
    except Exception as exc:
        logger.warning("Failed to delete checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

_SYNTHESISE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "synthesise_results",
        "description": (
            "Synthesise the outputs from multiple specialist agents into a coherent "
            "final answer. Call this tool once with the combined result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Coherent overall summary combining all specialist outputs.",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key findings from all specialists.",
                },
                "artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to artefacts produced by specialists.",
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recommended next steps.",
                },
            },
            "required": ["summary", "key_findings"],
        },
    },
}


async def _synthesise_results(
    *,
    original_prompt: str,
    specialist_payloads: Dict[str, Any],
    chat_client: ChatClient,
    model_cfg: ModelConfig,
    run_repository: RunRepository,
    run_id: RunId,
    event_queue: Optional[asyncio.Queue],
) -> Dict[str, Any]:
    """Make one LLM call to synthesise multi-specialist outputs.

    Returns the synthesis payload dict, or raises on LLM failure (caller catches).
    """
    payload_lines = "\n\n".join(
        f"**{sid}**:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        for sid, payload in specialist_payloads.items()
        if not isinstance(payload, BaseException)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a synthesis agent. Combine the outputs of multiple specialist agents "
                "into a coherent, concise final answer for the original task."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task:\n{original_prompt}\n\n"
                f"Specialist outputs:\n{payload_lines}\n\n"
                "Call synthesise_results with a coherent synthesis."
            ),
        },
    ]

    _synth_req_ev = {"step": "synthesis", "message_count": len(messages)}
    run_repository.append_event(run_id, "llm_request", _synth_req_ev, step="synthesis")
    _emit(event_queue, "llm_request", _synth_req_ev, step="synthesis")

    response = await chat_client.chat(
        messages=messages,
        model=model_cfg.model,
        tools=[_SYNTHESISE_TOOL_DEF],
        temperature=0.0,
        max_tokens=model_cfg.max_tokens,
    )

    if response.tool_calls:
        tc = response.tool_calls[0]
        if tc.tool_name == "synthesise_results":
            payload = {"action": "final", **tc.arguments}
            _synth_ev = {"step": "synthesis", "result": "tool_call"}
            run_repository.append_event(run_id, "synthesis_complete", _synth_ev, step="synthesis")
            _emit(event_queue, "synthesis_complete", _synth_ev, step="synthesis")
            return payload

    # Fallback: use text content as summary
    logger.warning("Synthesis LLM call returned no tool call; using text response as summary")
    return {
        "action": "final",
        "summary": response.content or "Synthesis produced no output.",
        "key_findings": [],
        "artifacts": [],
        "next_steps": [],
    }


# ---------------------------------------------------------------------------
# Independent reviewer (Gate 4)
# ---------------------------------------------------------------------------

async def _review_specialist_work(
    *,
    finish_payload: Dict[str, Any],
    pack: Any,
    model_cfg: ModelConfig,
    chat_client: ChatClient,
    run_repository: RunRepository,
    run_id: RunId,
    specialist_id: str,
    event_queue: Optional[asyncio.Queue],
    review_iteration: int,
    reviewer_model_cfg: Optional[ModelConfig] = None,
    available_models: Optional[List[str]] = None,
    finish_schema_key: Optional[str] = None,
) -> tuple:
    """Independently review a specialist's finish_task payload.

    Runs a lightweight reviewer mini-loop (max 5 steps).  The reviewer can
    inspect the workspace with read-only tools, then must call ``approve_work``
    or ``request_revision``.  Plain text or reaching max steps without a
    decision tool is treated as implicit approval (fail-open).

    When ``reviewer_model_cfg`` is provided, a separate chat client is built
    for the reviewer.  Otherwise falls back to the doer's model.

    Returns:
        ``(approved: bool, comment_or_critique: str)``
    """
    from agentic_concierge.infrastructure.specialists.prompts import (
        PROMPT_REVIEWER,
        PROMPT_REVIEWER_QUICK_ANSWER,
    )

    reviewer_prompt = (
        PROMPT_REVIEWER_QUICK_ANSWER
        if finish_schema_key == "quick_answer"
        else PROMPT_REVIEWER
    )

    # Select reviewer model: prefer a different model with reasoning capability
    rev_model_cfg = reviewer_model_cfg or model_cfg
    rev_client = chat_client
    if reviewer_model_cfg is None and available_models:
        try:
            from agentic_concierge.infrastructure.model_profiles import match_models
            reviewer_model = match_models(
                available_models,
                required_capabilities={"reasoning": 0.7, "instruction_following": 0.6},
                exclude_models=[model_cfg.model],
                prefer_smaller=True,
            )
            if reviewer_model and reviewer_model != model_cfg.model:
                rev_model_cfg = ModelConfig(
                    base_url=model_cfg.base_url,
                    model=reviewer_model,
                    backend=model_cfg.backend,
                    api_key=model_cfg.api_key,
                    temperature=0.0,
                    top_p=model_cfg.top_p,
                    max_tokens=model_cfg.max_tokens,
                    timeout_s=model_cfg.timeout_s,
                )
                rev_client = _rebuild_chat_client(rev_model_cfg, chat_client)
                logger.info(
                    "Reviewer using different model: %s (doer=%s)",
                    reviewer_model, model_cfg.model,
                )
        except Exception as exc:
            logger.debug("Reviewer model selection failed (%s); using doer model", exc)

    # Emit review_start event
    _rev_start = {
        "specialist_id": specialist_id,
        "review_iteration": review_iteration,
        "reviewer_model": rev_model_cfg.model,
    }
    run_repository.append_event(run_id, "review_start", _rev_start, step=None)
    _emit(event_queue, "review_start", _rev_start)

    # Build reviewer tool set: decision tools + safe read-only tools from doer
    reviewer_tools = [_APPROVE_WORK_TOOL_DEF, _REQUEST_REVISION_TOOL_DEF]
    for tool_def in pack.tool_definitions:
        if tool_def["function"]["name"] in _SAFE_REVIEWER_TOOLS:
            reviewer_tools.append(tool_def)

    # Build reviewer messages
    payload_json = json.dumps(finish_payload, indent=2, ensure_ascii=False)
    rev_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": reviewer_prompt},
        {
            "role": "user",
            "content": (
                f"Review this specialist's claimed output:\n\n{payload_json}"
            ),
        },
    ]

    _MAX_REVIEWER_STEPS = 5
    for _rev_step in range(_MAX_REVIEWER_STEPS):
        try:
            response = await rev_client.chat(
                messages=rev_messages,
                model=rev_model_cfg.model,
                tools=reviewer_tools,
                temperature=0.0,
                max_tokens=rev_model_cfg.max_tokens,
            )
        except Exception as exc:
            # Reviewer LLM failure → fail-open (implicit approval)
            logger.warning("Reviewer LLM call failed (%s); implicit approval", exc)
            comment = "Approved (reviewer error; fail-open)"
            _rev_ok = {
                "specialist_id": specialist_id,
                "comment": comment,
                "review_iteration": review_iteration,
            }
            run_repository.append_event(
                run_id, "review_approved", _rev_ok, step=None,
            )
            _emit(event_queue, "review_approved", _rev_ok)
            return True, comment

        if not response.has_tool_calls:
            # Plain text = implicit approval (fail-open)
            comment = response.content or "Approved (implicit)"
            _rev_ok = {
                "specialist_id": specialist_id,
                "comment": comment,
                "review_iteration": review_iteration,
            }
            run_repository.append_event(
                run_id, "review_approved", _rev_ok, step=None,
            )
            _emit(event_queue, "review_approved", _rev_ok)
            return True, comment

        # Check for decision tools first
        for tc in response.tool_calls:
            if tc.tool_name == "approve_work":
                comment = tc.arguments.get("comment", "Approved")
                _rev_ok = {
                    "specialist_id": specialist_id,
                    "comment": comment,
                    "review_iteration": review_iteration,
                }
                run_repository.append_event(
                    run_id, "review_approved", _rev_ok, step=None,
                )
                _emit(event_queue, "review_approved", _rev_ok)
                return True, comment

            if tc.tool_name == "request_revision":
                critique = tc.arguments.get("critique", "Revision requested")
                _rev_rej = {
                    "specialist_id": specialist_id,
                    "critique": critique,
                    "review_iteration": review_iteration,
                }
                run_repository.append_event(
                    run_id, "review_rejected", _rev_rej, step=None,
                )
                _emit(event_queue, "review_rejected", _rev_rej)
                return False, critique

        # No decision tool — execute safe inspection tools and continue
        rev_messages.append(
            _make_assistant_tool_turn(response.content, response.tool_calls)
        )
        for tc in response.tool_calls:
            if tc.tool_name in _SAFE_REVIEWER_TOOLS:
                try:
                    result = await pack.execute_tool(tc.tool_name, tc.arguments)
                except Exception:
                    result = {"error": "tool_failed"}
                rev_messages.append(
                    _make_tool_result(
                        tc.call_id,
                        json.dumps(result, ensure_ascii=False),
                    )
                )

    # Max reviewer steps without decision → implicit approval
    comment = "Approved (reviewer reached max steps without decision)"
    _rev_ok = {
        "specialist_id": specialist_id,
        "comment": comment,
        "review_iteration": review_iteration,
    }
    run_repository.append_event(run_id, "review_approved", _rev_ok, step=None)
    _emit(event_queue, "review_approved", _rev_ok)
    return True, comment


# ---------------------------------------------------------------------------
# Parallel task force helpers
# ---------------------------------------------------------------------------

async def _run_task_force_parallel(
    *,
    specialist_ids: List[str],
    task: Any,
    workspace_path: str,
    model_cfg: ModelConfig,
    chat_client: ChatClient,
    run_repository: RunRepository,
    run_id: RunId,
    max_steps: int,
    tracer: Any,
    event_queue: Optional[asyncio.Queue],
    specialist_registry: Any,
    plan: Optional[OrchestrationPlan] = None,
    available_models: Optional[List[str]] = None,
    max_review_iterations: int = 0,
    max_escalations: int = 0,
    approval_channel: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run all specialist packs concurrently and merge their payloads."""

    async def _run_one(specialist_id: str, pack_idx: int) -> Dict[str, Any]:
        assignment = _get_assignment(plan, specialist_id)
        pack = specialist_registry.get_pack(
            specialist_id, workspace_path, task.network_allowed,
            tools=assignment.tools if assignment else None,
            role=assignment.role if assignment else None,
            finish_schema_key=assignment.finish_schema if assignment else None,
        )

        # Per-specialist model selection (ADR-024)
        spec_model_cfg, spec_chat_client = _select_specialist_model(
            assignment, available_models or [], model_cfg, chat_client,
        )

        pack_start_ev = {"specialist_id": specialist_id, "pack_index": pack_idx}
        run_repository.append_event(run_id, "pack_start", pack_start_ev, step=None)
        _emit(event_queue, "pack_start", pack_start_ev)

        user_content = f"Task:\n{task.prompt}"
        brief_text = _get_brief(plan, specialist_id)
        if brief_text:
            user_content += f"\n\nYour specific assignment:\n{brief_text}"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": pack.system_prompt},
            {"role": "user", "content": user_content},
        ]
        step_prefix = f"{specialist_id}_"
        return await _execute_pack_loop(
            pack=pack,
            messages=messages,
            model_cfg=spec_model_cfg,
            chat_client=spec_chat_client,
            run_repository=run_repository,
            run_id=run_id,
            step_prefix=step_prefix,
            max_steps=max_steps,
            tracer=tracer,
            specialist_id=specialist_id,
            event_queue=event_queue,
            available_models=available_models,
            max_review_iterations=max_review_iterations,
            max_escalations=max_escalations,
            approval_channel=approval_channel,
            delegation_depth=_MAX_DELEGATION_DEPTH,
            specialist_registry=specialist_registry,
            workspace_path=workspace_path,
            config=config,
            finish_schema_key=assignment.finish_schema if assignment else None,
        )

    results = await asyncio.gather(
        *[_run_one(sid, idx) for idx, sid in enumerate(specialist_ids)],
        return_exceptions=True,
    )
    return _merge_parallel_payloads(list(results), specialist_ids)


def _merge_parallel_payloads(
    payloads: List[Any],
    specialist_ids: List[str],
) -> Dict[str, Any]:
    """Merge parallel pack payloads into a single combined result dict."""
    pack_results: Dict[str, Any] = {}
    summaries: List[str] = []

    for sid, payload in zip(specialist_ids, payloads):
        if isinstance(payload, BaseException):
            pack_results[sid] = {
                "error": str(payload),
                "error_type": type(payload).__name__,
            }
            summaries.append(f"{sid}: error — {payload}")
        else:
            pack_results[sid] = payload
            pack_summary = payload.get("summary") or payload.get("executive_summary") or ""
            if pack_summary:
                summaries.append(f"{sid}: {pack_summary}")

    combined_summary = " | ".join(summaries) if summaries else "Parallel task force completed."
    return {
        "action": "final",
        "pack_results": pack_results,
        "summary": combined_summary,
        "artifacts": [],
        "next_steps": [],
    }


# ---------------------------------------------------------------------------
# Chat client rebuild helper (preserves FallbackChatClient wrapper)
# ---------------------------------------------------------------------------

def _rebuild_chat_client(new_model_cfg: ModelConfig, old_client: ChatClient) -> ChatClient:
    """Build a new chat client for *new_model_cfg*, preserving FallbackChatClient wrapping.

    When the outer ``execute_task()`` has wrapped the client with
    ``FallbackChatClient`` for cloud fallback, a raw ``build_chat_client()``
    call would lose that wrapper.  This helper detects the wrapper and
    replaces only the inner local client, keeping cloud fallback intact.
    """
    from agentic_concierge.infrastructure.chat import build_chat_client
    new_local = build_chat_client(new_model_cfg)
    try:
        from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient
        if isinstance(old_client, FallbackChatClient):
            return FallbackChatClient(
                local=new_local,
                cloud=old_client._cloud,
                cloud_model=old_client._cloud_model,
                policy=old_client._policy,
            )
    except ImportError:
        pass
    return new_local


# ---------------------------------------------------------------------------
# Adaptive escalation helper (ADR-020)
# ---------------------------------------------------------------------------

def _attempt_escalation(
    *,
    trigger: str,
    model_cfg: ModelConfig,
    available_models: Optional[List[str]],
    escalation_count: int,
    escalated_triggers: set,
    max_escalations: int,
    run_repository: RunRepository,
    run_id: RunId,
    event_queue: Optional[asyncio.Queue],
    step_key: str,
) -> Optional[ModelConfig]:
    """Try to escalate to a larger model.  Returns new ModelConfig on success, None on failure."""
    if max_escalations <= 0:
        return None
    if escalation_count >= max_escalations:
        logger.debug("Escalation cap reached (%d); trigger %r ignored", escalation_count, trigger)
        return None
    if trigger in escalated_triggers:
        logger.debug("Trigger %r already escalated; ignoring", trigger)
        return None

    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    larger = pick_larger_model(available_models or [], model_cfg.model)
    if larger is None:
        logger.info("No larger model available for escalation (trigger=%s)", trigger)
        return None

    from agentic_concierge.infrastructure.llm_discovery import _scale_timeout
    warnings: list[str] = []
    new_timeout = _scale_timeout(larger, model_cfg, warnings)

    new_cfg = ModelConfig(
        base_url=model_cfg.base_url,
        model=larger,
        backend=model_cfg.backend,
        api_key=model_cfg.api_key,
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        max_tokens=model_cfg.max_tokens,
        timeout_s=new_timeout,
    )

    _esc_ev = {
        "trigger": trigger,
        "from_model": model_cfg.model,
        "to_model": larger,
        "escalation_count": escalation_count + 1,
    }
    run_repository.append_event(run_id, "model_escalated", _esc_ev, step=step_key)
    _emit(event_queue, "model_escalated", _esc_ev, step=step_key)
    logger.warning(
        "Model escalated: %s → %s (trigger=%s, escalation #%d)",
        model_cfg.model, larger, trigger, escalation_count + 1,
    )
    return new_cfg


# ---------------------------------------------------------------------------
# Pack tool loop
# ---------------------------------------------------------------------------

async def _execute_pack_loop(
    *,
    pack: Any,
    messages: List[Dict[str, Any]],
    model_cfg: ModelConfig,
    chat_client: ChatClient,
    run_repository: RunRepository,
    run_id: RunId,
    step_prefix: str = "",
    max_steps: int = 40,
    tracer: Any = None,
    specialist_id: str = "",
    event_queue: Optional[asyncio.Queue] = None,
    available_models: Optional[List[str]] = None,
    max_review_iterations: int = 0,
    max_escalations: int = 0,
    approval_channel: Optional[Any] = None,
    delegation_depth: int = 0,
    specialist_registry: Optional[Any] = None,
    workspace_path: Optional[str] = None,
    config: Optional[Any] = None,
    finish_schema_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one specialist pack's tool loop until ``finish_task`` or ``max_steps``.

    ``messages`` is mutated in place as the conversation accumulates.
    ``pack.aopen()`` is called before the loop; ``pack.aclose()`` in a
    ``finally`` block — ensuring MCP subprocess cleanup even on error.

    Returns the final payload dict (always has ``action: "final"``).
    """
    from agentic_concierge.infrastructure.telemetry import _NOOP_TRACER
    if tracer is None:
        tracer = _NOOP_TRACER

    payload: Dict[str, Any] = {}
    any_non_finish_tool_called = False
    consecutive_plain_text = 0
    total_plain_text = 0
    # Repetition detection: sliding window of recent (tool_name, args) signatures.
    tool_call_history: List[str] = []
    # Secondary: track tool names only for semantic loop detection.
    tool_name_history: List[str] = []
    # Gate 4: track how many times the reviewer has rejected.
    review_iteration_count = 0
    # Adaptive escalation state (ADR-020).
    escalation_count = 0
    escalated_triggers: set = set()
    # Delegation counter for step-key prefixing.
    delegation_num = 0

    # Build effective tool definitions: pack tools + inline tools.
    effective_tool_defs = list(pack.tool_definitions)
    if approval_channel is not None:
        effective_tool_defs.append(_REQUEST_APPROVAL_TOOL_DEF)
    if delegation_depth > 0 and specialist_registry is not None:
        effective_tool_defs.append(_DELEGATE_TOOL_DEF)

    try:
        await pack.aopen()
        for step in range(max_steps):
            step_key = f"{step_prefix}step_{step}"
            logger.debug("Step %s: %d messages in context", step_key, len(messages))
            _llm_req_ev = {"step": step, "message_count": len(messages)}
            run_repository.append_event(run_id, "llm_request", _llm_req_ev, step=step_key)
            _emit(event_queue, "llm_request", _llm_req_ev, step=step_key)

            with tracer.start_as_current_span("concierge.llm_call") as llm_span:
                llm_span.set_attribute("step", step)
                llm_span.set_attribute("specialist_id", specialist_id)
                llm_span.set_attribute("model", model_cfg.model)
                llm_span.set_attribute("message_count", len(messages))
                try:
                    response = await chat_client.chat(
                        messages=messages,
                        model=model_cfg.model,
                        tools=effective_tool_defs,
                        temperature=model_cfg.temperature,
                        top_p=model_cfg.top_p,
                        max_tokens=model_cfg.max_tokens,
                    )
                except httpx.ReadTimeout:
                    from agentic_concierge.infrastructure.llm_discovery import pick_smaller_model
                    original_model = model_cfg.model
                    smaller = pick_smaller_model(available_models or [], model_cfg.model)
                    if smaller is None:
                        raise RuntimeError(
                            f"LLM timed out after {model_cfg.timeout_s}s "
                            f"(model={model_cfg.model}). No smaller model available. "
                            "Try pulling a smaller model or increasing timeout_s."
                        )
                    logger.warning(
                        "Timeout with %s; retrying with smaller model %s",
                        model_cfg.model, smaller,
                    )
                    new_timeout = min(model_cfg.timeout_s * 1.5, 900.0)
                    model_cfg = ModelConfig(
                        base_url=model_cfg.base_url,
                        model=smaller,
                        backend=model_cfg.backend,
                        api_key=model_cfg.api_key,
                        temperature=model_cfg.temperature,
                        top_p=model_cfg.top_p,
                        max_tokens=model_cfg.max_tokens,
                        timeout_s=new_timeout,
                    )
                    chat_client = _rebuild_chat_client(model_cfg, chat_client)
                    _emit(event_queue, "timeout_recovery", {
                        "original_model": original_model,
                        "fallback_model": smaller,
                        "new_timeout_s": new_timeout,
                    }, step=step_key)
                    response = await chat_client.chat(
                        messages=messages,
                        model=smaller,
                        tools=effective_tool_defs,
                        temperature=model_cfg.temperature,
                        top_p=model_cfg.top_p,
                        max_tokens=model_cfg.max_tokens,
                    )
                llm_span.set_attribute("tool_calls_returned", len(response.tool_calls))

            # Drain pending cloud_fallback events
            if hasattr(chat_client, "pop_events"):
                for fb_event in chat_client.pop_events():
                    run_repository.append_event(
                        run_id, "cloud_fallback", fb_event, step=step_key,
                    )
                    _emit(event_queue, "cloud_fallback", fb_event, step=step_key)
                    logger.info(
                        "Cloud fallback used: step=%s reason=%s local=%s cloud=%s",
                        step_key, fb_event.get("reason"),
                        fb_event.get("local_model"), fb_event.get("cloud_model"),
                    )

            _llm_resp_ev = {
                "content": (response.content or "")[:MAX_LLM_CONTENT_IN_RUNLOG_CHARS],
                "tool_calls": [
                    {"name": tc.tool_name, "call_id": tc.call_id}
                    for tc in response.tool_calls
                ],
            }
            run_repository.append_event(run_id, "llm_response", _llm_resp_ev, step=step_key)
            _emit(event_queue, "llm_response", _llm_resp_ev, step=step_key)

            if not response.has_tool_calls:
                consecutive_plain_text += 1
                total_plain_text += 1
                if consecutive_plain_text <= _MAX_PLAIN_TEXT_RETRIES:
                    tool_names = [t["function"]["name"] for t in effective_tool_defs]
                    if any_non_finish_tool_called or total_plain_text >= 2:
                        correction = (
                            "You already have enough information. Do NOT call any more "
                            "search or data-gathering tools. Call finish_task now with "
                            "a summary of what you found.\n"
                            "Do not respond with plain text — you MUST call the "
                            "finish_task tool."
                        )
                    else:
                        correction = (
                            "You must call one of the available tools to continue — "
                            "do not respond with plain text.\n"
                            f"Available tools: {', '.join(tool_names)}.\n"
                            "If the task is fully complete, call finish_task. "
                            "Otherwise, use a tool to make progress."
                        )
                    messages.append({"role": "assistant", "content": response.content or ""})
                    messages.append({"role": "user", "content": correction})
                    logger.warning(
                        "Step %s: LLM returned plain text (attempt %d/%d); injecting corrective re-prompt",
                        step_key, consecutive_plain_text, _MAX_PLAIN_TEXT_RETRIES,
                    )
                    _reprompt_ev = {
                        "reason": "plain_text_response",
                        "attempt": consecutive_plain_text,
                        "max_retries": _MAX_PLAIN_TEXT_RETRIES,
                    }
                    run_repository.append_event(
                        run_id, "corrective_reprompt", _reprompt_ev, step=step_key,
                    )
                    _emit(event_queue, "corrective_reprompt", _reprompt_ev, step=step_key)
                    continue

                # Escalation trigger: plain_text_exhaustion (ADR-020)
                new_cfg = _attempt_escalation(
                    trigger="plain_text_exhaustion",
                    model_cfg=model_cfg,
                    available_models=available_models,
                    escalation_count=escalation_count,
                    escalated_triggers=escalated_triggers,
                    max_escalations=max_escalations,
                    run_repository=run_repository,
                    run_id=run_id,
                    event_queue=event_queue,
                    step_key=step_key,
                )
                if new_cfg is not None:
                    model_cfg = new_cfg
                    chat_client = _rebuild_chat_client(model_cfg, chat_client)
                    escalation_count += 1
                    escalated_triggers.add("plain_text_exhaustion")
                    consecutive_plain_text = 0  # reset for the new model
                    messages.append({"role": "assistant", "content": response.content or ""})
                    _esc_correction = (
                        "You already have enough information. Call finish_task now "
                        "with a summary of what you found. Do not respond with "
                        "plain text — you MUST call the finish_task tool."
                    ) if any_non_finish_tool_called else (
                        "You must call one of the available tools to continue — "
                        "do not respond with plain text.\n"
                        f"Available tools: {', '.join(t['function']['name'] for t in effective_tool_defs)}.\n"
                        "If the task is fully complete, call finish_task. "
                        "Otherwise, use a tool to make progress."
                    )
                    messages.append({"role": "user", "content": _esc_correction})
                    continue

                logger.warning(
                    "Step %s: LLM returned plain text %d time(s); using as final payload",
                    step_key, consecutive_plain_text,
                )
                payload = {
                    "action": "final",
                    "summary": response.content or "",
                    "artifacts": [],
                    "next_steps": [],
                    "notes": (
                        f"Model returned plain text {consecutive_plain_text} time(s) "
                        "without calling a tool; used text response as summary."
                    ),
                }
                break

            consecutive_plain_text = 0

            messages.append(
                _make_assistant_tool_turn(response.content, response.tool_calls)
            )

            finish_payload: Optional[Dict[str, Any]] = None

            for tc in response.tool_calls:
                _tc_ev = {"tool": tc.tool_name, "args": tc.arguments}
                run_repository.append_event(run_id, "tool_call", _tc_ev, step=step_key)
                _emit(event_queue, "tool_call", _tc_ev, step=step_key)

                if tc.tool_name == pack.finish_tool_name:
                    # Gate 1: LLM must attempt at least one non-finish tool first.
                    if not any_non_finish_tool_called:
                        logger.warning(
                            "Step %s: finish_task called before any tool was used; "
                            "sending error to LLM for retry",
                            step_key,
                        )
                        error_result = {
                            "error": "finish_task_called_without_doing_work",
                            "message": (
                                "You must use at least one tool to actually complete "
                                "the task before calling finish_task. Call finish_task "
                                "only after you have done the work and verified it."
                            ),
                            "hint": (
                                "Use your available tools first (e.g. shell, write_file, "
                                "web_search), then call finish_task."
                            ),
                        }
                        messages.append(
                            _make_tool_result(tc.call_id, json.dumps(error_result))
                        )
                        _no_work_ev = {"tool": tc.tool_name, "result": error_result}
                        run_repository.append_event(run_id, "tool_result", _no_work_ev, step=step_key)
                        _emit(event_queue, "tool_result", _no_work_ev, step=step_key)
                        continue

                    # Gate 2: Required fields must all be present.
                    missing = [
                        f for f in pack.finish_required_fields if f not in tc.arguments
                    ]
                    if missing:
                        logger.warning(
                            "Step %s: finish_task missing required fields %s; sending error to LLM for retry",
                            step_key, missing,
                        )
                        error_result = {
                            "error": "finish_task called with missing required fields",
                            "missing_fields": missing,
                            "required_fields": pack.finish_required_fields,
                            "hint": "Call finish_task again with all required fields populated.",
                        }
                        messages.append(
                            _make_tool_result(tc.call_id, json.dumps(error_result))
                        )
                        _missing_fields_ev = {"tool": tc.tool_name, "result": error_result}
                        run_repository.append_event(run_id, "tool_result", _missing_fields_ev, step=step_key)
                        _emit(event_queue, "tool_result", _missing_fields_ev, step=step_key)
                        continue

                    # Gate 3: Pack-defined quality gate (e.g. tests_verified check).
                    # Use hasattr so custom _Pack stubs in tests don't need the method,
                    # and isinstance(str) so MagicMock returns don't trigger the gate.
                    _vfp = getattr(pack, "validate_finish_payload", None)
                    quality_error = _vfp(tc.arguments) if _vfp is not None else None
                    if isinstance(quality_error, str) and quality_error:
                        logger.warning(
                            "Step %s: quality gate failed: %s", step_key, quality_error
                        )
                        error_result = {
                            "error": "quality_gate_failed",
                            "message": quality_error,
                            "hint": "Call run_tests, fix issues, then retry finish_task.",
                        }
                        messages.append(
                            _make_tool_result(tc.call_id, json.dumps(error_result))
                        )
                        _qg_ev = {"tool": tc.tool_name, "message": quality_error}
                        run_repository.append_event(run_id, "quality_gate_failed", _qg_ev, step=step_key)
                        _emit(event_queue, "quality_gate_failed", _qg_ev, step=step_key)
                        continue

                    finish_payload = {"action": "final", **tc.arguments}
                    messages.append(_make_tool_result(tc.call_id, _FINISH_TOOL_RESULT_CONTENT))
                    _finish_ev = {"tool": tc.tool_name, "result": {"status": "task_completed"}}
                    run_repository.append_event(run_id, "tool_result", _finish_ev, step=step_key)
                    _emit(event_queue, "tool_result", _finish_ev, step=step_key)
                    continue

                # --- Inline tool: request_approval (ADR-021) ---
                if tc.tool_name == "request_approval":
                    any_non_finish_tool_called = True
                    from agentic_concierge.application.approval import (
                        ApprovalRequest,
                        ApprovalDecision,
                    )
                    req = ApprovalRequest(
                        action=tc.arguments.get("action", ""),
                        reason=tc.arguments.get("reason", ""),
                        command=tc.arguments.get("command", ""),
                        specialist_id=specialist_id,
                    )
                    _appr_req_ev = {
                        "request_id": req.request_id,
                        "action": req.action,
                        "reason": req.reason,
                        "command": req.command,
                    }
                    run_repository.append_event(
                        run_id, "approval_requested", _appr_req_ev, step=step_key,
                    )
                    _emit(event_queue, "approval_requested", _appr_req_ev, step=step_key)

                    if approval_channel is not None:
                        try:
                            decision = await approval_channel.request_approval(req)
                        except Exception as exc:
                            logger.warning(
                                "Approval channel error (%s); denying (fail-closed)", exc
                            )
                            decision = ApprovalDecision(
                                approved=False, comment=f"Channel error: {exc}"
                            )
                    else:
                        decision = ApprovalDecision(approved=True, comment="auto-approved (no channel)")

                    if decision.approved:
                        _appr_ok = {
                            "request_id": req.request_id,
                            "approved": True,
                            "comment": decision.comment,
                        }
                        run_repository.append_event(
                            run_id, "approval_granted", _appr_ok, step=step_key,
                        )
                        _emit(event_queue, "approval_granted", _appr_ok, step=step_key)
                    else:
                        _appr_deny = {
                            "request_id": req.request_id,
                            "approved": False,
                            "comment": decision.comment,
                        }
                        run_repository.append_event(
                            run_id, "approval_denied", _appr_deny, step=step_key,
                        )
                        _emit(event_queue, "approval_denied", _appr_deny, step=step_key)

                    result_dict = {
                        "approved": decision.approved,
                        "comment": decision.comment,
                    }
                    messages.append(
                        _make_tool_result(tc.call_id, json.dumps(result_dict))
                    )
                    continue

                # --- Inline tool: delegate_to_specialist (ADR-022) ---
                if tc.tool_name == "delegate_to_specialist":
                    any_non_finish_tool_called = True
                    delegation_num += 1
                    sub_task = tc.arguments.get("sub_task", "")
                    spec_type = tc.arguments.get("specialist_type", "")
                    sub_tools = tc.arguments.get("tools")
                    sub_role = tc.arguments.get("role")

                    _del_start_ev = {
                        "parent_specialist": specialist_id,
                        "sub_specialist": spec_type,
                        "sub_task": sub_task[:200],
                        "delegation_num": delegation_num,
                    }
                    run_repository.append_event(
                        run_id, "delegation_start", _del_start_ev, step=step_key,
                    )
                    _emit(event_queue, "delegation_start", _del_start_ev, step=step_key)

                    if specialist_registry is None or workspace_path is None:
                        err = {"error": "delegation_not_available", "message": "Delegation infrastructure not configured."}
                        messages.append(_make_tool_result(tc.call_id, json.dumps(err)))
                        continue

                    try:
                        sub_pack = specialist_registry.get_pack(
                            spec_type, workspace_path, True,
                            tools=sub_tools, role=sub_role,
                        )
                    except (ValueError, KeyError) as exc:
                        err = {"error": "unknown_specialist", "message": str(exc)}
                        messages.append(_make_tool_result(tc.call_id, json.dumps(err)))
                        _del_err_ev = {
                            "parent_specialist": specialist_id,
                            "sub_specialist": spec_type,
                            "delegation_num": delegation_num,
                            "error": str(exc),
                        }
                        run_repository.append_event(
                            run_id, "delegation_complete", _del_err_ev, step=step_key,
                        )
                        _emit(event_queue, "delegation_complete", _del_err_ev, step=step_key)
                        continue

                    # Calculate step budget for sub-specialist.
                    remaining = max_steps - step - 1
                    sub_max = max(min(remaining // 2, _MAX_DELEGATION_STEPS), _MIN_DELEGATION_STEPS)

                    sub_messages: List[Dict[str, Any]] = [
                        {"role": "system", "content": sub_pack.system_prompt},
                        {"role": "user", "content": f"Task:\n{sub_task}"},
                    ]
                    sub_step_prefix = f"{step_prefix}step_{step}.d{delegation_num}."

                    try:
                        sub_payload = await _execute_pack_loop(
                            pack=sub_pack,
                            messages=sub_messages,
                            model_cfg=model_cfg,
                            chat_client=chat_client,
                            run_repository=run_repository,
                            run_id=run_id,
                            step_prefix=sub_step_prefix,
                            max_steps=sub_max,
                            tracer=tracer,
                            specialist_id=spec_type,
                            event_queue=event_queue,
                            available_models=available_models,
                            max_review_iterations=max_review_iterations,
                            max_escalations=max_escalations,
                            approval_channel=approval_channel,
                            delegation_depth=0,  # cannot delegate further
                        )
                    except Exception as exc:
                        logger.warning(
                            "Delegation to %r failed: %s", spec_type, exc,
                        )
                        sub_payload = {"error": "delegation_failed", "message": str(exc)}

                    _del_done_ev = {
                        "parent_specialist": specialist_id,
                        "sub_specialist": spec_type,
                        "delegation_num": delegation_num,
                        "summary": (sub_payload.get("summary") or sub_payload.get("executive_summary") or "")[:200],
                    }
                    run_repository.append_event(
                        run_id, "delegation_complete", _del_done_ev, step=step_key,
                    )
                    _emit(event_queue, "delegation_complete", _del_done_ev, step=step_key)

                    messages.append(
                        _make_tool_result(tc.call_id, json.dumps(sub_payload, ensure_ascii=False))
                    )
                    continue

                # Mark that the LLM has attempted at least one non-finish tool.
                any_non_finish_tool_called = True

                # Repetition detection: warn the LLM when it repeats the same call.
                _call_sig = tc.tool_name + ":" + json.dumps(tc.arguments, sort_keys=True)
                _recent_repeats = sum(
                    1 for s in tool_call_history[-_LOOP_DETECT_WINDOW:] if s == _call_sig
                )
                tool_call_history.append(_call_sig)
                # Secondary: detect same tool name with varying args (semantic loop).
                tool_name_history.append(tc.tool_name)
                _name_repeats = sum(
                    1 for n in tool_name_history[-_TOOL_NAME_LOOP_WINDOW:]
                    if n == tc.tool_name
                )
                if _name_repeats >= _TOOL_NAME_LOOP_THRESHOLD and _recent_repeats < _LOOP_DETECT_THRESHOLD:
                    _recent_repeats = _LOOP_DETECT_THRESHOLD  # promote to loop

                error_type: Optional[str] = None
                error_message: str = ""
                with tracer.start_as_current_span("concierge.tool_call") as tool_span:
                    tool_span.set_attribute("tool_name", tc.tool_name)
                    tool_span.set_attribute("specialist_id", specialist_id)
                    try:
                        result = await pack.execute_tool(tc.tool_name, tc.arguments)
                    except PermissionError as exc:
                        result = {"error": "permission_denied", "message": str(exc)}
                        error_type, error_message = "permission", str(exc)
                    except (ValueError, TypeError) as exc:
                        result = {"error": "invalid_arguments", "message": str(exc)}
                        error_type, error_message = "invalid_args", str(exc)
                    except OSError as exc:
                        result = {"error": "io_error", "message": str(exc)}
                        error_type, error_message = "io_error", str(exc)
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "error": "unexpected_error",
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                        }
                        error_type, error_message = "unexpected", str(exc)

                if error_type is not None:
                    logger.warning(
                        "Step %s: tool %r error (%s): %s",
                        step_key, tc.tool_name, error_type, error_message,
                    )
                    _terr_ev = {
                        "tool": tc.tool_name,
                        "error_type": error_type,
                        "error_message": error_message,
                    }
                    run_repository.append_event(run_id, "tool_error", _terr_ev, step=step_key)
                    _emit(event_queue, "tool_error", _terr_ev, step=step_key)
                    if error_type == "permission":
                        logger.warning(
                            "Security event: sandbox violation by tool %r: %s",
                            tc.tool_name, error_message,
                        )
                        _sec_ev = {
                            "event_type": "sandbox_violation",
                            "tool": tc.tool_name,
                            "error_message": error_message,
                        }
                        run_repository.append_event(run_id, "security_event", _sec_ev, step=step_key)
                        _emit(event_queue, "security_event", _sec_ev, step=step_key)
                else:
                    _tresult_ev = {"tool": tc.tool_name, "result": result}
                    run_repository.append_event(run_id, "tool_result", _tresult_ev, step=step_key)
                    _emit(event_queue, "tool_result", _tresult_ev, step=step_key)
                messages.append(_make_tool_result(tc.call_id, json.dumps(result, ensure_ascii=False)))

                if _recent_repeats >= _LOOP_DETECT_THRESHOLD:
                    run_repository.append_event(
                        run_id,
                        "loop_detected",
                        {"tool": tc.tool_name, "repeat_count": _recent_repeats},
                        step=step_key,
                    )
                    _emit(
                        event_queue,
                        "loop_detected",
                        {"tool": tc.tool_name, "repeat_count": _recent_repeats},
                        step=step_key,
                    )

                    # Escalation trigger: persistent_loop (ADR-020)
                    # On the second loop detection (repeat_count > threshold),
                    # try escalating instead of just warning again.
                    if _recent_repeats > _LOOP_DETECT_THRESHOLD:
                        new_cfg = _attempt_escalation(
                            trigger="persistent_loop",
                            model_cfg=model_cfg,
                            available_models=available_models,
                            escalation_count=escalation_count,
                            escalated_triggers=escalated_triggers,
                            max_escalations=max_escalations,
                            run_repository=run_repository,
                            run_id=run_id,
                            event_queue=event_queue,
                            step_key=step_key,
                        )
                        if new_cfg is not None:
                            model_cfg = new_cfg
                            chat_client = _rebuild_chat_client(model_cfg, chat_client)
                            escalation_count += 1
                            escalated_triggers.add("persistent_loop")
                            tool_call_history.clear()  # reset for the new model
                            tool_name_history.clear()

                    _loop_warning = (
                        f"[SYSTEM] LOOP DETECTED: you have called '{tc.tool_name}' "
                        f"excessively ({_name_repeats} time(s) in the last "
                        f"{_TOOL_NAME_LOOP_WINDOW} calls) without meaningful progress.\n"
                        "STOP repeating this tool. Instead:\n"
                        "1. Synthesise the information you ALREADY HAVE from previous results.\n"
                        "2. If results are insufficient, take a DIFFERENT action — use a "
                        "different tool, change your approach, or try a fundamentally "
                        "different query.\n"
                        "3. If you have enough information to answer the question, call "
                        "finish_task NOW with what you have.\n"
                        "4. If you truly cannot make progress, call finish_task and explain "
                        "what was attempted and what failed."
                    )
                    messages.append({"role": "user", "content": _loop_warning})
                    logger.warning(
                        "Step %s: loop detected — tool %r repeated %d time(s); "
                        "injected loop-break warning",
                        step_key, tc.tool_name, _recent_repeats,
                    )

            if finish_payload is not None:
                # Gate 4: Independent reviewer approval.
                if max_review_iterations > 0:
                    approved, review_comment = await _review_specialist_work(
                        finish_payload=finish_payload,
                        pack=pack,
                        model_cfg=model_cfg,
                        chat_client=chat_client,
                        run_repository=run_repository,
                        run_id=run_id,
                        specialist_id=specialist_id,
                        event_queue=event_queue,
                        review_iteration=review_iteration_count,
                        available_models=available_models,
                        finish_schema_key=finish_schema_key,
                    )
                    if not approved:
                        review_iteration_count += 1
                        if review_iteration_count >= max_review_iterations:
                            # Escalation trigger: review_rejection_max (ADR-020)
                            new_cfg = _attempt_escalation(
                                trigger="review_rejection_max",
                                model_cfg=model_cfg,
                                available_models=available_models,
                                escalation_count=escalation_count,
                                escalated_triggers=escalated_triggers,
                                max_escalations=max_escalations,
                                run_repository=run_repository,
                                run_id=run_id,
                                event_queue=event_queue,
                                step_key=step_key,
                            )
                            if new_cfg is not None:
                                model_cfg = new_cfg
                                chat_client = _rebuild_chat_client(model_cfg, chat_client)
                                escalation_count += 1
                                escalated_triggers.add("review_rejection_max")
                                review_iteration_count = 0  # reset for the new model
                                critique_msg = (
                                    f"[REVIEW REJECTED] An independent reviewer found "
                                    f"issues with your work:\n\n{review_comment}\n\n"
                                    f"Address these issues and call finish_task again."
                                )
                                messages.append(
                                    {"role": "user", "content": critique_msg}
                                )
                                continue
                            finish_payload["_review_warning"] = (
                                f"Accepted after {review_iteration_count} review "
                                f"rejection(s): {review_comment}"
                            )
                        else:
                            critique_msg = (
                                f"[REVIEW REJECTED] An independent reviewer found "
                                f"issues with your work:\n\n{review_comment}\n\n"
                                f"Address these issues and call finish_task again."
                            )
                            messages.append(
                                {"role": "user", "content": critique_msg}
                            )
                            continue

                payload = finish_payload
                logger.info(
                    "Pack loop completed: step=%s pack=%s",
                    step_key, step_prefix.rstrip("_") or "single",
                )
                break

        else:
            logger.warning(
                "max_steps (%d) reached without finish_task: run_id=%s step_prefix=%r",
                max_steps, run_id.value, step_prefix,
            )
            payload = {
                "action": "final",
                "summary": f"Reached max_steps ({max_steps}) without completion.",
                "artifacts": [],
                "next_steps": ["Increase max_steps or refine task."],
                "notes": "See runlog for details.",
            }
    finally:
        await pack.aclose()

    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dir_to_workspace_root(run_dir: str) -> str:
    """Derive workspace_root from run_dir (``{root}/runs/{run_id}`` → ``{root}``)."""
    from pathlib import Path
    return str(Path(run_dir).parent.parent)


def _make_assistant_tool_turn(
    content: Optional[str],
    tool_calls: list,
) -> Dict[str, Any]:
    """Build the ``assistant`` message dict for a turn that contains tool calls."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.tool_name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ],
    }


def _make_tool_result(call_id: str, content: str) -> Dict[str, Any]:
    """Build the ``tool`` message dict for a tool call result."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }
