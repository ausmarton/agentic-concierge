"""LLM Orchestrator: decomposes tasks, assigns specialists, plans execution.

The orchestrator uses **capability-driven routing** (ADR-028).  The routing
LLM specifies *what the task needs* (``required_capabilities``) and the
system resolves the best specialist template and model automatically.

Template names (engineering, research, enterprise_research) are internal
implementation details — they define tool sets and role descriptions but
are never exposed to the routing LLM.

For tasks requiring a custom tool combination not covered by any template,
the LLM can provide explicit ``tools`` and ``role`` fields (dynamic packs).

Falls back to the first available template on any error — zero regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentic_concierge.config import ConciergeConfig

if TYPE_CHECKING:
    from agentic_concierge.application.ports import ChatClient

logger = logging.getLogger(__name__)


@dataclass
class SpecialistBrief:
    """A targeted sub-task description for a specific specialist."""

    specialist_id: str
    brief: str  # targeted instructions / sub-task for this specialist
    tools: Optional[List[str]] = None  # tool names for dynamic packs
    role: Optional[str] = None  # role description for dynamic packs
    finish_schema: Optional[str] = None  # key into FINISH_SCHEMAS (e.g. "quick_answer")
    required_capabilities: Optional[List[str]] = None  # capability-driven routing


@dataclass
class OrchestrationPlan:
    """The orchestrator's task decomposition and assignment plan.

    ``specialist_assignments`` is an ordered list of (specialist_id, brief) pairs.
    ``mode`` is ``"sequential"`` or ``"parallel"``.
    ``synthesis_required`` is True when a synthesis step is needed after execution.
    ``routing_method`` is ``"orchestrator"`` on success or the fallback's method.
    ``required_capabilities`` is derived from the assigned specialists' capabilities
    for RunResult / runlog compatibility with the existing RecruitmentResult shape.
    """

    specialist_assignments: List[SpecialistBrief]
    mode: str  # "sequential" | "parallel"
    synthesis_required: bool
    reasoning: str
    routing_method: str  # "orchestrator" | "template_fallback"
    required_capabilities: List[str] = field(default_factory=list)
    recommended_model_key: str = "quality"  # "fast" or "quality"


def _build_orchestrator_tool_def() -> Dict[str, Any]:
    """Build the create_plan tool definition.

    Capability-driven (ADR-028): the LLM specifies what capabilities the
    task needs; the system resolves the best specialist template and model.
    """
    return {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "Create a task execution plan by assigning sub-tasks to specialists. "
                "Call this tool exactly once with the complete plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "description": "Ordered list of specialist assignments.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "brief": {
                                    "type": "string",
                                    "description": "Specific sub-task instructions for this specialist.",
                                },
                                "required_capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Capabilities this sub-task needs. The system "
                                        "selects the best specialist and model automatically. "
                                        "Values: web_comprehension, summarisation, "
                                        "code_python, code_sql, code_rust, reasoning, "
                                        "structured_output, instruction_following."
                                    ),
                                },
                                "finish_schema": {
                                    "type": "string",
                                    "enum": ["quick_answer", "research_report", "code", "general"],
                                    "description": (
                                        "Finish schema shape. Use 'quick_answer' for simple "
                                        "factual lookups, 'research_report' for deep research "
                                        "requiring bibliography and evidence, 'code' for "
                                        "engineering tasks, 'general' for everything else."
                                    ),
                                },
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Optional: explicit tool names for a custom pack "
                                        "(only when no standard capability set covers the "
                                        "need). Most tasks do NOT need this."
                                    ),
                                },
                                "role": {
                                    "type": "string",
                                    "description": (
                                        "Optional: role description for a custom pack. "
                                        "Only used when 'tools' is also provided."
                                    ),
                                },
                            },
                            "required": ["brief", "required_capabilities"],
                        },
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sequential", "parallel"],
                        "description": (
                            "'sequential' when specialists depend on each other's outputs; "
                            "'parallel' when tasks are independent."
                        ),
                    },
                    "synthesis_required": {
                        "type": "boolean",
                        "description": "True when a final synthesis step is needed to combine outputs.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence explaining the orchestration decision.",
                    },
                    "model_tier": {
                        "type": "string",
                        "enum": ["fast", "quality"],
                        "description": (
                            "Model tier for task execution. Use 'fast' for simple tasks "
                            "(lookups, searches, single-step queries, summarisation). "
                            "Use 'quality' for complex tasks (multi-step engineering, "
                            "code generation, analysis requiring deep reasoning)."
                        ),
                    },
                },
                "required": ["assignments", "mode", "synthesis_required", "reasoning"],
            },
        },
    }


def _filter_known_capabilities(raw_caps: List[str]) -> List[str]:
    """Filter to known capability names, warning on unknown ones.

    Returns only capabilities present in ``KNOWN_CAPABILITIES``.
    If everything is filtered out, returns ``["instruction_following"]``
    as a safe default (model-only, triggers degenerate fallback).
    """
    from agentic_concierge.infrastructure.model_profiles import KNOWN_CAPABILITIES

    known = [c for c in raw_caps if c in KNOWN_CAPABILITIES]
    unknown = [c for c in raw_caps if c not in KNOWN_CAPABILITIES]
    if unknown:
        logger.warning(
            "Orchestrator specified unknown capabilities %s; ignoring", unknown,
        )
    return known or ["instruction_following"]


def _resolve_specialist_from_capabilities(
    required_capabilities: List[str],
) -> str:
    """Resolve capability names to a specialist template ID.

    Thin wrapper around ``_resolve_pack_from_capabilities`` that returns
    only the specialist_id.  Kept for backward-compatible test imports.
    """
    result = _resolve_pack_from_capabilities(required_capabilities)
    return result[0]


def _resolve_pack_from_capabilities(
    required_capabilities: List[str],
) -> tuple:
    """Compose a pack configuration from capability requirements.

    This is the core of capability-driven routing (ADR-028).  Instead of
    mapping capabilities to a template name, it:

    1. Composes a tool set from the union of tools needed for all requested
       capabilities (via ``compose_tools_from_capabilities``).
    2. Checks if the composed tool set exactly matches a known template's
       tools — if so, uses the template (for its tuned role and defaults).
    3. Otherwise, builds a dynamic pack specification with a generated role
       and inferred finish schema.

    Returns:
        ``(specialist_id, tools, role, finish_schema_key)`` where:
        - ``specialist_id`` is a template name or ``"dynamic"``
        - ``tools`` is ``None`` for template matches, else the composed list
        - ``role`` is ``None`` for template matches, else a generated description
        - ``finish_schema_key`` is ``None`` for template matches (use default),
          else an inferred key or ``None``
    """
    from agentic_concierge.infrastructure.model_profiles import (
        compose_tools_from_capabilities,
        generate_role_from_capabilities,
        infer_finish_schema_from_capabilities,
    )
    from agentic_concierge.infrastructure.specialists.dynamic_pack import pack_templates

    composed_tools = compose_tools_from_capabilities(required_capabilities)
    composed_set = set(composed_tools)

    # Degenerate case: only base tools (read_file, list_files) — no useful
    # tool capabilities requested.  Fall back to research (most general).
    from agentic_concierge.infrastructure.model_profiles import _BASE_TOOLS
    if composed_set == set(_BASE_TOOLS):
        return ("research", None, None, "quick_answer")

    # Check if any template's tool set matches exactly.
    for tid, tpl in pack_templates().items():
        if set(tpl.tool_names) == composed_set:
            return (tid, None, None, None)

    # No template match → dynamic pack with composed tools.
    role = generate_role_from_capabilities(required_capabilities)
    finish_key = infer_finish_schema_from_capabilities(required_capabilities)
    return ("dynamic", composed_tools, role, finish_key)


def _build_orchestrator_messages(prompt: str, config: ConciergeConfig) -> List[Dict[str, Any]]:
    """Build the system + user messages for the orchestrator LLM call.

    The prompt is capability-driven (ADR-028): it tells the LLM to specify
    *what the task needs*, not which template to use.
    """
    from agentic_concierge.infrastructure.specialists.tool_catalog import tool_catalog_summary

    tools_summary = tool_catalog_summary()

    system = (
        "You are a task orchestrator. Analyse the task and decide what capabilities "
        "each sub-task needs. The system will automatically select the best specialist "
        "and model for each assignment.\n\n"
        "Available capabilities (use these in required_capabilities):\n"
        "  web_comprehension — web search, fetching URLs, extracting facts from pages\n"
        "  summarisation — condensing information into concise output\n"
        "  code_python — writing, debugging, or analysing Python code\n"
        "  code_rust — writing or analysing Rust code\n"
        "  code_sql — writing SQL queries\n"
        "  reasoning — complex logic, math, multi-step deduction\n"
        "  structured_output — following schemas, producing JSON output\n"
        "  instruction_following — adhering to complex multi-step instructions\n\n"
        f"Available tools (only needed for custom packs):\n{tools_summary}\n\n"
        "RULES:\n"
        "1. ONE SPECIALIST per task. Most tasks need only one assignment. Only create "
        "multiple assignments when the task explicitly asks for INDEPENDENT sub-tasks "
        "(e.g. 'research X AND build Y').\n"
        "2. Keep the brief identical to the original task when using a single specialist. "
        "Do not rephrase, narrow, or split the user's question.\n"
        "3. Set finish_schema='quick_answer' for simple factual questions (prices, dates, "
        "stock quotes, single-fact lookups). Set finish_schema='code' for engineering "
        "tasks. Set finish_schema='research_report' only for deep research that genuinely "
        "needs bibliography and evidence tables. Omit the field when in doubt.\n"
        "4. Set model_tier='fast' for simple tasks (lookups, web searches, quick questions). "
        "Set model_tier='quality' only for complex multi-step engineering, code generation, "
        "or deep analysis. Default to 'fast'.\n"
        "5. Only provide 'tools' and 'role' when the standard capability set does not cover "
        "the need (rare). Most tasks are covered by capabilities alone.\n"
        "6. Use 'sequential' mode only when later assignments need earlier ones' outputs.\n"
        "7. Set synthesis_required=true only when multiple assignments produce outputs "
        "that need combining.\n\n"
        "Call create_plan with the complete assignment plan."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {prompt}"},
    ]


def _derive_required_capabilities(
    specialist_ids: List[str], config: ConciergeConfig
) -> List[str]:
    """Derive required capabilities from the assigned specialists' declared capabilities."""
    caps: List[str] = []
    for sid in specialist_ids:
        spec_cfg = config.specialists.get(sid)
        if spec_cfg:
            for cap in spec_cfg.capabilities:
                if cap not in caps:
                    caps.append(cap)
    return caps


def _collapse_redundant_dynamic(
    assignments: List[SpecialistBrief],
) -> List[SpecialistBrief]:
    """Collapse redundant dynamic packs in multi-specialist plans.

    Only applied when there are 2+ assignments.  When a dynamic pack's tool
    set is a subset of a template that is *already present* in the plan,
    the dynamic assignment is merged into the existing template (its brief
    is appended).  This prevents the orchestrator from splitting work across
    a template and a redundant dynamic pack with the same tools.

    Single-specialist plans and dynamic packs that don't overlap with an
    already-assigned template are left untouched.
    """
    if len(assignments) <= 1:
        return assignments

    from agentic_concierge.infrastructure.specialists.dynamic_pack import pack_templates

    # Collect template IDs already in the plan
    template_ids_present = {
        a.specialist_id for a in assignments if a.specialist_id != "dynamic"
    }
    templates = pack_templates()

    result: List[SpecialistBrief] = []
    for a in assignments:
        if a.specialist_id != "dynamic" or not a.tools:
            result.append(a)
            continue

        tool_set = set(a.tools)
        # Only collapse into a template that's already in the plan
        matched_template: Optional[str] = None
        for tid in template_ids_present:
            tpl = templates.get(tid)
            if tpl and tool_set <= set(tpl.tool_names):
                matched_template = tid
                break

        if matched_template is None:
            result.append(a)
            continue

        # Merge brief into the existing template assignment
        existing = next((r for r in result if r.specialist_id == matched_template), None)
        if existing is not None:
            if a.brief and a.brief not in (existing.brief or ""):
                existing_brief = existing.brief or ""
                merged = f"{existing_brief}\n\nAlso: {a.brief}" if existing_brief else a.brief
                object.__setattr__(existing, "brief", merged)
            logger.info(
                "Collapsed dynamic pack (tools=%s) into existing %r",
                a.tools, matched_template,
            )
        else:
            # Template was supposed to be present but hasn't been added yet
            # (shouldn't happen since we iterate in order, but handle gracefully)
            result.append(a)

    return result


def _dedup_same_id(assignments: List[SpecialistBrief]) -> List[SpecialistBrief]:
    """Merge assignments that share the same specialist_id.

    When the orchestrator assigns e.g. ``research, research``, collapse them
    into a single ``research`` with merged briefs.  This prevents redundant
    parallel packs doing the same work.

    Dynamic packs (specialist_id="dynamic") are only merged when they have
    identical tool sets, since different dynamic packs may serve genuinely
    different purposes.
    """
    seen: dict[str, SpecialistBrief] = {}
    result: List[SpecialistBrief] = []
    for a in assignments:
        # Dynamic packs with different tool sets are NOT duplicates.
        if a.specialist_id == "dynamic":
            # Check if we've seen a dynamic pack with the SAME tools.
            _dedup_key = f"dynamic:{sorted(a.tools or [])}"
        else:
            _dedup_key = a.specialist_id

        if _dedup_key in seen:
            existing = seen[_dedup_key]
            if a.brief and a.brief not in (existing.brief or ""):
                existing_brief = existing.brief or ""
                merged = f"{existing_brief}\n\nAlso: {a.brief}" if existing_brief else a.brief
                object.__setattr__(existing, "brief", merged)
            logger.info("Merged duplicate specialist %r", a.specialist_id)
        else:
            seen[_dedup_key] = a
            result.append(a)
    return result


async def orchestrate_task(
    prompt: str,
    config: ConciergeConfig,
    *,
    chat_client: "ChatClient",
    model: str,
) -> OrchestrationPlan:
    """Decompose a task, assign specialists, and plan execution mode.

    Makes one LLM call with the ``create_plan`` tool.  Falls back to the
    first available template on any error or when the LLM returns no
    usable tool call — zero regression.

    Args:
        prompt: The task prompt to decompose.
        config: Concierge config (specialists, models).
        chat_client: LLM interface.
        model: Model name to use for the orchestrator call.

    Returns:
        ``OrchestrationPlan`` with routing_method ``"orchestrator"`` on success,
        or a plan built from the template fallback result.
    """
    from agentic_concierge.infrastructure.telemetry import get_tracer

    tracer = get_tracer()

    try:
        messages = _build_orchestrator_messages(prompt, config)
        tool_def = _build_orchestrator_tool_def()
        with tracer.start_as_current_span("concierge.orchestrator_call") as span:
            span.set_attribute("model", model)
            response = await chat_client.chat(
                messages,
                model,
                tools=[tool_def],
                temperature=0.0,
                max_tokens=512,
            )
            span.set_attribute("tool_call_returned", bool(response.tool_calls))
    except Exception as exc:
        logger.warning("Orchestrator LLM call failed (%s); falling back to template", exc)
        return _template_fallback_plan(prompt, config)

    # Parse the create_plan tool call
    if not response.tool_calls:
        logger.info("Orchestrator returned no tool call; falling back to template")
        return _template_fallback_plan(prompt, config)

    tc = response.tool_calls[0]
    if tc.tool_name != "create_plan":
        logger.info("Orchestrator called unexpected tool %r; falling back", tc.tool_name)
        return _template_fallback_plan(prompt, config)

    raw_assignments = tc.arguments.get("assignments", [])
    mode = tc.arguments.get("mode", "sequential")
    synthesis_required = tc.arguments.get("synthesis_required", False)
    reasoning = tc.arguments.get("reasoning", "")
    model_tier = tc.arguments.get("model_tier", "quality")
    if model_tier not in ("fast", "quality"):
        model_tier = "quality"

    # Validate assignments: known template IDs or "dynamic" with tools
    known_ids = set(config.specialists.keys())
    from agentic_concierge.infrastructure.specialists.dynamic_pack import pack_templates
    known_templates = set(pack_templates().keys())

    assignments: List[SpecialistBrief] = []
    from agentic_concierge.infrastructure.specialists.finish_schemas import FINISH_SCHEMA_KEYS

    for a in raw_assignments:
        brief = a.get("brief", "")
        llm_tools = a.get("tools")
        llm_role = a.get("role")
        raw_fs = a.get("finish_schema")
        finish_schema = raw_fs if raw_fs in FINISH_SCHEMA_KEYS else None
        raw_caps = a.get("required_capabilities")
        req_caps = _filter_known_capabilities(raw_caps) if isinstance(raw_caps, list) else None

        # Legacy backward compat: if the LLM (or a test) provides
        # specialist_id directly, accept it as an override.
        sid = a.get("specialist_id", "")

        if sid:
            # Explicit specialist_id — use it directly (backward compat).
            if sid == "dynamic" and llm_tools:
                assignments.append(SpecialistBrief(
                    specialist_id="dynamic",
                    brief=brief,
                    tools=llm_tools,
                    role=llm_role,
                    finish_schema=finish_schema,
                    required_capabilities=req_caps,
                ))
            elif sid in known_ids or sid in known_templates:
                assignments.append(SpecialistBrief(
                    specialist_id=sid, brief=brief, finish_schema=finish_schema,
                    required_capabilities=req_caps,
                ))
            else:
                logger.warning("Orchestrator specified unknown specialist %r; skipping", sid)
            continue

        # ── Capability-driven resolution (ADR-028) ──────────────────
        # The LLM specifies what the task needs; the system composes
        # the tool set and resolves the specialist.

        if llm_tools:
            # LLM provided explicit tools without specialist_id → dynamic
            assignments.append(SpecialistBrief(
                specialist_id="dynamic",
                brief=brief,
                tools=llm_tools,
                role=llm_role,
                finish_schema=finish_schema,
                required_capabilities=req_caps,
            ))
            continue

        effective_caps = req_caps or ["instruction_following"]
        resolved_sid, resolved_tools, resolved_role, resolved_fs = (
            _resolve_pack_from_capabilities(effective_caps)
        )
        logger.info(
            "Resolved capabilities %s → specialist=%r tools=%s",
            effective_caps, resolved_sid, resolved_tools,
        )

        # Merge: LLM-specified finish_schema takes precedence over inferred.
        effective_fs = finish_schema or resolved_fs

        if resolved_sid == "dynamic":
            assignments.append(SpecialistBrief(
                specialist_id="dynamic",
                brief=brief,
                tools=resolved_tools,
                role=resolved_role,
                finish_schema=effective_fs,
                required_capabilities=req_caps,
            ))
        elif resolved_sid in known_ids or resolved_sid in known_templates:
            assignments.append(SpecialistBrief(
                specialist_id=resolved_sid,
                brief=brief,
                finish_schema=effective_fs,
                required_capabilities=req_caps,
            ))
        else:
            logger.warning("Resolved unknown specialist %r; skipping", resolved_sid)

    if not assignments:
        logger.info("Orchestrator produced no valid assignments; falling back")
        return _template_fallback_plan(prompt, config)

    # --- plan deduplication ------------------------------------------------
    # 1. Collapse dynamic packs whose tools are subsets of assigned templates.
    assignments = _collapse_redundant_dynamic(assignments)
    # 2. Merge duplicate specialist IDs (e.g. research, research → one research).
    assignments = _dedup_same_id(assignments)

    specialist_ids = [a.specialist_id for a in assignments]
    required_capabilities = _derive_required_capabilities(specialist_ids, config)

    # Force synthesis when multiple specialists assigned
    if len(assignments) > 1:
        synthesis_required = True

    logger.info(
        "Orchestrator plan: specialists=%s mode=%s model_tier=%s synthesis=%s reasoning=%r",
        specialist_ids, mode, model_tier, synthesis_required, reasoning,
    )
    return OrchestrationPlan(
        specialist_assignments=assignments,
        mode=mode,
        synthesis_required=synthesis_required,
        reasoning=reasoning,
        routing_method="orchestrator",
        required_capabilities=required_capabilities,
        recommended_model_key=model_tier,
    )


def _template_fallback_plan(
    prompt: str,
    config: ConciergeConfig,
) -> OrchestrationPlan:
    """Fall back to the first available template specialist."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import pack_templates

    # Use first template that exists in config.
    # Default to "fast" model — adaptive escalation will upgrade if needed.
    for tid in pack_templates():
        if tid in config.specialists:
            logger.info("Template fallback: using %r", tid)
            return OrchestrationPlan(
                specialist_assignments=[SpecialistBrief(specialist_id=tid, brief="")],
                mode="sequential",
                synthesis_required=False,
                reasoning="template_fallback",
                routing_method="template_fallback",
                required_capabilities=[],
                recommended_model_key="fast",
            )

    # Last resort: first specialist in config
    first_id = next(iter(config.specialists))
    logger.info("Template fallback (last resort): using %r", first_id)
    return OrchestrationPlan(
        specialist_assignments=[SpecialistBrief(specialist_id=first_id, brief="")],
        mode="sequential",
        synthesis_required=False,
        reasoning="template_fallback",
        routing_method="template_fallback",
        required_capabilities=[],
        recommended_model_key="fast",
    )
