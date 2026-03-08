"""LLM Orchestrator: decomposes tasks, assigns specialists, plans execution.

The orchestrator can assign predefined specialist templates (engineering,
research, enterprise_research) or compose dynamic packs by selecting tools
and describing roles.  When the LLM selects specialist_id="dynamic", it must
also provide a ``tools`` list and ``role`` description.

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
    """Build the create_plan tool definition."""
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
                                "specialist_id": {
                                    "type": "string",
                                    "description": (
                                        "Specialist ID: one of the available templates "
                                        "(e.g. 'engineering', 'research', 'enterprise_research') "
                                        "or 'dynamic' for a custom tool composition."
                                    ),
                                },
                                "brief": {
                                    "type": "string",
                                    "description": "Specific sub-task instructions for this specialist.",
                                },
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Tool names to include when specialist_id='dynamic'. "
                                        "Ignored for template specialists."
                                    ),
                                },
                                "role": {
                                    "type": "string",
                                    "description": (
                                        "Role description when specialist_id='dynamic'. "
                                        "Ignored for template specialists."
                                    ),
                                },
                                "finish_schema": {
                                    "type": "string",
                                    "enum": ["quick_answer", "research_report", "code", "general"],
                                    "description": (
                                        "Finish schema shape. Use 'quick_answer' for simple "
                                        "factual lookups, 'research_report' for deep research "
                                        "requiring bibliography and evidence, 'code' for "
                                        "engineering tasks, 'general' for everything else. "
                                        "Default: template's built-in schema."
                                    ),
                                },
                                "required_capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Capabilities this sub-task needs. The system selects "
                                        "the best specialist template and model automatically. "
                                        "Capabilities: code_python, code_sql, code_rust, "
                                        "reasoning, web_comprehension, summarisation, "
                                        "structured_output, instruction_following. "
                                        "When provided, specialist_id may be omitted (the "
                                        "system resolves it)."
                                    ),
                                },
                            },
                            "required": ["brief"],
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


def _resolve_specialist_from_capabilities(
    required_capabilities: List[str],
) -> str:
    """Resolve capability names to the best matching specialist template.

    Scores each template by how many of the required capabilities it covers
    (using ``_TEMPLATE_CAPABILITIES`` from model_profiles).  Returns the
    template_id with the highest overlap.  Falls back to ``"research"``
    (the most general template) when no template matches.
    """
    from agentic_concierge.infrastructure.model_profiles import _TEMPLATE_CAPABILITIES

    cap_set = set(required_capabilities)
    best_id = "research"  # default fallback
    best_score = -1.0

    for template_id, template_caps in _TEMPLATE_CAPABILITIES.items():
        # Score = number of required capabilities that the template has a non-zero score for
        score = sum(
            template_caps.get(cap, 0.0)
            for cap in cap_set
        )
        if score > best_score:
            best_score = score
            best_id = template_id

    return best_id


def _build_orchestrator_messages(prompt: str, config: ConciergeConfig) -> List[Dict[str, Any]]:
    """Build the system + user messages for the orchestrator LLM call."""
    from agentic_concierge.infrastructure.specialists.tool_catalog import tool_catalog_summary
    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES

    specialist_lines = "\n".join(
        f"- {name}: {spec.description}"
        for name, spec in config.specialists.items()
    )

    template_lines = "\n".join(
        f"- {tid}: tools=[{', '.join(tpl.tool_names)}]"
        for tid, tpl in PACK_TEMPLATES.items()
    )

    tools_summary = tool_catalog_summary()

    # Capability-driven context (ADR-028)
    capability_section = (
        "Available capabilities:\n"
        "  code_python, code_rust, code_sql — code generation/analysis\n"
        "  reasoning — complex logic, math, multi-step deduction\n"
        "  web_comprehension — understanding web content, fact extraction\n"
        "  summarisation — condensing information into concise output\n"
        "  structured_output — following schemas, JSON output\n"
        "  instruction_following — adhering to complex multi-step instructions\n"
    )

    system = (
        "You are a task orchestrator. Analyse the task and determine what capabilities "
        "are needed, then assign to specialist agents.\n\n"
        f"{capability_section}\n"
        f"Available specialist templates:\n{template_lines}\n\n"
        f"Available specialists in config:\n{specialist_lines}\n\n"
        f"Available tools (for dynamic packs):\n{tools_summary}\n\n"
        "RULES (follow strictly):\n"
        "1. PREFER CAPABILITIES. For each sub-task, specify required_capabilities so the "
        "system can automatically select the best specialist template and model. You may "
        "also set specialist_id explicitly if you know the right template.\n"
        "2. PREFER ONE SPECIALIST. Most tasks need only one specialist. Only assign "
        "multiple specialists when the task explicitly asks for INDEPENDENT sub-tasks "
        "(e.g. 'research X AND build Y').\n"
        "3. PREFER TEMPLATES OVER DYNAMIC. The research template already has web_search "
        "and fetch_url — use it for ANY web lookup, question answering, or information "
        "gathering task. Only use specialist_id='dynamic' when no template covers the "
        "needed tool combination.\n"
        "4. NEVER create a dynamic pack without web tools (web_search, fetch_url) if the "
        "task requires web access.\n"
        "5. Keep the brief identical to the original task when using a single specialist. "
        "Do not rephrase, narrow, or split the user's question.\n"
        "6. Use 'sequential' mode only when later specialists need earlier specialists' "
        "outputs. Use 'parallel' when tasks are truly independent.\n"
        "7. Set synthesis_required=true only when multiple specialists produce outputs "
        "that need combining.\n"
        "8. Set model_tier='fast' for simple tasks (lookups, web searches, quick questions, "
        "summarisation). Set model_tier='quality' only for complex multi-step tasks, "
        "code generation, or deep analysis. Default to 'fast' when in doubt.\n"
        "9. Set finish_schema='quick_answer' for simple factual questions (prices, dates, "
        "stock quotes, single-fact lookups). Set finish_schema='research_report' only for "
        "deep research that genuinely needs bibliography and evidence tables. "
        "Set finish_schema='code' for engineering tasks. Default to the template's "
        "built-in schema when in doubt (omit the field).\n"
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

    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES

    # Collect template IDs already in the plan
    template_ids_present = {
        a.specialist_id for a in assignments if a.specialist_id != "dynamic"
    }

    result: List[SpecialistBrief] = []
    for a in assignments:
        if a.specialist_id != "dynamic" or not a.tools:
            result.append(a)
            continue

        tool_set = set(a.tools)
        # Only collapse into a template that's already in the plan
        matched_template: Optional[str] = None
        for tid in template_ids_present:
            tpl = PACK_TEMPLATES.get(tid)
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
    """
    seen: dict[str, SpecialistBrief] = {}
    result: List[SpecialistBrief] = []
    for a in assignments:
        if a.specialist_id in seen:
            existing = seen[a.specialist_id]
            if a.brief and a.brief not in (existing.brief or ""):
                existing_brief = existing.brief or ""
                merged = f"{existing_brief}\n\nAlso: {a.brief}" if existing_brief else a.brief
                object.__setattr__(existing, "brief", merged)
            logger.info("Merged duplicate specialist %r", a.specialist_id)
        else:
            seen[a.specialist_id] = a
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
    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
    known_templates = set(PACK_TEMPLATES.keys())

    assignments: List[SpecialistBrief] = []
    from agentic_concierge.infrastructure.specialists.finish_schemas import FINISH_SCHEMA_KEYS

    for a in raw_assignments:
        sid = a.get("specialist_id", "")
        brief = a.get("brief", "")
        tools = a.get("tools")
        role = a.get("role")
        raw_fs = a.get("finish_schema")
        finish_schema = raw_fs if raw_fs in FINISH_SCHEMA_KEYS else None
        raw_caps = a.get("required_capabilities")
        req_caps = raw_caps if isinstance(raw_caps, list) else None

        # Capability-driven resolution (ADR-028): when required_capabilities
        # are provided and specialist_id is empty/missing, resolve the best
        # template from the capabilities.
        if req_caps and not sid:
            sid = _resolve_specialist_from_capabilities(req_caps)
            logger.info(
                "Resolved capabilities %s → specialist %r", req_caps, sid,
            )

        if sid == "dynamic" and tools:
            assignments.append(SpecialistBrief(
                specialist_id="dynamic",
                brief=brief,
                tools=tools,
                role=role,
                finish_schema=finish_schema,
                required_capabilities=req_caps,
            ))
        elif sid in known_ids or sid in known_templates:
            assignments.append(SpecialistBrief(
                specialist_id=sid, brief=brief, finish_schema=finish_schema,
                required_capabilities=req_caps,
            ))
        else:
            logger.warning("Orchestrator assigned unknown specialist %r; skipping", sid)

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
    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES

    # Use first template that exists in config.
    # Default to "fast" model — adaptive escalation will upgrade if needed.
    for tid in PACK_TEMPLATES:
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
