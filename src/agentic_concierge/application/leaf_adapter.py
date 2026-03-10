"""Leaf adapter — bridges the graph executor to the existing pack loop.

The graph executor (``execute_graph_with_affinity``) needs a ``WorkExecutor``
callback with signature ``(TaskNode, TaskGraph, ModelAssignment) → Dict``.
The existing ``_execute_pack_loop`` in ``execute_task.py`` is the proven,
battle-tested tool-calling loop (quality gates, loop detection, delegation,
escalation, review).

This module provides ``build_leaf_executor`` which creates a ``WorkExecutor``
closure with all shared context (registry, run repository, config, etc.)
bound in.  Each invocation converts a single ``TaskNode`` into the
parameters ``_execute_pack_loop`` expects, calls it, and returns the result.

See GH issue #28 for context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic_concierge.application.agent_roles import ModelAssignment
from agentic_concierge.application.task_graph import TaskGraph, TaskNode
from agentic_concierge.config import ConciergeConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LeafExecutionContext:
    """Shared context for all leaf executions within a single run.

    Created once per ``execute_task`` invocation and passed into the
    ``WorkExecutor`` closure via ``build_leaf_executor``.
    """

    specialist_registry: Any  # SpecialistRegistry
    run_repository: Any  # RunRepository
    run_id: Any  # RunId
    workspace_path: str
    config: ConciergeConfig
    network_allowed: bool
    max_steps: int = 40
    max_review_iterations: int = 2
    max_escalations: int = 2
    approval_channel: Optional[Any] = None
    event_queue: Optional[Any] = None  # asyncio.Queue
    tracer: Optional[Any] = None
    # Injectable pack loop — defaults to the real _execute_pack_loop at build time.
    # Override in tests to avoid importing/running the real loop.
    pack_loop_fn: Optional[Any] = None
    # When task.specialist_id was set explicitly, override automatic specialist
    # resolution for all nodes.
    specialist_id_override: Optional[str] = None
    # Available models for escalation (populated from resolved_llm.available_models).
    available_models: List[str] = field(default_factory=list)
    # Collected results from completed nodes, for context forwarding.
    completed_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _node_to_specialist_id(node: TaskNode) -> str:
    """Determine the specialist_id for a TaskNode.

    Uses ``_resolve_pack_from_capabilities`` from the orchestrator to map
    the node's capabilities to the best template or dynamic pack.
    """
    from agentic_concierge.application.orchestrator import _resolve_pack_from_capabilities

    if node.required_capabilities:
        specialist_id, _tools, _role, _fs = _resolve_pack_from_capabilities(
            node.required_capabilities,
        )
        return specialist_id
    # Fallback: if node has required_tools, use "dynamic"
    if node.required_tools:
        return "dynamic"
    return "research"  # most general fallback


def _build_node_messages(
    node: TaskNode,
    graph: TaskGraph,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Build the initial messages for a leaf node's pack loop.

    Includes:
    - System prompt from the specialist pack
    - User message with the node description
    - Context from completed preceding siblings (if any)
    """
    # Gather context from completed preceding siblings
    context_parts: List[str] = []
    if node.parent_id is not None:
        parent = graph.nodes.get(node.parent_id)
        if parent is not None:
            for sib_id in parent.children:
                if sib_id == node.id:
                    break
                sib = graph.nodes.get(sib_id)
                if sib is not None and sib.status == "done" and sib.result:
                    context_block = json.dumps(sib.result, indent=2, ensure_ascii=False)
                    context_parts.append(
                        f"Result from preceding step '{sib.description[:80]}':\n{context_block}"
                    )

    # Also include the root task description for context
    root = graph.root
    user_content = f"Task:\n{root.description}\n\nYour specific assignment:\n{node.description}"

    if context_parts:
        context_text = "\n\n".join(context_parts)
        user_content += f"\n\nContext from previous steps:\n{context_text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_leaf_executor(ctx: LeafExecutionContext):
    """Create a ``WorkExecutor`` closure that bridges to ``_execute_pack_loop``.

    The returned async callable has signature:
        ``(TaskNode, TaskGraph, ModelAssignment) → Dict[str, Any]``

    Each call:
    1. Resolves the specialist pack for the node's capabilities
    2. Builds messages with context from preceding siblings
    3. Constructs a ModelConfig from the assignment's handle
    4. Calls ``_execute_pack_loop`` with the correct parameters
    5. Stores the result for context forwarding to subsequent siblings

    Returns:
        Async callable matching ``WorkExecutor`` type.
    """
    # Resolve the pack loop function: use injectable override or import the real one.
    if ctx.pack_loop_fn is not None:
        _execute_pack_loop = ctx.pack_loop_fn
    else:
        from agentic_concierge.application.execute_task import _execute_pack_loop

    async def _execute_leaf(
        node: TaskNode,
        graph: TaskGraph,
        assignment: ModelAssignment,
    ) -> Dict[str, Any]:
        handle = assignment.handle
        chat_client = handle.chat_client
        model_id = handle.model_id

        # Build a ModelConfig from the handle for _execute_pack_loop
        # The base_url comes from the handle's slot backend info.
        base_url = getattr(handle, "slot", None)
        if base_url is not None:
            backend_name = handle.slot.backend
            backend = ctx.config.models.get("quality")
            if backend is not None:
                model_cfg = ModelConfig(
                    base_url=backend.base_url,
                    model=model_id,
                    backend=backend_name,
                )
            else:
                model_cfg = ModelConfig(
                    base_url="http://localhost:11434",
                    model=model_id,
                    backend=backend_name,
                )
        else:
            model_cfg = ModelConfig(
                base_url="http://localhost:11434",
                model=model_id,
            )

        # Determine the specialist pack for this node
        if ctx.specialist_id_override:
            # Explicit specialist assignment: use the override directly
            sid = ctx.specialist_id_override
            tools = None
            role = None
            fs_key = None
        elif node.required_capabilities:
            # Resolve pack from capabilities (may use template or dynamic)
            from agentic_concierge.application.orchestrator import _resolve_pack_from_capabilities
            sid, tools, role, fs_key = _resolve_pack_from_capabilities(
                node.required_capabilities,
            )
        else:
            sid = _node_to_specialist_id(node)
            tools = node.required_tools if node.required_tools else None
            role = None
            fs_key = None

        # Empty string means "use pack's built-in default" (no override);
        # this preserves each template's native finish schema.
        raw_fs = fs_key or node.finish_schema_key
        finish_schema_key = raw_fs if raw_fs else None

        pack = ctx.specialist_registry.get_pack(
            sid,
            ctx.workspace_path,
            ctx.network_allowed,
            tools=tools,
            role=role,
            finish_schema_key=finish_schema_key,
        )

        # Build messages with context from preceding siblings
        messages = _build_node_messages(node, graph, pack.system_prompt)

        # Build step prefix from node id for unique step keys in runlog.
        # For single-specialist (explicit override) runs, use empty prefix
        # to maintain backward-compatible step keys (step_0, step_1, ...).
        if ctx.specialist_id_override:
            step_prefix = ""
        else:
            step_prefix = f"node_{node.id}_"

        logger.info(
            "Leaf adapter: executing node %s (%s) specialist=%s model=%s",
            node.id, node.description[:60], sid, model_id,
        )

        # Emit node_execution_start event to both runlog and event queue
        _node_start_data = {
            "node_id": node.id,
            "description": node.description,
            "specialist_id": sid,
            "model": model_id,
            "role": assignment.role,
        }
        try:
            ctx.run_repository.append_event(
                ctx.run_id, "node_execution_start", _node_start_data, step=None,
            )
        except Exception:
            pass
        if ctx.event_queue is not None:
            try:
                ctx.event_queue.put_nowait({
                    "kind": "node_execution_start",
                    "data": _node_start_data,
                    "step": None,
                })
            except Exception:
                pass

        # Synthesis nodes skip Gate 1 (must-use-tool) and plain-text
        # re-prompting because their job is to reason over existing data.
        # Auto-detect: if the planner marked it OR it has context from
        # completed siblings and only reasoning capabilities.
        _is_synth = getattr(node, "is_synthesis", False)
        if not _is_synth and node.parent_id is not None:
            _has_done_siblings = any(
                graph.nodes[sib_id].status == "done"
                for sib_id in graph.nodes[node.parent_id].children
                if sib_id != node.id and sib_id in graph.nodes
            )
            if _has_done_siblings:
                _SYNTHESIS_CAPS = {"reasoning", "summarisation", "structured_output",
                                   "instruction_following"}
                if node.required_capabilities and set(node.required_capabilities).issubset(_SYNTHESIS_CAPS):
                    _is_synth = True

        payload = await _execute_pack_loop(
            pack=pack,
            messages=messages,
            model_cfg=model_cfg,
            chat_client=chat_client,
            run_repository=ctx.run_repository,
            run_id=ctx.run_id,
            step_prefix=step_prefix,
            max_steps=ctx.max_steps,
            tracer=ctx.tracer,
            specialist_id=sid,
            event_queue=ctx.event_queue,
            available_models=ctx.available_models,
            max_review_iterations=ctx.max_review_iterations,
            max_escalations=ctx.max_escalations,
            approval_channel=ctx.approval_channel,
            delegation_depth=1,  # allow one level of delegation
            specialist_registry=ctx.specialist_registry,
            workspace_path=ctx.workspace_path,
            config=ctx.config,
            finish_schema_key=finish_schema_key,
            is_synthesis=_is_synth,
        )

        # Store result for context forwarding
        ctx.completed_results[node.id] = payload

        return payload

    return _execute_leaf
