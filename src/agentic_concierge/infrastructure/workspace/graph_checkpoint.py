"""Graph checkpoint persistence — atomic file I/O for TaskGraph snapshots.

Saves and loads ``GraphCheckpoint`` objects as JSON files in the run directory.
Uses atomic write (write to temp + rename) to prevent corruption.

The ``find_resumable_runs`` function scans the workspace for runs that have a
checkpoint but are not yet complete (root != "done"), providing the data needed
by ``concierge resume`` and ``concierge logs --list``.

This module handles ONLY I/O.  Serialization logic is in
``application.graph_checkpoint``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_concierge.application.graph_checkpoint import (
    deserialize_graph,
    serialize_graph,
)
from agentic_concierge.application.task_graph import TaskGraph

logger = logging.getLogger(__name__)

# File names within a run directory.
_CHECKPOINT_FILENAME = "graph_checkpoint.json"


@dataclass
class GraphCheckpoint:
    """Persisted snapshot of a task graph's execution state.

    Attributes:
        run_id: Unique identifier for the run.
        prompt: Original user prompt.
        model_name: Model used for planning / execution.
        specialist_ids: Specialist ids involved in the run.
        graph_data: Serialized TaskGraph dict (from ``serialize_graph``).
        created_at: Unix timestamp when the checkpoint was created.
        updated_at: Unix timestamp of the last update.
        routing_method: How the task was routed ("planner", "explicit").
        completed_node_results: Results from completed nodes, keyed by node id.
    """

    run_id: str
    prompt: str
    model_name: str
    specialist_ids: List[str] = field(default_factory=list)
    graph_data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    routing_method: str = "planner"
    completed_node_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _checkpoint_path(run_dir: str | Path) -> Path:
    """Return the path to the checkpoint file in *run_dir*."""
    return Path(run_dir) / _CHECKPOINT_FILENAME


def save_checkpoint(run_dir: str | Path, checkpoint: GraphCheckpoint) -> None:
    """Atomically write a checkpoint to the run directory.

    Writes to a temporary file in the same directory, then renames.
    This prevents partial reads on crash.

    Raises:
        OSError: If the write/rename fails.
    """
    target = _checkpoint_path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.updated_at = time.time()
    if checkpoint.created_at == 0.0:
        checkpoint.created_at = checkpoint.updated_at

    data = asdict(checkpoint)

    # Atomic write: temp file + rename (same filesystem guaranteed)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), suffix=".tmp", prefix=".ckpt_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, str(target))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(run_dir: str | Path) -> Optional[GraphCheckpoint]:
    """Load a checkpoint from *run_dir*.

    Returns ``None`` if no checkpoint file exists.

    Raises:
        ValueError: If the file exists but contains invalid data.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = _checkpoint_path(run_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return GraphCheckpoint(
        run_id=data["run_id"],
        prompt=data["prompt"],
        model_name=data["model_name"],
        specialist_ids=data.get("specialist_ids", []),
        graph_data=data.get("graph_data", {}),
        created_at=data.get("created_at", 0.0),
        updated_at=data.get("updated_at", 0.0),
        routing_method=data.get("routing_method", "planner"),
        completed_node_results=data.get("completed_node_results", {}),
    )


def delete_checkpoint(run_dir: str | Path) -> None:
    """Delete the checkpoint file if it exists.  No-op if absent."""
    path = _checkpoint_path(run_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to delete checkpoint %s: %s", path, exc)


def load_graph_from_checkpoint(
    run_dir: str | Path,
) -> Optional[TaskGraph]:
    """Load and deserialize the TaskGraph from a checkpoint.

    Convenience wrapper: loads the checkpoint JSON and deserializes the
    embedded ``graph_data`` into a ``TaskGraph``.

    Returns ``None`` if no checkpoint exists.
    """
    ckpt = load_checkpoint(run_dir)
    if ckpt is None or not ckpt.graph_data:
        return None
    return deserialize_graph(ckpt.graph_data)


@dataclass
class ResumableRun:
    """Summary of a run that can be resumed."""

    run_id: str
    prompt: str
    model_name: str
    specialist_ids: List[str]
    routing_method: str
    created_at: float
    updated_at: float
    total_nodes: int
    done_nodes: int
    failed_nodes: int
    pending_nodes: int


def find_resumable_runs(workspace_root: str | Path) -> List[ResumableRun]:
    """Scan the workspace for runs with checkpoints that can be resumed.

    A run is resumable when:
    - It has a ``graph_checkpoint.json`` file
    - The graph's root node is NOT ``done`` (i.e. execution is incomplete)

    Returns a list of ``ResumableRun`` summaries sorted by ``updated_at``
    (most recent first).
    """
    root = Path(workspace_root).resolve()
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []

    resumable: List[ResumableRun] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            ckpt = load_checkpoint(str(run_dir))
        except (ValueError, KeyError, json.JSONDecodeError):
            logger.debug("Skipping invalid checkpoint in %s", run_dir)
            continue
        if ckpt is None or not ckpt.graph_data:
            continue

        # Deserialize to check graph state
        try:
            graph = deserialize_graph(ckpt.graph_data)
        except (ValueError, KeyError):
            logger.debug("Skipping invalid checkpoint in %s", run_dir)
            continue

        # Only include non-complete runs
        if graph.root.status == "done":
            continue

        # Count node statuses
        status_counts: Dict[str, int] = {}
        for node in graph.nodes.values():
            status_counts[node.status] = status_counts.get(node.status, 0) + 1

        resumable.append(ResumableRun(
            run_id=ckpt.run_id,
            prompt=ckpt.prompt,
            model_name=ckpt.model_name,
            specialist_ids=ckpt.specialist_ids,
            routing_method=ckpt.routing_method,
            created_at=ckpt.created_at,
            updated_at=ckpt.updated_at,
            total_nodes=len(graph.nodes),
            done_nodes=status_counts.get("done", 0),
            failed_nodes=status_counts.get("failed", 0),
            pending_nodes=status_counts.get("pending", 0),
        ))

    # Sort by most recently updated
    resumable.sort(key=lambda r: r.updated_at, reverse=True)
    return resumable
