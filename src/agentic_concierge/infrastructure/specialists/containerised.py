"""ContainerisedSpecialistPack: runs the shell tool inside a Podman container.

Wraps any SpecialistPack (including MCPAugmentedPack) so that the 'shell'
tool executes inside an isolated Podman container with the workspace mounted
at /workspace.  All other tools (file I/O, MCP) are delegated to the inner
pack unchanged.

Lifecycle::

    pack = ContainerisedSpecialistPack(inner, "python:3.12-slim", workspace_path)
    # aopen() is called by _execute_pack_loop before the tool loop:
    await pack.aopen()   # starts container, then inner.aopen()
    # ... tool loop ...
    await pack.aclose()  # inner.aclose(), then stops container

Podman must be installed and in PATH.  The image must be available locally
(pull it first: ``podman pull <image>``).

Shell command allowlist matches the local SandboxPolicy default — the container
provides OS-level isolation but the allowlist remains as a defence-in-depth
measure.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from agentic_concierge.config.constants import MAX_TOOL_OUTPUT_CHARS, SHELL_DEFAULT_TIMEOUT_S

logger = logging.getLogger(__name__)

# Commands the LLM is allowed to run inside the container.
# Mirrors SandboxPolicy.allowed_commands — kept in sync manually.
# Container isolation removes the file-system escape risk, but the allowlist
# limits the blast radius of prompt-injection or unexpected model behaviour.
_ALLOWED_COMMANDS = (
    "python", "python3", "pytest", "bash", "sh", "git",
    "rg", "ls", "cat", "sed", "awk", "jq", "pip", "uv", "make",
)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


class ContainerisedSpecialistPack:
    """Decorator that runs the 'shell' tool inside a Podman container.

    On ``aopen()``:

    1. A Podman container is started from ``container_image`` with the
       workspace mounted at ``/workspace``.
    2. The inner pack's ``aopen()`` is called (connects MCP sessions, etc.).

    On ``aclose()``:

    1. The inner pack's ``aclose()`` is called (disconnects MCP sessions, etc.).
    2. The Podman container is stopped (auto-removed via ``--rm``).

    ``execute_tool()`` intercepts ``shell`` and forwards the command to
    ``podman exec`` inside the running container.  All other tools are
    delegated to the inner pack unchanged.

    This class is transparent with respect to the ``SpecialistPack`` protocol:
    ``specialist_id``, ``system_prompt``, ``finish_tool_name``,
    ``finish_required_fields``, and ``tool_definitions`` are forwarded to the
    inner pack.
    """

    def __init__(
        self,
        inner: Any,
        container_image: str,
        workspace_path: str,
    ) -> None:
        """
        Args:
            inner: Any object satisfying the ``SpecialistPack`` protocol.
            container_image: Podman image name (e.g. ``"python:3.12-slim"``).
            workspace_path: Absolute path on the host to mount as ``/workspace``.
        """
        self._inner = inner
        self._image = container_image
        self._workspace = workspace_path
        self._container_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aopen(self) -> None:
        """Start the Podman container then call inner.aopen()."""
        logger.debug(
            "ContainerisedSpecialistPack: starting container from image %r "
            "(workspace=%s)",
            self._image, self._workspace,
        )
        try:
            proc = subprocess.run(
                [
                    "podman", "run", "-d", "--rm",
                    # :Z applies a private SELinux label so the container process
                    # can read/write the mounted directory on SELinux-enabled hosts
                    # (e.g. Fedora/RHEL).  Ignored on non-SELinux systems.
                    "-v", f"{self._workspace}:/workspace:Z",
                    self._image, "sleep", "infinity",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "podman is not installed or not in PATH. "
                "Install Podman to use container_image in specialist config."
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start Podman container (image={self._image!r}): "
                f"{proc.stderr.strip()}"
            )

        self._container_id = proc.stdout.strip()
        logger.info(
            "ContainerisedSpecialistPack: container started (id=%s, image=%r)",
            self._container_id[:12], self._image,
        )

        await self._inner.aopen()

    async def aclose(self) -> None:
        """Call inner.aclose() then stop the Podman container."""
        await self._inner.aclose()

        if self._container_id:
            try:
                subprocess.run(
                    ["podman", "stop", self._container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.debug(
                    "ContainerisedSpecialistPack: stopped container %s",
                    self._container_id[:12],
                )
            except Exception as exc:
                logger.warning(
                    "ContainerisedSpecialistPack: failed to stop container %s: %s",
                    self._container_id[:12], exc,
                )
            finally:
                self._container_id = None

    # ------------------------------------------------------------------
    # SpecialistPack protocol properties (forwarded to inner pack)
    # ------------------------------------------------------------------

    @property
    def specialist_id(self) -> str:
        return self._inner.specialist_id

    @property
    def system_prompt(self) -> str:
        return self._inner.system_prompt

    @property
    def finish_tool_name(self) -> str:
        return self._inner.finish_tool_name

    @property
    def finish_required_fields(self) -> List[str]:
        return self._inner.finish_required_fields

    @property
    def tool_definitions(self) -> List[Any]:
        return self._inner.tool_definitions

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept 'shell' and run in container; delegate everything else."""
        if name == "shell":
            return self._exec_in_container(args)
        return await self._inner.execute_tool(name, args)

    def _exec_in_container(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a shell command inside the Podman container.

        Returns the same dict format as ``run_cmd`` in sandbox.py:
        ``{"cmd": ..., "returncode": ..., "stdout": ..., "stderr": ...}``.
        """
        if not self._container_id:
            return {"error": "Container is not running (call aopen() first)"}

        cmd = args.get("cmd", [])
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        timeout_s: int = int(args.get("timeout_s", SHELL_DEFAULT_TIMEOUT_S))

        if not cmd:
            raise ValueError("Empty command")

        exe = cmd[0]
        if exe not in _ALLOWED_COMMANDS:
            raise PermissionError(
                f"Command not allowed: {exe!r}. "
                f"Allowed commands: {list(_ALLOWED_COMMANDS)}"
            )

        podman_cmd = [
            "podman", "exec", "-w", "/workspace",
            self._container_id,
        ] + list(cmd)

        try:
            p = subprocess.run(
                podman_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "cmd": " ".join(shlex.quote(x) for x in cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_s}s",
            }

        return {
            "cmd": " ".join(shlex.quote(x) for x in cmd),
            "returncode": p.returncode,
            "stdout": _truncate(p.stdout, MAX_TOOL_OUTPUT_CHARS),
            "stderr": _truncate(p.stderr, MAX_TOOL_OUTPUT_CHARS),
        }
