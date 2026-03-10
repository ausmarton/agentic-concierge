"""Configuration schema. Defaults point at Ollama; any OpenAI-compatible backend works via base_url + model."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class MCPServerConfig(BaseModel):
    """Configuration for one MCP (Model Context Protocol) tool server.

    Each server exposes tools that are merged into the specialist pack's
    tool_definitions and dispatched via the MCP session at runtime.
    """

    name: str = Field(..., description="Server name — used as tool prefix: mcp__<name>__<tool>.")
    transport: str = Field("stdio", description="Transport type: 'stdio' or 'sse'.")
    # stdio fields
    command: Optional[str] = Field(None, description="Executable to launch (stdio transport).")
    args: List[str] = Field(default_factory=list, description="Arguments for the command.")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables for the subprocess.")
    # sse fields
    url: Optional[str] = Field(None, description="SSE endpoint URL (sse transport).")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for SSE transport.")
    timeout_s: float = Field(30.0, description="Timeout in seconds for MCP calls.")

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                f"MCPServerConfig {self.name!r}: transport='stdio' requires 'command' to be set."
            )
        if self.transport == "sse" and not self.url:
            raise ValueError(
                f"MCPServerConfig {self.name!r}: transport='sse' requires 'url' to be set."
            )
        return self


class ModelConfig(BaseModel):
    """LLM endpoint and model name (OpenAI chat-completions API). Defaults: Ollama."""
    base_url: str = Field(..., description="e.g. http://localhost:11434/v1 or http://localhost:8000/v1")
    model: str = Field(..., description="Model name (e.g. qwen2.5:7b for Ollama, or your server's model id)")
    api_key: str = Field("", description="Bearer token; empty for local backends (no header sent), set for cloud.")
    backend: str = Field(
        "ollama",
        description=(
            "LLM client backend to use. "
            "'ollama' (default): Ollama-compatible client with 400-retry and tool-support detection. "
            "'generic': bare OpenAI-compatible client for cloud providers (OpenAI, Anthropic via "
            "LiteLLM bridge, vLLM, LM Studio, etc.). "
            "'vllm': vLLM-specific client for high-throughput concurrent inference."
        ),
    )
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout_s: float = Field(default=360.0, description="HTTP timeout for chat request (read). Large models may need 300–600s.")


class SpecialistConfig(BaseModel):
    """Specialist pack definition in config."""
    description: str
    keywords: List[str] = Field(
        default_factory=list,
        description="Deprecated. Retained for config backward-compatibility; unused by routing.",
    )
    workflow: str = Field(
        "",
        description="Reserved for future use; today we use specialist_id to load pack.",
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description=(
            "Capability IDs this pack can provide "
            "(e.g. 'code_execution', 'systematic_review', 'web_search'). "
            "Used for required_capabilities derivation in orchestration plans."
        ),
    )
    tools: Optional[List[str]] = Field(
        None,
        description=(
            "Tool names for this specialist's pack (from tool_catalog). "
            "When set, the specialist is built dynamically from these tools "
            "instead of using a built-in template."
        ),
    )
    builder: Optional[str] = Field(
        None,
        description=(
            "Dotted import path to the pack factory function, "
            "e.g. 'mypackage.packs.custom:build_custom_pack'. "
            "Signature: (workspace_path: str, network_allowed: bool) -> SpecialistPack. "
            "When omitted the built-in pack for this specialist id is used."
        ),
    )
    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description=(
            "MCP tool servers to attach to this specialist pack. "
            "Tools from each server are merged into the pack's tool_definitions "
            "and dispatched via the MCP session at runtime."
        ),
    )
    container_image: Optional[str] = Field(
        None,
        description=(
            "Podman container image to run the specialist's shell tool inside, "
            "e.g. 'python:3.12-slim'. When set, the registry wraps the pack with "
            "ContainerisedSpecialistPack so every 'shell' tool call executes inside "
            "an isolated Podman container with the workspace mounted at /workspace. "
            "Requires Podman to be installed and the image to be available locally. "
            "Default: None (no container; shell commands run on the host)."
        ),
    )

    @model_validator(mode="after")
    def _check_mcp_server_names_unique(self) -> "SpecialistConfig":
        names = [s.name for s in self.mcp_servers]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate MCP server names in specialist config: {duplicates}. "
                "Each MCP server must have a unique 'name'."
            )
        return self


class CloudFallbackConfig(BaseModel):
    """Configuration for cloud LLM fallback (P6-4).

    When set on ``ConciergeConfig``, ``execute_task`` wraps the chat client with
    ``FallbackChatClient``.  The local model is tried first; if the configured
    policy triggers, the same call is re-issued to the cloud model instead.

    Fallback is *explicit* — this config must be present and correctly wired
    for any fallback to occur.  Absent config = identical behaviour to today.
    """

    model_key: str = Field(
        ...,
        description=(
            "Key into ``config.models`` for the cloud fallback model "
            "(e.g. 'cloud_quality').  Must exist in models when cloud_fallback is used."
        ),
    )
    policy: str = Field(
        "no_tool_calls",
        description=(
            "Trigger condition for falling back to cloud:\n"
            "  'no_tool_calls' — local model returned plain text with no tool calls.\n"
            "  'malformed_args' — a tool call has malformed JSON arguments (``_raw`` key).\n"
            "  'always' — always use cloud (for debugging/testing only).\n"
            "Unknown values are silently treated as 'never trigger' (no fallback)."
        ),
    )


class RunIndexConfig(BaseModel):
    """Configuration for the cross-run index (P6-1 keyword search; P7-1 semantic search;
    P11-6 ChromaDB vector store backend).

    When ``embedding_model`` is set, each run is embedded at write time and
    ``semantic_search_index()`` ranks by cosine similarity instead of substring
    matching.  When ``None`` (default), the index uses keyword/substring search
    with no extra dependencies.

    Set ``provider="chromadb"`` to store and search embeddings via ChromaDB
    (requires ``pip install 'agentic-concierge[embed]'``).  The default
    ``provider="jsonl"`` uses the existing JSONL + cosine scan path.
    """

    embedding_model: Optional[str] = Field(
        None,
        description=(
            "Ollama embedding model name, e.g. 'nomic-embed-text'. "
            "When set, each run entry is embedded at write time and semantic "
            "search (cosine similarity) is used instead of keyword matching. "
            "When None (default), keyword/substring search is used."
        ),
    )
    embedding_base_url: Optional[str] = Field(
        None,
        description=(
            "Base URL for the Ollama embeddings endpoint, e.g. 'http://localhost:11434'. "
            "When None, derived from the primary (fast/quality) model's base_url by "
            "stripping any /v1 suffix. Usually does not need to be set explicitly."
        ),
    )
    provider: str = Field(
        "jsonl",
        description=(
            "'jsonl' (default): JSONL flat file with optional in-process cosine scan. "
            "'chromadb': persist embeddings in ChromaDB for scalable ANN search "
            "(requires pip install 'agentic-concierge[embed]')."
        ),
    )
    chromadb_path: str = Field(
        "",
        description=(
            "ChromaDB storage directory path. "
            "Empty string (default) uses the OS-appropriate platformdirs "
            "user_data_path('agentic-concierge')/chromadb directory."
        ),
    )
    chromadb_collection: str = Field(
        "agentic_concierge_runs",
        description="ChromaDB collection name for the run index.",
    )


class FeaturesConfig(BaseModel):
    """Per-feature overrides for the auto-detected profile.

    Each field is ``Optional[bool]``:
    - ``None`` (default): use the profile's default for this feature.
    - ``True``: force-enable even if the profile disables it.
    - ``False``: force-disable even if the profile enables it.
    """

    ollama: Optional[bool] = None
    llama_cpp: Optional[bool] = None
    vllm: Optional[bool] = None
    cloud: Optional[bool] = None
    mcp: Optional[bool] = None
    browser: Optional[bool] = None
    embedding: Optional[bool] = None
    telemetry: Optional[bool] = None
    container: Optional[bool] = None


class ResourceLimitsConfig(BaseModel):
    """Resource caps for the concierge runtime."""

    max_concurrent_agents: int = Field(
        4,
        description="Maximum number of specialist agents to run in parallel.",
    )
    max_ram_mb: Optional[int] = Field(
        None,
        description="Hard cap on RAM usage in MB. None = no cap (use system total).",
    )
    max_gpu_vram_mb: Optional[int] = Field(
        None,
        description="Hard cap on GPU VRAM usage in MB. None = no cap.",
    )
    model_cache_path: str = Field(
        "",
        description=(
            "Directory for downloaded model weights. "
            "Empty string = use platformdirs user_data_path('agentic-concierge')."
        ),
    )


class TelemetryConfig(BaseModel):
    """Optional OpenTelemetry tracing configuration."""
    enabled: bool = False
    service_name: str = "agentic-concierge"
    exporter: str = Field(
        "none",
        description="Span exporter: 'none' (default), 'console' (stdout), or 'otlp' (gRPC endpoint).",
    )
    otlp_endpoint: str = Field(
        "",
        description="OTLP gRPC endpoint, e.g. 'http://localhost:4317'. Required when exporter='otlp'.",
    )


class ConciergeConfig(BaseModel):
    """Root config: models and specialists."""
    models: Dict[str, ModelConfig]
    specialists: Dict[str, SpecialistConfig]
    telemetry: Optional[TelemetryConfig] = None
    profile: str = Field(
        "auto",
        description=(
            "System profile tier: 'auto' (detect from hardware), 'nano', 'small', "
            "'medium', 'large', or 'server'. 'auto' runs system_probe() on first launch."
        ),
    )
    features: FeaturesConfig = Field(
        default_factory=FeaturesConfig,
        description=(
            "Per-feature overrides. Each field is True (force-enable), False (force-disable), "
            "or None (use profile default). See FeaturesConfig for available flags."
        ),
    )
    resource_limits: ResourceLimitsConfig = Field(
        default_factory=ResourceLimitsConfig,
        description="Resource caps for the concierge runtime.",
    )
    run_index: RunIndexConfig = Field(
        default_factory=RunIndexConfig,
        description=(
            "Run index configuration. Controls how the cross-run memory index is "
            "written and searched. Defaults to keyword-only search (no embedding). "
            "Set embedding_model to enable semantic search."
        ),
    )
    cloud_fallback: Optional[CloudFallbackConfig] = Field(
        None,
        description=(
            "Cloud LLM fallback configuration.  When set, the local model is tried first "
            "and the cloud model is used if the configured policy triggers.  Off by default "
            "(None) — identical behaviour to prior versions when absent."
        ),
    )

    @model_validator(mode="after")
    def _specialists_not_empty(self) -> "ConciergeConfig":
        """Validate that at least one specialist is defined.

        An empty specialists dict means every recruit_specialist() call fails
        at execution time with an opaque KeyError.  Catching this at config
        load time gives a clear, early error message.

        As new fields that reference specialist IDs are added to ConciergeConfig
        (e.g. a future ``default_specialist`` or routing rules), add those
        cross-reference checks here.
        """
        if not self.specialists:
            raise ValueError(
                "specialists must not be empty: at least one specialist must be defined. "
                "Add an 'engineering' or 'research' entry (or a custom pack) to your config."
            )
        return self
    require_human_approval_for: List[str] = Field(
        default_factory=lambda: ["deploy", "push", "write_external"]
    )
    approval_timeout_s: float = Field(
        600.0,
        description=(
            "Timeout in seconds for human approval requests. "
            "If no response within this time, the request is denied (fail-closed)."
        ),
    )
    # Local LLM is default and primary: ensure it's available (start if needed) by default.
    local_llm_ensure_available: bool = Field(
        True,
        description="If True (default), ensure local LLM is reachable; start it via local_llm_start_cmd when unreachable. Set False if you manage the server yourself.",
    )
    local_llm_start_cmd: List[str] = Field(
        default_factory=lambda: ["ollama", "serve"],
        description="Command to start the local LLM server when unreachable (e.g. ollama serve).",
    )
    local_llm_start_timeout_s: int = Field(
        90,
        description="Seconds to wait for local LLM server to become ready after start.",
    )
    # vLLM auto-start (ADR-033 OPS-1)
    vllm_ensure_available: bool = Field(
        False,
        description=(
            "If True, ensure vLLM is running on SERVER/LARGE profiles with GPU. "
            "Mirrors local_llm_ensure_available for Ollama. Default False because "
            "vLLM requires a GPU and is not universally available."
        ),
    )
    vllm_start_cmd: List[str] = Field(
        default_factory=lambda: [
            "python", "-m", "vllm.entrypoints.openai.api_server",
        ],
        description="Command to start vLLM. Model is appended as --model <model>.",
    )
    vllm_model: str = Field(
        "",
        description=(
            "Model to serve with vLLM. When empty, auto-selects the best available "
            "model from Ollama's model list. Set explicitly for deterministic startup."
        ),
    )
    vllm_start_timeout_s: int = Field(
        120,
        description="Seconds to wait for vLLM server to become healthy after start.",
    )
    vllm_gpu_memory_utilization: float = Field(
        0.9,
        description="Fraction of GPU memory for vLLM (--gpu-memory-utilization). Default 0.9.",
    )
    auto_pull_if_missing: bool = Field(
        True,
        description="If True (default), when no models are available at the configured backend (e.g. Ollama), pull auto_pull_model so one command works.",
    )
    auto_pull_model: str = Field(
        "qwen2.5:7b",
        description="Model to pull when auto_pull_if_missing is True and no models are available.",
    )
    routing_model_key: str = Field(
        "fast",
        description=(
            "Key into 'models' used for the LLM routing call (llm_recruit_specialist). "
            "Defaults to 'fast' so routing uses a lightweight model rather than the task "
            "model. Falls back to the task model if the key is not present in 'models'."
        ),
    )
    task_force_mode: str = Field(
        "sequential",
        description=(
            "Execution mode for multi-pack task forces.\n"
            "  'sequential' (default): packs run one after another; each pack receives "
            "    context (finish payload) from the previous pack.\n"
            "  'parallel': all packs run concurrently via asyncio.gather; each gets the "
            "    original task prompt with no inter-pack context forwarding; results are "
            "    merged into a combined payload with 'pack_results' key."
        ),
    )


# Default: Ollama on localhost:11434
DEFAULT_CONFIG = ConciergeConfig(
    models={
        "fast": ModelConfig(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
            temperature=0.1,
            max_tokens=1200,
        ),
        "quality": ModelConfig(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:14b",
            temperature=0.1,
            max_tokens=2400,
        ),
    },
    specialists={
        "engineering": SpecialistConfig(
            description="Plan → implement → test → review → iterate.",
            keywords=["build", "implement", "code", "service", "pipeline", "kubernetes", "gcp", "scala", "rust", "python"],
            workflow="engineering",
            capabilities=["code_execution", "file_io", "software_testing"],
        ),
        "research": SpecialistConfig(
            description="Scope → search → screen → extract → synthesize.",
            keywords=["literature", "systematic review", "paper", "arxiv", "survey", "bibliography", "citations"],
            workflow="research",
            capabilities=["systematic_review", "web_search", "citation_extraction", "file_io"],
        ),
        "enterprise_research": SpecialistConfig(
            description=(
                "Enterprise search: GitHub, Confluence, Jira, and internal sources "
                "via MCP. Produces structured reports with staleness/confidence notes."
            ),
            keywords=["confluence", "jira", "github", "internal docs", "knowledge base", "enterprise"],
            workflow="enterprise_research",
            capabilities=["enterprise_search", "github_search", "systematic_review", "web_search", "file_io"],
            # MCP servers are wired here in production configs.
            # Example (add to your CONCIERGE_CONFIG_PATH file):
            #   mcp_servers:
            #     - name: github
            #       transport: stdio
            #       command: npx
            #       args: ["--yes", "--", "@modelcontextprotocol/server-github"]
            #       env: {GITHUB_TOKEN: "${GITHUB_TOKEN}"}
        ),
    },
)
