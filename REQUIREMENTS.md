# agentic-concierge: Requirements and Validation

This document describes the **current system capabilities** as of v0.3.52. For the
long-term vision, principles, and phasing see [docs/VISION.md](docs/VISION.md).

## Purpose

agentic-concierge is a **quality-first agent fabric** for local inference, **built for Ollama**:

1. **Plans and decomposes** user tasks into a directed acyclic graph of sub-tasks via a planner+critic agent loop, selecting tools and roles, and assigning specialist packs (automatically or via explicit `--pack`).
2. **Runs** each specialist in a tool-calling loop until the task is completed, with quality gates (Gates 1–4) and an independent reviewer.
3. **Produces** a per-run directory with a structured runlog and a workspace of artifacts.

We **use Ollama** for local inference by default (default config points at localhost:11434 and Ollama model names). Local LLM is the **default and primary** path: the fabric ensures it's available (including starting it when unreachable) by default; opt out with `local_llm_ensure_available: false` if you manage the server yourself. Other backends (vLLM, in-process via mistral.rs, generic OpenAI-compatible) are supported via config. Cloud is used only when local **capability or quality** is insufficient (via explicit `cloud_fallback` config).

---

## Functional Requirements

### FR1: CLI and API

- **FR1.1** The CLI shall provide:
  - `concierge run PROMPT` — run a task (with optional `--pack`, `--model-key`, `--no-network-allowed`, `--stream`/`-s`, `--auto-approve`).
  - `concierge serve` — run the HTTP API.
  - `concierge logs list [--workspace] [--limit]` — list past runs.
  - `concierge logs show RUN_ID [--workspace] [--kinds]` — show runlog events.
  - `concierge logs search QUERY [--workspace] [--limit]` — keyword or semantic search over past runs.
  - `concierge plan PROMPT` — preview the planner's task decomposition without executing.
  - `concierge doctor` — show hardware profile, backend health, and feature flags.
  - `concierge bootstrap [--profile] [--non-interactive]` — detect hardware, configure profile, pull models.
- **FR1.2** The HTTP API shall expose:
  - `GET /health` returning `{"ok": true}`.
  - `POST /run` accepting `{ "prompt", "pack?", "model_key?", "network_allowed?" }` — blocking; returns finish_task payload + `_meta`.
  - `POST /run/stream` — SSE streaming of all run events until completion.
  - `GET /runs/{id}/status` — returns `completed`, `running`, or 404.
  - `POST /runs/{id}/approve` — submit human approval/denial response for a pending approval request.

### FR2: Routing and packs

- **FR2.1** If `--pack` is not specified, the **planner agent** decomposes the task:
  1. A planner+critic loop produces a `TaskGraph` (DAG of `TaskNode`s) with required capabilities, tool requirements, and finish schema per node.
  2. Leaf nodes are resolved to specialist packs via capability matching — may use **template packs** (engineering, research, enterprise_research) or **dynamic packs** (custom tool selection + role).
  3. On any planner error, falls back to a single-node graph with the first available template (zero regression).
  4. The plan's `reasoning`, `critique_feedback`, and graph structure are logged in the runlog as a `plan_result` event and included in the HTTP `_meta` response field.
- **FR2.2** Specialist packs are composed from a **central tool catalog** (8 tools: shell, read_file, write_file, list_files, run_tests, web_search, fetch_url, cross_run_search) and composable system prompt fragments. Three **template packs** provide curated defaults:
  - **engineering**: tools = shell, read_file, write_file, list_files, run_tests; workflow = plan → implement → test → review → iterate; quality gate requires `tests_verified`.
  - **research**: tools = web_search, fetch_url, write_file, read_file, list_files; workflow = scope → search → screen → extract → synthesize. Web tools only when `network_allowed=True`.
  - **enterprise_research**: tools = cross_run_search, web_search, fetch_url, read_file, write_file, list_files; workflow adds staleness/confidence assessment; cross-run memory via run index.
- **FR2.3** The orchestrator may also compose **dynamic packs** by selecting any subset of catalog tools and providing a role description. This allows task-specific specialist configurations without code changes.

### FR3: Execution

- **FR3.1** Each run shall create a unique run directory under `workspace_root/runs/<run_id>` and a `workspace` subdirectory for artifacts.
- **FR3.2** All LLM requests/responses and tool calls/results shall be appended to `runlog.jsonl` in the run directory.
- **FR3.3** The LLM client shall call the configured base URL at `/chat/completions` with the configured model name and parameters (temperature, top_p, max_tokens).
- **FR3.4** Multi-node task forces: when the planner creates multiple leaf nodes, sibling nodes under the same parent execute in parallel (via `asyncio.gather` in the graph executor). Completed node results are forwarded as context to subsequent siblings.
- **FR3.5** Result synthesis: when a task graph has multiple leaf nodes, a final LLM call with `synthesise_results` tool combines their outputs into a unified response.

### FR4: Configuration

- **FR4.1** Default configuration shall use **Ollama** (base_url http://localhost:11434/v1, models e.g. qwen2.5:7b / qwen2.5:14b) and define two model profiles (`fast`, `quality`) and three template specialists with their workflows. The fabric shall **ensure the local LLM is available by default** (check reachability; if unreachable, start via `local_llm_start_cmd` and wait for readiness); config may set `local_llm_ensure_available: false` to opt out.
- **FR4.2** If `CONCIERGE_CONFIG_PATH` is set to a valid file path, that file (JSON or YAML) shall be loaded and used as the fabric config; otherwise defaults are used.
- **FR4.3** Configuration supports:
  - `models`: keyed model configs with backend type (`ollama`, `generic`, `vllm`, `inprocess`), base_url, model name, API key, and parameters.
  - `specialists`: keyed specialist configs with description, capabilities, optional `tools` list (for dynamic packs), optional `builder` (custom factory), `mcp_servers`, and `container_image`.
  - `profile`: hardware profile tier (`auto`, `nano`, `small`, `medium`, `large`, `server`).
  - `features`: per-feature overrides (inprocess, ollama, vllm, cloud, mcp, browser, embedding, telemetry, container).
  - `resource_limits`: max_concurrent_agents, max_ram_mb, max_gpu_vram_mb.
  - `run_index`: embedding model, provider (`jsonl`/`chromadb`), ChromaDB settings.
  - `cloud_fallback`: model_key + policy (`no_tool_calls`/`malformed_args`/`always`).
  - `telemetry`: enabled flag, exporter (`none`/`console`/`otlp`), endpoint.
  - `routing_model_key`, `task_force_mode`, `require_human_approval_for`, `approval_timeout_s`.

### FR5: Quality and safety

- **FR5.1** Quality gates (enforced in `_execute_pack_loop`):
  - **Gate 1**: The LLM must have called at least one non-finish tool before `finish_task` is accepted.
  - **Gate 2**: All required fields in the `finish_task` payload must be present (derived from the pack's finish tool schema).
  - **Gate 3**: Pack-specific validation via `validate_finish_payload()` — e.g. engineering rejects `tests_verified=False`.
  - **Gate 4 (Review)**: An independent reviewer LLM inspects the workspace using read-only tools (`read_file`, `list_files`, `run_tests`) plus `approve_work`/`request_revision` decision tools. Fail-open: plain text = approval; errors = approval; max 2 rejections then accept with warning.
- **FR5.2** Engineering pack: the agent must not claim success without having run tests via `run_tests` tool; deploy/push steps must be proposed for human approval and not executed automatically.
- **FR5.3** Research pack: only URLs actually fetched via `fetch_url` may be cited; screening log and evidence table shall be maintained in the workspace.
- **FR5.4** When `network_allowed` is false, web tools (web_search, fetch_url) are excluded from the pack's tool definitions entirely; if invoked despite exclusion, they return a clear "network disabled" response.

### FR6: Sandbox and tools

- **FR6.1** File tools (read_file, write_file, list_files) shall be scoped to the run's workspace directory; paths must not escape the sandbox. Absolute paths are rejected with a clear error message and hint.
- **FR6.2** Shell commands shall be restricted to an allowlist (python, pytest, bash, git, pip, make, cargo, npm, …) and run with cwd within the workspace.
- **FR6.3** `run_tests` tool: auto-detects test framework (pytest, cargo, npm, unittest) and returns structured pass/fail results.

### FR7: LLM error recovery

- **FR7.1** Corrective re-prompt: up to 2 plain-text (no tool call) LLM responses trigger a corrective re-prompt nudging the LLM to use tools; after the limit, text is treated as final payload.
- **FR7.2** Loop detection: if the same (tool, args) signature appears ≥2 times in the last 8 tool calls, a `[SYSTEM] LOOP DETECTED` message is injected; `loop_detected` event emitted to runlog.
- **FR7.3** Timeout recovery: when a model times out, the system attempts to fall back to a smaller available model.
- **FR7.4** String-typed numerics: tool entry points coerce `timeout_s` from string to int/float, defending against small LLMs sending `"120"` instead of `120`.
- **FR7.5** Adaptive escalation: when quality failure signals accumulate (plain-text exhaustion, persistent loops, review rejection at max), the system escalates to a larger available model and continues the conversation. Each trigger type fires at most once; total escalations capped at `max_escalations` (default 2). Bidirectional convergence with timeout recovery: quality issues → go bigger, timeouts → go smaller.

### FR8: MCP tool servers

- **FR8.1** Any specialist can be augmented with MCP tool servers via `mcp_servers` config. Tools from MCP servers are merged into the pack's tool definitions with `mcp__<server_name>__<tool>` naming.
- **FR8.2** MCP sessions support `stdio` (subprocess) and `sse` (HTTP) transports with async lifecycle (`aopen`/`aclose`).
- **FR8.3** The MCP augmentation is transparent: no pack factory changes required.

### FR9: Containerised execution

- **FR9.1** When `container_image` is set on a specialist config, the registry wraps the pack with `ContainerisedSpecialistPack`. All `shell` tool calls execute inside a Podman container with the workspace mounted at `/workspace`.
- **FR9.2** Container wrapping is applied after MCP augmentation (wrapping order: inner → MCP → container).

### FR10: Cross-run memory (run index)

- **FR10.1** On successful completion, each run's summary is appended to `run_index.jsonl` and optionally embedded for semantic search.
- **FR10.2** `concierge logs search` queries the index (keyword or semantic via Ollama embeddings / ChromaDB).
- **FR10.3** The `cross_run_search` tool allows specialists to query prior run results.

### FR11: Streaming and observability

- **FR11.1** `POST /run/stream` provides SSE streaming of all runlog events in real-time.
- **FR11.2** `concierge run --stream` renders events with Rich terminal formatting.
- **FR11.3** OpenTelemetry tracing (optional dep `[otel]`): `fabric.execute_task`, `fabric.llm_call`, `fabric.tool_call` spans.
- **FR11.4** `CONCIERGE_RATE_LIMIT=<n>` enables per-IP sliding-window rate limiting on the HTTP API (429 + Retry-After).

### FR12: Multi-backend LLM support

- **FR12.1** `ModelConfig.backend` selects the LLM client: `ollama` (default, with 400-retry and tool-support detection), `generic` (bare OpenAI-compatible), `vllm`, `inprocess` (mistral.rs via PyO3).
- **FR12.2** `cloud_fallback` config wraps the chat client with `FallbackChatClient`; triggers when the configured policy fires (e.g. local model returns no tool calls).
- **FR12.3** Adaptive backend resolution: `resolve_llm()` falls back through `BACKEND_PRIORITY[tier]` when the primary backend is unreachable. Probes backends in order; first to return models is used.
- **FR12.4** Model selection: `select_model()` sorts available models by closest parameter-size distance to the configured model, with same-family preference and tool-incapable model filtering.

### FR13: Hardware profiles and bootstrap

- **FR13.1** `SystemProbe` detects CPU/RAM/GPU/disk/network/backends; `ProfileTier` (nano/small/medium/large/server) determines feature flags and model recommendations.
- **FR13.2** `FeatureSet` gates capabilities: disabled features consume zero resources (no imports, no processes).
- **FR13.3** `concierge bootstrap` orchestrates first-run: probe → advise → ensure backends → pull models → write `detected.json`.
- **FR13.4** Browser tool (Playwright, optional): available when `Feature.BROWSER` is enabled; provides browse, click, fill, screenshot, extract_text, navigate.

### FR14: Distribution (Rust launcher)

- **FR14.1** Static ~5 MB Rust binary bootstraps the Python environment and exec-replaces itself with the Python `concierge` binary. No Python or pip required to get started.
- **FR14.2** `--self-update`: checks GitHub Releases for newer version, downloads, verifies Ed25519 signature (optional/best-effort), applies atomic binary replacement, upgrades pip package.
- **FR14.3** `CONCIERGE_EXTRA` support: ensures requested pip extras are installed on existing venvs.
- **FR14.4** Targets: Linux (x86_64/aarch64 musl static), macOS (x86_64/aarch64 apple-darwin).

### FR16: Human Approval (ADR-021)

- **FR16.1** A `request_approval` tool is available to all specialists during `_execute_pack_loop`. When invoked, execution pauses and blocks on an `ApprovalChannel` until a human responds or `approval_timeout_s` expires.
- **FR16.2** Three `ApprovalChannel` implementations:
  - `AutoApprovalChannel` — always approves immediately (for testing and CI).
  - `CliApprovalChannel` — prompts the user interactively at the terminal.
  - `HttpApprovalChannel` — blocks until `POST /runs/{run_id}/approve` is called.
- **FR16.3** CLI: `concierge run --auto-approve` uses `AutoApprovalChannel`; default is `CliApprovalChannel`.
- **FR16.4** HTTP API: `POST /runs/{run_id}/approve` accepts `{ "approved": bool, "reason"?: str }` and unblocks the waiting `HttpApprovalChannel`.
- **FR16.5** Config: `approval_timeout_s` on `ConciergeConfig` sets the maximum wait time for an approval response.
- **FR16.6** Runlog events: `approval_requested`, `approval_granted`, `approval_denied`.

### FR17: Agent Delegation (ADR-022)

- **FR17.1** A `delegate_to_specialist` tool is available to specialists during `_execute_pack_loop`. When invoked, a sub-specialist is resolved from the registry and a nested `_execute_pack_loop` is spawned.
- **FR17.2** Maximum delegation depth is 1 — a sub-specialist cannot delegate further.
- **FR17.3** Sub-specialist step budget is capped at 15 to prevent runaway sub-tasks.
- **FR17.4** The sub-specialist shares the parent's workspace and runlog. Delegation events (`delegation_start`, `delegation_complete`) are recorded.
- **FR17.5** The sub-specialist's finish payload is returned to the parent as a tool result.

---

## Validation

### Manual validation

1. **CLI help** — `concierge --help`, `concierge run --help`, `concierge serve --help` run without error.
2. **Routing** — `concierge run "build a small API"` creates a run dir and selects a specialist via the orchestrator (or fails at LLM call if no server running).
3. **Run output structure** — after any run (even failed), the run directory contains `runlog.jsonl` and `workspace/`.
4. **API** — `concierge serve` and `curl http://127.0.0.1:8787/health` returns `{"ok": true}`.
5. **Doctor** — `concierge doctor` shows hardware, profile tier, feature flags, backend health.

### End-to-end validation (real LLM server required)

6. **Engineering (real verification)** — `python scripts/verify_working_real.py` exits 0; runlog contains `tool_call` and `tool_result`; workspace has artifacts.
7. **Research** — `concierge run "Mini systematic review of post-quantum crypto performance." --pack research`; expect web_search/fetch_url calls, workspace files, citations.

### Automated tests

From the repo root with `pip install -e ".[dev]"`:

```bash
make test          # fast CI: 1377 tests (no real LLM/MCP/Podman)
make test-rust     # Rust launcher: 22 tests
```

- **Fast CI:** `make test` runs 1377 tests with all real-LLM, real-MCP, and Podman tests deselected. Use for quick feedback on wiring, contracts, and behaviour.
- **Real-LLM E2E:** `test_execute_task_engineering_real_llm`, `test_execute_task_research_pack_real_llm`, `test_api_post_run_real_llm`, `test_verify_working_real_script`. Essential for integration assurance — must run and pass with a real LLM for full validation.
- **Real-MCP:** `pytest tests/ -k real_mcp -v` (requires npx + optional GITHUB_TOKEN).
- **Podman:** `pytest tests/ -k podman -v` (requires Podman + pulled image).
- **Rust launcher:** `cargo test --manifest-path launcher/Cargo.toml`.
- **Lint:** `ruff check src/ tests/ --select E,W,F --ignore E501,F401`.

### FR18: Model Runtime (V2 — Layer 1)

- **FR18.1** `LocalModelRuntime` provides `acquire(requirements) → ModelHandle` and `release(handle)` for model lifecycle management with reference counting and LRU eviction.
- **FR18.2** `BackendRegistry` discovers and monitors inference backends (Ollama, LlamaCpp) with health checks and priority-based failover. Configuration via `config/defaults/backends.yaml`.
- **FR18.3** `InferenceBackend` protocol defines: `load_model()`, `unload_model()`, `build_client()`, `estimate_memory()`, `list_available()`, `health_check()`.
- **FR18.4** `OllamaBackend` manages model lifecycle via Ollama API (keep_alive for load/unload, list loaded/available).
- **FR18.5** `LlamaCppBackend` manages llama.cpp server processes: one process per model, dynamic port allocation, process lifecycle.
- **FR18.6** `CapabilityProbe` validates model capabilities via micro-prompt probes (tool_calling, structured_output, instruction_following). Results cached to disk.
- **FR18.7** Memory-aware model management: tracks loaded model sizes, respects system RAM budget, evicts least-recently-used models when budget is exceeded.

### FR19: Recursive Task Decomposition (V2 — Layer 2)

- **FR19.1** `TaskGraph` provides a DAG of `TaskNode` objects replacing the flat `OrchestrationPlan`. Nodes track status via a state machine: pending → decomposing → critiqued → executing → reviewing → done/failed.
- **FR19.2** Planner agent decomposes root tasks into subtasks with `required_capabilities`, `required_tools`, and `finish_schema_key` per subtask.
- **FR19.3** Critic agent reviews decomposition plans. Approves or rejects with feedback; maximum 2 re-plans before accepting.
- **FR19.4** `execute_graph()` runs ready leaf nodes in parallel via `asyncio.gather`, marks nodes done/failed, and propagates completion upward through the DAG.
- **FR19.5** Adaptive depth control: `should_decompose()` stops recursion when leaf fits available model or max depth (3) is reached.

### FR20: Agent-Model Affinity (V2 — Layer 3)

- **FR20.1** Six agent roles (router, planner, critic, coder, researcher, reviewer) define capability requirements as `Dict[str, float]` (e.g., `{"reasoning": 0.8, "structured_output": 0.7}`).
- **FR20.2** `assign_model()` selects the best model for a role using `ModelRuntime.acquire()` with capability matching and `must_differ_from` constraints (e.g., reviewer must use a different model than the doer).
- **FR20.3** `execute_graph_with_affinity()` wraps the graph executor: for each leaf node, determines role → assigns model → executes work → releases handle.
- **FR20.4** Model preloading: `preload_hint()` fires background model loads for upcoming sibling nodes while the current node executes. Fire-and-forget via `asyncio.ensure_future`.
- **FR20.5** `must_differ_from` is fail-open: when no alternative model is available, the constraint is relaxed and the same model is used.

### FR21: System Diagnostics (V2)

- **FR21.1** `run_diagnostics()` runs 6 checks: config loadable, backends.yaml valid, memory budget sufficient, backend registry health, model availability, agent roles loadable.
- **FR21.2** `DiagnosticReport` provides `all_passed`, `failed_checks`, and `summary()` for structured reporting.

---

## Out of scope (current)

- Windows launcher binary (Phase 15).
- Web UI, multi-tenant auth, plugin registry (Phase 17+).
