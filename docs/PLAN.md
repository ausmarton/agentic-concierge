# agentic-concierge: Iterative Build Plan

This document defines **phases**, **deliverables**, and **verification gates** so the system can be built incrementally and any session (human or agent) can **resume** with full context.

---

## Resumability: how to pick up work across restarts

**When you (or another agent) start or resume work on this repo:**

1. **Read [STATE.md](STATE.md)** — current phase, what’s done, what’s next, last updated.
2. **Read this PLAN** — at least the section for the phase in STATE (e.g. Phase 1).
3. **Run the verification for that phase** (see “Verification” below and per-phase gates). If something fails, fix it before adding new work.
4. **Proceed** with the next unchecked deliverable for the current phase, or move to the next phase if the current one is complete.

**Key docs:**

| Doc | Purpose |
|-----|--------|
| [STATE.md](STATE.md) | Single source of truth: current phase, completed items, next steps. **Update STATE when you complete or start work.** |
| [PLAN.md](PLAN.md) (this file) | Phases, deliverables, verification criteria. |
| [VISION.md](VISION.md) | Long-term vision, principles, use cases (illustrative). |
| [../REQUIREMENTS.md](../REQUIREMENTS.md) | Functional requirements and validation for the MVP. |
| [../README.md](../README.md) | User-facing quickstart and usage. |

---

## Verification strategy (checks we run)

- **Automated (every change / before merge):**
  - `pytest tests/ -v` — all tests pass.
  - Lint (if configured): e.g. `ruff check src/agentic_concierge`.
- **Manual (per phase or before marking phase complete):**
  - CLI: `concierge --help`, `concierge run --help`; `concierge run "…" --pack engineering` (with or without LLM server) behaves as in REQUIREMENTS.
  - API: `concierge serve` + `curl http://127.0.0.1:8787/health`.
  - Run structure: `.concierge/runs/<id>/runlog.jsonl` and `workspace/` exist after a run.
- **E2E (when LLM server available):**
  - One engineering run and one research run as in REQUIREMENTS “End-to-end validation”; inspect artifacts and runlog.

**Rule:** Do not mark a phase complete until its verification gate passes. Update STATE.md when you run verification or complete deliverables.

---

## Phase 1: Solid MVP (current baseline)

**Goal:** A working fabric with one-pack-per-run recruitment, engineering and research packs, local LLM only, and enough tests and docs to iterate safely.

**What Phase 1 delivers (outcomes):**
- You can run `concierge run "your prompt"` (or `--pack engineering` / `--pack research`) and get a run directory with a structured runlog and workspace; the router picks a pack when you don’t specify one.
- You can run `concierge serve` and hit `GET /health` and `POST /run` to drive the same behaviour over HTTP.
- Config is default + optional file via `CONCIERGE_CONFIG_PATH`; model params (temperature, max_tokens) are passed to the LLM.
- Engineering and research packs each have tools and workflows; quality gates (no “works” without tests, deploy proposed only, citations only from fetch) are in the prompts.
- Sandbox keeps file and shell operations scoped and safe; automated tests plus a clear verification gate prove the above.

### Deliverables (Phase 1)

| # | Deliverable | Where it lives | Verification |
|---|-------------|----------------|--------------|
| 1.1 | CLI: `concierge run`, `concierge serve`, options | `src/agentic_concierge/interfaces/cli.py` | `concierge --help`, `concierge run --help` |
| 1.2 | HTTP API: `/health`, `POST /run` | `src/agentic_concierge/interfaces/http_api.py` | `curl .../health`, POST with prompt |
| 1.3 | Config: defaults + optional file via `CONCIERGE_CONFIG_PATH` | `src/agentic_concierge/config/` | Config load test; env override |
| 1.4 | Recruit: keyword scoring + fallback (engineering vs research) | `src/agentic_concierge/application/recruit.py` | `tests/test_router.py` |
| 1.5 | Execute task: run dir, workspace, runlog, one pack per run | `src/agentic_concierge/application/execute_task.py` | Run once; check run dir structure |
| 1.6 | Engineering specialist: tools + prompts | `src/agentic_concierge/infrastructure/specialists/engineering.py` | Run with `--pack engineering`; runlog has tool_call |
| 1.7 | Research specialist: tools + prompts | `src/agentic_concierge/infrastructure/specialists/research.py` | Run with `--pack research`; `network_allowed` gates web tools |
| 1.8 | Sandbox: path safety, shell allowlist | `src/agentic_concierge/infrastructure/tools/sandbox.py` | `tests/test_sandbox.py` |
| 1.9 | Runlog and model params passed to LLM | `src/agentic_concierge/infrastructure/workspace/run_log.py`; execute_task uses `model_cfg` | Runlog exists; temperature/max_tokens in use |
| 1.10 | Quality gates in prompts (no “works” without tests; deploy proposed only; citations only from fetch) | Workflow system rules, REQUIREMENTS FR5 | README + REQUIREMENTS |
| 1.11 | Automated tests for router, sandbox, json_tools, prompts, config, packs | `tests/` | `pytest tests/ -v` |
| 1.12 | Docs: README, REQUIREMENTS, VISION, PLAN, STATE | Various | All referenced docs exist and linked |
| 1.13 | Local LLM default and core (ensure available by default) | Config + ensure_llm_available in CLI/API; opt-out | local_llm_ensure_available: true by default; test_config, test_llm_bootstrap, test_backends_alignment |

### Phase 1 verification gate

- [x] **Full validation:** `python scripts/validate_full.py` passes (ensures real LLM, then all tests run including real-LLM E2E).
- [x] `concierge run “list files” --pack engineering` creates `.concierge/runs/<id>/runlog.jsonl` and `workspace/` (fails at LLM if no server; that’s OK).
- [x] `concierge serve` and `curl http://127.0.0.1:8787/health` return `{“ok”: true}`.
- [x] REQUIREMENTS.md “Manual validation” items 1–4 pass.

**Phase 1 acceptance (we're done when):** All 13 deliverables implemented; full validation (scripts/validate_full.py) run and passed so at least a couple of real-LLM E2E tests have run and passed (integration assurance); manual checks: CLI help, `concierge run` creates run dir + runlog + workspace, `concierge serve` + `/health` returns `{"ok": true}`. Update STATE.md to “Phase 1 complete” and set “Next: Phase 2”.

---

## Phase 2: Task decomposition and smarter routing

**Goal:** Move from “keyword router picks one pack” to “task is analysed → required capabilities → which pack(s) to recruit”. Still one pack per run as a first step, but the *decision* is capability-based and documented so we can later add multi-pack.

**What Phase 2 delivers (outcomes):** A capability model and config mapping packs to capabilities; for each run, required capabilities and selected pack(s) recorded in runlog or metadata; routing selects pack by capabilities (still one pack per run); tests and docs updated.

### Deliverables (Phase 2)

| # | Deliverable | Verification |
|---|-------------|--------------|
| 2.1 | **Capability model** — define capabilities (e.g. “data_engineering”, “systematic_review”, “web_search”) and map packs to capabilities in config. | Config schema + docs; router or new module uses it. |
| 2.2 | **Task → capabilities** — either (a) keyword/schema rules or (b) small router model + JSON schema that, given a task, outputs required capability IDs. | Unit tests; deterministic or model-based path. |
| 2.3 | **Recruitment** — select pack(s) that cover required capabilities (for Phase 2: still single pack; multi-pack in Phase 3). | Router returns one pack; log “required capabilities” and “selected pack”. |
| 2.4 | **Runlog / observability** — log “task”, “required_capabilities”, “selected_pack(s)” in run metadata or runlog. | Inspect runlog or result `_meta`. |
| 2.5 | **Docs** — update VISION §8 alignment and REQUIREMENTS to describe capability-based routing. | STATE and PLAN updated. |

**Phase 2 acceptance (we’re done when):**
- Capability model and task→capabilities (rules or model) implemented; recruitment selects pack from capabilities.
- Run metadata or runlog includes required_capabilities and selected_pack; tests and manual run confirm.
- Phase 1 verification gate still passes.

### Phase 2 verification gate

- [ ] `pytest tests/ -v` passes (including any new tests for capabilities/routing).
- [ ] For a few prompts, run and confirm selected pack and (if implemented) required capabilities are logged.
- [ ] Phase 1 verification gate still passes.

---

## Phase 3: Multi-pack task force — **complete**

**Goal:** For a single task, recruit and run **multiple** packs (e.g. engineering + research) that together form a task force. Orchestration may be sequential or coordinated.

### Deliverables (Phase 3)

| # | Deliverable | Status | Verification |
|---|-------------|--------|--------------|
| 3.1 | Task decomposition outputs *multiple* capability IDs when needed. | Done | `infer_capabilities()` returns all matching caps; `_greedy_select_specialists()` covers multi-pack selection. `tests/test_task_force.py`. |
| 3.2 | Supervisor runs multiple packs; shared workspace + combined runlog. | Done | `execute_task()` loops over `specialist_ids`; one run dir; `pack_start` events; `specialist_ids` on `RunResult`. |
| 3.3 | Coordination: sequential handoff with context forwarding. | Done | Each pack receives previous pack's `finish_task` payload as context; step names prefixed by specialist ID. |
| 3.4 | Docs and STATE updated. | Done | BACKLOG.md Phase 3 section; STATE.md; this PLAN updated. |

**Phase 3 acceptance:** All 4 deliverables implemented; fast CI: **144 pass** (+22). `RecruitmentResult.is_task_force` and `RunResult.is_task_force` both work correctly for mixed-capability prompts.

---

## Phase 4: Observability and multi-backend LLM — **complete**

**Goal:** Add production-grade observability (OpenTelemetry spans), a `concierge logs` CLI for inspecting past runs, and a generic LLM client so cloud/enterprise LLM endpoints work without Ollama quirks. All optional/additive: no breaking changes to existing functionality.

### Deliverables (Phase 4)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 4.1 | Generic/cloud LLM client + `ModelConfig.backend` | Done | `infrastructure/chat/__init__.py` (`build_chat_client()` factory); `GenericChatClient` (no Ollama 400 retry); shared `parse_chat_response()` in `_parser.py`; `backend: str = "ollama"` on `ModelConfig`; CLI + HTTP API updated; 15 new tests |
| 4.2 | `concierge logs` CLI subcommand | Done | `logs list` (Rich table, sorted most-recent-first) + `logs show` (pretty JSON with `--kinds` filter); `RunSummary` dataclass + `list_runs()` + `read_run_events()` in `infrastructure/workspace/run_reader.py`; 18 new tests |
| 4.3 | OpenTelemetry tracing (optional dep) | Done | `infrastructure/telemetry.py` (`_NoOpSpan`, `_NoOpTracer`, `setup_telemetry()`, `get_tracer()`); graceful no-op when OTEL not installed; `TelemetryConfig` in config schema; `fabric.execute_task` / `fabric.llm_call` / `fabric.tool_call` spans; `[otel]` extra in `pyproject.toml`; wired into CLI + HTTP API lifespan; 13 new tests |
| 4.4 | Docs update | Done | BACKLOG.md Phase 4 section; STATE.md phase + CI count; PLAN.md Phase 4 concrete deliverables |

**Phase 4 acceptance:** All 4 deliverables implemented; fast CI: **194 pass** (+50 vs Phase 3). `ModelConfig.backend = "generic"` routes to `GenericChatClient`; `concierge logs list` shows past runs; OTEL spans emitted when `telemetry.enabled=true`.

---

## Phase 5: MCP tool server support — **complete**

**Goal:** Enable specialist packs to delegate tool calls to external MCP (Model Context Protocol) servers without writing custom Python adapters. Zero changes required to pack factories — just add `mcp_servers:` entries to the YAML/JSON config. The MCP session lifecycle (connect, call, disconnect) is managed by the framework around the tool loop.

### Deliverables (Phase 5)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 5.1 | Config schema: `MCPServerConfig` + `mcp_servers` on `SpecialistConfig` | Done | `config/schema.py`; stdio and SSE transports; validators; duplicate-name check; +6 tests in `test_config.py` |
| 5.2 | Async `execute_tool` + `aopen`/`aclose` lifecycle | Done | `base.py`, `ports.py`, `execute_task.py` (_execute_pack_loop wrapped in try/finally); `_StubPack` in registry tests updated |
| 5.3 | `MCPSessionManager` + `mcp_tool_to_openai_def` converter | Done | `infrastructure/mcp/session.py`, `converter.py`; guarded mcp imports; +12 mocked tests |
| 5.4 | `MCPAugmentedPack` wrapper | Done | `infrastructure/mcp/augmented_pack.py`; asyncio.gather for connect/disconnect; +10 tests |
| 5.5 | Registry wraps pack transparently | Done | `registry.py`; RuntimeError when mcp not installed; +6 tests in `test_mcp_registry.py` |
| 5.6 | `pyproject.toml` + docs | Done | `mcp = ["mcp>=1.0"]` optional dep; `dev` dep includes mcp; all docs updated |

**Phase 5 acceptance:** All 6 deliverables implemented; fast CI: **~242 pass** (+33 vs Phase 4). `SpecialistConfig(mcp_servers=[MCPServerConfig(...)])` in config causes `get_pack()` to return `MCPAugmentedPack`; tool loop calls `aopen()`/`aclose()`; MCP tools are prefixed `mcp__<name>__<tool>`.

---

## Phase 6: Containerisation, memory, and cloud fallback — **complete**

**Goal:** OS-level workspace isolation (Podman), cross-run memory (run index), end-to-end MCP verification, and cloud LLM fallback quality gate. All are additive; existing behaviour unchanged when features are not configured.

### Deliverables (Phase 6)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 6.1 | Persistent run index + `concierge logs search` | Done | `infrastructure/workspace/run_index.py`; append-only JSONL; keyword/substring `search_index()`; `concierge logs search <query>` CLI; `execute_task` appends on success; 9 tests |
| 6.2 | Real MCP server smoke test | Done | `tests/test_mcp_real_server.py` — 5 tests using `@modelcontextprotocol/server-filesystem` via `npx`; `real_mcp` marker; `podman`/`real_llm`/`real_mcp` all declared in `pyproject.toml` |
| 6.3 | Containerised workspace isolation (Podman) | Done | `infrastructure/specialists/containerised.py` — `ContainerisedSpecialistPack`; `podman run -d --rm -v workspace:/workspace:Z`; shell intercepted via `podman exec`; `SpecialistConfig.container_image`; registry wraps after MCP; 26 tests (22 unit + 4 real Podman) |
| 6.4 | Cloud LLM fallback | Done | `infrastructure/chat/fallback.py` — `FallbackPolicy` (no_tool_calls / malformed_args / always) + `FallbackChatClient` + `pop_events()`; `CloudFallbackConfig` + `cloud_fallback` on `ConciergeConfig`; `execute_task` auto-wraps + drains events + logs `cloud_fallback` runlog events; 21 tests |

**Phase 6 acceptance:** All 4 deliverables implemented; fast CI: **304 pass** (+47 vs Phase 5). `SpecialistConfig(container_image="python:3.12-slim")` wraps pack with `ContainerisedSpecialistPack`; `ConciergeConfig(cloud_fallback=CloudFallbackConfig(...))` auto-wraps chat client; `concierge logs search` returns matching prior runs.

---

## Phase 7: Enterprise RAG and integrations

**Goal:** Upgrade the run index from keyword to semantic search (vector embeddings via Ollama); integrate first real enterprise MCP server (GitHub); add an enterprise research pack that can search GitHub, Confluence, and Jira via MCP; strengthen cross-run memory for the orchestrator.

### Deliverables (Phase 7)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 7.1 | Semantic run index search (vector embeddings) | Done | `run_index.py`: `RunIndexEntry.embedding`; `embed_text()` via Ollama `/api/embeddings` (strips `/v1`); `cosine_similarity()`; `semantic_search_index()` with keyword fallback; `RunIndexConfig` + `run_index` on `ConciergeConfig`; `execute_task` embeds entry when configured; `concierge logs search` uses semantic when available; 22 tests |
| 7.2 | GitHub MCP integration + tests | Done | `tests/test_mcp_real_github.py` — 4 tests (list_tools, search_repositories, get_file_contents, unknown_tool); `github_search` + `enterprise_search` capabilities in `capabilities.py`; `docs/MCP_INTEGRATIONS.md` with GitHub/Confluence/Jira/filesystem config examples |
| 7.3 | Enterprise research specialist | Done | `infrastructure/specialists/enterprise_research.py` — `cross_run_search` tool (queries run index); staleness/confidence system prompt; `enterprise_search` + `github_search` capabilities; entry in `DEFAULT_CONFIG`; registry `_DEFAULT_BUILDERS` updated; 16 tests |
| 7.4 | Docs update | Done | STATE.md (phase 7 complete, CI 342); PLAN.md (this table); VISION.md §7 (Phase 7 in history, Phase 8+ planned) + §8 (enterprise integrations row updated) |

---

## Phase 8: Streaming, parallelism, and run status — **complete**

**Goal:** Add real-time SSE streaming of run events, parallel task force execution, and a run status endpoint. All additive; existing behaviour unchanged.

### Deliverables (Phase 8)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 8.1 | Parallel task force execution | Done | `ConciergeConfig.task_force_mode` ('sequential'/'parallel'); `_run_task_force_parallel()` + `_merge_parallel_payloads()` in `execute_task.py`; asyncio.gather for concurrent packs; errors per-pack (non-fatal); 14 tests in `test_parallel_task_force.py` |
| 8.2 | SSE run event streaming | Done | `event_queue: Optional[asyncio.Queue]` on `execute_task()`; `_emit()` helper mirrors every runlog event to queue; `run_complete` event written at end of every successful run; `_run_done_`/`_run_error_` sentinels terminate stream; `POST /run/stream` returns `text/event-stream`; 6 tests in `test_run_streaming.py` |
| 8.3 | Run status endpoint | Done | `GET /runs/{run_id}/status` — reads runlog for `run_complete` event; returns `completed`/`running`/404; no full scan needed; 6 tests in `test_run_status.py` |
| 8.4 | Docs update | Done | ARCHITECTURE.md (Phase 4-8 components, streaming flow, full runlog table); README.md (Phase 8 features, HTTP API, full CLI); CONTRIBUTING.md (new); LICENSE (MIT); PLAN.md Phase 8; VISION.md §7 updated |

**Phase 8 acceptance:** All 4 deliverables implemented; fast CI: **368 pass** (+26 vs Phase 7).

---

## Summary

- **Resume by:** Reading STATE.md → PLAN.md (current phase) → run verification → do next deliverable.
- **Always:** Keep STATE.md updated when completing or starting work; run `make test` before considering a phase done.
- **Value by phase:**
  - **Phase 1:** Working fabric with engineering + research packs, keyword router, CLI, HTTP API, sandbox, runlog.
  - **Phase 2:** Capability-based routing (task → capabilities → pack by coverage).
  - **Phase 3:** Multi-pack task forces with sequential context handoff.
  - **Phase 4:** Multi-backend LLM (`ModelConfig.backend`), `concierge logs` CLI, OpenTelemetry tracing.
  - **Phase 5:** MCP tool server support (`MCPAugmentedPack`, transparent wrapping).
  - **Phase 6:** Podman workspace isolation, cross-run memory (run index), cloud LLM fallback.
  - **Phase 7:** Semantic search (vector embeddings), GitHub MCP integration, enterprise research specialist.
  - **Phase 8:** Parallel task forces, SSE streaming, run status endpoint.
  - **Phase 9:** CLI streaming, corrective re-prompt, rate limiting.
  - **Phase 10:** Hardware profiles, bootstrap, three-layer inference, feature flags.
  - **Phase 11:** Browser tool (Playwright), ChromaDB vector store.
  - **Phase 12:** Quality gates (Gates 1–3), `run_tests` tool, LLM orchestrator, session continuation.
  - **Phase 13:** Rust thin launcher (static binary, self-update, `install.sh`).
  - **Phase 14:** Pure-Rust tar extraction, Ed25519 signed updates, macOS targets, hot-path analysis.
  - **Post-Phase-14:** Adaptive backend resolution, model selection fixes, dynamic pack composition, universal review mechanism (Gate 4).

---

## Phase 9: CLI streaming, LLM error recovery, rate limiting — **complete**

### Deliverables (Phase 9)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 9.1 | `concierge run --stream` (`-s`) — Rich terminal rendering of all run events in real-time | Done | `interfaces/cli.py`; `StreamRenderer` per event kind; asyncio task + queue drain |
| 9.2 | Corrective re-prompt — LLM plain-text response triggers up to 2 re-prompts before fallback | Done | `_execute_pack_loop` in `execute_task.py`; `corrective_reprompt` runlog event |
| 9.3 | `CONCIERGE_RATE_LIMIT=<n>` — per-IP sliding-window rate limiting, 429 + Retry-After | Done | `interfaces/http_api.py`; in-process deque per IP |
| 9.4 | Sandbox absolute-path error message | Done | `infrastructure/tools/sandbox.py`; clear message with "use relative path e.g. 'app.py'" hint |

**Phase 9 acceptance:** All 4 deliverables implemented; fast CI: **402 pass**.

---

## Phase 10: Self-sizing bootstrap, three-layer inference, profile-based features

**Goal:** The system detects its host environment on first run, selects an appropriate hardware
profile, starts an in-process tiny model immediately (zero setup), and configures Ollama and/or
vLLM as full backends. Features that are not enabled by the profile consume zero resources
(no imports, no processes, no RAM). After this phase, `concierge` works out of the box on any
hardware from a 4 GB RAM laptop to a multi-GPU server with no manual configuration required.

**See also:** ADR-012 (three-layer inference), ADR-013 (feature flags), ADR-014 (in-process
bootstrap), ADR-015 (vLLM first-class), ADR-016 (Rust launcher — Phase 13).

### New files

| File | Purpose |
|------|---------|
| `src/agentic_concierge/bootstrap/__init__.py` | Package init |
| `src/agentic_concierge/bootstrap/system_probe.py` | Detect CPU/RAM/GPU/disk/network; `SystemProbe`, `GPUDevice` |
| `src/agentic_concierge/bootstrap/model_advisor.py` | `SystemProfile`, `ProfileTier`; model + backend recommendations per tier |
| `src/agentic_concierge/bootstrap/backend_manager.py` | `BackendManager`, `BackendHealth`, `BackendStatus`; probe/start/monitor backends |
| `src/agentic_concierge/bootstrap/first_run.py` | `FirstRunBootstrap`; orchestrates first-run: probe → advise → install → pull → write detected.json |
| `src/agentic_concierge/bootstrap/detected.py` | Read/write `detected.json` via `platformdirs` |
| `src/agentic_concierge/config/features.py` | `Feature` enum, `PROFILE_FEATURES` mapping, `FeatureSet`, `FeatureDisabledError` |
| `src/agentic_concierge/infrastructure/chat/inprocess.py` | `InProcessChatClient` — mistral.rs via PyO3; lazy import of `mistralrs` |
| `src/agentic_concierge/infrastructure/chat/vllm.py` | `VLLMChatClient` — thin wrapper over OpenAI-compat HTTP + health check |

### Modified files

| File | Change |
|------|--------|
| `src/agentic_concierge/config/schema.py` | Add `profile`, `features: FeaturesConfig`, `resource_limits: ResourceLimitsConfig` to `ConciergeConfig`; add `FeaturesConfig`, `ResourceLimitsConfig` models |
| `src/agentic_concierge/infrastructure/chat/__init__.py` | Handle `"inprocess"` and `"vllm"` backends in `build_chat_client()` |
| `src/agentic_concierge/interfaces/cli.py` | Add `concierge doctor` and `concierge bootstrap` subcommands |
| `src/agentic_concierge/interfaces/http_api.py` | Initialise `BackendManager` in lifespan; store on app state |
| `pyproject.toml` | Add `psutil>=5.9`, `platformdirs>=4.0` to core deps; add `[nano]`, `[vllm]`, `[browser]`, `[embed]` extras |

### Deliverables (Phase 10)

| # | Deliverable | Where | Verification |
|---|-------------|-------|--------------|
| 10.1 | `SystemProbe` — detect CPU/RAM/GPU/disk/internet/backends | `bootstrap/system_probe.py` | `tests/test_system_probe.py` (15 tests, all mocked) |
| 10.2 | `ModelAdvisor` — `ProfileTier` + model/backend recommendations | `bootstrap/model_advisor.py` | `tests/test_model_advisor.py` (10 tests) |
| 10.3 | `BackendManager` — probe/start/health-check all three backends | `bootstrap/backend_manager.py` | `tests/test_backend_manager.py` (12 tests) |
| 10.4 | `FirstRunBootstrap` — first-run orchestrator | `bootstrap/first_run.py` | `tests/test_first_run.py` (10 tests); `concierge bootstrap --non-interactive` |
| 10.5 | `detected.py` — cross-platform detected.json via platformdirs | `bootstrap/detected.py` | `tests/test_first_run.py`; manual: `~/.local/share/agentic-concierge/detected.json` exists after bootstrap |
| 10.6 | `FeatureSet` + profile feature mapping | `config/features.py` | `tests/test_features.py` (8 tests); disabled feature raises `FeatureDisabledError` |
| 10.7 | `ConciergeConfig` schema additions (`profile`, `features`, `resource_limits`) | `config/schema.py` | `tests/test_config.py` (extend existing); round-trip YAML |
| 10.8 | `InProcessChatClient` (mistral.rs via PyO3) | `infrastructure/chat/inprocess.py` | `tests/test_inprocess_client.py` (8 tests, mistralrs mocked) |
| 10.9 | `VLLMChatClient` (OpenAI-compat HTTP + health check) | `infrastructure/chat/vllm.py` | `tests/test_vllm_client.py` (8 tests, httpx mocked) |
| 10.10 | `build_chat_client()` handles `"inprocess"` and `"vllm"` | `infrastructure/chat/__init__.py` | existing + new chat factory tests |
| 10.11 | `concierge doctor` — show profile, backend health, feature flags | `interfaces/cli.py` | manual: `concierge doctor` on dev machine; `tests/test_doctor_cli.py` (5 tests) |
| 10.12 | `concierge bootstrap [--profile P] [--non-interactive]` | `interfaces/cli.py` | `concierge bootstrap --non-interactive` exits 0; detected.json written |
| 10.13 | `psutil` + `platformdirs` added to core deps | `pyproject.toml` | install + import without extras |
| 10.14 | Docs updated (STATE, PLAN, BACKLOG, DECISIONS, ARCHITECTURE) | `docs/` | this table; ADR-012 through ADR-016 in DECISIONS.md |

### Key data structures

```python
# bootstrap/system_probe.py
@dataclass
class GPUDevice:
    vendor: str          # "nvidia" | "amd" | "apple"
    name: str
    vram_mb: int
    index: int

@dataclass
class SystemProbe:
    cpu_cores: int
    cpu_arch: str        # "x86_64" | "aarch64" | "apple_silicon"
    ram_total_mb: int
    ram_available_mb: int
    gpu_devices: list[GPUDevice]
    disk_free_mb: int
    internet_reachable: bool
    ollama_installed: bool
    ollama_reachable: bool
    vllm_reachable: bool
    mistralrs_available: bool

# bootstrap/model_advisor.py
class ProfileTier(str, Enum):
    NANO = "nano" | SMALL = "small" | MEDIUM = "medium" | LARGE = "large" | SERVER = "server"

@dataclass
class SystemProfile:
    tier: ProfileTier
    max_concurrent_agents: int
    recommended_models: dict[str, str]   # "fast"/"quality"/"routing" -> model name
    recommended_backends: list[str]      # ordered preference
    resource_limits: ResourceLimitsConfig
    reasoning: str

# config/features.py
class Feature(str, Enum):
    INPROCESS | OLLAMA | VLLM | CLOUD | MCP | BROWSER | EMBEDDING | TELEMETRY | CONTAINER

@dataclass
class FeatureSet:
    enabled: frozenset[Feature]
    def is_enabled(self, f: Feature) -> bool
    def require(self, f: Feature, hint: str = "") -> None  # raises FeatureDisabledError if off
```

### Profile tier thresholds

| Profile | RAM | GPU VRAM | Primary backend | Routing backend | Max agents (formula) |
|---------|-----|----------|-----------------|-----------------|----------------------|
| nano    | < 8 GB | any | in-process | in-process | 1 |
| small   | 8–16 GB | < 4 GB | Ollama | in-process | 2 |
| medium  | 16–32 GB | 4–12 GB | Ollama or vLLM | in-process | 4 |
| large   | 32–64 GB | 12–24 GB | vLLM | in-process | 8 |
| server  | 64 GB+ | 24 GB+ or multi-GPU | vLLM | in-process | 16+ |

Max agents formula: `floor((available_ram_mb - reserved_system_mb - 512) / model_ctx_mb) clamped to cpu_cores - 1`

### Model recommendations per profile

| Profile | routing (in-process) | fast | quality |
|---------|----------------------|------|---------|
| nano    | qwen2.5:0.5b (GGUF) | qwen2.5:3b | phi3:mini |
| small   | qwen2.5:0.5b (GGUF) | qwen2.5:7b | qwen2.5:7b |
| medium  | qwen2.5:0.5b (GGUF) | qwen2.5:7b | qwen2.5:14b |
| large   | qwen2.5:0.5b (GGUF) | qwen2.5:14b | qwen2.5:32b |
| server  | qwen2.5:0.5b (GGUF) | qwen2.5:32b | qwen2.5:72b |

All models listed support native tool calling. Minimum viable model = ~3B parameters with instruction + tool-use fine-tuning.

### First-run bootstrap flow

```
1. Binary/CLI invoked; bootstrap/detected.py checks for detected.json — not found
2. FirstRunBootstrap.run() starts:
   a. In-process model loads immediately (system responds NOW, zero setup)
   b. SystemProbe runs in background (psutil, subprocess to nvidia-smi/rocm-smi)
3. ProfileTier determined from probe results
4. Interactive: Rich panel shows "Detected: medium profile — 16 GB RAM, NVIDIA RTX 3080 (8 GB)"
5. If Ollama not installed: prompt to install (or skip for inprocess/cloud-only)
6. Ollama started if installed but not running
7. Recommended models pulled in background with Rich progress bar
8. detected.json written to platformdirs user_data_path("agentic-concierge")
9. System fully operational — hands off to normal request handling
```

### New dependencies

```toml
# Added to [project] core dependencies
"psutil>=5.9"        # RAM/CPU/disk detection (lightweight, no native deps)
"platformdirs>=4.0"  # cross-platform config/data/cache paths (XDG on Linux, native on macOS/Windows)

# New optional extras
nano    = ["mistralrs>=0.3"]    # in-process inference wheel (CPU/CUDA/Metal variants)
vllm    = []                    # reserved — vLLM HTTP client uses existing httpx, no extra pkg needed
browser = ["playwright>=1.40"]  # Phase 11
embed   = ["chromadb>=0.4"]     # Phase 11+
all     = ["agentic-concierge[mcp,otel,embed,browser]"]  # nano excluded (platform-specific wheel)
```

### Phase 10 verification gate

- [ ] `concierge doctor` runs on dev machine; shows detected tier, backend health, active features
- [ ] `concierge bootstrap --non-interactive` exits 0; `detected.json` written to platform data dir
- [ ] `profile: nano` in config → `vllm`, `browser`, `embedding`, `container` features disabled: no imports, no processes
- [ ] Fast CI: **~473 pass** (+71 new tests across 7 new test files)
- [ ] `ruff check src/ tests/ --select E,W,F --ignore E501,F401,E741` passes clean
- [ ] On mocked "first run" (no detected.json): bootstrap flow runs without error in tests

**Phase 10 acceptance:** All 14 deliverables implemented; fast CI: **495 pass**; `concierge doctor` works on the development machine; `concierge bootstrap --non-interactive` writes `detected.json`; feature flag gating verified (disabled feature = zero resource cost confirmed by test).

---

## Phase 11: Browser tool and ChromaDB — **complete**

**Goal:** Add a Playwright-based browser tool (feature-gated per profile) and a ChromaDB vector store backend for the run index. Both are optional features that consume zero resources when disabled.

### Deliverables (Phase 11)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 11.1 | `infrastructure/tools/browser_tool.py` | Done | `BrowserTool`, `is_available()`; 6 async tool methods (browse, click, fill, screenshot, extract_text, navigate); 30s timeout; URL validation |
| 11.2 | `Feature.BROWSER` in `PROFILE_FEATURES` | Done | SMALL/MEDIUM/LARGE/SERVER enabled; NANO excluded |
| 11.3 | `BaseSpecialistPack` browser integration | Done | `feature_set`, `workspace_path`, `network_allowed` params; `aopen()`/`aclose()` lifecycle; `_register_browser_tools()` |
| 11.4 | Registry passes `FeatureSet` to packs | Done | `ConfigSpecialistRegistry.get_pack()` loads detected tier, builds FeatureSet |
| 11.5 | `RunIndexConfig` additions for ChromaDB | Done | `provider`, `chromadb_path`, `chromadb_collection` fields |
| 11.6 | `ChromaRunIndex` — ChromaDB vector store | Done | `infrastructure/workspace/run_index_chroma.py`; lazy import; `add()`/`search()` |
| 11.7 | Dispatch in `run_index.py` | Done | ChromaDB dispatch when `provider="chromadb"` with JSONL fallback |
| 11.8 | `concierge doctor` extras table | Done | Browser (playwright) and ChromaDB rows |
| 11.9 | `MCPAugmentedPack` aopen/aclose fix | Done | Now calls `inner.aopen()`/`inner.aclose()` so browser tools work when MCP-wrapped |
| 11.10 | Tests | Done | `test_browser_tool.py` (13), `test_run_index_chroma.py` (10), +4 test_config, +4 test_features, +2 test_doctor_cli; total **531 pass** |

**Phase 11 acceptance:** All 10 deliverables implemented; fast CI: **531 pass** (+36 vs Phase 10).

---

## Phase 12: Quality gates, orchestrator, session continuation — **complete**

**Goal:** Add engineering quality gates (run_tests tool, tests_verified validation), replace naive recruitment with an LLM orchestrator that decomposes tasks and assigns specialists with briefs, and add session continuation (checkpoint + resume) for interrupted runs.

**See also:** ARCHITECTURE.md §8 for detailed technical design.

### Deliverables (Phase 12)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 12.1 | `infrastructure/tools/test_runner.py` | Done | `run_tests()` — auto-detect pytest/cargo/npm/unittest; structured results |
| 12.2 | `run_tests` tool in engineering pack | Done | Always present regardless of `network_allowed` |
| 12.3 | `tests_verified` + `validate_finish_payload` quality gate | Done | Gates 1–3 in `_execute_pack_loop` |
| 12.4 | Engineering system prompt quality gate instructions | Done | `SYSTEM_PROMPT_ENGINEERING` updated |
| 12.5 | `application/orchestrator.py` | Done | `SpecialistBrief`, `OrchestrationPlan`; `orchestrate_task()` with `create_plan` tool |
| 12.6 | Brief injection in `execute_task.py` | Done | `_get_brief()` helper; brief appended to user message |
| 12.7 | Result synthesis step | Done | `_synthesise_results()` async function |
| 12.8 | `orchestration_plan` runlog event | Done | Emitted when `routing_method="orchestrator"` |
| 12.9 | Orchestrator wired into `execute_task.py` | Done | Replaces `llm_recruit_specialist` call; `plan.mode` overrides `task_force_mode` |
| 12.10 | `concierge plan` CLI command | Done | Calls `orchestrate_task`, prints Rich panel |
| 12.11 | `infrastructure/workspace/run_checkpoint.py` | Done | `RunCheckpoint`; atomic save/load/delete; `find_resumable_runs()` |
| 12.12 | Checkpoint write/delete in `execute_task.py` | Done | `_create_initial_checkpoint()`, `_update_checkpoint()`, `_delete_run_checkpoint()` |
| 12.13 | `concierge resume` + `(resumable)` in `logs list` | Done | `resume_cmd` CLI command; resumable marker |
| 12.14 | Tests | Done | 68 new tests across 6 files; total **599 pass** |

**Phase 12 acceptance:** All 14 deliverables implemented; fast CI: **599 pass** (+68 vs Phase 11).

---

## Phase 13: Rust thin launcher — **complete**

**Goal:** Add a static ~5 MB Rust binary that bootstraps the Python environment, manages self-update, and exec-replaces itself with the Python `concierge` binary. No Python or pip required for end users to get started.

**See also:** ADR-016 (Rust launcher), ARCHITECTURE.md §9.

### Deliverables (Phase 13)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 13.1 | `launcher/Cargo.toml` | Done | reqwest/serde/dirs/semver/anyhow/thiserror; musl-compatible |
| 13.2 | `launcher/src/config.rs` | Done | `LauncherConfig`; env overrides; 5 tests |
| 13.3 | `launcher/src/exec.rs` | Done | `exec_python_concierge()` via `execv`; 1 test |
| 13.4 | `launcher/src/setup.rs` | Done | `ensure_environment`, `upgrade_package`, `installed_version`; 3 tests |
| 13.5 | `launcher/src/update.rs` | Done | `check_latest_release`, `apply_update`, `is_newer`; 4 tests |
| 13.6 | `launcher/src/main.rs` | Done | Orchestration only; `--self-update` flag; passive hint |
| 13.7 | `.github/workflows/build-launcher.yml` | Done | CI: test + clippy + fmt; cross-compile x86_64/aarch64 musl |
| 13.8 | `release.yml` + `install.sh` | Done | Launcher binaries attached to GitHub Release; POSIX one-liner installer |
| 13.9 | Docs | Done | README, CHANGELOG, BACKLOG, STATE, ARCHITECTURE, DECISIONS |

**Phase 13 acceptance:** All 9 deliverables implemented; Rust tests: **13 pass**; Python fast CI unchanged.

---

## Phase 14: Native Rust hot paths + macOS support — **complete**

**Goal:** Replace system `tar` dependency with pure-Rust extraction, add Ed25519 signature verification for self-update, add macOS build targets, and audit application hot paths for potential Rust (PyO3) acceleration.

**See also:** ADR-017 (Ed25519 signing), ARCHITECTURE.md §10.

### Deliverables (Phase 14)

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 14.1 | `launcher/Cargo.toml` — add flate2, tar, ed25519-dalek | Done | All pure Rust; musl static linking unaffected |
| 14.2 | `setup.rs` — pure-Rust tar extraction | Done | `extract_uv()` via flate2+tar; 2 new tests |
| 14.3 | `update.rs` — Ed25519 signed verification | Done | `verify_binary_signature_with_key`; optional .sig (best-effort); 5 new tests |
| 14.4 | `exec.rs` — `#[cfg(unix)]` annotation | Done | Comment documents Phase 15 Windows path |
| 14.5 | `install.sh` — macOS platform dispatch | Done | OS+arch case-block; Darwin/arm64 → aarch64-apple-darwin |
| 14.6 | `build-launcher.yml` — macOS matrix | Done | 4 targets; portable size gate |
| 14.7 | `release.yml` — signing + macOS | Done | Sign binaries step (graceful if secret unset); all 4 targets |
| 14.8 | Application hot-path audit | Done | I/O-bound; PyO3 deferred to Phase 16 |
| 14.9 | Docs | Done | ADR-017; ARCHITECTURE §10; BACKLOG; STATE |
| 14.10 | `Makefile` — `setup-rust-toolchain` target | Done | User-local rustup install; `lint-rust` |
| 14.11 | `scripts/generate_signing_key.sh` | Done | One-time keygen helper; Ed25519 via openssl |

**Phase 14 acceptance:** All 11 deliverables implemented; Rust tests: **20 pass** (+7); Python fast CI: **618 pass** (unchanged).

---

## Post-Phase-14: Dynamic packs, review mechanism, model fixes — **complete**

After Phase 14, several cross-cutting improvements were made without forming a numbered phase:

| Feature | Key changes | Test impact |
|---------|-------------|-------------|
| Adaptive backend resolution (ADR-018) | `BACKEND_PRIORITY[tier]` fallback chain in `llm_discovery.py` | +11 tests |
| Model selection fixes (v0.3.13–v0.3.15) | Closest-distance sort, same-family pref, tool-incapable blocklist, routing model validation, timeout recovery | +13 tests |
| Dynamic pack composition | Central `tool_catalog.py` (8 tools), `dynamic_pack.py` (templates + dynamic builder), composable `prompts.py`; deleted `engineering.py`, `research.py`, `enterprise_research.py`, `recruit.py`, `capabilities.py` | +3 new test files |
| Universal work review (Gate 4, ADR-019) | `_review_specialist_work()` in `execute_task.py`; `PROMPT_REVIEWER`; approve/reject tools; fail-open | +8 tests in `test_review.py` |

**Current state:** Fast CI: **656 pass**; Rust: **22 pass**.

