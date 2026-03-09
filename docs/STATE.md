# agentic-concierge: Current State

**Purpose:** Single source of truth for “where we are” so any human or agent can resume work across restarts and sessions.

**Last updated:** 2026-03-09. Fast CI: **875 pass** (`make test`).
Rust launcher: **22 tests pass** (`make test-rust`).

---

## Specialist Marketplace Architecture — **complete** (2026-03-08)

Six ADRs (023–028) implemented in 5 phases. Adds capability-driven model selection,
non-tool-calling model consultation, adaptive finish schemas, independent reviewer model
selection, and capability-driven orchestrator routing.

| Phase | ADR | What | New files | Tests added |
|-------|-----|------|-----------|-------------|
| A | 023: Model Capability Registry | `ModelCapabilityProfile`, `get_profile()`, `match_models()`, `infer_task_capabilities()` | `infrastructure/model_profiles.py`, `tests/test_model_profiles.py` | ~10 |
| B | 026: Adaptive Finish Schemas | `FINISH_SCHEMAS`, `get_finish_schema()`, `finish_schema_key` threading | `infrastructure/specialists/finish_schemas.py`, `tests/test_finish_schemas.py` | ~8 |
| C | 027: Independent Reviewer Model | `_select_reviewer_model()`, reviewer uses different model | `tests/test_reviewer_model.py` | ~4 |
| D | 024+025: Per-Specialist Model + Consult | `_select_specialist_model()`, `consult_specialist_model` tool, `set_runtime_models()` | `infrastructure/tools/consult.py`, `tests/test_consult_model.py`, `tests/test_per_specialist_model.py` | ~14 |
| E | 028: Capability-Driven Orchestrator | `required_capabilities` on `SpecialistBrief`, `compose_tools_from_capabilities()`, `_resolve_pack_from_capabilities()` — true capability→tool composition with template matching | `tests/test_capability_routing.py`, `tests/test_specialist_marketplace_integration.py` | ~25+19 |

**Key architectural changes:**
- `SpecialistRegistry` protocol gains `set_runtime_models()` (no infrastructure params in app layer).
- `ConfigSpecialistRegistry` manages runtime model state internally via `_needs_consult_tool()`, `_maybe_add_consult()`, `_llm_kwargs()`.
- `_select_specialist_model()` picks per-specialist models using capability profiles.
- `consult_specialist_model` automatically injected when non-tool-calling models detected.
- Orchestrator routes by capabilities, not template names.
- `compose_tools_from_capabilities()` builds tool sets from capability lists; `_resolve_pack_from_capabilities()` matches against templates or creates dynamic packs.
- Mixed capabilities (e.g., `code_python + web_comprehension`) compose a union tool set and create a dynamic pack with generated role and inferred finish schema.

---

## Agent Delegation — **complete** (2026-03-05)

A `delegate_to_specialist` tool handled inline in `_execute_pack_loop` that spawns
a sub-specialist via nested `_execute_pack_loop`. Max depth 1, step budget capped at 15.

| Change | Files |
|--------|-------|
| `delegate_to_specialist` tool definition + inline handler | `application/execute_task.py` |
| Nested `_execute_pack_loop` with depth guard + step cap | `application/execute_task.py` |
| ADR-022 | `docs/DECISIONS.md` |
| 10 tests | `tests/test_delegation.py` |

---

## Human Approval Mechanism — **complete** (2026-03-05)

A `request_approval` tool handled inline in `_execute_pack_loop` that blocks on an
`ApprovalChannel` protocol. Three implementations: `AutoApprovalChannel`,
`CliApprovalChannel`, `HttpApprovalChannel`. CLI gets `--auto-approve`.
HTTP gets `POST /runs/{run_id}/approve`. Config gets `approval_timeout_s`.

| Change | Files |
|--------|-------|
| `ApprovalChannel` protocol + 3 implementations | `application/approval.py` (new), `infrastructure/approval/` (new) |
| `request_approval` tool definition + inline handler | `application/execute_task.py` |
| `--auto-approve` CLI flag | `interfaces/cli.py` |
| `POST /runs/{run_id}/approve` endpoint | `interfaces/http_api.py` |
| `approval_timeout_s` config | `config/schema.py` |
| `approval_requested`, `approval_granted`, `approval_denied` runlog events | `application/execute_task.py` |
| ADR-021 | `docs/DECISIONS.md` |
| 12 tests | `tests/test_approval.py` |

---

## Adaptive Escalation — **complete** (2026-03-05)

Bidirectional model convergence: quality issues → escalate to larger model,
timeouts → fall back to smaller model (existing). Three escalation triggers:
plain-text exhaustion, persistent loops, review rejection at max.

| Change | Files |
|--------|-------|
| `pick_larger_model()` | `infrastructure/llm_discovery.py` |
| `_attempt_escalation()`, `_rebuild_chat_client()`, escalation state, 3 trigger points | `application/execute_task.py` |
| `max_escalations` param threading | `execute_task.py` |
| `model_escalated` runlog event | `application/execute_task.py` |
| 26 tests (8 integration + 8 unit `pick_larger_model` + 6 unit `_attempt_escalation` + 2 `_rebuild_chat_client` + 2 persistent loop) | `tests/test_escalation.py` |
| ADR-020 | `docs/DECISIONS.md` |

---

## Universal work review mechanism (Gate 4) — **complete** (2026-03-05)

Independent reviewer LLM call after doer's `finish_task` passes Gates 1–3.
Reviewer uses read-only tools + `approve_work`/`request_revision` decision tools.
Fail-open design (plain text = approval; max iterations = accept with warning).

| Change | Files |
|--------|-------|
| `PROMPT_REVIEWER` fragment | `infrastructure/specialists/prompts.py` |
| `_review_specialist_work()`, constants, Gate 4 logic | `application/execute_task.py` |
| `max_review_iterations` param threading | `execute_task.py`, `ports.py` (unchanged) |
| Runlog events: `review_start`, `review_approved`, `review_rejected` | `application/execute_task.py` |
| 8 new tests | `tests/test_review.py` |
| 10+ existing test files updated (`max_review_iterations=0`) | `tests/test_*.py` |
| ADR-019 | `docs/DECISIONS.md` |

---

## Dynamic Pack Composition — **complete** (2026-03-05)

Replaced hardcoded specialist packs with dynamic/template-based composition.
Central tool catalog, composable prompt fragments, orchestrator-only routing.

| Change | Files |
|--------|-------|
| Central tool registry (8 tools) | `infrastructure/specialists/tool_catalog.py` (new) |
| Dynamic + template pack builders | `infrastructure/specialists/dynamic_pack.py` (new) |
| Composable prompt fragments | `infrastructure/specialists/prompts.py` |
| Data-driven quality gates | `infrastructure/specialists/base.py` |
| Template/dynamic resolution | `infrastructure/specialists/registry.py` |
| Orchestrator `tools`/`role` support | `application/orchestrator.py`, `execute_task.py`, `ports.py` |
| `tools` field on `SpecialistConfig` | `config/schema.py` |
| Deleted: `engineering.py`, `research.py`, `enterprise_research.py`, `recruit.py`, `capabilities.py` | — |
| New tests | `test_tool_catalog.py`, `test_dynamic_pack.py`, `test_orchestrator_dynamic.py` |

---

## Model selection & timeout resilience — **complete** (2026-03-04)

Fixes three interrelated bugs: model selection picking oversized models, routing
model not validated against available models, and timeout crashes being
unrecoverable.

| Change | Files |
|--------|-------|
| `_size_candidates` closest-distance sort | `infrastructure/llm_discovery.py` |
| `available_models` on `ResolvedLLM` | `infrastructure/llm_discovery.py` |
| `resolve_routing_model()`, `pick_smaller_model()` | `infrastructure/llm_discovery.py` |
| `_scale_timeout()` for oversized models | `infrastructure/llm_discovery.py` |
| `resolved_llm` param + timeout recovery in `_execute_pack_loop` | `application/execute_task.py` |
| Pass `resolved_llm` from callers | `interfaces/cli.py`, `interfaces/http_api.py` |
| 13 new tests | `tests/test_llm_discovery.py`, `tests/test_execute_task.py` |

---

## Adaptive backend resolution — **complete** (2026-03-04)

`resolve_llm()` now falls back through `BACKEND_PRIORITY[tier]` when the configured
primary backend is unreachable. Each backend is probed in order; the first to return
models is used with `fallback_used=True` and a warning. Covers vLLM, Ollama (with
auto-start), inprocess (mistral.rs + GGUF), and cloud (generic backend with api_key).
SERVER profile now includes Ollama as a fallback backend.

| Change | Files |
|--------|-------|
| `BACKEND_PRIORITY` + Ollama in SERVER | `config/features.py` |
| `DEFAULT_BACKEND_URLS`, GGUF constants | `config/constants.py` |
| `ResolvedLLM` fields + fallback chain | `infrastructure/llm_discovery.py` |
| GGUF download in bootstrap | `bootstrap/first_run.py` |
| Warning display + doctor hints | `interfaces/cli.py`, `interfaces/http_api.py` |
| 11 new tests | `tests/test_llm_discovery.py`, `tests/test_features.py` |
| ADR-018 | `docs/DECISIONS.md` |

---

## Current phase: **Phase 14 complete**

### Phase 14 checklist — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| P14-1 | `launcher/Cargo.toml` — add flate2, tar, ed25519-dalek | Done | All pure Rust; musl static linking unaffected |
| P14-2 | `launcher/src/setup.rs` — pure-Rust tar extraction | Done | `extract_uv()` via flate2+tar; `find_file()` removed; 2 new tests |
| P14-3 | `launcher/src/update.rs` — Ed25519 signed verification | Done | `verify_binary_signature_with_key`; `apply_update` downloads + verifies `.sig`; macOS asset names; 5 new tests |
| P14-4 | `launcher/src/exec.rs` — `#[cfg(unix)]` annotation | Done | Comment documents Phase 15 Windows path; no functional change |
| P14-5 | `install.sh` — macOS platform dispatch | Done | OS+arch case-block; Darwin/arm64 → aarch64-apple-darwin |
| P14-6 | `.github/workflows/build-launcher.yml` — macOS matrix | Done | `build-native` job; 4 targets; portable size gate (perl) |
| P14-7 | `.github/workflows/release.yml` — signing + macOS | Done | `Sign binaries` step (graceful if secret unset); all 4 targets |
| P14-8 | Application hot-path audit | Done | I/O-bound; PyO3 deferred to Phase 16; see ARCHITECTURE.md §10 |
| P14-9 | Docs | Done | ADR-017; ARCHITECTURE §10; BACKLOG Phase 14+futures; STATE; CHANGELOG |
| P14-10 | `Makefile` — `setup-rust-toolchain` target | Done | User-local rustup install; `lint-rust` uses `~/.cargo/bin/cargo` |
| P14-11 | `scripts/generate_signing_key.sh` | Done | One-time keygen helper; Ed25519 via openssl; instructions printed |

**Rust tests:** `make test-rust` → **20 pass** (was 13)
**Python fast CI:** `make test` → **618 pass** (unchanged)

---

## Current phase: **Phase 13 complete** (superseded by Phase 14)

### Phase 13 checklist — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| P13-1 | `launcher/Cargo.toml` | Done | reqwest/serde/dirs/semver/anyhow/thiserror; musl-compatible |
| P13-2 | `launcher/src/config.rs` | Done | `LauncherConfig`; env overrides; 5 tests |
| P13-3 | `launcher/src/exec.rs` | Done | `exec_python_concierge()` via `execv`; 1 test |
| P13-4 | `launcher/src/setup.rs` | Done | `ensure_environment`, `upgrade_package`, `installed_version`; 3 tests |
| P13-5 | `launcher/src/update.rs` | Done | `check_latest_release` (network-silent), `apply_update` (atomic rename), `is_newer`; 4 tests |
| P13-6 | `launcher/src/main.rs` | Done | Orchestration only; `--self-update` flag; passive hint |
| P13-7 | `.github/workflows/build-launcher.yml` | Done | CI: test + clippy + fmt; cross-compile x86_64/aarch64 musl |
| P13-8 | `release.yml` + `install.sh` | Done | Launcher binaries attached to GitHub Release; POSIX one-liner installer |
| P13-9 | Docs | Done | README, CHANGELOG, BACKLOG, STATE, ARCHITECTURE, DECISIONS |

**Rust tests:** `cargo test --manifest-path launcher/Cargo.toml` → **13 pass**
**Python fast CI:** `pytest tests/ -k "not real_llm and not real_mcp and not podman" -q` → **599 pass** (unchanged)

---

## Current phase: **Phase 12 complete** (superseded by Phase 13)

Phases 6, 7, and 8 are all **complete**. Phase 8 items (P8-1 through P8-4) are all done.

- **P6-1:** Persistent cross-run run index (`run_index.jsonl`) + `concierge logs search`.
- **P6-2:** Real MCP server smoke test (`tests/test_mcp_real_server.py`, `@pytest.mark.real_mcp`).
- **P6-3:** Containerised workspace isolation — `ContainerisedSpecialistPack` runs `shell` inside Podman; `SpecialistConfig.container_image` triggers transparent wrapping.
- **P6-4:** Cloud LLM fallback — `FallbackChatClient` + `FallbackPolicy`; `CloudFallbackConfig` on `ConciergeConfig`; `cloud_fallback` runlog events; auto-wrapping in `execute_task`.
- **P7-1:** Semantic run index search — `embed_text()` via Ollama `/api/embeddings`; `cosine_similarity()`; `semantic_search_index()` with keyword fallback; `RunIndexConfig` on `ConciergeConfig`; `execute_task` embeds on success; `concierge logs search` uses semantic when available. 22 tests.
- **P7-2:** GitHub MCP real integration test + `docs/MCP_INTEGRATIONS.md`; `github_search` + `enterprise_search` capabilities added.
- **P7-3:** `enterprise_research` specialist — `cross_run_search` tool (queries run index), staleness/confidence system prompt, `enterprise_search` + `github_search` capabilities; in `DEFAULT_CONFIG`. 16 tests.
- **P7-4:** Docs update — STATE.md, PLAN.md, VISION.md §7+§8, BACKLOG.md all updated.
- **P8-1:** Parallel task force execution — `task_force_mode` on `ConciergeConfig`; `_run_task_force_parallel()` + `_merge_parallel_payloads()` in `execute_task.py`; 14 tests.
- **P8-2:** SSE run event streaming — `event_queue: Optional[asyncio.Queue]` on `execute_task()`; `_emit()` helper; `POST /run/stream` SSE endpoint; `run_complete` runlog event; 6 tests.
- **P8-3:** Run status endpoint — `GET /runs/{run_id}/status`; reads `run_complete` event for completion detection; 6 tests.
- **P8-4:** Docs update — STATE.md, BACKLOG.md, PLAN.md updated.

---

## Phase 1 checklist (from [PLAN.md](PLAN.md))

| # | Deliverable | Status | Notes |
|---|-------------|--------|--------|
| 1.1 | CLI: `concierge run`, `concierge serve` | Done | `src/agentic_concierge/interfaces/cli.py` |
| 1.2 | HTTP API: `/health`, `POST /run` | Done | `src/agentic_concierge/interfaces/http_api.py` |
| 1.3 | Config: defaults + `CONCIERGE_CONFIG_PATH` | Done | `agentic_concierge.config.load_config` |
| 1.4 | Recruit: keyword + fallback | Done | `agentic_concierge.application.recruit`; `tests/test_router.py` |
| 1.5 | Execute task: run dir, workspace, runlog, one pack | Done | `agentic_concierge.application.execute_task` |
| 1.6 | Engineering specialist | Done | `src/agentic_concierge/infrastructure/specialists/engineering.py` |
| 1.7 | Research specialist | Done | `src/agentic_concierge/infrastructure/specialists/research.py`; web tools gated by `network_allowed` |
| 1.8 | Sandbox: path safety, shell allowlist | Done | `src/agentic_concierge/infrastructure/tools/sandbox.py`; `tests/test_sandbox.py` |
| 1.9 | Runlog + model params to LLM | Done | `model_cfg` passed; runlog in run dir |
| 1.10 | Quality gates in prompts | Done | FR5; deploy proposed only; citations from fetch only |
| 1.11 | Automated tests | Done | `tests/` — router, sandbox, json_tools, prompts, config, packs |
| 1.12 | Docs: README, REQUIREMENTS, VISION, PLAN, STATE | Done | This file + PLAN + VISION + REQUIREMENTS |
| 1.13 | Local LLM default and core (ensure available by default) | Done | `local_llm_ensure_available: true` by default; [SELF_CONTAINED_LLM.md](SELF_CONTAINED_LLM.md); `ensure_llm_available` in CLI/API; opt-out for managed server |

---

## Phase 1 verification gate (run before marking Phase 1 complete)

**Integration assurance** requires **at least a couple of E2E tests that run against a real LLM** to run and pass. Mocked and unit tests add value (fast feedback, wiring, contracts); real-LLM E2E are essential to ensure everything is integrated and working as expected.

- [x] **Full validation (proves system works):** `python scripts/validate_full.py` — ensures LLM is reachable (starts it if configured), then runs pytest so **all 42 tests** run (no skips). Must pass. If no LLM can be reached or started, the script exits with failure and does not run tests.
- [x] **Run dir:** `concierge run "list files" --pack engineering` → creates `.concierge/runs/<id>/runlog.jsonl` and `workspace/` (connection error without LLM server is expected).
- [x] **API:** `concierge serve` then `curl http://127.0.0.1:8787/health` → `{"ok": true}`. `POST /run` without LLM returns **503** with a clear detail message.
- [x] **REQUIREMENTS:** Manual validation items 1–4 in REQUIREMENTS.md hold (CLI help, routing, run structure, API health).
- [x] **E2E (real LLM):** With a real LLM available, `python scripts/verify_working_real.py` → exits 0; runlog has tool_call and tool_result; workspace has artifacts. Same is asserted by the real-LLM pytest tests when run via `validate_full.py`.

**Fast CI:** `pytest tests/ -k "not real_llm and not verify"` → **194 pass** (4 real-LLM tests deselected). Use for quick feedback on wiring and unit/integration behaviour; it does not replace the need to run real-LLM E2E for integration assurance.

**Phase 1 complete.** Full validation (2026-02-24): fast CI 45 pass; all 4 real-LLM E2E tests pass against Ollama 0.12.11 with llama3.1:8b (resolve_llm auto-discovers the available model). `verify_working_real.py` exits 0. Next: Phase 2.

**Verification passes (multi-pass checklist):** See [VERIFICATION_PASSES.md](VERIFICATION_PASSES.md). Last run 2026-02-24: fast CI 45 pass; real-LLM tests (engineering, research, API, verify_script) all PASS with llama3.1:8b on Ollama 0.12.11.

---

## Phase 1: what’s tested, what’s not

**Fully tested / demonstrated**

All Phase 1 functional requirements (FR1–FR6 in REQUIREMENTS.md) have automated test coverage or are covered by the verification gate and E2E runs.

| Area | How it’s tested |
|------|------------------|
| CLI `concierge run` / `concierge serve` | pytest (integration + API); real CLI run with real LLM (engineering task). |
| API `GET /health`, `POST /run` | pytest (health, POST with mocked execute_task); POST without LLM → 503. |
| Config, recruit, sandbox, runlog, packs | Unit and integration tests (test_config, test_router, test_sandbox, test_packs, test_integration, etc.). |
| Engineering pack with real LLM, tool use, artifacts | `verify_working_real.py` (exits 0; tool_call/tool_result; workspace e.g. hello.txt). |
| Run dir structure (runlog.jsonl, workspace/) | All E2E and integration tests. |
| Routing (keyword + fallback), research pack tool list (network_allowed) | test_router, test_packs. |
| Local LLM default (config, ensure_available in code) | test_config, test_llm_bootstrap; real run uses Ollama when available. |
| BACKENDS/REQUIREMENTS alignment (backend-agnostic, ensure when enabled, run dir only under workspace_root) | `tests/test_backends_alignment.py`: ChatClient port only, config defaults, API ensure_llm_available when enabled / skipped when opted out, run dir under workspace_root. |

**Recommended for full demonstration (manual or when LLM available)**

| Check | Command / how |
|-------|----------------|
| **Research pack with real LLM** (REQUIREMENTS §6) | `concierge run "Mini systematic review of post-quantum crypto performance." --pack research` (with `network_allowed` true if you want web tools). Inspect runlog for web_search/fetch_url and workspace for deliverables. |
| **API POST /run with real LLM** | `concierge serve` in one terminal; `curl -X POST http://127.0.0.1:8787/run -H "Content-Type: application/json" -d '{"prompt":"Create a file ok.txt with content OK","pack":"engineering"}'`. Expect 200 and JSON with `_meta` and payload. |
| **Local LLM bootstrap (start if unreachable)** | With Ollama stopped, run `concierge run "list files" --pack engineering` (default `local_llm_ensure_available: true`). Fabric should start `ollama serve` and then run; or fail with a clear “couldn’t start or reach” message if Ollama isn’t installed. |

**Not automated (prompt/behaviour)**

- **FR5.1 / FR5.2:** Quality gates (no “works” without tests; deploy proposed only; citations only from fetch) are in system prompts; compliance is by design and manual inspection, not automated assertion.
- **FR5.3:** Research with `network_allowed: false` omits web tools (tested in test_packs); tools return “network disabled” when invoked (in tool implementation).

---

## Phase 2 checklist (from [PLAN.md](PLAN.md)) — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 2.1 | Capability model: define capabilities, map packs in config | Done | `config/capabilities.py` (CAPABILITY_KEYWORDS); `capabilities` field on SpecialistConfig; DEFAULT_CONFIG updated |
| 2.2 | Task → capabilities (rules or router model) | Done | `infer_capabilities()` in `application/recruit.py`; keyword substring matching |
| 2.3 | Recruitment: select pack(s) from capabilities (single pack for Phase 2) | Done | `RecruitmentResult`; two-stage routing in `recruit_specialist()`; keyword fallback preserved |
| 2.4 | Runlog/metadata: log required_capabilities, selected_pack(s) | Done | `"recruitment"` event in runlog; `required_capabilities` on `RunResult`; in HTTP `_meta` |
| 2.5 | Docs: VISION §8, REQUIREMENTS, STATE updated | Done | `REQUIREMENTS.md` FR2.1 rewritten; VISION §8 alignment table updated; `docs/CAPABILITIES.md` new |

---

## Phase 3 checklist (from [PLAN.md](PLAN.md)) — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 3.1 | Task decomposition outputs multiple capability IDs | Done | `infer_capabilities()` returns all matching caps; `_greedy_select_specialists()` covers all of them |
| 3.2 | Supervisor runs multiple packs; shared workspace + combined runlog | Done | `execute_task()` loops over `specialist_ids`; single run dir; `pack_start` events in runlog |
| 3.3 | Sequential coordination with context handoff | Done | finish payload from pack N forwarded as context to pack N+1; step names prefixed by specialist ID |
| 3.4 | Docs and STATE updated | Done | BACKLOG.md Phase 3 section; STATE.md; PLAN.md ticks |

---

## Phase 4 checklist (from [PLAN.md](PLAN.md)) — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 4.1 | Generic/cloud LLM client + `ModelConfig.backend` field | Done | `infrastructure/chat/__init__.py` (build_chat_client factory); `GenericChatClient` in `infrastructure/chat/generic.py`; shared `parse_chat_response()` in `_parser.py`; `backend: str = “ollama”` on `ModelConfig` |
| 4.2 | `concierge logs` CLI subcommand | Done | `logs list` (Rich table) and `logs show` (pretty-printed JSON with kind filter) in `interfaces/cli.py`; `RunSummary` + `list_runs()` + `read_run_events()` in `infrastructure/workspace/run_reader.py` |
| 4.3 | OpenTelemetry tracing (optional dep) | Done | `infrastructure/telemetry.py` (`_NoOpSpan`, `_NoOpTracer`, `setup_telemetry()`, `get_tracer()`); graceful no-op when OTEL not installed; `TelemetryConfig` in `config/schema.py`; `fabric.execute_task` / `fabric.llm_call` / `fabric.tool_call` spans in `execute_task.py`; `[otel]` extra in `pyproject.toml` |
| 4.4 | Docs update | Done | BACKLOG.md Phase 4 section; STATE.md; PLAN.md Phase 4 concrete deliverables |

---

## Phase 5 checklist (from [PLAN.md](PLAN.md)) — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 5.1 | Config schema: MCPServerConfig + mcp_servers | Done | `config/schema.py`; validators for stdio/sse; duplicate-name check |
| 5.2 | Async execute_tool + pack lifecycle (aopen/aclose) | Done | `base.py`, `ports.py`, `execute_task.py`; try/finally in _execute_pack_loop |
| 5.3 | MCPSessionManager + converter | Done | `infrastructure/mcp/session.py`, `converter.py`; top-level mcp import guarded |
| 5.4 | MCPAugmentedPack | Done | `infrastructure/mcp/augmented_pack.py`; asyncio.gather connect/disconnect |
| 5.5 | Registry integration | Done | `registry.py` wraps pack when mcp_servers non-empty; RuntimeError if mcp not installed |
| 5.6 | pyproject.toml + docs | Done | `mcp = [“mcp>=1.0”]` optional dep; dev dep updated; all docs updated |

---

## Phase 10 checklist — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| P10-1 | `bootstrap/system_probe.py` | Done | `SystemProbe`, `GPUDevice`, `probe_system()` async; psutil+platformdirs |
| P10-2 | `bootstrap/model_advisor.py` | Done | `ProfileTier` in `config/features.py`; `SystemProfile`, `advise_profile()` |
| P10-3 | `config/features.py` | Done | `Feature`, `PROFILE_FEATURES`, `FeatureDisabledError`, `FeatureSet` |
| P10-4 | `config/schema.py` additions | Done | `FeaturesConfig`, `ResourceLimitsConfig`; `profile/features/resource_limits` on `ConciergeConfig` |
| P10-5 | `bootstrap/detected.py` | Done | `detected_path()`, `save_detected()`, `load_detected()`, `is_first_run()` via platformdirs |
| P10-6 | `bootstrap/backend_manager.py` | Done | `BackendStatus`, `BackendHealth`, `BackendManager`; feature-gated probing |
| P10-7 | `infrastructure/chat/inprocess.py` | Done | `InProcessChatClient` lazy-imports mistralrs; `is_available()` |
| P10-8 | `infrastructure/chat/vllm.py` | Done | `VLLMChatClient`; pure httpx; `health_check()`, `list_models()`, `chat()` |
| P10-9 | Update `build_chat_client()` | Done | Dispatches `”vllm”` and `”inprocess”` backends |
| P10-10 | `bootstrap/first_run.py` | Done | `run()` orchestrates probe→advise→ensure_ollama→pull→save |
| P10-11 | `concierge doctor` CLI | Done | Rich table: hardware, profile, feature flags, backend health |
| P10-12 | `concierge bootstrap` CLI | Done | Calls `first_run.run()`; `--profile`, `--non-interactive` |
| P10-13 | `pyproject.toml` dep/extras | Done | `psutil>=5.9`, `platformdirs>=4.0` core; `nano`, `embed`, `browser`, `all` extras |
| P10-14 | Tests | Done | 7 new test files, 93 new tests; total fast CI: **495 pass** |

## Phase 11 checklist — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| P11-1 | `infrastructure/tools/browser_tool.py` | Done | `BrowserTool`, `is_available()`; 6 async tool methods; 30s timeout; URL validation; workspace screenshot |
| P11-2 | `Feature.BROWSER` in `PROFILE_FEATURES` | Done | SMALL/MEDIUM/LARGE/SERVER; NANO excluded |
| P11-3 | `BaseSpecialistPack` browser integration | Done | `feature_set`, `workspace_path`, `network_allowed` params; `aopen()`/`aclose()` lifecycle; `_register_browser_tools()` |
| P11-4 | Registry passes `FeatureSet` to packs | Done | `ConfigSpecialistRegistry.get_pack()` loads detected tier, builds FeatureSet, sets `pack._feature_set` |
| P11-5 | `RunIndexConfig` additions for ChromaDB | Done | `provider`, `chromadb_path`, `chromadb_collection` fields |
| P11-6 | `ChromaRunIndex` — ChromaDB vector store | Done | `infrastructure/workspace/run_index_chroma.py`; lazy import; `add()`/`search()` |
| P11-7 | Dispatch in `run_index.py` | Done | `append_to_index`/`semantic_search_index` accept `run_index_config`; ChromaDB dispatch with JSONL fallback |
| P11-8 | `concierge doctor` extras table | Done | Browser (playwright) and ChromaDB rows via `importlib.util.find_spec` |
| P11-9 | Tests | Done | `test_browser_tool.py` (13), `test_run_index_chroma.py` (10); +4 test_config; +4 test_features; +2 test_doctor_cli; total **531 pass** |
| P11-10 | `MCPAugmentedPack` aopen/aclose fix | Done | Now calls `inner.aopen()`/`inner.aclose()` so browser tools work when MCP-wrapped |

## Phase 12 checklist — **complete**

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| P12-1 | `infrastructure/tools/test_runner.py` | Done | `run_tests()` — auto-detect pytest/cargo/npm; `_detect_framework`, `_parse_pytest_output`, `_parse_cargo_output`; sandbox allowlist extended |
| P12-2 | `run_tests` tool in engineering pack | Done | Registered in `build_engineering_pack()`; always present regardless of `network_allowed` |
| P12-3 | `tests_verified` + `validate_finish_payload` quality gate | Done | `tests_verified` in engineering finish required fields; `EngineeringSpecialistPack.validate_finish_payload()` rejects `False`; `BaseSpecialistPack` default no-op; Gate 3 in `_execute_pack_loop` |
| P12-4 | Engineering system prompt quality gate instructions | Done | `SYSTEM_PROMPT_ENGINEERING` updated with quality gate section |
| P12-5 | `application/orchestrator.py` | Done | `SpecialistBrief`, `OrchestrationPlan`; `orchestrate_task()` with `create_plan` tool; fallback to `llm_recruit_specialist` |
| P12-6 | Brief injection in `execute_task.py` | Done | `_get_brief()` helper; brief appended to user message in both sequential and parallel paths |
| P12-7 | Result synthesis step | Done | `_synthesise_results()` async function; called when `plan.synthesis_required=True` and >1 specialist |
| P12-8 | `orchestration_plan` runlog event | Done | Emitted after recruitment when `routing_method=”orchestrator”` |
| P12-9 | Orchestrator wired into `execute_task.py` | Done | `orchestrate_task` replaces `llm_recruit_specialist` call; `plan.mode` overrides `task_force_mode` |
| P12-10 | `concierge plan` CLI command | Done | Calls `orchestrate_task`, prints Rich panel with mode/synthesis/assignments |
| P12-11 | `infrastructure/workspace/run_checkpoint.py` | Done | `RunCheckpoint`; `save_checkpoint()` atomic; `load_checkpoint()`; `delete_checkpoint()`; `find_resumable_runs()` |
| P12-12 | Checkpoint write/delete in `execute_task.py` | Done | `_create_initial_checkpoint()`, `_update_checkpoint()`, `_delete_run_checkpoint()`; `resume_execute_task()` |
| P12-13 | `concierge resume` + `(resumable)` in `logs list` | Done | `resume_cmd` CLI command; resumable marker in `concierge logs list` |
| P12-14 | Tests | Done | `test_run_tests_tool.py` (15), `test_engineering_pack_quality.py` (5), `test_orchestrate_task.py` (20), `test_run_checkpoint.py` (16), `test_resume.py` (8), +4 `test_execute_task.py`; total **599 pass** |

## Next steps (what to do when resuming)

**The backlog is the canonical source for what to work on next.**

1. Read [BACKLOG.md](BACKLOG.md) — find the first non-done item; that is what to work on.
2. Run `pytest tests/ -k “not real_llm and not verify and not real_mcp”` — confirm **875 pass** before touching code.
3. Phase 12 is complete — see BACKLOG.md for Phase 13 planning or add new items.
4. See [DECISIONS.md](DECISIONS.md) for rationale behind key architectural choices.

---

## Quick commands (for copy-paste)

```bash
# From repo root
pip install -e ".[dev]"
pytest tests/ -v

# CLI
concierge --help
concierge run "list files" --pack engineering
concierge run "mini systematic review of X" --pack research

# API (background)
concierge serve
# then: curl http://127.0.0.1:8787/health
# POST: curl -X POST http://127.0.0.1:8787/run -H "Content-Type: application/json" -d '{"prompt":"list files","pack":"engineering"}'
```

---

## Architecture changes (2026-02-24 refactor)

The tool loop was completely reworked from a fragile JSON-in-content protocol to **native OpenAI function calling**:

- `ChatClient.chat()` now accepts `tools: list[dict] | None` and returns `LLMResponse` (not `str`)
- `LLMResponse` + `ToolCallRequest` are domain types in `domain/models.py`
- `SpecialistPack` now has `tool_definitions` and `finish_tool_name` properties
- `execute_task` runs a proper tool-calling loop; `finish_task` tool call signals completion
- `OllamaChatClient` detects “does not support tools” in 400 responses and raises a clear error
- `_param_size_sort_key` fixed: parses “8.0B” as 8.0 not 80 (was causing sqlcoder:15b to be selected over llama3.1:8b)
- `resolve_llm` is called via `asyncio.to_thread` in the FastAPI handler

## Blockers / open questions

- None at last update. Phase 10 spec is locked. Ready to implement.

---

## Doc map (for agents)

| Read first | Then | For |
|------------|------|-----|
| **STATE.md** (this file) | BACKLOG.md | Resuming work; current phase and what’s next |
| **BACKLOG.md** | — | Prioritised work items with full context; single source of truth for "what to do next" |
| **DECISIONS.md** | — | Architectural decisions and rationale; read before changing significant design |
| PLAN.md | REQUIREMENTS.md, VISION.md | Phase deliverables, verification gates, full context |
| REQUIREMENTS.md | — | MVP functional requirements and validation |
| VISION.md | — | Long-term vision, principles, use-case pillars |

**Workflow:** When you complete an item, tick it off in BACKLOG.md and move it to the Done table.
Update STATE.md with the new date. Run the fast CI check before and after every change.
