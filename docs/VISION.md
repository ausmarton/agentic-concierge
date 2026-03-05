# agentic-concierge: Long-term Vision

A single, coherent vision document for the agentic-concierge project. Use this to steer design and to check that the repo stays aligned with the vision.

---

## 1. Vision in one paragraph

We are building an **autonomous agentic system** that can act in many ways depending on the task. The following are **illustrative use cases**, not a fixed or comprehensive list—they help shape our **initial iterations**:

- **An engineering team** (software + data) taking ideas from development through deployment, monitoring, and fixes across cloud, local, web, and mobile.
- **A personal assistant** answering queries like “Find the best business-class tickets London–Lisbon for a week in May for two adults”, “Summarise my stocks today”, or “Tell me about RR.LSE”.
- **An enterprise research assistant** that searches Confluence, GitHub, Jira, Rally (and similar) and produces short, reasoned reports with links and explicit notes on what is likely valid vs stale.

The actual system should **eventually adapt to any new formation** required by the task: we add or recruit capabilities as needed rather than being limited to these examples.

We do **not** run every agent all the time. We **recruit on demand**: we look at the task, break it down into the capabilities required, and form a **task force** of only those agents that are needed. For example, a pure data-engineering project would not spin up mobile-app or financial-modelling specialists. The thing that is always available (or readily started) is an **orchestrator** that can decide what to spin up and get the task done—not the full roster of all possible agents. Teams are formed on demand; agents are spun up based on the specific task.

---

## 2. Principles (non‑negotiable)

- **Quality over speed**  
  We prefer precision and correctness. Where we must trade off, we choose quality.
- **Local-first**  
  Local LLM is the **default and primary** path. Prefer local models and local tooling (MCP, etc.). The fabric ensures the local LLM is available (including starting it when unreachable) by default; opt-out only for “I manage the server myself.” Use cloud only where local cannot meet quality or **capability** demands, with an explicit fallback path.
- **Portable and clean**  
  Implement in a way that stays portable and eventually supports cross‑platform use; for now, optimise for the current hardware and OS.
- **Phased and aligned**  
  Build in phases, with the full blueprint written down upfront and the design iterated as we move through phases.

---

## 3. Platform and hardware

- **OS:** Fedora Linux.
- **Local inference:** We **use Ollama** for local LLM inference. Install Ollama, pull models (e.g. qwen2.5:7b, qwen2.5:14b), run the fabric; no extra config by default.
- **Hardware (reference):** AMD Ryzen AI Max+ Pro 395, Radeon 8060S (×32), 128 GB RAM. No NVIDIA; use Vulkan/AMD-friendly runtimes (e.g. llama.cpp with appropriate backends).
- **Implication:** Build and document with Ollama as the default; other backends remain supported via config override.

---

## 4. Use-case pillars (long-term)

The examples below are **illustrative**, not exhaustive. They give concrete directions for early iterations; the goal is a system that can **adapt to whatever formation a task requires** (new capabilities, new agent types, new combinations). Not all of these need to be in the first phase.

### 4.1 Engineering (software + data)

- **Scope:** From idea → prototype → test → demo → revise (from feedback) → deploy → monitor → test, critique, and fix issues.
- **Domains:** Rust, Python, data engineering, ML/AI pipelines, Scala, GCP, autonomous pipelines and tooling, infra (e.g. Minikube, Kubernetes, GKE), Podman, JVM, SRE, testing, data quality, data provenance, modelling, architecture (including enterprise-scale).
- **Organisation:** One or more “teams” of agents; specialise where it improves accuracy, generalise where that works better. The system may use the internet when needed to gather context, like a human engineering team would.

### 4.2 Research (systematic and general)

- **Scope:** Full systematic literature review and general research (academic, professional, web).
- **Standard:** PhD‑researcher level: scoping, search, screening, extraction, synthesis, critique, with rigour and critical thinking.
- **Organisation:** As many agents as needed, structured so that rigour and traceability (screening logs, evidence tables, citations) are maintained.

### 4.3 Enterprise search and reporting

- **Scope:** Search Confluence, GitHub, Jira, Rally (and similar) for a user-defined topic; produce a short report with links and reasoning about what is valid vs potentially stale.
- **Example:** “What can you find about Supply management in our org?” → search across sources → distilled report with links and staleness/confidence notes.

### 4.4 Personal and life-assistant style queries

- **Examples:** Travel (e.g. best ticket prices, itineraries), portfolio summaries (“what’s happening today across my stocks”), instrument lookups (“Tell me about RR.LSE”).
- **Note:** May share infrastructure with research/enterprise (search, summarisation, citations) but with different tools and data sources.

### 4.5 Other specialised areas (candidate)

- Financial planning and investment optimisation.
- Open source: find issues, implement fixes, contribute back.
- Learning: organise learning goals and find/resources to support them.
- Social / trends: track social media trends with a dedicated agent ecosystem.

We can start with a small set of very specialised pillars (e.g. engineering + research) and add others incrementally.

---

## 5. Architecture and resource model

- **Specialist pool:** Many agents, each with distinct capabilities (A, B, C, D, …)—e.g. data engineering, mobile, financial modelling, research, enterprise search. None of them need to be “always on”.
- **Task → breakdown → recruit → task force:** For each task we (1) look at what’s required, (2) break it down into explicit capabilities, (3) recruit only the agents that have those capabilities, and (4) spin them up to form a **task force** for that problem. Example: pure data-engineering work → recruit data-engineering (and any supporting) agents only; no mobile-app or financial-modelling agents.
- **Orchestrator, not full roster:** What is always available (or quickly started) is something that can **decide what to spin up** and orchestrate the task—not the entire set of agents. So we don’t “toggle” which pre-defined team is active; we **form a team on demand** and spin up only what that task needs.
- **Single fabric, multiple packs:** One agentic-concierge with many “packs” (capability areas). The orchestrator/router analyses the task, maps it to required capabilities, and recruits the right pack(s) or sub-agents. The system is designed to **adapt to any new formation** the task demands—new capability areas and new combinations can be added without being limited to a fixed set of use cases. The current repo uses “one pack per run” chosen by a router; the long-term model extends this to task decomposition and **multi-pack recruitment** so that the right task force is assembled and started for each request.

---

## 6. Technical direction

- **Models:** We use **Ollama** for local models by default; aim for quality and correctness on par with strong cloud models for the tasks we support. Ensuring the local LLM is available (including starting it when unreachable) is **default behaviour**. Explicit **cloud fallback** only when the local **model** cannot meet the bar (quality or capability), not when the server is unreachable.
- **Automation and tools:** MCP and other local automation; enterprise connectors (Confluence, Jira, GitHub, Rally) via MCP or custom tools, least‑privilege and sandboxed where possible.
- **Observability:** Export traces (e.g. OpenTelemetry) and maintain runlogs and audit trails so we can verify behaviour and debug.
- **Deployment and safety:** Deploy/push and other high-impact actions require human approval via the `request_approval` tool (ADR-021). Specialists can also delegate sub-tasks to other specialists via `delegate_to_specialist` (ADR-022, max depth 1).

**Cloud fallback (future).** When we add cloud support, it will be used only when the **local model** cannot meet **quality or capability** (e.g. task needs a model we don’t have locally, or the local model fails a quality bar). It will **not** be used when the local server is unreachable—that case is handled by ensuring the local LLM is available (start if needed). Implementation will be explicit (e.g. capability or quality check, or user choice), not “connection failed → try cloud”.

---

## 7. Phasing and blueprint

- **Phases 1–8 (complete):**
  - Phase 1: Engineering + research packs, keyword router, CLI, HTTP API, local Ollama, sandbox, runlog.
  - Phase 2: Capability model; two-stage routing (prompt → required capabilities → pack by coverage); capability logged in runlog and HTTP `_meta`.
  - Phase 3: Multi-pack task forces; sequential execution with context handoff; shared workspace + runlog; `pack_start` events.
  - Phase 4: Generic/cloud LLM client (`ModelConfig.backend`); `concierge logs` CLI subcommand; OpenTelemetry tracing (optional dep; no-op shim when absent); LLM-driven orchestrator routing with `routing_model_key`.
  - Phase 5: MCP tool server support — `MCPServerConfig` in config; `MCPAugmentedPack` wraps any specialist pack transparently; `aopen`/`aclose` lifecycle; tool names prefixed `mcp__<server>__<tool>`; optional `mcp` dep group.
  - Phase 6: Persistent run index + `concierge logs search`; real MCP server smoke test (filesystem); containerised workspace isolation via Podman (`ContainerisedSpecialistPack`; `:Z` SELinux label); cloud LLM fallback (`FallbackPolicy` + `FallbackChatClient`; `CloudFallbackConfig`).
  - Phase 7: Semantic run index search (`embed_text` + `cosine_similarity` + `semantic_search_index` via Ollama; `RunIndexConfig` with `embedding_model`); GitHub MCP real integration tests + `docs/MCP_INTEGRATIONS.md`; `enterprise_research` specialist (`cross_run_search` tool, staleness/confidence notation, `enterprise_search` + `github_search` capabilities).
  - Phase 8: Parallel task force execution (`task_force_mode: parallel`; `asyncio.gather`; `_merge_parallel_payloads`); SSE streaming (`POST /run/stream`; `event_queue` on `execute_task`; `_emit` helper; `run_complete` event); run status endpoint (`GET /runs/{id}/status`).

- **Phases 9–14 (complete):**
  - Phase 9: CLI streaming (`--stream` / `-s` with Rich rendering); corrective re-prompt (up to 2 plain-text retries); per-IP rate limiting (`CONCIERGE_RATE_LIMIT`); sandbox absolute-path error hint.
  - Phase 10: Self-sizing bootstrap (`SystemProbe`, `ProfileTier`, `FirstRunBootstrap`); three-layer inference (in-process via mistral.rs, Ollama, vLLM); profile-based feature flags (`FeatureSet`); `concierge doctor` and `concierge bootstrap` CLI commands.
  - Phase 11: Browser tool (Playwright, feature-gated); ChromaDB vector store backend for run index; `MCPAugmentedPack` lifecycle fix for inner pack `aopen`/`aclose`.
  - Phase 12: Quality gates (Gates 1–3: prior work, required fields, pack-specific validation); `run_tests` tool; LLM orchestrator (`orchestrate_task`, `create_plan` tool, brief injection, result synthesis); session continuation (checkpoint, `concierge resume`, resumable runs).
  - Phase 13: Rust thin launcher (static binary, venv bootstrap, self-update, `install.sh`).
  - Phase 14: Pure-Rust tar extraction (no system `tar`); Ed25519 signed self-update (ADR-017); macOS targets; hot-path analysis (I/O-bound, no PyO3 needed).

- **Post-Phase-14 (complete):**
  - Adaptive backend resolution: `BACKEND_PRIORITY[tier]` fallback chain when primary backend unreachable (ADR-018).
  - Model selection fixes: closest-distance sort, same-family preference, tool-incapable model blocklist, routing model validation, timeout recovery with smaller-model fallback.
  - Dynamic pack composition: central tool catalog (8 tools), composable prompt fragments, template-based pack builders. Removed hardcoded `engineering.py`, `research.py`, `enterprise_research.py`, `recruit.py`, `capabilities.py`.
  - Universal work review mechanism (Gate 4, ADR-019): independent reviewer LLM after doer finishes; read-only tools + approve/reject; fail-open design.

- **Post-Phase-14 continued:**
  - Adaptive escalation (ADR-020): bidirectional model convergence — quality issues escalate to larger model, timeouts fall back to smaller model. Three triggers, max 2 escalations.
  - Human approval mechanism (ADR-021): `request_approval` tool handled inline in `_execute_pack_loop`; blocks on `ApprovalChannel` protocol. Three implementations (Auto, CLI, HTTP). CLI `--auto-approve`; HTTP `POST /runs/{run_id}/approve`; `approval_timeout_s` config.
  - Agent-to-agent delegation (ADR-022): `delegate_to_specialist` tool spawns nested `_execute_pack_loop`. Max depth 1, step budget 15.

- **Future:**
  - Dynamic re-recruitment: re-plan when a pack's output reveals additional capability needs.
  - Phase 15: Windows launcher + Homebrew tap.
  - Phase 16: PyO3 extension (if profiling justifies) + additional specialist packs.
  - Phase 17+: Multi-tenant, Web UI, plugin registry.

The full blueprint is reflected in PLAN.md, BACKLOG.md, REQUIREMENTS.md, and this document.

---

## 8. Alignment with the repo (how to “follow the vision”)

Use this checklist to keep the repo aligned with the vision.

| Vision element | Where it lives in repo | Status / notes |
|----------------|------------------------|----------------|
| Quality over speed | REQUIREMENTS (FR5, Gates 1–4), composable prompt fragments in `prompts.py` | Enforced: Gates 1–3 in `_execute_pack_loop`; Gate 4 (reviewer) in `_review_specialist_work`; `run_tests` tool with quality gate. |
| Local-first | Config `base_url`, `local_llm_ensure_available` (default True), `BACKEND_PRIORITY` fallback chain | Local LLM is default; fabric ensures available (start if needed); adaptive backend resolution falls back through `BACKEND_PRIORITY[tier]`. |
| Cloud fallback | `infrastructure/chat/fallback.py`; `CloudFallbackConfig` in `config/schema.py`; `FallbackPolicy`; auto-wrap in `execute_task`; `cloud_fallback` runlog events | **Done (Phase 6):** Triggers when local model fails a quality bar (no tool calls, malformed args), not on connection failure. |
| Engineering pack: plan→implement→test→review | `PACK_TEMPLATES[“engineering”]` in `dynamic_pack.py`; `ROLE_ENGINEERING` in `prompts.py` | **Done.** Template pack with shell, read/write/list files, run_tests; quality gate requires `tests_verified`; deploy/push proposed only. |
| Research pack: systematic review, citations, screening | `PACK_TEMPLATES[“research”]` in `dynamic_pack.py`; `ROLE_RESEARCH` in `prompts.py` | **Done.** Template pack with web_search, fetch_url, read/write/list files; citations only from fetch_url. |
| Deploy/push require human approval | `request_approval` tool in `execute_task.py`; `ApprovalChannel` in `application/approval.py`; `infrastructure/approval/` (Auto, CLI, HTTP); `approval_timeout_s` config | **Done (ADR-021).** Interactive pause-and-wait via `request_approval` tool; CLI `--auto-approve`; HTTP `POST /runs/{run_id}/approve`. |
| Orchestrator decides what to spin up | `application/orchestrator.py`; `orchestrate_task()` with `create_plan` tool | **Done (Phase 12 + dynamic packs).** LLM orchestrator decomposes tasks, assigns template or dynamic packs, sets execution mode. Fallback to first template on error. |
| Dynamic pack composition | `infrastructure/specialists/tool_catalog.py`, `dynamic_pack.py`, `prompts.py` | **Done (post-Phase 14).** Central tool catalog (8 tools); composable prompt fragments; `build_dynamic_pack()` for runtime composition. |
| Agent-to-agent delegation | `delegate_to_specialist` tool in `execute_task.py`; ADR-022 | **Done (ADR-022).** Nested `_execute_pack_loop` with depth=1 guard and 15-step budget cap. |
| Universal work review (Gate 4) | `_review_specialist_work()` in `execute_task.py`; `PROMPT_REVIEWER` in `prompts.py`; ADR-019 | **Done (post-Phase 14).** Independent reviewer LLM; read-only tools; approve/reject; fail-open; max 2 rejections. |
| Enterprise (Confluence/Jira/GitHub/Rally) | `PACK_TEMPLATES[“enterprise_research”]` in `dynamic_pack.py`; `MCP_INTEGRATIONS.md`; `cross_run_search` tool in `tool_catalog.py` | **Done (Phase 7).** Enterprise research template with cross-run search; GitHub MCP integration tested; Confluence/Jira config examples in docs. |
| Session continuation | `infrastructure/workspace/run_checkpoint.py`; `resume_execute_task()` in `execute_task.py`; `concierge resume` CLI | **Done (Phase 12).** Atomic checkpoint; resume interrupted runs; `(resumable)` marker in `logs list`. |
| Multi-backend LLM | `infrastructure/chat/` — Ollama, Generic, vLLM, InProcess clients; `build_chat_client()` factory | **Done (Phases 4, 10).** Four backends; `ModelConfig.backend` selects; adaptive fallback via `BACKEND_PRIORITY`. |
| Hardware profiles and bootstrap | `bootstrap/` package; `config/features.py`; `concierge doctor`/`bootstrap` CLI | **Done (Phase 10).** `SystemProbe`, `ProfileTier`, `FeatureSet`; zero-resource for disabled features. |
| Browser tool | `infrastructure/tools/browser_tool.py`; `Feature.BROWSER` gating | **Done (Phase 11).** Playwright-based; feature-gated per profile; 6 async tool methods. |
| Run index (cross-run memory) | `infrastructure/workspace/run_index.py`, `run_index_chroma.py`; `RunIndexConfig` | **Done (Phases 6–7, 11).** Keyword + semantic (Ollama embeddings) + ChromaDB vector store. |
| Observability | `infrastructure/telemetry.py`; `TelemetryConfig`; runlog events | **Done (Phase 4).** OTEL spans + graceful no-op; SSE streaming (Phase 8); rate limiting (Phase 9). |
| Rust launcher distribution | `launcher/` Rust crate; `install.sh`; GitHub Actions CI | **Done (Phases 13–14).** Static binary for Linux + macOS; Ed25519 signed self-update; pure-Rust tar extraction. |
| AMD / Fedora / Vulkan-friendly | README Quickstart | Documented; no NVIDIA assumption. |
| Portable, clean, extensible | Hexagonal architecture; dynamic packs; tool catalog; MCP; config-driven | Packs composable at runtime; tools registered in catalog; MCP for external integrations. |
| Phased build, blueprint upfront | PLAN.md, BACKLOG.md, REQUIREMENTS.md, this doc | 14 phases + post-phase work complete; 22 ADRs in DECISIONS.md. |

When adding features or refactoring, check this table and the principles in §2 to ensure the repo continues to follow the vision.
