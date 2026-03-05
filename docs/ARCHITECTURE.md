# agentic-concierge: Architecture

**Purpose:** Layer boundaries, key classes, data flow, and extension points.
Read this before making structural changes or adding a new specialist pack.

See [DECISIONS.md](DECISIONS.md) for the *why* behind each major design choice.
See [BACKLOG.md](BACKLOG.md) for what is next.

---

## 1. Layer overview

agentic-concierge uses a strict hexagonal (ports-and-adapters) architecture.
Arrows show allowed import directions — the application core never imports
from infrastructure or interfaces.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Interfaces  (entry points — wire everything together)              │
│                                                                     │
│   cli.py (Typer)                 http_api.py (FastAPI)              │
│   concierge run / serve / logs      GET /health                     │
│   concierge plan / resume / doctor  POST /run  (blocking)           │
│   concierge bootstrap               POST /run/stream  (SSE)        │
│                                     GET /runs/{id}/status           │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ calls
┌──────────────────▼──────────────────────────────────────────────────┐
│  Application  (orchestration + ports)                               │
│                                                                     │
│   execute_task()  ·  _execute_pack_loop()  ·  resume_execute_task() │
│   _run_task_force_parallel()  ·  _merge_parallel_payloads()         │
│   _review_specialist_work()                                         │
│   _handle_request_approval()  ·  _handle_delegate_to_specialist()   │
│   orchestrate_task()  (OrchestrationPlan, SpecialistBrief)          │
│   ApprovalChannel protocol  (approval.py)                           │
│                                                                     │
│   Ports (Protocol interfaces defined here):                         │
│     ChatClient  ·  RunRepository                                    │
│     SpecialistRegistry  ·  SpecialistPack  ·  ApprovalChannel       │
└──────┬─────────────────────────────────┬────────────────────────────┘
       │ imports domain                  │ imports domain
┌──────▼──────────┐        ┌─────────────▼────────────────────────────┐
│  Domain         │        │  Infrastructure  (adapters)              │
│  (pure data)    │        │                                          │
│                 │        │  OllamaChatClient                        │
│  Task           │        │    → implements ChatClient               │
│  RunId          │        │  GenericChatClient (cloud / generic)     │
│  RunResult      │        │    → implements ChatClient               │
│  LLMResponse    │        │  VLLMChatClient (vLLM HTTP)              │
│  ToolCallRequest│        │    → implements ChatClient               │
│  RecruitError   │        │  InProcessChatClient (mistral.rs)        │
│  FabricError    │        │    → implements ChatClient               │
│                 │        │  FallbackChatClient (cloud quality gate) │
│                 │        │    → wraps ChatClient                    │
│                 │        │  FileSystemRunRepository                 │
│                 │        │    → implements RunRepository            │
│                 │        │  ConfigSpecialistRegistry                │
│                 │        │    → implements SpecialistRegistry       │
│                 │        │  BaseSpecialistPack                      │
│                 │        │    → implements SpecialistPack           │
│                 │        │  MCPAugmentedPack (MCP tool servers)     │
│                 │        │  ContainerisedSpecialistPack (Podman)    │
│                 │        │  Dynamic / template pack composition     │
│                 │        │  Tool catalog (8 tools)                  │
│                 │        │  sandbox · file / shell / web / browser  │
│                 │        │  test_runner · llm_discovery · bootstrap │
└─────────────────┘        │  telemetry (OpenTelemetry, optional)     │
                           │  approval/ (Auto, CLI, HTTP channels)    │
                           └──────────────────────────────────────────┘

  Config  (cross-cutting — any layer may import)
  ┌────────────────────────────────────────────────────────────────────┐
  │  ConciergeConfig · ModelConfig · SpecialistConfig · MCPServerConfig│
  │  CloudFallbackConfig · RunIndexConfig · TelemetryConfig            │
  │  FeaturesConfig · ResourceLimitsConfig                             │
  │  load_config() [lru_cache] · features.py · constants.py            │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component map

```
src/agentic_concierge/
│
├── domain/
│   ├── models.py        Task · RunId · RunResult
│   │                    LLMResponse · ToolCallRequest
│   └── errors.py        FabricError · RecruitError
│
├── application/
│   ├── execute_task.py  Main use-case: orchestrate → create run → tool loop(s) → result
│   │                    _execute_pack_loop(): one specialist's tool-calling loop
│   │                    _review_specialist_work(): Gate 4 reviewer mini-loop
│   │                    _run_task_force_parallel(): asyncio.gather for concurrent packs
│   │                    _merge_parallel_payloads(): combines parallel results
│   │                    _emit(): mirrors every event to optional event_queue (SSE)
│   │                    Loop detection: _LOOP_DETECT_WINDOW=8, _LOOP_DETECT_THRESHOLD=2
│   │                    Review: _MAX_REVIEW_ITERATIONS=2, _SAFE_REVIEWER_TOOLS
│   ├── orchestrator.py  orchestrate_task() — LLM-driven task decomposition
│   │                    OrchestrationPlan, SpecialistBrief (with tools/role for dynamic packs)
│   │                    create_plan tool; fallback to template_fallback on error
│   ├── approval.py      ApprovalChannel protocol; request_approval inline handler
│   │                    Defines the approval contract used by execute_task
│   ├── json_parsing.py  JSON extraction helpers
│   └── ports.py         ChatClient · RunRepository · SpecialistRegistry
│                        SpecialistPack · ApprovalChannel  (Protocol interfaces)
│
├── bootstrap/
│   ├── system_probe.py    SystemProbe · GPUDevice · probe_system()
│   ├── model_advisor.py   SystemProfile · advise_profile()
│   ├── backend_manager.py BackendManager · BackendHealth · BackendStatus
│   ├── first_run.py       FirstRunBootstrap — probe → advise → pull → save
│   └── detected.py        detected_path() · save/load_detected() · is_first_run()
│
├── infrastructure/
│   ├── chat/
│   │   ├── __init__.py      build_chat_client() — factory dispatches on ModelConfig.backend
│   │   │                      handles: "ollama", "generic", "vllm", "inprocess"
│   │   ├── generic.py       GenericChatClient — bare OpenAI-compatible (no Ollama quirks)
│   │   ├── inprocess.py     InProcessChatClient — mistral.rs via PyO3; lazy import
│   │   ├── vllm.py          VLLMChatClient — OpenAI-compat HTTP + health check
│   │   ├── _parser.py       parse_chat_response() — shared across clients
│   │   └── fallback.py      FallbackChatClient — wraps client; applies FallbackPolicy
│   │                          FallbackPolicy: no_tool_calls | malformed_args | always
│   │                          pop_events() — drain cloud_fallback events for runlog
│   ├── ollama/
│   │   └── client.py        OllamaChatClient
│   │                          • POST /v1/chat/completions (OpenAI format)
│   │                          • native tool calling (tool_calls in response)
│   │                          • 400 retry on "does not support tools" models
│   ├── mcp/
│   │   ├── session.py       MCPSessionManager(config: MCPServerConfig)
│   │   │                      connect() / disconnect() (stdio subprocess or SSE)
│   │   │                      list_tools() → OpenAI-format defs with mcp__name__tool prefix
│   │   │                      call_tool(name, args) → result dict
│   │   ├── converter.py     mcp_tool_to_openai_def() — MCP tool → OpenAI function schema
│   │   └── augmented_pack.py MCPAugmentedPack(inner, sessions)
│   │                          aopen(): inner.aopen() + asyncio.gather session connects
│   │                          aclose(): inner.aclose() + asyncio.gather session disconnects
│   │                          tool_definitions: inner tools + MCP tools merged
│   │                          execute_tool(): dispatch to owning session or inner pack
│   ├── workspace/
│   │   ├── run_repository.py  FileSystemRunRepository
│   │   ├── run_directory.py   create_run_directory()
│   │   │                        → .concierge/runs/<uuid>/{workspace/, runlog.jsonl}
│   │   ├── run_log.py         append_event() — one JSON line per event
│   │   ├── run_index.py       RunIndexEntry · append_to_index() · search_index()
│   │   │                        semantic_search_index() (cosine similarity via Ollama)
│   │   │                        embed_text() via Ollama /api/embeddings
│   │   │                        ChromaDB dispatch when provider="chromadb"
│   │   ├── run_index_chroma.py ChromaRunIndex — ChromaDB vector store backend
│   │   ├── run_checkpoint.py  RunCheckpoint · save/load/delete_checkpoint()
│   │   │                        find_resumable_runs() — scan for incomplete runs
│   │   └── run_reader.py      list_runs() · read_run_events() → RunSummary
│   ├── specialists/
│   │   ├── base.py            BaseSpecialistPack
│   │   │                        holds: system_prompt, tool_map, tool_definitions
│   │   │                        quality_gates: data-driven validate_finish_payload()
│   │   │                        async execute_tool() · aopen()/aclose() (browser lifecycle)
│   │   │                        _register_browser_tools() when Feature.BROWSER enabled
│   │   ├── tool_catalog.py    Central tool registry (8 tools: ToolEntry with metadata)
│   │   │                        Categories: file_io, code, web, search
│   │   │                        Each entry: openai_def, executor_factory, requires_network,
│   │   │                          quality_gate; executors are lazy-bound to SandboxPolicy
│   │   ├── dynamic_pack.py    build_dynamic_pack() — runtime pack from tool selections
│   │   │                        build_template_pack() — build from PackTemplate
│   │   │                        PACK_TEMPLATES: engineering, research, enterprise_research
│   │   │                        Finish-tool schemas per template
│   │   ├── registry.py        ConfigSpecialistRegistry
│   │   │                        Resolution: dynamic (tools+role) → template → custom builder
│   │   │                        wraps with MCPAugmentedPack when mcp_servers non-empty
│   │   │                        wraps with ContainerisedSpecialistPack when container_image set
│   │   │                        Passes FeatureSet (for browser tool gating)
│   │   ├── containerised.py   ContainerisedSpecialistPack(inner, image)
│   │   │                        intercepts execute_tool("shell") → podman exec
│   │   ├── tool_defs.py       make_tool_def() · make_finish_tool_def()
│   │   │                        READ_FILE_TOOL_DEF · WRITE_FILE_TOOL_DEF · LIST_FILES_TOOL_DEF
│   │   └── prompts.py         Composable prompt fragment system:
│   │                            PROMPT_CORE_RULES · PROMPT_ENGINEERING_QUALITY ·
│   │                            PROMPT_RESEARCH_RULES · PROMPT_ENTERPRISE_RULES ·
│   │                            PROMPT_LOOP_AVOIDANCE · PROMPT_REVIEWER
│   │                            Role descriptions: ROLE_ENGINEERING, ROLE_RESEARCH,
│   │                              ROLE_ENTERPRISE_RESEARCH
│   │                            generate_system_prompt(role, tools, quality_gates)
│   │                            Legacy constants preserved for backward-compatible tests
│   ├── tools/
│   │   ├── sandbox.py         SandboxPolicy · run_cmd() · safe_path()
│   │   │                        path-escape prevention + command allowlist
│   │   │                        absolute-path rejection with relative-path hint
│   │   ├── shell_tools.py     run_shell() — wraps run_cmd; timeout_s int coercion
│   │   ├── file_tools.py      read_text() · write_text() · list_tree()
│   │   ├── web_tools.py       web_search() · fetch_url(); timeout_s float coercion
│   │   ├── test_runner.py     run_tests() — auto-detect pytest/cargo/npm/unittest
│   │   │                        _detect_framework(), _parse_pytest_output(),
│   │   │                        _parse_cargo_output(), _parse_unittest_output()
│   │   └── browser_tool.py    BrowserTool (Playwright, optional)
│   │                            is_available() · 6 async tool methods · 30s timeout
│   │                            browse, click, fill, screenshot, extract_text, navigate
│   ├── approval/
│   │   ├── auto.py            AutoApprovalChannel — always approves (testing, CI)
│   │   ├── cli.py             CliApprovalChannel — interactive terminal prompt
│   │   └── http.py            HttpApprovalChannel — awaits POST /runs/{id}/approve
│   ├── llm_discovery.py       resolve_llm() — probe backend, select model
│   │                            discover_ollama_models() / discover_openai_models()
│   │                            select_model() — closest-distance sort, same-family pref
│   │                            resolve_routing_model() · pick_smaller_model()
│   │                            _TOOL_INCAPABLE_NAMES blocklist (sqlcoder etc.)
│   │                            Adaptive fallback: BACKEND_PRIORITY[tier] chain
│   ├── llm_bootstrap.py       ensure_llm_available() — start Ollama if needed
│   └── telemetry.py           setup_telemetry() · get_tracer()
│                                _NoOpSpan / _NoOpTracer — graceful no-op without OTEL
│                                spans: fabric.execute_task / fabric.llm_call / fabric.tool_call
│
├── interfaces/
│   ├── cli.py            Typer app:
│   │                       concierge run [--pack] [--model-key] [--stream] [--verbose] [--auto-approve]
│   │                       concierge serve [--host] [--port]
│   │                       concierge logs list [--workspace] [--limit]
│   │                       concierge logs show RUN_ID [--workspace] [--kinds]
│   │                       concierge logs search QUERY [--workspace] [--limit]
│   │                       concierge plan PROMPT
│   │                       concierge resume RUN_ID
│   │                       concierge doctor
│   │                       concierge bootstrap [--profile] [--non-interactive]
│   │                       StreamRenderer: Rich terminal rendering of SSE events
│   └── http_api.py       FastAPI:
│                           GET  /health
│                           POST /run               (blocking; returns finish_task payload + _meta)
│                           POST /run/stream        (SSE; streams all events until _run_done_)
│                           GET  /runs/{id}/status  (completed | running | 404)
│                           POST /runs/{id}/approve (human approval response)
│                           CONCIERGE_RATE_LIMIT=<n>: per-IP sliding-window rate limiting
│
└── config/
    ├── schema.py         ConciergeConfig · ModelConfig · SpecialistConfig
    │                       MCPServerConfig · CloudFallbackConfig · RunIndexConfig
    │                       TelemetryConfig · FeaturesConfig · ResourceLimitsConfig
    │                       DEFAULT_CONFIG (Ollama @ localhost:11434)
    ├── loader.py         load_config() — lru_cache(maxsize=1)
    │                       reads CONCIERGE_CONFIG_PATH env var (JSON or YAML)
    ├── features.py       Feature enum · ProfileTier enum · PROFILE_FEATURES mapping
    │                       FeatureSet · FeatureDisabledError · BACKEND_PRIORITY
    └── constants.py      MAX_TOOL_OUTPUT_CHARS · MAX_LLM_CONTENT_IN_RUNLOG_CHARS
                          LLM_DISCOVERY_TIMEOUT_S · SHELL_DEFAULT_TIMEOUT_S
                          DEFAULT_BACKEND_URLS · GGUF constants
```

---

## 3. Task execution: data flow

### ASCII flow (single pack)

```
 User / HTTP client
       │
       │  Task(prompt, specialist_id?, model_key, network_allowed)
       ▼
  execute_task()
       │
       ├─ [specialist_id is None?]
       │    orchestrate_task(prompt, config, chat_client, model=routing_model)
       │      → OrchestrationPlan(specialist_assignments, mode, synthesis_required)
       │      On error: fallback to first available template (zero regression)
       │
       ├─ [cloud_fallback configured?]
       │    wrap chat_client with FallbackChatClient(local, cloud, policy)
       │
       ├─ RunRepository.create_run()
       │    creates .concierge/runs/<uuid>/workspace/
       │    → (RunId, run_dir, workspace_path)
       │
       ├─ _create_initial_checkpoint()
       │
       ├─ [task_force_mode == "parallel" and len(specialist_ids) > 1]?
       │    _run_task_force_parallel(...)
       │      asyncio.gather(_execute_pack_loop × N)
       │      → _merge_parallel_payloads() → combined payload
       │
       └─ [sequential, default]
            for each specialist_id:
              SpecialistRegistry.get_pack(id, workspace_path, network_allowed,
                                          tools=brief.tools, role=brief.role)
                → dynamic pack (if tools+role) or template pack or custom builder
                → wraps with MCPAugmentedPack  (if mcp_servers)
                → wraps with ContainerisedSpecialistPack  (if container_image)
              _execute_pack_loop(pack, messages, …)
                previous pack's finish_payload forwarded as context
              _update_checkpoint(completed=..., payloads=...)
```

### _execute_pack_loop detail

```
 _execute_pack_loop()
       │
       ├─ await pack.aopen()    ← MCPAugmentedPack connects sessions;
       │                          BaseSpecialistPack starts browser if enabled
       │
       └─ Tool loop (up to max_steps, default 40)
              │
              ├─ [plain text, no tool calls?]
              │    corrective_reprompt (up to _MAX_PLAIN_TEXT_RETRIES=2)
              │    then escalate to larger model (ADR-020) or treat text as final payload
              │
              ├─ [loop detection: same (tool, args) ≥2× in last 8 calls?]
              │    inject "[SYSTEM] LOOP DETECTED" message
              │    emit loop_detected event
              │    if persistent (>threshold): escalate to larger model (ADR-020)
              │
              ├─ append_event("llm_request")  + _emit(event_queue, …)
              ├─ ChatClient.chat(messages, model, tools=pack.tool_definitions)
              │    [FallbackChatClient: try local → maybe retry on cloud]
              │    [timeout → pick_smaller_model() and retry]
              ├─ append_event("llm_response")  + _emit(…)
              │    drain FallbackChatClient.pop_events() → cloud_fallback events
              │
              └─ for each tool_call in response.tool_calls:
                   │
                   ├─ append_event("tool_call")  + _emit(…)
                   │
                   ├─ [tool_name == finish_task]
                   │    Gate 1: must have called at least one non-finish tool
                   │    Gate 2: required fields present (from pack schema)
                   │    Gate 3: pack.validate_finish_payload(args) → error or None
                   │    ├─ gate failed → send error to LLM, continue loop
                   │    └─ gates passed →
                   │         Gate 4: _review_specialist_work()
                   │           reviewer LLM with read-only tools + approve/reject
                   │           max _MAX_REVIEW_ITERATIONS=2 rejections
                   │           emit review_start / review_approved / review_rejected
                   │           ├─ approved → accept finish_payload, break loop
                   │           ├─ rejected (< max) → inject revision feedback, continue loop
                   │           └─ rejected (at max) → escalate to larger model (ADR-020)
                   │                or accept with _review_warning
                   │
                   ├─ [tool_name == request_approval]
                   │    emit approval_requested event
                   │    await approval_channel.request_approval(description, run_id)
                   │    ├─ approved → emit approval_granted, continue loop
                   │    └─ denied  → emit approval_denied, inject denial to LLM
                   │    Blocks until human responds or approval_timeout_s expires
                   │
                   ├─ [tool_name == delegate_to_specialist]
                   │    Guard: delegation_depth must be 0 (max depth 1)
                   │    Resolve sub-specialist pack from registry
                   │    Spawn nested _execute_pack_loop(depth=1, max_steps=15)
                   │    Return sub-specialist's result to parent as tool result
                   │
                   └─ [regular tool]
                        await pack.execute_tool(name, args)
                          runs inside SandboxPolicy (path-escape check, command allowlist)
                        ├─ success       → append_event("tool_result")  + _emit(…)
                        ├─ PermissionError → append_event("tool_error") + _emit(…)
                        │                    append_event("security_event") + _emit(…)
                        └─ other error   → append_event("tool_error")  + _emit(…)
                        result appended to messages → next LLM call
       │
       └─ await pack.aclose()   ← in finally block (MCP + browser cleanup)
```

After the pack loop(s):

```
  [synthesis_required and multiple payloads?]
    _synthesise_results() — one LLM call with synthesise_results tool
    exception → fallback to last specialist's payload

  append_event("run_complete") + _emit(event_queue, "run_complete", …)
  append entry to run_index.jsonl (cross-run memory)
  _delete_run_checkpoint()
  _emit(event_queue, "_run_done_", …)   ← terminates SSE stream
  return RunResult(…)
```

### SSE streaming (POST /run/stream)

```
  POST /run/stream
       │
       ├─ asyncio.Queue(maxsize=256)  ← event_queue
       │
       ├─ asyncio.create_task(_run_task_background())
       │    calls execute_task(…, event_queue=event_queue)
       │    on exception: put _run_error_ sentinel
       │
       └─ StreamingResponse(_sse_event_generator(event_queue))
              yields "data: {json}\n\n" for each event
              stops on _run_done_ or _run_error_ sentinel
```

### Sequence diagram (happy path, single pack)

```
 CLI/HTTP   execute_task  orchestrator  SpecReg  RunRepo   ChatClient  Pack
    │            │            │            │        │           │        │
    │──Task──────▶            │            │        │           │        │
    │            │──prompt────▶            │        │           │        │
    │            │◀──plan─────│            │        │           │        │
    │            │──get_pack(id,tools,role)─▶        │           │        │
    │            │◀──pack──────────────────│        │           │        │
    │            │──create_run──────────────────────▶│           │        │
    │            │◀──(run_id, dirs)─────────────────│           │        │
    │            │                                  │           │        │
    │         ┌──┤ step 0..N                        │           │        │
    │         │  │──append(llm_request)─────────────▶           │        │
    │         │  │──chat(msgs, tools)───────────────────────────▶        │
    │         │  │◀──LLMResponse(tool_calls)───────────────────│        │
    │         │  │──append(llm_response)────────────▶           │        │
    │         │  │──execute_tool(name, args)─────────────────────────────▶
    │         │  │◀──result dict────────────────────────────────────────│
    │         │  │──append(tool_result)─────────────▶           │        │
    │         └──┤ finish_task (Gates 1-4) → break              │        │
    │            │                                              │        │
    │            │──append(run_complete)────────────▶                    │
    │◀──RunResult│                                                       │
```

---

## 4. Runlog events

Every run produces `.concierge/runs/<id>/runlog.jsonl`.
Each line is a JSON record:

```json
{"ts": 1708800000.123, "kind": "<kind>", "step": "step_0", "payload": {...}}
```

| `kind` | When | Key payload fields |
|---|---|---|
| `orchestration_plan` | Orchestrator assigned specialists | `assignments`, `mode`, `synthesis_required`, `reasoning`, `routing_method` |
| `recruitment` | Specialist(s) selected (legacy compat) | `specialist_id`, `specialist_ids`, `required_capabilities`, `routing_method`, `is_task_force` |
| `task_force_parallel` | Parallel task force started | `specialist_ids`, `mode: "parallel"` |
| `pack_start` | One specialist starts (task forces) | `specialist_id`, `pack_index` |
| `llm_request` | Before each LLM call | `step`, `message_count` |
| `llm_response` | After each LLM call | `content` (truncated to 2 000 chars), `tool_calls` |
| `corrective_reprompt` | LLM returned plain text, re-prompting | `attempt`, `max_retries` |
| `loop_detected` | Same (tool, args) repeated in recent calls | `signature`, `count` |
| `cloud_fallback` | Local model fell back to cloud | `reason`, `local_model`, `cloud_model` |
| `tool_call` | Before executing a tool | `tool`, `args` |
| `tool_result` | Successful tool result, or accepted `finish_task` | `tool`, `result` |
| `tool_error` | Tool raised an exception | `tool`, `error_type`, `error_message` |
| `security_event` | `PermissionError` from tool (sandbox escape) | `event_type: "sandbox_violation"`, `tool`, `error_message` |
| `review_start` | Gate 4 reviewer begins inspection | `iteration` |
| `review_approved` | Reviewer accepted the work | `comment` |
| `review_rejected` | Reviewer requested revision | `feedback`, `iteration` |
| `model_escalated` | Adaptive escalation: switched to larger model | `trigger`, `from_model`, `to_model`, `escalation_count` |
| `approval_requested` | Specialist requested human approval | `description`, `tool_name` |
| `approval_granted` | Human approved the requested action | `run_id` |
| `approval_denied` | Human denied the requested action | `run_id`, `reason` |
| `delegation_start` | Specialist delegated to sub-specialist | `parent_specialist`, `child_specialist`, `sub_task` |
| `delegation_complete` | Sub-specialist delegation finished | `child_specialist`, `result` |
| `run_complete` | Run finished successfully | `run_id`, `specialist_ids`, `task_force_mode` |

`tool_error` and `security_event` are both emitted when a `PermissionError` occurs — the former for error classification, the latter as an explicit audit trail.

`run_complete` is written at the end of every successful run. `GET /runs/{id}/status` uses this event for completion detection.

---

## 5. Extension points

### New LLM backend

Implement the `ChatClient` protocol (`application/ports.py`):

```python
class MyBackendClient:
    async def chat(
        self, messages, model, *, tools=None,
        temperature, top_p, max_tokens,
    ) -> LLMResponse: ...
```

Register in `build_chat_client()` (`infrastructure/chat/__init__.py`) under a new backend name. No changes to `execute_task` or any other layer.

### New specialist pack

**Option A — dynamic pack (no code change):**

The orchestrator can compose a pack at runtime by selecting tools from the catalog and providing a role description. No config or code changes needed.

**Option B — new template:**

1. Add a `PackTemplate` entry to `PACK_TEMPLATES` in `infrastructure/specialists/dynamic_pack.py`.
2. Add a specialist entry to `DEFAULT_CONFIG` in `config/schema.py`.
3. The template specifies tool names, role description, and finish-tool schema.

**Option C — config-driven custom builder:**

1. Write a factory: `build_my_pack(workspace_path: str, network_allowed: bool) -> SpecialistPack`
2. In your `CONCIERGE_CONFIG_PATH` config, set `builder: "mymodule:build_my_pack"` on the specialist entry.

All options use `BaseSpecialistPack` and `tool_defs.make_tool_def()` / `make_finish_tool_def()`.

### New tool

1. Add the tool function in `infrastructure/tools/`.
2. Add an executor factory and `ToolEntry` to `TOOL_CATALOG` in `infrastructure/specialists/tool_catalog.py`.
3. The tool is now available for dynamic packs and templates.

### New MCP server (zero Python)

Add an `mcp_servers` entry to any specialist in config — see [MCP_INTEGRATIONS.md](MCP_INTEGRATIONS.md).

---

## 6. Config and startup

```
CONCIERGE_CONFIG_PATH (env var, optional JSON or YAML)
        │
        ▼
 load_config()  ← lru_cache(maxsize=1): read once per process
        │              call load_config.cache_clear() to force reload
        ▼
 ConciergeConfig
   ├── models: {key → ModelConfig(base_url, model, backend, api_key, temperature, …)}
   ├── specialists: {id → SpecialistConfig(description, capabilities, tools?, builder?,
   │                                        mcp_servers, container_image)}
   ├── profile: str              ("auto" | "nano" | "small" | "medium" | "large" | "server")
   ├── features: FeaturesConfig  (per-feature True/False/None overrides)
   ├── resource_limits: ResourceLimitsConfig (max_concurrent_agents, max_ram_mb, …)
   ├── routing_model_key: str    (default "fast")
   ├── task_force_mode: str      (default "sequential")
   ├── approval_timeout_s: int   (default timeout for human approval requests)
   ├── require_human_approval_for: list[str]  (default ["deploy", "push", "write_external"])
   ├── local_llm_ensure_available: bool  (default True)
   ├── local_llm_start_cmd: list[str]    (default ["ollama", "serve"])
   ├── auto_pull_if_missing: bool  (default True)
   ├── run_index: RunIndexConfig(embedding_model?, provider, chromadb_path?, …)
   ├── cloud_fallback: CloudFallbackConfig(model_key, policy)?
   └── telemetry: TelemetryConfig(enabled, exporter, otlp_endpoint)?

 CLI / HTTP startup:
   resolve_llm(config, model_key)
     ├── [ensure_available] ensure_llm_available() — start server if down
     ├── [adaptive fallback] probe BACKEND_PRIORITY[tier] chain
     ├── discover_ollama_models()  or  discover_openai_models()
     └── select_model() — closest-distance sort, same-family pref, tool-capable filter
         returns ResolvedLLM(base_url, model, model_config, available_models, fallback_used)
```

---

## 7. Dependency rule summary

| Layer | May import from |
|---|---|
| `domain` | stdlib only |
| `application` | `domain`, `config` |
| `infrastructure` | `domain`, `application.ports`, `config` |
| `interfaces` | all layers |
| `config` | stdlib + pydantic only |

Violations of this rule break testability: `execute_task` tests run with
mocked ports and never touch a real LLM or filesystem because the application
layer is kept clean.

---

## 8. Quality gates and review

### 8.1 Quality Gates (Gates 1–3)

```
_execute_pack_loop()
  ├── Gate 1: no prior tool call before finish_task → error to LLM
  ├── Gate 2: required fields missing (from pack's finish tool schema) → error to LLM
  └── Gate 3: pack.validate_finish_payload(args) → str or None
              BaseSpecialistPack: data-driven from quality_gates dict
                e.g. engineering: tests_verified=False → "run run_tests first"
              Default: always None (no gate)

run_tests(policy, framework, path, timeout_s) → dict
  Auto-detects: Cargo.toml→cargo, package.json+test→npm, pytest.ini/pyproject/test_*.py→pytest,
                test_*.py→unittest
  Runs via run_cmd() (sandbox allowlist applies)
  Returns: {passed, failed_count, error_count, summary, output, framework}
```

### 8.2 Gate 4: Universal Work Review (ADR-019)

After a specialist's `finish_task` passes Gates 1–3, an independent reviewer LLM inspects the workspace:

```
_review_specialist_work(pack, finish_payload, messages, ...)
  │
  ├─ Build reviewer prompt: PROMPT_REVIEWER + task summary + specialist's finish payload
  ├─ Reviewer tools: safe subset from pack (read_file, list_files, run_tests)
  │    + approve_work + request_revision decision tools
  │
  └─ Mini-loop (max 5 LLM steps, NOT a full _execute_pack_loop):
       ├─ LLM may call read_file/list_files/run_tests to inspect
       ├─ approve_work → emit review_approved, return approved
       ├─ request_revision → emit review_rejected, return feedback
       └─ plain text → treat as approval (fail-open)

  Fail-open rules:
  - Plain text response = approval
  - Any error = approval
  - Max steps exceeded = approval
  - Max _MAX_REVIEW_ITERATIONS=2 rejections → accept with _review_warning in payload
```

### 8.3 Dynamic Pack Composition

Packs are composed from a central tool catalog rather than hardcoded builder functions:

```
Tool Catalog (tool_catalog.py):
  8 tools: shell, read_file, write_file, list_files, run_tests,
           web_search, fetch_url, cross_run_search
  Each ToolEntry: name, category, openai_def, executor_factory,
                  requires_network, quality_gate

Pack Builders (dynamic_pack.py):
  build_dynamic_pack(tools, role, workspace_path, network_allowed)
    → select ToolEntry objects from catalog
    → generate_system_prompt(role, tool_names, quality_gates)
    → BaseSpecialistPack with quality_gates from tool metadata

  build_template_pack(template_name, workspace_path, network_allowed)
    → look up PACK_TEMPLATES[template_name]
    → build_dynamic_pack with template's tools/role/finish_schema

Prompt System (prompts.py):
  Composable fragments: PROMPT_CORE_RULES, PROMPT_ENGINEERING_QUALITY,
    PROMPT_RESEARCH_RULES, PROMPT_ENTERPRISE_RULES, PROMPT_LOOP_AVOIDANCE
  generate_system_prompt() selects relevant fragments based on
    which tools are included and what quality gates apply
```

### 8.4 LLM Orchestrator

The orchestrator decomposes tasks and assigns specialists:

```
execute_task()
  if task.specialist_id is None:
    plan = await orchestrate_task(prompt, config, chat_client, model=routing_model)
      ├── One LLM call with create_plan tool
      ├── Parses: assignments (specialist_id + brief + tools? + role?), mode,
      │   synthesis_required, reasoning
      ├── Filters unknown specialist IDs
      ├── Forces synthesis_required=True when len(assignments) > 1
      └── Falls back to first template on any error (zero regression)
    specialist_ids = [a.specialist_id for a in plan.specialist_assignments]
    task_force_mode = plan.mode  # may override config.task_force_mode
  else:
    specialist_ids = [task.specialist_id]  # explicit, no orchestrator call

  # Brief injection (per specialist):
  if brief_text:
    user_content += f"\n\nYour specific assignment:\n{brief_text}"

  # After all specialists complete:
  if plan.synthesis_required and len(all_payloads) > 1:
    final_payload = await _synthesise_results(...)
```

**CLI command:** `concierge plan "<prompt>"` — calls `orchestrate_task`, prints Rich panel, no run directory created.

### 8.5 Session Continuation

```
Checkpoint file: {run_dir}/checkpoint.json  (plain JSON, atomic write via .tmp + rename)

RunCheckpoint fields:
  run_id, run_dir, workspace_path, task_prompt
  specialist_ids, completed_specialists, payloads
  task_force_mode, model_key, routing_method, required_capabilities
  orchestration_plan (serialized dict or None)
  created_at, updated_at

execute_task() lifecycle:
  1. After create_run() + orchestration: _create_initial_checkpoint()
  2. After each sequential specialist: _update_checkpoint(completed=..., payloads=...)
  3. After run_complete event: _delete_run_checkpoint()

resume_execute_task(run_id, workspace_root, ...)
  ├── load_checkpoint() → ValueError if missing or all complete
  ├── Reconstructs plan from checkpoint.orchestration_plan
  ├── Loops specialists: skips completed, runs remaining
  ├── Updates checkpoint after each specialist
  ├── Emits run_complete and deletes checkpoint
  └── Returns RunResult

find_resumable_runs(workspace_root):
  Scans */checkpoint.json; returns run_ids with no run_complete in runlog
```

**CLI commands:**
- `concierge resume <run-id>` — loads checkpoint, resumes run, streams events
- `concierge logs list` — shows `(resumable)` next to interrupted run IDs

### 8.6 Human Approval (ADR-021)

Specialists can request human approval for high-impact actions via the `request_approval`
tool, which is handled inline in `_execute_pack_loop` (not dispatched to the pack):

```
_execute_pack_loop()
  └─ tool_call.name == "request_approval"
       ├─ emit approval_requested event
       ├─ await approval_channel.request_approval(description, run_id)
       │    ApprovalChannel protocol (application/approval.py):
       │      AutoApprovalChannel  — always approves (testing, --auto-approve)
       │      CliApprovalChannel   — interactive terminal prompt (default for CLI)
       │      HttpApprovalChannel  — awaits POST /runs/{run_id}/approve
       ├─ approved → emit approval_granted, return result to LLM
       └─ denied   → emit approval_denied, return denial reason to LLM

  Config: approval_timeout_s on ConciergeConfig (max wait before timeout)
  CLI:    concierge run --auto-approve (uses AutoApprovalChannel)
  HTTP:   POST /runs/{run_id}/approve  (unblocks HttpApprovalChannel)
```

### 8.7 Agent Delegation (ADR-022)

Specialists can delegate sub-tasks to other specialists via the `delegate_to_specialist`
tool, which spawns a nested `_execute_pack_loop`:

```
_execute_pack_loop(depth=0)
  └─ tool_call.name == "delegate_to_specialist"
       ├─ Guard: delegation_depth must be 0 (max nesting depth = 1)
       ├─ Resolve sub-specialist pack from SpecialistRegistry
       ├─ emit delegation_start event
       ├─ Spawn nested _execute_pack_loop(depth=1, max_steps=15)
       │    Sub-specialist runs in same workspace, shared runlog
       │    Step budget capped at 15 (prevents runaway sub-tasks)
       │    Cannot delegate further (depth guard)
       ├─ emit delegation_complete event
       └─ Return sub-specialist's finish payload as tool result to parent

  Constraints:
  - Max delegation depth: 1 (no recursive delegation)
  - Sub-specialist step budget: 15 (hard cap)
  - Sub-specialist shares workspace and runlog with parent
```

---

## 9. Rust thin launcher (Phase 13–14)

The `launcher/` Rust crate adds a static ~5 MB binary that bootstraps the Python environment and
then exec-replaces itself with the Python `concierge` binary. No Python or pip required to get started.

### Launcher flow

```
User runs: concierge [args]
    │
    ├── parse_launcher_args() → self_update?
    │
    ├── launcher_config()
    │     CONCIERGE_DATA_DIR   → data_dir (default: ~/.local/share/agentic-concierge)
    │     CONCIERGE_NO_UPDATE_CHECK=1 → skip update hint
    │     CONCIERGE_EXTRA      → pip extras (e.g. "mcp,otel")
    │
    ├── [if --self-update]  check_latest_release → apply_update → upgrade_package → exit 0
    │
    ├── [else if !skip_update]  check_latest_release → is_newer → print hint (never blocks)
    │
    ├── ensure_environment(config)
    │     Fast path: venv/bin/concierge exists? → ensure_extras() → return path.
    │     First-time:
    │       try_system_python() → >= 3.10 in PATH?
    │       If None: ensure_uv() → download uv, extract with flate2+tar (pure Rust)
    │       python3 -m venv  OR  uv venv --python 3.12
    │       pip install --upgrade agentic-concierge[{extra}]
    │       write installed_version + installed_extras markers
    │       return venv/bin/concierge
    │
    └── exec_python_concierge(bin)
          exec() replaces process image — Python inherits launcher PID, correct signals
```

### Module dependency graph

```
main.rs → config.rs        (data_dir, paths, env constants)
       → update.rs → config.rs   (GitHub Releases API; Ed25519 sig verification; atomic self-update)
       → setup.rs  → config.rs   (Python/venv/pip; pure-Rust flate2+tar extraction)
       → exec.rs   (no deps)     (execv the Python concierge binary; #[cfg(unix)])
```

### Distribution

| Channel | Who | How |
|---------|-----|-----|
| GitHub Releases binary | End users (Linux + macOS) | `install.sh` one-liner or direct download |
| PyPI wheel | Developers, CI | `pip install agentic-concierge` |
| Docker (GHCR) | Operators | `docker compose up` |

The launcher binary is a **thin distribution shim only** — all application logic stays in Python.

---

## 10. Hot-path analysis (Phase 14)

**Finding**: The Python application is I/O-bound on every hot path. No PyO3 extension
module is justified at current scale.

| Call site | Type | Typical latency | Rust (PyO3) benefit |
|-----------|------|-----------------|---------------------|
| LLM HTTP call | I/O | 100 ms – 10 s | None |
| Tool subprocess | I/O | 10 ms – 5 s | None |
| `safe_path()` | CPU | ~5 μs | Negligible |
| `cosine_similarity()` | CPU | ~50 μs/pair | Only if index > 50 k entries |
| JSON parsing | CPU | ~10 μs (C-backed) | None |

**Verdict**: No PyO3 extension justified at current scale. The one candidate,
`cosine_similarity`, is already superseded by ChromaDB for large-scale deployments.
Deferred pending profiling evidence at production scale (> 50 k entries).
