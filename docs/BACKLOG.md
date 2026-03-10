# agentic-concierge: Prioritised Backlog

**Purpose:** Single source of truth for *what to work on next, in what order, and why*.
Each item is self-contained: a fresh session can pick up any item using only this file, the
referenced source files, and [DECISIONS.md](DECISIONS.md).

**How to use this file**
- Items within each tier are ordered by priority (top = most urgent).
- When starting an item: add `**Status: IN PROGRESS — <date>**` to the item.
- When complete: move the item to the **Done** section; add `**Completed: <date>**`.
- New items go into the appropriate tier with full context written in at the time.
- Never leave an item as "IN PROGRESS" without updating STATE.md too.

**How to resume after an interruption**
1. Read [STATE.md](STATE.md) — confirms current phase and last verified state.
2. Read this file — find the first non-done item; that is what to work on.
3. Run `make test` — confirm **1429 pass** before touching code.
4. Start the item; mark it IN PROGRESS here and in STATE.md.

---

## Tier 1 — Fix before Phase 2 (correctness and robustness)

These items are defects or gaps that make the system incorrect or fragile in ways that will
compound as Phase 2 adds more complexity. Do these before adding any new Phase 2 work.

---

### ~~T1-1: Validate `finish_task` payload before returning it~~ **DONE 2026-02-24**

**Why:** The LLM can call `finish_task` with an empty or partial argument object. The current
code spreads `tc.arguments` directly into the payload (`{"action": "final", **tc.arguments}`)
with no validation. If `summary` is missing the result is silently malformed. Callers (HTTP API,
CLI) will return a payload that is missing required fields with no error surfaced to the user.

**What to change:**
- `src/agentic_concierge/application/execute_task.py` — where `finish_payload` is set (around
  the `if tc.tool_name == pack.finish_tool_name` block):
  - Validate `tc.arguments` contains at minimum `"summary"`.
  - If validation fails: log a `tool_result` event with the error, send the error back to the
    LLM as a tool result (so it can retry), and do **not** set `finish_payload`.
- `src/agentic_concierge/domain/errors.py` — add `FinishTaskValidationError` if needed.

**Acceptance criteria:**
- [ ] LLM calling `finish_task({})` causes the error to be returned to the LLM as a tool result,
      not silently accepted as a final payload.
- [ ] LLM calling `finish_task({"summary": "x"})` succeeds as before.
- [ ] New unit test in `tests/test_execute_task.py` (create this file) covering both cases.
- [ ] `pytest tests/ -k "not real_llm and not verify"` still passes (45+).

**Files:** `src/agentic_concierge/application/execute_task.py`, `tests/test_execute_task.py` (new)

---

### ~~T1-2: Replace bare `except Exception` in tool execution~~ **DONE 2026-02-24**

**Why:** `execute_task.py` catches `except Exception` around tool execution. This swallows
`KeyboardInterrupt`, `SystemExit`, `MemoryError`, and other non-recoverable signals. More
importantly, it hides the *nature* of failures: a sandbox `PermissionError` (security event),
a `FileNotFoundError` (tool bug), and a `ValueError` (bad arguments) are all treated identically.

**What to change:**
- `src/agentic_concierge/application/execute_task.py` — around `pack.execute_tool(...)`:
  - Catch specific exceptions: `PermissionError` (sandbox violation), `ValueError`/`TypeError`
    (bad args), `OSError` (filesystem), `Exception` as final fallback — but log each distinctly.
  - Add a `kind: "tool_error"` event to the runlog when a tool fails, distinct from a normal
    `tool_result`. Include `tool_name`, `error_type`, `error_message`.
  - Do NOT re-raise — the LLM should receive the error as a tool result so it can adapt.
- `src/agentic_concierge/infrastructure/workspace/run_log.py` — add `log_tool_error()` if not present.

**Acceptance criteria:**
- [ ] A tool that raises `PermissionError` (sandbox escape) produces a `tool_error` runlog event.
- [ ] A tool that raises `ValueError` (bad args) produces a `tool_error` runlog event.
- [ ] `KeyboardInterrupt` propagates up normally (is not caught).
- [ ] Tests in `tests/test_execute_task.py` covering sandbox violation and bad-args paths.
- [ ] `pytest tests/ -k "not real_llm and not verify"` still passes.

**Files:** `src/agentic_concierge/application/execute_task.py`,
`src/agentic_concierge/infrastructure/workspace/run_log.py`, `tests/test_execute_task.py`

---

### ~~T1-3: Add structured logging (Python `logging` module)~~ **DONE 2026-02-24**

**Why:** There is currently no `logging` usage anywhere. Diagnosing failures in the HTTP API
or CLI requires inspecting `runlog.jsonl` files on disk. You cannot enable debug output,
cannot see what the server is doing at runtime, and cannot integrate with log aggregators.
This is below the bar for any system intended to run unattended.

**What to add:**
- Every module that has observable behaviour should log at appropriate levels:
  - **DEBUG:** LLM request/response payloads, tool arguments/results, step counter.
  - **INFO:** Task started/completed, model resolved, specialist recruited, run_id created.
  - **WARNING:** Fallback paths (minimal payload retry, plain-text LLM response), 400 errors.
  - **ERROR:** Unrecoverable errors (model not found, sandbox violation, config invalid).
- Use a single logger per module: `logger = logging.getLogger(__name__)`.
- In `interfaces/cli.py`: configure root logger at INFO or DEBUG based on `--verbose` flag.
- In `interfaces/http_api.py`: configure at startup (use uvicorn's existing logging config).
- Do NOT log sensitive data (API keys, file contents by default).

**Key files to instrument first (highest value):**
1. `src/agentic_concierge/application/execute_task.py` — task start/end, each step, LLM fallback
2. `src/agentic_concierge/infrastructure/ollama/client.py` — request sent, response received, retries
3. `src/agentic_concierge/infrastructure/llm_discovery.py` — model resolved, fallbacks
4. `src/agentic_concierge/interfaces/http_api.py` — request received, result returned
5. `src/agentic_concierge/interfaces/cli.py` — add `--verbose` flag wiring

**Acceptance criteria:**
- [ ] `concierge run "list files" --pack engineering --verbose` prints INFO-level log lines to stderr.
- [ ] Running the HTTP server and hitting `POST /run` produces log output at INFO.
- [ ] No sensitive data (API keys) in default logs.
- [ ] No existing tests broken.
- [ ] Logging is silent by default in unit tests (configure `logging.NullHandler` at library root).

**Files:** Almost all `src/` modules. Start with execute_task, client, llm_discovery, interfaces.

---

### ~~T1-4: Make the specialist registry extensible (config-driven, not hardcoded)~~ **DONE 2026-02-24**

**Why:** `infrastructure/specialists/registry.py` has a hardcoded `_BUILDERS` dict. Every new
specialist pack requires editing core code. The vision explicitly expects new capability areas to
be added without architectural surgery. This also blocks Phase 2 which adds capability-based
routing — that feature is useless if packs can only be added by editing the registry.

**What to change:**
- `src/agentic_concierge/infrastructure/specialists/registry.py` — replace the hardcoded dict with
  one of these two strategies (prefer A):
  - **Strategy A (recommended): Config-driven factory map.**
    `ConciergeConfig.specialists` already exists as `dict[str, SpecialistConfig]`. Extend
    `SpecialistConfig` with an optional `builder` field (dotted module path, e.g.
    `"agentic_concierge.infrastructure.specialists.engineering:build_engineering_pack"`).
    The registry imports and calls the builder at `get_pack()` time.
    Built-in packs are registered via a default factory map keyed by `specialist_id`; config
    can override or add new ones.
  - **Strategy B: `importlib.metadata` entry points.**
    Define a `"agentic_concierge.specialists"` entry point group. Built-in packs are registered in
    `pyproject.toml`; external packs can register themselves the same way.
    This is the most Pythonic plugin pattern but requires a bit more setup.
- Either strategy must preserve backward compatibility with existing tests and config.

**Acceptance criteria:**
- [ ] Adding a new specialist pack does NOT require editing `registry.py`.
- [ ] Existing `engineering` and `research` packs work as before.
- [ ] A test in `tests/test_specialist_registry.py` (new) demonstrates registering a minimal
      custom pack without modifying core code.
- [ ] `pytest tests/ -k "not real_llm and not verify"` still passes.

**Files:** `src/agentic_concierge/infrastructure/specialists/registry.py`,
`src/agentic_concierge/config/schema.py` (if Strategy A), `pyproject.toml` (if Strategy B),
`tests/test_specialist_registry.py` (new)

---

## Tier 2 — Important, not blocking (quality and maintainability)

Do these after all T1 items, or interleaved if a T1 item is blocked/waiting.

---

### ~~T2-1: Extract shared tool-definition helpers (DRY)~~ **DONE 2026-02-24**

**Why:** `_tool()` helper and the `finish_task` tool definition are duplicated between
`engineering.py` and `research.py`. They will diverge over time. Also blocks the extensibility
work (T1-4) because new packs will copy-paste the same boilerplate.

**What to change:**
- Create `src/agentic_concierge/infrastructure/specialists/tool_defs.py` with:
  - `def make_tool_def(name, description, parameters, required=None) -> dict` — the `_tool()` helper.
  - `def make_finish_tool_def(description, extra_properties=None, extra_required=None) -> dict`
    — builds the finish_task definition with common base fields (summary, artifacts, next_steps,
    notes) plus any pack-specific extras.
- Update `engineering.py` and `research.py` to import from `tool_defs.py`.

**Acceptance criteria:**
- [ ] `_tool()` is not defined in either `engineering.py` or `research.py`.
- [ ] `finish_task` base schema (summary, artifacts, next_steps, notes) is defined once.
- [ ] All existing pack tests still pass.

**Files:** `src/agentic_concierge/infrastructure/specialists/tool_defs.py` (new),
`engineering.py`, `research.py`

---

### ~~T2-2: Cache `load_config()` to avoid re-parsing on every HTTP request~~ **DONE 2026-02-24**

**Why:** `http_api.py` calls `load_config()` on every `POST /run`. The function reads the file
from disk, parses JSON, and constructs a Pydantic model — every single time. This is a silent
per-request cost that will matter at any reasonable call rate.

**What to change:**
- `src/agentic_concierge/config/loader.py` — use `functools.lru_cache` with `maxsize=1` on
  `load_config()` OR cache the result at module level with a `_cache: ConciergeConfig | None`.
  - The cache must be invalidatable in tests (use `load_config.cache_clear()` if lru_cache).
  - Config should be reloaded if `CONCIERGE_CONFIG_PATH` changes (accept this limitation for now;
    document it).
- `src/agentic_concierge/interfaces/http_api.py` — no changes needed if caching is in loader.

**Acceptance criteria:**
- [ ] `load_config()` only reads the filesystem once per process (subsequent calls return cached).
- [ ] Tests can reset cache between test runs (via `cache_clear()` or module-level reset).
- [ ] No existing tests broken.

**Files:** `src/agentic_concierge/config/loader.py`

---

### ~~T2-3: Expand test coverage for error paths in `execute_task`~~ **DONE 2026-02-24**

**Why:** The error paths in the tool loop are entirely untested:
- LLM returns plain text (no tool calls) — code at execute_task.py lines ~106-117
- `max_steps` exhausted — code at execute_task.py lines ~163-171
- Tool raises an exception — covered by T1-2 but needs tests written
- LLM returns malformed tool arguments (`{"_raw": "..."}` fallback in client.py)

Without these tests, regressions in error handling will go undetected.

**What to add:** `tests/test_execute_task.py` (started in T1-1 and T1-2):
- `test_plain_text_response_is_final_payload` — mock LLM returns `LLMResponse(content="done", tool_calls=[])`.
- `test_max_steps_exceeded_produces_timeout_payload` — mock LLM always returns a non-finish tool call.
- `test_tool_exception_is_returned_to_llm` — mock tool raises `ValueError`; next LLM call should see error in messages.
- `test_malformed_tool_arguments_logged` — mock LLM returns `tool_calls` with `arguments="{not json}"`.

**Acceptance criteria:**
- [ ] All 4 tests above are implemented and pass.
- [ ] `pytest tests/ -k "not real_llm and not verify"` passes (49+ tests).

**Files:** `tests/test_execute_task.py`

---

### ~~T2-4: Log sandbox violations as security events (audit trail)~~ **DONE 2026-02-24**

**Why:** When a tool call tries to escape the sandbox (`PermissionError` from `safe_path()`),
this is currently silently returned as `{"error": "..."}` in the tool result with no
distinguishing mark. There is no audit trail that a potentially adversarial input attempted
a path escape. For any system running LLM-generated code, this is a meaningful security gap.

**What to change:**
- `src/agentic_concierge/application/execute_task.py` — in the scoped exception handler (T1-2):
  when the caught exception is `PermissionError`, write a `kind: "security_event"` entry to
  the runlog in addition to the `tool_error` entry.
- `src/agentic_concierge/infrastructure/workspace/run_log.py` — add `log_security_event()`.

**Acceptance criteria:**
- [ ] A `PermissionError` from tool execution produces a `security_event` runlog entry with
      `tool_name`, `error_message`, and `timestamp`.
- [ ] Test in `tests/test_execute_task.py`.

**Files:** `execute_task.py`, `run_log.py`, `tests/test_execute_task.py`

---

### ~~T2-5: Document and centralise magic numbers~~ **DONE 2026-02-24**

**Why:** `50_000` (output truncation in sandbox), `2000` (content truncation in execute_task),
`10.0` / `120.0` / `360.0` (various timeouts) are scattered through the code with no
explanation. Future maintainers cannot tell whether these are safe to change.

**What to change:**
- `src/agentic_concierge/config/schema.py` or a new `src/agentic_concierge/config/constants.py`:
  - `MAX_TOOL_OUTPUT_CHARS: int = 50_000` — explain: prevents OOM from runaway shell output.
  - `MAX_LLM_CONTENT_IN_RUNLOG_CHARS: int = 2_000` — explain: runlog size control.
  - `LLM_DISCOVERY_TIMEOUT_S: float = 10.0` — explain: fast check, don't block startup.
  - `LLM_CHAT_DEFAULT_TIMEOUT_S: float = 120.0` — explain: single generation step.
- Update all references to use these constants.
- Add a comment on each explaining the rationale.

**Acceptance criteria:**
- [ ] No bare numeric literals for the above values; all use named constants.
- [ ] Each constant has a docstring or comment explaining its purpose and rationale.
- [ ] All existing tests still pass.

**Files:** `config/constants.py` (new), `sandbox.py`, `execute_task.py`, `client.py`,
`llm_discovery.py`

---

## Tier 3 — Nice to have (polish and developer experience)

Do after T1 and T2, or pick up opportunistically when adjacent work is in progress.

---

### ~~T3-1: Add architecture diagram~~ **DONE 2026-02-24**

A single ASCII or Mermaid diagram in `docs/ARCHITECTURE.md` showing the layer boundaries,
key classes, and data flow (task in → LLM loop → result out). Invaluable for onboarding
new contributors and for reasoning about Phase 2 changes.

### ~~T3-2: Parametrize tests where multiple scenarios are similar~~ **DONE 2026-02-24**

`tests/test_packs.py` and `tests/test_router.py` repeat similar structures. Use
`@pytest.mark.parametrize` to reduce duplication and cover more cases with less code.

### ~~T3-3: Tie-breaking and documentation in `recruit.py`~~ **DONE 2026-02-24**

Current max() on keyword scores has undefined tie-breaking behaviour (Python dict ordering).
Document this; add a deterministic tie-break (e.g., config order) and a test for it.

### ~~T3-4: Validate specialist IDs at config load time~~ **DONE 2026-02-24**

Config can reference a `specialist_id` that doesn't exist in `config.specialists`. This only
fails at execution time (when `get_pack()` raises). Add a validator in `ConciergeConfig` that
ensures every specialist referenced in routes/defaults exists in `config.specialists`.

### ~~T3-5: Extract `Task` construction to shared helper~~ **DONE 2026-02-24**

CLI (`cli.py:63`) and HTTP API (`http_api.py:59-64`) both construct `Task(...)` with identical
field mapping. Extract to `_build_task(prompt, pack, model_key, network_allowed) -> Task` in
a shared module (e.g., `application/task_factory.py` or directly in `domain`).

---

## Phase 2 items (next phase, not started)

These are the Phase 2 deliverables from [PLAN.md](PLAN.md). Do not start these until all T1
items are done and the Phase 1 verification gate still passes.

---

### ~~P2-1: Capability model — define capabilities and map packs~~ **DONE 2026-02-24**

**What:** Define a set of capability IDs (e.g., `"code_execution"`, `"file_io"`,
`"systematic_review"`, `"web_search"`) and declare which capabilities each pack provides
in `ConciergeConfig.specialists[id].capabilities: list[str]`.

**Why:** Enables task→capabilities→pack routing that is grounded in what packs can actually do,
not keyword heuristics.

**Files:** `src/agentic_concierge/config/schema.py`, `docs/CAPABILITIES.md` (new)

---

### ~~P2-2: Task-to-capabilities mapping~~ **DONE 2026-02-24**

**What:** Given a task prompt, determine the required capability IDs. Start with a rules/keyword
approach (similar to current routing but keyed to capability IDs, not pack names). Later replace
with a small router model + JSON schema.

**Why:** Decouples "what capability is needed" from "which pack provides it" — enabling multi-pack
task forces in Phase 3.

**Files:** `src/agentic_concierge/application/recruit.py` (rewrite or extend)

---

### ~~P2-3: Recruit pack from capabilities~~ **DONE 2026-02-24**

**What:** Select the pack(s) whose declared capabilities cover the required capabilities.
For Phase 2: still single pack per run. Log `required_capabilities` and `selected_pack` in
run metadata.

**Files:** `src/agentic_concierge/application/recruit.py`, `execute_task.py`, `run_log.py`

---

### ~~P2-4: Log required capabilities and selected pack in run metadata~~ **DONE 2026-02-24**

**What:** `runlog.jsonl` and/or the `RunResult` metadata (`_meta` in HTTP response) should
include `required_capabilities: [...]` and `selected_pack: "..."`. This makes routing
decisions observable and debuggable.

**Files:** `execute_task.py`, `run_log.py`, `http_api.py`

---

### ~~P2-5: Update docs for Phase 2~~ **DONE 2026-02-24**

**What:** Update `STATE.md` (Phase 2 complete), `PLAN.md` (tick off deliverables), `VISION.md §8`
(alignment table), and `REQUIREMENTS.md` (describe capability-based routing as a functional
requirement).

---

## Phase 3 items (multi-pack task force) — **complete**

---

### ~~P3-1: Multi-pack recruitment (RecruitmentResult.specialist_ids)~~ **DONE 2026-02-24**

**What:** Changed `RecruitmentResult` from a single `specialist_id: str` to
`specialist_ids: List[str]` with `specialist_id` as a backward-compatible property.
Added `is_task_force` property. Added `_greedy_select_specialists()` to pick the
minimum set of specialists that covers all required capabilities.

**Files:** `src/agentic_concierge/application/recruit.py`

---

### ~~P3-2: Multi-pack execute_task (sequential execution, shared runlog)~~ **DONE 2026-02-24**

**What:** Extracted `_execute_pack_loop()` from `execute_task()`. `execute_task` now
loops over `specialist_ids`, running each pack in turn. Logs `pack_start` events for
multi-pack runs; step names are prefixed with specialist ID (`engineering_step_0`,
`research_step_0`) so the runlog clearly shows which pack each step belongs to.

**Files:** `src/agentic_concierge/application/execute_task.py`

---

### ~~P3-3: Context handoff and domain/HTTP updates~~ **DONE 2026-02-24**

**What:** Each subsequent pack receives the previous pack's `finish_task` payload as
context in its user message. Added `specialist_ids: List[str]` and `is_task_force`
property to `RunResult`. Updated `http_api.py` `_meta` to include `specialist_ids`
and `is_task_force`.

**Files:** `src/agentic_concierge/domain/models.py`, `src/agentic_concierge/interfaces/http_api.py`

---

### ~~P3-4: Tests for Phase 3~~ **DONE 2026-02-24**

**What:** Updated `test_capabilities.py` — replaced `test_mixed_prompt_routes_to_best_coverage`
with `test_mixed_prompt_routes_to_task_force` and added two more task-force specific tests.
New `tests/test_task_force.py` with 17 tests covering greedy selection, multi-pack recruitment,
sequential execution, runlog structure, context handoff, and `RunResult` properties.
Fast CI: **144 pass** (+22).

**Files:** `tests/test_capabilities.py`, `tests/test_task_force.py` (new)

---

### ~~P3-5: Docs and STATE updated for Phase 3~~ **DONE 2026-02-24**

**What:** BACKLOG.md Phase 3 section; STATE.md phase and CI count updated;
PLAN.md Phase 3 deliverables ticked off.

---

## Phase 4 items (observability and multi-backend LLM) — **complete**

---

### ~~P4-1: Generic/cloud LLM client + `ModelConfig.backend` field~~ **DONE 2026-02-24**

**What:** Added `backend: str = "ollama"` to `ModelConfig`. Created `infrastructure/chat/__init__.py` with `build_chat_client()` factory (dispatches on `backend`). Created `GenericChatClient` in `infrastructure/chat/generic.py` — bare OpenAI-compatible client, no Ollama 400 retry, raises immediately on non-2xx. Extracted shared `parse_chat_response()` into `infrastructure/chat/_parser.py`. Both `OllamaChatClient` and `GenericChatClient` import the shared parser. CLI and HTTP API now use `build_chat_client(resolved.model_config)` instead of hardcoded `OllamaChatClient`.

**Tests:** `tests/test_generic_client.py` — 15 tests.

---

### ~~P4-2: `concierge logs` CLI subcommand~~ **DONE 2026-02-24**

**What:** Added `concierge logs list` (Rich table of runs, sorted most-recent-first, respects `--limit`) and `concierge logs show <run_id>` (pretty-printed JSON events, optional `--kinds` filter) to `interfaces/cli.py`. Created `infrastructure/workspace/run_reader.py` with `RunSummary` dataclass, `list_runs()`, `read_run_events()`, `_parse_runlog()`, `_summarise_run()`. Silently skips malformed runlog lines.

**Tests:** `tests/test_logs_cli.py` — 18 tests.

---

### ~~P4-3: OpenTelemetry tracing (optional dep)~~ **DONE 2026-02-24**

**What:** Created `infrastructure/telemetry.py` with `_NoOpSpan`, `_NoOpTracer`, `setup_telemetry()`, `get_tracer()`, `reset_for_testing()`. Graceful no-op when `opentelemetry-sdk` is not installed. Added `TelemetryConfig` to `config/schema.py` (`enabled`, `service_name`, `exporter`, `otlp_endpoint`; supports `"none"` | `"console"` | `"otlp"`). Added `telemetry: Optional[TelemetryConfig]` to `ConciergeConfig`. Instrumented `execute_task.py` with `fabric.execute_task` (root span), `fabric.llm_call` (wraps `chat_client.chat()`), and `fabric.tool_call` (wraps `pack.execute_tool()`). Added `[otel]` optional dep to `pyproject.toml`. Wired `setup_telemetry()` into CLI `run` command and HTTP API `lifespan`.

**Tests:** `tests/test_telemetry.py` — 13 tests (no-op shim, OTEL-only span emission with `InMemorySpanExporter`, `TelemetryConfig` schema).

---

### ~~P4-4: Docs update for Phase 4~~ **DONE 2026-02-24**

**What:** Updated BACKLOG.md (this section), STATE.md (phase → Phase 4 complete, CI count → 194), PLAN.md (concrete Phase 4 deliverables + acceptance criteria).

---

## Phase 5 items (MCP tool server support) — **complete**

---

### ~~P5-1: Config schema — MCPServerConfig + mcp_servers~~ **DONE 2026-02-24**

**What:** Added `MCPServerConfig(BaseModel)` to `config/schema.py` (`name`, `transport`, `command`/`args`/`env` for stdio, `url`/`headers` for SSE, `timeout_s`; validators require `command` for stdio and `url` for SSE). Added `mcp_servers: List[MCPServerConfig]` to `SpecialistConfig` with a validator that rejects duplicate server names. +6 tests in `tests/test_config.py`.

**Files:** `src/agentic_concierge/config/schema.py`, `tests/test_config.py`

---

### ~~P5-2: Async execute_tool + pack lifecycle~~ **DONE 2026-02-24**

**What:** Made `BaseSpecialistPack.execute_tool()` async (calls sync tool functions directly — no executor needed). Added no-op `aopen()`/`aclose()` to `BaseSpecialistPack`. Updated `SpecialistPack` Protocol (`execute_tool` async, `aopen`/`aclose` added). Updated `_execute_pack_loop` to `await pack.execute_tool(...)` and wrap the step loop in `try/finally: await pack.aopen() / await pack.aclose()`. Updated `_StubPack.execute_tool` in `tests/test_specialist_registry.py`.

**Files:** `src/agentic_concierge/infrastructure/specialists/base.py`, `src/agentic_concierge/application/ports.py`, `src/agentic_concierge/application/execute_task.py`, `tests/test_specialist_registry.py`

---

### ~~P5-3: MCPSessionManager + converter~~ **DONE 2026-02-24**

**What:** Created `infrastructure/mcp/` package. `converter.py`: `mcp_tool_to_openai_def(prefixed_name, tool)` — substitutes empty schema when `inputSchema` is None. `session.py`: `MCPSessionManager` with `connect()`/`disconnect()` via `AsyncExitStack`, `list_tools()` returning prefixed OpenAI defs, `call_tool()` (strips prefix, returns `{"result": text}` or `{"error": text}`), `owns_tool()`. Top-level `mcp` imports guarded with try/except for graceful no-op when package absent. +12 tests in `tests/test_mcp_session.py` (all mocked).

**Files:** `src/agentic_concierge/infrastructure/mcp/__init__.py`, `session.py`, `converter.py`; `tests/test_mcp_session.py` (new)

---

### ~~P5-4: MCPAugmentedPack~~ **DONE 2026-02-24**

**What:** Created `infrastructure/mcp/augmented_pack.py` with `MCPAugmentedPack(inner, sessions)`: `aopen()` — `asyncio.gather()` connects + populates `_mcp_tool_defs`; `aclose()` — `asyncio.gather(..., return_exceptions=True)` ignores individual failures; `tool_definitions` — inner + MCP tools; `execute_tool()` — dispatches to owning session or inner pack; forwards `specialist_id`, `system_prompt`, `finish_tool_name`, `finish_required_fields`. +10 tests in `tests/test_mcp_augmented_pack.py`.

**Files:** `src/agentic_concierge/infrastructure/mcp/augmented_pack.py` (new); `tests/test_mcp_augmented_pack.py` (new)

---

### ~~P5-5: Registry integration~~ **DONE 2026-02-24**

**What:** Updated `ConfigSpecialistRegistry.get_pack()` to wrap the built pack with `MCPAugmentedPack` when `spec_cfg.mcp_servers` is non-empty. Import is guarded: raises `RuntimeError("mcp package not installed")` if `agentic_concierge.infrastructure.mcp` cannot be imported. +6 tests in `tests/test_mcp_registry.py`.

**Files:** `src/agentic_concierge/infrastructure/specialists/registry.py`, `tests/test_mcp_registry.py` (new)

---

### ~~P5-6: pyproject.toml + docs~~ **DONE 2026-02-24**

**What:** Added `mcp = ["mcp>=1.0"]` to `[project.optional-dependencies]`; also added `mcp>=1.0` to `dev` so CI tests can mock it. Updated BACKLOG.md (this section), STATE.md (phase → Phase 5 complete, CI count → ~242), PLAN.md (Phase 5 concrete deliverables).

**Files:** `pyproject.toml`, `docs/BACKLOG.md`, `docs/STATE.md`, `docs/PLAN.md`

---

## Phase 8 items — **complete**

Phase 7 is complete (P7-1 through P7-4 all done; fast CI: 342 pass). Phase 8 focused on concurrency (parallel task forces), real-time streaming, and run status observability. All P8-1 through P8-4 done; fast CI: **368 pass**.

---

### ~~P8-1: Parallel task force execution~~ **DONE 2026-02-25**

- `config/schema.py`: Added `task_force_mode: str = Field("sequential", ...)` to `ConciergeConfig`.
- `execute_task.py`: `_run_task_force_parallel()` + `_merge_parallel_payloads()`; parallel path via `asyncio.gather`; sequential is default/unchanged.
- 14 tests in `tests/test_parallel_task_force.py`.

---

### ~~P8-2: SSE run event streaming (HTTP API)~~ **DONE 2026-02-25**

- `execute_task.py`: `event_queue: Optional[asyncio.Queue]` param; `_emit()` helper; all events mirrored to queue; `run_complete` + `_run_done_` sentinels.
- `interfaces/http_api.py`: `POST /run/stream` returns `text/event-stream`; background asyncio task.
- 6 tests in `tests/test_run_streaming.py`.

---

### ~~P8-3: Run status endpoint~~ **DONE 2026-02-25**

- `interfaces/http_api.py`: `GET /runs/{run_id}/status` — reads runlog; returns `completed/running`; 404 if not found.
- 6 tests in `tests/test_run_status.py`.

---

### ~~P8-4: Docs update for Phase 8~~ **DONE 2026-02-25**

- STATE.md, BACKLOG.md, PLAN.md updated.

---

## Phase 7 items (next)

Phase 6 is complete (P6-1 through P6-4 all done; fast CI: 304 pass). Phase 7 focuses on upgrading the run index to semantic search, first real enterprise MCP integration (GitHub), and a dedicated enterprise research specialist. Work on P7-1 first.

---

### ~~P7-1: Semantic run index search (vector embeddings via Ollama)~~ **DONE 2026-02-25**

**Why:** P6-1 added keyword/substring matching for the run index. This is limited: "authentication" won't find runs about "login flow" or "OAuth". The vision requires "cross-run memory" that understands semantics. Vector embeddings give us semantic similarity search without any external cloud service — just the local Ollama embedding endpoint.

**What to build:**
- Extend `infrastructure/workspace/run_index.py`:
  - `RunIndexEntry` gets an optional `embedding: Optional[list[float]]` field.
  - New `embed_text(text: str, model: str, base_url: str) -> list[float]` async function — calls `POST /api/embeddings` on the Ollama endpoint.
  - New `semantic_search_index(query: str, workspace_root: str, config: ConciergeConfig, top_k: int = 10) -> list[RunIndexEntry]` — embeds the query, loads all entries, computes cosine similarity, returns top-k. Falls back to `search_index()` keyword search when no entries have embeddings.
  - `append_to_index()` updated to optionally embed the entry at write time (new `embed: bool` param; default False for backward compatibility — can opt in via config).
- `config/schema.py`: Add `RunIndexConfig(BaseModel)` with `embedding_model: Optional[str] = None` (e.g. `"nomic-embed-text"`). Add `run_index: RunIndexConfig = Field(default_factory=RunIndexConfig)` to `ConciergeConfig`.
- `execute_task.py`: Pass `config.run_index` to `append_to_index()` — embed when `embedding_model` is set.
- `interfaces/cli.py`: `concierge logs search` uses `semantic_search_index()` when embeddings are available; falls back to keyword search otherwise.

**Acceptance criteria:**
- [ ] `RunIndexEntry` can serialise/deserialise with `embedding` field (None = not embedded).
- [ ] `embed_text()` calls `POST /api/embeddings` and returns a float list; gracefully raises on HTTP error.
- [ ] `semantic_search_index()` ranks entries by cosine similarity to query embedding; falls back to keyword when no embeddings exist.
- [ ] Existing `search_index()` still works unchanged (backward compat).
- [ ] `RunIndexConfig(embedding_model="nomic-embed-text")` in config → `execute_task` embeds each run; absent/None → no embedding (unchanged behaviour).
- [ ] Tests: `tests/test_run_index_semantic.py` — 10–12 tests; all mocked (no real Ollama needed for fast CI). Fast CI stays green.

**Files:** `infrastructure/workspace/run_index.py`, `config/schema.py`, `application/execute_task.py`, `interfaces/cli.py`, `tests/test_run_index_semantic.py` (new)

---

### ~~P7-2: GitHub MCP integration + real tests~~ **DONE 2026-02-25**

**Why:** Phase 5 built the MCP infrastructure; P6-2 verified it with a filesystem server. The most immediately useful enterprise integration is GitHub — searching issues, PRs, code, and commit history. The `@modelcontextprotocol/server-github` package is the official MCP server.

**What to build:**
- `MCPServerConfig.env` already exists for auth tokens. No new config fields needed.
- Add a `github_search` capability ID to `config/capabilities.py` (`CAPABILITY_KEYWORDS`).
- `tests/test_mcp_real_github.py` (new, `@pytest.mark.real_mcp`):
  - Fixture `skip_if_github_token_missing` — skips when `GITHUB_TOKEN` env var is absent.
  - Fixture `github_mcp_config` — returns `MCPServerConfig(name="github", transport="stdio", command="npx", args=["@modelcontextprotocol/server-github"], env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]})`.
  - 4 tests: `test_list_tools_returns_non_empty`, `test_search_repositories`, `test_get_file_contents`, `test_unknown_tool_returns_error`.
- Add a config example in `docs/MCP_INTEGRATIONS.md` (new) showing how to wire the GitHub server into an `enterprise_research` specialist.

**Acceptance criteria:**
- [ ] `tests/test_mcp_real_github.py` passes when `GITHUB_TOKEN` is set and `npx` is in PATH.
- [ ] Tests are deselected from fast CI (`-k "not real_mcp"`).
- [ ] `docs/MCP_INTEGRATIONS.md` includes complete `ConciergeConfig` YAML examples for GitHub, Confluence, and Jira stubs (even if Confluence/Jira tests are deferred).
- [ ] Fast CI count unchanged.

**Files:** `tests/test_mcp_real_github.py` (new), `config/capabilities.py`, `docs/MCP_INTEGRATIONS.md` (new)

---

### ~~P7-3: Enterprise research specialist~~ **DONE 2026-02-25**

**Why:** §4.3 of the vision describes an enterprise research assistant that searches Confluence, GitHub, Jira, and Rally and produces reports with staleness and confidence notes. We now have all the infrastructure (MCP, multi-pack, run index). The missing piece is a specialist pack designed for this use case.

**What to build:**
- `infrastructure/specialists/enterprise_research.py` (new):
  - `build_enterprise_research_pack(cfg: SpecialistConfig) -> SpecialistPack`
  - System prompt: enterprise research mode — search across MCP-backed sources (GitHub, Confluence, Jira), produce structured report with links, confidence notes, and explicit staleness caveats.
  - Tools: all research tools + a `cross_run_search` tool that calls `semantic_search_index()` to retrieve relevant prior run summaries.
  - `specialist_id = "enterprise_research"`, `capabilities = ["enterprise_search", "systematic_review", "web_search"]`
- `config/capabilities.py`: Add `"enterprise_search"` to `CAPABILITY_KEYWORDS` with relevant keywords (`confluence`, `jira`, `github issue`, `enterprise`, `knowledge base`, `internal docs`).
- Default config: add `enterprise_research` specialist entry (with `mcp_servers: []` as placeholder; comment explaining how to add GitHub/Confluence servers).
- `tests/test_enterprise_research_pack.py` (new): 10 tests — system prompt content, capability declaration, `cross_run_search` tool definition, tool dispatch to inner pack or index search, `finish_task` definition.

**Acceptance criteria:**
- [ ] `infer_capabilities("search confluence for supply management policies")` returns `["enterprise_search"]`.
- [ ] `recruit_specialist(task)` selects `enterprise_research` for enterprise-search prompts.
- [ ] `cross_run_search` tool calls `semantic_search_index()` (or `search_index()` fallback).
- [ ] Tests: 10+ pass; fast CI stays green.

**Files:** `infrastructure/specialists/enterprise_research.py` (new), `config/capabilities.py`, default config, `tests/test_enterprise_research_pack.py` (new)

---

### ~~P7-4: Docs update for Phase 7~~ **DONE 2026-02-25**

**What:** Update STATE.md (phase → Phase 7 complete, CI count), PLAN.md (Phase 7 deliverables ticked off), VISION.md §7+§8 (Phase 7 in history; enterprise integrations row updated), BACKLOG.md done table.

---

## Phase 6 items (complete)

---

### ~~P6-1: Persistent cross-run memory (run index + summary store)~~ **DONE 2026-02-24**

**Why:** Every run is currently a black box — the fabric cannot build on previous work, refer to past findings, or avoid repeating itself across conversations or tasks. The vision describes an enterprise assistant that accumulates context. Without memory, each task starts cold.

**What to build:**
- A `RunIndex` infrastructure component that maintains a lightweight JSONL index of all past runs: `run_id`, `specialist_ids`, `prompt_prefix` (first 200 chars), `finish_summary` (from `payload["summary"]`), `timestamp`, `workspace_path`.
- Index is written atomically after each successful run (append to `.concierge/run_index.jsonl`).
- A `search_runs(query: str) -> list[RunSummary]` function (keyword / substring, no vector store yet) that lets a future pack or the orchestrator retrieve relevant prior results.
- Expose via `concierge logs search <query>` CLI subcommand.
- Phase 6.2 will add vector search; Phase 6.1 uses substring matching — simple, no new deps.

**Acceptance criteria:**
- [ ] Every `execute_task` run appends to the run index on success.
- [ ] `concierge logs search "kubernetes"` returns all prior runs whose prompt or summary contains that string.
- [ ] The index file is append-only JSONL; survives partial writes (write + atomic rename).
- [ ] Tests: `tests/test_run_index.py` — 8–10 tests; fast CI stays green.

**Files:** `infrastructure/workspace/run_index.py` (new), `interfaces/cli.py`, `application/execute_task.py`

---

### ~~P6-2: Real MCP server smoke test (filesystem server)~~ **DONE 2026-02-25**

**Why:** Phase 5 built all the MCP wiring but every test mocks the transport layer. We have never run a real MCP server end-to-end through the fabric. The `mcp` package is installed in dev but its integration with a real subprocess has never been verified.

**What to build:**
- A `tests/test_mcp_real_server.py` marked `@pytest.mark.real_mcp` (like `real_llm`) — deselected from fast CI but runnable with `-k real_mcp`.
- Uses the official `@modelcontextprotocol/server-filesystem` npm package (must be installed: `npm install -g @modelcontextprotocol/server-filesystem`) with a tmp directory as root.
- Test: connect `MCPSessionManager`, call `list_tools()`, call `read_file` (or equivalent), assert result dict contains `"result"`.
- A fixture `skip_if_npx_unavailable` that skips gracefully if `npx` is not in PATH.

**What was built:**
- `tests/test_mcp_real_server.py`: 5 tests (`test_list_tools_returns_non_empty`, `test_owns_tool_prefix`, `test_read_file_via_call_tool`, `test_unknown_tool_returns_error`, `test_reconnect_after_disconnect`). Two fixtures: `skip_if_npx_unavailable` (checks `npx` in PATH + probes package), `skip_if_mcp_not_installed` (skips if `mcp` Python package absent). All 5 pass against real `@modelcontextprotocol/server-filesystem` server.
- `pyproject.toml`: Added `[tool.pytest.ini_options] markers` with `real_llm` and `real_mcp` entries.
- Fast CI: 257 pass (unchanged).

**Files:** `tests/test_mcp_real_server.py` (new), `pyproject.toml` (add `real_mcp` marker)

---

### ~~P6-3: Containerised workspace isolation (Podman)~~ **DONE 2026-02-25**

**Why:** The engineering pack runs shell commands in a shared workspace directory. There is no OS-level isolation — a rogue LLM call can reach the host filesystem beyond the workspace. The vision explicitly calls for Podman-based containment for specialist workers.

**What to build:**
- A `ContainerisedSpecialistPack` wrapper (similar to `MCPAugmentedPack`) that:
  - On `aopen()`: starts a Podman container from a base image (configurable), mounts the workspace as `/workspace`.
  - Overrides the `shell` tool to forward commands into `podman exec` rather than `subprocess.run` locally.
  - On `aclose()`: stops and removes the container.
- `SpecialistConfig.container_image: Optional[str]` — when set, the registry wraps with `ContainerisedSpecialistPack`.
- Default: no container (existing behaviour unchanged).

**What was built:**
- `infrastructure/specialists/containerised.py`: `ContainerisedSpecialistPack` — starts `podman run -d --rm -v workspace:/workspace:Z image sleep infinity` on `aopen()`, runs shell via `podman exec -w /workspace`, stops container on `aclose()`. Applies command allowlist for defence-in-depth. `:Z` volume option for SELinux hosts (Fedora/RHEL). Calls `inner.aopen()`/`inner.aclose()` to propagate lifecycle to MCP sessions.
- `config/schema.py`: `container_image: Optional[str]` field on `SpecialistConfig`.
- `registry.py`: Wraps with `ContainerisedSpecialistPack` after MCP wrap when `container_image` is set.
- `pyproject.toml`: Added `podman` marker.
- `tests/test_containerised_pack.py`: 26 tests — 22 unit (mocked subprocess) + 4 `@pytest.mark.podman` integration (run against real python:3.11-slim; all pass).
- Fast CI: 283 pass (+26).

**Files:** `infrastructure/specialists/containerised.py` (new), `config/schema.py`, `infrastructure/specialists/registry.py`, `pyproject.toml`, `tests/test_containerised_pack.py` (new)

---

### ~~P6-4: Cloud LLM fallback (quality/capability gate)~~ **DONE 2026-02-25**

**Why:** ADR-008 defines the policy: cloud only when the local model cannot meet the quality or capability bar, not on connection failure. Phase 4 added `ModelConfig.backend = "generic"` and `GenericChatClient`. The fallback logic itself — detecting "local cannot meet bar" and routing to a cloud model — is missing.

**What was built:**
- `infrastructure/chat/fallback.py`: `FallbackPolicy(mode)` — evaluates `LLMResponse` against `"no_tool_calls"` / `"malformed_args"` / `"always"` policies (unknown mode = never trigger). `FallbackChatClient(local, cloud, cloud_model, policy)` — calls local first; if policy triggers, calls cloud and queues a `cloud_fallback` event in `pop_events()`.
- `config/schema.py`: `CloudFallbackConfig(model_key, policy="no_tool_calls")` + `cloud_fallback: Optional[CloudFallbackConfig]` on `ConciergeConfig`. Defaults to `None` — identical behaviour when absent.
- `execute_task.py`: Auto-wraps injected `chat_client` with `FallbackChatClient` when `config.cloud_fallback` is set (local import + `build_chat_client` for cloud). Drains `pop_events()` after each LLM call and logs `cloud_fallback` runlog events with `reason`, `local_model`, `cloud_model`.
- `tests/test_chat_fallback.py`: 21 tests — policy unit tests (8), client unit tests (8), config tests (3), execute_task integration tests (2). All mocked; no real cloud call needed. Fast CI: 304 pass (+21).

**Files:** `infrastructure/chat/fallback.py` (new), `config/schema.py`, `application/execute_task.py`, `tests/test_chat_fallback.py` (new)

---

## Done

| Item | Completed | Summary |
|------|-----------|---------|
| V2 Layer 3: Agent-Model Affinity (Issues #24–#27) | 2026-03-09 | 6 agent roles (router, planner, critic, coder, researcher, reviewer) with capability requirements in `agent_roles.py`. `assign_model()` with `must_differ_from` constraints (fail-open). `execute_graph_with_affinity()` wraps graph executor with per-node model assignment. Model preloading via `preload_hint()` for upcoming siblings (fire-and-forget). `concierge doctor` diagnostics with 6 checks. ~53 new tests across 4 test files. Fast CI: 1429 pass. |
| V2 Layer 2: Recursive Task Decomposition (Issues #21–#23) | 2026-03-09 | `TaskGraph` + `TaskNode` DAG with state machine replacing flat orchestration. `plan_task()` planner + critic loop (max 2 re-plans). `execute_graph()` runs ready leaves in parallel via `asyncio.gather`. `should_decompose()` for adaptive depth control. ~99 new tests across 3 test files. |
| V2 Layer 1: Model Runtime (Issues #15–#20) | 2026-03-09 | `InferenceBackend` protocol, `OllamaBackend`, `LlamaCppBackend`, `BackendRegistry` (discovery + health + failover), `LocalModelRuntime` (acquire/release with refcounting, LRU eviction, memory budgets), `CapabilityProbe` (micro-prompt probes with disk cache). `config/defaults/backends.yaml`. ~110+ new tests across 6 test files. |
| Specialist Marketplace Architecture (ADR-023 through ADR-028) | 2026-03-08 | 5-phase implementation: (A) Model Capability Registry — `ModelCapabilityProfile`, `get_profile()`, `match_models()`, `infer_task_capabilities()` in `infrastructure/model_profiles.py`. (B) Adaptive Finish Schemas — `FINISH_SCHEMAS` dict with `quick_answer`, `research_report`, `code`, `general` shapes in `finish_schemas.py`; `finish_schema_key` threaded through pack builders. (C) Independent Reviewer Model — `_select_reviewer_model()` picks reasoning-focused model different from doer. (D) Per-Specialist Model Selection + Consult Tool — `_select_specialist_model()` picks best model per task; `consult_specialist_model` tool in `infrastructure/tools/consult.py` lets TC agents consult non-TC models; `set_runtime_models()` on `SpecialistRegistry` protocol. (E) Capability-Driven Orchestrator — `required_capabilities` on `SpecialistBrief`; `_resolve_specialist_from_capabilities()` maps caps to templates; orchestrator prompt uses capability language. Key architectural fix: LLM config params (`base_url`, `backend`, `api_key`) moved out of `SpecialistRegistry` protocol into `ConfigSpecialistRegistry` instance state. ~61 new tests across 7 test files. Fast CI: 799 pass (+92). |
| Agent Delegation (ADR-022) | 2026-03-05 | `delegate_to_specialist` tool handled inline in `_execute_pack_loop`; spawns sub-specialist via nested `_execute_pack_loop`. Max depth 1, step budget capped at 15. 10 tests (`test_delegation.py`). Fast CI: 707 pass (+10). |
| Human Approval Mechanism (ADR-021) | 2026-03-05 | `request_approval` tool handled inline in `_execute_pack_loop`; blocks on `ApprovalChannel` protocol. Three implementations: `AutoApprovalChannel`, `CliApprovalChannel`, `HttpApprovalChannel`. CLI `--auto-approve` flag; HTTP `POST /runs/{run_id}/approve` endpoint; `approval_timeout_s` config. Runlog events: `approval_requested`, `approval_granted`, `approval_denied`. 12 tests (`test_approval.py`). Fast CI: 697 pass (+10). |
| Universal work review mechanism (Gate 4) | 2026-03-05 | Independent reviewer LLM call after doer's `finish_task` passes Gates 1–3. Reviewer uses read-only tools (`read_file`, `list_files`, `run_tests`) + `approve_work`/`request_revision` decision tools. Fail-open (plain text = approval). Max 2 review iterations; `_review_warning` annotation on exhaustion. `PROMPT_REVIEWER` in `prompts.py`; `_review_specialist_work()`, `_SAFE_REVIEWER_TOOLS`, `_APPROVE_WORK_TOOL_DEF`, `_REQUEST_REVISION_TOOL_DEF` in `execute_task.py`. `max_review_iterations` param threaded through `execute_task`/`resume_execute_task`/`_execute_pack_loop`/`_run_task_force_parallel`. Runlog events: `review_start`, `review_approved`, `review_rejected`. ADR-019. 8 new tests (`test_review.py`); 10+ existing test files updated. Fast CI: 655 pass (+5). |
| Dynamic Pack Composition | 2026-03-05 | Replaced hardcoded specialist packs with dynamic/template-based composition. `tool_catalog.py` (central registry of 8 tools), `dynamic_pack.py` (`build_dynamic_pack()` + `build_template_pack()` + `PackTemplate` + `PACK_TEMPLATES`). Orchestrator is sole routing path; keyword routing removed. Deleted: `engineering.py`, `research.py`, `enterprise_research.py`, `recruit.py`, `capabilities.py`. New tests: `test_tool_catalog.py`, `test_dynamic_pack.py`, `test_orchestrator_dynamic.py`. Fast CI: 650 pass. |
| Adaptive backend resolution | 2026-03-04 | `resolve_llm()` refactored with fallback chain: tries configured primary, then `BACKEND_PRIORITY[tier]` backends in order (vLLM → Ollama → inprocess → cloud). `ResolvedLLM` extended with `warnings`, `fallback_used`, `resolved_backend`. SERVER profile now includes Ollama as fallback. `_ensure_nano_model()` GGUF download in bootstrap. Doctor install hints detect launcher venv. HTTP API `_meta` includes fallback info. ADR-018. 11 new tests. Fast CI: 635 pass (+7). |
| P7-4: Docs update for Phase 7 | 2026-02-25 | STATE.md (phase 7 in progress → complete, CI 342); PLAN.md (Phase 7 deliverables all ticked); VISION.md §7 (Phase 7 in history) + §8 (enterprise integrations row updated); BACKLOG.md done table. |
| P7-3: Enterprise research specialist | 2026-02-25 | infrastructure/specialists/enterprise_research.py — cross_run_search tool (queries run index), file tools, web tools (network_allowed); SYSTEM_PROMPT_ENTERPRISE_RESEARCH (staleness/confidence notation, multi-source, structured reports); enterprise_research in DEFAULT_CONFIG with enterprise_search + github_search caps; registry._DEFAULT_BUILDERS updated; 16 tests — system prompt, capabilities, tool defs, cross_run_search execution, routing. Fast CI: 342 pass (+16). |
| P7-2: GitHub MCP integration | 2026-02-25 | tests/test_mcp_real_github.py — 4 tests (list_tools, search_repositories, get_file_contents, unknown_tool_returns_error); skip_if_github_token_missing + skip_if_npx_unavailable + skip_if_mcp_not_installed fixtures; github_search + enterprise_search capability IDs added to capabilities.py; docs/MCP_INTEGRATIONS.md with GitHub/Confluence/Jira/filesystem config examples. Fast CI: 326 pass (unchanged — real_mcp deselected). |
| P7-1: Semantic run index search | 2026-02-25 | RunIndexEntry.embedding (Optional[List[float]]); embed_text() via Ollama /api/embeddings (strips /v1 suffix); cosine_similarity(); semantic_search_index() with fallback to keyword; RunIndexConfig(embedding_model, embedding_base_url) on ConciergeConfig; execute_task embeds entry when configured; concierge logs search uses semantic when available. 22 tests. Fast CI: 326 pass (+22). |
| P6-4: Cloud LLM fallback | 2026-02-25 | FallbackPolicy (no_tool_calls / malformed_args / always) + FallbackChatClient with pop_events(); CloudFallbackConfig on ConciergeConfig; execute_task auto-wraps + logs cloud_fallback runlog events. 21 tests, all mocked. Fast CI: 304 pass (+21). |
| P6-3: Containerised workspace isolation (Podman) | 2026-02-25 | ContainerisedSpecialistPack — podman run/exec/stop lifecycle; :Z SELinux volume label; shell intercepted, other tools delegated; container_image on SpecialistConfig; registry wraps after MCP; 26 tests (22 unit + 4 real Podman). Fast CI: 283 pass (+26). |
| P6-2: Real MCP server smoke test | 2026-02-25 | tests/test_mcp_real_server.py — 5 tests using @modelcontextprotocol/server-filesystem via npx; fixtures skip gracefully when npx or mcp package absent; all 5 pass end-to-end. pyproject.toml: real_llm + real_mcp markers declared. Fast CI: 257 pass (unchanged). |
| P5-1 through P5-6: Phase 5 MCP tool server support | 2026-02-24 | MCPServerConfig + mcp_servers on SpecialistConfig; execute_tool async + aopen/aclose lifecycle; MCPSessionManager + converter; MCPAugmentedPack wrapper; registry transparent wrap; [mcp] optional dep. Fast CI: 243 pass (+34) |
| P4-1 through P4-4: Phase 4 observability + multi-backend LLM | 2026-02-24 | GenericChatClient + build_chat_client() factory + ModelConfig.backend; concierge logs list/show CLI; OpenTelemetry no-op shim + optional real OTEL (console/otlp); TelemetryConfig schema; execute_task spans (execute_task, llm_call, tool_call); setup_telemetry() wired into CLI + HTTP API lifespan; [otel] pyproject.toml extra. Fast CI: 194 pass (+50) |
| P3-1 through P3-5: Phase 3 multi-pack task force | 2026-02-24 | RecruitmentResult.specialist_ids (greedy selection); _execute_pack_loop(); sequential multi-pack execution with context handoff; pack_start events + prefixed step names; RunResult.specialist_ids + is_task_force; HTTP _meta updated; 17 new tests in test_task_force.py + 2 in test_capabilities.py. Fast CI: 144 pass (+22) |
| P2-1 through P2-5: Phase 2 capability routing | 2026-02-24 | CAPABILITY_KEYWORDS + capabilities on SpecialistConfig; infer_capabilities(); RecruitmentResult; two-stage routing (caps → keyword fallback); recruitment runlog event; required_capabilities in RunResult + HTTP _meta; docs/CAPABILITIES.md; REQUIREMENTS FR2 + VISION §8 updated. Fast CI: 122 pass (+17) |
| T3-5: Extract build_task() to domain | 2026-02-24 | build_task() in domain/models.py; (pack or "").strip() or None fixes subtle whitespace-only inconsistency between CLI and HTTP paths; exported from domain/__init__; 6 new tests |
| T3-4: Config validation at load time | 2026-02-24 | @model_validator on ConciergeConfig rejects empty specialists dict; docstring marks extension point for future cross-reference checks; 3 new tests |
| T3-3: Tie-breaking in recruit_specialist | 2026-02-24 | Explicit min(-score, config_index) replaces implicit max; docstring documents contract; 2 parametrized tie-break tests |
| T3-2: Parametrize tests | 2026-02-24 | test_packs, test_router, test_sandbox, test_llm_discovery; 94 pass (was 82; +12 named cases) |
| T3-1: Architecture diagram | 2026-02-24 | Complete rewrite of `docs/ARCHITECTURE.md`: ASCII layer overview, component map with all source files, data flow + sequence diagram, runlog events table, extension points, config/startup flow, dependency rule table |
| T2-5: Centralise magic numbers | 2026-02-24 | `config/constants.py` with 6 named constants + rationale comments; updated sandbox.py, execute_task.py, client.py, llm_discovery.py, engineering.py |
| T2-4: Log sandbox violations as security events | 2026-02-24 | `security_event` runlog entry alongside `tool_error` when `PermissionError` caught; 3 new tests |
| T2-3: Expand error-path test coverage | 2026-02-24 | 4 new tests: malformed args, multiple tool calls, finish_task + regular tool coexisting, unknown tool name |
| T2-2: Cache `load_config()` | 2026-02-24 | `@lru_cache(maxsize=1)`; autouse fixture in conftest.py resets cache between tests; 2 new tests |
| T2-1: Extract shared tool-definition helpers | 2026-02-24 | `tool_defs.py` with `make_tool_def()`, `make_finish_tool_def()`, shared file tool constants; engineering.py and research.py updated |
| T1-4: Extensible specialist registry | 2026-02-24 | `SpecialistConfig.builder` optional field; `_load_builder()` via importlib; `_DEFAULT_BUILDERS` dict; 12 new tests |
| T1-3: Structured logging | 2026-02-24 | `logging.NullHandler` at library root; per-module loggers; DEBUG/INFO/WARNING coverage; `--verbose` CLI flag |
| T1-2: Scoped exception handling in tool execution | 2026-02-24 | Four specific except clauses; `tool_error` runlog event (distinct from `tool_result`); 7 new tests; KeyboardInterrupt/SystemExit propagate normally |
| T1-1: Validate `finish_task` payload | 2026-02-24 | Required fields (pack-specific) validated before accepting; error returned to LLM as tool result so it can retry; 9 new tests in `tests/test_execute_task.py` |
| Phase 1: all 13 deliverables | 2026-02-23 | MVP with engineering + research packs, CLI, HTTP API, sandbox, runlog, local LLM |
| Native tool calling refactor | 2026-02-24 | Replaced JSON-in-content protocol with OpenAI function calling; `LLMResponse`/`ToolCallRequest` domain types; `finish_task` terminal tool |
| Fix `_param_size_sort_key` float parsing | 2026-02-24 | "8.0B" was parsed as 80 not 8; `sqlcoder:15b` was selected over `llama3.1:8b` as fallback; fixed with regex float parse |
| Fix "does not support tools" 400 handling | 2026-02-24 | `OllamaChatClient` now detects this specific error before retrying; raises clear `RuntimeError` |
| Fix `asyncio.to_thread` for `resolve_llm` in HTTP handler | 2026-02-24 | `resolve_llm` is blocking; was called directly in async FastAPI handler |
| Un-skip and fix `test_resolve_llm_filters_embedding_models` | 2026-02-24 | Wrong patch target fixed; test now passes |
| 5 new tests (packs, prompt content) | 2026-02-24 | `finish_task` in definitions, OpenAI format validation, prompt content checks |
| All 4 real-LLM E2E tests passing | 2026-02-24 | engineering, research, API POST, verify_working_real.py — all pass against Ollama 0.12.11 with llama3.1:8b |

---

## Adaptive Escalation — Dynamic Model Right-Sizing

**Status: DONE — ADR-020, implemented 2026-03-05**

**Concept:** Start every task with the smallest suitable model, monitor progress signals,
and escalate to a more capable model when the current approach isn't working. This is the
inverse of the existing timeout recovery (which downgrades to a smaller model on timeout).
Together they form a bidirectional convergence: quality issues → go bigger, timeouts → go
smaller.

**Escalation signals (detected during `_execute_pack_loop`):**
- Review rejections (Gate 4) — reviewer finds quality/correctness issues
- Loop detection triggers — same tool+args repeated without progress
- Quality gate failures (Gate 3) — `validate_finish_payload()` rejections
- Corrective re-prompts — repeated finish_task errors for missing fields
- Stalled steps — no meaningful tool calls after N turns

**Escalation actions (in order of severity):**
1. Swap to a larger model within the same family (e.g. qwen2.5:7b → qwen2.5:14b)
2. Swap to a different model family if same-family upgrade unavailable
3. Recruit a different specialist with a role better suited to the problem
4. Modify the tool/role mix (add tools the specialist didn't originally have)

**De-escalation (already exists):**
- Timeout recovery (`pick_smaller_model()` in `llm_discovery.py`) — on timeout, retry
  with the next smaller available model

**Key design questions (to resolve before implementation):**
- Where does escalation state live? (Per-pack loop? Per-task? Shared across task force?)
- How many escalation attempts before giving up? (Probably 2, matching review iterations.)
- Should escalation reset the doer's message history or continue from where it left off?
- How to surface escalation decisions to the user? (Runlog events, CLI warnings, HTTP _meta?)
- Should the reviewer model also be escalated, or should it always match the doer?

**Depends on:** Universal work review mechanism (Gate 4) — provides the primary quality
signal for escalation decisions.

**Files likely affected:** `application/execute_task.py` (`_execute_pack_loop`),
`infrastructure/llm_discovery.py` (model upgrade logic), `config/schema.py`
(escalation policy config), `application/orchestrator.py` (specialist re-recruitment).

---

## Phase 17+ — Multi-tenant, Web UI, Plugin Registry

**Status: FUTURE — not yet started**

| Item | Summary |
|------|---------|
| Multi-tenant HTTP auth | API keys + JWT; per-tenant rate limits; tenant isolation in run directories |
| Web UI / dashboard | React or HTMX frontend; real-time run streaming via SSE; run history browser |
| Plugin registry | Third-party specialist packs discoverable and installable from a registry URL |

These items require Phase 15 and 16 to be shipped first.  No implementation details
are locked — requirements will be elaborated when Phase 16 is nearing completion.

---

## Phase 16 — PyO3 Extension Module + Additional Specialist Packs

**Status: FUTURE — not yet started**

**Trigger:** Phase 16 should be started only after profiling confirms that
`cosine_similarity` (in `run_index.py`) is a CPU bottleneck at production scale
(index > 50 000 entries).  For smaller scales ChromaDB already handles the
vector work efficiently.

### P16-1: PyO3 `cosine_similarity` extension
- Scope: `cosine_similarity(a, b)` only — the Python application is I/O-bound
  on all other hot paths (LLM HTTP, tool subprocesses, file I/O).  No other
  Python→Rust FFI boundary is justified at current scale.
- Toolchain: `maturin` build; wheel published alongside Python wheel in `release.yml`.
- ADR: will add ADR-019 documenting the decision and the bottleneck evidence.

### P16-2: Additional specialist packs
- `data-analysis` pack — pandas / polars / matplotlib tools; structured-output finish.
- `devops` pack — kubectl, docker, terraform tooling (containerised by default).
- `documentation` pack — generates RST/Markdown docs from source; `sphinx`/`mkdocs` tools.

---

## Phase 15 — Windows Launcher + Homebrew Tap

**Status: FUTURE — not yet started**

### P15-1: Windows launcher (`x86_64-windows-msvc`)
- `exec.rs`: add `#[cfg(windows)]` variant using `CreateProcess` +
  `WaitForSingleObject` + `exit(child_exit_code)` to preserve correct exit
  codes without leaving a zombie launcher process.
- `setup.rs`: uv download URL → `uv-x86_64-pc-windows-msvc.zip`; use
  `ZipArchive` (pure-Rust `zip` crate) instead of tar+gz.
- CI: add `windows-latest` runner to `build-native` matrix in
  `build-launcher.yml` and `release.yml`; artifact name `concierge-x86_64-pc-windows-msvc.exe`.
- `install.ps1`: PowerShell one-liner to complement `install.sh`.

### P15-2: Homebrew tap formula
- `homebrew-tap/Formula/concierge.rb` in a separate `ausmarton/homebrew-tap` repo.
- Formula downloads `aarch64-apple-darwin` or `x86_64-apple-darwin` binary from
  GitHub Releases and verifies its Ed25519 signature (shell `openssl pkeyutl`).
- `release.yml` auto-bumps the formula via `gh pr create` on the tap repo.

---

## Phase 14 — Native Rust Hot Paths

**Status: COMPLETE — 2026-02-27**

### P14-1: `launcher/Cargo.toml` — add 3 deps — DONE
Pure-Rust deps: `flate2 = "1"`, `tar = "0.4"`, `ed25519-dalek = { version = "2", … }`.
Static musl linking unaffected — all three are pure Rust with no system C libs.

### P14-2: `launcher/src/setup.rs` — pure Rust tar extraction — DONE
`extract_uv(archive_path, dest_dir)` using `flate2::read::GzDecoder` + `tar::Archive`.
Iterates entries; finds the file named `uv`; unpacks and chmod 0o755.
Replaces `std::process::Command::new("tar")` subprocess in `ensure_uv()`.
`find_file()` helper removed (superseded). 2 new tests:
`test_extract_uv_from_synthetic_archive`, `test_extract_uv_missing_binary`.

### P14-3: `launcher/src/update.rs` — Ed25519 signed verification — DONE
`SIGNING_PUBLIC_KEY: [u8; 32]` — placeholder (all-zeros); replace with
`scripts/generate_signing_key.sh` output before first release.
`verify_binary_signature_with_key(binary_path, sig_path, pub_key_bytes)` —
inner function (testable with custom keys without dependency on placeholder).
`verify_binary_signature(binary_path, sig_path)` — public wrapper using the embedded key.
`apply_update()` updated: downloads `{binary_url}.sig`, calls `verify_binary_signature`,
cleans up on failure, atomic rename only on success.
Asset name updated: Linux → `concierge-{arch}-unknown-linux-musl`;
macOS → `concierge-{arch}-apple-darwin`.  5 new tests.

### P14-4: `launcher/src/exec.rs` — macOS cfg annotation — DONE
`#[cfg(unix)]` on `exec_python_concierge`.  Comment documents Phase 15 Windows path.
No functional change — `std::os::unix::process::CommandExt::exec()` is POSIX,
works on both Linux and macOS without branching.

### P14-5: `install.sh` — macOS platform dispatch — DONE
Replaced `[ "$(uname -s)" = "Linux" ]` guard with OS+arch case-block.
Linux: `*-unknown-linux-musl`. macOS: `*-apple-darwin`. Other OS: helpful error + exit 1.

### P14-6: `.github/workflows/build-launcher.yml` — macOS matrix — DONE
Renamed `build-musl` → `build-native`. Added `x86_64-apple-darwin` (macos-13)
and `aarch64-apple-darwin` (macos-latest) entries. `use_cross: false` for macOS
(native runners use `cargo build` directly). Binary size gate uses `perl -e` for
portable file size (GNU `stat -c%s` not available on macOS).

### P14-7: `.github/workflows/release.yml` — signing + macOS artifacts — DONE
`build-launcher-release` matrix extended with macOS targets.  `Sign binaries` step
added: downloads `LAUNCHER_SIGNING_KEY_PEM` CI secret, signs each binary with
`openssl pkeyutl -rawin`, outputs `{binary}.sig` alongside the binary.  Signing step
is graceful: if secret is not set a warning is emitted, release continues unsigned.
`Create GitHub Release` glob updated to include all `launcher-bins/concierge-*`
(catches `.sig` files too).

### P14-8: Application hot-path audit — DONE
All Python application hot paths are I/O-bound; no PyO3 extension justified at
current scale.  See ARCHITECTURE.md Section 10 for the analysis table.
PyO3 deferred to Phase 16 (only if `cosine_similarity` becomes a measurable bottleneck).

### P14-9: Docs — DONE
ADR-017 (Ed25519 signed self-update), ARCHITECTURE.md Section 10 (hot-path table),
BACKLOG.md (Phase 14 DONE; Phase 15/16/17+ futures), STATE.md, CHANGELOG.md,
`scripts/generate_signing_key.sh` (new).

### P14-10: `Makefile` — `setup-rust-toolchain` target — DONE
`make setup-rust-toolchain`: idempotent rustup user-local install (no system packages);
installs `stable` + `clippy` + `rustfmt` into `~/.cargo/bin/`.
`lint-rust` target updated to use `$(HOME)/.cargo/bin/cargo` so it works even when
system cargo (e.g. Fedora dnf) lacks clippy/rustfmt.

**Rust tests:** `make test-rust` → **20 pass** (was 13)
**Python fast CI:** `make test` → **618 pass** (unchanged)

---

## Phase 13 — Rust Thin Launcher

**Status: COMPLETE — 2026-02-26**

### P13-1: `launcher/Cargo.toml` — DONE
Rust crate manifest. Deps: `reqwest` (blocking + rustls-tls for musl), `serde`, `dirs`, `semver`, `anyhow`, `thiserror`. Dev: `tempfile`.

### P13-2: `launcher/src/config.rs` — DONE
`LauncherConfig` struct; `launcher_config()` constructor. Env overrides: `CONCIERGE_DATA_DIR`, `CONCIERGE_NO_UPDATE_CHECK`, `CONCIERGE_EXTRA`. 5 unit tests.

### P13-3: `launcher/src/exec.rs` — DONE
`exec_python_concierge()` — `execv()` replaces process image; strips `--self-update` from forwarded args. 1 unit test.

### P13-4: `launcher/src/setup.rs` — DONE
`ensure_environment()`, `upgrade_package()`, `installed_version()`. System-Python detection, uv download via GitHub Releases + system `tar`, venv creation, pip install. Error types: `NoPython`, `VenvCreation`, `PackageInstall`, `UvNotExecutable`. 3 unit tests.

### P13-5: `launcher/src/update.rs` — DONE
`check_latest_release()` (network errors silently swallowed), `apply_update()` (atomic rename), `is_newer()` (semver). `ARCH_STR` const. 4 unit tests.

### P13-6: `launcher/src/main.rs` — DONE
Orchestration only. `parse_launcher_args()` (no clap). `main()` flow: self-update → passive hint → `ensure_environment` → `exec`. Module dependency graph enforced.

### P13-7: `.github/workflows/build-launcher.yml` — DONE
CI: `cargo test` + `cargo clippy -D warnings` + `cargo fmt --check`. Cross-compile matrix (x86_64/aarch64 musl) via `cross`. Binary size gate < 15 MB. Artifact upload.

### P13-8: `release.yml` + `install.sh` — DONE
`build-launcher-release` job added to release workflow; launcher binaries attached to GitHub Release. `install.sh`: POSIX one-liner; detects arch; atomic install to `~/.local/bin`.

### P13-9: Docs — DONE
README (Quick install section), CHANGELOG (Unreleased), BACKLOG (this section), STATE.md, ARCHITECTURE.md (Section 9), DECISIONS.md (ADR-016 consequences updated).

---

## Phase 12 — Quality Gates, LLM Orchestrator, and Session Continuation

**Status: COMPLETE — 2026-02-26 — 599 fast CI pass**

### Phase 12A — Engineering Quality Gates (P12-1 to P12-4) — DONE

**Deliverables:**
- `infrastructure/tools/test_runner.py`: `run_tests()` — auto-detects pytest/cargo/npm; returns `{passed, failed_count, error_count, summary, output, framework}`. `pytest`, `cargo`, `npm` added to sandbox allowlist.
- Engineering pack: `run_tests` tool registered; `tests_verified: bool` added to `_FINISH_TOOL_DEF` required fields; `EngineeringSpecialistPack.validate_finish_payload()` rejects `tests_verified=False`.
- `BaseSpecialistPack.validate_finish_payload()` no-op default added; `execute_task._execute_pack_loop` calls it as Gate 3 (after required-fields Gate 2).
- `SYSTEM_PROMPT_ENGINEERING` updated with quality gate instructions.
- **New test files:** `tests/test_run_tests_tool.py` (15 tests), `tests/test_engineering_pack_quality.py` (5 tests).

### Phase 12B — LLM Orchestrator (P12-5 to P12-10) — DONE

**Deliverables:**
- `application/orchestrator.py`: `SpecialistBrief`, `OrchestrationPlan` dataclasses; `orchestrate_task()` makes one LLM call with `create_plan` tool; filters unknown IDs; forces `synthesis_required=True` for multi-specialist; falls back to `llm_recruit_specialist` on any error.
- `execute_task.py`: Replaced `llm_recruit_specialist` with `orchestrate_task`; `_get_brief()` helper; brief injected into specialist user messages; `orchestration_plan` runlog event emitted; `_synthesise_results()` async function called when `synthesis_required=True`; `plan.mode` overrides `task_force_mode` for multi-specialist runs.
- `interfaces/cli.py`: `concierge plan "<prompt>"` command — calls `orchestrate_task`, prints Rich panel with mode/synthesis/assignments.
- **New test file:** `tests/test_orchestrate_task.py` (20 tests); `+4` to `tests/test_execute_task.py`.

### Phase 12C — Session Continuation (P12-11 to P12-13) — DONE

**Deliverables:**
- `infrastructure/workspace/run_checkpoint.py`: `RunCheckpoint` dataclass; `save_checkpoint()` (atomic write via tmp+rename); `load_checkpoint()` (returns None on missing/corrupt); `delete_checkpoint()`; `find_resumable_runs()`.
- `infrastructure/workspace/__init__.py`: checkpoint functions exported.
- `execute_task.py`: `_create_initial_checkpoint()`, `_update_checkpoint()`, `_delete_run_checkpoint()` helpers wired into `execute_task()`; `resume_execute_task()` function that skips completed specialists and seeds `prev_finish_payload` from checkpoint.
- `interfaces/cli.py`: `concierge resume <run-id>` command; `(resumable)` marker in `concierge logs list`.
- **New test files:** `tests/test_run_checkpoint.py` (16 tests), `tests/test_resume.py` (8 tests).

---

## Phase 10 — Self-sizing bootstrap, three-layer inference, profile-based features

**Status: COMPLETE — 2026-02-26 — 495 fast CI pass**
**Spec:** See `docs/PLAN.md` Phase 10 section.
**ADRs:** ADR-012 through ADR-016 in `docs/DECISIONS.md`.
**Target fast CI:** ~473 pass (+71 new tests across 7 new test files).

Items are listed in implementation order — earlier items are prerequisites for later ones.

---

### P10-1: `bootstrap/system_probe.py` — detect system resources

**Why first:** Everything else (profile selection, feature flags, backend decisions) depends on knowing what the machine looks like. This is the foundation of the entire phase.

**Deliverable:** `SystemProbe` dataclass + `probe_system()` function. Detects:
- CPU: `os.cpu_count()`, `platform.machine()` → `cpu_arch` ("x86_64" / "aarch64" / "apple_silicon")
- RAM: `psutil.virtual_memory()` → `ram_total_mb`, `ram_available_mb`
- GPU: `subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])` for NVIDIA; `["rocm-smi", "--showmeminfo", "vram", "--json"]` for AMD; `platform.machine() == "arm64"` + Darwin check for Apple Silicon. Returns `list[GPUDevice]` (empty if none found).
- Disk: `shutil.disk_usage(cache_path)` → `disk_free_mb`
- Internet: `httpx.head("https://1.1.1.1", timeout=3.0)` → `internet_reachable: bool`
- Ollama: `shutil.which("ollama")` → `ollama_installed`; `httpx.get("http://localhost:11434/api/tags", timeout=2)` → `ollama_reachable`
- vLLM: `httpx.get(vllm_base_url + "/health", timeout=2)` → `vllm_reachable` (default `http://localhost:8000`)
- mistral.rs: `importlib.util.find_spec("mistralrs") is not None` → `mistralrs_available`

**Test file:** `tests/test_system_probe.py` — 15 tests. All external calls mocked. Cover: NVIDIA GPU parse, AMD GPU parse, Apple Silicon detection, no-GPU path, internet unreachable, Ollama not installed, Ollama reachable, vLLM reachable, mistralrs absent.

---

### P10-2: `bootstrap/model_advisor.py` — profile and model recommendations

**Why second:** Depends on `SystemProbe`. Produces `SystemProfile` used by everything downstream.

**Deliverable:** `ProfileTier` enum + `SystemProfile` dataclass + `advise_profile(probe: SystemProbe) -> SystemProfile`.

Tier thresholds:
- nano: ram < 8 GB (regardless of GPU)
- small: 8–16 GB RAM, VRAM < 4 GB
- medium: 16–32 GB RAM OR 4–12 GB VRAM
- large: 32–64 GB RAM OR 12–24 GB VRAM
- server: 64 GB+ RAM OR 24 GB+ VRAM OR multi-GPU (2+ devices)

Max concurrent agents: `max(1, min(floor((available_ram_mb - max(2048, total_ram_mb * 0.15) - 512) / model_ctx_mb), cpu_cores - 1))`

Model recommendations per tier (all support tool calling):

| Tier | routing (in-process GGUF) | fast | quality |
|------|--------------------------|------|---------|
| nano | qwen2.5:0.5b | qwen2.5:3b | phi3:mini |
| small | qwen2.5:0.5b | qwen2.5:7b | qwen2.5:7b |
| medium | qwen2.5:0.5b | qwen2.5:7b | qwen2.5:14b |
| large | qwen2.5:0.5b | qwen2.5:14b | qwen2.5:32b |
| server | qwen2.5:0.5b | qwen2.5:32b | qwen2.5:72b |

**Test file:** `tests/test_model_advisor.py` — 10 tests. Each tier threshold; edge cases (exactly 8 GB, Apple Silicon unified memory, multi-GPU server detection, max_concurrent_agents formula).

---

### P10-3: `config/features.py` — FeatureSet and profile feature mapping

**Why third:** Feature flags must exist before any infrastructure factory checks them.

**Deliverable:**
```python
class Feature(str, Enum):
    INPROCESS | OLLAMA | VLLM | CLOUD | MCP | BROWSER | EMBEDDING | TELEMETRY | CONTAINER

PROFILE_FEATURES: dict[ProfileTier, frozenset[Feature]]

class FeatureDisabledError(RuntimeError):
    feature: Feature
    hint: str

@dataclass
class FeatureSet:
    enabled: frozenset[Feature]
    def is_enabled(self, f: Feature) -> bool
    def require(self, f: Feature, hint: str = "") -> None   # raises FeatureDisabledError if off
    @classmethod
    def from_profile(cls, tier: ProfileTier, overrides: FeaturesConfig) -> "FeatureSet"
```

`FeaturesConfig` (added to `config/schema.py`): all fields `Optional[bool] = None` (None = use profile default).

Feature defaults by profile: nano={inprocess,cloud}; small adds {ollama,mcp}; medium adds {vllm,embedding}; large adds {container}; server swaps ollama out, adds {telemetry}.

**Test file:** `tests/test_features.py` — 8 tests. Cover: each profile produces correct FeatureSet; override enables a disabled feature; override disables an enabled feature; `require()` raises FeatureDisabledError; `require()` passes silently when enabled.

---

### P10-4: `config/schema.py` additions — profile, features, resource_limits

**Deliverable:** Add to `ConciergeConfig`:
```python
profile: str = "auto"
features: FeaturesConfig = FeaturesConfig()
resource_limits: ResourceLimitsConfig = ResourceLimitsConfig()
```
New models:
```python
class ResourceLimitsConfig(BaseModel):
    max_concurrent_agents: int = 4
    max_ram_mb: Optional[int] = None
    max_gpu_vram_mb: Optional[int] = None
    model_cache_path: str = ""    # empty = use platformdirs default

class FeaturesConfig(BaseModel):
    inprocess: Optional[bool] = None
    ollama: Optional[bool] = None
    vllm: Optional[bool] = None
    cloud: Optional[bool] = None
    mcp: Optional[bool] = None
    browser: Optional[bool] = None
    embedding: Optional[bool] = None
    telemetry: Optional[bool] = None
    container: Optional[bool] = None
```
**Tests:** Extend `tests/test_config.py` — 6 new tests. Round-trip YAML; defaults; override serialises; `profile: auto` accepted.

---

### P10-5: `bootstrap/detected.py` — cross-platform detected.json

**Deliverable:** `detected_path() -> Path`; `save_detected(profile: SystemProfile) -> None`; `load_detected() -> SystemProfile | None`; `is_first_run() -> bool`.

Platform paths: Linux `~/.local/share/agentic-concierge/detected.json`; macOS `~/Library/Application Support/agentic-concierge/detected.json`; Windows `%LOCALAPPDATA%\agentic-concierge\detected.json`.

**Tests:** In `tests/test_first_run.py` — use `tmp_path` fixture to override platform path.

---

### P10-6: `bootstrap/backend_manager.py` — probe and manage backends

**Deliverable:** `BackendStatus(str, Enum)` + `BackendHealth` dataclass + `BackendManager`. Probes only backends enabled in the `FeatureSet`. Key methods:
- `async probe_all(feature_set: FeatureSet) -> dict[str, BackendHealth]`
- `async ensure_ollama(config: ConciergeConfig) -> BackendHealth`
- `async probe_vllm(base_url: str) -> BackendHealth`
- `probe_inprocess(feature_set: FeatureSet) -> BackendHealth` (sync)
- `get_healthy_backends() -> list[str]`

**Test file:** `tests/test_backend_manager.py` — 12 tests. All healthy; Ollama down but installed; vLLM disabled → not probed; inprocess not available; feature-disabled backend skipped entirely.

---

### P10-7: `infrastructure/chat/inprocess.py` — InProcessChatClient

**Deliverable:** `InProcessChatClient(ChatClient)`. Lazy-imports `mistralrs` only when instantiated. `is_available() -> bool` (checks importability without importing). `chat()` converts OpenAI format to mistralrs API, returns `LLMResponse`. Raises `FeatureDisabledError` if `mistralrs` absent.

Testable without the real wheel: mock `mistralrs` in `sys.modules`.

**Test file:** `tests/test_inprocess_client.py` — 8 tests. `is_available()` true/false; chat with mocked mistralrs; tool call parsing; plain text response; `FeatureDisabledError` on missing dep.

---

### P10-8: `infrastructure/chat/vllm.py` — VLLMChatClient

**Deliverable:** `VLLMChatClient(ChatClient)`. Pure httpx — no `vllm` Python package needed. `health_check() -> bool` (GET `/health`); `list_models() -> list[str]` (GET `/v1/models`); `chat()` delegates to `GenericChatClient`-style OpenAI-compat HTTP call.

**Test file:** `tests/test_vllm_client.py` — 8 tests. Health check healthy/unhealthy; list models; chat happy path; non-2xx raises; timeout.

---

### P10-9: Update `build_chat_client()` for new backends

**Deliverable:** In `infrastructure/chat/__init__.py`, dispatch `"inprocess"` → `InProcessChatClient` and `"vllm"` → `VLLMChatClient`. Lazy imports inside each branch. Extend existing chat factory tests (+4 tests).

---

### P10-10: `bootstrap/first_run.py` — FirstRunBootstrap orchestrator

**Deliverable:** `async run(interactive: bool = True, force_profile: str | None = None) -> SystemProfile`:
1. Check `is_first_run()` — return cached profile if detected.json exists (unless `--force`).
2. Load in-process model concurrently with `probe_system()`.
3. `advise_profile()` from probe.
4. If interactive: Rich panel with profile summary.
5. `BackendManager.ensure_ollama()` if ollama feature enabled.
6. Pull recommended models with Rich progress (non-interactive: silent).
7. `save_detected(profile)`.
8. Return `SystemProfile`.

**Test file:** `tests/test_first_run.py` — 10 tests. Happy path; skip when detected.json exists; force_profile overrides; non-interactive; Ollama not installed (graceful skip); model pull failure (warns, continues).

---

### P10-11: `concierge doctor` CLI command

**Deliverable:** New `doctor` subcommand in `interfaces/cli.py`. Calls `BackendManager.probe_all()`. Rich table output: detected hardware, profile tier, backend health per backend (icon + status + models available), active features checklist, suggestions for unhealthy backends.

**Test file:** `tests/test_doctor_cli.py` — 5 tests. Output contains expected sections; unhealthy backend shows fix hint; all-healthy path; no detected.json falls back to live probe.

---

### P10-12: `concierge bootstrap` CLI command

**Deliverable:** `bootstrap` subcommand. Options: `--profile PROFILE` (override auto-detection), `--non-interactive` (no prompts/progress). Calls `FirstRunBootstrap.run()`. Idempotent: re-running overwrites detected.json. Tests covered in `test_first_run.py` via CLI runner.

---

### P10-13: Add `psutil` and `platformdirs` to core deps; update extras

**Deliverable:** `pyproject.toml`:
- Add `"psutil>=5.9"` and `"platformdirs>=4.0"` to `[project] dependencies`
- Add `nano = ["mistralrs>=0.3"]`
- Add `embed = ["chromadb>=0.4"]` (placeholder, Phase 11)
- Add `browser = ["playwright>=1.40"]` (placeholder, Phase 11)
- Update `all = ["agentic-concierge[mcp,otel,embed,browser]"]`

---

### P10-14: ARCHITECTURE.md update

**Deliverable:** Update `docs/ARCHITECTURE.md` to reflect: bootstrap layer, three-layer inference stack, feature flag gating, platformdirs paths, new `BackendManager` in lifespan, new CLI commands. Do this when implementation begins, not before (keep docs honest).

---

## Tier 1 — Operational Excellence (ADR-033)

These items directly improve end-user experience and system performance. They build on the V2
multi-backend architecture to fully utilise available hardware.

---

### OPS-1: vLLM auto-start on SERVER/LARGE profiles

**Why:** vLLM supports continuous batching — true concurrent model inference. The task graph
already runs independent nodes in parallel via `asyncio.gather()`, but Ollama serialises
inference on a single GPU. On SERVER/LARGE profiles with GPU, vLLM should be auto-started
just like Ollama is today.

**What to change:**
- `bootstrap/backend_manager.py` — add `ensure_vllm()` mirroring `ensure_ollama()`.
  Auto-start vLLM with the best available model when feature is enabled and GPU detected.
- `config/schema.py` — add `vllm_start_cmd`, `vllm_model`, `vllm_gpu_memory_utilization`
  config fields with sensible defaults.
- `config/defaults/backends.yaml` — add vLLM start command template.
- Backend priority already prefers vLLM on LARGE/SERVER tiers (`_FALLBACK_BACKEND_PRIORITY`).

**Acceptance criteria:**
- On a SERVER profile with GPU, `concierge run` auto-starts vLLM if not already running.
- `concierge doctor` shows vLLM as healthy after auto-start.
- Existing Ollama-only setups are unaffected (vLLM requires GPU).
- New tests in `tests/test_backend_manager.py`.

**ADR:** ADR-033

---

### OPS-2: Parallel backend preference for concurrent task graph nodes

**Why:** When the planner decomposes a task into 3+ independent sub-tasks, they can execute
in parallel. But if all use Ollama (serial inference), the parallelism is wasted. The model
assignment layer should prefer backends that support concurrent requests when the graph has
concurrent nodes.

**What to change:**
- `application/affinity_executor.py` — when assigning models, check if >1 nodes will run
  concurrently. If so, prefer vLLM handles over Ollama handles for the concurrent batch.
- `infrastructure/backends/model_runtime.py` — add `supports_concurrent_requests: bool`
  property to `InferenceBackend` protocol.

**Acceptance criteria:**
- With both Ollama and vLLM available, concurrent graph nodes prefer vLLM.
- Sequential nodes (single ready node at a time) can still use Ollama.
- Fallback: if vLLM unavailable, Ollama handles all nodes (as today).
- Tests in `tests/test_affinity_executor.py`.

**ADR:** ADR-033

---

### OPS-3: Ollama connection pooling for concurrent requests

**Why:** Even without vLLM, Ollama (v0.14+) can handle concurrent requests. The current
client creates a fresh `httpx.AsyncClient` per call with no connection pooling. Adding a
shared client with connection pooling would improve performance for concurrent graph nodes.

**What to change:**
- `infrastructure/ollama/client.py` — use a shared `httpx.AsyncClient` instance with
  connection pooling instead of creating one per call.
- `infrastructure/backends/ollama.py` — same pattern for backend health checks.

**Acceptance criteria:**
- Concurrent graph nodes reuse HTTP connections to Ollama.
- No change in single-node behaviour.
- Tests verify shared client lifecycle.

