# agentic-concierge: Architecture Decision Records

**Purpose:** Records of significant technical decisions — *what* was decided, *why*, and
*what it means for future work*. Prevents re-litigating settled questions. When a decision
is revisited or superseded, mark the old record as Superseded and add a new one.

**Format:** Each record has Status, Context, Decision, and Consequences.

---

## ADR-001: Hexagonal architecture (ports and adapters)

**Status:** Accepted
**Date:** 2026-02-23

**Context:** The system needs to work with multiple LLM backends (Ollama, vLLM, OpenAI), multiple
interfaces (CLI, HTTP API), and multiple specialist packs. We also need to unit-test the application
logic without a real LLM or filesystem.

**Decision:** Use a strict layered hexagonal architecture:
- `domain/` — pure data structures and errors; no I/O, no external dependencies.
- `application/` — orchestration logic; depends on `domain/` and defines ports (protocols) that
  infrastructure must implement. Never imports from `infrastructure/` or `interfaces/`.
- `infrastructure/` — concrete adapters (LLM client, filesystem, specialists, tools).
- `interfaces/` — CLI (Typer) and HTTP (FastAPI) entry points; inject concrete infrastructure into application.
- `config/` — schema (Pydantic) and loading; can be imported by any layer.

**Consequences:**
- Adding a new LLM backend = implement `ChatClient` protocol (~30 lines). No other changes needed.
- Adding a new specialist = implement `SpecialistPack` and register it (see ADR-006).
- `application/execute_task.py` is fully testable with mocks — no real LLM or filesystem needed.
- All interfaces (CLI, HTTP) inject the same application function; behaviour is identical.

---

## ADR-002: Native OpenAI function calling (tools API) over JSON-in-content

**Status:** Accepted (supersedes original JSON-in-content design)
**Date:** 2026-02-24

**Context:** The original implementation required the LLM to output a specific JSON schema in
its message content (`{"action": "tool", "tool_name": "...", "args": {...}}`). This was fragile:
LLMs would add prose, wrap the JSON in markdown, or produce invalid JSON. Parsing required a
custom extraction pass. Every LLM call could fail in a new way.

**Decision:** Use the standard OpenAI `tools` parameter and `tool_calls` response field. The LLM
receives tool definitions as structured API input (not as prompt text); it emits tool calls as
structured API output (not as freeform text). The `finish_task` tool is the terminal signal
(see ADR-003).

**Consequences:**
- System prompts are clean — no JSON schema embedded in prompts.
- Tool calls are reliably parsed from structured API fields.
- Requires a tool-capable model (see ADR-007).
- `ChatClient.chat()` now accepts `tools: list[dict] | None` and returns `LLMResponse`
  (with `content` + `tool_calls` fields) instead of `str`.
- The 400 error handling in `OllamaChatClient` detects "does not support tools" and raises a
  clear `RuntimeError` before attempting a retry.

---

## ADR-003: `finish_task` as the terminal tool signal

**Status:** Accepted
**Date:** 2026-02-24

**Context:** The tool loop needs a stopping condition. Options considered:
1. LLM returns a message with no tool calls → treat as done.
2. LLM calls a special `finish_task` tool → treat as done.
3. A separate `stop` field in the response.

**Decision:** Option 2: a `finish_task` tool is included in every specialist pack's tool
definitions. When the LLM calls it, the loop terminates and the tool arguments become the
`RunResult.payload`. Option 1 is also handled as a fallback (plain text response with no tool
calls produces a minimal final payload) but is not the expected path.

**Consequences:**
- `finish_task` arguments are the final output format. Each pack defines its own schema
  (engineering: summary/artifacts/next_steps/notes; research: richer with citations, etc.).
- `finish_task` is NOT in `BaseSpecialistPack._tools` (the executor map) — it is handled
  specially in `execute_task.py`. Attempting to call `pack.execute_tool("finish_task", ...)`
  will raise `KeyError`. This is intentional.
- Required fields are validated before accepting the payload (BACKLOG T1-1 — done 2026-02-24).
  If any required field is missing, the error is returned to the LLM as a tool result so it can retry.

---

## ADR-004: `resolve_llm` for model discovery (don't hard-require a specific model)

**Status:** Accepted
**Date:** 2026-02-24

**Context:** The default config references `qwen2.5:7b` and `qwen2.5:14b`, but these models
may not be pulled on the user's machine. Failing with "model not found" on first run is a bad
experience. We also want the system to work out-of-the-box with whatever model the user has.

**Decision:** `resolve_llm(config, model_key)` queries the backend for available models and
selects the best match: the configured model if it exists, otherwise the smallest available
chat-capable model by parameter size. This means "best available" rather than "exact match".

**Consequences:**
- First run works without pulling a specific model.
- The `_param_size_sort_key` function must parse `"8.0B"` correctly as 8.0 (not 80 — a bug
  fixed 2026-02-24 with regex float parse).
- Embedding-only models are excluded from selection (filter in `_is_ollama_chat_capable`).
- Models that don't support tool calling (e.g. `sqlcoder:15b`) will cause a clear error if
  selected; the fix is to have a tool-capable model available (see ADR-007).
- `resolve_llm` is a synchronous blocking call (HTTP). In async contexts (FastAPI handler)
  it must be called via `asyncio.to_thread`.

---

## ADR-005: Sandbox scoping for file and shell tools

**Status:** Accepted
**Date:** 2026-02-23

**Context:** The engineering pack gives the LLM access to shell execution and file I/O. Without
scoping, the LLM could read or write arbitrary files on the host, or run arbitrary commands.

**Decision:**
- File tools (`read_file`, `write_file`, `list_files`) use `safe_path()` which resolves the
  real path and checks it is within the workspace root. `PermissionError` is raised otherwise.
- Shell tool (`shell`) uses a command allowlist (`SandboxPolicy.allowed_commands`). Commands not
  on the list raise `PermissionError`. The subprocess runs with `cwd=workspace_root`.
- `network_allowed` in the research pack gates web tools (web_search, fetch_url) — not
  implemented at the OS/network layer, just at the tool level.

**Consequences:**
- The shell allowlist must be maintained as new tools/languages are needed.
- Network is not OS-blocked even when `network_allowed=False` — the engineering pack's shell
  can still reach the network. This is intentional (documented) and acceptable for the current
  phase; true network sandboxing would require containers (Phase 4).
- Sandbox violations (PermissionError) produce a `tool_error` runlog event *and* a distinct
  `security_event` entry with `event_type: "sandbox_violation"` (BACKLOG T2-4 — done 2026-02-24).

---

## ADR-006: Extensible specialist registry (config-driven builders + MCP transparent wrap)

**Status:** Accepted — fully implemented
**Date:** 2026-02-24 (T1-4 completed); 2026-02-24 (Phase 5 MCP wrap added)

**Context:** Specialist packs need to be discoverable and constructable from a `specialist_id`
string. The original implementation used a hardcoded `_BUILDERS` dict in `registry.py`.

**Decision (T1-4 — done):** `SpecialistConfig` carries an optional `builder` field (dotted
import path, e.g. `"mypackage.packs.custom:build_custom_pack"`). `ConfigSpecialistRegistry`
imports and calls it at `get_pack()` time. A fallback `_DEFAULT_BUILDERS` map covers the
built-in `engineering` and `research` packs. Adding a new pack requires only a config entry —
no changes to `registry.py`.

**Decision (Phase 5 — done):** When `SpecialistConfig.mcp_servers` is non-empty, `get_pack()`
transparently wraps the returned pack in `MCPAugmentedPack`. The inner pack factory is
unaware of MCP; MCP attachment is a registry concern. Import of the `mcp` infrastructure is
lazy and guarded: a clear `RuntimeError` is raised if the optional `mcp` package is absent.

**Consequences:**
- Adding a new specialist pack = add a `builder:` entry to config. No registry edits needed.
- MCP tool servers are attached per-specialist in config. Pack factories need no changes.
- The `SpecialistPack` protocol is stable; the registry is the only place that handles wrapping.
- `execute_tool` is `async def` (ADR-011); sync tool functions are called directly without an executor.

---

## ADR-007: Require a tool-capable model; no fallback to JSON-in-content

**Status:** Accepted
**Date:** 2026-02-24

**Context:** When a model doesn't support tool calling (e.g. `sqlcoder:15b`), Ollama returns
`400 {"error": "... does not support tools"}`. We considered falling back to the original
JSON-in-content protocol.

**Decision:** No fallback. If the model doesn't support tools, raise a clear `RuntimeError`
with instructions to use a tool-capable model. Do not silently degrade to JSON-in-content.

**Rationale:** JSON-in-content is unreliable. Maintaining two code paths adds complexity.
The ecosystem of tool-capable local models is large enough (llama3.1, mistral, qwen2.5-coder,
deepseek-coder, etc.) that requiring one is reasonable. A clear error with guidance is better
than silently degraded behaviour.

**Consequences:**
- Users must have at least one tool-capable model pulled. The README should document this.
- `resolve_llm` currently selects by size without checking tool capability. If the smallest
  model happens to be tool-incapable, the user gets a clear error at runtime. A future
  improvement would probe tool capability during discovery (adds latency; deferred).

---

## ADR-008: Local-first LLM; cloud only when local capability/quality is insufficient

**Status:** Accepted
**Date:** 2026-02-23

**Context:** The vision is explicit: local LLM is the default and primary path. Cloud is used
only when local cannot meet quality or capability (not when the server is unreachable).

**Decision:** All current code targets local Ollama. `local_llm_ensure_available: true` by
default means the fabric starts Ollama if it isn't running. No cloud path exists yet.

**Consequences:**
- Cloud fallback is future (Phase 4+). When implemented, it must be triggered by "local model
  cannot meet quality or capability bar" — not by connection failures.
- The distinction matters architecturally: "server unreachable → start it" vs
  "model capability insufficient → use cloud model" are different code paths.

---

## ADR-009: Runlog as primary observability artifact

**Status:** Accepted
**Date:** 2026-02-23

**Context:** We need to be able to replay, debug, and audit every task run. Options:
1. Structured log per run (`runlog.jsonl`).
2. Global application log.
3. OpenTelemetry traces.

**Decision:** Per-run `runlog.jsonl` as the primary artifact. Every LLM request/response and
tool call/result is appended. Global application logging (option 2) is a pending addition
(BACKLOG T1-3) for operational concerns (HTTP request handling, startup, errors). OpenTelemetry
(option 3) is Phase 4+.

**Consequences:**
- Debugging a specific task = open its `runlog.jsonl`.
- Operational monitoring (what's the server doing right now?) is not possible until T1-3 is done.
- `runlog.jsonl` format is append-only JSONL; each line is `{"kind": "...", "ts": "...", ...}`.
  This format is stable and any change must be backward-compatible.

---

## ADR-010: Async-first application layer; sync tool implementations are acceptable

**Status:** Accepted (amended by Phase 5 — see ADR-011)
**Date:** 2026-02-24

**Context:** The tool loop is async (LLM calls are awaited). Individual tools (shell, file I/O,
web fetch) are sync functions. We could make tools async to allow concurrent execution.

**Decision:** The `SpecialistPack.execute_tool()` method is `async def` (as of Phase 5;
see ADR-011), but the underlying tool implementations remain sync functions called directly
from within `execute_tool`. Sequential execution (one tool per LLM turn) is preserved.
This is acceptable because:
1. Current pack tools are fast relative to LLM round-trips.
2. Sequential tool execution is predictable and easier to reason about.
3. The LLM drives the loop; it does not issue parallel tool calls within a single turn.

**Consequences:**
- Blocking sync calls in tools (subprocess, file I/O, httpx.Client) block the event loop for
  their duration. For short-running tools this is fine.
- `fetch_url` (research pack) uses `httpx.Client` (sync). If tool execution ever becomes
  concurrent, this will need refactoring to `httpx.AsyncClient`.
- `resolve_llm` (blocking) is correctly offloaded via `asyncio.to_thread` in the HTTP handler
  because it runs at request startup, outside the tool loop.
- MCP tool calls (`session.call_tool()`) are natively async and benefit from the async signature.

---

## ADR-011: Async pack lifecycle (`aopen`/`aclose`) for MCP subprocess management

**Status:** Accepted
**Date:** 2026-02-24

**Context:** MCP tool servers run as subprocesses (stdio transport) or long-lived HTTP
connections (SSE transport). They must be started before the tool loop begins and shut down
after it ends — even if the loop raises an exception. A sync interface cannot cleanly express
this because the connection/disconnection calls are themselves async (MCP SDK uses anyio).

**Decision:**
- `SpecialistPack.execute_tool()` is promoted to `async def` (Phase 5).
- `aopen()` and `aclose()` async lifecycle hooks are added to the `SpecialistPack` Protocol
  and `BaseSpecialistPack` (no-op defaults, so existing packs need no changes).
- `MCPAugmentedPack` overrides both: `aopen()` connects all sessions and populates MCP tool
  definitions; `aclose()` disconnects all sessions with `return_exceptions=True` so one
  failing disconnect never prevents the others from running.
- In `_execute_pack_loop`, `aopen()` is called *inside* a `try/finally` block so that `aclose()`
  is guaranteed to run even if `aopen()` itself raises partway through. This prevents leaking
  partially-connected MCP sessions.
- Tool names are namespaced `mcp__<server_name>__<tool>` to avoid collisions with native tools.

**Alternatives considered:**
- Context-manager protocol (`__aenter__`/`__aexit__`): cleaner for direct `async with` use,
  but would require changes to `_execute_pack_loop` callsites and adds complexity for callers
  that don't want lifecycle management (e.g. tests). The explicit `aopen`/`aclose` pair is
  simpler and works identically from the loop's perspective.
- Making sync tools async via `asyncio.to_thread`: unnecessary overhead. Sync functions called
  from an `async def execute_tool` are fine as long as they complete quickly.

**Consequences:**
- All test stubs implementing `SpecialistPack` must change `execute_tool` to `async def`.
- MCP server subprocesses are always cleaned up via the `finally` block in `_execute_pack_loop`.
- The optional `mcp` package is never imported at module level in `session.py`; the import is
  guarded by `try/except ImportError` and a `_MCP_AVAILABLE` flag, so the `infrastructure/mcp`
  package is importable without the dep installed. The registry performs a lazy import and raises
  a clear `RuntimeError` with an install hint when `mcp_servers` is configured without the package.

---

## ADR-012: Three-layer inference stack (in-process / Ollama / vLLM)

**Status:** Superseded by ADR-034
**Date:** 2026-02-26

**Context:** We need a single system that works on hardware ranging from a 4 GB RAM laptop to a
multi-GPU server, and that delivers an immediately useful response on first run before any model
server is set up. Ollama alone is the wrong choice for all scenarios: it is not designed for
high-throughput concurrent requests (it serves one request at a time per model), which is
exactly the workload produced by parallel specialist task forces. vLLM is the right choice for
concurrent workloads but requires more setup and does not serve small models efficiently.

**Decision:** Three inference layers coexist and complement each other:

1. **In-process (mistral.rs via PyO3 wheel)** — always present on every profile. Starts in
   milliseconds, no server required. Used as: (a) the primary inference engine on `nano`
   profile; (b) the dedicated routing/planning brain on all profiles (routing decisions are
   low-complexity tasks that should not consume Ollama/vLLM capacity); (c) the bootstrap
   agent that guides setup while heavier backends install in the background.

2. **Ollama (local model server)** — primary task-execution backend for `small` and `medium`
   profiles. Easy cross-platform install, good quantisation support, excellent model registry.
   On `large`/`server` profiles it is kept for development and testing only.

3. **vLLM (production model server)** — primary task-execution backend for `large` and
   `server` profiles. Continuous batching and paged attention allow multiple concurrent
   agents to be served efficiently. Both CUDA (NVIDIA) and ROCm (AMD) are supported.

Cloud API (OpenAI, Anthropic, etc.) is a fourth optional layer available on all profiles as
fallback or primary when no local capability is present (nano + no in-process dep installed).

**Consequences:**
- `build_chat_client()` gains `"inprocess"` and `"vllm"` as valid `backend` values.
- In-process client (`InProcessChatClient`) uses lazy import of `mistralrs`; raises
  `FeatureDisabledError` if the `[nano]` extra is not installed.
- vLLM client (`VLLMChatClient`) is a thin wrapper over the OpenAI-compatible HTTP API;
  the `vllm` Python package is not required on the client side.
- `BackendManager` probes all three local backends at startup and caches health.
- The routing model key (`routing_model_key` config field) should point to an `inprocess`
  backend model on all profiles to minimise latency and avoid consuming task-execution capacity.

---

## ADR-013: Profile-based feature flags — disabled means truly zero resource cost

**Status:** Accepted
**Date:** 2026-02-26

**Context:** The system must work on everything from a 4 GB RAM nano install to a 64 GB+ server.
Features like vLLM, browser automation, vector embedding, and container isolation must not
consume any RAM, CPU, or disk I/O when not needed. Simply not configuring a feature is
insufficient if the code still imports the module, spawns health-check loops, or holds
background threads.

**Decision:** A `FeatureSet` derived from a `profile` config value (or `auto`-detected) controls
which features are active. Features are gated at four levels simultaneously:

1. **Install-time:** Optional pip extras (`[browser]`, `[nano]`, `[embed]`, `[otel]`) mean the
   dependency is never installed unless requested.
2. **Import-time:** Lazy imports inside factory functions (the import only executes when the
   feature is enabled and the factory function is called).
3. **Config-time:** Objects are never instantiated for disabled features. `BackendManager`
   skips disabled backends entirely.
4. **Process-time:** No background processes are spawned. MCP servers, Playwright browsers,
   and backend health-check loops only start if the feature is enabled.

Profile -> feature defaults:

| Profile | inprocess | ollama | vllm | cloud | mcp | browser | embedding | container | telemetry |
|---------|-----------|--------|------|-------|-----|---------|-----------|-----------|-----------|
| nano    | yes | — | — | yes | — | — | — | — | — |
| small   | yes | yes | — | yes | yes | — | — | — | — |
| medium  | yes | yes | yes | yes | yes | — | yes | — | — |
| large   | yes | yes | yes | yes | yes | — | yes | yes | — |
| server  | yes | yes* | yes | yes | yes | — | yes | yes | yes |

*\* Added in ADR-018: Ollama enabled on server as fallback backend (was previously excluded).*

Individual features can be overridden in `config.yaml` `features:` block regardless of profile.

**Consequences:**
- `config/features.py` is a new module defining `Feature` enum, `PROFILE_FEATURES` mapping,
  `FeatureSet` dataclass, and `FeatureDisabledError`.
- `ConciergeConfig` gains `profile: str` and `features: FeaturesConfig` fields.
- All infrastructure factories accept a `FeatureSet` and call `feature_set.require(Feature.X)`
  before doing any work for feature X.
- Browser (`[browser]` extra) is not enabled by default on any profile in Phase 10; it is
  reserved for Phase 11 when Playwright integration is built.

---

## ADR-014: In-process inference as bootstrap layer and permanent routing brain

**Status:** Superseded by ADR-034
**Date:** 2026-02-26

**Context:** A key design goal is "works on first run without any prior setup". If the only
inference backend is Ollama, the user must install Ollama and pull a model before the system
can do anything. This creates a chicken-and-egg problem: we want an agentic system to guide
setup, but the agentic system needs a model to run. Additionally, routing decisions (which
specialists to recruit) happen on every request and are low-complexity tasks that should not
consume the same capacity as actual task execution.

**Decision:**
- A small quantised model (~1–2 GB) is bundled or auto-downloaded on first run for in-process
  inference via `mistralrs` (PyO3 bindings to mistral.rs, a Rust inference engine supporting
  GGUF models — same format as llama.cpp).
- This in-process model starts in under 1 second and is immediately available before Ollama or
  vLLM are set up. The `FirstRunBootstrap` orchestrator uses it to guide the user through setup.
- On all profiles (not just nano), `routing_model_key` is wired to the in-process backend.
  Routing calls are therefore sub-100ms and consume no Ollama/vLLM capacity.
- On nano profile the in-process model also handles task execution (Ollama is optional).

**Why mistral.rs over llama.cpp:**
- Same GGUF model format; identical model compatibility.
- Pure Rust implementation — consistent with the planned Rust launcher binary (Phase 13).
  One native toolchain rather than mixing C++ (llama.cpp) and Rust.
- PyO3 bindings are maintained alongside the core library.
- Supports CPU, CUDA, ROCm, and Apple Metal via the same interface.

**Consequences:**
- `[nano]` optional extra: `mistralrs>=0.3` (platform-specific wheel: -cpu, -cuda, -metal).
- `InProcessChatClient` is a new `ChatClient` implementation. It is the only client that
  does not require a network connection.
- The `routing_model_key` default changes from `"fast"` to `"routing"` where `"routing"` maps
  to an `inprocess` backend `ModelConfig`.
- On nano profile with no `[nano]` extra installed and no cloud key: `concierge` raises a
  clear error on install pointing to `pip install agentic-concierge[nano]`.

---

## ADR-015: vLLM is a first-class concurrent-agent backend from Phase 10, not deferred

**Status:** Accepted
**Date:** 2026-02-26

**Context:** The earlier plan deferred vLLM to Phase 12. This was reconsidered when analysing
the system's actual concurrent workload. When `task_force_mode: parallel` runs three specialist
agents simultaneously, all three issue LLM requests at the same time. Ollama serves requests
sequentially per model: three simultaneous agents wait 3x the per-request latency. vLLM's
continuous batching serves all three in approximately the time of one request.

**Decision:** vLLM is added as a first-class backend in Phase 10 alongside Ollama:
- `VLLMChatClient` added to `infrastructure/chat/vllm.py`.
- `BackendManager` probes vLLM at startup alongside Ollama and in-process.
- `"vllm"` added as a valid `ModelConfig.backend` value in `build_chat_client()`.
- Profile defaults: medium/large/server profiles use vLLM as primary task-execution backend.
- vLLM supports CUDA (NVIDIA) and ROCm (AMD) — it is not CUDA-only.

**Implementation note:** vLLM exposes an OpenAI-compatible /v1/chat/completions API.
`VLLMChatClient` is a thin wrapper that adds health-checking and model listing on top of
the existing `GenericChatClient` HTTP logic. The `vllm` Python package is NOT required on
the client side — we speak to it over HTTP. The `[vllm]` optional extra is reserved for
future use if we need to manage a vLLM server process programmatically.

**Consequences:**
- Three-way backend selection at startup: in-process -> Ollama -> vLLM (profile-dependent).
- `concierge doctor` shows vLLM health alongside Ollama health.
- Parallel task forces on medium+ profiles are now genuinely concurrent at the LLM layer,
  not serialised through Ollama.

---

## ADR-016: Distribution via Rust thin launcher; Python application core unchanged

**Status:** Accepted
**Date:** 2026-02-26

**Context:** PyPI alone is the wrong distribution channel for the target user base. A user who
"just downloads and runs" needs a single executable that works without Python, pip, or any
package manager. However, replacing Python with Rust for the application code would be
counterproductive: the system is I/O-bound (waiting on LLM inference, network calls, file I/O),
not CPU-bound. Rewriting orchestration, routing, and tool dispatch in Rust would save
microseconds in a system where LLM calls take seconds. The only places where raw compute
matters (token generation) are already handled by Rust internally (mistral.rs, Ollama's
llama.cpp backend, vLLM's CUDA kernels).

**Decision:**
- The Python application (`src/agentic_concierge/`) remains Python — no Rust in orchestration,
  routing, HTTP clients, config, or MCP management.
- A Rust thin launcher binary (~5 MB) is added in Phase 13. It handles:
  - Platform detection and first-run bootstrap
  - Managed Python venv setup (similar to how `uv` and `rye` work)
  - Self-update
  - Launching the Python application via exec
- Distribution channels (Phase 13): GitHub Releases binary, Homebrew tap, one-liner install
  script. PyPI and Docker are kept for developers and operators respectively.
- In-process inference (`mistralrs` PyO3 wheel) is a narrow Rust boundary — a Python-callable
  wheel, not application logic.

**Why Rust for the launcher (not Go or Python/PyInstaller):**
- Static binary: no runtime dependencies, no Python needed on the target machine.
- Cross-compiles to x86_64-linux, aarch64-linux, x86_64-apple-darwin,
  aarch64-apple-darwin (M-series), x86_64-pc-windows-msvc from one CI job.
- Consistent with mistral.rs (Phase 10): single Rust toolchain for all native components.
- PyInstaller produces 150-300 MB bundles with 2-3s startup; Rust binary is ~5 MB, <50ms startup.

**Consequences:**
- Phases 10–12 distributed via PyPI and Docker only (existing channels).
- Phase 13 added the `launcher/` Rust crate to the repo, CI jobs for cross-compilation, and
  `install.sh` one-liner. Binaries attached to GitHub Releases as
  `concierge-x86_64-unknown-linux-musl` and `concierge-aarch64-unknown-linux-musl`.
- `pyproject.toml` and Python packaging are unchanged.
- Developers continue to work with pure Python (`pip install -e ".[dev]"`).
- Module boundaries (`config.rs` / `setup.rs` / `update.rs` / `exec.rs`) are enforced by the
  rule that only `main.rs` may import from other modules. This enables Phase 14+ to replace
  any single module (e.g. swap `setup.rs` for a native Rust Python manager) without touching
  the others.
- Phase 14 replaced the `tar xzf` subprocess in `setup.rs` with pure-Rust `flate2`+`tar`
  crates and added Ed25519 signed binary verification in `update.rs` (see ADR-017).
- Phase 14 also added macOS targets (`x86_64/aarch64-apple-darwin`) to CI and `install.sh`.

---

## ADR-017: Ed25519 signed binary verification for self-update

**Status:** Accepted
**Date:** 2026-02-27

**Context:** The self-update mechanism in `apply_update()` downloads a binary from GitHub
Releases and replaces the running binary with an atomic rename. Without signature verification,
a compromised GitHub account or a network-level MITM attack (even with HTTPS, if a CDN or
cache layer is compromised) could deliver a malicious binary that passes the HTTP check. The
threat is small in practice but binary self-update is one of the highest-impact attack surfaces
in a developer tool.

**Decision:** Every release binary is signed with an Ed25519 keypair during the `release.yml`
CI job.  The public key is embedded at compile time as a 32-byte constant
(`SIGNING_PUBLIC_KEY` in `update.rs`).  Before atomically renaming a downloaded binary into
place, `apply_update()` downloads the corresponding `.sig` file and calls
`verify_binary_signature()`.  An invalid signature causes immediate `Err` return; the
partial-download files are cleaned up; the installed binary is unchanged.

**Key management:**
- **Private key:** stored as CI secret `LAUNCHER_SIGNING_KEY_PEM` (PEM-encoded Ed25519
  private key).  Generated once with `scripts/generate_signing_key.sh`.  Never written to
  disk on a developer machine; injected only into the release runner.
- **Public key:** embedded in compiled binary.  Changing it requires a new release.  Key
  rotation procedure: run `generate_signing_key.sh`, update `SIGNING_PUBLIC_KEY` in
  `update.rs`, update the CI secret, tag a new release.
- **Placeholder:** until the first signed release, `SIGNING_PUBLIC_KEY = [0u8; 32]`.  The
  all-zeros key causes `VerifyingKey::from_bytes` or signature verification to fail, so the
  update is silently skipped.  This is safe: unsigned updates are rejected, not silently
  applied.

**Failure policy:**
- Any verification failure (missing sig file, wrong length, invalid signature) → `Err`.
- `apply_update` caller in `main.rs` catches the error and prints it; the launcher continues
  running with the existing binary.
- The passive hint (non-self-update path) never downloads a binary, so verification never
  runs there.

**Why Ed25519 (not RSA, ECDSA, or a checksum file):**
- Ed25519 is pure-Rust (curve25519-dalek), musl-compatible, no system OpenSSL dependency.
  RSA and ECDSA verification would require linking OpenSSL or a large Rust implementation.
- Detached raw-byte signatures are 64 bytes — trivial to download alongside the binary.
- Ed25519 is the same curve used by SSH and modern Git signing; the OpenSSL CLI used for
  key generation and release signing (`pkeyutl -rawin`) is standard.
- A checksum file (SHA-256) provides tamper detection but not authentication: anyone who
  can replace the binary can also replace the checksum.

**Consequences:**
- New CI secret `LAUNCHER_SIGNING_KEY_PEM` required before a signed release can be made.
- `launcher/Cargo.toml` gains `ed25519-dalek = { version = "2", … }`.
- `verify_binary_signature_with_key` is an inner function accepting a custom key — this
  enables unit tests without depending on the placeholder key.
- Tests: 5 new tests in `update.rs` covering valid sig, tampered binary, wrong key,
  truncated sig, and apply-update blocked on bad sig.
- The signing step in `release.yml` is graceful: if `LAUNCHER_SIGNING_KEY_PEM` is unset
  (e.g. a fork or early release), binaries are published unsigned with a CI warning.

---

## ADR-018: Adaptive backend resolution — fallback chain in `resolve_llm()`

**Status:** Accepted
**Date:** 2026-03-04

**Context:** The system picks a single backend from config, probes it, and hard-fails
if it is unavailable. A SERVER-profile machine with 94 GB RAM but no vLLM running gets
`RuntimeError` and exits — even if Ollama is installed and could serve the request.
The profile system disabled backends (SERVER excluded Ollama) rather than expressing
preference order. The system should always be able to run *something*.

**Decision:** Refactor `resolve_llm()` to try backends in priority order, with the
configured primary backend tried first (preserving backward compatibility). On failure,
iterate through `BACKEND_PRIORITY[tier]` — a per-profile ordered list of backend names
— filtered by enabled features, skipping the already-tried primary.

The priority order per profile tier:
- NANO: `["ollama", "llama_cpp", "cloud"]`
- SMALL: `["ollama", "llama_cpp", "cloud"]`
- MEDIUM: `["ollama", "llama_cpp", "cloud"]`
- LARGE: `["ollama", "llama_cpp", "vllm", "cloud"]`
- SERVER: `["vllm", "ollama", "llama_cpp", "cloud"]`

*(Updated by ADR-034: in-process removed, llama_cpp added to all tiers.)*

Each backend is probed with a dedicated `_try_<backend>()` helper:
- `_try_ollama()`: discovers models; if unreachable and `shutil.which("ollama")` finds
  the binary, auto-starts via `ensure_llm_available()` with a 30 s timeout.
- `_try_vllm()`: probes the OpenAI-compatible `/v1/models` endpoint.
- `_try_cloud()`: scans `config.models` for a `backend="generic"` entry with a non-empty
  `api_key`.

**Changes to ResolvedLLM:**
- `warnings: list[str]` — human-readable warnings (e.g. "Primary backend failed; using
  fallback: ollama"). Displayed in CLI and included in HTTP API `_meta`.
- `fallback_used: bool` — `True` when resolution used a non-primary backend.
- `resolved_backend: str` — the backend name that was actually used.

All fields have defaults, so existing callers are unaffected.

**Changes to SERVER profile:** `Feature.OLLAMA` added to
`PROFILE_FEATURES[ProfileTier.SERVER]` so Ollama is eligible as a fallback. Users can
still force-disable it with `features.ollama: false`. This supersedes the comment in
ADR-013's table that said "server drops Ollama".

**Alternatives considered:**
- Moving the fallback logic to `execute_task` or a new orchestration layer: rejected
  because `resolve_llm()` already encapsulates all discovery logic and its return type
  (`ResolvedLLM`) flows directly into `build_chat_client()`. Keeping the change inside
  `resolve_llm()` means zero changes to callers, `execute_task`, `ChatClient` protocol,
  or specialist packs.
- A separate `BackendResolver` class: unnecessary abstraction at this stage. The
  `_try_backend()` dispatch function and per-backend helpers are sufficient.

**Consequences:**
- The system no longer hard-fails when the configured backend is down — it degrades
  gracefully through the fallback chain.
- Warnings are surfaced to the user (CLI terminal, HTTP `_meta`) so they know a fallback
  is in use and can fix the primary backend.
- `BACKEND_PRIORITY` is a new constant in `config/features.py` that must be maintained
  alongside `PROFILE_FEATURES` when new profiles or backends are added.
- *(Note: `_ensure_nano_model()` and `_try_inprocess()` were removed by ADR-034.)*
- 11 new tests cover fallback scenarios, feature gating, auto-start, comprehensive error
  messages, and `_try_backend` unit tests.

---

## ADR-019: Universal work review mechanism (Gate 4 on `finish_task`)

**Status:** Accepted
**Date:** 2026-03-05

**Context:** The system has quality gates (required fields, `tests_verified`,
`validate_finish_payload`) but no independent verification of a specialist's claims.
A specialist can assert "tests pass" or "feature complete" without any second opinion.
Human code review exists because developers are unreliable self-assessors; the same
principle applies to LLM agents. We need a built-in review/critique phase that runs
automatically for every specialist execution.

**Decision:** After the doer calls `finish_task` and passes Gates 1–3 (prior tool call,
required fields, quality gate), a new **Gate 4** triggers an independent reviewer LLM
call. The reviewer inspects the workspace and finish payload using read-only tools, then
either approves (`approve_work`) or rejects with actionable critique (`request_revision`).
If rejected, the critique is returned to the doer as a `finish_task` error, and the doer
loop continues.

**Reviewer design:**
- `_review_specialist_work()` is a lightweight async function (max 5 steps), not a full
  `_execute_pack_loop`. It builds a reviewer system prompt + user message containing the
  finish payload, gives the reviewer read-only tools from the doer's pack plus
  `approve_work` / `request_revision`, and runs a simple tool-calling mini-loop.
- **Fail-open:** Plain text from reviewer = implicit approval. Reviewer errors = implicit
  approval. Max steps without a decision tool call = implicit approval. This prevents
  reviewer failures from blocking work.
- **Same model:** The reviewer uses the same `model_cfg` as the doer (no separate model
  selection). This keeps things simple and avoids model-selection complexity.
- **Safe tool set:** `_SAFE_REVIEWER_TOOLS = {"read_file", "list_files", "run_tests"}`.
  Explicitly excluded: `shell`, `write_file`, `web_search`, `fetch_url`,
  `cross_run_search`. The reviewer reads and verifies; it does not modify.

**Max review iterations:** `_MAX_REVIEW_ITERATIONS = 2`. After 2 rejections, the work
is accepted with a `_review_warning` annotation in the payload. This prevents infinite
doer↔reviewer loops. Exposed as `max_review_iterations` parameter on `execute_task()`,
`resume_execute_task()`, and `_execute_pack_loop()` (threaded via `LeafExecutionContext` in V2).

**Test strategy:** `execute_task()` gets `max_review_iterations: int = 2` (enabled by
default). Test helpers (`_run()`, `_run_task_force()`, etc.) pass
`max_review_iterations=0` so existing tests work unchanged. Dedicated review tests in
`tests/test_review.py` pass `max_review_iterations=2` explicitly.

**Runlog events:**
- `review_start` — emitted when Gate 4 begins; payload includes `specialist_id` and
  `review_iteration`.
- `review_approved` — reviewer approves; payload includes `comment` and `review_iteration`.
- `review_rejected` — reviewer rejects; payload includes `critique` and `review_iteration`.

**Alternatives considered:**
- Separate reviewer specialist pack: rejected as over-engineered. The reviewer is a
  lightweight mini-loop, not a full pack with its own lifecycle. It reuses the doer's
  pack for tool execution (`pack.execute_tool()` for read-only tools).
- Reviewer uses a different (smaller/cheaper) model: deferred. Using the same model is
  simpler and ensures the reviewer can understand the doer's work. A future ADR may
  revisit this when adaptive model selection is implemented.
- Review as a separate template/pack type: rejected. Review is inherent to how every
  specialist execution works, not a specialist-specific concern.

**Consequences:**
- Every specialist execution now has an independent verification step (when enabled).
- The reviewer tool definitions (`_APPROVE_WORK_TOOL_DEF`, `_REQUEST_REVISION_TOOL_DEF`)
  and `_SAFE_REVIEWER_TOOLS` are defined in `execute_task.py` alongside the synthesis
  tool, not in `tool_catalog.py` — they are execute-task-specific.
- `PROMPT_REVIEWER` is a self-contained fragment in `prompts.py` (not assembled via
  `generate_system_prompt()`), following the same pattern as the synthesis agent prompt.
- 8 new tests in `tests/test_review.py`; 10+ existing test files updated to pass
  `max_review_iterations=0`. Fast CI: 655 pass (+5).

---

### ADR-020: Adaptive Escalation — Dynamic Model Right-Sizing

**Date:** 2026-03-05

**Status:** Accepted

**Context:** The system currently starts each specialist on the configured model and has
only one direction of model switching: *downward* on timeout (`pick_smaller_model()` in
`llm_discovery.py`). When a small model produces low-quality work — repeated plain-text
responses, tool-call loops, quality gate failures, or review rejections — the loop either
retries with the same model (wasting steps) or gives up. The BACKLOG spec describes a
bidirectional convergence: quality issues → go bigger, timeouts → go smaller. Gate 4
(ADR-019) now provides reliable quality signals; it's time to act on them.

**Decision:** Implement adaptive escalation as a per-pack-loop concern. When quality
failure signals accumulate, the loop swaps to a larger available model and continues the
conversation (preserving message history). This is the upward counterpart to the existing
timeout-based downward switching.

**Escalation triggers (quality failure signals):**
1. **Plain-text exhaustion** — After `_MAX_PLAIN_TEXT_RETRIES` (2) corrective re-prompts
   fail to elicit tool calls, escalate instead of treating text as final payload.
2. **Loop detection** — After `[SYSTEM] LOOP DETECTED` is injected and the loop persists
   for one more iteration, escalate instead of continuing the stall.
3. **Review rejection at max** — After `_MAX_REVIEW_ITERATIONS` (2) rejections, escalate
   instead of accepting with `_review_warning`. If escalation succeeds, reset the review
   iteration counter (the new model gets fresh review attempts).

Each trigger escalates at most once per signal type per pack loop. This prevents cascading
escalations from a single persistent problem.

**Non-triggers (handled by existing mechanisms):**
- Timeout → existing `pick_smaller_model()` downward switching (unchanged).
- Gate 1/2 failures → corrective error messages to the LLM (unchanged).
- Gate 3 failures → existing `validate_finish_payload()` error messages (unchanged).

**Escalation mechanics:**
- `pick_larger_model(available_models, current_model)` in `llm_discovery.py`: returns the
  smallest model larger than the current one (same-family preference, matching the existing
  `pick_smaller_model()` pattern). Returns `None` if no larger model is available.
- On escalation: rebind `model_cfg` to a new `ModelConfig` and rebuild `chat_client` via
  `_rebuild_chat_client()` which preserves `FallbackChatClient` wrapping if present.
- **Continue message history** — do not reset. The larger model benefits from seeing the
  conversation context including prior failures. This also preserves any partial work the
  smaller model accomplished.
- Emit `model_escalated` runlog event with payload: `{trigger, from_model, to_model,
  escalation_count}`.

**Escalation state:**
- Lives per-pack loop (reset for each specialist in a task force).
- `escalation_count: int` — total escalations in this loop (across all trigger types).
- `escalated_triggers: set[str]` — which trigger types have already caused escalation
  (each type fires at most once).
- `_MAX_ESCALATIONS = 2` — hard cap on total escalations per pack loop, regardless of
  trigger type. After this cap, fall through to existing behaviour (accept with warning,
  treat text as payload, etc.).

**Reviewer model:**
- The reviewer always uses the same model as the doer (matching ADR-019). When the doer
  model is escalated, the reviewer automatically gets the escalated model since they share
  `model_cfg`.

**Config:**
- `_MAX_ESCALATIONS = 2` constant in `execute_task.py`. Exposed as `max_escalations`
  parameter on `execute_task()`, `resume_execute_task()`, and `_execute_pack_loop()`
  (threaded via `LeafExecutionContext` in V2).
- `max_escalations=0` disables escalation entirely (for tests and opt-out).
- Not yet exposed on `ConciergeConfig` (deferred until we need per-config tuning).

**Test strategy:**
- Existing tests are unaffected: escalation only fires when `available_models` is populated
  AND a quality failure occurs; test mocks typically have no available_models.
- Dedicated escalation tests in `tests/test_escalation.py`:
  - Plain-text exhaustion triggers escalation to larger model.
  - Loop detection triggers escalation after warning fails.
  - Review rejection at max triggers escalation and resets review counter.
  - Escalation cap (`_MAX_ESCALATIONS=2`) prevents runaway escalation.
  - No larger model available → fall through to existing behaviour.
  - Each trigger type fires at most once.
  - `model_escalated` runlog event is emitted with correct payload.

**Scope boundaries (explicitly excluded from this ADR):**
- Actions 3 and 4 from the BACKLOG spec (re-recruit specialist, modify tool/role mix) are
  deferred. They require orchestrator changes and are a separate concern. This ADR covers
  only model-level escalation (BACKLOG actions 1 and 2).
- Starting on the *smallest* model by default (the "start small" half of adaptive
  escalation) is deferred. Currently the system starts on the configured model. A future
  ADR may address automatic smallest-model-first selection.

**Alternatives considered:**
- Reset message history on escalation: rejected. Partial work and failure context help the
  larger model understand what went wrong and what's already been attempted.
- Escalation state shared across task force: rejected. Each specialist's quality issues are
  independent. A research pack struggling doesn't mean the engineering pack needs a bigger
  model.
- Separate escalation policy config (per-signal thresholds, etc.): rejected as
  over-engineered. A single `max_escalations` cap with per-trigger-type dedup is sufficient.
  Can be refined later if needed.
- Escalate on Gate 3 (quality gate) failures: deferred. Gate 3 failures are pack-specific
  validation (e.g. `tests_verified=False`) and the corrective error message is usually
  sufficient for the LLM to self-correct. If this proves insufficient in practice, Gate 3
  can be added as a trigger in a future iteration.

**Consequences:**
- Every specialist execution can now converge on the right model size: timeouts push down,
  quality issues push up.
- `pick_larger_model()` is added to `llm_discovery.py` alongside `pick_smaller_model()`.
- `_execute_pack_loop` gains escalation state tracking and three escalation insertion points.
- New `model_escalated` runlog event for observability.
- Tests must pass `max_escalations=0` to avoid escalation side effects (same pattern as
  review).


### ADR-021: Human Approval — Interactive Pause-and-Wait via `request_approval` Tool

**Date:** 2026-03-05
**Status:** Accepted

**Context:**
`require_human_approval_for` exists in config and `PROMPT_CORE_RULES` tells the LLM to
include an approval note in `finish_task`, but this is prompt-only enforcement — the LLM
can ignore it, and even when it complies the human must manually run the command after the
task finishes. There is no mechanical pause-and-wait in the tool loop.

**Decision:**
Add a `request_approval` tool handled inline in `_execute_pack_loop` (like `finish_task`).
When the LLM calls it, the loop blocks on an `ApprovalChannel` protocol until the human
responds. The tool result tells the LLM whether the action was approved or denied.

**Design:**
- `ApprovalChannel` protocol in `application/approval.py` with dataclasses
  `ApprovalRequest` and `ApprovalDecision`.
- Three implementations: `AutoApprovalChannel` (always approves; for tests),
  `CliApprovalChannel` (stdin via `asyncio.to_thread`), `HttpApprovalChannel`
  (SSE event + REST callback + `asyncio.Event`).
- `_REQUEST_APPROVAL_TOOL_DEF` constant in `execute_task.py`; appended to effective
  tool defs when `approval_channel` is not `None`.
- Timeout: fail-closed (deny). Default 600s, configurable via `approval_timeout_s`
  on `ConciergeConfig`.
- Runlog events: `approval_requested`, `approval_granted`, `approval_denied`.

**Backward compatibility:**
- `approval_channel` defaults to `None`. When `None`, the tool is not in definitions.
- If the LLM somehow calls `request_approval` without a channel, it auto-approves.
- All existing tests pass unchanged.

**Alternatives considered:**
- Intercept tool calls matching config's `require_human_approval_for` list: rejected.
  Fragile pattern matching; LLM names vary; can't distinguish "deploy to staging" from
  "deploy to prod".
- Add approval as a `finish_task` field: rejected. Conflates completion with approval;
  the LLM may need to request approval mid-task (not just at the end).

**Consequences:**
- The tool loop can now block on human input within a session.
- CLI gains `--auto-approve` flag.
- HTTP API gains `POST /runs/{run_id}/approve` endpoint.
- `_execute_pack_loop` gains `approval_channel` parameter (threaded through all callers).


### ADR-022: Agent-to-Agent Delegation via `delegate_to_specialist` Tool

**Date:** 2026-03-05
**Status:** Accepted

**Context:**
The orchestrator plans the full task force upfront. Once a specialist is running, it cannot
recruit a sub-specialist. If an engineering agent needs research, it must attempt it with its
own tools or punt to `finish_task` notes.

**Decision:**
Add a `delegate_to_specialist` tool handled inline in `_execute_pack_loop`. When the LLM
calls it, the loop creates a sub-specialist via the registry and runs a nested
`_execute_pack_loop()`. The sub-specialist's finish payload becomes the tool result.

**Design:**
- `_DELEGATE_TOOL_DEF` constant; appended to tool defs when `delegation_depth > 0`.
- `delegation_depth` parameter on `_execute_pack_loop`; defaults to 0. At depth 0, the
  tool is not included.
- Sub-specialist gets `min(remaining_steps // 2, _MAX_DELEGATION_STEPS)` with floor of 3.
- `_MAX_DELEGATION_DEPTH = 1`: sub-specialists cannot delegate further.
- `_MAX_DELEGATION_STEPS = 15`: independent cap on sub-specialist steps.
- Same workspace, same run ID, same chat client and model config.
- Sub-specialist gets only the sub_task prompt (clean context).
- Inherited settings: `max_review_iterations`, `approval_channel`; escalation state fresh.
- Runlog events: `delegation_start`, `delegation_complete`.

**Backward compatibility:**
- `delegation_depth` defaults to 0; tool not included in defs; existing tests unchanged.
- `specialist_registry` and `workspace_path` params default to `None` on
  `_execute_pack_loop`; delegation is only attempted when both are provided.

**Alternatives considered:**
- Allow unlimited recursion: rejected. Stack overflow risk; step budget explosion;
  hard to debug.
- Delegation via orchestrator re-planning: rejected. Orchestrator is a task-level
  concern; delegation is a tool-level concern within a running specialist.

**Consequences:**
- Specialists can now compose: an engineering agent can delegate research mid-task.
- `_execute_pack_loop` gains `delegation_depth`, `specialist_registry`, `workspace_path`,
  and `config` parameters.
- Step key prefixing enables attribution in runlog (`s5.d1.s0`).
- Max depth = 1 keeps the system predictable and debuggable.


### ADR-023: Model Capability Registry — Scored Profiles per Model Family

**Date:** 2026-03-08
**Status:** Accepted

**Context:**
`_TOOL_INCAPABLE_NAMES` is a static blocklist. `select_model()` picks by size only. There
is no way to say "this task needs code generation, prefer qwen2.5-coder:7b over llama3.1:7b."
Non-tool-calling models (sqlcoder, deepseek-coder, codellama) are blocked entirely even though
they could be valuable for specific sub-tasks.

**Decision:**
Create `infrastructure/model_profiles.py` with `ModelCapabilityProfile` dataclass. Each model
family gets scored capabilities (0.0–1.0) and a `supports_tool_calling` bool. Profiles are
loaded from a bundled dict, extensible via config. `_TOOL_INCAPABLE_NAMES` is replaced by
`get_profile(name).supports_tool_calling`.

**Design:**
- `ModelCapabilityProfile`: `family`, `supports_tool_calling`, `capabilities: dict[str, float]`.
- `BUILTIN_PROFILES`: bundled profiles for qwen2.5, qwen2.5-coder, llama3.1, phi-4-mini,
  deepseek-r1-distill, sqlcoder, deepseek-coder, codellama, gemma2.
- `get_profile(model_name)`: fuzzy-match family from model name; unknown → permissive default.
- `match_models()`: filter available models by capability thresholds, return best match.
- `infer_task_capabilities()`: map template ID or tool names → required capabilities.

**Backward compatibility:**
- `_is_tool_capable()` produces identical results for all currently-blocked families.
- Unknown model families get a permissive default (tool_calling=true, all caps=0.5).

**Consequences:**
- Model selection can now be capability-driven rather than size-driven.
- Non-tool-calling models are still excluded from the tool loop but can be used via consult.
- New capabilities and families can be added without code changes (config override).


### ADR-024: Per-Specialist Model Selection — Task-Capability Matching

**Date:** 2026-03-08
**Status:** Accepted

**Context:**
All specialists use the same globally-resolved model. If qwen2.5-coder:7b is pulled alongside
qwen2.5:7b, the engineering specialist should prefer the coder variant. The system has no
mechanism for per-specialist model selection.

**Decision:**
Each specialist assignment gets its own model selected by matching task capabilities against
model profiles. `_select_specialist_model()` in `execute_task.py` infers capabilities from the
specialist template and/or the orchestrator's `required_capabilities`, then calls `match_models()`
to find the best-scoring available model.

**Design:**
- `_select_specialist_model(assignment, available_models, base_model_cfg, chat_client)`:
  returns `(ModelConfig, ChatClient)`, potentially switching the model.
- When `assignment.required_capabilities` is set (from ADR-028), it overrides template inference.
- When only one model is available or the base model is already best, returns originals (no-op).
- Available as a utility; V2 path uses `affinity_executor` for per-node model assignment.

**Backward compatibility:**
- Single-model setups (common case): returns the same model unchanged.
- No changes to `_execute_pack_loop` signature.

**Consequences:**
- Engineering tasks prefer coder models; research tasks prefer generalist models.
- Model switching happens before the pack loop, not during.
- `_rebuild_chat_client()` constructs a new client for the selected model.


### ADR-025: Non-Tool-Calling Execution Path (`consult_specialist_model`)

**Date:** 2026-03-08
**Status:** Accepted

**Context:**
Models like sqlcoder, deepseek-coder, codellama are blocked because they can't do OpenAI tool
calling. But they could be valuable for specific sub-tasks (SQL generation, code completion).

**Decision:**
A `consult_specialist_model` tool in the tool catalog that lets a tool-calling agent consult a
non-tool-calling model for a specific sub-task. The tool executor does a single chat completion
(no tool loop) and returns the response as the tool result.

**Design:**
- `infrastructure/tools/consult.py`: `execute_consult()` function + `_consult_specialist_executor()`
  closure factory that captures `all_chat_models` and `base_url`.
- Tool definition: `specialty` (enum: code, sql, reasoning) + `prompt` string.
- `SPECIALTY_CAPABILITIES` maps specialties to capability requirements with
  `require_tool_calling=False`.
- `match_models()` called with `require_tool_calling=False` to find the best specialist.
- `ConfigSpecialistRegistry.set_runtime_models()` injects `all_chat_models` once after LLM
  discovery. `_needs_consult_tool()` checks if any model lacks tool-calling support.
- `_maybe_add_consult()` automatically appends `consult_specialist_model` to pack tool lists
  when non-TC models are detected.

**Backward compatibility:**
- Tool only included in packs when non-tool-calling models are discovered.
- When all models support tool calling, no consult tool is injected.
- `set_runtime_models()` added to `SpecialistRegistry` protocol; callers updated.

**Consequences:**
- Non-tool-calling models are now accessible as domain experts via the consult tool.
- Single-shot call (no loop) keeps latency bounded.
- Infrastructure concerns (base_url, model list) stay in `ConfigSpecialistRegistry`,
  not in the application layer protocol.


### ADR-026: Adaptive Finish Schemas — Task-Appropriate Output Structure

**Date:** 2026-03-08
**Status:** Accepted

**Context:**
The research template forces `_RESEARCH_FINISH_SCHEMA` with `executive_summary`, `citations`,
`bibliography_path`, etc. When someone asks "How much does Amex Platinum cost?", the LLM wastes
steps creating bibliography files and fabricates citation metadata to satisfy the schema.

**Decision:**
Replace per-template finish schemas with a schema selector. `finish_schemas.py` defines four
schema shapes: `quick_answer`, `research_report`, `code`, `general`. The orchestrator can
specify `finish_schema` per specialist assignment, overriding the template default.

**Design:**
- `infrastructure/specialists/finish_schemas.py`: `FINISH_SCHEMAS` dict, `get_finish_schema()`.
- `SpecialistBrief.finish_schema: Optional[str]` — when set, overrides template default.
- `build_dynamic_pack()` and `build_template_pack()` accept `finish_schema_key` param.
- `create_plan` tool schema includes `finish_schema` enum field.
- Existing schemas moved from `dynamic_pack.py` to `finish_schemas.py`.

**Backward compatibility:**
- When `finish_schema` is None (default), the template's built-in schema is used — identical
  to current behavior.
- Existing test packs that pass `finish_schema=None` see no change.

**Consequences:**
- Simple factual questions can use `quick_answer` (no artifacts, no bibliography).
- LLM orchestrator makes the schema decision, not the template.
- Schema shapes are extensible without modifying template code.


### ADR-027: Independent Reviewer Model — Cross-Model Quality Gate

**Date:** 2026-03-08
**Status:** Accepted

**Context:**
Gate 4 (reviewer) in `_execute_pack_loop` uses the same model as the doer. A 7b model
rubber-stamps its own 7b output. The reviewer should use a different model, preferably one
with strong reasoning capabilities.

**Decision:**
The reviewer model is selected independently using the Model Capability Registry. It prefers
reasoning-focused models (phi-4-mini, deepseek-r1-distill) and excludes the doer's model.
`_review_specialist_work()` accepts `reviewer_model_cfg` and builds a separate chat client.

**Design:**
- `_select_reviewer_model()` in `execute_task.py`: calls `match_models()` with
  `required_capabilities={"reasoning": 0.7, "instruction_following": 0.6}`,
  excluding the current model.
- `_review_specialist_work()` gains `reviewer_model_cfg` param.
- Fallback: when no alternative model is available, uses same model (identical to today).

**Backward compatibility:**
- `reviewer_model_cfg=None` → identical behavior to today.
- Only activates when multiple models are available.
- Existing `max_review_iterations=0` tests unchanged.

**Consequences:**
- Review quality improves when multiple models are available.
- No behavioral change on single-model setups.


### ADR-028: Capability-Driven Orchestrator — Route by What, Not by Name

**Date:** 2026-03-08 (revised 2026-03-09)
**Status:** Accepted

**Context:**
The orchestrator routes by template name. Its system prompt says "Available specialist templates:
engineering, research." This forces the LLM to think in template terms rather than capability
terms. Adding a new specialist type requires prompt changes.

**Decision:**
Capabilities compose tools directly. The routing LLM specifies `required_capabilities`; the
system composes a tool set from the union of tools needed for all requested capabilities, then
either matches a template (if the composed set is identical) or builds a dynamic pack.

Template names are internal implementation details — never exposed to the routing LLM.

**Design:**
- `_CAPABILITY_TOOLS` in model_profiles.py: maps each capability to tools it requires
  (e.g. `web_comprehension → [web_search, fetch_url]`, `code_python → [shell, write_file, run_tests]`).
  Model-only capabilities (`reasoning`, `summarisation`, `instruction_following`) map to `[]`.
- `_BASE_TOOLS = [read_file, list_files]`: every specialist gets these.
- `compose_tools_from_capabilities()`: union of capability tools + base tools.
- `_resolve_pack_from_capabilities()`: composes tools, checks template match, returns
  `(specialist_id, tools, role, finish_schema_key)`. Template match → template ID + None fields.
  No match → `"dynamic"` + composed tools + generated role + inferred schema.
- `generate_role_from_capabilities()`: composes mission fragments per capability.
- `infer_finish_schema_from_capabilities()`: code-only→"code", web-only→"quick_answer", mixed→None.
- `create_plan` schema: `required_capabilities` is **required**, `specialist_id` removed from schema.
- `_dedup_same_id()`: dynamic packs with different tool sets are not merged (fixes prior bug).

**Examples:**
- `["web_comprehension"]` → `{web_search, fetch_url, read_file, list_files}` = research template
- `["code_python"]` → `{shell, write_file, read_file, list_files, run_tests}` = engineering template
- `["code_python", "web_comprehension"]` → `{shell, write_file, run_tests, web_search, fetch_url, read_file, list_files}` → no template match → dynamic pack with composed tools
- `["reasoning"]` → `{read_file, list_files}` (degenerate) → research fallback

**Backward compatibility:**
- `specialist_id` still accepted in LLM output as override (legacy/tests).
- Templates exist as internal implementation — tuned role descriptions and finish schemas.
- Downstream code (execute_task.py, registry.py) unchanged — `SpecialistBrief.specialist_id`
  is populated by the system, consumed identically.
- Enterprise research is config-driven (cross_run_search has no capability mapping).

**Consequences:**
- The orchestrator LLM describes tasks in capability terms, not template terms.
- Mixed-capability tasks (code + web) automatically get a dynamic pack with all needed tools.
- New capabilities can be added to `_CAPABILITY_TOOLS` without touching the orchestrator prompt.
- Templates are just pre-tuned shortcuts for common capability combinations.

---

## V2 Architecture Decisions

The following ADRs document the V2 three-layer architecture redesign.
See [DESIGN_V2.md](DESIGN_V2.md) for the full design document.

---

### ADR-029: Model Runtime — Acquire/Release with Lifecycle Management

**Status:** Accepted
**Date:** 2026-03-09

**Context:** V1 used a single `base_url` for all specialists — whatever model happened to be
loaded in Ollama was used for everything. There was no way to load/unload models dynamically,
track memory usage, or assign different models to different tasks. The system also couldn't
use llama.cpp alongside Ollama.

**Decision:** Introduce a `ModelRuntime` protocol in `application/ports.py` with:
- `acquire(requirements: Dict[str, float]) → ModelHandle` — select and load the best model
- `release(handle: ModelHandle)` — decrement refcount, eligible for eviction
- `preload_hint(requirements)` — background load for upcoming work
- `status() → RuntimeStatus` — loaded models, memory usage

The concrete implementation `LocalModelRuntime` in `infrastructure/backends/model_runtime.py`
manages model lifecycle with reference counting, LRU eviction, and memory budget tracking.

`BackendRegistry` discovers and monitors inference backends (Ollama, LlamaCpp) with health
checks and priority-based failover. Configuration loaded from `config/defaults/backends.yaml`.

`InferenceBackend` protocol defines the per-backend contract: `load_model()`, `unload_model()`,
`build_client()`, `estimate_memory()`, `list_available()`, `health_check()`.

**Consequences:**
- Different tasks can use different models concurrently (within memory budget).
- Models are loaded lazily and evicted LRU when memory is tight.
- New backends (MLX for macOS) can be added by implementing `InferenceBackend`.
- `CapabilityProbe` validates model capabilities via micro-prompts before trusting model profiles.

---

### ADR-030: Recursive Task Decomposition — TaskGraph with Planner + Critic

**Status:** Accepted
**Date:** 2026-03-09

**Context:** V1's `OrchestrationPlan` produced a flat list of specialist assignments in one
LLM call. Complex tasks like "build a CRUD app with tests" need recursive decomposition:
plan → implement → test → fix failures, where each step may itself decompose further.

**Decision:** Replace flat `OrchestrationPlan` with a recursive `TaskGraph` (DAG of `TaskNode`
objects). Each node has a state machine: `pending → decomposing → critiqued → executing →
reviewing → done/failed`.

The decomposition uses two agents:
- **Planner**: LLM decomposes a task into subtasks with `required_capabilities`, `required_tools`,
  and `finish_schema_key`.
- **Critic**: Independent LLM reviews the plan and approves or rejects with feedback
  (max 2 re-plans, fail-open).

`execute_graph()` is the core executor — finds ready leaf nodes, executes them in parallel
via `asyncio.gather`, marks done/failed, propagates completion upward.

`should_decompose()` implements adaptive depth control: stops when a leaf fits the available
model or max depth (3) is reached.

**Consequences:**
- Complex tasks are decomposed recursively instead of once.
- Failed subtasks block dependent siblings (only `done` unblocks).
- The planner and critic can use different models (via `must_differ_from` in agent roles).
- The graph executor is agnostic to leaf execution — any async callback works.

---

### ADR-031: Agent-Model Affinity — Right Model for Right Task

**Status:** Accepted
**Date:** 2026-03-09

**Context:** V1 used the same model for every role (routing, planning, coding, reviewing).
A 7B generalist model fails at each specialised task. The reviewer rubber-stamps its own work
because it's the same model. Different agent roles need different model capabilities.

**Decision:** Define six agent roles with explicit capability requirements:

| Role | Key requirements | Constraints |
|------|-----------------|-------------|
| router | structured_output (0.8), instruction_following (0.7) | prefer_small |
| planner | reasoning (0.8), structured_output (0.7), instruction_following (0.8) | — |
| critic | reasoning (0.8), instruction_following (0.7) | must_differ_from planner |
| coder | code_python (0.8), instruction_following (0.7) | — |
| researcher | web_comprehension (0.7), summarisation (0.7) | — |
| reviewer | reasoning (0.8), instruction_following (0.8) | must_differ_from coder, researcher |

`assign_model()` acquires the best model from `ModelRuntime` using role requirements +
task capabilities. `must_differ_from` is fail-open: when no alternative model is available,
the constraint is relaxed.

`execute_graph_with_affinity()` wraps the graph executor: for each leaf node, determines
the agent role → assigns model → executes work → releases handle in `finally`.

Preload hints are issued for upcoming sibling nodes via `asyncio.ensure_future`, so models
are loaded in the background while the current step executes.

**Consequences:**
- Different tasks get different models based on capability matching.
- The reviewer always attempts to use a different model than the doer.
- Model handles are guaranteed to be released (via `finally`), preventing memory leaks.
- Preload hints reduce cold-start latency between sequential graph nodes.

---

## ADR-032: V2 graph checkpointing and resume

**Status:** Accepted
**Date:** 2026-03-10

**Context:**
After removing V1 checkpoint/resume (`RunCheckpoint`, `save_checkpoint`, `resume_execute_task`),
the system lost the ability to resume interrupted runs. The V2 architecture uses a `TaskGraph`
DAG instead of flat `OrchestrationPlan`, so checkpointing needs to serialize the entire graph
state including per-node status, results, and the plan structure.

**Decision:**
Implement V2-native graph checkpointing in two layers:

1. **Application layer** (`application/graph_checkpoint.py`):
   - `serialize_graph()` / `deserialize_graph()`: Pure-data TaskGraph ↔ dict round-trip.
   - `prepare_graph_for_resume()`: Resets in-flight nodes (`executing`, `reviewing`,
     `decomposing`) back to `pending`; collects results from `done` nodes into
     `prior_results` dict. This is an administrative bypass of the state machine.
   - Schema version (`SCHEMA_VERSION = 1`) for forward compatibility.

2. **Infrastructure layer** (`infrastructure/workspace/graph_checkpoint.py`):
   - `GraphCheckpoint` dataclass: `run_id`, `prompt`, `model_name`, `specialist_ids`,
     `graph_data` (serialized TaskGraph), `completed_node_results`, timestamps.
   - Atomic file I/O: write to temp file + `os.replace` to prevent corruption.
   - `save_checkpoint()`, `load_checkpoint()`, `delete_checkpoint()`.
   - `find_resumable_runs()`: Scans workspace for runs with checkpoints where root != done.

3. **Graph executor** (`graph_executor.py`, `affinity_executor.py`):
   - New `prior_results` parameter on `execute_graph()` and `execute_graph_with_affinity()`.
   - Pre-populates the `results` dict so completed nodes appear in the output without
     re-execution.

4. **Execution wiring** (`execute_task.py`):
   - `_create_graph_checkpoint()`: Called after planning, before execution.
   - `_update_graph_checkpoint()`: Called on each `node_done` / `node_failed` event.
   - `_delete_graph_checkpoint()`: Called after successful completion.
   - `resume_execute_task()`: Loads checkpoint, calls `prepare_graph_for_resume()`,
     creates `LeafExecutionContext` with prior results, calls `execute_graph_with_affinity()`
     with `prior_results`.

5. **Interfaces**:
   - CLI: `concierge resume <run_id>` command; `logs list` shows resumable markers.
   - HTTP: `POST /runs/{run_id}/resume`, `GET /runs/resumable`.

Checkpoint failures are fail-open (logged, not raised) — they never crash the main execution.

**Consequences:**
- Interrupted multi-step tasks can be resumed, skipping already-completed nodes.
- The checkpoint file (`graph_checkpoint.json`) is atomically written to prevent corruption.
- 82 new tests across 4 test files cover serialization, I/O, resume logic, and CLI/HTTP wiring.
- Backward compatible: existing runs without checkpoints work unchanged.

---

## ADR-033: Maximal-default install and concurrent backend utilisation

**Status:** Implemented
**Date:** 2026-03-10

**Context:**
Real-world usage revealed three friction points:
1. Default install (`CONCIERGE_EXTRA` unset) gives a bare-bones package — no MCP, no browser, no
   embeddings. Users must discover and set `CONCIERGE_EXTRA=all` manually.
2. Independent task graph nodes execute in parallel (`asyncio.gather`), but Ollama serialises
   model inference on a single GPU — negating the parallelism benefit.
3. vLLM supports continuous batching (true concurrent requests) but requires manual startup.
   There is no auto-start equivalent to `ensure_ollama()`.

**Principles:**
- The default install should detect hardware and install everything it can use. Trimming down
  is the customisation, not the default.
- If resources are available, use them — parallel inference backends should be preferred over
  serial ones when the task graph has concurrent nodes.

**Decision:**
1. **Maximal-default extras** (implemented): Launcher defaults `pypi_extra` to `"all"` when
   `CONCIERGE_EXTRA` is unset. Users opt out with `CONCIERGE_EXTRA=""` for bare-bones.
2. **vLLM auto-start** (implemented): `ensure_vllm()` added to `BackendManager`, mirroring
   `ensure_ollama()`. On SERVER profiles with GPU, auto-starts vLLM with the best available
   model. Config fields: `vllm_ensure_available`, `vllm_start_cmd`, `vllm_model`,
   `vllm_start_timeout_s`, `vllm_gpu_memory_utilization`.
3. **Backend preference for parallelism** (implemented): `supports_concurrent` property on
   `InferenceBackend` protocol. `prefer_concurrent` parameter threaded from affinity executor
   (detects >1 executing nodes) through `assign_model()` to `runtime.acquire()`.
   `LocalModelRuntime._acquire_locked()` filters to concurrent-capable backends first.
4. **Connection pooling** (implemented): Lazy-initialized shared `httpx.AsyncClient` in
   `OllamaChatClient`, `GenericOpenAIChatClient`, `VLLMChatClient`, and `OllamaBackend`.
   Replaces per-call client creation for better connection reuse under concurrency.

**Consequences:**
- First-time `concierge` launch installs all extras by default (browser, embed, mcp, otel).
  Slightly longer first install, but the system is immediately capable.
- Existing users with `CONCIERGE_EXTRA` set are unaffected.
- Concurrent graph nodes automatically prefer vLLM/llama_cpp(parallel>1) over Ollama.
- HTTP connection pooling reduces overhead for concurrent Ollama/vLLM requests.
- 35 new tests in `tests/test_ops_items.py` covering all items.

---

## ADR-034: Drop in-process backend; promote llama_cpp with --parallel

**Status:** Accepted
**Date:** 2026-03-10

**Context:**
Three backends — Ollama, llama_cpp (``llama-server``), and in-process (``mistralrs``) — all
run the same underlying GGUF inference engine (llama.cpp). The in-process backend added
complexity (PyO3 bindings, platform-specific wheels, ``[nano]`` optional extra) for marginal
benefit: it eliminated one IPC hop but was limited to a single model, couldn't serve
concurrent requests, and the ``mistralrs`` wheels were fragile across Python/OS versions.

Meanwhile, ``llama-server`` supports ``--parallel N`` for concurrent request slots with
shared KV cache, Vulkan GPU acceleration (works on AMD gfx1151 without ROCm), and process
isolation. Ollama provides model registry, pulling, and system service management. The two
are complementary, not redundant.

**Decision:**
1. **Remove the in-process backend entirely.** Delete ``Feature.INPROCESS``,
   ``InProcessChatClient``, ``_ensure_nano_model()``, ``probe_inprocess()``, nano GGUF
   config (``nano_gguf_filename``, ``nano_gguf_url``), and the ``[nano]`` optional extra
   from ``pyproject.toml``. Remove ``mistralrs`` as a dependency.

2. **Add ``--parallel N`` to LlamaCppBackend.** New ``parallel`` parameter (default 1,
   clamped to min 1). The subprocess command uses ``--ctx-size (ctx_size * parallel)``
   and ``--parallel N``. Default in ``backends.yaml`` set to ``parallel: 4``.

3. **Update tier priorities.** Remove ``inprocess`` from all tiers. Add ``llama_cpp`` to
   all tiers (NANO through SERVER). SERVER tier puts ``vllm`` first. Specific ordering:
   - nano/small/medium: ``[ollama, llama_cpp, cloud]``
   - large: ``[ollama, llama_cpp, vllm, cloud]``
   - server: ``[vllm, ollama, llama_cpp, cloud]``

4. **Keep both Ollama and llama_cpp** as complementary backends. Ollama handles model
   management/UX (~90% of local LLM users). llama_cpp handles concurrent inference
   (``--parallel``), Vulkan GPU support, and process isolation.

**Consequences:**
- Simpler dependency tree: no PyO3 bindings, no platform-specific ``mistralrs`` wheels.
- ``Feature`` enum has 9 members (was 10); ``_FALLBACK_BACKEND_PRIORITY`` references only
  ``ollama``, ``llama_cpp``, ``vllm``, ``cloud``.
- ``LlamaCppBackend(parallel=4)`` enables 4 concurrent request slots per process, at the
  cost of ~4x KV cache memory. Performance: ~25 t/s at 1 slot → ~17 t/s at 3 slots.
- Supersedes ADR-012 (three-layer stack) and ADR-014 (in-process as bootstrap layer).
