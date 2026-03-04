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

**Status:** Accepted
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

**Status:** Accepted
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
- NANO: `["inprocess", "ollama", "cloud"]`
- SMALL: `["ollama", "inprocess", "cloud"]`
- MEDIUM: `["ollama", "vllm", "inprocess", "cloud"]`
- LARGE: `["vllm", "ollama", "inprocess", "cloud"]`
- SERVER: `["vllm", "ollama", "inprocess", "cloud"]`

Each backend is probed with a dedicated `_try_<backend>()` helper:
- `_try_ollama()`: discovers models; if unreachable and `shutil.which("ollama")` finds
  the binary, auto-starts via `ensure_llm_available()` with a 30 s timeout.
- `_try_vllm()`: probes the OpenAI-compatible `/v1/models` endpoint.
- `_try_inprocess()`: checks `is_available()` from the mistral.rs module and looks for
  a GGUF model file at the platformdirs data path.
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
- Bootstrap (`first_run.py`) gains `_ensure_nano_model()` for GGUF auto-download, making
  the inprocess backend viable as a fallback without manual model download.
- 11 new tests cover fallback scenarios, feature gating, auto-start, comprehensive error
  messages, and `_try_backend` unit tests.
