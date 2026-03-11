# agentic-concierge v2: Ground-Up System Redesign

**Purpose:** Comprehensive design for the next-generation model runtime and agentic system,
informed by everything learned building v1 (Phases 1–14 + Specialist Marketplace).

**Target hardware:**
- Primary: AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S (RDNA 3.5 / gfx1151, unified memory, 128 GB RAM)
- Secondary: Apple M series (M1 Pro/Max/Ultra through M4)

**Target OS (priority order):** Fedora Linux > macOS > Windows

**Status:** IMPLEMENTED — Layer 1 (Model Runtime), Layer 2 (Task Decomposition), Layer 3 (Agent-Model Affinity) all implemented and tested on `main`
**Last verified:** 2026-03-11 — **1499 Python tests pass**, 27 Rust tests pass

---

## Table of Contents

1. [Lessons Learned from v1](#1-lessons-learned-from-v1)
2. [Problem Statement](#2-problem-statement)
3. [Design Principles](#3-design-principles)
4. [Hardware & Inference Backend Analysis](#4-hardware--inference-backend-analysis)
5. [Programming Language Analysis](#5-programming-language-analysis)
6. [Architecture: Three-Layer Design](#6-architecture-three-layer-design)
7. [Layer 1: Model Runtime](#7-layer-1-model-runtime)
8. [Layer 2: Recursive Task Decomposition](#8-layer-2-recursive-task-decomposition)
9. [Layer 3: Agent-Model Affinity](#9-layer-3-agent-model-affinity)
10. [Tool System Redesign](#10-tool-system-redesign)
11. [Resilience and Adaptability](#11-resilience-and-adaptability)
12. [Migration Strategy](#12-migration-strategy)
13. [Risks and Mitigations](#13-risks-and-mitigations)
14. [Verification Plan](#14-verification-plan)

---

## 1. Lessons Learned from v1

### 1.1 What Works Well

| Area | What works | Evidence |
|------|-----------|----------|
| Hexagonal architecture | Clean layer boundaries enable testing without real LLMs | 875 tests pass with mocked ports |
| Tool-calling protocol | Native OpenAI function calling is reliable | ADR-002; eliminated JSON-in-content fragility |
| Dynamic pack composition | Runtime tool+role composition is more flexible than hardcoded packs | tool_catalog.py + dynamic_pack.py |
| Quality gates | Gates 1–4 catch incomplete/incorrect work | run_tests tool, reviewer, finish validation |
| Capability-driven routing | Capabilities → tools → pack is the right abstraction | ADR-028; compose_tools_from_capabilities() |
| Rust launcher | Thin binary bootstrap is elegant; pure-Rust extraction works | ~5 MB static binary; no system tar needed |
| Runlog + checkpointing | Full audit trail; resumable runs | runlog.jsonl + checkpoint.json |

### 1.2 What Breaks in Practice

These are systemic issues discovered through real-world testing, not hypothetical concerns.

#### Issue A: LLMs Hallucinate Everything — Types, Capabilities, Tool Arguments

Small local LLMs (7–14B) routinely violate JSON schemas:
- `cmd` sent as string `"python -m pytest"` instead of array `["python", "-m", "pytest"]` → sandbox rejects ALL commands (Issue #6)
- `timeout_s` sent as string `"120"` instead of integer → `TypeError` crash (v0.3.9)
- Capability names hallucinated: `"web_browsing"` instead of `"web_comprehension"` → wrong model selected, wrong tools composed, wrong finish schema (Issue #1)

**Root cause:** Small models are unreliable schema followers. Every boundary between LLM output and system input must validate and coerce.

**Lesson:** Defensive coercion at every tool boundary. Never trust LLM output types. Validate capability names against a known set.

#### Issue B: One Model Does Everything Badly

A single 7B model handles routing, research, code generation, and review. It fails at each because:
- Routing requires instruction-following and structured output, not domain knowledge
- Research requires web comprehension and synthesis
- Code requires code generation capability
- Review requires reasoning about correctness — same model rubber-stamps its own work

**Root cause:** No model specialisation. The system has model profiles (Phase A) but uses a single base_url, so all specialists share whatever model happens to be loaded.

**Lesson:** Different tasks need different models. The system needs to load/unload models dynamically.

#### Issue C: Flat Task Decomposition

The orchestrator decomposes once into a flat list of specialist assignments. Complex tasks need recursive decomposition:
- "Build a CRUD app" should decompose into: plan architecture → create models → create API → create tests → run tests → fix failures
- Each sub-task may need further decomposition
- Currently, one specialist gets the entire task and must figure out the sequence internally

**Root cause:** Single-level orchestration with no feedback loop between planning and execution.

**Lesson:** Task decomposition should be recursive, with per-subtask critique before execution.

#### Issue D: No Model Lifecycle Management

The current system has no concept of model loading/unloading:
- Ollama manages models opaquely — no API to load/unload/query VRAM usage
- vLLM serves one model per process
- llama_cpp manages one process per model with `--parallel` for concurrent slots
- No resource awareness: can't answer "do I have VRAM for a 14B alongside an 8B?"
- No acquire/release semantics: parallel specialists may try to use models that aren't loaded

**Root cause:** Missing infrastructure layer between application and inference backend.

**Lesson:** Need explicit model lifecycle: acquire (ensure loaded) → use → release (allow unload).

#### Issue E: Reviewer Uses Same Model as Doer

Gate 4 reviewer using the same model as the doer produces meaningless reviews:
- "Looks good" on completely wrong answers
- Rejects correct answers because workspace has no files (quick_answer schema)
- No ability to select a reasoning-focused model for review

**Root cause:** No per-role model selection with different capability requirements.

**Lesson:** Agent roles (planner, doer, reviewer, critic) should declare model requirements independently.

#### Issue F: Rigid Finish Schemas

The research template forces academic structure (executive_summary, citations, bibliography) for simple factual questions like "How much does Amex Platinum cost?" This causes:
- LLM wastes steps creating bibliography files
- Fabricates citation metadata to satisfy the schema
- Simple questions take 20+ steps instead of 3

**Root cause:** Schema is per-template, not per-task-complexity.

**Lesson:** Finish schema should be selected based on task complexity, not specialist template.

### 1.3 LLM Reliability Patterns (Universal)

From 14 phases and hundreds of real-world runs, these patterns are universal:

| Pattern | Frequency | Impact | Mitigation |
|---------|-----------|--------|------------|
| Wrong JSON types | Every run with <14B models | Tool crashes | Coerce at boundary |
| Hallucinated names | ~30% of orchestrator calls | Wrong routing | Validate against known set |
| Infinite loops | ~15% of complex tasks | Wasted compute | Loop detection + escalation |
| Plain text instead of tool calls | ~20% with small models | Stalled progress | Corrective reprompt + escalation |
| Schema violations | ~40% of finish_task calls | Rejected output | Gate 2 validation + retry |
| Fabricated data | ~25% of research tasks | Wrong answers | Reviewer + source verification |

---

## 2. Problem Statement

The current system can route tasks and execute tool-calling loops, but lacks:

1. **Dynamic model management** — load/unload models based on current task needs and available resources
2. **Recursive task decomposition** — break complex tasks into smaller pieces, each with its own model and tool selection
3. **Per-subtask critique** — validate each decomposition step before execution
4. **Heterogeneous model mix** — use multiple specialized models concurrently (coder for code, reasoning model for review, small model for routing)
5. **Resource-aware scheduling** — know how much VRAM/RAM is available and schedule accordingly
6. **Multi-backend orchestration** — use Ollama for quick tasks, vLLM for throughput, llama_cpp for concurrent slots

The ideal system operates like a consultancy firm: many specialised consultants who don't know everything but know who to bring in for each sub-problem.

---

## 3. Design Principles

Carried forward from v1 (VISION.md §2) plus new principles from lessons learned:

1. **Quality over speed** — Prefer precision and correctness. Trade-off → quality.
2. **Local-first** — Local LLM is default and primary. Cloud only when local cannot meet quality/capability bar.
3. **Defensive at every boundary** — Never trust LLM output types. Validate and coerce at every tool/system interface.
4. **Right model for the right task** — Different agent roles need different model capabilities. MoE models (3B active params) can route; Qwen3.5-9B can do nearly everything; Phi-4-reasoning-14B excels at review; specialist coders outperform generalists.
5. **Recursive decomposition** — Complex tasks are broken down recursively until each leaf is manageable by available models.
6. **Resource-aware** — The system knows what hardware is available, what models fit, and schedules accordingly.
7. **Fail-open, fail-safe** — Review failures → accept with warning. Resource exhaustion → queue, don't crash.
8. **Observable** — Every decision (model selection, task decomposition, tool execution) is logged and auditable.
9. **Portable** — Fedora Linux first, macOS second. Same codebase, different backends.
10. **Resilient to ecosystem change** — Model names, backend URLs, GPU quirks, and capability profiles live in configuration, not source code. New models, backends, and hardware require config changes, not code changes. See §11.

---

## 4. Hardware & Inference Backend Analysis

### 4.1 Target Hardware Profiles

#### AMD RYZEN AI MAX+ PRO 395 ("Strix Halo")

| Property | Value |
|----------|-------|
| CPU | 16 cores / 32 threads, Zen 5 |
| GPU | Radeon 8060S, 40 CUs, **RDNA 3.5** (gfx1151) — NOT RDNA 4 |
| Memory | 128 GB unified (~115–120 GB usable for inference) |
| Memory bandwidth | ~215 GB/s |
| GPU compute | ROCm 6.4.1+ (gfx1151); ROCm 7.x via TheRock nightly builds |
| NPU | XDNA 2 (50 TOPS) — limited software support currently |

**Key advantage:** 128 GB unified memory means large models (70B Q4 at ~40 GB) run comfortably. Can even fit 70B Q8_0 (~75 GB). Memory bandwidth of ~215 GB/s feeding 40 CUs gives ~3–5 tok/s for 70B models, ~50 tok/s for 7B.

**Key challenges (from real benchmarks, issue trackers, and March 2026 verification):**
- **ROCm/HIP is BROKEN in Ollama on Linux** (ollama #13589, still open). HIP backend crashes with VM fault in `libhsa-runtime64.so.1` during GPU discovery → **silent fallback to CPU**. This is the critical blocker.
- **Vulkan outperforms ROCm/HIP by up to 50%** for token generation even where ROCm works. ROCm KV cache bug (llama.cpp #18011, closed as NOT_PLANNED — driver issue) forces KV cache into shared memory.
- **Model loading extremely slow past ~64 GB on ROCm** — workaround: `--no-mmap` flag (llama.cpp #15018, closed with workaround).
- **ROCm 7.2.0** (latest stable) requires `HSA_OVERRIDE_GFX_VERSION` for gfx1151. **ROCm 7.9.0 preview** adds official gfx1151 support.
- Linux kernel >= 6.18.4 mandatory for Strix Halo KFD driver fixes.
- GPU hangs under combined AI + video encoding workloads (ROCm #5665).

**Practical recommendation (updated March 2026):** ROCm 7.2+ supports gfx1151 via the `gfx11-generic` ISA target. Both ROCm and Vulkan backends work. ROCm may underutilise unified memory on APUs (allocates VRAM only, not GTT), making Vulkan competitive for prompt processing. Both options are valid; the platform profile no longer forces either.

#### Apple M Series (M1 Pro through M4 Max)

| Property | Value |
|----------|-------|
| CPU | 8–16 cores (performance + efficiency) |
| GPU | 16–40 cores, Metal |
| Memory | 16–192 GB unified |
| Frameworks | Metal (llama.cpp), MLX (Apple's ML framework) |

**Key advantage:** Best-in-class local LLM support. Both llama.cpp (Metal) and MLX are highly optimised for Apple Silicon's unified memory architecture.

**MLX vs llama.cpp on Apple Silicon:**
- MLX is **20–87% faster** than llama.cpp for token generation, with the gap widening on larger models (arXiv:2601.19139v1, M4 Max benchmarks).
- vllm-mlx achieves up to **525 tok/s** on text models (M4 Max) with continuous batching.
- An M2 Ultra (192 GB) study confirmed MLX and MLC-LLM deliver the highest sustained throughput and lowest per-token latency vs Ollama, llama.cpp, PyTorch MPS (arXiv:2511.05502).

**Key challenge:** Smaller memory configs (16–32 GB) limit model size. MLX uses SafeTensors/MLX format rather than GGUF (conversion tools exist).

### 4.2 Inference Backend Comparison

**(Verified March 9, 2026 against latest releases)**

| Backend | Version | GPU Support | Multi-model | Key facts |
|---------|---------|-------------|-------------|-----------|
| **Ollama** | **v0.17.7** (Mar 5) | CUDA, Metal, Vulkan, ROCm (gfx1151 via gfx11-generic) | Up to N concurrent (`OLLAMA_MAX_LOADED_MODELS`) | New engine (v0.17) replaces llama.cpp server mode; 40% faster prompt processing; KV cache 8-bit quant; MLX engine on macOS; `keep_alive` lifecycle control |
| **llama.cpp** | **b8248** (Mar 9) | CUDA, ROCm/HIP, Vulkan, Metal, SYCL, MUSA, CANN, OpenCL, Hexagon, CPU | One per process | **11 backends**; daily releases; `--no-mmap` workaround for >64GB ROCm loading; GPU token sampling (35% faster on NVIDIA) |
| **vLLM** | **v0.17.0** (Mar 7) | CUDA, ROCm (gfx1151 supported via gfx11-generic in ROCm 7.2+) | One per process | FlashAttention 4; pipeline parallelism; first-class ROCm support (93% test pass rate); PyTorch 2.10; pre-built Docker images available |
| **MLX / vllm-mlx** | MLX **v0.31.0** / vllm-mlx **v0.2.6** | Metal + **CUDA** (new!) | Continuous batching via vllm-mlx | 20–87% faster than llama.cpp on Apple Silicon; now runs on NVIDIA too; MCP tool calling; Anthropic API compat |
| **SGLang** | **v0.5.9** (Feb 24) | CUDA, ROCm (Instinct only) | One per process | ROCm 7 standardized; no gfx1151; LoRA overlap; Anthropic API compat |
| **LM Studio** | **v0.4.6** (Feb 27) | CUDA, Metal, Vulkan, ROCm | Multiple via UI | AMD Variable Graphics Memory for 128B on Strix Halo; continuous batching; LM Link remote; Vulkan regressions in v0.4.4+ |
| ~~**mistral.rs**~~ | — | CUDA, Metal, CPU | Per-process | *(Removed by ADR-034 — replaced by managed llama-server with `--parallel`)* |

### 4.3 Backend Strategy (revised with March 2026 findings)

**Primary (Fedora Linux / AMD Strix Halo):**
1. **vLLM with ROCm** — Highest throughput for single-model serving. ROCm 7.2+ supports gfx1151 via `gfx11-generic` ISA target. First-class ROCm platform in vLLM (93% CI pass rate as of Jan 2026). Pre-built Docker images available.
2. **Ollama** — Default multi-model server. Supports ROCm and Vulkan backends. Concurrent model support via `OLLAMA_MAX_LOADED_MODELS`.
3. **llama.cpp server** — Direct control fallback. One process per pinned model. Explicit `--n-gpu-layers`. Use when Ollama's concurrent model management is insufficient.

**Primary (macOS / Apple Silicon):**
1. **Ollama** — Same API, Metal backend. Simplest setup. v0.17 includes MLX engine integration.
2. **vllm-mlx** (v0.2.6) — 20–87% faster than llama.cpp; continuous batching; OpenAI + Anthropic compat; MCP tool calling. Preferred for throughput.
3. **llama.cpp server with Metal** — Fallback; direct control.

**Model runtime must abstract over all backends** — the application layer should not know or care which backend is serving a particular model.

### 4.3.1 Ollama Model Management (Key Discovery)

Ollama provides more control than previously assumed:

| Operation | How | Notes |
|-----------|-----|-------|
| **List loaded models** | `GET /api/ps` | Shows VRAM usage per model |
| **Preload a model** | `POST /api/generate` with empty prompt | Forces model into memory |
| **Unload a model** | Request with `keep_alive: "0"` | Immediately frees memory |
| **Keep model loaded** | `keep_alive: "-1"` | Never auto-unload |
| **Concurrent models** | `OLLAMA_MAX_LOADED_MODELS=N` | Default 3; requires sufficient memory |
| **Parallel requests** | `OLLAMA_NUM_PARALLEL=N` | Default 1; RAM scales with N × context |
| **Queue management** | `OLLAMA_MAX_QUEUE=N` | Default 512; rejects when full |

This means the model runtime can use Ollama as a managed backend with explicit load/unload control — not just fire-and-forget HTTP requests.

### 4.4 Multi-Model Concurrency Analysis

The core challenge: **running multiple specialized models simultaneously**.

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Single Ollama, concurrent models** | `OLLAMA_MAX_LOADED_MODELS=5`; preload via empty generate; unload via `keep_alive: 0` | Simplest; proven; Ollama manages GPU offloading | Memory management opaque; can't pin GPU layers per model |
| **Ollama + dedicated llama.cpp servers** | Ollama for general; llama.cpp for pinned specialist models on dedicated ports | Full control for specialists; Ollama ease-of-use for transient models | More processes to manage |
| **Multiple llama.cpp servers** | One server per model, different ports | Full control; explicit `--n-gpu-layers` | Must manage all processes; manual model loading |
| **vLLM per model class** | One vLLM per model class | High throughput per model; continuous batching | Heavy resource usage; one model per process |

**Recommended approach (revised based on research):** **Single Ollama instance** with `OLLAMA_MAX_LOADED_MODELS` set to 3–5. Use `/api/ps` to monitor loaded models, empty-prompt generate to preload, and `keep_alive: "0"` to unload. This gives the model runtime explicit lifecycle control without managing separate processes.

**Fallback:** If Ollama's concurrent model handling proves insufficient (e.g., can't control GPU layer allocation per model), escalate to the Ollama + dedicated llama.cpp server approach for pinned specialist models.

### 4.5 Model Landscape (March 2026)

The model landscape has shifted dramatically. MoE architecture is now dominant, and small models have leapfrogged larger ones on agentic benchmarks.

**Recommended models by agent role (verified March 2026):**

| Role | Model | Params | Memory (Q4_K_M) | Why |
|------|-------|--------|-----------------|-----|
| **Router / Orchestrator** | Qwen3.5-9B | 9B | ~5 GB | Beats GPT-OSS-120B on reasoning (81.7 GPQA Diamond); 66.1 BFCL-V4 tool calling; 79.1 TAU2 tool use. Outperforms 80B Qwen3-Next on agentic benchmarks. The sweet spot. |
| **Coder (primary)** | Qwen2.5-Coder-32B | 32B dense | ~18 GB | Rivals GPT-4o on coding; 88.4% HumanEval; 128K context; 92+ languages |
| **Coder (lightweight)** | Qwen2.5-Coder-7B | 7B dense | ~4 GB | 88.4% HumanEval; beats Codestral-22B; FIM support |
| **Coder (frontier MoE)** | Qwen3-Coder-Next | 80B/3B active | ~46 GB | 512 experts, 10 active; SWE-Bench-Pro competitive with 10-20x larger models |
| **Reasoner / Reviewer** | Phi-4-reasoning | 14B | ~8 GB | Approaches full DeepSeek R1 (671B) performance; outperforms R1-Distill-70B |
| **Reasoner (larger)** | DeepSeek-R1-Distill-Qwen-32B | 32B dense | ~18 GB | Outperforms OpenAI o1-mini; best balance of reasoning + local runnability |
| **Agentic (MoE)** | GLM-4.7-Flash | 30B/3B active | ~17 GB | Interleaved thinking — thinks before every tool call; 87.4 tau2-Bench (highest open-source) |
| **Agentic (MoE)** | Nemotron 3 Nano | 30B/3B active | ~17 GB | Hybrid Mamba-Transformer MoE; 1M context; optimized for multi-agent |
| **Tool calling specialist** | xLAM-2-8b-fc-r | 8B | ~5 GB | SOTA on BFCL and tau-bench; outperforms GPT-4o on function calling |

**Key insight:** Qwen3.5-9B is a generational leap — a single 9B model that would have been unthinkable a year ago. For our project, it should replace llama3.1:8b and qwen2.5:7b as the default model. The 5 GB memory footprint means we can run 4–5 instances simultaneously on 128 GB hardware.

**MoE memory note:** MoE models (Qwen3-Coder-Next, GLM-4.7-Flash, Nemotron 3 Nano) have ALL parameters in memory but only activate a fraction per token. A "30B total / 3B active" model needs ~17 GB RAM but generates at speeds comparable to a 3B dense model.

### 4.6 Memory Budget Analysis (128 GB unified, AMD)

Approximate GGUF memory requirements (weights only; add 1–4 GB per model for KV cache overhead depending on context length):

| Model | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | FP16 |
|-------|--------|--------|--------|------|------|------|
| 3B | ~1.5 GB | ~2 GB | ~2.5 GB | ~3 GB | ~3.5 GB | ~6 GB |
| 7B | ~3 GB | ~3.8 GB | ~4.7 GB | ~5.5 GB | ~7.5 GB | ~14 GB |
| 14B | ~6 GB | ~7.6 GB | ~9.4 GB | ~11 GB | ~15 GB | ~28 GB |
| 32B | ~14 GB | ~18 GB | ~22 GB | ~25 GB | ~34 GB | ~64 GB |
| 70B | ~30 GB | ~38-40 GB | ~47 GB | ~55 GB | ~75 GB | ~140 GB |

Q4_K_M offers ~75% size reduction vs FP16 with <2% perplexity increase — the sweet spot.
Q5_K_M offers <1% perplexity increase for quality-conscious workloads.

**Concurrent model scenarios on Strix Halo (~115 GB usable after OS/BIOS):**

| Scenario | Models loaded | Total memory | Feasible? | Notes |
|----------|-------------|------------|-----------|-------|
| One large reasoner | 70B Q4_K_M | ~44 GB | Yes | ~3–5 tok/s generation |
| One large Q8 | 70B Q8_0 | ~80 GB | Yes | Better quality; ~3 tok/s |
| **Recommended team** | Qwen3.5-9B (router) + Qwen2.5-Coder-32B + Phi-4-reasoning-14B + xLAM-2-8b (tools) | ~36 GB | Yes, comfortably | 4 specialists; ~79 GB headroom |
| **Dense specialist team** | Qwen3.5-9B × 2 + 14B + 7B × 3 | ~32 GB | Comfortably | 6 models; ~83 GB headroom |
| **MoE-heavy team** | GLM-4.7-Flash (30B/3B) + Qwen3-Coder-Next (80B/3B) + Qwen3.5-9B | ~68 GB | Yes | 3 MoE + 1 dense; ~47 GB headroom |
| Maximum density | 9B Q4 × 10 | ~50 GB | Yes | 10 concurrent Qwen3.5-9B instances |
| Quality team | 32B Q5 + 14B Q5 + 9B Q5 × 2 + 9B Q5 | ~44 GB | Yes | Better quality quants |

**Conclusion:** 128 GB unified memory is extremely generous. The system can comfortably run 3–5 models simultaneously at Q4_K_M, or 2–3 at the higher-quality Q5_K_M. Even a full 15-model dense configuration fits. The bottleneck is not memory but **memory bandwidth** (~215 GB/s) — inference throughput scales with how fast weights can be read, not how much memory is available.

**Performance expectations (Strix Halo, Vulkan backend):**

| Model size | Approx tok/s (generation) | Approx tok/s (prompt) |
|-----------|--------------------------|----------------------|
| 3B Q4 | ~100–120 | ~1500+ |
| 7B Q4 | ~45–55 | ~1400–1500 |
| 14B Q4 | ~20–30 | ~800–1000 |
| 32B Q4 | ~10–15 | ~400–600 |
| 70B Q4 | ~3–5 | ~200–300 |

---

## 5. Programming Language Analysis

### 5.1 Current Split

| Component | Language | Rationale |
|-----------|----------|-----------|
| Application core (agentic logic, orchestration, tools) | Python | Rapid development; async/await; LLM ecosystem |
| Launcher (bootstrap, self-update) | Rust | Static binary; no runtime dependencies; fast startup |

### 5.2 Analysis for v2

| Concern | Python | Rust | Recommendation |
|---------|--------|------|----------------|
| **Agentic logic** (orchestration, routing, tool dispatch) | Natural fit; async/await; rapid iteration | Over-engineered for I/O-bound orchestration | **Python** — I/O-bound; development speed matters |
| **Model runtime** (acquire/release, scheduling, resource tracking) | Adequate for orchestration layer | Better for resource management, concurrency, reliability | **Python** — adequate; Rust not justified unless profiling shows otherwise |
| **Inference backends** | httpx async HTTP; good ecosystem | Direct llama.cpp bindings; zero-overhead | **Python** — all backends expose HTTP APIs; no CPU-bound bottleneck |
| **Tool execution** (sandbox, file I/O, shell) | Current implementation works; subprocess-based | Faster sandboxing; no GIL for parallel tools | **Python** — subprocess-based tools are I/O-bound |
| **Launcher** (bootstrap, distribution) | Requires Python installed | Static binary; zero dependencies | **Rust** — keep current approach |
| **Hot path** (if any) | Phase 14 analysis: all I/O-bound | Would help if CPU-bound | **N/A** — no CPU-bound hot path identified |

### 5.3 Verdict

**Keep the current split:** Python for all application logic, Rust for the launcher.

Rationale:
- Phase 14 hot-path analysis confirmed all execution paths are I/O-bound (LLM HTTP calls, subprocess tools)
- Python async/await maps perfectly to concurrent model queries and tool execution
- The LLM ecosystem (tokenizers, embeddings, GGUF parsing) is Python-first
- Development velocity matters more than micro-optimisation for an agentic system
- The only Rust candidate (model runtime) would add build complexity for no measured benefit

**Exception:** If profiling reveals a CPU bottleneck (e.g., embedding computation at scale), add a targeted PyO3 extension — don't rewrite.

---

## 6. Architecture: Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Interfaces (CLI, HTTP API, Web UI)                                     │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────┐
│  Layer 3: Agent-Model Affinity                                          │
│                                                                         │
│  Declares per-role model requirements:                                  │
│    Planner: {reasoning: 0.8, structured_output: 0.7}                    │
│    Coder:   {code_python: 0.9, instruction_following: 0.7}              │
│    Reviewer:{reasoning: 0.8, instruction_following: 0.8}                │
│    Router:  {structured_output: 0.8, instruction_following: 0.7}        │
│                                                                         │
│  Maps role requirements to model runtime requests                       │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────┐
│  Layer 2: Recursive Task Decomposition                                  │
│                                                                         │
│  Task Graph (DAG) replaces flat specialist list:                        │
│    Root task → subtasks → sub-subtasks → leaf tasks                     │
│  Per-node:                                                              │
│    - Decompose (planner agent)                                          │
│    - Critique decomposition (critic agent)                              │
│    - Execute leaf (doer agent)                                          │
│    - Review result (reviewer agent)                                     │
│                                                                         │
│  Adaptive depth: stop decomposing when leaf is within model capability  │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────┐
│  Layer 1: Model Runtime                                                 │
│                                                                         │
│  Manages model lifecycle across backends:                               │
│    acquire(model, capabilities) → ModelHandle                           │
│    release(handle)                                                      │
│                                                                         │
│  Responsibilities:                                                      │
│    - Model inventory (what's available, what's loaded)                  │
│    - Resource tracking (VRAM/RAM usage per model)                       │
│    - Backend abstraction (Ollama, llama.cpp, vLLM, MLX)                │
│    - Load/unload scheduling (LRU eviction when full)                   │
│    - Multi-backend routing (different models on different backends)     │
│    - Preloading (anticipate next model needs from task graph)           │
└─────────────────────────────────────────────────────────────────────────┘
```

Each layer is independently testable. Layer boundaries are defined by protocols (Python Protocol classes).

---

## 7. Layer 1: Model Runtime

### 7.1 Core Concepts

```python
@dataclass
class ModelSlot:
    """A model loaded and ready for inference on a specific backend."""
    model_id: str           # e.g. "qwen2.5:7b", "deepseek-coder-v2:16b"
    backend: str            # "ollama", "llama_cpp", "vllm", "mlx"
    base_url: str           # endpoint URL
    estimated_vram_mb: int  # estimated memory consumption
    loaded_at: float        # timestamp
    refcount: int           # number of active users
    capabilities: dict[str, float]  # from model profile

@dataclass
class ModelHandle:
    """Opaque handle returned by acquire(). Must be released."""
    slot: ModelSlot
    chat_client: ChatClient  # ready-to-use client
    # context manager: auto-releases on exit

class ModelRuntime(Protocol):
    """Central model lifecycle manager."""

    async def acquire(
        self,
        requirements: dict[str, float],  # capability requirements
        *,
        prefer_model: str | None = None,  # hint: prefer specific model
        require_tool_calling: bool = True,
        timeout_s: float = 30.0,
    ) -> ModelHandle:
        """Ensure a model matching requirements is loaded; return handle."""
        ...

    async def release(self, handle: ModelHandle) -> None:
        """Decrement refcount; model eligible for eviction."""
        ...

    async def inventory(self) -> list[ModelInfo]:
        """List all known models (loaded and available-to-load)."""
        ...

    async def status(self) -> RuntimeStatus:
        """Current resource usage: loaded models, VRAM/RAM used/free."""
        ...
```

### 7.2 Resource Management

```
                    ┌─────────────────────────┐
                    │    Resource Tracker      │
                    │                          │
                    │  total_memory: 128 GB    │
                    │  reserved_os: 20 GB      │
                    │  available: 108 GB       │
                    │                          │
                    │  loaded_models:           │
                    │    qwen2.5:7b   → 4.5 GB │
                    │    phi-4-mini   → 2.5 GB │
                    │    deepseek:16b → 10 GB  │
                    │                          │
                    │  used: 17 GB             │
                    │  free: 91 GB             │
                    └─────────────────────────┘
```

**Eviction policy:** When `acquire()` needs memory:
1. Find models with `refcount == 0` (no active users)
2. Sort by last-used timestamp (LRU)
3. Evict oldest unused models until enough memory is free
4. If still insufficient and all models are in use → queue the request (with timeout)

### 7.3 Backend Abstraction

```python
class InferenceBackend(Protocol):
    """Abstraction over inference serving backends."""

    async def load_model(self, model_id: str) -> ModelSlot:
        """Load a model and return slot info."""
        ...

    async def unload_model(self, model_id: str) -> None:
        """Unload a model to free resources."""
        ...

    async def list_loaded(self) -> list[str]:
        """List currently loaded model IDs."""
        ...

    async def list_available(self) -> list[str]:
        """List models available to load (downloaded/cached)."""
        ...

    def build_client(self, model_id: str) -> ChatClient:
        """Create a ChatClient for the given loaded model."""
        ...

    async def estimate_memory(self, model_id: str) -> int:
        """Estimate memory in MB for loading this model."""
        ...
```

**Implementations:**

1. **OllamaBackend** (primary for both platforms) — Uses Ollama v0.17+ lifecycle API:
   - `load_model()`: `POST /api/generate` with empty prompt + `keep_alive: "-1"` (pin in memory)
   - `unload_model()`: `POST /api/generate` with `keep_alive: "0"` (immediate unload)
   - `list_loaded()`: `GET /api/ps` (returns model names + VRAM usage + expiration)
   - `list_available()`: `GET /api/tags` (downloaded models)
   - `estimate_memory()`: Parse model metadata from tags/show response
   - Ollama v0.17 manages GPU offloading internally with new engine (not llama.cpp server)
   - **Strix Halo Linux:** ROCm 7.2+ supported via gfx11-generic ISA target; Vulkan also available
   - `keep_alive` accepts: duration strings ("10m"), seconds (3600), -1 (forever), 0 (immediate unload)

2. **LlamaCppBackend** (fallback for explicit GPU control) — Manages llama.cpp b8248+ server processes:
   - `load_model()`: Start `llama-server --model X --port N --n-gpu-layers M`
   - `unload_model()`: Kill the process
   - Explicit `--n-gpu-layers` control for fine-grained GPU allocation
   - One process per model; port managed by backend
   - **Strix Halo:** Use Vulkan backend (`-ngl 999` with Vulkan build)

3. ~~**VLLMBackend**~~ — **Removed from design.** gfx1151 not supported upstream (closed "not planned"). For datacenter AMD (MI series) or NVIDIA only. Could be re-added as an optional backend for those deployments.

4. **MLXBackend** (macOS, and experimentally NVIDIA via MLX v0.30.4+) — Uses vllm-mlx v0.2.6 or mlx-lm serve. 20–87% faster than llama.cpp on Apple Silicon. OpenAI + Anthropic compatible API. MCP tool calling support. Continuous batching.

### 7.4 Model Selection Algorithm

```python
async def acquire(self, requirements, *, prefer_model=None, ...):
    # 1. Check if a loaded model satisfies requirements
    for slot in self._loaded_models:
        if self._satisfies(slot, requirements):
            slot.refcount += 1
            return ModelHandle(slot, slot.build_client())

    # 2. Check if an unloaded model satisfies requirements
    candidates = self._match_available(requirements)
    if not candidates:
        raise NoModelAvailable(requirements)

    # 3. Pick best candidate (highest capability score)
    best = candidates[0]

    # 4. Ensure enough memory (evict if needed)
    needed = await self._backend.estimate_memory(best.model_id)
    await self._ensure_memory(needed)

    # 5. Load the model
    slot = await self._backend.load_model(best.model_id)
    slot.refcount = 1
    self._loaded_models.append(slot)

    return ModelHandle(slot, self._backend.build_client(best.model_id))
```

### 7.5 Preloading from Task Graph

The model runtime accepts hints from the task decomposition layer:

```python
async def preload_hint(self, requirements: dict[str, float]) -> None:
    """Hint that a model with these requirements will be needed soon.

    Non-blocking. The runtime may preload the model if resources allow.
    """
    # Find best unloaded model matching requirements
    # If memory available, start loading in background
    # If not enough memory, do nothing (will evict when actually needed)
```

This enables the task graph to tell the runtime "I'm about to need a coder model" while the current planning step is still running.

---

## 8. Layer 2: Recursive Task Decomposition

### 8.1 Task Graph

Replace the flat `OrchestrationPlan` with a task DAG:

```python
@dataclass
class TaskNode:
    """A node in the task decomposition graph."""
    id: str
    description: str
    parent_id: str | None

    # Decomposition state
    status: Literal["pending", "decomposing", "critiqued", "executing", "reviewing", "done", "failed"]
    children: list[str]  # child TaskNode IDs

    # Execution requirements (set during decomposition)
    required_capabilities: list[str]
    required_tools: list[str]
    finish_schema_key: str  # quick_answer, code, research_report, general

    # Results
    result: dict | None
    critique: str | None  # critique feedback if decomposition was revised

@dataclass
class TaskGraph:
    """DAG of task nodes. Root is the user's original request."""
    nodes: dict[str, TaskNode]
    root_id: str

    def leaves(self) -> list[TaskNode]:
        """Return executable leaf nodes (no children)."""
        ...

    def ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all satisfied."""
        ...
```

### 8.2 Decomposition Loop

```
User prompt
    │
    ▼
┌─────────────────────┐
│  Planner Agent       │  (model: reasoning-focused, e.g. phi-4-mini, deepseek-r1)
│                      │
│  Decomposes task     │
│  into subtasks       │
│  with capability     │
│  requirements        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Critic Agent        │  (model: different from planner — prevents self-approval)
│                      │
│  Reviews plan:       │
│  - Are subtasks      │
│    achievable?       │
│  - Missing steps?    │
│  - Over-decomposed?  │
│  - Capabilities      │
│    realistic?        │
└──────────┬──────────┘
           │
           ├─── Critique passes ──► Execute leaves
           │
           └─── Critique fails  ──► Re-plan with feedback
                                    (max 2 re-plans)

For each leaf node:
    │
    ▼
┌─────────────────────┐
│  Doer Agent          │  (model: matched to leaf's capabilities)
│                      │
│  Executes leaf task  │
│  using appropriate   │
│  tools               │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Reviewer Agent      │  (model: reasoning-focused, different from doer)
│                      │
│  Reviews result      │
│  against leaf        │
│  description         │
└──────────┬──────────┘
           │
           ├─── Approved ──► Mark node done; check if parent ready
           │
           └─── Rejected ──► Re-execute with feedback (max 2)
```

### 8.3 Adaptive Depth Control

Not every task needs deep decomposition. The system should stop decomposing when:

1. **Leaf is simple enough:** A single model with the right capabilities can handle it in < 10 tool calls
2. **Max depth reached:** Hard limit (default 3 levels) prevents runaway decomposition
3. **Available models suffice:** If the available 7B model can handle the task, don't decompose further just to use a 3B

```python
def should_decompose(node: TaskNode, available_models: list[ModelInfo]) -> bool:
    """Decide whether to further decompose a task node."""
    # Never decompose beyond max depth
    if node.depth >= MAX_DECOMPOSITION_DEPTH:
        return False

    # Simple tasks don't need decomposition
    if node.estimated_complexity <= "simple":
        return False

    # If a model with high enough capabilities is available, execute directly
    best_model = match_models(available_models, node.required_capabilities)
    if best_model and best_model.capability_score >= 0.8:
        return False

    return True
```

### 8.4 Parallelism in the Task Graph

Leaf nodes with no data dependencies can execute in parallel:

```
Task: "Build a CRUD app with tests"
    │
    ├── Plan architecture (sequential — blocks all)
    │
    ├── Create data model (depends on architecture)
    │
    ├── Create API endpoints (depends on data model)
    │   │
    │   ├── GET /todos (parallel)
    │   ├── POST /todos (parallel)
    │   ├── PUT /todos/:id (parallel)
    │   └── DELETE /todos/:id (parallel)
    │
    ├── Create tests (depends on API endpoints)
    │
    └── Run tests and fix (depends on tests)
```

The task graph scheduler identifies independent nodes and runs them concurrently, each potentially using a different model.

---

## 9. Layer 3: Agent-Model Affinity

### 9.1 Agent Roles

Each agent role declares its model requirements. Recommended models based on March 2026 benchmarks:

```python
AGENT_ROLES = {
    "router": {
        "description": "Routes user requests to appropriate decomposition",
        "requirements": {
            "structured_output": 0.8,
            "instruction_following": 0.7,
        },
        "prefer_small": True,  # Fast response; Qwen3.5-9B is ideal
        "require_tool_calling": True,
        # Recommended: Qwen3.5-9B (66.1 BFCL-V4, 79.1 TAU2)
        # Alternative: xLAM-2-8b-fc-r (SOTA on BFCL)
    },
    "planner": {
        "description": "Decomposes tasks into subtask graphs",
        "requirements": {
            "reasoning": 0.8,
            "structured_output": 0.7,
            "instruction_following": 0.8,
        },
        "prefer_small": False,  # Reasoning benefits from larger models
        "require_tool_calling": True,
        # Recommended: Qwen3.5-9B (81.7 GPQA Diamond — beats 120B models)
        # Alternative: Phi-4-reasoning-14B, DeepSeek-R1-Distill-32B
    },
    "critic": {
        "description": "Reviews plans and identifies issues",
        "requirements": {
            "reasoning": 0.8,
            "instruction_following": 0.7,
        },
        "prefer_small": True,
        "require_tool_calling": True,
        "must_differ_from": ["planner"],  # Different model than planner
        # Recommended: Phi-4-reasoning-14B (approaches full R1 performance)
    },
    "coder": {
        "description": "Writes and modifies code",
        "requirements": {
            "code_python": 0.8,  # Or code_rust, code_sql depending on task
            "instruction_following": 0.7,
        },
        "require_tool_calling": True,
        # Recommended: Qwen2.5-Coder-32B (88.4% HumanEval, rivals GPT-4o)
        # Lightweight: Qwen2.5-Coder-7B (88.4% HumanEval at 4GB)
        # Frontier MoE: Qwen3-Coder-Next (80B/3B active, 46GB)
    },
    "researcher": {
        "description": "Searches web and synthesises information",
        "requirements": {
            "web_comprehension": 0.7,
            "summarisation": 0.7,
            "instruction_following": 0.7,
        },
        "require_tool_calling": True,
        # Recommended: Qwen3.5-9B (general-purpose excellence)
    },
    "reviewer": {
        "description": "Reviews completed work for correctness",
        "requirements": {
            "reasoning": 0.8,
            "instruction_following": 0.8,
        },
        "prefer_small": True,
        "require_tool_calling": True,
        "must_differ_from": ["coder", "researcher"],  # Never review own work
    },
}
```

### 9.2 Model Assignment

When the task graph scheduler needs to execute a node:

```python
async def assign_model(node: TaskNode, runtime: ModelRuntime) -> ModelHandle:
    """Select and acquire the best model for a task node."""

    # Determine agent role from node type
    role = determine_role(node)  # "coder", "researcher", "reviewer", etc.
    role_spec = AGENT_ROLES[role]

    # Merge role requirements with task-specific capabilities
    requirements = {**role_spec["requirements"]}
    for cap in node.required_capabilities:
        if cap not in requirements:
            requirements[cap] = 0.6  # Default threshold for task caps

    # Acquire from model runtime
    handle = await runtime.acquire(
        requirements=requirements,
        require_tool_calling=role_spec.get("require_tool_calling", True),
    )

    return handle
```

### 9.3 "Must Differ From" Constraint

The reviewer must use a different model than the doer. This is enforced by the affinity system:

```python
# During task graph execution, track which model handled each role:
node_model_map: dict[str, str] = {}  # node_id → model_id

# When assigning reviewer:
doer_model = node_model_map[node.id]
handle = await runtime.acquire(
    requirements=reviewer_requirements,
    exclude_models=[doer_model],  # Different model
)
```

If no alternative model is available, fall back to the same model (fail-open, like v1).

---

## 10. Tool System Redesign

### 10.1 Current Tool Catalog (Keep)

The 8-tool catalog from v1 is solid:

| Tool | Category | Keep? | Notes |
|------|----------|-------|-------|
| `shell` | code | Yes | Core execution capability |
| `read_file` | file_io | Yes | |
| `write_file` | file_io | Yes | |
| `list_files` | file_io | Yes | |
| `run_tests` | code | Yes | Quality gate integration |
| `web_search` | web | Yes | |
| `fetch_url` | web | Yes | |
| `cross_run_search` | search | Yes | Cross-run memory |

### 10.2 New Tools for v2

| Tool | Category | Purpose |
|------|----------|---------|
| `think` | meta | Scratchpad for reasoning (no side effects) — helps small models plan |
| `ask_user` | meta | Request clarification from user (breaks out of agent loop) |
| `search_codebase` | code | Grep/ripgrep over workspace (faster than shell + grep) |

### 10.3 Tool Safety (Hardened)

Carry forward all v1 safety measures plus:

1. **Type coercion at every boundary** — `cmd: str → shlex.split()`, `timeout_s: str → int()`, etc.
2. **Schema validation before tool dispatch** — Validate all required params exist and have correct types
3. **Tool output truncation** — Already implemented (`MAX_TOOL_OUTPUT_CHARS`)
4. **Sandbox unchanged** — SandboxPolicy with path safety + command allowlist

---

## 11. Resilience and Adaptability

The local LLM ecosystem changes faster than any other AI domain — new model families every few weeks,
backend engines being rewritten or deprecated, GPU driver stacks maturing, quantization formats emerging.
v1's codebase has **12 categories of hardcoded assumptions** that would break when the landscape shifts.
This section designs each one out, establishing a system that adapts to change without code modifications.

### 11.1 Design Philosophy: Configuration, Discovery, Contracts

Three principles govern resilience:

1. **Configuration over code** — Every assumption about the environment (model names, backend URLs,
   GPU capabilities, model recommendations) lives in externalized configuration, not source code.
   The codebase defines *contracts* (protocols, schemas) and *defaults* (shipped config files).
   Users override defaults; the system adapts.

2. **Runtime discovery over static tables** — Instead of hardcoding what backends exist or what
   models can do, the system *probes* at startup: which backends respond? What models are available?
   What are their actual capabilities? Discovery results populate the same data structures as
   static configuration, so the rest of the system is indifferent to the source.

3. **Versioned contracts with fallback** — Each integration point (backend API, model profile schema,
   tool definition format) has a version. When a backend's API changes, the system detects the version
   mismatch and falls back to a compatible interaction pattern rather than crashing.

### 11.2 Brittleness Audit and Solutions

The following table maps every hardcoded assumption in v1 to its v2 solution:

| # | Category | v1 Location | v1 Problem | v2 Solution |
|---|----------|-------------|------------|-------------|
| 1 | Model names | `model_advisor.py` `_MODEL_TABLE` | Hardcodes qwen2.5 family; not configurable | **Externalized model catalog** (§11.3) |
| 2 | Model capabilities | `model_profiles.py` `BUILTIN_PROFILES` | 21 families; `get_profile()` override param never used in production | **Discovery + overlay profiles** (§11.4) |
| 3 | Backend URLs | `constants.py` `DEFAULT_BACKEND_URLS` | Scattered across 7+ files; only localhost | **Backend registry with health probing** (§11.5) |
| 4 | Ollama-specific calls | `llm_discovery.py` | Direct `/api/tags`, `ollama pull` CLI bypassing abstractions | **InferenceBackend protocol** (§7.3) absorbs all backend-specific calls |
| 5 | Backend priority | `features.py` `BACKEND_PRIORITY` | Hardcoded dict per tier; ordering not configurable | **Configurable backend preference chain** (§11.5) |
| 6 | Tool-incapable blocklist | `llm_discovery.py` `_TOOL_INCAPABLE_NAMES` | Deprecated static list; blocks entire model families | **Capability profiles replace blocklist** (§11.4) |
| 7 | GPU/hardware thresholds | `model_advisor.py` | Fixed RAM/VRAM breakpoints (8/16/32/64 GB) | **Runtime resource detection** (§11.6) |
| 8 | Backend detection | `backend_manager.py` | Only probes localhost default URLs | **Configurable endpoints + auto-discovery** (§11.5) |
| 9 | ~~Nano GGUF model~~ | ~~`constants.py` `NANO_GGUF_*`~~ | *(Removed by ADR-034)* | **Smallest-available from catalog** (§11.3) |
| 10 | ~~In-process engine~~ | ~~`mistral.rs`~~ | *(Removed by ADR-034 — replaced by llama_cpp `--parallel`)* | **InferenceBackend protocol** (§7.3) |
| 11 | Specialist templates | `dynamic_pack.py` `PACK_TEMPLATES` | 3 hardcoded templates with fixed role descriptions | **Template files loaded from config directory** (§11.7) |
| 12 | Profile tier features | `features.py` `PROFILE_FEATURES` | Hardcoded per tier (small/medium/large) | **Derived from actual available models** (§11.6) |

### 11.3 Externalized Model Catalog

**Problem:** `_MODEL_TABLE` hardcodes `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b` etc. When Qwen3.5-9B
releases, someone must edit Python source code. *(Note: `NANO_GGUF_*` constants removed by ADR-034.)*

**Solution:** A YAML model catalog shipped with the package but overridable via `~/.config/concierge/models.yaml`
or `CONCIERGE_MODELS_CATALOG` environment variable.

```yaml
# models.yaml — shipped defaults (overridable)
catalog_version: 1

# Role-based recommendations (replaces _MODEL_TABLE)
recommendations:
  router:
    preferred: ["qwen3.5:9b", "xlam-2:8b"]
    minimum_size_b: 3
    max_size_b: 14
  planner:
    preferred: ["qwen3.5:9b", "phi-4-reasoning:14b"]
    minimum_size_b: 7
  coder:
    preferred: ["qwen2.5-coder:32b", "qwen3-coder-next:80b"]
    fallback: ["qwen2.5-coder:7b", "qwen3.5:9b"]
  reviewer:
    preferred: ["phi-4-reasoning:14b", "deepseek-r1-distill:14b"]
    must_differ_from_doer: true
  researcher:
    preferred: ["qwen3.5:9b"]

# Nano model section removed by ADR-034 (in-process backend dropped)
# Bootstrap uses Ollama-pulled models instead of standalone GGUF files
```

**Resolution order:** User config → environment variable → shipped defaults.

**Key invariant:** The system never fails because a recommended model isn't available. Every
recommendation has a fallback chain that terminates with "use whatever is available and satisfies
the capability floor." The catalog informs *preference*; capability matching decides *eligibility*.

### 11.4 Discovery-Driven Model Profiles

**Problem:** `BUILTIN_PROFILES` has 21 hardcoded families. `get_profile()` accepts an `overrides`
parameter but no production code path ever passes overrides. New model families (GLM-4.7, Nemotron 3,
xLAM-2) require code changes to be recognized.

**Solution:** Three-tier profile resolution:

```
Tier 1: Runtime discovery    — probe model's actual capabilities (fastest, most accurate)
Tier 2: User-supplied YAML   — ~/.config/concierge/model_profiles.yaml
Tier 3: Shipped defaults      — BUILTIN_PROFILES dict (current 21 families)
Tier 4: Unknown-family default — permissive profile (tool_calling=true, all caps=0.5)
```

**Tier 1: Runtime capability probing:**

When a model is first seen (via `list_available()` on any backend), the runtime can optionally
run a lightweight probe:

```python
async def probe_model_capabilities(
    client: ChatClient, model_id: str
) -> dict[str, float]:
    """Quick probe of a model's actual capabilities.

    Runs 3-5 targeted micro-prompts (< 50 tokens each) to test:
    - tool_calling: can it produce valid function calls?
    - structured_output: can it produce valid JSON?
    - instruction_following: does it follow a specific format constraint?

    Results cached to disk (models_cache.json) to avoid re-probing.
    """
```

Probing is **optional and lazy** — it only runs when no profile exists for a model family AND the
model is actually selected for use. Results are cached to `~/.local/share/concierge/model_probes.json`
so each model is probed at most once.

**Tier 2: User-supplied profiles:**

```yaml
# ~/.config/concierge/model_profiles.yaml
profiles:
  glm-4.7-flash:
    supports_tool_calling: true
    capabilities:
      reasoning: 0.85
      code_python: 0.7
      web_comprehension: 0.7
      structured_output: 0.8
    notes: "30B/3B MoE; interleaved thinking"

  nemotron-3-nano:
    supports_tool_calling: true
    capabilities:
      reasoning: 0.7
      code_python: 0.6
      structured_output: 0.7
    notes: "30B/3B Mamba-Transformer MoE; 1M context"
```

**Profile merging:** User profiles override shipped profiles for the same family. Discovery
results override both. This means:
- New model released today? User adds 5 lines of YAML — no code change needed.
- Model capabilities change with a new quantization? Probe detects the difference.
- Shipped defaults wrong? User corrects locally.

**_TOOL_INCAPABLE_NAMES elimination:** The blocklist is replaced by `supports_tool_calling: false`
in the profile. Any model with `supports_tool_calling: false` is eligible for `consult_specialist_model`
but excluded from the tool-calling loop. No model is permanently blocked.

### 11.5 Backend Registry with Health Probing

**Problem:** `DEFAULT_BACKEND_URLS` hardcodes `{"ollama": "http://localhost:11434", ...}`. Backend
detection probes only localhost. `BACKEND_PRIORITY` hardcodes ordering per tier.

**Solution:** A backend registry that combines configuration with runtime health checks.

```yaml
# ~/.config/concierge/backends.yaml (or shipped defaults)
backends:
  ollama:
    urls: ["http://localhost:11434"]
    health_endpoint: "/api/tags"        # GET → 200 means healthy
    priority: 1                         # Lower = preferred
    env: {}                             # ROCm 7.2+ works natively on gfx1151

  llama_cpp:
    # No fixed URL — managed processes get dynamic ports
    managed: true                       # Runtime starts/stops processes
    binary: "llama-server"              # Must be in PATH
    priority: 2

  vllm:
    urls: ["http://localhost:8000"]
    health_endpoint: "/health"
    priority: 3
    platforms: ["linux-cuda", "linux-rocm-mi"]  # Only on these platforms

  mlx:
    urls: ["http://localhost:8080"]
    health_endpoint: "/health"
    priority: 1                         # Preferred on macOS
    platforms: ["darwin"]
```

**Startup sequence:**

```python
async def discover_backends(config: BackendRegistryConfig) -> list[ActiveBackend]:
    """Probe all configured backends; return those that are healthy."""
    active = []
    for name, cfg in config.backends.items():
        if cfg.platforms and current_platform() not in cfg.platforms:
            continue  # Skip backends not for this platform
        if cfg.managed:
            if shutil.which(cfg.binary):
                active.append(ManagedBackend(name, cfg))
            continue
        for url in cfg.urls:
            try:
                resp = await httpx.AsyncClient().get(
                    f"{url}{cfg.health_endpoint}", timeout=5.0
                )
                if resp.status_code == 200:
                    active.append(RemoteBackend(name, url, cfg))
                    break  # First healthy URL wins
            except httpx.ConnectError:
                continue
    return sorted(active, key=lambda b: b.priority)
```

**Key properties:**
- Adding a remote backend (e.g., a LAN server) = add a URL to `backends.yaml`. No code change.
- Backend goes down mid-run = health check fails = model runtime routes to next backend.
- New backend type (e.g., `exo`, `petals`) = implement `InferenceBackend` protocol, add to config.
- Platform-aware: vLLM config exists in defaults but is skipped on Strix Halo Linux automatically.

**Backend version detection:**

Each `InferenceBackend` implementation probes the backend's version at startup:

```python
async def detect_version(self) -> str:
    """Detect backend version for API compatibility."""
    # Ollama: parse from /api/version
    # llama.cpp: parse from --version or /health response
    # Returns semver string; stored on ActiveBackend
```

This enables version-gated behavior: e.g., `keep_alive` parameter format changed between
Ollama 0.14 and 0.17 — the backend adapter can branch on version rather than failing.

### 11.6 Runtime Resource Detection

**Problem:** `model_advisor.py` uses fixed thresholds (8/16/32/64 GB) to recommend models. This
breaks on non-standard memory configs and doesn't account for other processes using memory.

**Solution:** The Model Runtime (§7) queries actual available resources at startup and before
each `acquire()` call.

```python
@dataclass
class ResourceSnapshot:
    """Point-in-time system resource state."""
    total_ram_mb: int           # Total physical RAM
    available_ram_mb: int       # Currently available (not just free — includes reclaimable)
    gpu_devices: list[GPUDevice]  # Detected GPUs with memory info
    loaded_models: list[ModelSlot]
    estimated_free_for_models_mb: int  # available - os_reserve - loaded model total

@dataclass
class GPUDevice:
    index: int
    name: str                   # e.g. "Radeon 8060S"
    vram_total_mb: int          # Total VRAM (or GTT for APUs)
    vram_used_mb: int           # Currently used
    backend: str                # "vulkan", "rocm", "cuda", "metal"
    compute_capability: str     # e.g. "gfx1151", "sm_89"
```

**Sources for resource data:**

| Platform | RAM | GPU |
|----------|-----|-----|
| Linux | `/proc/meminfo` (MemAvailable) | `vulkaninfo`, ROCm `rocm-smi`, backend's own reporting |
| macOS | `sysctl hw.memsize` + `vm_stat` | Metal via `system_profiler SPDisplaysDataType` |

**How this replaces hardcoded tiers:**

Instead of:
```python
# v1: hardcoded thresholds
if ram_gb >= 64: tier = "large"
elif ram_gb >= 32: tier = "medium"
else: tier = "small"
```

v2 uses:
```python
# v2: actual availability drives decisions
snapshot = await runtime.resource_snapshot()
available_mb = snapshot.estimated_free_for_models_mb

# Model catalog recommendations filtered by what actually fits
candidates = [m for m in catalog.recommendations[role]
              if m.estimated_size_mb <= available_mb]
```

No tiers. No thresholds. The system works with whatever resources are actually available
right now.

**Ollama APU VRAM misreporting workaround:** Ollama reports only fixed VRAM (512 MB) for APUs,
ignoring GTT (~108 GB). The OllamaBackend implementation includes a correction:

```python
async def _corrected_memory(self) -> int:
    """Work around Ollama APU VRAM misreporting (ollama #12062)."""
    ps_response = await self._get("/api/ps")
    # If reported total < 4 GB and system has > 32 GB RAM,
    # assume APU with shared memory — use system RAM as budget
    if reported_vram_mb < 4096 and system_ram_mb > 32768:
        return int(system_ram_mb * 0.85)  # Reserve 15% for OS
    return reported_vram_mb
```

### 11.7 Externalized Specialist Templates

**Problem:** `PACK_TEMPLATES` in `dynamic_pack.py` hardcodes 3 templates (`engineering`,
`research`, `enterprise_research`) with fixed role descriptions and tool lists. Adding a
new specialist type requires a code change.

**Solution:** Templates loaded from a config directory, with shipped defaults.

```yaml
# ~/.config/concierge/templates/engineering.yaml (or shipped defaults)
template_id: engineering
role: "Software engineer"
description: "Writes, tests, and debugs code"
tools:
  - shell
  - read_file
  - write_file
  - list_files
  - run_tests
finish_schema: code
default_capabilities:
  - code_python
  - code_rust
quality_gates:
  - tests_pass
  - no_syntax_errors
```

**Resolution:** Templates from `~/.config/concierge/templates/` override shipped defaults.
The dynamic pack builder is indifferent to the source — it receives a `PackTemplate` dataclass
regardless.

**Adding a new specialist:** Drop a YAML file. No code change, no restart needed (templates
reloaded on each run).

### 11.8 Hardware-Adaptive Backend Configuration

**Problem:** Different hardware may need different backend configuration. Apple Silicon
needs nothing special, NVIDIA needs CUDA paths, AMD APUs work with ROCm or Vulkan.
Currently this is documented but not automated.

**Solution:** Platform detection with automatic environment configuration:

```python
def detect_platform() -> PlatformProfile:
    """Detect hardware and return appropriate configuration."""
    gpu = detect_gpu()  # Parses lspci, system_profiler, etc.

    if gpu.vendor == "AMD" and gpu.arch == "RDNA3.5":
        return PlatformProfile(
            name="strix-halo",
            backend_env={},
            preferred_backends=["vllm", "ollama", "llama_cpp"],
            notes="ROCm 7.2+ supported via gfx11-generic",
        )
    elif gpu.vendor == "Apple":
        return PlatformProfile(
            name="apple-silicon",
            preferred_backends=["mlx", "ollama"],
        )
    elif gpu.vendor == "NVIDIA":
        return PlatformProfile(
            name="nvidia-cuda",
            preferred_backends=["vllm", "ollama", "llama_cpp"],
        )
    else:
        return PlatformProfile(
            name="generic",
            preferred_backends=["ollama"],
        )
```

**Overridable:** `~/.config/concierge/platform.yaml` or `CONCIERGE_PLATFORM` env var.
Auto-detection is a sensible default; explicit configuration is always preferred.

### 11.9 Ecosystem Change Response Playbook

This table shows how the system responds to each category of ecosystem change **without code modifications**:

| Change | Response | Who acts | Time to adopt |
|--------|----------|----------|---------------|
| **New model family released** (e.g., Llama 5) | Add profile to `model_profiles.yaml`; add to recommendations in `models.yaml` | User or maintainer | Minutes (config edit) |
| **Model gains/loses tool calling** | Update `supports_tool_calling` in profile | User or maintainer | Minutes |
| **New backend released** (e.g., `exo v2`) | Implement `InferenceBackend` protocol; add to `backends.yaml` | Developer | Hours (one new file) |
| **Backend API changes** (e.g., Ollama v1.0) | Version-detect in backend adapter; branch on version | Developer | Hours (adapter update) |
| **New GPU architecture** | Add to `detect_platform()`; add platform profile | Developer | Minutes to hours |
| **GPU driver fix** (e.g., ROCm works on gfx1151) | Update `platform.yaml` — remove Vulkan workaround | User | Minutes |
| **New quantization format** | Backend handles natively (GGUF additions are backward-compatible) | None | Zero |
| **Memory config changes** | Runtime resource detection adapts automatically | None | Zero |
| **New tool needed** | Add `ToolEntry` to catalog + executor function | Developer | Hours (one new entry) |
| **Specialist workflow changes** | Edit template YAML | User or maintainer | Minutes |

### 11.10 Health Monitoring and Self-Healing

Beyond startup discovery, the system monitors backend health continuously:

```python
class HealthMonitor:
    """Periodic backend health checks with automatic failover."""

    async def check_all(self) -> list[BackendHealth]:
        """Probe all active backends; mark unhealthy ones for failover."""
        ...

    async def on_backend_failure(self, backend: str, error: Exception) -> None:
        """Called when a backend request fails.

        1. Mark backend as degraded
        2. If model was loaded there, attempt re-acquire on another backend
        3. Emit health_degraded event to runlog
        4. After N consecutive failures, mark backend as down
        5. Periodically retry downed backends (exponential backoff)
        """
```

**Self-healing scenarios:**

| Scenario | Detection | Response |
|----------|-----------|----------|
| Ollama crashes mid-run | `acquire()` timeout or connection refused | Re-discover backends; fall back to llama.cpp if available |
| Model OOM during inference | Backend returns error or process killed | Evict other models; retry with smaller quantization if available |
| GPU driver hang | Backend stops responding | Kill stuck process; restart backend; fall back to CPU |
| Network backend unreachable | Health check fails | Remove from active set; retry in background |

### 11.11 Configuration Hierarchy Summary

All externalized configuration follows a consistent hierarchy:

```
Priority (highest → lowest):
  1. Environment variables    (CONCIERGE_*)
  2. CLI flags                (--backend-url, --model, etc.)
  3. User config directory    (~/.config/concierge/)
  4. Project config           (.concierge.yaml in workspace)
  5. Shipped defaults         (bundled in package)
```

| Config file | Purpose | Shipped default? |
|-------------|---------|-----------------|
| `models.yaml` | Model recommendations per role | Yes |
| `model_profiles.yaml` | Capability profiles per model family | Yes (as Python dict; YAML for overrides) |
| `backends.yaml` | Backend URLs, priorities, platform filters | Yes |
| `platform.yaml` | Hardware-specific env vars and preferences | Auto-detected; overridable |
| `templates/*.yaml` | Specialist pack templates | Yes (3 shipped) |

### 11.12 Testing Resilience

Resilience is tested, not hoped for:

```python
# test_backend_failover.py
async def test_acquire_falls_back_when_primary_backend_down():
    """When Ollama is down, acquire() falls back to llama.cpp."""

async def test_acquire_with_unknown_model_family():
    """Unknown model family gets default profile; acquire succeeds."""

async def test_model_catalog_override():
    """User models.yaml overrides shipped recommendations."""

async def test_backend_health_recovery():
    """After backend recovers from failure, it re-enters the active set."""

# test_platform_detection.py
def test_strix_halo_detected():
    """Strix Halo GPU detected → Vulkan env vars set."""

def test_unknown_gpu_falls_back_to_generic():
    """Unknown GPU → generic platform profile → Ollama default."""

# test_profile_discovery.py
async def test_probed_profile_overrides_shipped():
    """Runtime probe results take precedence over BUILTIN_PROFILES."""

async def test_user_profile_overrides_shipped():
    """User YAML profiles override BUILTIN_PROFILES."""
```

---

## 12. Migration Strategy

### 12.1 Approach: Evolutionary, Not Rewrite

Despite the title "ground-up redesign", the recommended approach is **evolutionary** — add the three layers incrementally while keeping all 875 tests passing.

**Rationale:**
- The hexagonal architecture is solid — ports and adapters allow swapping internals
- The tool system, sandbox, quality gates, and runlog work well
- The specialist pack composition is flexible
- A full rewrite would lose 14 phases of battle-tested edge cases

### 12.2 Implementation Phases

#### Phase V2-A: Externalized Configuration + Model Runtime (Foundation)

**Goal:** Externalize all hardcoded assumptions (§11) AND replace `resolve_llm()` + `build_chat_client()` with `ModelRuntime.acquire()/release()`.

These are combined into one phase because the model runtime needs the externalized config to avoid
re-introducing hardcoded assumptions.

1. **Externalized configuration layer:**
   - Create `config/loader.py` — YAML config loader with resolution hierarchy (env → CLI → user → project → shipped)
   - Create shipped defaults: `config/defaults/models.yaml`, `model_profiles.yaml`, `backends.yaml`
   - Create `config/platform.py` — hardware/platform detection → `PlatformProfile`
   - Migrate `_MODEL_TABLE`, `DEFAULT_BACKEND_URLS`, `BACKEND_PRIORITY`, `PROFILE_FEATURES` to YAML defaults *(NANO_GGUF_* removed by ADR-034)*
   - Remove `_TOOL_INCAPABLE_NAMES`; replace with `supports_tool_calling` in profiles
   - Migrate `PACK_TEMPLATES` to `config/defaults/templates/*.yaml`

2. **Model Runtime:**
   - Create `ModelRuntime` protocol in `application/ports.py`
   - Implement `LocalModelRuntime` in `infrastructure/model_runtime/`
     - Wraps Ollama backend initially (using config from `backends.yaml`)
     - Adds acquire/release with refcounting
     - Adds resource tracking (runtime detection, not hardcoded tiers)
     - Adds eviction policy (LRU when memory needed)
   - Add `ModelHandle` context manager pattern
   - Wire into `execute_task.py` — replace `build_chat_client()` calls with `runtime.acquire()`
   - All existing tests pass with `MockModelRuntime` that returns mock handles

3. **Backend health probing:**
   - `discover_backends()` at startup — probe all configured backends, return healthy ones
   - Version detection per backend — enables version-gated behavior
   - Health monitor with failover (mark degraded → retry with backoff)

**Backward compatibility:** `resolve_llm()` becomes a convenience wrapper. Zero-config path (no YAML files) uses shipped defaults — identical to v1 behavior.

**Tests:** ~30 new tests covering config loading, platform detection, backend discovery, model runtime acquire/release/evict, health monitoring, failover.

#### Phase V2-B: Multi-Backend Support + Capability Probing

**Goal:** Model runtime manages multiple backends simultaneously. Unknown model families
get probed for capabilities.

1. Add `InferenceBackend` protocol
2. Implement `OllamaBackend`, `LlamaCppBackend` (§7.3)
3. Model runtime routes to appropriate backend based on model and resource state
4. llama.cpp backend manages server processes (start/stop = load/unload)
5. **Capability probing** for unknown model families (§11.4 Tier 1)
   - Micro-prompt probes (< 50 tokens each) for tool calling, structured output, instruction following
   - Results cached to `~/.local/share/concierge/model_probes.json`
   - Lazy: only probes when model selected for use and no profile exists

#### Phase V2-C: Recursive Task Decomposition

**Goal:** Replace flat `OrchestrationPlan` with `TaskGraph`.

1. Create `TaskGraph`, `TaskNode` in `application/task_graph.py`
2. Create planner and critic agents (reuse tool-calling loop infrastructure)
3. Implement adaptive depth control
4. Wire into `execute_task.py` — `TaskGraph` replaces `OrchestrationPlan` as the execution plan
5. Parallel leaf execution reuses existing `asyncio.gather` pattern

**Backward compatibility:** Simple tasks (single leaf) produce identical behaviour to current system.

#### Phase V2-D: Agent-Model Affinity

**Goal:** Different agent roles use different models, selected from externalized recommendations.

1. Load `AGENT_ROLES` from `models.yaml` recommendations (not hardcoded dict)
2. Implement `assign_model()` using model runtime + model catalog
3. Add `must_differ_from` constraint for reviewer
4. Thread model assignment through task graph execution

#### Phase V2-E: Preloading & Optimization

**Goal:** Anticipate model needs from task graph; preload while current step executes.

1. Implement `preload_hint()` on model runtime
2. Task graph scheduler calls preload for upcoming nodes
3. Background loading with progress events
4. Add `concierge doctor` command — validates config, probes backends, checks model availability

### 12.3 What Gets Replaced vs. Kept

| Component | Action | Rationale |
|-----------|--------|-----------|
| `resolve_llm()` | Wrapped by ModelRuntime | Still useful as a discovery helper |
| `build_chat_client()` | Kept; called by backend implementations | Factory pattern still needed |
| `_execute_pack_loop()` | Kept with minor changes | Core tool-calling loop is solid |
| `orchestrate_task()` | Replaced by TaskGraph planner | New decomposition strategy |
| `_review_specialist_work()` | Kept; reviewer gets its own model | Review logic is good; model selection changes |
| `model_profiles.py` | **Externalized to YAML** + shipped defaults | Profiles loadable from config; `BUILTIN_PROFILES` becomes shipped default |
| `model_advisor.py` `_MODEL_TABLE` | **Replaced by `models.yaml`** | Recommendations externalized; no code change needed for new models |
| ~~`constants.py` `NANO_GGUF_*`~~ | *(Removed by ADR-034)* | In-process backend dropped; llama_cpp uses Ollama-managed models |
| `constants.py` `DEFAULT_BACKEND_URLS` | **Replaced by `backends.yaml`** | URLs, priorities, platform filters all in config |
| `features.py` `BACKEND_PRIORITY` | **Replaced by `backends.yaml` priorities** | Ordering configurable; platform-aware |
| `features.py` `PROFILE_FEATURES` | **Replaced by runtime resource detection** | No tiers; actual availability drives decisions |
| `llm_discovery.py` `_TOOL_INCAPABLE_NAMES` | **Deleted** | Replaced by `supports_tool_calling` in profiles |
| `dynamic_pack.py` `PACK_TEMPLATES` | **Externalized to `templates/*.yaml`** | Loaded from config directory; new specialists = new YAML file |
| `tool_catalog.py` | Kept; add `think`, `ask_user`, `search_codebase` | Solid foundation |
| `prompts.py` | Extended with planner/critic prompts | Composable fragment system works |
| `VLLMChatClient` | Active | ROCm 7.2+ supports gfx1151; first-class vLLM backend on Strix Halo |
| Rust launcher | Kept; platform detection applies backend env vars | No special env vars needed for Strix Halo (ROCm works natively) |
| Quality gates 1–4 | Kept | Battle-tested |
| Runlog + checkpointing | Extended for task graph | Add task_graph events |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **ROCm APU memory underutilisation** | Known | Medium | ROCm may allocate only VRAM (not GTT) on APUs, ~60% slower than Vulkan for prompt processing; monitor ROCm releases for unified memory fixes |
| **vLLM ROCm maturity on consumer GPUs** | Medium | Medium | 93% CI pass rate (Jan 2026); pre-built Docker images available; gfx1151 supported via gfx11-generic ISA |
| **Multiple backend options on Strix Halo** | N/A | Low | All three backends (vLLM, Ollama, llama.cpp) work; platform profile lists all as preferred |
| Small models can't plan recursively | Low (revised) | High | Qwen3.5-9B has unprecedented reasoning for 9B; adaptive depth; fallback to flat decomposition |
| Multi-model concurrency causes OOM | Low | High | Resource tracker; Ollama's `OLLAMA_MAX_LOADED_MODELS`; eviction via `keep_alive: 0` |
| Model swap latency hurts UX | Medium | Medium | Preloading from task graph; `keep_alive: "-1"` for hot models |
| MoE memory vs performance confusion | Medium | Medium | All MoE params in memory but only fraction active; memory budget must account for total params |
| Critic disagrees with planner endlessly | Low | Medium | Max 2 critique rounds; then proceed with warning |
| Migration breaks existing tests | Low | High | Incremental phases; all 875 tests pass at every phase boundary |
| Ollama APU VRAM misreporting | Confirmed | Medium | May need config override for GPU memory allocation; ollama #12062 |
| Complexity increase for diminishing returns | Medium | Medium | Phase V2-A alone provides significant value; later phases are optional |
| **Configuration drift** — user config diverges from reality | Medium | Medium | Startup validation warns on invalid model names, unreachable URLs; `concierge doctor` command validates config |
| **Probe overhead** — capability probing adds startup latency | Low | Low | Probes are lazy (only unknown families) + cached to disk; typical startup adds 0s |
| **Over-configuration** — too many YAML files overwhelm users | Medium | Medium | Zero config works (shipped defaults); YAML only needed when customising; `concierge init` generates annotated templates |
| **Backend version regression** — adapter assumes newer API | Low | High | Version detection at startup; adapter branches on version; tests cover oldest supported version |
| **Discovery returns stale data** — model unloaded between inventory and acquire | Medium | Low | Acquire re-checks; retry with eviction if model disappeared; idempotent load |

---

## 14. Verification Plan

### Per-Phase Gates

Each phase must pass before proceeding:

1. `make test` — all existing tests pass (875+ at start)
2. `ruff check` — clean
3. `cargo test --manifest-path launcher/Cargo.toml` — 22 pass
4. New tests for new functionality (target: 20+ per phase)
5. Manual smoke test with real models

### End-to-End Verification Scenarios

| Scenario | What it tests |
|----------|--------------|
| `concierge run "What's KO stock price?"` | Quick answer path; no decomposition; small model |
| `concierge run "Build a CRUD todo app with tests"` | Recursive decomposition; coder model; run_tests quality gate |
| `concierge run "Compare KO and BAC dividends"` | Research + synthesis; web tools; reviewer with different model |
| `concierge run "Write a SQL query for top customers"` | Consult non-tool-calling model (sqlcoder) |
| Two concurrent runs | Model runtime manages shared resources; no OOM |

### Resource Monitoring

During verification, monitor:
- VRAM usage over time (should see load/unload cycles)
- Model swap latency (target: < 5s for 7B Q4)
- Total tokens/second across concurrent models
- Memory headroom (should never exceed 90% of available)

---

## Appendix A: Ollama Model Management API (v0.17.7, March 2026)

**Architecture note:** Ollama v0.17.0 (Feb 21, 2026) replaced the llama.cpp server mode with a new custom "Ollama engine" — 40% faster prompt processing on NVIDIA, KV cache 8-bit quantization, improved tensor parallelism. The API is unchanged but the internals are significantly different.

Ollama exposes these relevant endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/tags` | GET | List downloaded models |
| `/api/ps` | GET | List currently loaded models (shows VRAM usage) |
| `/api/generate` | POST | Generate text (forces model load if not loaded) |
| `/api/chat` | POST | Chat completion (forces model load) |
| `/api/pull` | POST | Download a model |
| `/api/delete` | DELETE | Delete a model from disk |

**Lifecycle control (more capable than initially assumed):**

| Operation | Method | Notes |
|-----------|--------|-------|
| **Preload** | `POST /api/generate` with empty prompt + model name | Forces model into memory without generating |
| **Unload** | Any request with `keep_alive: "0"` | Immediately unloads after response |
| **Pin indefinitely** | Any request with `keep_alive: "-1"` | Never auto-unloads |
| **Global idle timeout** | `OLLAMA_KEEP_ALIVE` env var | Default 5 minutes |
| **Max concurrent models** | `OLLAMA_MAX_LOADED_MODELS` env var | Default 3 per GPU (or 3 for CPU) |
| **Parallel requests per model** | `OLLAMA_NUM_PARALLEL` env var | Default 1; RAM scales with N × context |
| **Request queue depth** | `OLLAMA_MAX_QUEUE` env var | Default 512; rejects when full |

**Key insight:** Ollama supports **explicit model lifecycle management** sufficient for a model runtime:
1. Use `GET /api/ps` to inventory loaded models + memory usage
2. Use empty `POST /api/generate` to preload anticipated models
3. Use `keep_alive: "-1"` to pin specialist models that should stay hot
4. Use `keep_alive: "0"` to unload models when their refcount drops to zero
5. `OLLAMA_MAX_LOADED_MODELS` controls concurrency ceiling

This eliminates the need for managing separate llama.cpp server processes in most scenarios.

## Appendix B: llama.cpp Server as Managed Backend

llama.cpp's `llama-server` (formerly `server`) can be managed as a subprocess:

```bash
# Start with explicit GPU layers and memory budget
llama-server \
    --model /path/to/model.gguf \
    --n-gpu-layers 999 \          # Full GPU offload
    --ctx-size 8192 \             # Context window
    --port 8081 \                 # Unique port per model
    --host 127.0.0.1

# OpenAI-compatible API at http://127.0.0.1:8081/v1/chat/completions
```

**Advantages over Ollama for managed deployment:**
- Explicit model assignment per process (no opaque scheduling)
- Full control over GPU layer offloading
- Known memory footprint per process
- Clean start/stop = load/unload

**Model runtime integration:** Start a llama-server process per pinned model. Track PID + port. Kill process to unload. Use `GenericChatClient` (already implemented) to talk to it.

## Appendix C: Key Technical References (verified March 9, 2026)

### Infrastructure & Backends

| Topic | Source | Key finding |
|-------|--------|-------------|
| Ollama v0.17.7 | github.com/ollama/ollama/releases | New engine (not llama.cpp server); 40% faster prompt; KV cache 8-bit quant |
| Ollama HIP broken on gfx1151 | ollama #13589 (open) | VM fault in libhsa-runtime64.so; must use Vulkan |
| Ollama APU VRAM reporting | ollama #12062 | Reads only fixed VRAM (512MB), ignores GTT (~108GB); GPU flagged "too small" |
| llama.cpp b8248 | github.com/ggml-org/llama.cpp/releases | 11 backends; daily releases; MXFP4 quantization; GPU token sampling |
| llama.cpp KV cache bug | llama.cpp #18011 (closed NOT_PLANNED) | Driver issue; KV cache in shared memory on gfx1151; no fix coming |
| llama.cpp slow loading >64GB | llama.cpp #15018 (closed) | Workaround: `--no-mmap` flag |
| vLLM gfx1151 NOT supported | vllm #22644 (closed "not planned") | Community workarounds only; not viable for production |
| vLLM v0.17.0 | github.com/vllm-project/vllm/releases | FlashAttention 4; pipeline parallelism; ROCm = datacenter MI only |
| ROCm 7.2.0 stable | github.com/ROCm/ROCm/releases | gfx1151 via HSA_OVERRIDE; ROCm 7.9.0 preview = official gfx1151 |
| ROCm 7.9.0 preview | ROCm docs + Phoronix | Official Strix Halo support; Linux kernel >= 6.18.4 required |
| MLX v0.31.0 | github.com/ml-explore/mlx/releases | Now supports CUDA (NVIDIA); no longer Apple-only |
| vllm-mlx v0.2.6 | pypi.org/project/vllm-mlx | 400+ tok/s; MCP tool calling; Anthropic API compat |
| Vulkan vs ROCm (Phoronix) | Phoronix ROCm 7.1 review | ROCm faster for long prompts; Vulkan faster for short generation |

### Model Landscape

| Topic | Source | Key finding |
|-------|--------|-------------|
| Qwen3.5-9B | VentureBeat; HuggingFace | 66.1 BFCL-V4; 79.1 TAU2; 81.7 GPQA Diamond; beats GPT-OSS-120B |
| Qwen3-Coder-Next | HuggingFace; Ollama library | 80B/3B MoE (512 experts); SWE-Bench-Pro competitive |
| Phi-4-reasoning-14B | HuggingFace; SiliconANGLE | Approaches full DeepSeek R1 (671B); outperforms R1-Distill-70B |
| GLM-4.7-Flash | Zhipu AI; Unsloth docs | 30B/3B MoE; interleaved thinking; 87.4 tau2-Bench (highest open-source) |
| xLAM-2-8b-fc-r | Salesforce; Berkeley BFCL V4 | SOTA function calling; outperforms GPT-4o |
| Nemotron 3 Nano | NVIDIA Newsroom | 30B/3B hybrid Mamba-Transformer MoE; 1M context |
| GGUF still dominant | Community consensus | No challenger; MXFP4 new quant type; Unsloth dynamic quants |
| MoE architecture dominance | Qwen3.5, Llama 4, GLM-4.7, Nemotron 3 | All frontier models are MoE; dense models for specialists only |
