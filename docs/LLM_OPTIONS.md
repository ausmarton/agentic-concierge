# LLM options for the agent fabric: comprehensive reference

This document is a **critical review of every practical way** LLMs can run for the agent fabric. Use it when choosing or changing how the fabric uses LLMs: deployment, backend choice, where inference runs, and trade-offs.

**Related:** [BACKENDS.md](BACKENDS.md) (we are not locked to Ollama; deployment philosophy). [VISION.md](VISION.md) (local-first, quality, portability).

> **Note (ADR-034, 2026-03-10):** The "in-process" option (§5) was evaluated and implemented
> as `InProcessChatClient` via mistral.rs/PyO3, then **removed** in favour of managed
> `llama-server` processes (`LlamaCppBackend` with `--parallel N`). The section below is
> retained for reference but is no longer an active deployment option. Current backends:
> **Ollama**, **llama_cpp** (managed llama-server), **vLLM**, **cloud/generic**.

---

## 1. What the fabric needs from an LLM

The fabric depends only on the **OpenAI chat-completions API**:

- **Endpoint:** `POST {base_url}/chat/completions` (typically `base_url` ends in `/v1`).
- **Request:** JSON with `model`, `messages`, `temperature`, `top_p`, `max_tokens`.
- **Response:** JSON with `choices[0].message.content` (and optionally `choices[0].message.tool_calls` for tool use).
- **Auth:** Optional `Authorization: Bearer <api_key>` (empty for local).

So any backend that exposes this contract works. The options below differ in **where** and **how** that endpoint is provided, not in the API shape.

---

## 2. Dimensions that matter

| Dimension | Choices | Why it matters for the fabric |
|-----------|---------|-------------------------------|
| **Execution model** | Separate server process / In-process library / Remote API | Affects startup, teardown, resource isolation, and performance. |
| **Location** | On-host (native) / In container (same machine) / Remote (cloud or other host) | Affects latency, data locality, repeatable deployment, and “local-first” alignment. |
| **Deployment** | Manual install / Bootstrap script / Docker–compose / K8s–Helm / Managed cloud | Affects “one command to same state” and clean teardown. |
| **Hardware** | CPU only / Single GPU (NVIDIA / AMD / Apple) / Multi-GPU | Affects throughput, model size, and which backends are viable. |
| **Who runs it** | User starts server / Fabric starts server (`local_llm_start_cmd`) / Orchestrator (K8s, cloud) | Affects default UX and clean teardown (no leftover processes). |

The following sections group options by **how** the LLM runs (execution model and location), then summarize **advantages**, **disadvantages**, and **when it’s a good fit** for this project.

---

## 3. Option categories (overview)

1. **Local, separate server (on-host)** — e.g. Ollama, llama-server, vLLM, LiteLLM. Process(es) on the same machine as the fabric; fabric talks to them via HTTP.
2. **Local, in-process** — e.g. llama-cpp-python, Hugging Face Transformers in the same process as the fabric. No separate server; library loads the model in the fabric process.
3. **Local, in container** — Same as (1) or (2), but the server or the process runs inside Docker/Podman (or similar) on the same host.
4. **Remote API (cloud or other host)** — OpenAI, Azure OpenAI, Anthropic, Google, or a self-hosted server on another machine. Fabric only needs `base_url` + optional `api_key`.
5. **Orchestrated / managed** — LLM runs in Kubernetes, serverless, or a managed service; fabric still talks to an HTTP endpoint.

Below we go through each in detail.

---

### 3.1 Which local option is the most performant?

**Short answer:** It depends on **what you optimize for** and **your hardware**; there is no single “fastest” for every case.

| Metric / workload | Most performant local option | Why |
|-------------------|------------------------------|-----|
| **Throughput (tokens/sec, many requests)** | **vLLM** (native on host, NVIDIA GPU) | Continuous batching, PagedAttention, built for high QPS. |
| **Latency (time to first token, single request)** | **llama.cpp server** or **in-process llama-cpp-python** (native) | Minimal stack, no extra HTTP/process overhead; tuned C++ core. Ollama is in the same ballpark for typical single-user use. |
| **Performance per watt / CPU or non-NVIDIA** | **llama.cpp** (server or in-process) | Very efficient C++ implementation; good CPU and AMD/Apple builds. Ollama uses llama.cpp under the hood on some paths and is close. |
| **Ease + good performance (single user, one agent)** | **Ollama** (native on host) | One binary, auto hardware detection, “good enough” speed for interactive use; not necessarily the absolute maximum. |
| **Multi-GPU / very large models** | **vLLM** (or TGI/OpenLLM) on host | Designed for batching and multi-GPU; Ollama and llama.cpp are single-node/single-process focused. |

**Practical takeaway for the fabric:** For **single-user, single-agent** use (our default), **Ollama** or **llama.cpp server**, both **native on the host**, are the best balance of performance and simplicity. If you need **maximum throughput** or **multi-GPU**, **vLLM** (native, not in a container) is the most performant local option. **Avoid running the LLM inside a container** if you care about top speed (GPU passthrough and memory overhead). In-process (**llama-cpp-python**) can be slightly faster than a local server (no HTTP round-trip) but uses more memory in the fabric process and blocks unless offloaded to a thread.

---

## 4. Local, separate server (on-host, native)

**What it is:** A dedicated LLM server process (or daemon) runs on the same machine as the fabric. The fabric calls `http://localhost:<port>/v1/chat/completions` (or a custom host/port). The server is started by the user, by the OS (systemd), or by the fabric via `local_llm_start_cmd`.

**Examples:** Ollama, llama.cpp’s `llama-server` (or `llama-cli --server`), vLLM server, LiteLLM server, Text Generation Inference (TGI), OpenLLM server.

### 4.1 Ollama

| Aspect | Details |
|--------|---------|
| **How it runs** | Single binary; `ollama serve` listens on 11434; models pulled on demand (`ollama pull <name>`). |
| **Where** | On-host, native (no container required). |
| **Hardware** | CPU, NVIDIA GPU, AMD (ROCm/Vulkan), Apple Metal. Auto-detects. |
| **Deployment** | Install per OS (script, .deb, .rpm, .dmg, Windows installer); then `ollama serve` and `ollama pull`. Not yet “one command” from clone. |

**Advantages**

- Simple UX: one binary, built-in OpenAI-compatible `/v1` endpoint.
- Cross-platform (Linux, macOS, Windows); works on CPU for small models.
- Aligns with VISION: Fedora/Linux, AMD-friendly (Vulkan/ROCm).
- Fabric can start it when unreachable (`local_llm_start_cmd: ["ollama", "serve"]`).
- No Python/CUDA version lock-in for the server; separate from concierge runtime.

**Disadvantages**

- Install and model pull are **not** part of the repo’s “one command”; they are OS-specific and manual unless wrapped in a bootstrap.
- We do not control release cadence or model format (Ollama’s own).
- Single-node; no built-in multi-GPU or distributed inference.

**Good fit for the fabric when**

- You want the **default** local backend with minimal config.
- You care about **local-first**, **portability**, and **AMD/CPU** support.
- You are okay with a **bootstrap script** (or manual steps) for “same state on every machine” and do not need the LLM inside a container.

---

### 4.2 llama.cpp server (llama-server / llama-cli --server)

| Aspect | Details |
|--------|---------|
| **How it runs** | Standalone C++ server; you build or download a binary; run with `--host` / `--port`; serve one or more GGUF models. |
| **Where** | On-host, native. |
| **Hardware** | CPU, NVIDIA (CUDA), AMD (ROCm), Apple (Metal); build-time or runtime flags. |
| **Deployment** | Download OS-specific binary or build from source; point to GGUF files. Can be scripted in a bootstrap. |

**Advantages**

- **Full control** over build (CPU/CUDA/ROCm/Metal) and model files (GGUF).
- No separate “pull” service; you own the model files and binaries.
- Very good performance per watt and broad hardware support.
- OpenAI-compatible endpoints available in many builds; fabric just sets `base_url` and `model`.

**Disadvantages**

- You must obtain GGUF models and (often) manage binaries per OS; no single “ollama pull” UX.
- Server process management (start/stop) is your responsibility unless wrapped in bootstrap or systemd.
- Multiple models = multiple processes or one server with multiple model paths, depending on build.

**Good fit for the fabric when**

- You want **maximum performance** and **control** on a known platform (e.g. Linux + AMD GPU).
- You are willing to maintain a **bootstrap** that fetches the right binary and (optionally) a default GGUF.
- You prefer **no dependency on Ollama** and are fine with GGUF as the model format.

---

### 4.3 vLLM server

| Aspect | Details |
|--------|---------|
| **How it runs** | Python server; typically `python -m vllm.entrypoints.openai.api_server` with model path or Hugging Face ID. |
| **Where** | On-host native or in container. Often used in GPU clusters. |
| **Hardware** | Strongly GPU-oriented (NVIDIA); CPU possible but not the main use case. |
| **Deployment** | pip/venv or container; model download from Hugging Face or local path. |

**Advantages**

- High throughput (continuous batching, PagedAttention); good for **batch or high-QPS** workloads.
- OpenAI-compatible API; supports many Hugging Face models.
- Fits multi-GPU and cluster setups.

**Disadvantages**

- Heavier and more GPU-centric than Ollama/llama.cpp; overkill for “one user, one agent” on a laptop.
- Python/CUDA dependency matrix; version alignment with the rest of the stack can be painful.
- Less “one binary, one command” than Ollama; usually deployed via venv or container.

**Good fit for the fabric when**

- You need **high throughput** or **larger models** on one or more NVIDIA GPUs.
- You already run vLLM (e.g. in a lab or cluster) and want the fabric to call it; set `base_url` to that server.
- You are **not** optimizing for the smallest default footprint or the simplest “run on any machine” story.

---

### 4.4 LiteLLM (proxy/server)

| Aspect | Details |
|--------|---------|
| **How it runs** | Proxy that translates to many backends (OpenAI, Anthropic, Ollama, llama.cpp, vLLM, etc.). Single process. |
| **Where** | On-host or in container; can sit in front of local or remote backends. |
| **Hardware** | Depends on the underlying backend(s). |
| **Deployment** | pip; config file or env for which backend to use. |

**Advantages**

- **One endpoint** for the fabric; you can switch backends (local Ollama vs cloud) via config without changing fabric code.
- Handles API key and format differences across providers.
- Useful for **fallback** (try local, then cloud) or **A/B** by model.

**Disadvantages**

- Extra hop and process; slight latency and operational complexity.
- You still need to run and manage the underlying LLM(s); LiteLLM only routes.

**Good fit for the fabric when**

- You want **multiple backends** or **fallback** (local → cloud) behind a single `base_url`.
- You are okay with an extra component and config in the stack.

---

### 4.5 Other local servers (TGI, OpenLLM, etc.)

**Text Generation Inference (TGI)** and **OpenLLM** are server processes that expose OpenAI-compatible or similar APIs. They are typically **container-first** and **GPU-oriented**, with strong Hugging Face integration. Trade-offs are similar to vLLM: higher throughput and flexibility, more moving parts and heavier deployment. Good when you already standardize on Hugging Face or need their specific features (e.g. adapters, multi-GPU); less ideal as the “simplest” default for a single-machine fabric.

---

## 5. Local, in-process

**What it is:** The fabric process loads the model via a Python (or other) library and runs inference in the same process. No separate HTTP server; the ChatClient implementation would call the library directly (or a tiny in-process wrapper that still implements the same `chat(...)` port).

**Examples:** llama-cpp-python (GGUF), Hugging Face Transformers (with PyTorch/Flax), other Python bindings to onnxruntime, etc.

### 5.1 llama-cpp-python

| Aspect | Details |
|--------|---------|
| **How it runs** | Python package; loads a GGUF file; inference in the fabric process (or a subprocess you manage). |
| **Where** | Same process as the fabric (or same machine, same memory space). |
| **Hardware** | CPU, CUDA, ROCm, Metal via package variants. |
| **Deployment** | `pip install llama-cpp-python` (plus optional GPU wheels); download GGUF; point config to model path. Can be “one command” if we add a ChatClient that loads GGUF and a default model URL. |

**Advantages**

- **No separate server**; no port, no “is the server up?” — simplifies bootstrap and teardown.
- **Same-process** latency; no HTTP round-trip.
- Model format (GGUF) and hardware support similar to llama.cpp server; you control the binary and model files.
- Can be packaged so that `pip install` + first-run model download gives “clone and run” without Docker.

**Disadvantages**

- **Memory**: model lives in the fabric process; large models increase RAM and can compete with tools/workspace.
- **Blocking**: inference blocks the event loop unless run in a thread/process pool; need to design for async.
- **Versioning**: Python wheel and CUDA/ROCm versions must align with the rest of the stack.
- Process crash takes down both fabric and model; no isolation.

**Good fit for the fabric when**

- You want **true “one command”** (e.g. `pip install` + optional model download) without a separate server or container.
- You run **smaller models** and are okay with **higher memory** in the fabric process.
- You value **simplest possible** local path and can accept the in-process and lifecycle trade-offs.

---

### 5.2 Hugging Face Transformers (and similar)

| Aspect | Details |
|--------|---------|
| **How it runs** | Python library; loads PyTorch/Flax/JAX models; inference in-process. |
| **Where** | Same process as the fabric. |
| **Hardware** | CPU, NVIDIA GPU (CUDA), Apple (MPS); AMD support varies. |
| **Deployment** | pip; large dependencies (PyTorch, etc.); model download from Hugging Face. |

**Advantages**

- Huge model ecosystem; easy to try new models and adapters.
- No separate server; single process.

**Disadvantages**

- **Heavy** dependencies and **memory**; slower cold start and larger disk footprint.
- Less aligned with “lightweight, AMD-friendly” default; CUDA-centric in practice.
- Need a thin ChatClient adapter (messages → model format, model output → content string).

**Good fit for the fabric when**

- You already use **Transformers** in the same environment and want to reuse it.
- You need **specific Hugging Face models or adapters** that are not available as GGUF or via Ollama.
- You are **not** optimizing for minimal footprint or fastest “clone and run” on arbitrary hardware.

---

## 6. Local, in container

**What it is:** The LLM runs inside a container (Docker, Podman, etc.) on the **same machine** as the fabric. The fabric still calls an HTTP endpoint (e.g. `http://localhost:8000/v1` or host network).

**Examples:** Ollama in Docker, vLLM in Docker, a custom image with llama-server + GGUF.

### 6.1 General trade-offs (any backend in a container)

**Advantages**

- **Reproducible environment**: same image everywhere; dependency and driver versions are fixed.
- **Isolation**: different Python/CUDA/ROCm from the host; easier “clean” teardown (stop container).
- **Portable** across Linux, Mac, Windows (where Docker runs); good for “one command” if the only prerequisite is Docker.

**Disadvantages**

- **Performance**: GPU passthrough (NVIDIA Docker, etc.) adds overhead; memory and NUMA can be worse than native. For **best inference speed**, we prefer **native** on the host (see BACKENDS.md).
- **Resource limits**: containers can be constrained (CPU, memory); if too tight, inference degrades.
- **Operational**: image size, build time, and updates (base image, model in image vs volume) to manage.

**Good fit for the fabric when**

- **Repeatability and isolation** matter more than squeezing maximum inference speed (e.g. CI, shared dev environments).
- You want “one command” and are okay with **Docker as the single prerequisite**; LLM in container is acceptable for that use case.
- You **cannot** install the backend cleanly on the host (e.g. conflicting system libs); container is the cleaner option.

**When not to use**

- When you need **best performance** on the machine (prefer native LLM on host, fabric in container or native).

---

## 7. Remote API (cloud or another host)

**What it is:** The LLM runs elsewhere. The fabric calls a `base_url` on the internet (or another machine on the network) and sends an `api_key` if required. No local inference.

**Examples:** OpenAI, Azure OpenAI, Anthropic, Google AI, or a self-hosted server (Ollama, vLLM, etc.) on another box.

### 7.1 Trade-offs

**Advantages**

- **No local GPU or large RAM**; good for thin clients, CI without GPU, or when local hardware is insufficient.
- **Managed** options (OpenAI, Azure) reduce ops; scaling and availability are the provider’s concern.
- **Quality/capability**: access to very large or specialized models that are impractical to run locally.

**Disadvantages**

- **Latency** and **availability** depend on the network and provider.
- **Cost** (per token or per request); **data leaves the machine** (privacy, compliance).
- **Not local-first**; contradicts the fabric’s default principle. Intended as **fallback** or **optional** when local capability/quality is insufficient (VISION, REQUIREMENTS).

**Good fit for the fabric when**

- Local model is **not good enough** for the task (quality or capability) and you explicitly opt in to cloud.
- You run the fabric in **CI or a minimal environment** without GPU and accept cost/latency.
- You use the fabric as a **thin client** and are okay with all prompts going to a remote endpoint.

**When not to use as default**

- When adhering to **local-first** and **data on the machine**; keep remote as an optional or fallback path.

---

## 8. Orchestrated / managed (K8s, serverless, managed services)

**What it is:** The LLM is run by an orchestrator (Kubernetes, Helm, a managed inference service) or as serverless. The fabric still talks to an HTTP endpoint; the endpoint might be local (e.g. port-forward to a pod) or remote (managed API).

### 8.1 Kubernetes / Helm (self-hosted LLM in cluster)

**Advantages**

- Same “declarative, repeatable” idea as BACKENDS: apply manifests, get a known state; scaling and placement (GPU nodes) are configurable.
- Good for **multi-user** or **shared** inference in an organization.

**Disadvantages**

- Operational complexity; need a cluster, GPU nodes, and image/Helm maintenance.
- Overkill for a single developer or single-machine fabric.

**Good fit when**

- You already run **K8s** and want the fabric to use an LLM service in the cluster; fabric’s `base_url` points at that service.
- You are building a **platform** (many users, many tasks) rather than a single-workstation tool.

### 8.2 Managed inference (e.g. Azure OpenAI, AWS Bedrock, GCP Vertex)

Same as “remote API” (section 7); the “orchestration” is the provider’s. Fabric just needs `base_url` and `api_key`. Use when you want no local inference and are okay with cost and data leaving the machine.

---

## 9. Summary: when to use which

| Goal | Preferred option(s) | Avoid or use only when |
|------|---------------------|-------------------------|
| **Local-first, default** | Ollama or llama.cpp server, **on-host native** | LLM in container (if you care about max performance) |
| **One command from clone** | Bootstrap script (native) or **in-process** (llama-cpp-python + default GGUF) or **one container** (Docker as only prereq) | Manual per-OS install without bootstrap |
| **Best inference performance** | **Native on host** (Ollama, llama-server, vLLM). See **§3.1** for which is most performant by workload (throughput vs latency, GPU vs CPU). | LLM inside container (GPU passthrough overhead) |
| **Clean teardown, no clutter** | Native + bootstrap that doesn’t leave stray files; or container (stop container = stop LLM) | Ad-hoc processes, temp dirs outside workspace_root |
| **Repeatable, same state everywhere** | Bootstrap (native) or Docker/Podman image | Manual, different steps per machine |
| **Multiple backends / fallback** | LiteLLM or config switch; or fabric logic “try local then cloud” | Hardcoding one backend |
| **Cloud / no local GPU** | Remote API (OpenAI, Azure, etc.) as **opt-in** | As default (breaks local-first) |
| **CI / smoke tests** | Mock or tiny model in container; or skip real LLM | Heavy local GPU dependency in CI |
| **AMD / Vulkan / ROCm** | Ollama, llama.cpp (with ROCm build) on host | Backends that are NVIDIA-only unless you have a wrapper |
| **Minimal deps, no server** | **In-process** (llama-cpp-python) with small GGUF | Large Transformers stack if you want “light” default |

---

## 10. How this doc should be used

- **Before adding a new backend or deployment path:** Read the relevant section (e.g. in-process vs server, container vs native) and the summary table; align with BACKENDS (prefer native, clean teardown) and VISION (local-first).
- **Before changing default:** Check section 4.1 (Ollama) and 9; ensure the new default still fits “local-first” and “best performance when cleanly manageable.”
- **When debugging “where should the LLM run?”:** Use dimensions in section 2 and the option categories in section 3; then drill into the specific option (sections 4–8) for pros/cons and “good fit.”
- **When evaluating a new tool** (e.g. another server or library): Add a short subsection under the right category (e.g. 4.x or 5.x) using the same structure (how it runs, where, hardware, deployment; advantages; disadvantages; good fit when).

Keeping this document updated when we add or drop backends or change deployment philosophy will keep future changes consistent with the fabric’s goals.
