# Agentic Concierge User Guide

This guide covers installation, configuration, and daily use of
agentic-concierge — an on-demand specialist-pack system powered by local
LLMs (Ollama by default).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [First Run Setup](#3-first-run-setup)
4. [Running Tasks](#4-running-tasks)
5. [Specialist Packs](#5-specialist-packs)
6. [Orchestration and Multi-Specialist Tasks](#6-orchestration-and-multi-specialist-tasks)
7. [Run History and Search](#7-run-history-and-search)
8. [Resuming Interrupted Runs](#8-resuming-interrupted-runs)
9. [HTTP API](#9-http-api)
10. [Configuration](#10-configuration)
11. [MCP Integrations](#11-mcp-integrations)
12. [Containerised Workspaces](#12-containerised-workspaces)
13. [Environment Variables Reference](#13-environment-variables-reference)
14. [Workspace Layout](#14-workspace-layout)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Ollama** | Install from <https://ollama.com>. Run `ollama serve` to start. |
| **At least one model** | `ollama pull llama3.1:8b` (minimum) or `ollama pull qwen2.5:7b`. |
| **Python 3.10+** | Only required for `pip install`. The native binary provisions its own Python via uv. |

Verify Ollama is running:

```bash
curl -s http://localhost:11434/v1/models | head -c 200
```

---

## 2. Installation

### 2.1 Native binary (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/ausmarton/agentic-concierge/main/install.sh | sh
```

Override the install directory (default `~/.local/bin`):

```bash
CONCIERGE_INSTALL_DIR=~/bin \
  curl -fsSL https://raw.githubusercontent.com/ausmarton/agentic-concierge/main/install.sh | sh
```

If `~/.local/bin` is not in your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
# Add the line above to ~/.bashrc or ~/.zshrc to persist.
```

**What happens on first run:** the Rust launcher downloads `uv`, creates a
Python virtual environment in `~/.local/share/agentic-concierge/venv`, and
installs the `agentic-concierge` package from PyPI. Subsequent launches
reuse the cached venv.

To install optional extras via the launcher, set `CONCIERGE_EXTRA`:

```bash
export CONCIERGE_EXTRA="mcp,otel"
concierge doctor
```

### 2.2 pip install

```bash
pip install agentic-concierge
```

Optional extras:

| Extra | What it adds |
|-------|-------------|
| `mcp` | MCP tool-server support (`mcp>=1.0`) |
| `otel` | OpenTelemetry tracing (`opentelemetry-api`, `opentelemetry-sdk`) |
| `embed` | ChromaDB vector index (`chromadb>=0.4`) |
| `browser` | Headless browser tools (`playwright>=1.40`) |
| `nano` | In-process inference (`mistralrs>=0.3`) |
| `all` | All of the above (except `nano`) |

Install with extras:

```bash
pip install "agentic-concierge[all]"
```

### 2.3 From source

```bash
git clone https://github.com/ausmarton/agentic-concierge.git
cd agentic-concierge
pip install -e ".[dev]"
```

### 2.4 Updating

**Native binary:**

```bash
concierge --self-update
```

The launcher checks for updates on every invocation by default. Suppress
the advisory with:

```bash
export CONCIERGE_NO_UPDATE_CHECK=1
```

**pip:**

```bash
pip install --upgrade agentic-concierge
```

---

## 3. First Run Setup

### Bootstrap

Detect hardware, select a profile tier, and pull recommended models:

```bash
concierge bootstrap
```

Override the detected profile:

```bash
concierge bootstrap --profile medium
```

Skip interactive prompts (CI-friendly):

```bash
concierge bootstrap --non-interactive
```

### Health check

```bash
concierge doctor
```

Add `--verbose` / `-v` for full error details. The output includes four
panels:

1. **System** — profile tier, CPU, RAM, GPU, disk, internet.
2. **Features** — which features are enabled/disabled for your tier.
3. **Backends** — Ollama / vLLM / cloud status, available models.
4. **Extras** — optional tools (Playwright, ChromaDB) with install hints.

### Profile tiers

| Tier | RAM | VRAM | Key features enabled |
|------|-----|------|----------------------|
| **NANO** | < 8 GB | — | In-process inference, cloud |
| **SMALL** | 8–16 GB | < 4 GB | Ollama, MCP, browser, cloud |
| **MEDIUM** | 16–32 GB | 4–12 GB | + vLLM, embeddings |
| **LARGE** | 32–64 GB | 12–24 GB | + containerised workspaces |
| **SERVER** | 64 GB+ | 24 GB+ | + telemetry; vLLM replaces Ollama |

### Quick smoke test

```bash
concierge run "Create hello.txt with content Hello World" --pack engineering
```

---

## 4. Running Tasks

### Basic usage

```bash
concierge run "Implement a Python function that checks if a number is prime"
```

Without `--pack`, the orchestrator auto-routes to the best specialist.

### Forcing a specialist

```bash
concierge run "Search for recent papers on RLHF" --pack research
concierge run "Fix the failing tests in src/" --pack engineering
concierge run "Find Jira tickets about auth" --pack enterprise_research
```

### Streaming output

```bash
concierge run "Build a REST API in Flask" --stream
# or
concierge run "Build a REST API in Flask" -s
```

### Full options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--pack` | string | (auto) | Force a specialist: `engineering`, `research`, `enterprise_research` |
| `--model-key` | string | `quality` | Model profile to use (`quality` or `fast`) |
| `--stream` / `-s` | flag | off | Stream events as they happen |
| `--network-allowed` | bool | `true` | Allow network tools (`web_search`, `fetch_url`) |
| `--verbose` / `-v` | flag | off | DEBUG logging to stderr |

### Understanding output

When a task completes you see:

1. **Result panel** — summary from the specialist.
2. **JSON payload** — structured output (artifacts, test results, citations).
3. **Run directory** — path to the workspace with all generated files.

---

## 5. Specialist Packs

### 5.1 Engineering

Autonomous software engineering agent for code, tests, and debugging.

**Tools:**

| Tool | Description |
|------|-------------|
| `shell` | Run shell commands in a sandboxed workspace. Parameters: `cmd` (string array), `timeout_s` (int, default 120). |
| `read_file` | Read a file from the workspace (relative path). |
| `write_file` | Write or overwrite a file (creates parent dirs). Parameters: `path`, `content`. |
| `list_files` | List files in workspace. Parameter: `max_files` (int, default 500). |
| `run_tests` | Run the project test suite. Parameters: `framework` (`auto`/`pytest`/`unittest`/`cargo`/`npm`), `path` (default `.`). |
| `finish_task` | Declare the task complete. Required fields: `summary`, `tests_verified`. |

**Quality gate:** the LLM must call `run_tests` and pass
`tests_verified: true` to `finish_task`. Calling `finish_task` with
`tests_verified: false` is rejected — the agent is told to fix failures
first.

**Example tasks:**

```bash
concierge run "Add input validation to the User model" --pack engineering
concierge run "Write unit tests for utils.py" --pack engineering
concierge run "Fix the TypeError in data_processor.py" --pack engineering
```

### 5.2 Research

Systematic review and web research agent.

**Tools:**

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo. Parameters: `query`, `max_results` (default 8). |
| `fetch_url` | Fetch full text of a URL. Parameter: `url`. |
| `read_file` | Read workspace file. |
| `write_file` | Write workspace file. |
| `list_files` | List workspace files. |
| `finish_task` | Declare complete. Required: `executive_summary`. Optional: `key_findings`, `citations`, `evidence_table_path`, `bibliography_path`. |

`web_search` and `fetch_url` are omitted when `--network-allowed false`.

**Workflow:** The agent scopes the question, searches, screens sources
(maintaining a screening log), extracts findings, writes a bibliography,
and synthesises an executive summary. Only URLs actually fetched via
`fetch_url` may be cited.

**Example tasks:**

```bash
concierge run "Survey recent advances in transformer architectures" --pack research
concierge run "Compare Redis vs Memcached for session storage" --pack research
```

### 5.3 Enterprise Research

Extended research pack for internal enterprise sources (GitHub, Confluence,
Jira) with confidence and staleness annotations.

**Additional tools beyond Research:**

| Tool | Description |
|------|-------------|
| `cross_run_search` | Search prior runs for relevant past research. Parameters: `query`, `limit` (default 5). |
| `mcp__<server>__<tool>` | MCP-provided tools from configured servers (e.g., `mcp__github__search_repositories`). |

**Confidence annotations:** Each key finding is annotated:

| Tag | Meaning |
|-----|---------|
| `[HIGH]` | Recent, authoritative source with clear date |
| `[MEDIUM]` | Credible but date unclear or moderately old |
| `[LOW]` | Potentially stale, unofficial, or second-hand |
| `[STALE?]` | Likely outdated (old version, archived) |
| `[UNVERIFIED]` | Claim cannot be verified from available sources |

**Example:**

```bash
concierge run "Find all Jira tickets about authentication failures this quarter" \
  --pack enterprise_research
```

### Browser tools (all packs)

When the `browser` extra is installed and the `BROWSER` feature is enabled
for your profile tier, all packs gain six additional tools:

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL; returns title and status code. |
| `browser_get_text` | Extract text from a CSS selector (default `body`). |
| `browser_get_links` | Return all anchor links on the current page. |
| `browser_click` | Click an element matching a CSS selector. |
| `browser_fill` | Fill a form field matching a CSS selector. |
| `browser_screenshot` | Take a screenshot and save to the workspace. |

Install: `pip install "agentic-concierge[browser]"` then
`playwright install chromium`.

---

## 6. Orchestration and Multi-Specialist Tasks

### Preview a plan

```bash
concierge plan "Build a REST API and write a literature review of REST best practices"
```

This shows the orchestration plan — which specialists are assigned, the
execution mode, and whether synthesis is needed — without running anything.

### Task force modes

When multiple specialists are needed, the orchestrator selects a mode:

| Mode | Behaviour |
|------|-----------|
| **sequential** | Specialists run one after another. The output of each specialist is forwarded as context to the next. |
| **parallel** | All specialists run concurrently with the same initial prompt. No context sharing during execution. |

Set the default in your config:

```json
{
  "task_force_mode": "parallel"
}
```

The orchestrator may override this per-task if it determines that one
specialist's output is a prerequisite for another.

### Synthesis

When multiple specialists complete, a synthesis step automatically combines
their outputs into a single cohesive result. This step uses the same model
as the routing decision (`routing_model_key`).

---

## 7. Run History and Search

### List recent runs

```bash
concierge logs list
concierge logs list --limit 50
concierge logs list --workspace /path/to/workspace
```

Output columns: Run ID, Started, Specialists, Event count, Summary.
Resumable runs are marked `(resumable)`.

### Show a run's event log

```bash
concierge logs show 20260303-141530-a1b2c3
```

Filter by event kind:

```bash
concierge logs show 20260303-141530-a1b2c3 --kinds tool_call,tool_result
```

### Search past runs

```bash
concierge logs search "authentication bug"
concierge logs search "REST API" --limit 10
```

When `run_index.embedding_model` is set in your config (e.g.,
`nomic-embed-text`), search uses semantic similarity (cosine via Ollama
embeddings). Otherwise it falls back to keyword/substring matching.

### Event kinds reference

| Kind | Description |
|------|-------------|
| `recruitment` | Specialist(s) selected for the task |
| `orchestration_plan` | Multi-specialist plan created |
| `pack_start` | A specialist pack begins execution |
| `llm_request` | LLM API call issued |
| `llm_response` | LLM returns a response |
| `tool_call` | LLM requests tool execution |
| `tool_result` | Tool execution succeeded |
| `tool_error` | Tool execution failed |
| `security_event` | Sandbox violation detected |
| `corrective_reprompt` | LLM returned plain text; nudging to use a tool |
| `cloud_fallback` | Fell back to cloud LLM |
| `quality_gate_failed` | `finish_task` payload rejected by quality gate |
| `loop_detected` | Same tool+args called repeatedly; loop broken |
| `synthesis_complete` | Multi-specialist synthesis finished |
| `run_complete` | Run finished successfully |

---

## 8. Resuming Interrupted Runs

Checkpoints are saved automatically during execution. If a run is
interrupted (network timeout, process kill, crash), it can be resumed.

### Find resumable runs

```bash
concierge logs list
```

Runs with a checkpoint but no `run_complete` event show `(resumable)`.

A run is considered resumable when:
- `checkpoint.json` exists in the run directory, AND
- the run log does not contain a `run_complete` event.

### Resume

```bash
concierge resume 20260303-141530-a1b2c3
```

Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--workspace` / `-w` | path | `.concierge` | Workspace root |
| `--model-key` | string | `quality` | Model profile |
| `--verbose` / `-v` | flag | off | DEBUG logging |

The resumed run continues from the last checkpoint, reusing the original
specialist assignment and conversation history.

---

## 9. HTTP API

### Start the server

```bash
concierge serve
concierge serve --host 0.0.0.0 --port 9000
```

Default: `http://127.0.0.1:8787`

### Endpoints

#### Health check

```bash
curl http://localhost:8787/health
```

```json
{"ok": true}
```

#### Run a task (blocking)

```bash
curl -X POST http://localhost:8787/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python hello world script", "pack": "engineering"}'
```

Request body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | (required) | Task description |
| `pack` | string | `null` | Force a specialist (`engineering`, `research`, `enterprise_research`) |
| `model_key` | string | `"quality"` | Model profile |
| `network_allowed` | bool | `true` | Allow network tools |

Response (200):

```json
{
  "payload": { "summary": "...", "artifacts": ["hello.py"], "tests_verified": true },
  "_meta": {
    "pack": "engineering",
    "specialist_ids": ["engineering"],
    "is_task_force": false,
    "run_dir": ".concierge/runs/20260303-141530-a1b2c3/workspace",
    "workspace": ".concierge",
    "model": "qwen2.5:7b",
    "run_id": "20260303-141530-a1b2c3",
    "required_capabilities": ["code_execution", "file_io", "software_testing"]
  }
}
```

Error codes: `503` (LLM unreachable / model not found / timeout).

#### Stream a task (SSE)

```bash
curl -N -X POST http://localhost:8787/run/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Build a Flask API"}'
```

Returns `text/event-stream`. Each event is a JSON line prefixed with
`data: `:

```
data: {"kind": "recruitment", "data": {"specialist_ids": ["engineering"]}, "step": null}
data: {"kind": "pack_start", "data": {"specialist_id": "engineering"}, "step": null}
data: {"kind": "tool_call", "data": {"tool": "shell", "args": {"cmd": ["python", "--version"]}}, "step": 1}
data: {"kind": "tool_result", "data": {"tool": "shell", "result": {"stdout": "Python 3.12.0"}}, "step": 1}
data: {"kind": "run_complete", "data": {"run_id": "...", "ok": true}, "step": null}
```

The stream ends with either a `run_complete` or `_run_error_` event.

#### Check run status

```bash
curl http://localhost:8787/runs/20260303-141530-a1b2c3/status
```

Response (completed):

```json
{
  "status": "completed",
  "run_id": "20260303-141530-a1b2c3",
  "specialist_ids": ["engineering"],
  "task_force_mode": "sequential"
}
```

Returns `404` if the run ID is not found.

### Authentication

Set `CONCIERGE_API_KEY` to require a bearer token on all endpoints except
`/health`:

```bash
export CONCIERGE_API_KEY="my-secret-key"
concierge serve
```

```bash
curl -H "Authorization: Bearer my-secret-key" \
  -X POST http://localhost:8787/run \
  -d '{"prompt": "hello"}'
```

Returns `401 Unauthorized` if the token is missing or invalid.

### Rate limiting

Set `CONCIERGE_RATE_LIMIT` to limit requests per IP per minute:

```bash
export CONCIERGE_RATE_LIMIT=60
concierge serve
```

Excess requests receive `429 Too Many Requests` with a `Retry-After`
header. `/health` is always exempt.

---

## 10. Configuration

### Config file location

Set `CONCIERGE_CONFIG_PATH` to point at a JSON file:

```bash
export CONCIERGE_CONFIG_PATH=~/.concierge/config.json
```

If unset or the file does not exist, built-in defaults are used (Ollama on
`localhost:11434`, `qwen2.5:7b` for fast, `qwen2.5:14b` for quality).

**Important:** the config loader uses `json.loads()` — only JSON is
supported, not YAML.

### Minimal config example

```json
{
  "models": {
    "fast": {
      "base_url": "http://localhost:11434/v1",
      "model": "llama3.1:8b",
      "backend": "ollama"
    },
    "quality": {
      "base_url": "http://localhost:11434/v1",
      "model": "qwen2.5:14b",
      "backend": "ollama"
    }
  },
  "specialists": {
    "engineering": {
      "description": "Code, build, test, debug",
      "workflow": "pack",
      "keywords": ["code", "build", "test"],
      "capabilities": ["code_execution", "file_io", "software_testing"]
    },
    "research": {
      "description": "Web research and systematic review",
      "workflow": "pack",
      "keywords": ["research", "survey", "paper"],
      "capabilities": ["web_search", "systematic_review", "file_io"]
    }
  }
}
```

### Model profiles

Each entry in `"models"` defines an LLM endpoint:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | (required) | OpenAI-compatible API base (e.g., `http://localhost:11434/v1`) |
| `model` | string | (required) | Model name (e.g., `llama3.1:8b`, `gpt-4`) |
| `api_key` | string | `""` | Bearer token; leave empty for local Ollama |
| `backend` | string | `"ollama"` | `"ollama"` (400-retry, tool detection) or `"generic"` (bare OpenAI-compatible) |
| `temperature` | float | `0.1` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `max_tokens` | int | `2048` | Maximum output tokens |
| `timeout_s` | float | `360.0` | HTTP timeout in seconds |

### Non-Ollama backends

**OpenAI:**

```json
{
  "models": {
    "cloud_quality": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4",
      "api_key": "sk-...",
      "backend": "generic",
      "max_tokens": 4096
    }
  }
}
```

**vLLM:**

```json
{
  "models": {
    "quality": {
      "base_url": "http://localhost:8000/v1",
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "backend": "generic"
    }
  }
}
```

### LLM auto-management

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_llm_ensure_available` | bool | `true` | Auto-start Ollama if unreachable |
| `local_llm_start_cmd` | string[] | `["ollama", "serve"]` | Command to launch Ollama |
| `local_llm_start_timeout_s` | int | `90` | Seconds to wait for startup |
| `auto_pull_if_missing` | bool | `true` | Auto-pull model if not available locally |
| `auto_pull_model` | string | `"qwen2.5:7b"` | Which model to pull |

### Semantic search

Enable cross-run semantic search by setting an embedding model:

```json
{
  "run_index": {
    "embedding_model": "nomic-embed-text",
    "provider": "jsonl"
  }
}
```

For persistent vector storage with ChromaDB:

```json
{
  "run_index": {
    "embedding_model": "nomic-embed-text",
    "provider": "chromadb",
    "chromadb_collection": "agentic_concierge_runs"
  }
}
```

Requires `pip install "agentic-concierge[embed]"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding_model` | string | `null` | Ollama embedding model (e.g., `nomic-embed-text`). `null` disables semantic search. |
| `embedding_base_url` | string | `null` | Base URL for embeddings API. Auto-derived from primary model if not set. |
| `provider` | string | `"jsonl"` | `"jsonl"` (flat file + cosine scan) or `"chromadb"` (persistent ANN) |
| `chromadb_path` | string | `""` | ChromaDB storage directory. Empty uses OS default. |
| `chromadb_collection` | string | `"agentic_concierge_runs"` | ChromaDB collection name |

### Cloud fallback

Fall back to a cloud LLM when the local model fails:

```json
{
  "models": {
    "quality": {
      "base_url": "http://localhost:11434/v1",
      "model": "qwen2.5:14b",
      "backend": "ollama"
    },
    "cloud_quality": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4",
      "api_key": "sk-...",
      "backend": "generic"
    }
  },
  "cloud_fallback": {
    "model_key": "cloud_quality",
    "policy": "no_tool_calls"
  }
}
```

| Policy | Trigger |
|--------|---------|
| `no_tool_calls` | Local LLM returned plain text without calling any tools |
| `malformed_args` | Local LLM returned tool calls with unparseable JSON arguments |
| `always` | Always use cloud (debug/testing only) |

### Feature overrides

Force-enable or force-disable individual features regardless of profile:

```json
{
  "features": {
    "browser": true,
    "container": false,
    "telemetry": true
  }
}
```

`null` (or omitted) = use profile default. `true` = force enable.
`false` = force disable.

Available features: `inprocess`, `ollama`, `vllm`, `cloud`, `mcp`,
`browser`, `embedding`, `telemetry`, `container`.

### Telemetry (OpenTelemetry)

```json
{
  "telemetry": {
    "enabled": true,
    "service_name": "agentic-concierge",
    "exporter": "otlp",
    "otlp_endpoint": "http://localhost:4317"
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable tracing |
| `service_name` | string | `"agentic-concierge"` | OTEL service name |
| `exporter` | string | `"none"` | `"none"`, `"console"` (stdout), or `"otlp"` (gRPC) |
| `otlp_endpoint` | string | `""` | gRPC endpoint (required when exporter is `"otlp"`) |

Requires `pip install "agentic-concierge[otel]"`.

### Resource limits

```json
{
  "resource_limits": {
    "max_concurrent_agents": 4,
    "max_ram_mb": 16384,
    "max_gpu_vram_mb": 8192
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_concurrent_agents` | int | `4` | Max parallel specialist agents |
| `max_ram_mb` | int | `null` | Hard RAM cap (MB); `null` = no cap |
| `max_gpu_vram_mb` | int | `null` | Hard GPU VRAM cap (MB); `null` = no cap |
| `model_cache_path` | string | `""` | Model weights directory; empty = OS default |

---

## 11. MCP Integrations

MCP (Model Context Protocol) servers provide additional tools to
specialists. Tools are automatically discovered and prefixed
`mcp__<server_name>__<tool_name>` to avoid collisions with built-in tools.

**Install the MCP extra:**

```bash
pip install "agentic-concierge[mcp]"
```

Or via the launcher:

```bash
export CONCIERGE_EXTRA="mcp"
```

### GitHub MCP server

```json
{
  "specialists": {
    "enterprise_research": {
      "description": "Enterprise research with GitHub access",
      "workflow": "pack",
      "keywords": ["github", "code search"],
      "capabilities": ["enterprise_search", "github_search", "web_search", "file_io"],
      "mcp_servers": [
        {
          "name": "github",
          "transport": "stdio",
          "command": "npx",
          "args": ["--yes", "--", "@modelcontextprotocol/server-github"],
          "env": {
            "GITHUB_TOKEN": "${GITHUB_TOKEN}"
          },
          "timeout_s": 30.0
        }
      ]
    }
  }
}
```

Set your token: `export GITHUB_TOKEN="ghp_..."`.

Exposed tools include `mcp__github__search_repositories`,
`mcp__github__get_file_contents`, `mcp__github__list_issues`,
`mcp__github__search_code`, and more.

Optional pre-install for faster startup:

```bash
npm install -g @modelcontextprotocol/server-github
```

### Confluence MCP server (SSE transport)

```json
{
  "mcp_servers": [
    {
      "name": "confluence",
      "transport": "sse",
      "url": "https://your-org.atlassian.net/rest/mcp/v1",
      "headers": {
        "Authorization": "Bearer ${CONFLUENCE_API_TOKEN}",
        "X-Atlassian-Token": "no-check"
      },
      "timeout_s": 30.0
    }
  ]
}
```

### Confluence MCP server (stdio transport, community)

```json
{
  "mcp_servers": [
    {
      "name": "confluence",
      "transport": "stdio",
      "command": "npx",
      "args": ["--yes", "--", "@your-org/confluence-mcp-server"],
      "env": {
        "CONFLUENCE_BASE_URL": "https://your-org.atlassian.net",
        "CONFLUENCE_EMAIL": "you@example.com",
        "CONFLUENCE_API_TOKEN": "${CONFLUENCE_API_TOKEN}"
      },
      "timeout_s": 30.0
    }
  ]
}
```

### Jira MCP server

```json
{
  "mcp_servers": [
    {
      "name": "jira",
      "transport": "stdio",
      "command": "npx",
      "args": ["--yes", "--", "@your-org/jira-mcp-server"],
      "env": {
        "JIRA_BASE_URL": "https://your-org.atlassian.net",
        "JIRA_EMAIL": "you@example.com",
        "JIRA_API_TOKEN": "${JIRA_API_TOKEN}"
      },
      "timeout_s": 30.0
    }
  ]
}
```

### Filesystem MCP server (testing)

```json
{
  "mcp_servers": [
    {
      "name": "fs",
      "transport": "stdio",
      "command": "npx",
      "args": ["--yes", "--", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"],
      "timeout_s": 15.0
    }
  ]
}
```

### MCP server config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Server name (used as tool prefix) |
| `transport` | string | `"stdio"` | `"stdio"` or `"sse"` |
| `command` | string | — | Executable to launch (required for stdio) |
| `args` | string[] | `[]` | Command arguments |
| `env` | object | `null` | Environment variables; `${VAR}` syntax expands from host env |
| `url` | string | — | SSE endpoint URL (required for sse) |
| `headers` | object | `{}` | HTTP headers (sse only) |
| `timeout_s` | float | `30.0` | Timeout per MCP call |

---

## 12. Containerised Workspaces

Isolate shell command execution inside a Podman container for security and
environment control.

### Setup

1. Install Podman: <https://podman.io/docs/installation>
2. Pull the desired image:

```bash
podman pull python:3.12-slim
```

3. Set `container_image` in your specialist config:

```json
{
  "specialists": {
    "engineering": {
      "description": "Isolated engineering workspace",
      "workflow": "pack",
      "keywords": ["code", "build"],
      "capabilities": ["code_execution", "file_io", "software_testing"],
      "container_image": "python:3.12-slim"
    }
  }
}
```

### How it works

- Shell commands (`shell`, `run_tests`) execute inside the container via
  `podman exec`.
- The workspace is mounted at `/workspace` inside the container with the
  `:Z` flag for SELinux compatibility (Fedora/RHEL).
- File I/O tools (`read_file`, `write_file`, `list_files`) operate on the
  host filesystem directly.
- MCP tools also run on the host, not inside the container.
- The container is started with `--rm` for automatic cleanup.

### Combining container + MCP

Both `container_image` and `mcp_servers` can be set on the same specialist.
The wrapping order is:

```
Base pack -> MCPAugmentedPack (MCP on host) -> ContainerisedSpecialistPack (shell in container)
```

```json
{
  "specialists": {
    "secure_engineering": {
      "description": "Isolated shell + GitHub MCP",
      "workflow": "pack",
      "keywords": ["code"],
      "capabilities": ["code_execution", "github_search"],
      "container_image": "python:3.12-slim",
      "mcp_servers": [
        {
          "name": "github",
          "transport": "stdio",
          "command": "npx",
          "args": ["--yes", "--", "@modelcontextprotocol/server-github"],
          "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
        }
      ]
    }
  }
}
```

---

## 13. Environment Variables Reference

### Launcher (Rust binary)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCIERGE_DATA_DIR` | `~/.local/share/agentic-concierge` | Data directory (venv, uv binary, version file) |
| `CONCIERGE_INSTALL_DIR` | `~/.local/bin` | Binary install directory (used by `install.sh`) |
| `CONCIERGE_NO_UPDATE_CHECK` | unset | Set to `1` to skip the update advisory on startup |
| `CONCIERGE_EXTRA` | unset | Pip extras for `uv pip install` (e.g., `"mcp,otel"`) |

### Python application

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCIERGE_CONFIG_PATH` | unset | Path to JSON config file. Unset = built-in defaults. |
| `CONCIERGE_WORKSPACE` | `.concierge` | Workspace root for run logs and checkpoints |
| `CONCIERGE_API_KEY` | unset | Bearer token for HTTP API auth. Unset = auth disabled. |
| `CONCIERGE_RATE_LIMIT` | unset | Max requests per IP per minute. Unset = no limit. |

---

## 14. Workspace Layout

```
$CONCIERGE_WORKSPACE/          (default: .concierge/)
  runs/
    20260303-141530-a1b2c3/     (run ID: YYYYMMDD-HHMMSS-<6 hex chars>)
      runlog.jsonl               Append-only event log (one JSON object per line)
      workspace/                 Task execution directory (generated files live here)
      checkpoint.json            Resume state (present only if run was interrupted)
    20260303-142000-d4e5f6/
      ...
  index/
    runs.jsonl                   Index of all runs (for search)
```

### Run log event format

Each line in `runlog.jsonl` is a JSON object:

```json
{"ts": 1709472930.123, "kind": "tool_call", "step": "1", "payload": {"tool": "shell", "args": {"cmd": ["ls"]}}}
```

| Field | Description |
|-------|-------------|
| `ts` | Unix epoch timestamp |
| `kind` | Event type (see [Event kinds reference](#event-kinds-reference)) |
| `step` | LLM turn number (`"1"`, `"2"`, ...) or `"synthesis"` or `null` |
| `payload` | Event-specific data |

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on `localhost:11434` | Ollama not running | `ollama serve` |
| `model "xyz" not found` | Model not pulled | `ollama pull xyz` |
| Task times out after 360s | Model too slow or prompt too large | Increase `timeout_s` in model config; use a smaller model |
| `RuntimeError: mcp package not installed` | MCP extra missing | `pip install "agentic-concierge[mcp]"` |
| `RuntimeError: Failed to start Podman container` | Podman not installed or image not pulled | Install Podman; `podman pull <image>` |
| `FileNotFoundError: npx` | Node.js not installed | Install Node.js; or `npm install -g @modelcontextprotocol/server-github` |
| LLM loops (same tool called repeatedly) | Small model stuck in a loop | Automatic loop detection breaks the cycle after 2 repeats in 8 calls. Try a larger model. |
| `finish_task` rejected: `tests_verified is False` | Engineering quality gate | Run `run_tests` and fix failures before calling `finish_task` |
| `concierge: command not found` | Binary not on PATH | `export PATH="$HOME/.local/bin:$PATH"` |
| Config not loading | Wrong path or invalid JSON | Check `CONCIERGE_CONFIG_PATH`; validate with `python -m json.tool < config.json` |
| `401 Unauthorized` on API | API key mismatch | Ensure `Authorization: Bearer <key>` matches `CONCIERGE_API_KEY` |
| `429 Too Many Requests` | Rate limit exceeded | Wait for `Retry-After` seconds or raise `CONCIERGE_RATE_LIMIT` |
| Semantic search returns no results | Embedding model not configured | Set `run_index.embedding_model` (e.g., `"nomic-embed-text"`) and pull the model |
| `TypeError: time.monotonic() + "120"` | Old version with string timeout bug | Update to v0.3.9+: `pip install --upgrade agentic-concierge` |
| `--self-update` fails on `.sig` | Expected behaviour (signing key not yet configured) | Update to v0.3.8+; the warning is informational, the update still applies |
