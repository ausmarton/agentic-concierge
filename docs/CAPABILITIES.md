# agentic-concierge: Capability Model

> **Note (2026-03-05):** This document describes the Phase 2 keyword-based capability routing
> system which has been **superseded** by the LLM orchestrator (Phase 12) and dynamic pack
> composition (post-Phase 14). The files it references (`config/capabilities.py`,
> `application/recruit.py`, `tests/test_capabilities.py`) have been deleted. Capability IDs
> still exist on `SpecialistConfig.capabilities` for metadata purposes, but routing is now
> handled by `orchestrate_task()` in `application/orchestrator.py`. See ARCHITECTURE.md §8.4
> for the current routing design.

**Purpose:** Reference for all defined capability IDs, which packs provide them, and how to extend the model.

---

## What is a capability?

A *capability* is a named unit of what a specialist pack can do.  The capability model
decouples **"what does this task need?"** from **"which pack provides it?"**, making routing
decisions observable, testable, and extensible without changing core application code.

---

## Defined capability IDs

| Capability ID | What it means | Provided by |
|---|---|---|
| `code_execution` | Run shell commands, compile code, execute scripts | `engineering` |
| `software_testing` | Write and run tests (pytest, unittest, etc.) | `engineering` |
| `file_io` | Read, write, and list files in the workspace | `engineering`, `research` |
| `systematic_review` | Search, screen, extract, and synthesise literature | `research` |
| `citation_extraction` | Extract and format references / bibliographies | `research` |
| `web_search` | Search the web and fetch URLs | `research` (when `network_allowed`) |

---

## How routing works (Phase 2)

```
Task prompt
    │
    ▼
infer_capabilities(prompt, CAPABILITY_KEYWORDS)
    │  keyword substring match → list of required capability IDs
    │
    ▼ required_capabilities
Score each specialist by:
    sum(1 for cap in required_capabilities if cap in specialist.capabilities)
    │
    ├─ highest score wins → specialist_id
    ├─ tie → first in cfg.specialists (config order)
    └─ all-zero → fall back to keyword scoring → hardcoded heuristic
    │
    ▼
RecruitmentResult(specialist_id, required_capabilities)
    │
    ▼
RunResult._meta.required_capabilities   (in HTTP API response)
runlog.jsonl  "recruitment" event        (always written at run start)
```

---

## Capability keywords (inference rules)

Defined in `src/agentic_concierge/config/capabilities.py → CAPABILITY_KEYWORDS`.
A capability is inferred when **any** of its keywords appears as a substring
of the lowercased task prompt.

| Capability ID | Trigger keywords (sample) |
|---|---|
| `code_execution` | `build`, `implement`, `code`, `service`, `pipeline`, `kubernetes`, `deploy`, `python`, `scala`, … |
| `software_testing` | `test`, `pytest`, `unittest`, `coverage`, `tdd`, … |
| `file_io` | `read file`, `write file`, `create file`, `list files` |
| `systematic_review` | `literature`, `systematic review`, `paper`, `arxiv`, `survey`, `bibliography`, `citations` |
| `citation_extraction` | `citations`, `references`, `bibliography` |
| `web_search` | `search the web`, `web search`, `fetch url`, `browse the internet` |

---

## Adding a new capability

1. **Define the capability ID** — add it to `CAPABILITY_KEYWORDS` in
   `src/agentic_concierge/config/capabilities.py` with trigger keywords.

2. **Declare it on the pack** — add the capability ID to `capabilities` in
   `SpecialistConfig` for every pack that can provide it.  For built-in packs
   this is `DEFAULT_CONFIG` in `src/agentic_concierge/config/schema.py`; for custom
   packs set it in your `CONCIERGE_CONFIG_PATH` YAML.

3. **Write a test** — add a case to `tests/test_capabilities.py` verifying that
   the new keyword infers the capability and routes to the correct pack.

---

## Adding a new pack with capabilities

Declare the pack's capabilities in `SpecialistConfig.capabilities`:

```yaml
# CONCIERGE_CONFIG_PATH YAML example
specialists:
  data_analysis:
    description: "Analyse datasets, produce charts, run statistical models."
    keywords: ["analyse", "dataset", "statistics", "chart"]
    workflow: data_analysis
    capabilities:
      - code_execution
      - file_io
    builder: "myorg.packs.data:build_data_analysis_pack"
```

No changes to application code are required.

---

## Observability

Every run logs a `"recruitment"` event as the first entry in `runlog.jsonl`:

```json
{
  "ts": 1708800000.1,
  "kind": "recruitment",
  "step": null,
  "payload": {
    "specialist_id": "engineering",
    "required_capabilities": ["code_execution"],
    "routing_method": "capability_routing"
  }
}
```

`routing_method` is `"capability_routing"` when the pack was selected by
capability matching, or `"explicit"` when the caller specified `specialist_id`
directly (e.g. via `--pack` CLI flag or `"pack"` in the HTTP request body).

The HTTP API `_meta` field also includes `required_capabilities`:

```json
{
  "_meta": {
    "pack": "engineering",
    "required_capabilities": ["code_execution"],
    ...
  }
}
```
