# agentic-concierge documentation

All docs live here and in the repo root. Use this index to find the right document for your purpose.

---

## For users

| Document | Purpose |
|---|---|
| [../README.md](../README.md) | **Start here.** Install, quickstart, CLI reference, HTTP API, configuration, extending. |
| [BACKENDS.md](BACKENDS.md) | Using backends other than Ollama (vLLM, LiteLLM, OpenAI, llama.cpp). |
| [MCP_INTEGRATIONS.md](MCP_INTEGRATIONS.md) | Connecting MCP tool servers (GitHub, Confluence, Jira, filesystem). Config examples. |
| [CAPABILITIES.md](CAPABILITIES.md) | Capability model, routing keywords, and how to add a new capability. |
| [LLM_OPTIONS.md](LLM_OPTIONS.md) | All LLM deployment options: on-host, container, remote. Pros/cons. |

---

## For contributors and engineers

| Document | Purpose |
|---|---|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | **Start here.** Dev setup, running tests, code style, adding packs/tools/backends. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Hexagonal layer design, component map, data flow, SSE streaming, runlog events, extension points. |
| [DECISIONS.md](DECISIONS.md) | Architecture Decision Records (ADR-001 to ADR-011). Read before making structural changes. |
| [../REQUIREMENTS.md](../REQUIREMENTS.md) | MVP functional requirements and validation items. |

---

## For resuming work (agents / any session)

| Document | Purpose |
|---|---|
| [STATE.md](STATE.md) | **Start here.** Current phase (8), fast CI count (368), what's done, what's next, quick commands. |
| [BACKLOG.md](BACKLOG.md) | Prioritised work items with context. First non-done item = what to work on. |
| [PLAN.md](PLAN.md) | Phases 1–8: deliverables, verification gates, acceptance criteria. |

---

## For product direction

| Document | Purpose |
|---|---|
| [VISION.md](VISION.md) | Long-term vision, principles (non-negotiable), use-case pillars, phase history. |
| [DESIGN.md](DESIGN.md) | Design from first principles: naming, structure, fit-for-purpose analysis. |
| [SELF_CONTAINED_LLM.md](SELF_CONTAINED_LLM.md) | Local-first philosophy and why it matters. |

---

## Full document list

| Document | Lines | Summary |
|---|---|---|
| [../README.md](../README.md) | ~230 | User quickstart, full CLI/HTTP/config reference |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | ~180 | Contributor guide (dev setup, tests, extensions) |
| [../REQUIREMENTS.md](../REQUIREMENTS.md) | ~120 | MVP functional requirements and validation |
| [../LICENSE](../LICENSE) | 21 | MIT License |
| [ARCHITECTURE.md](ARCHITECTURE.md) | ~300 | Hexagonal layers, component map, data flow, runlog |
| [DECISIONS.md](DECISIONS.md) | ~1400 | 34 ADRs with rationale and consequences |
| [VISION.md](VISION.md) | ~145 | Vision, principles, phases 1–8 history, phase 9+ roadmap |
| [PLAN.md](PLAN.md) | ~220 | Phase deliverables and verification gates |
| [STATE.md](STATE.md) | ~210 | Phase 8 complete; CI 368; resumability guide |
| [BACKLOG.md](BACKLOG.md) | ~480 | Prioritised items; done tables; what to do next |
| [CAPABILITIES.md](CAPABILITIES.md) | ~80 | Capability model and routing keyword lists |
| [BACKENDS.md](BACKENDS.md) | ~70 | Backend portability and deployment philosophy |
| [MCP_INTEGRATIONS.md](MCP_INTEGRATIONS.md) | ~120 | MCP config examples (GitHub, Confluence, Jira, filesystem) |
| [LLM_OPTIONS.md](LLM_OPTIONS.md) | — | All LLM deployment options with pros/cons |
| [VERIFICATION_PASSES.md](VERIFICATION_PASSES.md) | ~100 | 5-pass verification checklist (Phase 1 baseline) |
| [DESIGN.md](DESIGN.md) | — | First-principles design rationale |
| [DESIGN_ASSESSMENT.md](DESIGN_ASSESSMENT.md) | — | How well implementation matches vision |
| [ENGINEERING.md](ENGINEERING.md) | — | Engineering specialist design notes |
| [SELF_CONTAINED_LLM.md](SELF_CONTAINED_LLM.md) | — | Local-first philosophy |
| [LOCAL_FIRST_REWORK_PLAN.md](LOCAL_FIRST_REWORK_PLAN.md) | — | Local-first architecture notes |
