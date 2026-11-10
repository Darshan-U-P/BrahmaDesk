# BrahmaDesk + ULTK Integration Plan

**Status:** Integration foundation in progress  
**Target mode:** Embedded ULTK runtime in the BrahmaDesk FastAPI backend  
**Scope:** All implemented ULTK capability packs, integrated incrementally  
**Working branch:** `feat/ultk-integration-foundation`  
**Baseline:** BrahmaDesk `main` at `3c9aa366956a44ce6b54fb9b39971d269425daca`

## 1. Product boundary

BrahmaDesk remains the desktop application, local inference host, conversation/profile/workspace store, and user-facing control plane. ULTK becomes the provider-independent tool and agent execution subsystem embedded in the BrahmaDesk Python backend.

Responsibilities:

- **BrahmaDesk desktop:** chat UI, tool activity UI, approval prompts, user settings, profile/project selection.
- **BrahmaDesk backend:** API boundary, profile and conversation ownership, prompt/context assembly, local inference lifecycle, SSE delivery, app persistence and packaging.
- **Brahma Engine:** model validation, resource planning, model process lifecycle, generation and streaming.
- **ULTK:** tool contracts, registry, schemas, policy gates, provider-neutral agent loop, bounded execution, approval/audit/workflow/plugin services and tool packs.
- **Workers:** optional isolated processes for high-risk or resource-intensive tools. Embedding ULTK does not mean executing every tool in the API process.

Do not merge BrahmaDesk and ULTK into one conceptual runtime. Keep the inference engine and tool execution engine independently replaceable.

## 2. Source integration strategy

ULTK's implemented capabilities are currently distributed across sequential and divergent feature branches. The ULTK README and verification ledger explicitly identify the implementation as pre-alpha, not fully integrated and not comprehensively validated.

Before adding ULTK to BrahmaDesk's production dependency graph:

1. Create a dedicated ULTK integration branch from the latest comprehensive tool-pack branch.
2. Reconcile the persistent-memory branch with the later security/data/documents/REST/provider/plugin/media/email chain.
3. Repair and validate package metadata, optional dependency groups, imports and public exports.
4. Run the complete ULTK test/lint suite on the consolidated branch and record unverified platform/security behavior.
5. Pin BrahmaDesk to the resulting immutable ULTK commit. Prefer a Git submodule under `third_party/ultk` or an equivalent pinned source dependency; do not track a moving branch in application builds.
6. Ensure the PyInstaller build includes the ULTK package and only the selected optional dependencies.

Do not copy feature branches into BrahmaDesk manually or depend on an unpinned branch.

## 3. Runtime architecture

```text
Electron / React
  |  chat, tool activity, approvals, settings
  v
BrahmaDesk FastAPI
  |-- Profile / conversation / project context
  |-- Chat and SSE event protocol
  |-- ULTK lifecycle + session adapter
  |       |-- BrahmaDeskProviderAdapter
  |       |-- ToolRegistry and capability manifest
  |       |-- Policy / approval bridge
  |       |-- Bounded AgentRuntime
  |       |-- Audit / workflow / plugin services
  |       '-- Built-in tool packs
  |
  v
Brahma Engine InferenceBackend
  |  structured tool-call-capable generation
  v
llama.cpp now; future Brahma native inference backend

ULTK tools -> in-process low-risk tools or bounded worker processes
BrahmaDesk SQLite -> conversations, profiles, projects, notes and memories
ULTK audit store -> execution audit and approval records, correlated to app IDs
```

## 4. Critical interface work

The existing Brahma Engine generation contract streams text chunks. ULTK's ProviderAdapter contract returns an AssistantTurn containing optional ToolCallRequest objects. The integration must add a structured completion path without breaking existing text-only generation.

Required changes:

- Extend the generation request with provider-compatible tool schemas and an explicit non-streaming structured-completion mode.
- Extend generation results with structured tool-call data, including call ID, function name and parsed JSON arguments.
- Parse both non-streaming tool-call responses and streamed tool-call deltas safely. Never interpret malformed or incomplete arguments as executable calls.
- Add a BrahmaDesk provider adapter implementing ULTK's ProviderAdapter protocol.
- Preserve BrahmaDesk sampling settings and selected model identity in host-owned configuration.
- Keep ordinary chat SSE behavior unchanged. Agent SSE should additionally report tool lifecycle events and stream final assistant text where supported.
- Preserve error and incomplete-stream semantics: no completed conversation exchange is persisted until the agent run finishes successfully.

The current ULTK ProviderAdapter is request/response rather than streaming. Add a compatible optional streaming protocol/event interface instead of silently changing existing provider implementations.

## 5. Identity, context and persistence

Every agent run must be bound to a server-validated context object containing:

- request ID;
- profile ID;
- conversation ID (when present);
- project ID (when selected);
- ULTK session/task ID;
- enabled tool set and capability grants;
- cancellation token/deadline.

Never trust profile or project ownership supplied only by model output or a tool argument. Resolve and validate all IDs through BrahmaDesk's existing database helpers.

BrahmaDesk remains the system of record for user conversations, profile settings, projects, notes and memories. Implement a memory adapter for existing BrahmaDesk stores where appropriate; do not introduce duplicate global/profile memory stores by default. Keep ULTK approval and audit records in a separate ULTK-owned store, linked by opaque correlation IDs.

## 6. Tool capability rollout

Integrate all implemented tool packs through a capability manifest, not a global unrestricted registry.

1. Read-only filesystem, document extraction, structured data read and Git inspection.
2. System inspection and browser read/navigation.
3. Computer/desktop automation and browser interaction.
4. Terminal/process and development operations.
5. REST integrations, database writes, plugin execution and media processing.
6. External communications (SMTP) and multi-step workflows.

Every tool must declare its required capabilities, scope, timeout, output ceiling, optional dependencies, side-effect classification and approval requirement. Destructive, external, credentialed or otherwise consequential actions require an explicit policy and human approval. A tool being registered must not imply that it is enabled or authorized.

## 7. Security requirements

- Bind local APIs to loopback by default; do not expose the backend to untrusted networks.
- Keep credentials in host-controlled environment/secret references. Never pass secrets to the model or include them in tool schemas/results/audit payloads.
- Use least-privilege filesystem roots, executable allowlists, host-configured network allowlists and bounded request/response sizes.
- Treat model tool calls, tool output, plugin metadata, documents and browser content as untrusted data.
- Require exact-argument, one-time, expiring approval for high-impact actions.
- Authenticate the approval actor in the BrahmaDesk host layer; do not trust a caller-supplied actor label.
- Make audit failure fail closed before side effects. Define recovery behavior for failures after a side effect.
- Do not claim ULTK's in-process plugin manager is a sandbox. Untrusted plugins and parsers need isolated workers.
- Add OS-level isolation before enabling arbitrary command execution, untrusted document parsing or third-party plugin execution for general users.

## 8. Implementation phases and acceptance gates

### M0 — Integration baseline
- [x] Create an isolated BrahmaDesk integration branch.
- [ ] Record source branch/commit inventory and ULTK consolidation strategy.
- [ ] Add feature flags with all ULTK execution disabled by default.
- [ ] Add architecture and threat-boundary documentation.

**Gate:** BrahmaDesk main remains unchanged; integration work is reviewable and reversible.

### M1 — ULTK source consolidation and package validation
- [ ] Consolidate the comprehensive tool-pack chain and persistent-memory increment.
- [ ] Fix package metadata and dependency extras.
- [ ] Validate imports and public APIs.
- [ ] Run pytest and Ruff; add missing integration tests.
- [ ] Pin the validated ULTK commit in BrahmaDesk.

**Gate:** one reproducible ULTK source revision, no reliance on moving feature branches.

### M2 — Embedded runtime and inference adapter
- [ ] Initialize one ULTK runtime in FastAPI lifespan.
- [ ] Add a BrahmaDesk ProviderAdapter backed by the Brahma Engine.
- [ ] Add structured tool-call request/response contracts.
- [ ] Preserve existing normal chat streaming.
- [ ] Add bounded agent-run API and lifecycle events.

**Gate:** mock-provider tests pass; real llama.cpp tool-call smoke test passes on a supported chat template.

### M3 — Session, profile and memory adapters
- [ ] Add validated request context and server-side ownership checks.
- [ ] Bind every tool call to profile/conversation/project/session IDs.
- [ ] Adapt BrahmaDesk profile/project memory without duplicate storage.
- [ ] Add cancellation, deadlines and per-session concurrency controls.

**Gate:** profile-isolation and conversation-persistence regression suites pass.

### M4 — Full implemented tool ecosystem
- [ ] Integrate all implemented ULTK built-in packs.
- [ ] Add per-tool configuration, optional-dependency detection and enablement.
- [ ] Add worker boundary for high-risk/resource-heavy tools.
- [ ] Expose capability inventory and health diagnostics.

**Gate:** each enabled pack passes its focused tests and cannot exceed its configured scope.

### M5 — Approvals, audit and workflows
- [ ] Add authenticated approval endpoints and desktop approval surfaces.
- [ ] Bind approvals to exact tool + arguments + session + expiry.
- [ ] Add durable audit views and correlation IDs.
- [ ] Integrate bounded DAG workflows and cancellation.

**Gate:** approval replay/expiry/concurrency and audit integrity tests pass.

### M6 — Desktop experience
- [ ] Add tool-call timeline and execution states.
- [ ] Add approval/deny/cancel flows.
- [ ] Add tool enablement and capability settings.
- [ ] Render normalized errors and partial/incomplete run states.

**Gate:** Electron E2E tests cover chat-only and tool-using sessions.

### M7 — Windows packaging and release
- [ ] Update backend packaging and PyInstaller hidden imports/data.
- [ ] Include optional dependencies only when selected and supported.
- [ ] Handle Playwright browser installation/runtime packaging explicitly.
- [ ] Test clean install, upgrade, restart, data preservation and rollback.
- [ ] Produce a security/verification ledger for the integrated product.

**Gate:** packaged Windows build passes backend, live model, tool, approval, and desktop E2E tests.

## 9. Initial configuration proposal

All settings are host-owned and default to safe-off:

- `BRAHMA_ULTK_ENABLED=0`
- `BRAHMA_ULTK_MAX_STEPS=6`
- `BRAHMA_ULTK_MAX_CONCURRENT_TOOLS=2`
- `BRAHMA_ULTK_DEFAULT_POLICY=deny`
- `BRAHMA_ULTK_AUDIT_PATH=<BRAHMA_DATA_DIR>/ultk/audit.sqlite3`
- `BRAHMA_ULTK_APPROVAL_TTL_SECONDS=300`
- `BRAHMA_ULTK_WORKSPACE_ROOT=<explicit user-selected directory>`

Tool packs and capabilities must be configured per profile and host policy. A user-facing enable toggle cannot grant capabilities that the host has not authorized.

## 10. Known blockers discovered during repository inspection

- ULTK feature branches are not yet consolidated; the persistent-memory branch is separate from the later email/media/plugin branch chain.
- The inspected ULTK `pyproject.toml` contains literal escaped newline text in its optional-dependency section and needs correction before packaging.
- ULTK's current provider contract is non-streaming.
- Brahma Engine's current GenerationChunk carries text and finish metadata but no structured tool calls.
- The current BrahmaDesk ChatRequest accepts only system/user/assistant messages; tool messages must be kept inside the backend agent/provider path and must not be accepted as arbitrary desktop-supplied roles.
- The current chat stream persists only user/assistant message records. Agent tool messages and lifecycle events need explicit handling without corrupting existing history.
- ULTK's verification ledger defers comprehensive tests for many later tool packs and identifies important sandbox, SSRF, browser, approval/audit, plugin and cross-platform gaps.

## 11. Non-goals for initial cutover

- Do not replace llama.cpp or claim the native Brahma inference runtime is complete.
- Do not enable every tool by default.
- Do not expose arbitrary shell/Python execution.
- Do not expose FastAPI to LAN/Internet as part of this integration.
- Do not merge the integration branch into BrahmaDesk main until all relevant gates pass.
