"""Embedded ULTK runtime composition for BrahmaDesk."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.engine.supervisor import EngineSupervisor

from .config import ULTKSettings
from .provider import BrahmaEngineProvider

AgentEventHandler = Callable[[Mapping[str, Any]], Awaitable[None] | None]
ApprovalEventHandler = Callable[[Any], Awaitable[None] | None]


class ULTKRuntime:
    """Host-owned ULTK engine, provider, agent and workflow composition."""

    def __init__(
        self,
        *,
        settings: ULTKSettings,
        tool_engine: Any,
        agent_runtime: Any,
        workflow_engine: Any,
        provider: BrahmaEngineProvider,
    ) -> None:
        self.settings = settings
        self.tool_engine = tool_engine
        self.agent_runtime = agent_runtime
        self.workflow_engine = workflow_engine
        self.provider = provider

    @property
    def available_tools(self) -> tuple[str, ...]:
        return tuple(sorted(tool.spec.name for tool in self.tool_engine.registry.list()))

    async def run(
        self,
        prompt: str,
        *,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        tool_metadata: Mapping[str, Any],
        on_event: AgentEventHandler | None = None,
    ) -> str:
        return await self.agent_runtime.run(
            prompt,
            session_id=session_id,
            enabled_tools=self.settings.enabled_tools,
            granted_capabilities=self.settings.granted_capabilities,
            messages=messages,
            tool_metadata=tool_metadata,
            on_event=on_event,
        )


def _require_workspace(settings: ULTKSettings, pack: str) -> Path:
    if settings.workspace_root is None:
        raise ValueError(
            f"ULTK pack '{pack}' requires BRAHMA_ULTK_WORKSPACE_ROOT "
            "to point to an explicitly selected existing directory."
        )
    return settings.workspace_root


def create_ultk_runtime(
    supervisor: EngineSupervisor,
    settings: ULTKSettings,
    *,
    on_approval_request: ApprovalEventHandler | None = None,
    memory_tools: Sequence[Any] | None = None,
    grounder: Any = None,
) -> ULTKRuntime:
    """Create one embedded ULTK runtime using host-owned policy and paths.

    Tool packs are registered only when explicitly enabled. Tool exposure is a
    separate exact-name allowlist, and capability grants are separately required.
    """
    if not settings.enabled:
        raise ValueError("ULTK runtime creation requested while BRAHMA_ULTK_ENABLED is false")

    from ultk.agents.runtime import AgentRuntime
    from ultk.engine import ToolEngine
    from ultk.permissions import AllowListPolicy, DenyAllPolicy

    if "database" in settings.enabled_packs and not settings.database_profiles:
        raise ValueError("The database pack requires BRAHMA_ULTK_DATABASE_PROFILES_JSON.")
    if "rest" in settings.enabled_packs and not settings.rest_endpoints:
        raise ValueError("The REST pack requires BRAHMA_ULTK_REST_ENDPOINTS_JSON.")
    if "email" in settings.enabled_packs and not settings.smtp_config:
        raise ValueError("The email pack requires BRAHMA_ULTK_SMTP_CONFIG_JSON.")
    if "memory" in settings.enabled_packs and not memory_tools:
        raise ValueError("The memory pack requires the BrahmaDesk memory adapter.")
    if "computer_grounding" in settings.enabled_packs and grounder is None:
        raise ValueError("The computer_grounding pack requires a host-provided visual grounder.")

    sensitive_markers = (
        "write", "delete", "execute", "send", "commit", "push",
        "publish", "launch", "input", "terminate", "shutdown", "install",
    )
    sensitive_grants = {
        capability
        for capability in settings.granted_capabilities
        if any(marker in capability.casefold() for marker in sensitive_markers)
    }
    missing_approvals = sensitive_grants - settings.approval_capabilities
    if missing_approvals:
        raise ValueError(
            "Sensitive capabilities must also require human approval: "
            + ", ".join(sorted(missing_approvals))
        )

    policy = (
        DenyAllPolicy()
        if settings.default_policy == "deny"
        else AllowListPolicy(settings.enabled_tools)
    )
    tool_engine = ToolEngine(policy=policy)

    app_state = settings.audit_path.parent
    app_state.mkdir(parents=True, exist_ok=True)
    tool_engine.enable_audit(settings.audit_path)

    if settings.approval_capabilities:
        tool_engine.require_approval(
            settings.audit_path.with_name("approvals.sqlite3"),
            approval_capabilities=settings.approval_capabilities,
            ttl_seconds=settings.approval_ttl_seconds,
            on_request=on_approval_request,
        )

    packs = settings.enabled_packs
    workspace = settings.workspace_root
    state_dir = app_state / "tool-state"
    state_dir.mkdir(parents=True, exist_ok=True)

    if "filesystem" in packs:
        tool_engine.install_filesystem_tools(_require_workspace(settings, "filesystem"))
    if "system" in packs:
        tool_engine.install_system_tools()
    if "process" in packs:
        tool_engine.install_process_management_tools()
    if "terminal" in packs:
        tool_engine.install_terminal_tools(
            _require_workspace(settings, "terminal"),
            allowed_executables=settings.allowed_executables,
        )
    if "computer" in packs:
        tool_engine.install_computer_tools(
            image_root=state_dir / "screenshots",
            allowed_apps=settings.allowed_apps,
        )
    if "computer_advanced" in packs:
        tool_engine.install_advanced_computer_tools(
            image_root=state_dir / "screenshots",
        )
    if "computer_os" in packs:
        tool_engine.install_computer_os_tools(
            filesystem_root=_require_workspace(settings, "computer_os"),
            state_database=state_dir / "computer-os.sqlite3",
        )
    if "computer_visual" in packs:
        tool_engine.install_computer_visual_tools()
    if "computer_vision" in packs:
        tool_engine.install_computer_vision_tools(
            template_root=state_dir / "computer-templates",
        )
    if "computer_grounding" in packs:
        tool_engine.install_computer_grounding_tools(grounder=grounder)
    if "browser" in packs:
        tool_engine.install_advanced_browser_tools(
            session_root=state_dir / "browser-sessions",
            monitor_database=state_dir / "browser-monitors.sqlite3",
        )
    if "browser_comprehensive" in packs:
        tool_engine.install_comprehensive_browser_tools(
            profile_root=state_dir / "browser-profiles",
            download_root=state_dir / "downloads",
        )
    if "browser_semantic" in packs:
        tool_engine.install_semantic_browser_tools(
            session_root=state_dir / "browser-semantic-sessions",
        )
    if "development" in packs:
        tool_engine.install_development_tools(
            project_root=_require_workspace(settings, "development"),
            allowed_executables=settings.allowed_executables,
        )
    if "development_advanced" in packs:
        tool_engine.install_advanced_development_tools(
            project_root=_require_workspace(settings, "development_advanced"),
        )
    if "git" in packs:
        tool_engine.install_advanced_git_tools(
            project_root=_require_workspace(settings, "git"),
            worktree_root=state_dir / "git-worktrees",
        )
    if "documents" in packs:
        tool_engine.install_document_tools(
            root_dir=_require_workspace(settings, "documents"),
        )
    if "web" in packs:
        tool_engine.install_web_tools(
            download_root=state_dir / "downloads",
        )
    if "data" in packs:
        tool_engine.install_data_tools(
            root_dir=_require_workspace(settings, "data"),
            sqlite_database=state_dir / "workspace-data.sqlite3",
            allow_database_writes=settings.allow_database_writes,
        )
    if "structured_data" in packs:
        tool_engine.install_structured_data_tools(
            root_dir=_require_workspace(settings, "structured_data"),
        )
    if "database" in packs:
        tool_engine.install_database_tools(settings.database_profiles)
    if "rest" in packs:
        tool_engine.install_rest_tools(
            settings.rest_endpoints,
            allow_writes=settings.allow_rest_writes,
        )
    if "email" in packs:
        tool_engine.install_email_tools(**settings.smtp_config)
    if "memory" in packs:
        tool_engine.register_tools(memory_tools or ())
    if "media" in packs:
        tool_engine.install_media_tools(
            root_dir=state_dir / "media",
        )
    if "scheduler" in packs:
        tool_engine.install_scheduler_tools(
            database=state_dir / "scheduler.sqlite3",
            handlers={},
        )
    if "plugins" in packs:
        tool_engine.configure_plugins(trusted_plugins=settings.trusted_plugins)

    registered = {tool.spec.name for tool in tool_engine.registry.list()}
    unknown_tools = sorted(settings.enabled_tools - registered)
    if unknown_tools:
        raise ValueError(
            "BRAHMA_ULTK_ENABLED_TOOLS contains tools not registered by the selected packs: "
            + ", ".join(unknown_tools)
        )

    provider = BrahmaEngineProvider(supervisor)
    agent_runtime = AgentRuntime(
        provider,
        tool_engine.registry,
        tool_engine.dispatcher,
        max_steps=settings.max_steps,
    )
    workflow_engine = tool_engine.create_workflow_engine(
        database=state_dir / "workflows.sqlite3",
        max_steps=settings.max_steps,
        max_concurrency=settings.max_concurrent_tools,
    )
    return ULTKRuntime(
        settings=settings,
        tool_engine=tool_engine,
        agent_runtime=agent_runtime,
        workflow_engine=workflow_engine,
        provider=provider,
    )
