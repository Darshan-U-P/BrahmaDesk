"""Host-owned, fail-closed configuration for the embedded ULTK runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SUPPORTED_PACKS = frozenset({
    "filesystem", "system", "process", "terminal", "computer",
    "computer_advanced", "computer_os", "computer_visual", "computer_vision",
    "browser", "browser_comprehensive", "browser_semantic", "development",
    "development_advanced", "git", "documents", "data", "structured_data",
    "database", "media", "rest", "email", "scheduler", "plugins", "memory",
})


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("BRAHMA_ULTK_ENABLED must be a boolean value")


def _parse_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = env.get(key)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _parse_set(env: Mapping[str, str], key: str) -> frozenset[str]:
    values = frozenset(
        item.strip()
        for item in env.get(key, "").split(",")
        if item.strip()
    )
    if any(len(item) > 128 for item in values):
        raise ValueError(f"{key} entries must be at most 128 characters")
    return values


def _default_data_dir(env: Mapping[str, str]) -> Path:
    override = env.get("BRAHMA_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    roaming = env.get("APPDATA")
    if roaming:
        return (Path(roaming).expanduser() / "brahma-desk").resolve()
    return (Path.home() / ".local" / "share" / "brahma-desk").resolve()


@dataclass(frozen=True, slots=True)
class ULTKSettings:
    """Validated host configuration. Enabling ULTK does not grant tool capabilities."""

    enabled: bool
    max_steps: int
    max_concurrent_tools: int
    default_policy: str
    audit_path: Path
    approval_ttl_seconds: int
    workspace_root: Path | None
    enabled_packs: frozenset[str]
    enabled_tools: frozenset[str]
    granted_capabilities: frozenset[str]
    approval_capabilities: frozenset[str]
    trusted_plugins: frozenset[str]

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        data_dir: Path | None = None,
    ) -> "ULTKSettings":
        source = os.environ if env is None else env
        enabled = _parse_bool(source.get("BRAHMA_ULTK_ENABLED"), default=False)
        max_steps = _parse_int(
            source, "BRAHMA_ULTK_MAX_STEPS", 6, minimum=1, maximum=12
        )
        max_concurrent_tools = _parse_int(
            source, "BRAHMA_ULTK_MAX_CONCURRENT_TOOLS", 2, minimum=1, maximum=8
        )
        approval_ttl_seconds = _parse_int(
            source, "BRAHMA_ULTK_APPROVAL_TTL_SECONDS", 300,
            minimum=1, maximum=86_400,
        )

        default_policy = source.get(
            "BRAHMA_ULTK_DEFAULT_POLICY", "deny"
        ).strip().casefold()
        if default_policy not in {"deny", "interactive"}:
            raise ValueError("BRAHMA_ULTK_DEFAULT_POLICY must be 'deny' or 'interactive'")

        root_value = source.get("BRAHMA_ULTK_WORKSPACE_ROOT", "").strip()
        workspace_root = Path(root_value).expanduser().resolve() if root_value else None
        if workspace_root is not None and (
            not workspace_root.exists() or not workspace_root.is_dir()
        ):
            raise ValueError("BRAHMA_ULTK_WORKSPACE_ROOT must point to an existing directory")

        enabled_packs = _parse_set(source, "BRAHMA_ULTK_ENABLED_PACKS")
        unknown_packs = sorted(enabled_packs - SUPPORTED_PACKS)
        if unknown_packs:
            raise ValueError("Unsupported ULTK pack(s): " + ", ".join(unknown_packs))

        enabled_tools = _parse_set(source, "BRAHMA_ULTK_ENABLED_TOOLS")
        granted_capabilities = _parse_set(source, "BRAHMA_ULTK_GRANTED_CAPABILITIES")
        approval_capabilities = _parse_set(source, "BRAHMA_ULTK_APPROVAL_CAPABILITIES")
        trusted_plugins = _parse_set(source, "BRAHMA_ULTK_TRUSTED_PLUGINS")

        root_data_dir = (data_dir or _default_data_dir(source)).expanduser().resolve()
        audit_path = Path(
            source.get(
                "BRAHMA_ULTK_AUDIT_PATH",
                str(root_data_dir / "ultk" / "audit.sqlite3"),
            )
        ).expanduser().resolve()

        return cls(
            enabled=enabled,
            max_steps=max_steps,
            max_concurrent_tools=max_concurrent_tools,
            default_policy=default_policy,
            audit_path=audit_path,
            approval_ttl_seconds=approval_ttl_seconds,
            workspace_root=workspace_root,
            enabled_packs=enabled_packs,
            enabled_tools=enabled_tools,
            granted_capabilities=granted_capabilities,
            approval_capabilities=approval_capabilities,
            trusted_plugins=trusted_plugins,
        )
