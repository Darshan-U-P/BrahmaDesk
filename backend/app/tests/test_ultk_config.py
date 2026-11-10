from __future__ import annotations

from pathlib import Path

import pytest

from app.integrations.ultk.config import ULTKSettings


def test_ultk_is_disabled_and_deny_by_default(tmp_path: Path) -> None:
    settings = ULTKSettings.from_env({}, data_dir=tmp_path)

    assert settings.enabled is False
    assert settings.default_policy == "deny"
    assert settings.max_steps == 6
    assert settings.max_concurrent_tools == 2
    assert settings.audit_path == (tmp_path / "ultk" / "audit.sqlite3").resolve()
    assert settings.workspace_root is None
    assert settings.enabled_tools == frozenset()
    assert settings.enabled_packs == frozenset()
    assert settings.granted_capabilities == frozenset()
    assert settings.approval_capabilities == frozenset()
    assert settings.trusted_plugins == frozenset()


def test_ultk_settings_parse_host_owned_limits(tmp_path: Path) -> None:
    settings = ULTKSettings.from_env(
        {
            "BRAHMA_ULTK_ENABLED": "true",
            "BRAHMA_ULTK_MAX_STEPS": "8",
            "BRAHMA_ULTK_MAX_CONCURRENT_TOOLS": "3",
            "BRAHMA_ULTK_APPROVAL_TTL_SECONDS": "600",
            "BRAHMA_ULTK_DEFAULT_POLICY": "interactive",
            "BRAHMA_ULTK_ENABLED_TOOLS": "filesystem.read,documents.extract",
        },
        data_dir=tmp_path,
    )

    assert settings.enabled is True
    assert settings.max_steps == 8
    assert settings.max_concurrent_tools == 3
    assert settings.approval_ttl_seconds == 600
    assert settings.default_policy == "interactive"
    assert settings.enabled_tools == frozenset(
        {"filesystem.read", "documents.extract"}
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("BRAHMA_ULTK_ENABLED", "sometimes"),
        ("BRAHMA_ULTK_MAX_STEPS", "0"),
        ("BRAHMA_ULTK_MAX_CONCURRENT_TOOLS", "99"),
        ("BRAHMA_ULTK_APPROVAL_TTL_SECONDS", "0"),
        ("BRAHMA_ULTK_DEFAULT_POLICY", "allow-all"),
    ],
)
def test_ultk_settings_reject_invalid_values(
    tmp_path: Path, key: str, value: str
) -> None:
    with pytest.raises(ValueError):
        ULTKSettings.from_env({key: value}, data_dir=tmp_path)


def test_ultk_workspace_root_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        ULTKSettings.from_env(
            {"BRAHMA_ULTK_WORKSPACE_ROOT": str(tmp_path / "missing")},
            data_dir=tmp_path,
        )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = ULTKSettings.from_env(
        {"BRAHMA_ULTK_WORKSPACE_ROOT": str(workspace)},
        data_dir=tmp_path,
    )
    assert settings.workspace_root == workspace.resolve()



def test_ultk_settings_parse_packs_tools_and_capabilities(tmp_path: Path) -> None:
    settings = ULTKSettings.from_env(
        {
            "BRAHMA_ULTK_ENABLED_PACKS": "filesystem,documents,git",
            "BRAHMA_ULTK_ENABLED_TOOLS": "filesystem.read_file,documents.extract_text",
            "BRAHMA_ULTK_GRANTED_CAPABILITIES": "filesystem:read,documents:extract",
            "BRAHMA_ULTK_APPROVAL_CAPABILITIES": "filesystem:write",
            "BRAHMA_ULTK_TRUSTED_PLUGINS": "com.example.trusted",
        },
        data_dir=tmp_path,
    )

    assert settings.enabled_packs == frozenset({"filesystem", "documents", "git"})
    assert settings.enabled_tools == frozenset(
        {"filesystem.read_file", "documents.extract_text"}
    )
    assert settings.granted_capabilities == frozenset(
        {"filesystem:read", "documents:extract"}
    )
    assert settings.approval_capabilities == frozenset({"filesystem:write"})
    assert settings.trusted_plugins == frozenset({"com.example.trusted"})


def test_ultk_settings_reject_unknown_tool_packs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported ULTK pack"):
        ULTKSettings.from_env(
            {"BRAHMA_ULTK_ENABLED_PACKS": "unrestricted_shell"},
            data_dir=tmp_path,
        )
