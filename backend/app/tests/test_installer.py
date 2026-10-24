from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException

from app import installer


def test_catalog_ids_are_unique_and_paths_are_fixed() -> None:
    ids = [item["id"] for item in installer.CATALOG]
    assert len(ids) == len(set(ids))
    for item in installer.CATALOG:
        path = installer._model_path(item["id"])
        assert path.parent == installer.MODEL_DIR
        assert path.name == item["filename"]
        assert path.suffix == ".gguf"


def test_unknown_model_cannot_select_arbitrary_path() -> None:
    with pytest.raises(HTTPException) as error:
        installer._model_path("../../outside")
    assert error.value.status_code == 404


def test_recommendation_uses_ram_vram_and_disk_constraints() -> None:
    model = installer.CATALOG[0]
    low_memory = {
        "ram_gb": 4,
        "gpus": [],
        "model_directory_free_gb": 20,
    }
    status, reasons = installer._recommendation(model, low_memory)
    assert status == "unsupported"
    assert reasons

    capable = {
        "ram_gb": 32,
        "gpus": [{"vram_gb_reported": 12}],
        "model_directory_free_gb": 50,
    }
    status, reasons = installer._recommendation(model, capable)
    assert status == "recommended"
    assert reasons


def test_sha256_streaming_hash(tmp_path: Path) -> None:
    sample = tmp_path / "sample.gguf"
    sample.write_bytes(b"GGUF" + b"payload")
    assert installer._sha256(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_registry_only_returns_existing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(installer, "REGISTRY_DB", tmp_path / "test.sqlite3")
    entry = installer.CATALOG[0]
    model_path = tmp_path / entry["filename"]
    model_path.write_bytes(b"GGUF" + b"x" * 32)

    result = installer._register_model(
        entry, model_path, model_path.stat().st_size, installer._sha256(model_path),
        True, {"revision": "test"},
    )
    assert result["verified"] is True
    registered = installer._registry_rows()
    assert entry["id"] in registered
    assert registered[entry["id"]]["path"] == str(model_path)

    model_path.unlink()
    assert entry["id"] not in installer._registry_rows()


def test_scan_validate_and_configure_local_gguf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(installer, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(installer, "REGISTRY_DB", tmp_path / "test.sqlite3")
    installer.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = installer.MODEL_DIR / "existing-model.gguf"
    model_path.write_bytes(b"GGUF" + b"x" * (1024 * 1024))

    scan = asyncio.run(installer.scan_local_models())
    assert len(scan["discovered"]) == 1
    model_id = scan["discovered"][0]["model_id"]
    assert model_id.startswith("local-")

    validation = asyncio.run(installer.validate_model(model_id))
    assert validation["status"] == "valid"
    assert validation["official_checksum_verified"] is False

    configured = asyncio.run(installer.configure_inference(
        installer.ConfigureRequest(model_id=model_id, context_size=4096, mode="auto")
    ))
    assert configured["status"] == "configured"
    assert configured["config"]["model_path"] == str(model_path)
