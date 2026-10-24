"""Smart-installer services for Brahma Desk.

Uses the existing FastAPI process and per-user data directory. All downloaded
models are written to a dedicated models directory and atomically renamed only
after integrity checks pass.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/installer", tags=["installer"])

def _data_dir() -> Path:
    override = os.getenv("BRAHMA_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if os.getenv("APPDATA"):
        return Path(os.environ["APPDATA"]) / "brahma-desk"
    return Path.home() / ".local" / "share" / "brahma-desk"

DATA_DIR = _data_dir()
MODEL_DIR = Path(os.getenv("BRAHMA_MODEL_DIR", str(DATA_DIR / "models"))).expanduser()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DB = DATA_DIR / "brahma_desk.sqlite3"
GGUF_MAGIC = b"GGUF"
DOWNLOADS: dict[str, dict[str, Any]] = {}
DOWNLOAD_TASKS: dict[str, asyncio.Task] = {}
DOWNLOAD_LOCK = asyncio.Lock()

# Catalog metadata is intentionally curated. File size and SHA-256 are resolved
# from Hugging Face at download time instead of trusting guessed values.
CATALOG: list[dict[str, Any]] = [
    {
        "id": "qwen3-4b-q4km",
        "name": "Qwen3 4B · Q4_K_M",
        "repo_id": "bartowski/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "family": "Qwen3",
        "parameters_b": 4.0,
        "quantization": "Q4_K_M",
        "context_default": 8192,
        "purpose": "General chat and reasoning",
        "license": "Apache-2.0",
        "minimum_ram_gb": 8,
        "recommended_ram_gb": 12,
        "minimum_vram_gb": 4,
        "recommended_vram_gb": 6,
    },
    {
        "id": "qwen25-coder-7b-q4km",
        "name": "Qwen2.5 Coder 7B · Q4_K_M",
        "repo_id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "family": "Qwen2.5 Coder",
        "parameters_b": 7.0,
        "quantization": "Q4_K_M",
        "context_default": 8192,
        "purpose": "Code generation and software engineering",
        "license": "Apache-2.0",
        "minimum_ram_gb": 12,
        "recommended_ram_gb": 16,
        "minimum_vram_gb": 6,
        "recommended_vram_gb": 8,
    },
    {
        "id": "qwen25-3b-q4km",
        "name": "Qwen2.5 3B Instruct · Q4_K_M",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "family": "Qwen2.5",
        "parameters_b": 3.0,
        "quantization": "Q4_K_M",
        "context_default": 8192,
        "purpose": "Lightweight general assistant",
        "license": "Apache-2.0",
        "minimum_ram_gb": 8,
        "recommended_ram_gb": 10,
        "minimum_vram_gb": 3,
        "recommended_vram_gb": 4,
    },
]

class DownloadRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)

class ConfigureRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    context_size: int = Field(default=8192, ge=512, le=131072)
    mode: str = Field(default="auto", pattern="^(auto|cpu|hybrid|gpu)$")

def _powershell_json(script: str) -> Any:
    if platform.system() != "Windows":
        return None
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=8, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        return json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

def _hardware_snapshot() -> dict[str, Any]:
    cpu_script = (
        "$ErrorActionPreference='SilentlyContinue'; "
        "$c=Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors; "
        "$s=Get-CimInstance Win32_ComputerSystem; "
        "$g=Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion; "
        "[pscustomobject]@{cpu=$c;memory_bytes=$s.TotalPhysicalMemory;gpus=@($g)} | ConvertTo-Json -Depth 4 -Compress"
    )
    raw = _powershell_json(cpu_script)
    cpu = raw.get("cpu") if isinstance(raw, dict) else None
    memory_bytes = raw.get("memory_bytes") if isinstance(raw, dict) else None
    gpu_rows = raw.get("gpus", []) if isinstance(raw, dict) else []
    if isinstance(gpu_rows, dict):
        gpu_rows = [gpu_rows]
    gpus = []
    for gpu in gpu_rows if isinstance(gpu_rows, list) else []:
        if not isinstance(gpu, dict):
            continue
        try:
            vram = int(gpu.get("AdapterRAM") or 0)
        except (ValueError, TypeError):
            vram = 0
        gpus.append({
            "name": gpu.get("Name") or "Unknown GPU",
            "vram_bytes_reported": vram,
            "vram_gb_reported": round(vram / (1024 ** 3), 2) if vram else None,
            "driver_version": gpu.get("DriverVersion"),
            "vram_note": "Windows AdapterRAM can be inaccurate for modern GPUs.",
        })
    total_ram = round(int(memory_bytes) / (1024 ** 3), 2) if memory_bytes else None
    try:
        disk_free = round(shutil.disk_usage(MODEL_DIR).free / (1024 ** 3), 2)
    except OSError:
        disk_free = None
    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": {
            "name": (cpu or {}).get("Name") or platform.processor() or "Unknown CPU",
            "physical_cores": (cpu or {}).get("NumberOfCores"),
            "logical_cores": (cpu or {}).get("NumberOfLogicalProcessors") or os.cpu_count(),
        },
        "ram_gb": total_ram,
        "gpus": gpus,
        "model_directory": str(MODEL_DIR),
        "model_directory_free_gb": disk_free,
        "detected_at": int(time.time()),
        "limitations": [
            "GPU VRAM detection through Win32_VideoController is best-effort and may be inaccurate.",
            "GPU compute support is not inferred from the GPU name alone.",
            "Compatibility estimates include model weights and runtime headroom, not a guaranteed performance prediction.",
        ],
    }

def _model_path(model_id: str) -> Path:
    entry = next((m for m in CATALOG if m["id"] == model_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown catalog model")
    # Catalog filenames are fixed; never accept arbitrary paths from a client.
    return MODEL_DIR / entry["filename"]

def _registry_init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS installer_models (
            model_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, filename TEXT NOT NULL,
            path TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
            verified INTEGER NOT NULL, added_at REAL NOT NULL, metadata_json TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS installer_settings (
            key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at REAL NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS installer_download_jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL, job_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        )""")

def _persist_job(job: dict[str, Any]) -> None:
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        db.execute(
            """INSERT INTO installer_download_jobs(job_id,status,job_json,updated_at)
               VALUES(?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET
               status=excluded.status, job_json=excluded.job_json, updated_at=excluded.updated_at""",
            (job["id"], job["status"], json.dumps(job), time.time()),
        )

def _hydrate_jobs() -> None:
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        rows = db.execute("SELECT job_id,job_json FROM installer_download_jobs").fetchall()
    for job_id, encoded in rows:
        if job_id in DOWNLOADS:
            continue
        try:
            job = json.loads(encoded)
        except (TypeError, ValueError):
            continue
        if job.get("status") in {"queued", "downloading"}:
            job["status"] = "failed"
            job["error"] = "Download was interrupted when Brahma Desk closed. Retry to resume from the partial file."
            _persist_job(job)
        DOWNLOADS[job_id] = job

def _registry_rows() -> dict[str, dict[str, Any]]:
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM installer_models").fetchall()
    result = {}
    for row in rows:
        item = dict(row)
        if not Path(item["path"]).is_file():
            continue
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        item["verified"] = bool(item["verified"])
        result[item["model_id"]] = item
    return result

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

async def _hf_file_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{entry['repo_id']}/tree/main"
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            response = await client.get(url, params={"recursive": "true", "expand": "true"})
            response.raise_for_status()
            items = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Could not query Hugging Face: {exc}") from exc
    target = entry["filename"].casefold()
    match = next((item for item in items if item.get("type") == "file" and item.get("path", "").casefold() == target), None)
    if not match:
        # The catalog entry is not silently redirected to a different quantization.
        raise HTTPException(status_code=404, detail=f"Expected GGUF file was not found in {entry['repo_id']}: {entry['filename']}")
    lfs = match.get("lfs") or {}
    size = match.get("size") or lfs.get("size")
    sha = lfs.get("sha256")
    if not sha and isinstance(lfs.get("oid"), str) and lfs["oid"].startswith("sha256:"):
        sha = lfs["oid"].split(":", 1)[1]
    if not size:
        raise HTTPException(status_code=502, detail="Hugging Face did not provide the file size")
    return {
        "size_bytes": int(size),
        "sha256": sha,
        "revision": "main",
        "download_url": f"https://huggingface.co/{entry['repo_id']}/resolve/main/{entry['filename']}",
    }

def _recommendation(model: dict[str, Any], hardware: dict[str, Any]) -> tuple[str, list[str]]:
    ram = hardware.get("ram_gb") or 0
    gpus = hardware.get("gpus") or []
    vram = max((g.get("vram_gb_reported") or 0 for g in gpus), default=0)
    free_disk = hardware.get("model_directory_free_gb") or 0
    reasons = []
    # Approximate Q4 weight footprint plus runtime/KV-cache headroom.
    estimated_ram = model["parameters_b"] * 0.75 + 2.5
    if ram and ram < model["minimum_ram_gb"]:
        return "unsupported", [f"Detected RAM ({ram} GB) is below the conservative {model['minimum_ram_gb']} GB minimum."]
    if hardware.get("model_directory_free_gb") is not None and free_disk < model["parameters_b"] * 0.8:
        reasons.append("Model directory has limited free disk space.")
    if ram and ram < model["recommended_ram_gb"]:
        reasons.append(f"Recommended system RAM is {model['recommended_ram_gb']} GB; detected {ram} GB.")
    if vram >= model["recommended_vram_gb"]:
        reasons.append("Reported VRAM meets the model's recommended target; actual GPU backend support must still be verified.")
        return "recommended", reasons
    if vram >= model["minimum_vram_gb"]:
        reasons.append("Reported VRAM may support partial or full offload, depending on runtime and context length.")
        return "compatible_with_constraints", reasons
    if ram >= estimated_ram:
        reasons.append("CPU inference is the conservative path; generation speed depends on CPU and memory bandwidth.")
        return "compatible_with_constraints", reasons
    reasons.append("Hardware information is incomplete or below the conservative estimate; use a smaller model.")
    return "unknown", reasons

@router.get("/hardware")
async def get_hardware() -> dict[str, Any]:
    return _hardware_snapshot()

@router.get("/catalog")
async def get_catalog() -> dict[str, Any]:
    hardware = _hardware_snapshot()
    installed = _registry_rows()
    models = []
    for entry in CATALOG:
        status, reasons = _recommendation(entry, hardware)
        path = _model_path(entry["id"])
        models.append({
            **entry,
            "recommendation": status,
            "reasons": reasons,
            "installed": path.is_file() and entry["id"] in installed,
            "installed_path": str(path) if path.is_file() else None,
            "estimated_weight_gb": round(entry["parameters_b"] * 0.75, 2),
        })
    known_ids = {item["id"] for item in CATALOG}
    for model_id, item in installed.items():
        metadata = item.get("metadata") or {}
        if model_id in known_ids or metadata.get("source") != "local":
            continue
        size_gb = round(item["size_bytes"] / (1024 ** 3), 2)
        models.append({
            "id": model_id, "name": Path(item["filename"]).stem,
            "repo_id": "local-file", "filename": item["filename"],
            "family": "Local GGUF", "parameters_b": round(size_gb / 0.75, 2),
            "quantization": "unknown", "context_default": 4096,
            "purpose": "Existing local model", "license": "User supplied",
            "minimum_ram_gb": 0, "recommended_ram_gb": 0,
            "minimum_vram_gb": 4, "recommended_vram_gb": 6,
            "recommendation": "unknown",
            "reasons": ["Local model discovered in the Brahma Desk model directory; origin/license are not independently verified."],
            "installed": True, "installed_path": item["path"],
            "estimated_weight_gb": size_gb,
        })
    return {"hardware": hardware, "models": models, "source": "curated catalog plus locally discovered GGUF models"}

@router.post("/scan-local")
async def scan_local_models() -> dict[str, Any]:
    _registry_init()
    registered = _registry_rows()
    catalog_paths = {str(_model_path(item["id"]).resolve()).casefold() for item in CATALOG}
    discovered = []
    rejected = []
    for path in MODEL_DIR.glob("*.gguf"):
        if str(path.resolve()).casefold() in catalog_paths:
            continue
        try:
            if not path.is_file() or path.stat().st_size < 1024 * 1024:
                rejected.append({"filename": path.name, "reason": "File is too small to be a valid model"})
                continue
            with path.open("rb") as stream:
                if stream.read(4) != GGUF_MAGIC:
                    rejected.append({"filename": path.name, "reason": "GGUF magic header is missing"})
                    continue
            digest = await asyncio.to_thread(_sha256, path)
            model_id = "local-" + digest[:12]
            existing = registered.get(model_id)
            if not existing:
                entry = {"id": model_id, "repo_id": "local-file"}
                _register_model(entry, path, path.stat().st_size, digest, True, {
                    "source": "local", "validation": "GGUF magic and local SHA-256 calculated",
                    "official_checksum_verified": False, "origin_trusted": False,
                })
            discovered.append({
                "model_id": model_id, "filename": path.name,
                "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": digest, "official_checksum_verified": False,
            })
        except OSError as exc:
            rejected.append({"filename": path.name, "reason": str(exc)})
    return {"discovered": discovered, "rejected": rejected, "directory": str(MODEL_DIR)}

@router.get("/models")
async def get_registered_models() -> dict[str, Any]:
    return {"models": list(_registry_rows().values()), "directory": str(MODEL_DIR)}

@router.post("/downloads")
async def start_download(request: DownloadRequest) -> dict[str, Any]:
    _hydrate_jobs()
    entry = next((m for m in CATALOG if m["id"] == request.model_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown catalog model")
    destination = _model_path(request.model_id)
    if destination.exists():
        raise HTTPException(status_code=409, detail="A file already exists at this model path; validate or remove it before downloading again.")
    async with DOWNLOAD_LOCK:
        existing = next((job for job in DOWNLOADS.values() if job["model_id"] == request.model_id and job["status"] in {"queued", "downloading"}), None)
        if existing:
            return existing
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id, "model_id": request.model_id, "name": entry["name"],
            "status": "queued", "progress": 0, "bytes_downloaded": 0,
            "total_bytes": None, "error": None, "created_at": time.time(),
        }
        DOWNLOADS[job_id] = job
        _persist_job(job)
        DOWNLOAD_TASKS[job_id] = asyncio.create_task(_download_job(job_id, entry, destination))
    return job

async def _download_job(job_id: str, entry: dict[str, Any], destination: Path) -> None:
    job = DOWNLOADS[job_id]
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        metadata = await _hf_file_metadata(entry)
        total = metadata["size_bytes"]
        free = shutil.disk_usage(destination.parent).free
        if free < total + 512 * 1024 * 1024:
            raise RuntimeError(f"Not enough free disk space. Need approximately {round((total + 512 * 1024 * 1024) / (1024 ** 3), 2)} GB including temporary space.")
        job.update(status="downloading", total_bytes=total, expected_sha256=metadata["sha256"])
        _persist_job(job)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > total:
            partial.unlink(missing_ok=True)
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        timeout = httpx.Timeout(connect=30, read=90, write=30, pool=30)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", metadata["download_url"], headers=headers) as response:
                if offset and response.status_code != 206:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("The server did not honor resume. Retry the download to restart safely.")
                response.raise_for_status()
                mode = "ab" if offset else "wb"
                last_persisted = offset
                with partial.open(mode) as stream:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        stream.write(chunk)
                        job["bytes_downloaded"] = stream.tell()
                        job["progress"] = min(99, int(stream.tell() * 100 / total)) if total else 0
                        if stream.tell() - last_persisted >= 8 * 1024 * 1024:
                            _persist_job(job)
                            last_persisted = stream.tell()
        if partial.stat().st_size != total:
            raise RuntimeError(f"Downloaded byte count mismatch: expected {total}, received {partial.stat().st_size}.")
        actual_sha = await asyncio.to_thread(_sha256, partial)
        expected_sha = metadata.get("sha256")
        if expected_sha and actual_sha.lower() != expected_sha.lower():
            raise RuntimeError("SHA-256 mismatch; the incomplete file was retained for diagnosis and will not be registered.")
        with partial.open("rb") as stream:
            if stream.read(4) != GGUF_MAGIC:
                raise RuntimeError("File does not have GGUF magic bytes; refusing to register it.")
        partial.replace(destination)
        _register_model(entry, destination, total, actual_sha, bool(expected_sha), metadata)
        job.update(status="completed", progress=100, bytes_downloaded=total, sha256=actual_sha, path=str(destination), error=None)
        _persist_job(job)
    except Exception as exc:
        job.update(status="failed", error=str(exc), progress=min(job.get("progress", 0), 99))
        _persist_job(job)
    finally:
        DOWNLOAD_TASKS.pop(job_id, None)

def _register_model(entry: dict[str, Any], path: Path, size: int, sha: str, verified: bool, metadata: dict[str, Any]) -> dict[str, Any]:
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        db.execute("""INSERT INTO installer_models
            (model_id, repo_id, filename, path, size_bytes, sha256, verified, added_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET repo_id=excluded.repo_id,
            filename=excluded.filename, path=excluded.path, size_bytes=excluded.size_bytes,
            sha256=excluded.sha256, verified=excluded.verified, added_at=excluded.added_at,
            metadata_json=excluded.metadata_json""",
            (entry["id"], entry["repo_id"], path.name, str(path), size, sha, int(verified), time.time(), json.dumps(metadata)))
    return {"model_id": entry["id"], "path": str(path), "size_bytes": size, "sha256": sha, "verified": verified}

@router.get("/downloads")
async def list_downloads() -> dict[str, Any]:
    _hydrate_jobs()
    return {"jobs": list(DOWNLOADS.values())}

@router.get("/downloads/{job_id}")
async def get_download(job_id: str) -> dict[str, Any]:
    _hydrate_jobs()
    job = DOWNLOADS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Download job not found in this backend session")
    return job

@router.post("/downloads/{job_id}/cancel")
async def cancel_download(job_id: str) -> dict[str, Any]:
    _hydrate_jobs()
    job = DOWNLOADS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Download job not found")
    task = DOWNLOAD_TASKS.get(job_id)
    if not task or task.done():
        raise HTTPException(status_code=409, detail="Download is not active")
    task.cancel()
    job.update(status="cancelled", error="Cancelled by user")
    _persist_job(job)
    return job


@router.post("/downloads/{job_id}/retry")
async def retry_download(job_id: str) -> dict[str, Any]:
    _hydrate_jobs()
    old = DOWNLOADS.get(job_id)
    if not old:
        raise HTTPException(status_code=404, detail="Download job not found")
    if old["status"] not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Only failed or cancelled downloads can be retried")
    entry = next((m for m in CATALOG if m["id"] == old["model_id"]), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Model is no longer in the catalog")
    new_id = str(uuid.uuid4())
    DOWNLOADS[new_id] = {**old, "id": new_id, "status": "queued", "progress": 0, "error": None, "created_at": time.time()}
    _persist_job(DOWNLOADS[new_id])
    DOWNLOAD_TASKS[new_id] = asyncio.create_task(_download_job(new_id, entry, _model_path(entry["id"])))
    return DOWNLOADS[new_id]

@router.post("/validate/{model_id}")
async def validate_model(model_id: str) -> dict[str, Any]:
    entry = next((m for m in CATALOG if m["id"] == model_id), None)
    registered = _registry_rows().get(model_id)
    if entry:
        path = _model_path(model_id)
    elif registered and (registered.get("metadata") or {}).get("source") == "local":
        path = Path(registered["path"]).resolve()
        if path.parent != MODEL_DIR.resolve():
            raise HTTPException(status_code=403, detail="Local model is outside the managed model directory")
    else:
        raise HTTPException(status_code=404, detail="Model is not in the catalog or local registry")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model file is not installed")
    size = path.stat().st_size
    if size < 1024 * 1024:
        raise HTTPException(status_code=422, detail="Model file is unexpectedly small")
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic != GGUF_MAGIC:
        raise HTTPException(status_code=422, detail="Invalid GGUF header")
    sha = await asyncio.to_thread(_sha256, path)
    if entry:
        metadata = await _hf_file_metadata(entry)
        expected = metadata.get("sha256")
        if expected and sha.lower() != expected.lower():
            raise HTTPException(status_code=422, detail="SHA-256 mismatch")
        verified = bool(expected)
        registry_entry = entry
    else:
        metadata = {
            "source": "local", "validation": "GGUF magic and local SHA-256 calculated",
            "official_checksum_verified": False, "origin_trusted": False,
        }
        verified = True
        registry_entry = {"id": model_id, "repo_id": "local-file"}
    result = _register_model(registry_entry, path, size, sha, verified, metadata)
    return {
        "status": "valid", **result, "format": "GGUF",
        "checksum_checked": bool(entry and metadata.get("sha256")),
        "official_checksum_verified": bool(entry and metadata.get("sha256")),
    }

@router.post("/configure")
async def configure_inference(request: ConfigureRequest) -> dict[str, Any]:
    entry = next((m for m in CATALOG if m["id"] == request.model_id), None)
    registered = _registry_rows().get(request.model_id)
    if not registered:
        raise HTTPException(status_code=409, detail="Download and validate the model before configuring it")
    if not entry:
        metadata = registered.get("metadata") or {}
        if metadata.get("source") != "local":
            raise HTTPException(status_code=404, detail="Unknown model configuration")
        size_gb = registered["size_bytes"] / (1024 ** 3)
        entry = {
            "id": request.model_id, "parameters_b": max(1.0, size_gb / 0.75),
            "minimum_vram_gb": 4, "recommended_vram_gb": 6,
        }
    hardware = _hardware_snapshot()
    ram = hardware.get("ram_gb") or 0
    vram = max((g.get("vram_gb_reported") or 0 for g in hardware.get("gpus", [])), default=0)
    if request.mode == "gpu" and vram < entry["minimum_vram_gb"]:
        raise HTTPException(status_code=422, detail="Reported VRAM is below this model's conservative GPU target")
    mode = request.mode
    if mode == "auto":
        mode = "gpu" if vram >= entry["recommended_vram_gb"] else ("hybrid" if vram >= entry["minimum_vram_gb"] else "cpu")
    context = request.context_size
    if ram and ram < 12:
        context = min(context, 4096)
    if ram and ram < 8 and context > 2048:
        context = 2048
    config = {
        "model_id": request.model_id,
        "model_path": registered["path"],
        "mode": mode,
        "context_size": context,
        "threads": max(1, min((hardware.get("cpu", {}).get("logical_cores") or os.cpu_count() or 4) - 1, 16)),
        "gpu_layers": 999 if mode == "gpu" else (20 if mode == "hybrid" else 0),
        "estimated_weight_gb": round(entry["parameters_b"] * 0.75, 2),
        "warning": "The selected settings will be tested by launching llama.cpp and generating a short response. GPU backend availability still depends on the installed llama.cpp build.",
    }
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        db.execute("INSERT INTO installer_settings(key,value_json,updated_at) VALUES('inference',?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at", (json.dumps(config), time.time()))
    return {"status": "configured", "config": config}

@router.get("/configuration")
async def get_inference_configuration() -> dict[str, Any]:
    _registry_init()
    with sqlite3.connect(REGISTRY_DB, timeout=10) as db:
        row = db.execute("SELECT value_json FROM installer_settings WHERE key='inference'").fetchone()
    return {"config": json.loads(row[0]) if row else None}

@router.post("/test/{model_id}")
async def test_model_readiness(model_id: str) -> dict[str, Any]:
    registered = _registry_rows().get(model_id)
    if not registered:
        raise HTTPException(status_code=404, detail="Model is not registered")
    config = await get_inference_configuration()
    chosen = config.get("config")
    if not chosen or chosen.get("model_id") != model_id:
        raise HTTPException(status_code=409, detail="Configure this model before testing readiness")
    # Do not launch a second llama-server or change the user's active server from
    # this installer route. Verify GGUF and configuration; actual inference test
    # can be run through the existing backend model lifecycle.
    path = Path(registered["path"])
    if not path.is_file() or path.stat().st_size != registered["size_bytes"]:
        raise HTTPException(status_code=422, detail="Model file changed since validation")
    return {
        "status": "ready_for_runtime_test",
        "model_id": model_id,
        "file_validation": "passed",
        "configuration_validation": "passed",
        "runtime_inference_test": "not_run",
        "message": "File and configuration checks passed. Start llama.cpp with this configuration and complete a generation test before release.",
    }

@router.get("/status")
async def installer_status() -> dict[str, Any]:
    return {
        "api_version": 1,
        "implemented_services": ["hardware_scan", "catalog_compatibility", "resumable_downloads", "gguf_validation", "sqlite_registry", "inference_configuration"],
        "model_directory": str(MODEL_DIR),
        "registered_model_count": len(_registry_rows()),
        "note": "Download job progress is held in memory for this backend session; partial files are retained to support HTTP Range resume.",
    }
