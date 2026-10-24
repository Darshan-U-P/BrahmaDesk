import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault(
    "BRAHMA_DATA_DIR",
    str(Path(__file__).resolve().parent / ".test-data"),
)

main = importlib.import_module("main")


@pytest.fixture
def client():
    # Do not enter the TestClient context manager: its lifespan may start
    # a real llama.cpp process. These tests mock model connectivity directly.
    return TestClient(main.app)


def test_health_reports_unreachable_model_server(client):
    async def unavailable(*args, **kwargs):
        raise httpx.ConnectError("simulated model server outage")

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        instance = mock_client.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=unavailable)

        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["backend"] == "ok"
    assert response.json()["llama_server"] == "unreachable"


def test_health_recovers_when_model_server_returns(client):
    async def available(*args, **kwargs):
        return httpx.Response(
            200,
            json={"data": [{"id": "test-model"}]},
            request=httpx.Request("GET", "http://127.0.0.1:8080/v1/models"),
        )

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        instance = mock_client.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=available)

        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["backend"] == "ok"
    assert response.json()["llama_server"] == "ok"
    assert response.json()["model"] == "test-model"


def test_health_recovers_after_temporary_outage(client):
    with patch.object(main.httpx, "AsyncClient") as mock_client:
        instance = mock_client.return_value.__aenter__.return_value
        instance.get = AsyncMock(
            side_effect=[
                httpx.ConnectError("temporary outage"),
                httpx.Response(
                    200,
                    json={"data": [{"id": "test-model"}]},
                    request=httpx.Request(
                        "GET", "http://127.0.0.1:8080/v1/models"
                    ),
                ),
            ]
        )

        first = client.get("/api/health")
        second = client.get("/api/health")

    assert first.status_code == 200
    assert first.json()["llama_server"] == "unreachable"
    assert second.status_code == 200
    assert second.json()["llama_server"] == "ok"
    assert second.json()["model"] == "test-model"
