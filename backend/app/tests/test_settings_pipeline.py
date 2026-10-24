"""Focused regression tests for Brahma Desk sampling/thinking settings."""

import importlib
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
# Keep test-created profile/database files outside a real user's application data.
os.environ.setdefault("BRAHMA_DATA_DIR", str(Path(__file__).resolve().parent / ".test-data"))
main = importlib.import_module("main")


def request(**overrides):
    data = {
        "messages": [{"role": "user", "content": "Test prompt"}],
        "reasoning_budget": "medium",
    }
    data.update(overrides)
    return main.ChatRequest.model_validate(data)


def test_reasoning_guidance_uses_real_newlines_and_preserves_system_message():
    req = request(reasoning_budget="low")
    payload = main.build_generation_payload(
        req,
        [{"role": "system", "content": "Base system instruction"},
         {"role": "user", "content": "Test prompt"}],
    )
    system_content = payload["messages"][0]["content"]
    assert "Base system instruction\n\nReasoning effort preference:" in system_content
    assert r"\n\nReasoning effort preference:" not in system_content
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_thinking_off_disables_template_flag_and_adds_no_guidance():
    req = request(reasoning_budget="off")
    payload = main.build_generation_payload(req, [{"role": "user", "content": "Test prompt"}])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(payload["messages"]) == 1
    assert "Reasoning effort preference" not in payload["messages"][0]["content"]


def test_sampling_values_are_forwarded_to_llama_payload():
    req = request(
        temperature=1.15,
        dynatemp_range=0.25,
        dynatemp_exponent=1.4,
        top_k=27,
        top_p=0.88,
        min_p=0.04,
        xtc_probability=0.2,
        xtc_threshold=0.12,
        typ_p=0.91,
        min_keep=3,
        max_tokens=777,
        sampler_order="top_k;typ_p;top_p;min_p;temperature",
        backend_sampling=True,
        repeat_last_n=96,
        repeat_penalty=1.12,
        presence_penalty=0.2,
        frequency_penalty=0.15,
        dry_multiplier=0.4,
        dry_base=1.9,
        dry_allowed_length=3,
        dry_penalty_last_n=128,
    )
    payload = main.build_generation_payload(req, [{"role": "user", "content": "Test prompt"}])
    expected = {
        "temperature": 1.15, "dynatemp_range": 0.25, "dynatemp_exponent": 1.4,
        "top_k": 27, "top_p": 0.88, "min_p": 0.04, "xtc_probability": 0.2,
        "xtc_threshold": 0.12, "typical_p": 0.91, "min_keep": 3, "max_tokens": 777,
        "samplers": ["top_k", "typ_p", "top_p", "min_p", "temperature"],
        "backend_sampling": True, "repeat_last_n": 96, "repeat_penalty": 1.12,
        "presence_penalty": 0.2, "frequency_penalty": 0.15, "dry_multiplier": 0.4,
        "dry_base": 1.9, "dry_allowed_length": 3, "dry_penalty_last_n": 128,
        "stream": True,
    }
    for key, value in expected.items():
        assert payload[key] == value, key


@pytest.mark.parametrize("field,value", [
    ("temperature", 2.1),
    ("top_p", -0.1),
    ("min_p", 1.1),
    ("max_tokens", -2),
    ("reasoning_budget", "extreme"),
    ("dry_base", 0.5),
])
def test_invalid_sampling_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        request(**{field: value})


@pytest.mark.parametrize("sampler_order", [
    "", " ; ; ", "top_k;made_up_sampler", "top_k;top_k",
])
def test_invalid_sampler_order_is_rejected(sampler_order):
    with pytest.raises(ValidationError):
        request(sampler_order=sampler_order)


def test_sampler_order_trims_whitespace():
    req = request(sampler_order=" top_k ; typ_p ; temperature ")
    assert req.sampler_order == "top_k;typ_p;temperature"


def test_api_rejects_invalid_sampling_request_with_http_422():
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    response = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "Test prompt"}],
        "temperature": 99,
    })
    assert response.status_code == 422


def test_backend_temperature_default_matches_frontend_default():
    assert request().temperature == 0.8
