from __future__ import annotations

import json
import socket
from pathlib import Path

from app import main


def test_find_available_llama_port_returns_bindable_loopback_port() -> None:
    port = main._find_available_llama_port()
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((main.LLAMA_HOST, port))


def test_persist_llama_endpoint_writes_port_for_restart(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "active-llama-endpoint.json"
    monkeypatch.setattr(main, "LLAMA_ENDPOINT_STATE_PATH", state_path)

    main._persist_llama_endpoint(18765)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {"host": main.LLAMA_HOST, "port": 18765}
    assert not state_path.with_suffix(".tmp").exists()
