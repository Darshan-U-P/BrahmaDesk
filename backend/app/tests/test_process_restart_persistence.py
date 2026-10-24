import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def run_isolated_process(data_dir: Path, script: str) -> str:
    env = os.environ.copy()
    env["BRAHMA_DATA_DIR"] = str(data_dir)
    env["BRAHMA_BACKEND_DIR"] = str(BACKEND_DIR)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result.stdout.strip()


def test_conversation_survives_application_process_restart(tmp_path):
    data_dir = tmp_path / "restart-test-data"

    writer_script = r"""
import json, os, sys
sys.path.insert(0, os.environ["BRAHMA_BACKEND_DIR"])
import main
from fastapi.testclient import TestClient

client = TestClient(main.app)
response = client.post("/api/conversations/import", json={
    "conversation": {
        "title": "process-restart-persistence-test",
        "messages": [
            {"role": "user", "content": "restart-user-message-123"},
            {"role": "assistant", "content": "restart-assistant-message-456"},
        ],
    }
})
assert response.status_code == 200, response.text
print(json.dumps({"id": response.json()["id"]}))
"""

    output = run_isolated_process(data_dir, writer_script)
    conversation_id = json.loads(output.splitlines()[-1])["id"]

    # Start a separate interpreter and verify the persisted conversation.
    reader_script = r"""
import json, os, sys
sys.path.insert(0, os.environ["BRAHMA_BACKEND_DIR"])
import main
from fastapi.testclient import TestClient

client = TestClient(main.app)
conversation_id = os.environ["RESTART_TEST_CONVERSATION_ID"]
response = client.get(f"/api/conversations/{conversation_id}")
assert response.status_code == 200, response.text

conversation = response.json()
assert conversation["title"] == "process-restart-persistence-test"
assert [m["content"] for m in conversation["messages"]] == [
    "restart-user-message-123",
    "restart-assistant-message-456",
]
print("RESTART_PERSISTENCE_OK")
"""

    env = os.environ.copy()
    env["BRAHMA_DATA_DIR"] = str(data_dir)
    env["BRAHMA_BACKEND_DIR"] = str(BACKEND_DIR)
    env["RESTART_TEST_CONVERSATION_ID"] = conversation_id

    result = subprocess.run(
        [sys.executable, "-c", reader_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "RESTART_PERSISTENCE_OK" in result.stdout
