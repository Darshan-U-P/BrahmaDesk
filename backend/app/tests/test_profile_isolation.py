import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault(
    "BRAHMA_DATA_DIR",
    str(Path(__file__).resolve().parent / ".test-data"),
)

main = importlib.import_module("main")


def test_conversations_are_isolated_between_profiles():
    with TestClient(main.app) as client:
        # Record the default profile.
        response = client.get("/api/active-profile")
        assert response.status_code == 200
        default_id = response.json()["id"]

        # Create a conversation under the default profile.
        response = client.post(
            "/api/conversations",
            json={"title": "Default profile conversation"},
        )
        assert response.status_code == 200
        conversation_id = response.json()["id"]

        # Create and activate a separate profile.
        response = client.post(
            "/api/profiles",
            json={"display_name": "Isolation Test"},
        )
        assert response.status_code == 200
        other_profile_id = response.json()["id"]

        response = client.put(
            "/api/active-profile",
            json={"profile_id": other_profile_id},
        )
        assert response.status_code == 200

        response = client.post(
            "/api/conversations",
            json={"title": "Other profile conversation"},
        )
        assert response.status_code == 200
        other_conversation_id = response.json()["id"]

        # The other profile must not see the original conversation.
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert conversation_id not in [
            item["id"] for item in response.json()["items"]
        ]
        assert other_conversation_id in [
            item["id"] for item in response.json()["items"]
        ]
        assert client.get(
            f"/api/conversations/{other_conversation_id}"
        ).status_code == 200

        # Direct access and mutations must be denied.
        assert client.get(
            f"/api/conversations/{conversation_id}"
        ).status_code == 404

        assert client.patch(
            f"/api/conversations/{conversation_id}",
            json={"title": "Unauthorized rename"},
        ).status_code == 404

        assert client.patch(
            f"/api/conversations/{conversation_id}/messages/nonexistent-message",
            json={"content": "Unauthorized edit"},
        ).status_code == 404

        assert client.delete(
            f"/api/conversations/{conversation_id}"
        ).status_code == 404

        assert client.post(
            "/api/chat/stream",
            json={
                "conversation_id": conversation_id,
                "messages": [{"role": "user", "content": "Unauthorized chat"}],
            },
        ).status_code == 404

        # Switching back must restore access to the original conversation.
        response = client.put(
            "/api/active-profile",
            json={"profile_id": default_id},
        )
        assert response.status_code == 200

        response = client.get(
            f"/api/conversations/{conversation_id}"
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Default profile conversation"


def test_default_profile_cannot_be_deleted():
    with TestClient(main.app) as client:
        response = client.get("/api/profiles")
        assert response.status_code == 200

        defaults = [
            profile for profile in response.json()["items"]
            if profile["is_default"]
        ]
        assert defaults

        response = client.delete(
            f"/api/profiles/{defaults[0]['id']}"
        )
        assert response.status_code == 400
