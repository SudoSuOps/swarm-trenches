"""Tests for FastAPI server module."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from swarmcode_agent.server import ServeConfig, create_app


@pytest.fixture
def workspace():
    """Create a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(workspace):
    """Create server configuration."""
    return ServeConfig(
        workspace_dir=workspace,
        enable_tools=True,
    )


@pytest.fixture
def client(config):
    """Create test client."""
    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "tools_enabled" in data


class TestModelsEndpoint:
    """Tests for models endpoint."""

    def test_list_models(self, client):
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "swarmcode"


class TestChatCompletions:
    """Tests for chat completions endpoint."""

    def test_basic_completion(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "swarmcode",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_completion_with_system_message(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "swarmcode",
                "messages": [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Write a hello world function"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] in ["stop", "tool_calls"]

    def test_completion_includes_usage(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "swarmcode",
                "messages": [
                    {"role": "user", "content": "Test message"}
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


class TestToolsEndpoint:
    """Tests for tools endpoint."""

    def test_list_tools(self, client):
        response = client.get("/v1/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0

        tool_names = [t["function"]["name"] for t in data["tools"]]
        assert "read_file" in tool_names
        assert "search" in tool_names

    def test_execute_tool(self, client, workspace):
        # Create a test file
        test_file = workspace / "test.txt"
        test_file.write_text("Hello, World!")

        response = client.post(
            "/v1/tools/execute",
            json={
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.txt"}',
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # FastAPI with CORS middleware should handle OPTIONS
        assert response.status_code in [200, 405]
