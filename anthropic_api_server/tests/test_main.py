from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from anthropic_api_server.backends.base import BaseBackend

# Use absolute imports assuming pytest is run from the project root
from anthropic_api_server.main import app

# --- Mock Backend ---


class MockBackend(BaseBackend):
    def __init__(self, model_path: str, **kwargs):
        # Don't call super().__init__ as it's an abstract class
        print(f"MockBackend initialized with model_path: {model_path}")

    def create_generator(self, messages, max_tokens, temperature, stop):
        # Simulate a simple generator for streaming tests
        yield "This "
        yield "is a "
        yield "test."


# --- Pytest Fixtures ---


@pytest.fixture
def mock_backend(monkeypatch):
    """Mocks the backend initialization in the lifespan manager."""
    mock_instance = MockBackend(model_path="mock/model")

    # Replace the actual backend classes with our mock using full import paths
    monkeypatch.setattr(
        "anthropic_api_server.main.LlamaCppBackend",
        lambda *args, **kwargs: mock_instance,
    )
    monkeypatch.setattr(
        "anthropic_api_server.main.MlxBackend", lambda *args, **kwargs: mock_instance
    )

    return mock_instance


@pytest.fixture
def client(mock_backend):
    """
    Provides a TestClient for the FastAPI app where the backend is mocked.
    This allows testing the API endpoints without loading a real model.
    """
    # We need to manually trigger the lifespan events for the test client
    with TestClient(app) as test_client:
        yield test_client


# --- Test Cases ---


def test_health_check(client):
    """Tests if the /health endpoint returns a 200 OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_non_streaming_message_endpoint(client, mock_backend):
    """Tests the /v1/messages endpoint for a non-streaming request."""
    # Mock the generator method to return a predictable full string
    mock_backend.create_generator = MagicMock(return_value=["This is a test."])

    response = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert data["content"][0]["text"] == "This is a test."
    assert "id" in data


def test_streaming_message_endpoint(client, mock_backend):
    """Tests the /v1/messages endpoint for a streaming request."""
    response = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "stream": True,
        },
    )

    assert response.status_code == 200
    # Check if the response is a streaming response
    assert "text/event-stream" in response.headers["content-type"]

    # Process the SSE stream to verify its content
    lines = response.text.split("\n")
    events = [line for line in lines if line.startswith("data:")]

    assert '{"type":"message_start"' in events[0]
    assert '{"type":"content_block_start"' in events[1]

    # Check for the delta chunks
    assert '{"type":"text_delta","text":"This "}' in events[2]
    assert '{"type":"text_delta","text":"is a "}' in events[3]
    assert '{"type":"text_delta","text":"test."}' in events[4]

    assert '{"type":"content_block_stop"' in events[5]
    assert '{"type":"message_stop"' in events[6]


def test_backend_initialization_failure(monkeypatch):
    """Tests if the server handles a backend initialization failure gracefully."""

    # Force an exception during backend initialization
    def faulty_initializer(*args, **kwargs):
        raise ValueError("Test initialization failure")

    monkeypatch.setattr(
        "anthropic_api_server.main.LlamaCppBackend", faulty_initializer
    )

    # Use a raw StarletteTestClient to control lifespan events
    with StarletteTestClient(app, raise_server_exceptions=False) as client:
        # The lifespan startup should have failed, so the backend is None
        assert client.app.state.backend is None

        # Requests to the message endpoint should now fail with a 503
        response = client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
            },
        )
        assert response.status_code == 503
        assert response.json() == {"error": "Backend not initialized."}
