import os
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from typing import List, Literal, Optional, Dict, Any, Generator

# Import backend classes
from .backends.base import BaseBackend
from .backends.llama_cpp_backend import LlamaCppBackend
from .backends.mlx_backend import MlxBackend

# --- Pydantic Models for Anthropic Compatibility ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str  # Although we load one model, the API spec requires this field.
    max_tokens: int = 1024
    temperature: float = 0.8
    stop: Optional[List[str]] = None
    stream: bool = False

# --- Logging Setup ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")


# --- Backend Management ---

BACKEND_MAPPING = {
    "llama_cpp": LlamaCppBackend,
    "mlx": MlxBackend,
}

# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the model's lifecycle. The model is loaded on startup and
    kept in memory until the server shuts down.
    """
    backend_type = os.environ.get("BACKEND_TYPE", "llama_cpp").lower()
    model_path = os.environ.get("MODEL_PATH")

    if not model_path:
        raise ValueError("MODEL_PATH environment variable must be set.")

    if backend_type not in BACKEND_MAPPING:
        raise ValueError(
            f"Invalid BACKEND_TYPE '{backend_type}'. "
            f"Available options are: {list(BACKEND_MAPPING.keys())}"
        )

    logger.info(f"Initializing backend: {backend_type}")
    backend_class = BACKEND_MAPPING[backend_type]

    try:
        # Generalize backend kwarg passing from environment variables
        backend_kwargs = {}
        prefix_map = {
            "GGUF_": "llama_cpp",
            "MLX_": "mlx"
        }

        if backend_type in prefix_map.values():
            target_prefix = [p for p, bt in prefix_map.items() if bt == backend_type][0]

            for k, v in os.environ.items():
                if k.upper().startswith(target_prefix):
                    key = k.lower().replace(target_prefix.lower(), "")
                    # Convert numerical strings to integers
                    if v.isdigit():
                        backend_kwargs[key] = int(v)
                    else:
                        backend_kwargs[key] = v

        app.state.backend = backend_class(model_path, **backend_kwargs)
        logger.info("Backend initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing backend: {e}")
        app.state.backend = None

    yield

    # Clean up resources on shutdown
    if hasattr(app.state, "backend") and app.state.backend:
        del app.state.backend
    logger.info("Backend resources cleaned up.")

# --- FastAPI App Initialization ---

app = FastAPI(lifespan=lifespan)

# --- API Endpoint ---

def generate_sse_chunk(event: str, data: Dict[str, Any]) -> str:
    """Formats a dictionary into a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

async def stream_generator(backend: BaseBackend, request: ChatCompletionRequest, request_id: str, model_name: str) -> Generator[str, None, None]:
    """
    A generator function that yields SSE formatted chunks for streaming responses.
    """
    # 1. Send the message_start event
    start_data = {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}, # Placeholder
        },
    }
    yield generate_sse_chunk("message_start", start_data)

    # 2. Stream content_block_delta events
    content_generator = backend.create_generator(
        messages=[msg.model_dump() for msg in request.messages],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=request.stop,
    )

    for text_chunk in content_generator:
        delta_data = {
            "type": "content_block_delta",
            "index": 0,  # The index should be constant for a single content block
            "delta": {"type": "text_delta", "text": text_chunk},
        }
        yield generate_sse_chunk("content_block_delta", delta_data)

    # 3. Send the message_stop event
    stop_data = {"type": "message_stop", "amazon-bedrock-invocationMetrics": {}} # Placeholder
    yield generate_sse_chunk("message_stop", stop_data)


@app.post("/v1/messages")
async def create_message(raw_request: Request):
    """
    Handles chat completion requests, supporting both streaming and non-streaming.
    """
    body = await raw_request.json()
    request = ChatCompletionRequest.model_validate(body)

    backend = raw_request.app.state.backend
    if not backend:
        return {"error": "Backend not initialized."}, 503

    request_id = f"msg_{uuid.uuid4().hex}"
    model_name = request.model

    if request.stream:
        # Return a streaming response
        return StreamingResponse(
            stream_generator(backend, request, request_id, model_name),
            media_type="text/event-stream",
        )
    else:
        # Handle non-streaming response
        content_generator = backend.create_generator(
            messages=[msg.model_dump() for msg in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )

        full_content = "".join([chunk for chunk in content_generator])

        response_data = {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [{"type": "text", "text": full_content}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}, # Placeholder
        }
        return response_data

@app.get("/health")
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok"}
