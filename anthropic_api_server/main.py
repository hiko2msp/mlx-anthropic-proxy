import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Generator, List, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

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
    model: str
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
        backend_kwargs = {}
        prefix_map = {"GGUF_": "llama_cpp", "MLX_": "mlx"}

        if backend_type in prefix_map.values():
            target_prefix = [p for p, bt in prefix_map.items() if bt == backend_type][0]

            for k, v in os.environ.items():
                if k.upper().startswith(target_prefix):
                    key = k.lower().replace(target_prefix.lower(), "")
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

    if hasattr(app.state, "backend") and app.state.backend:
        del app.state.backend
    logger.info("Backend resources cleaned up.")


# --- API Endpoint ---


def generate_sse_chunk(event: str, data: Dict[str, Any]) -> str:
    """Formats a dictionary into a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- FastAPI App Initialization ---

app = FastAPI(lifespan=lifespan)


async def stream_generator(
    backend: BaseBackend,
    request: ChatCompletionRequest,
    request_id: str,
    model_name: str,
) -> Generator[str, None, None]:
    """
    A stateful generator that parses model output for <thinking> tags and yields
    a structured, interleaved SSE stream compliant with the Anthropic API.
    """
    # 1. Send message_start event
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
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield generate_sse_chunk("message_start", start_data)

    content_generator = backend.create_generator(
        messages=[msg.model_dump() for msg in request.messages],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=request.stop,
    )

    # State machine variables
    buffer = ""
    state = "text"  # Can be 'text' or 'thinking'
    block_index = 0

    # Start the first content block (always 'text')
    yield generate_sse_chunk(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        },
    )

    for text_chunk in content_generator:
        buffer += text_chunk

        # Process buffer for thinking tags
        while True:
            if state == "text":
                if "<thinking>" in buffer:
                    # Split buffer at the start of the thinking block
                    before_think, after_think = buffer.split("<thinking>", 1)

                    # Yield any text before the tag
                    if before_think:
                        yield generate_sse_chunk(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": before_think},
                            },
                        )

                    # Stop the current text block
                    yield generate_sse_chunk(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_index},
                    )

                    # Start a new thinking block
                    block_index += 1
                    state = "thinking"
                    yield generate_sse_chunk(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )  # Anthropic uses 'text' type even for thinking

                    buffer = after_think
                else:
                    # No tag found, yield the whole buffer and clear it
                    yield generate_sse_chunk(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": buffer},
                        },
                    )
                    buffer = ""
                    break  # Exit the while loop

            elif state == "thinking":
                if "</thinking>" in buffer:
                    # Split buffer at the end of the thinking block
                    thought, after_thought = buffer.split("</thinking>", 1)

                    # Yield the thought content
                    if thought:
                        yield generate_sse_chunk(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": thought},
                            },
                        )

                    # Stop the thinking block
                    yield generate_sse_chunk(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_index},
                    )

                    # Start a new text block for the final answer
                    block_index += 1
                    state = "text"
                    yield generate_sse_chunk(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )

                    buffer = after_thought
                else:
                    # No closing tag found, yield the whole buffer and clear it
                    yield generate_sse_chunk(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": buffer},
                        },
                    )
                    buffer = ""
                    break  # Exit the while loop

    # After the loop, yield any remaining text in the buffer
    if buffer:
        yield generate_sse_chunk(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": buffer},
            },
        )

    # Stop the final content block
    yield generate_sse_chunk(
        "content_block_stop", {"type": "content_block_stop", "index": block_index}
    )

    # Send the message_stop event
    stop_data = {"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}
    yield generate_sse_chunk("message_stop", stop_data)


@app.post("/v1/messages")
async def create_message(raw_request: Request):
    body = await raw_request.json()
    request = ChatCompletionRequest.model_validate(body)

    backend = raw_request.app.state.backend
    if not backend:
        return {"error": "Backend not initialized."}, 503

    request_id = f"msg_{uuid.uuid4().hex}"
    model_name = request.model

    if request.stream:
        return StreamingResponse(
            stream_generator(backend, request, request_id, model_name),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming remains the same, it doesn't support interleaved thinking
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
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return response_data


@app.get("/health")
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok"}
