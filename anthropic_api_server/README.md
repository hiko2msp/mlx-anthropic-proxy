# Anthropic-Compatible API Server with Dual Backend

This project provides a high-performance, Anthropic Messages API compatible server that can be powered by two different inference backends: `llama-cpp-python` for broad CPU and GPU support, and `mlx-engine` for Apple Silicon Macs.

## Features

- **Anthropic Messages API Compatibility**: Implements the `/v1/messages` endpoint, supporting the official message structure.
- **Dual Backend Support**: Choose between `llama-cpp` for cross-platform compatibility (Linux, Windows, macOS) or `mlx` for optimized performance on Apple Silicon.
- **Streaming & Non-Streaming**: Supports both `stream: true` for real-time token streaming (Server-Sent Events) and `stream: false` for single JSON responses.
- **Interleaved Thinking Support**: For streaming requests, the server can parse `<thinking>` and `</thinking>` tags from the model's output to create distinct "thought" and "text" content blocks, fully compliant with the Anthropic API spec.
- **Dynamic Model Loading**: Specify a model on startup via environment variables. For `llama-cpp`, models can be automatically downloaded from Hugging Face Hub if not found locally.
- **Configuration via Environment Variables**: Easily configure the backend, model path, and backend-specific parameters.

## Prerequisites

- Python 3.11+
- `pip` for installing packages

## Quick Start & Setup

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Backend Selection and Installation

You must choose which backend to use. The installation steps differ based on your choice.

#### Option A: `llama-cpp` Backend (Recommended for Linux, Windows, and Intel Macs)

This is the most compatible option and works on a wide range of hardware.

1.  **Install Dependencies**:
    The requirements are defined in `anthropic_api_server/requirements.txt`.

    ```bash
    pip install -r anthropic_api_server/requirements.txt
    ```
    *Note: Depending on your system, you may need to install `llama-cpp-python` with specific compilation flags to enable GPU support (e.g., CUDA, Metal). Please refer to the [official llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for detailed instructions.*

#### Option B: `mlx` Backend (Apple Silicon Macs Only)

This backend is highly optimized for M1/M2/M3 chips and requires macOS 14.0+ and Python 3.11. The `mlx` backend depends on the `mlx-engine` library, which is included in this repository as a git submodule.

1.  **Initialize and Update the `mlx-engine` Submodule**:
    After cloning the main repository, you must initialize the `mlx-engine` submodule.

    ```bash
    git submodule update --init --recursive
    ```

2.  **Install Dependencies**:
    Install the base dependencies and then install the `mlx-engine` package in editable mode.

    ```bash
    # Install base server dependencies
    pip install -r anthropic_api_server/requirements.txt

    # Install mlx-engine and its specific dependencies
    pip install -e mlx-engine
    ```
    This command will install `mlx-engine` from the submodule directory and automatically handle its dependencies (like `mlx`, `mlx-lm`, etc.) based on its own `requirements.txt`.


### 3. Configure Environment Variables

Create a `.env` file in the root of the project or export the variables directly.

```bash
# .env file

# 1. Choose your backend: "llama_cpp" or "mlx"
BACKEND_TYPE="llama_cpp"

# 2. Set the path to your model.
# For llama-cpp: This can be a local path to a .gguf file or a Hugging Face repo_id/filename.
# For mlx: This must be a local path to a model compatible with mlx-engine.
MODEL_PATH="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# 3. (Optional) Set backend-specific parameters for llama-cpp.
# These will be passed to the Llama constructor.
GGUF_N_CTX=4096
GGUF_N_GPU_LAYERS=50 # Example: offload 50 layers to GPU
```

### 4. Run the Server

```bash
uvicorn anthropic_api_server.main:app --host 0.0.0.0 --port 8080
```
The server will start, load the model according to your configuration, and be ready to accept requests.

## API Usage

The server exposes a `/v1/messages` endpoint compatible with the Anthropic Messages API.

### Example: Non-Streaming Request

```bash
curl -X POST http://localhost:8080/v1/messages \
-H "Content-Type: application/json" \
-d '{
  "messages": [{"role": "user", "content": "Explain the importance of low-latency LLMs in one sentence."}],
  "model": "your-model-name",
  "max_tokens": 50,
  "temperature": 0.7,
  "stream": false
}'
```

**Expected Response:**
A single JSON object containing the full response from the assistant.

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "your-model-name",
  "content": [
    {
      "type": "text",
      "text": "Low-latency LLMs are crucial for creating responsive, real-time conversational AI experiences that feel natural and engaging to users."
    }
  ],
  "stop_reason": "end_turn",
  ...
}
```

### Example: Streaming Request

```bash
curl -N -X POST http://localhost:8080/v1/messages \
-H "Content-Type: application/json" \
-d '{
  "messages": [{"role": "user", "content": "Write a short poem about code."}],
  "model": "your-model-name",
  "max_tokens": 100,
  "stream": true
}'
```

**Expected Response:**
A stream of Server-Sent Events (SSE) that you can process in real-time.

```
event: message_start
data: {"type":"message_start", "message":{...}}

event: content_block_delta
data: {"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":"A silent,"}}

event: content_block_delta
data: {"type":"content_block_delta", "index":1, "delta":{"type":"text_delta", "text":" world"}}

...

event: message_stop
data: {"type":"message_stop", ...}
```

## Known Limitations

-   **Token Usage Statistics**: The `usage` field in the API responses (`input_tokens` and `output_tokens`) is currently a placeholder and does not reflect the actual token count. This feature may be implemented in a future release.
