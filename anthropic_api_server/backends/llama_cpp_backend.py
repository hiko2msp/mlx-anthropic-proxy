import os
from typing import Any, Dict, List, Optional, Generator

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .base import BaseBackend

class LlamaCppBackend(BaseBackend):
    """
    Backend implementation for llama-cpp-python.
    """

    def __init__(self, model_path: str, **kwargs: Any):
        """
        Initializes the backend and loads the GGUF model specified by model_path.
        If the model_path does not exist locally, it is treated as a
        Hugging Face repository ID and downloaded.
        """
        # Default parameters for Llama constructor, can be overridden by kwargs
        llama_params = {
            "n_ctx": 4096,
            "n_gpu_layers": 0,  # No GPU layers by default, as it might not be available
            "verbose": True,
            **kwargs,
        }

        # hf_hub_download expects repo_id and filename. We assume the model_path
        # is in the format "repo_id/filename" if it's not a local file.
        if not os.path.exists(model_path):
            try:
                repo_id, filename = os.path.split(model_path)
                if not repo_id or not filename:
                    raise ValueError("Invalid model path format for Hugging Face download.")

                print(f"Model not found locally. Downloading '{filename}' from '{repo_id}'...")
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                 raise FileNotFoundError(
                    f"Failed to download model '{model_path}'. Please ensure it's a valid local path "
                    f"or a 'repo_id/filename' on Hugging Face Hub. Error: {e}"
                 )

        llama_params["model_path"] = model_path
        self.model = Llama(**llama_params)

    def create_generator(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Creates a generator that yields text chunks based on the input messages.
        """
        streamer = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        )

        for chunk in streamer:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
