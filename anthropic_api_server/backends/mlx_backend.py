from typing import Any, Dict, Generator, List, Optional

from .base import BaseBackend

# MLX Engine is only available on macOS with Apple Silicon.
# We use a try-except block to handle the import error gracefully
# on other platforms.
try:
    from mlx_engine.generate import create_generator, load_model
    from transformers import AutoTokenizer

    MLX_ENGINE_AVAILABLE = True
except ImportError:
    # If the import fails, we set a flag and the backend will not be usable.
    MLX_ENGINE_AVAILABLE = False
    # Define dummy classes to avoid errors if this file is imported elsewhere
    load_model = None
    create_generator = None
    AutoTokenizer = None


class MlxBackend(BaseBackend):
    """
    Backend implementation for mlx-engine.
    This backend is only functional on macOS with Apple Silicon.
    """

    def __init__(self, model_path: str, **kwargs: Any):
        """
        Initializes the backend and loads the MLX model.
        Raises a RuntimeError if MLX Engine is not available.
        """
        if not MLX_ENGINE_AVAILABLE:
            raise RuntimeError(
                "MLX Engine backend is not available. Please install mlx-engine "
                "and its dependencies, and ensure you are running on a compatible "
                "macOS system with Apple Silicon."
            )

        print("Loading MLX model...")
        # MLX models require a separate tokenizer for applying chat templates
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model using mlx-engine's utility
        self.model_kit = load_model(model_path, **kwargs)
        print("MLX model loaded successfully.")

    def create_generator(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Creates a generator that yields text chunks using mlx-engine.
        """
        if not MLX_ENGINE_AVAILABLE:
            # This check is redundant if the constructor already ran,
            # but it's good practice for safety.
            raise RuntimeError("MLX Engine backend is not available.")

        # 1. Apply the chat template to the messages
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Tokenize the resulting prompt string
        prompt_tokens = self.model_kit.tokenize(prompt)

        # 3. Create the generator from mlx-engine
        generator = create_generator(
            model_kit=self.model_kit,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temp=temperature,
            stop_strings=stop,
        )

        # 4. Yield the text from each generation result
        for result in generator:
            if result.text:
                yield result.text
