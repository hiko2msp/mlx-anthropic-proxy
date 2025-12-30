from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator

class BaseBackend(ABC):
    """
    Abstract base class for inference backends.
    Defines the common interface for model loading and generation.
    """

    @abstractmethod
    def __init__(self, model_path: str, **kwargs: Any):
        """
        Initializes the backend and loads the model specified by model_path.
        Backend-specific arguments can be passed via kwargs.
        """
        raise NotImplementedError

    @abstractmethod
    def create_generator(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[List[str]],
    ) -> Generator[str, None, None]:
        """
        Creates a generator that yields text chunks based on the input messages.

        Args:
            messages: A list of message dictionaries, compatible with the chat template.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.
            stop: A list of stop strings.

        Yields:
            A string containing the next generated text chunk.
        """
        raise NotImplementedError
