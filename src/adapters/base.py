"""Abstract base class and decorator-based registry for VLM adapters.

Defines the contract that every model adapter must fulfil and provides the
machinery that lets adapters self-register via ``@register_adapter(name)``.
"""

from __future__ import annotations

import abc
from typing import Any

from PIL import Image

# --- Registry ---

MODEL_REGISTRY: dict[str, type["BaseVLMAdapter"]] = {}
"""Global adapter registry mapping provider names to adapter classes."""


def register_adapter(name: str):
    """Class decorator that registers an adapter class under *name*.

    Example::

        @register_adapter("qwen_local")
        class QwenAdapter(BaseVLMAdapter):
            ...

    Raises:
        TypeError: If the decorated class is not a ``BaseVLMAdapter``.
        KeyError: If *name* is already registered.
    """

    def _decorator(cls: type[BaseVLMAdapter]) -> type[BaseVLMAdapter]:
        if not issubclass(cls, BaseVLMAdapter):
            raise TypeError(f"{cls.__name__} must be a subclass of BaseVLMAdapter")
        if name in MODEL_REGISTRY:
            raise KeyError(
                f"Adapter name '{name}' is already registered "
                f"({MODEL_REGISTRY[name].__name__})."
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


# --- Abstract interface ---


class BaseVLMAdapter(abc.ABC):
    """Abstract interface for Vision-Language Model adapters.

    Every adapter must implement ``generate_answer``, which accepts a text
    prompt plus optional PIL images and returns the raw model output string.
    The pipeline's answer parser handles extraction — adapters are NOT
    responsible for structured parsing.
    """

    @property
    def model_name(self) -> str:
        """The model identifier used by this adapter (e.g. ``"gemma4:e2b"``)."""
        return getattr(self, "_model_id", "unknown")

    @abc.abstractmethod
    def generate_answer(
        self,
        prompt: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        """Run inference and return the raw model output.

        Args:
            prompt: The text prompt (question + options).
            images: List of PIL images associated with the prompt.
            **kwargs: Adapter-specific overrides (e.g. ``max_new_tokens``).

        Returns:
            The raw model output string (may be empty on error).
        """
        ...
