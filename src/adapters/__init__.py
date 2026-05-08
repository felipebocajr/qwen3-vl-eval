"""Model adapter factory.

Import this sub-package and call ``get_model_adapter(name, **kwargs)`` to
obtain a configured ``BaseVLMAdapter`` instance.  Adapters self-register
via the ``@register_adapter`` decorator in ``src.adapters.base`` — just
adding a new module with the decorator and importing it here makes it
available through the factory.
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import BaseVLMAdapter, MODEL_REGISTRY

# Import adapter modules so @register_adapter decorators fire at import time.
# Add new adapter imports below when extending the pipeline.
from src.adapters import medgemma  # noqa: F401
from src.adapters import qwen  # noqa: F401


# --- Public factory ---


def get_model_adapter(provider_name: str, **kwargs: Any) -> BaseVLMAdapter:
    """Instantiate and return a VLM adapter from the registry.

    Args:
        provider_name: Registered adapter name (e.g. ``"qwen_local"``).
        **kwargs: Forwarded to the adapter's ``__init__`` (e.g.
            ``model_id``, ``max_new_tokens``).

    Returns:
        A fully initialised ``BaseVLMAdapter`` subclass instance.

    Raises:
        ValueError: If *provider_name* is not registered.
        TypeError: If the registered class is not a ``BaseVLMAdapter``
            subclass.
    """
    if provider_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown adapter name '{provider_name}'. "
            f"Available: {available}"
        )

    cls = MODEL_REGISTRY[provider_name]

    if not issubclass(cls, BaseVLMAdapter):
        raise TypeError(
            f"Registered class {cls.__name__} for '{provider_name}' "
            f"is not a BaseVLMAdapter subclass."
        )

    return cls(**kwargs)
