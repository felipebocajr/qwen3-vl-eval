"""Skeleton adapter — copy this file and fill in the marked sections.

How to add a new model in 3 minutes:
    1. Copy this file to  src/adapters/<your_model>.py
    2. Change the class name and @register_adapter name
    3. Fill in __init__ (load / authenticate your model)
    4. Fill in generate_answer (run inference, return raw string)
    5. Add  from src.adapters import <your_model>  to  __init__.py
    6. Change  "qwen_local" → "<your_name>"  in  src/main.py

The pipeline, parser, metrics, and visualization layers are untouched.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.adapters.base import BaseVLMAdapter, register_adapter


# ═══════════════════════════════════════════════════════════════════════════
# 1. Choose a unique name.  This is the string you'll pass to the factory:
#       adapter = get_model_adapter("my_model_name")
# ═══════════════════════════════════════════════════════════════════════════
@register_adapter("CHANGE_ME__provider_name")
class CHANGE_ME__AdapterClassName(BaseVLMAdapter):
    """Adapter for <insert model description here>."""

    # ── Initialisation ────────────────────────────────────────────────────
    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        """Set up the model, processor/client, and any caches.

        Args:
            model_id: Optional override for the model identifier.
            **kwargs: Forward-compatibility slot (ignored).
        """
        super().__init__()
        _ = kwargs  # consume unused kwargs

        # ------------------------------------------------------------------
        # TODO: store your model identifier (use a sensible default)
        # ------------------------------------------------------------------
        self._model_id: str = model_id or "CHANGE_ME__default_model_id"

        # ------------------------------------------------------------------
        # TODO: load / authenticate your model
        #   - Local HF model:  AutoModel.from_pretrained(self._model_id, ...)
        #   - API client:      your_client = YourSDK(api_key=...)
        # ------------------------------------------------------------------
        # self._model = ...
        # self._processor = ...

        print(f"Adapter '{CHANGE_ME__AdapterClassName.__name__}' ready "
              f"(model={self._model_id}).")

    # ── Inference ─────────────────────────────────────────────────────────
    def generate_answer(
        self,
        prompt: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        """Run inference and return the raw model output string.

        Args:
            prompt: The text prompt (question + choices formatted by
                ``src.pipeline.build_prompt``).
            images: PIL images extracted from the MMMU sample.
            **kwargs: Adapter-specific overrides.  The pipeline always
                passes ``max_new_tokens`` (int).

        Returns:
            The raw model output.  This will be sent to ``src.parser``
            for structured extraction — **do not** parse it yourself.
        """
        max_new_tokens: int = kwargs.get("max_new_tokens", 512)

        # ------------------------------------------------------------------
        # TODO: run inference and return the raw output
        #
        #   Local model example (Outlines):
        #       generator = self._get_cached_generator()
        #       return generator([formatted_prompt] + image_inputs,
        #                        max_new_tokens=max_new_tokens).strip()
        #
        #   API example (Anthropic / OpenAI):
        #       response = self._client.messages.create(
        #           model=self._model_id,
        #           max_tokens=max_new_tokens,
        #           messages=[{"role": "user", "content": prompt}],
        #       )
        #       return response.content[0].text
        # ------------------------------------------------------------------
        raise NotImplementedError("Fill in generate_answer")
