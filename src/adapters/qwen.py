"""Qwen3-VL adapter — local inference via Hugging Face + Outlines.

Wraps model loading, Outlines generator caching, and structured JSON
generation inside a ``BaseVLMAdapter``-compliant class.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from PIL import Image
from outlines import from_transformers, Generator, inputs
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src import config
from src.adapters.base import BaseVLMAdapter, register_adapter
from src.schemas import make_response_schema

_MAX_OPTIONS = 10
"""Maximum number of option letters (A-J) the cached Generator supports.

A single schema covers all MMMU samples (max 9 options). The parser
post-validates that the extracted letter falls within each sample's actual
range, avoiding the need to rebuild heavy FSM indices per sample.
"""


@register_adapter("qwen_local")
class QwenAdapter(BaseVLMAdapter):
    """Qwen3-VL-2B-Instruct adapter with Outlines constrained generation.

    Loads the model and processor eagerly on construction and builds a single
    cached ``outlines.Generator`` instance that constrains output to valid
    JSON with an enumerated answer field (A-J).
    """

    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        """Initialise the adapter.

        Args:
            model_id: Hugging Face model checkpoint identifier.  Defaults to
                ``config.DEFAULT_MODEL_ID`` when ``None`` or omitted.
            **kwargs: Ignored (reserved for factory forward-compatibility).
        """
        super().__init__()
        _ = kwargs  # explicitly consumed

        self._model_id: str = model_id or config.DEFAULT_MODEL_ID
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._generator: Generator | None = None

        print(f"Loading model {self._model_id} on {self._device} ...")

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._model.eval()

        self._build_generator()
        print("Model loaded.")

    # --- Internal helpers ---

    def _build_generator(self) -> None:
        """Build (or rebuild) the cached Outlines Generator.

        Uses a single ``JsonSchema`` covering options A-J so every sample
        shares the same FSM index, avoiding per-sample rebuild costs.
        """
        outlines_model = from_transformers(self._model, self._processor)
        output_schema = make_response_schema(_MAX_OPTIONS)
        self._generator = Generator(outlines_model, output_type=output_schema)
        print("Outlines Generator built (single instance, max_options=J).")

    def _free_memory(self) -> None:
        """Release CUDA memory cached by PyTorch's allocator after generation.

        Without this, the allocator hoards freed blocks and RAM appears to
        grow without bound across samples with varying input sizes.
        """
        if self._device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # --- Public interface ---

    def generate_answer(
        self,
        prompt: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        """Run structured JSON generation on a multimodal prompt.

        Args:
            prompt: Text prompt (question + options) for the user message.
            images: List of PIL images to include in the prompt.
            **kwargs: Adapter-specific overrides:
                - ``max_new_tokens`` (int, default 512): Token budget.

        Returns:
            The raw JSON string produced by the model, stripped of
            surrounding whitespace.
        """
        if self._generator is None:
            self._build_generator()

        max_new_tokens: int = kwargs.get("max_new_tokens")

        system_text = (
            "You are a helpful assistant. For multiple-choice questions, analyze "
            "the question and any provided images carefully, think through your "
            "reasoning concisely, then respond with ONLY valid JSON matching the "
            "provided schema. Do not repeat steps you have already completed. "
            "Do not include markdown or any text outside the JSON."
        )

        # Build standard HF multimodal messages.  Each image is represented by
        # a bare ``{"type": "image"}`` dict so the chat template counts it and
        # emits ``<|vision_start|><|image_pad|><|vision_end|>``.  The actual
        # PIL images are supplied separately to the processor via Outlines.
        user_content: list[dict[str, Any]] = []
        for _ in images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]

        formatted_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Outlines' TransformersMultiModal adapter expects a list where the
        # first element is the formatted prompt string and the remaining
        # elements are ``outlines.inputs.Image`` wrappers around the raw PIL
        # images.
        model_input: list[Any] = [formatted_prompt]
        for img in images:
            model_input.append(inputs.Image(img))

        raw_answer = self._generator(
            model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        self._free_memory()

        return raw_answer.strip()
