"""Gemma 4 adapter — local inference via Ollama API.

Wraps the Ollama chat API behind a ``BaseVLMAdapter``-compliant class.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import requests
from PIL import Image

from src.adapters.base import BaseVLMAdapter, register_adapter


@register_adapter("gemma_ollama")
class GemmaAdapter(BaseVLMAdapter):
    """Gemma 4 adapter via Ollama.

    Talks to a local Ollama server at ``http://localhost:11434``. The model
    must already be pulled: ``ollama pull gemma4:e2b``.
    """

    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        _ = kwargs

        self._model_id: str = model_id or "gemma4:e2b"
        self._api_url: str = "http://localhost:11434/api/chat"

        print(f"Adapter 'GemmaAdapter' ready (model={self._model_id}).")

    # --- Public interface ---

    def generate_answer(
        self,
        prompt: str,
        images: list[Image.Image],
        **kwargs: Any,
    ) -> str:
        max_new_tokens: int = kwargs.get("max_new_tokens", 1024)

        images_b64: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": max_new_tokens, "temperature": 0.0},
            "stream": False,
        }
        if images_b64:
            payload["messages"][0]["images"] = images_b64

        response = requests.post(self._api_url, json=payload, timeout=300)
        response.raise_for_status()

        return response.json()["message"]["content"].strip()
