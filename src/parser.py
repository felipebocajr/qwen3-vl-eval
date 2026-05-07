"""Answer extraction from model responses using Pydantic validation."""

import json
import re
from typing import Tuple

from pydantic import ValidationError

from src.schemas import make_response_model


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences so JSON can be parsed."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_answer(text: str, num_choices: int = 4) -> Tuple[str | None, bool]:
    """
    Extract a multiple-choice letter answer from raw model text.

    Primary strategy: structured JSON via Pydantic.
    Fallback: none – Outlines guarantees JSON conformance.

    Returns:
        (extracted_letter, succeeded)
    """
    text = text.strip()
    if not text:
        return None, False

    cleaned = _strip_markdown_fences(text)

    try:
        parsed = json.loads(cleaned)
        ResponseModel = make_response_model(num_choices)
        validated = ResponseModel.model_validate(parsed)
        return str(validated.answer), True
    except (json.JSONDecodeError, Exception):
        pass

    return None, False
