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


def _fallback_extract_answer(text: str, num_choices: int) -> Tuple[str | None, bool]:
    """Extract an answer via regex fallback when JSON parsing fails.

    Applies multiple case-insensitive patterns and returns the last match
    (closest to the end of the text). This is the sole backup layer when the
    primary Pydantic-based extraction cannot parse the model output.
    """
    try:
        patterns = [
            r"answer is ([A-Z])",
            r"is ([A-Z])\b",
            r"think ([A-Z])",
            r"Option ([A-Z])",
            r"(?:choose|chose) ([A-Z])",
            r"select ([A-Z])",
            r"correct.*?([A-Z])",
            r'"answer"[:\s]*"?([A-Z])',
        ]

        if num_choices <= 0:
            # Non-multiple-choice: capture full answer text after "answer is"
            patterns = [
                r"answer is (.+?)(?:[.,!?]|$)",
                r'"answer"[:\s]*"?([^"]+)',
            ]

        best_match: tuple[int, str] | None = None  # (end_pos, letter)

        for pat in patterns:
            compiled = re.compile(pat, re.IGNORECASE)
            for m in compiled.finditer(text):
                end = m.end()
                letter = m.group(1)
                if num_choices > 0:
                    letter = letter.upper()
                if best_match is None or end > best_match[0]:
                    best_match = (end, letter)

        if best_match is None:
            return None, False

        _, letter = best_match

        if num_choices > 0:
            valid_range = chr(ord("A") + num_choices - 1)
            if "A" <= letter <= valid_range:
                return letter, True
            return None, False

        # num_choices <= 0: non-multiple-choice — return stripped text
        return letter.strip(), True

    except Exception:
        return None, False


def extract_answer(text: str, num_choices: int = 4) -> Tuple[str | None, bool]:
    """Extract a multiple-choice letter answer from raw model text.

    Primary strategy: parse the response as structured JSON and validate via
    Pydantic. Falls back to regex-based extraction if JSON parsing fails.

    Args:
        text: Raw model output string.
        num_choices: Number of valid answer options (e.g., 4 for A-D).

    Returns:
        Tuple of (extracted_letter, succeeded). *extracted_letter* is ``None``
        if extraction failed; *succeeded* indicates whether a valid answer was
        recovered.
    """
    text = text.strip()
    if not text:
        return None, False

    cleaned = _strip_markdown_fences(text)

    try:
        parsed = json.loads(cleaned)
        ResponseModel = make_response_model(num_choices)
        validated = ResponseModel.model_validate(parsed)
        answer = validated.answer
        return answer.value if hasattr(answer, 'value') else answer, True
    except (json.JSONDecodeError, Exception):
        return _fallback_extract_answer(text, num_choices)
