"""Pydantic schemas for structured model outputs."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelResponse(BaseModel):
    """
    Expected JSON schema for model answers.
    reasoning precedes answer so the constrained decoder emits thought first.
    """

    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning that leads to the final answer.",
    )

    answer: Literal["A", "B", "C", "D"] = Field(
        ...,
        description="The selected option letter.",
    )

    @field_validator("answer", mode="before")
    @classmethod
    def uppercase_answer(cls, v):
        """Normalise answer to uppercase."""
        if isinstance(v, str):
            return v.strip().upper()
        return v
