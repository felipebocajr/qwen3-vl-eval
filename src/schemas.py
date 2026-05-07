"""Pydantic schemas for structured model outputs."""

from enum import Enum

from pydantic import Field, create_model
from outlines.types import JsonSchema


def make_response_model(num_choices: int):
    """Create a Pydantic model with a constrained answer field.

    The ``answer`` field is restricted to the option letters A through the
    letter corresponding to the number of choices (e.g., A-D for 4 choices).
    For non-multiple-choice questions (``num_choices <= 0``), the answer field
    is an unconstrained string.
    """
    if num_choices <= 0:
        return create_model(
            "ModelResponse_0",
            reasoning=(
                str,
                Field(
                    default="",
                    max_length=1200,
                    description="Step-by-step reasoning that leads to the final answer.",
                ),
            ),
            answer=(
                str,
                Field(..., description="The answer text (non-multiple-choice)."),
            ),
        )

    letters = [chr(ord("A") + i) for i in range(num_choices)]
    enum_class = Enum("AnswerEnum", {letter: letter for letter in letters})

    return create_model(
        f"ModelResponse_{num_choices}",
        reasoning=(
            str,
            Field(
                default="",
                max_length=1200,
                description="Step-by-step reasoning that leads to the final answer.",
            ),
        ),
        answer=(
            enum_class,
            Field(..., description="The selected option letter."),
        ),
    )


def make_response_schema(num_choices: int) -> JsonSchema:
    """Return an Outlines JsonSchema that constrains structured generation.

    Limits the ``answer`` field to the exact set of option letters valid for
    the current MMMU sample (e.g., A-D for 4 choices).
    """
    model_cls = make_response_model(num_choices)
    return JsonSchema(model_cls)
