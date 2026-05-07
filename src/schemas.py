"""Pydantic schemas for structured model outputs."""

from enum import Enum

from pydantic import Field, create_model
from outlines.types import JsonSchema


def make_response_model(num_choices: int):
    """Create a Pydantic model whose ``answer`` field allows exactly the
    letters ``A`` … up to the number of choices provided.

    For non-multiple-choice questions (``num_choices <= 0``) the answer
    field is a plain ``str``.
    """
    if num_choices <= 0:
        return create_model(
            "ModelResponse_0",
            reasoning=(
                str,
                Field(
                    default="",
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
                description="Step-by-step reasoning that leads to the final answer.",
            ),
        ),
        answer=(
            enum_class,
            Field(..., description="The selected option letter."),
        ),
    )


def make_response_schema(num_choices: int) -> JsonSchema:
    """Return an Outlines ``JsonSchema`` term that constrains generation.

    The schema limits the ``answer`` field to the exact set of option
    letters present in the current MMMU sample.
    """
    model_cls = make_response_model(num_choices)
    return JsonSchema(model_cls)
