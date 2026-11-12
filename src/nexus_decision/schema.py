"""Typed decision protocol for nexus-decision-4b.

A request has a ``state`` (any JSON-serialisable object or plain text) and a mapping of named
questions. Every question resolves to a closed set of options. The model always answers with
exactly one of the declared options and returns a probability distribution over all of them.
"""

from __future__ import annotations

import string
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LETTERS: str = string.ascii_uppercase
MAX_OPTIONS: int = len(LETTERS)  # 26: every option maps to one single-token letter

Mode = Literal["fast", "think"]


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1, description="What the model must decide.")

    def option_items(self) -> list[tuple[str, str]]:
        """Canonical (label, description) pairs. Order here is the *canonical* order."""
        raise NotImplementedError


class Choice(_QuestionBase):
    """Pick exactly one of the declared options."""

    type: Literal["choice"] = "choice"
    options: dict[str, str] = Field(description="Mapping of option label -> criteria/description.")

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(v) <= MAX_OPTIONS:
            raise ValueError(f"a choice needs between 2 and {MAX_OPTIONS} options, got {len(v)}")
        for label in v:
            if not label.strip():
                raise ValueError("option labels must be non-empty")
        return v

    def option_items(self) -> list[tuple[str, str]]:
        return [(k, d) for k, d in self.options.items()]


class Score(_QuestionBase):
    """Place the state on an ordinal rubric (e.g. 1-5). Returns an expected score too."""

    type: Literal["score"] = "score"
    rubric: dict[int, str] = Field(description="Mapping of integer level -> level description.")

    @field_validator("rubric")
    @classmethod
    def _check_rubric(cls, v: dict[int, str]) -> dict[int, str]:
        if not 2 <= len(v) <= MAX_OPTIONS:
            raise ValueError(f"a rubric needs between 2 and {MAX_OPTIONS} levels, got {len(v)}")
        return v

    def option_items(self) -> list[tuple[str, str]]:
        return [(str(k), self.rubric[k]) for k in sorted(self.rubric)]


class Binary(_QuestionBase):
    """A yes/no proposition. Returns P(yes)."""

    type: Literal["binary"] = "binary"

    def option_items(self) -> list[tuple[str, str]]:
        return [
            ("yes", "The proposition in the instructions is true for this state."),
            ("no", "The proposition in the instructions is false for this state."),
        ]


Question = Annotated[Choice | Score | Binary, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] | list[Any] | str
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("at least one question is required")
        return v


class Answer(BaseModel):
    """The typed answer for one question."""

    question: str
    type: Literal["choice", "score", "binary"]
    choice: str
    probabilities: dict[str, float]
    confidence: float
    entropy: float
    mode: Mode
    expected_score: float | None = None
    p_yes: float | None = None
    reasoning: str | None = None
    model_id: str = ""


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    model_id: str


def parse_question(obj: dict[str, Any]) -> Choice | Score | Binary:
    """Parse a raw question dict (with a ``type`` key) into a typed question."""
    kind = obj.get("type")
    cls = {"choice": Choice, "score": Score, "binary": Binary}.get(kind)
    if cls is None:
        raise ValueError(f"unknown question type: {kind!r}")
    return cls.model_validate(obj)
