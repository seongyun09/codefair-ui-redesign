from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RequestedAction = Literal[
    "definition",
    "fact_lookup",
    "comparison",
    "evaluation",
    "recommendation",
    "decision",
    "interpretation",
]
InformationSource = Literal["question", "user_context"]


def _information_key(value: str) -> str:
    """Normalize harmless presentation differences for overlap checks."""
    return re.sub(r"[\s_\-:()\[\]{}]+", "", value.strip().casefold())


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    user_context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def normalize_question_input(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("question must contain at least two non-space characters")
        return normalized


class KnownInformation(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    value: Any
    source: InformationSource

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        if not (normalized := value.strip()):
            raise ValueError("name must not be blank")
        return normalized


class MissingInformation(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=5, max_length=300)

    @field_validator("name", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not (normalized := value.strip()):
            raise ValueError("value must not be blank")
        return normalized


class SubQuestion(BaseModel):
    id: str = Field(pattern=r"^q[1-5]$")
    question: str = Field(min_length=2, max_length=500)
    requested_action: RequestedAction
    purpose: str = Field(min_length=2, max_length=200)
    depends_on_information: list[str] = Field(default_factory=list, max_length=7)

    @field_validator("question", "purpose")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        if len(normalized := value.strip()) < 2:
            raise ValueError("value must contain at least two non-space characters")
        return normalized

    @field_validator("depends_on_information")
    @classmethod
    def remove_blank_dependencies(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class QuestionAnalysis(BaseModel):
    original_question: str = Field(min_length=2, max_length=4000)
    normalized_question: str = Field(min_length=2, max_length=1000)
    main_intent: str = Field(min_length=2, max_length=300)
    assumptions: list[str] = Field(default_factory=list, max_length=7)
    known_information: list[KnownInformation] = Field(default_factory=list, max_length=15)
    missing_information: list[MissingInformation] = Field(default_factory=list, max_length=10)
    is_compound: bool
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=5)
    analysis_warnings: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("original_question", "normalized_question", "main_intent")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        if len(normalized := value.strip()) < 2:
            raise ValueError("value must contain at least two non-space characters")
        return normalized

    @field_validator("assumptions", "analysis_warnings")
    @classmethod
    def remove_blank_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def validate_relationships(self) -> QuestionAnalysis:
        expected_ids = [f"q{index}" for index in range(1, len(self.sub_questions) + 1)]
        actual_ids = [item.id for item in self.sub_questions]
        if actual_ids != expected_ids:
            raise ValueError("sub-question IDs must be unique and consecutive from q1")
        if not self.is_compound and len(self.sub_questions) != 1:
            raise ValueError("a non-compound question must have exactly one sub-question")

        known_names = {_information_key(item.name) for item in self.known_information}
        self.missing_information = [
            item
            for item in self.missing_information
            if _information_key(item.name) not in known_names
        ]
        return self
