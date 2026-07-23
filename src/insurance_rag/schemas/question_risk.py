from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.question import QuestionAnalysis

RiskLevel = Literal["low", "medium", "high", "blocked"]
RiskCategory = Literal[
    "general_information",
    "policy_interpretation",
    "individual_claim_determination",
    "legal_conclusion",
    "financial_recommendation",
    "product_recommendation",
    "cancellation_recommendation",
    "fraud_or_evasion",
    "privacy_sensitive",
    "insufficient_information",
    "other",
]


class QuestionRiskGateRequest(BaseModel):
    original_question: str = Field(min_length=2, max_length=4000)
    normalized_question: str = Field(min_length=2, max_length=1000)
    requested_action: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_compound: bool = False
    sub_questions: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("original_question", "normalized_question")
    @classmethod
    def required_text(cls, value: str) -> str:
        if len(value := value.strip()) < 2:
            raise ValueError("question must contain at least two characters")
        return value

    @field_validator(
        "assumptions", "missing_information", "warnings", "sub_questions"
    )
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @classmethod
    def from_analysis(cls, analysis: QuestionAnalysis) -> "QuestionRiskGateRequest":
        requested_actions = list(dict.fromkeys(
            item.requested_action for item in analysis.sub_questions
        ))
        return cls(
            original_question=analysis.original_question,
            normalized_question=analysis.normalized_question,
            requested_action=",".join(requested_actions) or None,
            assumptions=analysis.assumptions,
            missing_information=[item.name for item in analysis.missing_information],
            warnings=analysis.analysis_warnings,
            is_compound=analysis.is_compound,
            sub_questions=[item.question for item in analysis.sub_questions],
        )


class QuestionRiskGateResult(BaseModel):
    risk_level: RiskLevel
    categories: list[RiskCategory] = Field(min_length=1)
    allow_retrieval: bool
    allow_claim_extraction: bool
    allow_final_answer: bool
    requires_disclaimer: bool
    requires_human_review: bool
    requires_additional_information: bool
    blocked_reason: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("categories", "missing_information", "warnings")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def consistent_permissions(self):
        if not self.allow_retrieval and self.allow_claim_extraction:
            raise ValueError("claim extraction requires retrieval")
        if self.risk_level == "blocked":
            if (
                self.allow_retrieval
                or self.allow_claim_extraction
                or self.allow_final_answer
                or not self.blocked_reason
            ):
                raise ValueError("blocked risk requires all permissions off and a reason")
        elif self.blocked_reason is not None:
            raise ValueError("blocked_reason is only valid for blocked risk")
        if self.requires_additional_information and not self.missing_information:
            raise ValueError("additional information requires a missing information list")
        return self
