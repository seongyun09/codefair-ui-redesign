from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.insurance_claim import InsuranceEvidenceClaim
from insurance_rag.schemas.insurance_pipeline import InsuranceEvidencePipelineResult
from insurance_rag.schemas.question_risk import RiskCategory, RiskLevel

JudgmentDisposition = Literal[
    "supported",
    "partially_supported",
    "not_supported",
    "insufficient_information",
    "requires_human_review",
]
ClaimAssessmentValue = Literal[
    "supports", "contradicts", "conditional", "irrelevant", "unclear"
]
JudgmentConfidence = Literal["high", "medium", "low"]
JudgmentPipelineStatus = Literal[
    "completed", "blocked", "insufficient_evidence", "partial", "failed"
]


def _clean(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class IndependentJudgmentModelConfig(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    provider: str = "openai"
    enabled: bool = True
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"

    @field_validator("model_id", "provider")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("model configuration values must not be blank")
        return value


class IndependentJudgmentRequest(BaseModel):
    original_question: str = Field(min_length=1, max_length=4000)
    normalized_question: str = Field(min_length=1, max_length=1000)
    requested_action: str | None = None
    risk_level: RiskLevel
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    allow_final_answer: bool
    requires_disclaimer: bool
    requires_human_review: bool
    requires_additional_information: bool
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    claims: list[InsuranceEvidenceClaim] = Field(min_length=1)
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    max_judgments: int = Field(default=2, ge=1, le=5)

    @field_validator("original_question", "normalized_question")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("question must not be blank")
        return value

    @field_validator(
        "risk_categories", "missing_information", "warnings"
    )
    @classmethod
    def unique_lists(cls, values: list[str]) -> list[str]:
        return _clean(values)

    @model_validator(mode="after")
    def unique_claim_ids(self):
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim_id values must be unique")
        return self


class ClaimAssessment(BaseModel):
    claim_id: str
    assessment: ClaimAssessmentValue
    reasoning: str = Field(min_length=1, max_length=1000)
    conditions_considered: list[str] = Field(default_factory=list)
    exceptions_considered: list[str] = Field(default_factory=list)
    source_reference_ids: list[str] = Field(default_factory=list)
    confidence: JudgmentConfidence

    @field_validator("claim_id", "reasoning")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("assessment text must not be blank")
        return value

    @field_validator(
        "conditions_considered", "exceptions_considered", "source_reference_ids"
    )
    @classmethod
    def unique_lists(cls, values: list[str]) -> list[str]:
        return _clean(values)


class IndependentModelJudgment(BaseModel):
    judgment_id: str = Field(
        pattern=r"^(j[1-9][0-9]*|q[1-5]-j[1-9][0-9]*)$"
    )
    model_id: str = Field(min_length=1)
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    disposition: JudgmentDisposition
    conclusion: str = Field(min_length=1, max_length=700)
    claim_assessments: list[ClaimAssessment] = Field(min_length=1)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    limiting_claim_ids: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requires_disclaimer: bool
    requires_human_review: bool
    confidence: JudgmentConfidence
    warnings: list[str] = Field(default_factory=list)

    @field_validator("model_id", "conclusion")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("judgment text must not be blank")
        return value

    @field_validator(
        "supporting_claim_ids", "limiting_claim_ids", "conflicting_claim_ids",
        "missing_information", "assumptions", "warnings",
    )
    @classmethod
    def unique_lists(cls, values: list[str]) -> list[str]:
        return _clean(values)

    @model_validator(mode="after")
    def consistent_human_review(self):
        if (
            self.disposition == "requires_human_review"
            and not self.requires_human_review
        ):
            raise ValueError("human-review disposition requires human review")
        ids = [item.claim_id for item in self.claim_assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("claim assessments must not contain duplicates")
        return self


class ModelJudgmentFailure(BaseModel):
    model_id: str
    error_type: str
    message: str
    retry_count: int = Field(ge=0, le=1)


class IndependentJudgmentBatchResult(BaseModel):
    original_question: str
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    judgments: list[IndependentModelJudgment] = Field(default_factory=list)
    requested_model_count: int = Field(ge=1, le=5)
    completed_model_count: int = Field(ge=0)
    failed_model_count: int = Field(ge=0)
    sufficient_for_comparison: bool
    failures: list[ModelJudgmentFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, values: list[str]) -> list[str]:
        return _clean(values)

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.completed_model_count != len(self.judgments):
            raise ValueError("completed count must equal judgment count")
        if self.failed_model_count != len(self.failures):
            raise ValueError("failed count must equal failure count")
        if (
            self.completed_model_count + self.failed_model_count
            != self.requested_model_count
        ):
            raise ValueError("completed and failed counts must equal requested count")
        if self.sufficient_for_comparison != (self.completed_model_count >= 2):
            raise ValueError("comparison requires at least two completed judgments")
        return self


class IndependentJudgmentPipelineResult(BaseModel):
    status: JudgmentPipelineStatus
    evidence: InsuranceEvidencePipelineResult
    single_result: IndependentJudgmentBatchResult | None = None
    sub_question_results: list[IndependentJudgmentBatchResult] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, values: list[str]) -> list[str]:
        return _clean(values)

    @model_validator(mode="after")
    def valid_result_shape(self):
        if self.evidence.analysis.is_compound:
            if self.single_result is not None:
                raise ValueError("compound pipeline cannot have a single result")
        elif self.sub_question_results:
            raise ValueError("simple pipeline cannot have sub-question results")
        return self
