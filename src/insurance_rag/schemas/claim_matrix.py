from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
    IndependentJudgmentPipelineResult,
    JudgmentConfidence,
    JudgmentDisposition,
)
from insurance_rag.schemas.insurance_claim import ClaimType, InsuranceEvidenceClaim
from insurance_rag.schemas.question_risk import QuestionRiskGateResult

MatrixAssessment = Literal[
    "supports", "contradicts", "conditional", "irrelevant", "unclear",
    "not_assessed", "model_failed",
]
AgreementLevel = Literal[
    "unanimous", "strong", "mixed", "conflicted", "insufficient"
]
ComparisonStatus = Literal[
    "ready", "insufficient_models", "no_claims", "invalid_input"
]
OverallAgreement = Literal["high", "moderate", "low", "indeterminate"]
DisagreementType = Literal[
    "claim_assessment_conflict", "condition_interpretation_difference",
    "exception_interpretation_difference", "legal_effect_difference",
    "evidence_usage_difference", "confidence_gap", "disposition_conflict",
    "missing_information_difference", "human_review_difference",
    "risk_constraint_violation", "claim_omission", "other",
]
DisagreementSeverity = Literal["low", "medium", "high", "critical"]
RecommendedNextStep = Literal[
    "review_conditions", "review_exceptions", "review_sources",
    "request_more_information", "human_review", "no_action",
]


class ClaimMatrixRequest(BaseModel):
    original_question: str = Field(min_length=1, max_length=4000)
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    claims: list[InsuranceEvidenceClaim]
    judgment_batch: IndependentJudgmentBatchResult
    risk: QuestionRiskGateResult | None = None
    minimum_models_for_comparison: int = Field(default=2, ge=2, le=5)

    @field_validator("original_question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def validate_references(self):
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        judgments = self.judgment_batch.judgments
        judgment_ids = [item.judgment_id for item in judgments]
        model_ids = [item.model_id for item in judgments]
        failure_model_ids = [item.model_id for item in self.judgment_batch.failures]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("successful model_id values must be unique")
        if len(judgment_ids) != len(set(judgment_ids)):
            raise ValueError("judgment_id values must be unique")
        if set(model_ids) & set(failure_model_ids):
            raise ValueError("a model cannot both succeed and fail")
        if len(failure_model_ids) != len(set(failure_model_ids)):
            raise ValueError("failed model_id values must be unique")
        if self.judgment_batch.sub_question_id != self.sub_question_id:
            raise ValueError("batch sub_question_id must match matrix request")
        known = set(claim_ids)
        for judgment in judgments:
            if judgment.sub_question_id != self.sub_question_id:
                raise ValueError("judgment sub_question_id must match matrix request")
            assessed = {item.claim_id for item in judgment.claim_assessments}
            referenced = (
                assessed
                | set(judgment.supporting_claim_ids)
                | set(judgment.limiting_claim_ids)
                | set(judgment.conflicting_claim_ids)
            )
            if not referenced <= known:
                raise ValueError("judgment references an unknown claim_id")
        return self


class ClaimMatrixCell(BaseModel):
    model_id: str
    assessment: MatrixAssessment
    reasoning: str | None = None
    conditions_considered: list[str] = Field(default_factory=list)
    exceptions_considered: list[str] = Field(default_factory=list)
    source_reference_ids: list[str] = Field(default_factory=list)
    confidence: JudgmentConfidence | None = None


class ClaimMatrixRow(BaseModel):
    claim_id: str
    claim_type: ClaimType
    statement: str
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    legal_effect: str | None = None
    cells: list[ClaimMatrixCell]
    assessment_counts: dict[str, int]
    assessed_model_count: int = Field(ge=0)
    agreement: AgreementLevel


class JudgmentDisagreement(BaseModel):
    disagreement_id: str = Field(pattern=r"^(d[1-9][0-9]*|q[1-5]-d[1-9][0-9]*)$")
    disagreement_type: DisagreementType
    severity: DisagreementSeverity
    claim_id: str | None = None
    model_ids: list[str] = Field(min_length=1)
    description: str
    recommended_next_step: RecommendedNextStep


class ClaimMatrixResult(BaseModel):
    original_question: str
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    model_ids: list[str]
    rows: list[ClaimMatrixRow]
    claim_count: int = Field(ge=0)
    comparison_status: ComparisonStatus
    sufficient_for_comparison: bool
    overall_agreement: OverallAgreement
    disagreements: list[JudgmentDisagreement]
    disagreement_counts: dict[str, int]
    requires_human_review: bool
    warnings: list[str] = Field(default_factory=list)


class InsuranceClaimMatrixPipelineResult(BaseModel):
    judgments: IndependentJudgmentPipelineResult
    single_result: ClaimMatrixResult | None = None
    sub_question_results: list[ClaimMatrixResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
