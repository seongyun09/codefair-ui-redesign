from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.claim_matrix import ClaimMatrixResult, OverallAgreement
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
)
from insurance_rag.schemas.insurance_claim import InsuranceEvidenceClaim
from insurance_rag.schemas.insurance_pipeline import InsuranceEvidencePipelineResult
from insurance_rag.schemas.question_risk import QuestionRiskGateResult

FinalAnswerStatus = Literal[
    "completed", "limited", "needs_information", "insufficient_evidence", "blocked"
]
AnswerPipelineStatus = Literal[
    "completed", "partial", "limited", "blocked", "insufficient_evidence", "failed"
]


def _clean(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class FinalAnswerSource(BaseModel):
    claim_id: str
    document_id: str | None = None
    article: str | None = None
    title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_file: str | None = None


class FinalInsuranceAnswer(BaseModel):
    status: FinalAnswerStatus
    answer: str = Field(min_length=1, max_length=6000)
    key_points: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    important_exceptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    sources: list[FinalAnswerSource] = Field(default_factory=list)
    model_agreement: OverallAgreement
    requires_disclaimer: bool
    requires_human_review: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "key_points", "applicable_conditions", "important_exceptions",
        "missing_information", "warnings",
    )
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return _clean(values)


class FinalAnswerSynthesisRequest(BaseModel):
    original_question: str = Field(min_length=1, max_length=4000)
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    risk: QuestionRiskGateResult
    claims: list[InsuranceEvidenceClaim] = Field(min_length=1)
    judgments: IndependentJudgmentBatchResult
    claim_matrix: ClaimMatrixResult

    @model_validator(mode="after")
    def aligned_inputs(self):
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim_id values must be unique")
        if self.judgments.sub_question_id != self.sub_question_id:
            raise ValueError("judgment sub_question_id must match synthesis request")
        if self.claim_matrix.sub_question_id != self.sub_question_id:
            raise ValueError("matrix sub_question_id must match synthesis request")
        if {row.claim_id for row in self.claim_matrix.rows} != claim_ids:
            raise ValueError("matrix rows must cover the synthesis claims")
        return self


class SubQuestionFinalAnswer(BaseModel):
    sub_question_id: str = Field(pattern=r"^q[1-5]$")
    question: str
    independent_judgments: IndependentJudgmentBatchResult
    claim_matrix: ClaimMatrixResult
    final_answer: FinalInsuranceAnswer


class InsuranceAnswerPipelineResult(BaseModel):
    status: AnswerPipelineStatus
    evidence_pipeline: InsuranceEvidencePipelineResult
    independent_judgments: IndependentJudgmentBatchResult | None = None
    claim_matrix: ClaimMatrixResult | None = None
    final_answer: FinalInsuranceAnswer | None = None
    sub_question_results: list[SubQuestionFinalAnswer] = Field(default_factory=list)
    stopped_at: str | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
