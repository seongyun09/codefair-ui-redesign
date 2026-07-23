from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.insurance_claim import InsuranceClaimExtractionResult
from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalResult
from insurance_rag.schemas.question import QuestionAnalysis
from insurance_rag.schemas.question_risk import QuestionRiskGateResult

PipelineStatus = Literal[
    "completed", "blocked", "needs_information",
    "insufficient_evidence", "partial", "failed",
]
PipelineStage = Literal["risk_gate", "retrieval", "claim_extraction"]


class SubQuestionEvidenceResult(BaseModel):
    sub_question_id: str = Field(pattern=r"^q[1-5]$")
    question: str = Field(min_length=2)
    retrieval: InsuranceRetrievalResult | None = None
    claims: InsuranceClaimExtractionResult | None = None
    status: PipelineStatus
    stopped_at: PipelineStage | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class InsuranceEvidencePipelineResult(BaseModel):
    status: PipelineStatus
    analysis: QuestionAnalysis
    risk: QuestionRiskGateResult
    retrieval: InsuranceRetrievalResult | None = None
    claims: InsuranceClaimExtractionResult | None = None
    sub_question_results: list[SubQuestionEvidenceResult] = Field(
        default_factory=list
    )
    stopped_at: PipelineStage | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def valid_shape(self):
        if self.analysis.is_compound:
            if not self.sub_question_results:
                raise ValueError("compound analysis requires sub-question results")
            if self.retrieval is not None or self.claims is not None:
                raise ValueError("compound results use sub-question retrieval and claims")
        elif self.sub_question_results:
            raise ValueError("simple analysis must not have sub-question results")
        if self.status == "blocked" and self.stopped_at != "risk_gate":
            raise ValueError("blocked pipeline must stop at risk gate")
        return self
