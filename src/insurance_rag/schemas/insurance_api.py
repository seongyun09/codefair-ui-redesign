from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from insurance_rag.schemas.claim_matrix import OverallAgreement

ApiAnalysisStatus = Literal[
    "completed", "limited", "needs_information", "insufficient_evidence",
    "blocked", "partial", "failed",
]


class InsuranceAnalysisRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    user_context: dict[str, Any] = Field(default_factory=dict)
    vector_store_id: str | None = None
    include_debug: bool = False

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        if len(value := value.strip()) < 2:
            raise ValueError("question must contain at least two characters")
        return value

    @field_validator("vector_store_id", mode="before")
    @classmethod
    def clean_store_id(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None


class ApiSourceReference(BaseModel):
    claim_id: str
    document_id: str | None = None
    company: str | None = None
    product: str | None = None
    document_type: str | None = None
    document_version: str | None = None
    article: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_file: str | None = None


class InsuranceSubAnswer(BaseModel):
    sub_question_id: str = Field(pattern=r"^q[1-5]$")
    question: str
    status: ApiAnalysisStatus
    answer: str | None = None
    sources: list[ApiSourceReference] = Field(default_factory=list)
    model_agreement: OverallAgreement | None = None
    disagreements: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class InsuranceAnalysisResponse(BaseModel):
    request_id: str
    status: ApiAnalysisStatus
    answer: str | None = None
    key_points: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    important_exceptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    sources: list[ApiSourceReference] = Field(default_factory=list)
    sub_answers: list[InsuranceSubAnswer] = Field(default_factory=list)
    model_agreement: OverallAgreement | None = None
    disagreements: list[str] = Field(default_factory=list)
    requires_disclaimer: bool = False
    requires_human_review: bool = False
    stopped_at: str | None = None
    reason_code: str | None = None
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: int = Field(ge=0)


class InsuranceAnalysisErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    retryable: bool
    details: list[str] = Field(default_factory=list)
