from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class InsuranceRetrievalRequest(BaseModel):
    original_question: str = Field(min_length=1, max_length=4000)
    normalized_question: str = Field(min_length=1, max_length=1000)
    company: str | None = None
    company_code: str | None = None
    product: str | None = None
    product_code: str | None = None
    product_type: str | None = None
    document_type: str = "insurance_terms"
    document_version: str | None = None
    requested_action: str | None = None
    search_topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=20)

    @field_validator("original_question", "normalized_question")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("question must not be blank")
        return value

    @field_validator("company", "company_code", "product", "product_code", "product_type", "document_type", "document_version", "requested_action", mode="before")
    @classmethod
    def optional_text(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("search_topics", "keywords")
    @classmethod
    def unique_nonblank(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class RetrievedInsuranceEvidence(BaseModel):
    rank: int = Field(ge=1)
    score: float | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    company: str | None = None
    product: str | None = None
    document_version: str | None = None
    part: str | None = None
    chapter: str | None = None
    article: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    source_file: str | None = None


class InsuranceRetrievalResult(BaseModel):
    original_question: str
    search_query: str
    applied_filters: dict[str, object] | None = None
    evidence: list[RetrievedInsuranceEvidence] = Field(default_factory=list)
    result_count: int
    sufficient_evidence: bool
    insufficiency_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
