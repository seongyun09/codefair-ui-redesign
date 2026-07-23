from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from insurance_rag.schemas.insurance_retrieval import (
    InsuranceRetrievalResult,
    RetrievedInsuranceEvidence,
)

ClaimType = Literal[
    "definition", "benefit_eligibility", "payment_condition", "payment_amount",
    "payment_timing", "exclusion", "premium_obligation", "premium_waiver",
    "contract_formation", "contract_change", "contract_termination",
    "contract_cancellation", "contract_invalidation", "contract_reinstatement",
    "refund", "claim_procedure", "document_requirement", "limitation_period",
    "dispute_resolution", "other",
]


def _nonblank(value: str) -> str:
    if not (value := value.strip()):
        raise ValueError("value must not be blank")
    return value


class InsuranceClaimExtractionRequest(BaseModel):
    original_question: str
    normalized_question: str
    sub_question_id: str | None = Field(default=None, pattern=r"^q[1-5]$")
    requested_action: str | None = None
    evidence: list[RetrievedInsuranceEvidence] = Field(default_factory=list)
    max_claims: int = Field(default=12, ge=1, le=30)
    retrieval_sufficient_evidence: bool | None = None
    retrieval_insufficiency_reason: str | None = None
    retrieval_warnings: list[str] = Field(default_factory=list)

    _questions_nonblank = field_validator(
        "original_question", "normalized_question"
    )(_nonblank)

    @field_validator(
        "sub_question_id", "requested_action", "retrieval_insufficiency_reason",
        mode="before",
    )
    @classmethod
    def optional_text(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @model_validator(mode="after")
    def evidence_text_is_required(self):
        if any(not item.text.strip() for item in self.evidence):
            raise ValueError("every evidence item must contain text")
        return self

    @field_validator("retrieval_warnings")
    @classmethod
    def clean_retrieval_warnings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @classmethod
    def from_retrieval(
        cls,
        retrieval: InsuranceRetrievalResult,
        *,
        normalized_question: str,
        sub_question_id: str | None = None,
        requested_action: str | None = None,
        max_claims: int = 12,
    ) -> "InsuranceClaimExtractionRequest":
        return cls(
            original_question=retrieval.original_question,
            normalized_question=normalized_question,
            sub_question_id=sub_question_id,
            requested_action=requested_action,
            evidence=retrieval.evidence,
            max_claims=max_claims,
            retrieval_sufficient_evidence=retrieval.sufficient_evidence,
            retrieval_insufficiency_reason=retrieval.insufficiency_reason,
            retrieval_warnings=retrieval.warnings,
        )


class ClaimSourceReference(BaseModel):
    evidence_rank: int | None = Field(default=None, ge=1)
    document_id: str | None = None
    chunk_id: str | None = None
    company: str | None = None
    product: str | None = None
    document_version: str | None = None
    part: str | None = None
    chapter: str | None = None
    article: str | None = None
    title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_file: str | None = None
    supporting_quote: str

    _quote_nonblank = field_validator("supporting_quote")(_nonblank)

    @field_validator(
        "document_id", "chunk_id", "company", "product", "document_version",
        "part", "chapter", "article", "title", "source_file",
        mode="before",
    )
    @classmethod
    def optional_text(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @model_validator(mode="after")
    def valid_page_range(self):
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        return self


class InsuranceEvidenceClaim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    legal_effect: str | None = None
    applies_when: list[str] = Field(default_factory=list)
    does_not_apply_when: list[str] = Field(default_factory=list)
    source_references: list[ClaimSourceReference] = Field(min_length=1)
    relevance: Literal["direct", "supporting", "context"]
    confidence: Literal["high", "medium", "low"]
    warnings: list[str] = Field(default_factory=list)

    _required_nonblank = field_validator("claim_id", "statement")(_nonblank)

    @field_validator(
        "subject", "predicate", "object", "legal_effect", mode="before"
    )
    @classmethod
    def optional_text(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator(
        "conditions", "exceptions", "applies_when", "does_not_apply_when", "warnings"
    )
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class InsuranceClaimExtractionResult(BaseModel):
    original_question: str
    claims: list[InsuranceEvidenceClaim] = Field(default_factory=list)
    claim_count: int = Field(ge=0)
    sufficient_evidence: bool
    insufficiency_reason: Literal[
        "no_evidence", "no_relevant_claims", "partial_evidence"
    ] | None = None
    retrieval_insufficiency_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    _question_nonblank = field_validator("original_question")(_nonblank)

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def consistent_result(self):
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count must equal the number of claims")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim_id values must be unique")
        if self.sufficient_evidence and self.insufficiency_reason is not None:
            raise ValueError("sufficient evidence cannot have an insufficiency reason")
        if not self.sufficient_evidence and self.insufficiency_reason is None:
            raise ValueError("insufficient evidence must include a reason")
        return self
