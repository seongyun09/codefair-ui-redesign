from __future__ import annotations

import re
from difflib import SequenceMatcher

from insurance_rag.core.evidence_text_normalizer import quote_occurs_in_evidence
from insurance_rag.schemas.insurance_claim import (
    InsuranceClaimExtractionRequest,
    InsuranceClaimExtractionResult,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.insurance_retrieval import RetrievedInsuranceEvidence


class InsuranceClaimValidationError(ValueError):
    pass


_INJECTION_MARKERS = (
    "이전 지시를 무시", "시스템 프롬프트", "system prompt",
    "ignore previous instructions",
)


def _identifier_matches(reference, item: RetrievedInsuranceEvidence) -> bool:
    return (
        (reference.chunk_id is not None and item.chunk_id == reference.chunk_id)
        or (
            reference.evidence_rank is not None
            and item.rank == reference.evidence_rank
        )
        or (
            reference.document_id is not None
            and item.document_id == reference.document_id
            and (reference.article is not None or reference.page_start is not None)
        )
    )


def _source_mismatch_fields(
    reference, item: RetrievedInsuranceEvidence
) -> list[str]:
    mismatches = []
    # Once an evidence item is identified, every supplied source field must agree
    # with it. This prevents a real chunk ID from masking a fabricated article.
    for field in (
        "document_id", "chunk_id", "company", "product", "document_version",
        "part", "chapter", "article", "title", "source_file",
    ):
        value = getattr(reference, field)
        if value is not None and value != getattr(item, field):
            mismatches.append(field)
    if reference.evidence_rank is not None and reference.evidence_rank != item.rank:
        mismatches.append("evidence_rank")
    if reference.page_start is not None:
        if item.page_start is None or item.page_end is None:
            mismatches.append("page_range")
        else:
            end = reference.page_end or reference.page_start
            if end < item.page_start or reference.page_start > item.page_end:
                mismatches.append("page_range")
    return mismatches


def _source_matches(reference, item: RetrievedInsuranceEvidence) -> bool:
    return _identifier_matches(reference, item) and not _source_mismatch_fields(
        reference, item
    )


def _validate_and_canonicalize_sources(claim: InsuranceEvidenceClaim, request):
    for reference in claim.source_references:
        identified = [
            item for item in request.evidence
            if _identifier_matches(reference, item)
        ]
        if not identified:
            raise InsuranceClaimValidationError(
                "source reference identifier does not match evidence"
            )
        quote_matches = [
            item for item in identified
            if quote_occurs_in_evidence(reference.supporting_quote, item.text)
        ]
        if not quote_matches:
            raise InsuranceClaimValidationError("supporting quote is absent from evidence")
        # The model only chooses an evidence identifier and verbatim quote.
        # All descriptive metadata is replaced with the trusted retrieval value,
        # preventing fabricated article/page data without retrying harmless drift.
        source = quote_matches[0]
        for field in (
            "document_id", "chunk_id", "company", "product", "document_version",
            "part", "chapter", "article", "title", "page_start", "page_end",
            "source_file",
        ):
            setattr(reference, field, getattr(source, field))
        reference.evidence_rank = source.rank


def _statement_key(value: str) -> str:
    return re.sub(r"[\s.,!?·'\"“”‘’]", "", value).casefold()


def _source_specificity(claim: InsuranceEvidenceClaim) -> int:
    return max(
        sum(value is not None for value in (
            ref.chunk_id, ref.document_id, ref.article, ref.page_start, ref.source_file
        )) for ref in claim.source_references
    )


def _preferred(left: InsuranceEvidenceClaim, right: InsuranceEvidenceClaim):
    confidence = {"low": 0, "medium": 1, "high": 2}
    return max(
        (left, right),
        key=lambda claim: (confidence[claim.confidence], _source_specificity(claim)),
    )


def _deduplicate(claims: list[InsuranceEvidenceClaim]) -> list[InsuranceEvidenceClaim]:
    kept: list[InsuranceEvidenceClaim] = []
    for claim in claims:
        duplicate_index = None
        for index, prior in enumerate(kept):
            exact = _statement_key(claim.statement) == _statement_key(prior.statement)
            nearly = (
                claim.claim_type == prior.claim_type
                and claim.legal_effect == prior.legal_effect
                and claim.conditions == prior.conditions
                and claim.exceptions == prior.exceptions
                and SequenceMatcher(
                    None, _statement_key(claim.statement), _statement_key(prior.statement)
                ).ratio() >= .94
            )
            same_rule = (
                claim.claim_type == prior.claim_type
                and claim.conditions == prior.conditions
                and claim.exceptions == prior.exceptions
                and {(r.chunk_id, r.document_id, r.evidence_rank) for r in claim.source_references}
                == {(r.chunk_id, r.document_id, r.evidence_rank) for r in prior.source_references}
            )
            if exact or nearly or same_rule:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(claim)
        else:
            kept[duplicate_index] = _preferred(kept[duplicate_index], claim)
    return kept


def _mark_tensions(claims: list[InsuranceEvidenceClaim]) -> list[str]:
    positive = {"benefit_eligibility", "payment_condition", "payment_amount", "payment_timing"}
    has_positive = any(c.claim_type in positive or c.legal_effect == "보험금 지급" for c in claims)
    has_exclusion = any(c.claim_type == "exclusion" or c.legal_effect == "보험금 지급 제외" for c in claims)
    if has_positive and has_exclusion:
        for claim in claims:
            if claim.claim_type in positive | {"exclusion"}:
                claim.warnings = list(dict.fromkeys([
                    *claim.warnings, "potential_claim_tension", "requires_condition_comparison"
                ]))
        return ["potential_claim_tension", "requires_condition_comparison"]
    return []


def validate_and_finalize_claims(
    raw_result: dict, request: InsuranceClaimExtractionRequest
) -> InsuranceClaimExtractionResult:
    raw_claims = raw_result.get("claims")
    if not isinstance(raw_claims, list):
        raise InsuranceClaimValidationError("claims must be a list")
    if raw_result.get("claim_count") != len(raw_claims):
        raise InsuranceClaimValidationError(
            "claim_count must equal the number of claims"
        )
    if len(raw_claims) > request.max_claims:
        raise InsuranceClaimValidationError("max_claims exceeded")
    claims = [InsuranceEvidenceClaim.model_validate(item) for item in raw_claims]
    ids = [claim.claim_id for claim in claims]
    if len(ids) != len(set(ids)):
        raise InsuranceClaimValidationError("duplicate claim_id")
    for claim in claims:
        _validate_and_canonicalize_sources(claim, request)
        normalized_statement = claim.statement.casefold()
        if any(marker.casefold() in normalized_statement for marker in _INJECTION_MARKERS):
            raise InsuranceClaimValidationError(
                "prompt injection text must not become a claim"
            )
        if claim.confidence == "low" and not claim.warnings:
            raise InsuranceClaimValidationError("low confidence claim requires a warning")
    context_limit = max(1, request.max_claims // 4)
    if sum(claim.relevance == "context" for claim in claims) > context_limit:
        raise InsuranceClaimValidationError(
            f"context claims exceed limit of {context_limit}"
        )
    claims = _deduplicate(claims)
    for index, claim in enumerate(claims, 1):
        prefix = f"{request.sub_question_id}-" if request.sub_question_id else ""
        claim.claim_id = f"{prefix}c{index}"
    warnings = list(dict.fromkeys([
        *(str(value).strip() for value in raw_result.get("warnings", []) if str(value).strip()),
        *_mark_tensions(claims),
    ]))
    if not claims:
        return InsuranceClaimExtractionResult(
            original_question=request.original_question, claims=[], claim_count=0,
            sufficient_evidence=False, insufficiency_reason="no_relevant_claims",
            retrieval_insufficiency_reason=request.retrieval_insufficiency_reason,
            warnings=warnings,
        )
    partial = all(
        claim.confidence == "low" and "partial_clause" in claim.warnings
        for claim in claims
    )
    return InsuranceClaimExtractionResult(
        original_question=request.original_question,
        claims=claims,
        claim_count=len(claims),
        sufficient_evidence=not partial,
        insufficiency_reason="partial_evidence" if partial else None,
        retrieval_insufficiency_reason=request.retrieval_insufficiency_reason,
        warnings=warnings,
    )
