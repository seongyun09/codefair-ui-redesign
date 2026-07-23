from __future__ import annotations

import re

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentRequest,
    IndependentModelJudgment,
)


class IndependentJudgmentError(Exception):
    pass


class IndependentJudgmentValidationError(IndependentJudgmentError):
    pass


class IndependentJudgmentConfigurationError(IndependentJudgmentError):
    pass


_DEFINITIVE_PATTERNS = (
    r"보험금(?:은|이)?\s*무조건\s*지급",
    r"보험사(?:는|가)?\s*반드시\s*패소",
    r"(?:즉시|당장)\s*해지해야",
    r"(?:당신은|귀하는).{0,20}보험금을\s*받을\s*수\s*있",
)


def _source_ids(request: IndependentJudgmentRequest) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for claim in request.claims:
        values[claim.claim_id] = {
            reference.chunk_id
            for reference in claim.source_references
            if reference.chunk_id
        }
    return values


def validate_model_judgment(
    raw_result: dict,
    request: IndependentJudgmentRequest,
    *,
    model_id: str,
) -> IndependentModelJudgment:
    judgment = IndependentModelJudgment.model_validate(raw_result)
    claim_ids = {claim.claim_id for claim in request.claims}
    assessed = {item.claim_id for item in judgment.claim_assessments}
    if assessed != claim_ids:
        raise IndependentJudgmentValidationError(
            "claim_assessments must cover every input claim exactly once"
        )
    for field in (
        "supporting_claim_ids", "limiting_claim_ids", "conflicting_claim_ids"
    ):
        unknown = set(getattr(judgment, field)) - claim_ids
        if unknown:
            raise IndependentJudgmentValidationError(
                f"{field} contains an unknown claim_id"
            )
    source_ids = _source_ids(request)
    for assessment in judgment.claim_assessments:
        unknown_sources = (
            set(assessment.source_reference_ids) - source_ids[assessment.claim_id]
        )
        if unknown_sources:
            raise IndependentJudgmentValidationError(
                "source_reference_ids contain an unknown source"
            )
    if any(
        re.search(pattern, judgment.conclusion)
        for pattern in _DEFINITIVE_PATTERNS
    ):
        raise IndependentJudgmentValidationError(
            "conclusion contains a prohibited definitive statement"
        )

    judgment.model_id = model_id
    judgment.sub_question_id = request.sub_question_id
    judgment.requires_human_review = (
        judgment.requires_human_review or request.requires_human_review
    )
    judgment.requires_disclaimer = (
        judgment.requires_disclaimer or request.requires_disclaimer
    )
    if request.requires_human_review:
        judgment.warnings = list(dict.fromkeys([
            *judgment.warnings, "risk_gate_requires_human_review",
        ]))

    conflicting = set(judgment.conflicting_claim_ids)
    limiting = set(judgment.limiting_claim_ids) - conflicting
    supporting = (
        set(judgment.supporting_claim_ids) - conflicting - limiting
    )
    judgment.conflicting_claim_ids = [
        value for value in judgment.conflicting_claim_ids if value in conflicting
    ]
    judgment.limiting_claim_ids = [
        value for value in judgment.limiting_claim_ids if value in limiting
    ]
    judgment.supporting_claim_ids = [
        value for value in judgment.supporting_claim_ids if value in supporting
    ]
    return judgment
