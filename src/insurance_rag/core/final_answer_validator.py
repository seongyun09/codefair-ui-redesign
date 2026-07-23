from __future__ import annotations

import re

from insurance_rag.schemas.final_answer import (
    FinalAnswerSource,
    FinalAnswerSynthesisRequest,
    FinalInsuranceAnswer,
)


class FinalAnswerValidationError(ValueError):
    pass


_PROHIBITED = (
    r"보험금(?:은|이)?\s*무조건\s*지급",
    r"보험사(?:는|가)?\s*반드시\s*패소",
    r"(?:즉시|당장)\s*(?:이\s*)?보험을?\s*해지",
    r"(?:당신은|귀하는).{0,20}확실히.{0,20}보험금을\s*받",
)


def _source_key(value) -> tuple:
    return (
        value.document_id, value.article, value.title,
        value.page_start, value.page_end, value.source_file,
    )


def _matches_verified_source(source, reference) -> bool:
    for field in (
        "document_id", "article", "title", "page_start", "page_end", "source_file"
    ):
        supplied = getattr(source, field)
        if supplied is not None and supplied != getattr(reference, field):
            return False
    return True


def validate_final_answer(
    raw_result: dict,
    request: FinalAnswerSynthesisRequest,
) -> FinalInsuranceAnswer:
    answer = FinalInsuranceAnswer.model_validate(raw_result)
    combined = " ".join([answer.answer, *answer.key_points])
    if any(re.search(pattern, combined) for pattern in _PROHIBITED):
        raise FinalAnswerValidationError(
            "answer contains a prohibited definitive statement"
        )

    claims = {claim.claim_id: claim for claim in request.claims}
    for source in answer.sources:
        if source.claim_id not in claims:
            raise FinalAnswerValidationError("source references an unknown claim_id")
        if not any(
            _matches_verified_source(source, reference)
            for reference in claims[source.claim_id].source_references
        ):
            raise FinalAnswerValidationError(
                "source metadata does not match a verified Claim source"
            )

    # These user-facing lists are derived exclusively from verified Claims.
    # Model paraphrases are deliberately discarded rather than exposed.
    answer.applicable_conditions = list(dict.fromkeys(
        value
        for claim in request.claims
        for value in [*claim.conditions, *claim.applies_when]
    ))
    answer.important_exceptions = list(dict.fromkeys(
        value
        for claim in request.claims
        for value in [*claim.exceptions, *claim.does_not_apply_when]
    ))
    answer.missing_information = list(dict.fromkeys([
        *answer.missing_information,
        *request.risk.missing_information,
        *(
            value
            for judgment in request.judgments.judgments
            for value in judgment.missing_information
        ),
    ]))
    existing_sources = {
        (source.claim_id, _source_key(source)) for source in answer.sources
    }
    for claim in request.claims:
        for reference in claim.source_references:
            key = (claim.claim_id, _source_key(reference))
            if key in existing_sources:
                continue
            answer.sources.append(FinalAnswerSource(
                claim_id=claim.claim_id,
                document_id=reference.document_id,
                article=reference.article,
                title=reference.title,
                page_start=reference.page_start,
                page_end=reference.page_end,
                source_file=reference.source_file,
            ))
            existing_sources.add(key)

    answer.model_agreement = request.claim_matrix.overall_agreement
    answer.requires_disclaimer = (
        answer.requires_disclaimer or request.risk.requires_disclaimer
    )
    answer.requires_human_review = (
        answer.requires_human_review
        or request.risk.requires_human_review
        or request.claim_matrix.requires_human_review
    )
    if not request.risk.allow_final_answer:
        if answer.status not in {"limited", "needs_information"}:
            answer.status = (
                "needs_information"
                if request.risk.requires_additional_information
                else "limited"
            )
        answer.requires_human_review = True
        answer.warnings = list(dict.fromkeys([
            *answer.warnings, "risk_gate_disallows_final_answer",
        ]))
    if request.judgments.completed_model_count < 2:
        answer.status = "limited"
        answer.requires_human_review = True
        answer.warnings = list(dict.fromkeys([
            *answer.warnings, "insufficient_independent_models",
        ]))
    if request.claim_matrix.overall_agreement == "low":
        summaries = [
            item.description for item in request.claim_matrix.disagreements
            if item.severity in {"critical", "high"}
        ]
        if summaries:
            answer.answer = (
                f"{answer.answer}\n\n모델 간 주요 불일치: "
                + " ".join(summaries[:3])
            )
        answer.requires_human_review = True
        answer.warnings = list(dict.fromkeys([
            *answer.warnings, "low_model_agreement",
        ]))
    return answer
