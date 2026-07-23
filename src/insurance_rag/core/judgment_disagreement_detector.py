from __future__ import annotations

from itertools import combinations

from insurance_rag.core.judgment_consistency_validator import normalized_values
from insurance_rag.schemas.claim_matrix import (
    ClaimMatrixRequest,
    ClaimMatrixRow,
    JudgmentDisagreement,
)


def _pairs(values):
    return combinations(values, 2)


def detect_disagreements(
    request: ClaimMatrixRequest,
    rows: list[ClaimMatrixRow],
) -> list[JudgmentDisagreement]:
    found: list[tuple] = []
    judgments = request.judgment_batch.judgments
    by_model = {item.model_id: item for item in judgments}

    def add(kind, severity, models, description, step, claim_id=None):
        found.append((kind, severity, sorted(models), description, step, claim_id))

    for row in rows:
        assessed = [
            cell for cell in row.cells
            if cell.assessment not in {"not_assessed", "model_failed"}
        ]
        omitted = [cell.model_id for cell in row.cells if cell.assessment == "not_assessed"]
        if omitted and assessed:
            add(
                "claim_omission", "medium" if len(omitted) >= len(assessed) else "low",
                [*omitted, *(cell.model_id for cell in assessed)],
                f"{row.claim_id} was omitted by one or more successful models.",
                "review_sources", row.claim_id,
            )
        for left, right in _pairs(assessed):
            pair = {left.assessment, right.assessment}
            if pair == {"supports", "contradicts"}:
                add(
                    "claim_assessment_conflict", "high",
                    [left.model_id, right.model_id],
                    f"{row.claim_id} is both supported and contradicted.",
                    "human_review", row.claim_id,
                )
                if row.legal_effect:
                    add(
                        "legal_effect_difference", "high",
                        [left.model_id, right.model_id],
                        f"Models disagree on the legal effect of {row.claim_id}.",
                        "human_review", row.claim_id,
                    )
            elif pair == {"supports", "conditional"}:
                add(
                    "claim_assessment_conflict", "medium",
                    [left.model_id, right.model_id],
                    f"Models differ on whether {row.claim_id} is conditional.",
                    "review_conditions", row.claim_id,
                )
            elif pair == {"conditional", "unclear"}:
                add(
                    "claim_assessment_conflict", "medium",
                    [left.model_id, right.model_id],
                    f"Models differ on conditional interpretation of {row.claim_id}.",
                    "review_conditions", row.claim_id,
                )
            if normalized_values(left.conditions_considered) != normalized_values(
                right.conditions_considered
            ):
                add(
                    "condition_interpretation_difference", "medium",
                    [left.model_id, right.model_id],
                    f"Models considered different conditions for {row.claim_id}.",
                    "review_conditions", row.claim_id,
                )
            if normalized_values(left.exceptions_considered) != normalized_values(
                right.exceptions_considered
            ):
                add(
                    "exception_interpretation_difference", "high",
                    [left.model_id, right.model_id],
                    f"Models considered different exceptions for {row.claim_id}.",
                    "review_exceptions", row.claim_id,
                )
            def evidence_usage(cell):
                judgment = by_model[cell.model_id]
                roles = set(cell.source_reference_ids)
                for name, values in (
                    ("supporting", judgment.supporting_claim_ids),
                    ("limiting", judgment.limiting_claim_ids),
                    ("conflicting", judgment.conflicting_claim_ids),
                ):
                    if row.claim_id in values:
                        roles.add(f"role:{name}")
                return roles

            if evidence_usage(left) != evidence_usage(right):
                add(
                    "evidence_usage_difference", "low",
                    [left.model_id, right.model_id],
                    f"Models used different source references for {row.claim_id}.",
                    "review_sources", row.claim_id,
                )
            confidence = {"low": 0, "medium": 1, "high": 2}
            gap = abs(confidence[left.confidence] - confidence[right.confidence])
            if gap:
                add(
                    "confidence_gap", "medium" if gap == 2 else "low",
                    [left.model_id, right.model_id],
                    f"Models report different confidence for {row.claim_id}.",
                    "no_action", row.claim_id,
                )

    disposition_severity = {
        frozenset({"supported", "not_supported"}): "high",
        frozenset({"supported", "requires_human_review"}): "high",
        frozenset({"partially_supported", "insufficient_information"}): "medium",
    }
    for left, right in _pairs(judgments):
        if left.disposition != right.disposition:
            add(
                "disposition_conflict",
                disposition_severity.get(
                    frozenset({left.disposition, right.disposition}), "low"
                ),
                [left.model_id, right.model_id],
                "Models reached different overall dispositions.",
                "human_review",
            )
        if normalized_values(left.missing_information) != normalized_values(
            right.missing_information
        ):
            add(
                "missing_information_difference", "medium",
                [left.model_id, right.model_id],
                "Models identified different missing information.",
                "request_more_information",
            )
        if left.requires_human_review != right.requires_human_review:
            add(
                "human_review_difference", "medium",
                [left.model_id, right.model_id],
                "Models disagree on whether human review is required.",
                "human_review",
            )

    if request.risk:
        definitive = {"supported", "partially_supported", "not_supported"}
        for judgment in judgments:
            violates_review = (
                request.risk.requires_human_review
                and not judgment.requires_human_review
            )
            violates_answer = (
                not request.risk.allow_final_answer
                and judgment.disposition in definitive
            )
            if violates_review or violates_answer:
                add(
                    "risk_constraint_violation", "critical", [judgment.model_id],
                    "Model judgment violates risk-gate constraints.", "human_review",
                )

    prefix = f"{request.sub_question_id}-" if request.sub_question_id else ""
    return [
        JudgmentDisagreement(
            disagreement_id=f"{prefix}d{index}",
            disagreement_type=item[0],
            severity=item[1],
            model_ids=item[2],
            description=item[3],
            recommended_next_step=item[4],
            claim_id=item[5],
        )
        for index, item in enumerate(found, 1)
    ]
