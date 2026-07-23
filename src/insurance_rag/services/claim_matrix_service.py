from __future__ import annotations

from collections import Counter

from insurance_rag.core.claim_matrix_builder import build_claim_matrix_rows
from insurance_rag.core.judgment_consistency_validator import (
    validate_claim_matrix_request,
)
from insurance_rag.core.judgment_disagreement_detector import detect_disagreements
from insurance_rag.schemas.claim_matrix import ClaimMatrixRequest, ClaimMatrixResult


def build_claim_matrix(request: ClaimMatrixRequest) -> ClaimMatrixResult:
    """Build a deterministic comparison without retrieval or model calls."""
    validate_claim_matrix_request(request)
    rows = build_claim_matrix_rows(request)
    model_ids = [
        *(item.model_id for item in request.judgment_batch.judgments),
        *(item.model_id for item in request.judgment_batch.failures),
    ]
    completed = request.judgment_batch.completed_model_count
    if not request.claims:
        status = "no_claims"
    elif completed < request.minimum_models_for_comparison:
        status = "insufficient_models"
    else:
        status = "ready"
    disagreements = detect_disagreements(request, rows)
    severity_counts = Counter(item.severity for item in disagreements)
    row_counts = Counter(row.agreement for row in rows)
    if status != "ready":
        overall = "indeterminate"
    elif severity_counts["critical"] or severity_counts["high"]:
        overall = "low"
    elif rows and row_counts["conflicted"] / len(rows) >= 0.25:
        overall = "low"
    elif row_counts["mixed"]:
        overall = "moderate"
    elif rows and all(row.agreement in {"unanimous", "strong"} for row in rows):
        overall = "high"
    else:
        overall = "moderate"
    review_types = {
        "exception_interpretation_difference", "legal_effect_difference",
        "missing_information_difference", "risk_constraint_violation",
    }
    requires_review = bool(
        (request.risk and request.risk.requires_human_review)
        or any(
            item.severity in {"critical", "high"}
            or item.disagreement_type in review_types
            or (
                item.disagreement_type == "disposition_conflict"
                and item.severity == "high"
            )
            for item in disagreements
        )
    )
    warnings = list(request.judgment_batch.warnings)
    if status == "insufficient_models":
        warnings.append("insufficient_models_for_claim_matrix_comparison")
    if status == "no_claims":
        warnings.append("claim_matrix_skipped_no_claims")
    return ClaimMatrixResult(
        original_question=request.original_question,
        sub_question_id=request.sub_question_id,
        model_ids=model_ids,
        rows=rows,
        claim_count=len(rows),
        comparison_status=status,
        sufficient_for_comparison=status == "ready",
        overall_agreement=overall,
        disagreements=disagreements,
        disagreement_counts=dict(sorted(
            Counter(item.disagreement_type for item in disagreements).items()
        )),
        requires_human_review=requires_review,
        warnings=list(dict.fromkeys(warnings)),
    )
