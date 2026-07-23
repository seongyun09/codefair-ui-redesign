from __future__ import annotations

from collections import Counter

from insurance_rag.schemas.claim_matrix import (
    ClaimMatrixCell,
    ClaimMatrixRequest,
    ClaimMatrixRow,
)


def _agreement(cells: list[ClaimMatrixCell]) -> str:
    assessed = [
        cell.assessment for cell in cells
        if cell.assessment not in {"not_assessed", "model_failed"}
    ]
    if len(assessed) < 2:
        return "insufficient"
    counts = Counter(assessed)
    if len(counts) == 1:
        return "unanimous"
    if "supports" in counts and "contradicts" in counts:
        return "conflicted"
    if max(counts.values()) / len(assessed) >= 0.8:
        return "strong"
    return "mixed"


def build_claim_matrix_rows(request: ClaimMatrixRequest) -> list[ClaimMatrixRow]:
    judgments = {item.model_id: item for item in request.judgment_batch.judgments}
    failed_ids = [item.model_id for item in request.judgment_batch.failures]
    failed = set(failed_ids)
    model_ids = [*judgments, *failed_ids]
    rows = []
    for claim in request.claims:
        cells = []
        for model_id in model_ids:
            if model_id in failed:
                cells.append(ClaimMatrixCell(
                    model_id=model_id, assessment="model_failed"
                ))
                continue
            assessment = next((
                item for item in judgments[model_id].claim_assessments
                if item.claim_id == claim.claim_id
            ), None)
            if assessment is None:
                cells.append(ClaimMatrixCell(
                    model_id=model_id, assessment="not_assessed"
                ))
            else:
                cells.append(ClaimMatrixCell(
                    model_id=model_id, **assessment.model_dump(exclude={"claim_id"})
                ))
        counts = Counter(cell.assessment for cell in cells)
        rows.append(ClaimMatrixRow(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            statement=claim.statement,
            conditions=claim.conditions,
            exceptions=claim.exceptions,
            legal_effect=claim.legal_effect,
            cells=cells,
            assessment_counts=dict(sorted(counts.items())),
            assessed_model_count=sum(
                cell.assessment not in {"not_assessed", "model_failed"}
                for cell in cells
            ),
            agreement=_agreement(cells),
        ))
    return rows
