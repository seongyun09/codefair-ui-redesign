from __future__ import annotations

import re

from insurance_rag.schemas.claim_matrix import ClaimMatrixRequest


def normalize_interpretation(value: str) -> str:
    """Normalize only presentation differences, preserving semantic words."""
    value = re.sub(r"[.,;:!?()\[\]{}'\"`]", "", value.casefold().strip())
    return re.sub(r"\s+", " ", value)


def normalized_values(values: list[str]) -> set[str]:
    return {normalize_interpretation(value) for value in values if value.strip()}


def validate_claim_matrix_request(request: ClaimMatrixRequest) -> None:
    """Expose schema consistency validation as an explicit pure boundary."""
    ClaimMatrixRequest.model_validate(request.model_dump())
