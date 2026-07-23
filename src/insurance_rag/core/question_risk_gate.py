from __future__ import annotations

import re

from insurance_rag.schemas.question import QuestionAnalysis
from insurance_rag.schemas.question_risk import (
    QuestionRiskGateRequest,
    QuestionRiskGateResult,
)

_FRAUD = ("들키지 않고", "고지의무를 피", "속여", "허위 청구", "보험사기")
_LEGAL = ("법적으로", "무조건 패소", "무조건 승소", "소송에서 이기", "위법인가")
_CANCELLATION = ("해지하는 게", "해지해야", "당장 해지", "해약해야")
_PRODUCT = ("보험 추천", "상품 추천", "어떤 보험", "보험을 골라")
_FINANCIAL = ("재정적으로", "경제적으로 유리", "돈을 아끼려면")
_INDIVIDUAL = ("내 경우", "제가", "나는", "우리 가족", "무조건 나오", "받을 수 있나")
_POLICY = ("약관", "면책", "지급하지 않는 사유", "지급 조건", "해약환급금")
_PRIVACY = re.compile(
    r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)|"
    r"(?<!\d)01[016789]-?\d{3,4}-?\d{4}(?!\d)|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def evaluate_question_risk(
    value: QuestionAnalysis | QuestionRiskGateRequest,
) -> QuestionRiskGateResult:
    """Classify question risk without answering or making an insurance decision."""
    request = (
        QuestionRiskGateRequest.from_analysis(value)
        if isinstance(value, QuestionAnalysis)
        else value
    )
    text = " ".join([
        request.original_question,
        request.normalized_question,
        *request.sub_questions,
        request.requested_action or "",
    ]).casefold()
    missing = list(request.missing_information)
    categories: list[str] = []
    warnings = list(request.warnings)

    if _contains(text, _FRAUD):
        return QuestionRiskGateResult(
            risk_level="blocked",
            categories=["fraud_or_evasion"],
            allow_retrieval=False,
            allow_claim_extraction=False,
            allow_final_answer=False,
            requires_disclaimer=False,
            requires_human_review=True,
            requires_additional_information=False,
            blocked_reason="fraud_or_evasion",
            warnings=warnings,
        )

    if _contains(text, _LEGAL):
        categories.append("legal_conclusion")
    if _contains(text, _CANCELLATION):
        categories.append("cancellation_recommendation")
    if _contains(text, _PRODUCT):
        categories.append("product_recommendation")
    if _contains(text, _FINANCIAL):
        categories.append("financial_recommendation")
    if _contains(text, _INDIVIDUAL) and any(
        marker in text for marker in ("보험금", "지급", "나오", "받")
    ):
        categories.append("individual_claim_determination")
    if _contains(text, _POLICY):
        categories.append("policy_interpretation")
    if _PRIVACY.search(request.original_question):
        categories.append("privacy_sensitive")
        warnings.append("privacy_sensitive_input")
    if missing:
        categories.append("insufficient_information")

    high_categories = {
        "individual_claim_determination", "legal_conclusion",
        "financial_recommendation", "product_recommendation",
        "cancellation_recommendation",
    }
    if not categories:
        categories.append("general_information")
    high = bool(high_categories.intersection(categories))
    medium = "policy_interpretation" in categories or "privacy_sensitive" in categories

    if high and not missing:
        missing = ["개별 판단에 필요한 계약 및 사실관계"]
        categories.append("insufficient_information")
    requires_information = bool(missing) and (
        high or "insufficient_information" in categories
    )
    return QuestionRiskGateResult(
        risk_level="high" if high else ("medium" if medium else "low"),
        categories=list(dict.fromkeys(categories)),
        allow_retrieval=True,
        allow_claim_extraction=True,
        allow_final_answer=not high,
        requires_disclaimer=high or medium,
        requires_human_review=high,
        requires_additional_information=requires_information,
        missing_information=missing if requires_information else [],
        warnings=list(dict.fromkeys(warnings)),
    )
