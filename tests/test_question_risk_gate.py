import pytest

from insurance_rag.core.question_risk_gate import evaluate_question_risk
from insurance_rag.schemas.question_risk import QuestionRiskGateRequest


def request(question: str, **changes):
    values = {
        "original_question": question,
        "normalized_question": question,
    }
    values.update(changes)
    return QuestionRiskGateRequest(**values)


@pytest.mark.parametrize(
    ("question", "level", "category", "final_answer"),
    [
        ("종신보험의 해약환급금이 뭐야?", "medium", "policy_interpretation", True),
        (
            "이 약관에서 사망보험금을 지급하지 않는 사유가 뭐야?",
            "medium", "policy_interpretation", True,
        ),
        (
            "내 경우 보험금이 무조건 나오나요?",
            "high", "individual_claim_determination", False,
        ),
        (
            "보험사가 법적으로 무조건 패소하나요?",
            "high", "legal_conclusion", False,
        ),
        ("어떤 보험을 추천해?", "high", "product_recommendation", False),
        (
            "이 보험을 지금 당장 해지하는 게 맞나요?",
            "high", "cancellation_recommendation", False,
        ),
    ],
)
def test_risk_categories(question, level, category, final_answer):
    result = evaluate_question_risk(request(question))
    assert result.risk_level == level
    assert category in result.categories
    assert result.allow_retrieval and result.allow_claim_extraction
    assert result.allow_final_answer is final_answer


def test_plain_definition_is_low_risk():
    result = evaluate_question_risk(request("보험수익자는 누구인가요?"))
    assert result.risk_level == "low"
    assert result.categories == ["general_information"]


def test_fraud_or_evasion_blocks_downstream_steps():
    result = evaluate_question_risk(
        request("보험사에 들키지 않고 고지의무를 피하려면?")
    )
    assert result.risk_level == "blocked"
    assert not result.allow_retrieval
    assert not result.allow_claim_extraction
    assert result.blocked_reason == "fraud_or_evasion"


def test_privacy_and_missing_information_are_preserved():
    result = evaluate_question_risk(request(
        "제 주민번호 900101-1234567로 보험금을 받을 수 있나요?",
        missing_information=["가입 약관", "사고 경위"],
    ))
    assert "privacy_sensitive" in result.categories
    assert "insufficient_information" in result.categories
    assert result.requires_additional_information
    assert result.missing_information == ["가입 약관", "사고 경위"]
    assert "privacy_sensitive_input" in result.warnings


def test_compound_question_combines_risk_categories():
    result = evaluate_question_risk(request(
        "약관도 설명하고 해지해야 하는지 알려줘",
        is_compound=True,
        sub_questions=[
            "이 약관의 면책 사유는?",
            "이 보험을 당장 해지하는 게 맞나요?",
        ],
    ))
    assert result.risk_level == "high"
    assert {"policy_interpretation", "cancellation_recommendation"} <= set(
        result.categories
    )
