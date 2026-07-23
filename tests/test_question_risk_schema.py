import pytest
from pydantic import ValidationError

from insurance_rag.schemas.question_risk import QuestionRiskGateResult


def result(**changes):
    values = {
        "risk_level": "low",
        "categories": ["general_information"],
        "allow_retrieval": True,
        "allow_claim_extraction": True,
        "allow_final_answer": True,
        "requires_disclaimer": False,
        "requires_human_review": False,
        "requires_additional_information": False,
    }
    values.update(changes)
    return QuestionRiskGateResult(**values)


def test_retrieval_block_also_blocks_extraction():
    with pytest.raises(ValidationError, match="requires retrieval"):
        result(allow_retrieval=False, allow_claim_extraction=True)


def test_blocked_requires_reason_and_all_permissions_off():
    with pytest.raises(ValidationError):
        result(risk_level="blocked", allow_final_answer=False)
    blocked = result(
        risk_level="blocked", categories=["fraud_or_evasion"],
        allow_retrieval=False, allow_claim_extraction=False,
        allow_final_answer=False, blocked_reason="fraud_or_evasion",
    )
    assert blocked.blocked_reason == "fraud_or_evasion"


def test_additional_information_requires_list():
    with pytest.raises(ValidationError, match="missing information"):
        result(requires_additional_information=True)
