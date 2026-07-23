import pytest

from insurance_rag.core.independent_judgment_validator import (
    IndependentJudgmentValidationError,
    validate_model_judgment,
)
from test_independent_judgment_schema import claim, request


def raw(**changes):
    value = {
        "judgment_id": "j99",
        "model_id": "untrusted-model",
        "disposition": "partially_supported",
        "conclusion": "Claims는 지급 조건을 지지하지만 개별 적용은 확정할 수 없다.",
        "claim_assessments": [{
            "claim_id": "c1",
            "assessment": "conditional",
            "reasoning": "조건 충족 여부 확인이 필요하다.",
            "conditions_considered": ["보험기간 중 사망"],
            "source_reference_ids": ["chunk-1"],
            "confidence": "medium",
        }],
        "supporting_claim_ids": ["c1"],
        "limiting_claim_ids": [],
        "conflicting_claim_ids": [],
        "requires_disclaimer": False,
        "requires_human_review": False,
        "confidence": "medium",
    }
    value.update(changes)
    return value


def test_validates_claims_sources_and_forces_model_and_risk_restrictions():
    value = request(
        allow_final_answer=False,
        requires_human_review=True,
        requires_disclaimer=True,
    )
    result = validate_model_judgment(raw(), value, model_id="model-a")
    assert result.model_id == "model-a"
    assert result.requires_human_review
    assert result.requires_disclaimer
    assert "risk_gate_requires_human_review" in result.warnings


@pytest.mark.parametrize(
    "changes",
    [
        {"claim_assessments": [{
            "claim_id": "c99", "assessment": "supports",
            "reasoning": "없는 Claim", "confidence": "low",
        }]},
        {"supporting_claim_ids": ["c99"]},
        {"claim_assessments": [{
            "claim_id": "c1", "assessment": "supports",
            "reasoning": "출처 오류",
            "source_reference_ids": ["missing-chunk"], "confidence": "low",
        }]},
    ],
)
def test_rejects_unknown_claim_or_source_ids(changes):
    with pytest.raises(IndependentJudgmentValidationError):
        validate_model_judgment(raw(**changes), request(), model_id="model-a")


def test_rejects_definitive_conclusion_when_final_answer_is_blocked():
    with pytest.raises(IndependentJudgmentValidationError, match="definitive"):
        validate_model_judgment(
            raw(conclusion="보험금은 무조건 지급됩니다."),
            request(allow_final_answer=False),
            model_id="model-a",
        )


def test_conflicting_classification_takes_precedence():
    result = validate_model_judgment(
        raw(
            supporting_claim_ids=["c1"],
            limiting_claim_ids=["c1"],
            conflicting_claim_ids=["c1"],
        ),
        request(),
        model_id="model-a",
    )
    assert result.conflicting_claim_ids == ["c1"]
    assert result.supporting_claim_ids == []
    assert result.limiting_claim_ids == []


def test_conditions_and_exceptions_are_assessed_per_claim():
    claims = [
        claim("c1"),
        claim("c2").model_copy(update={
            "claim_type": "exclusion",
            "statement": "고의 사고에는 보험금을 지급하지 않는다.",
            "conditions": ["고의 사고"],
            "legal_effect": "보험금 지급 제외",
        }),
        claim("c3").model_copy(update={
            "claim_type": "exclusion",
            "statement": "자유로운 의사결정이 불가능한 상태는 예외다.",
            "exceptions": ["자유로운 의사결정 불가"],
        }),
    ]
    assessments = [
        {
            "claim_id": "c1", "assessment": "supports",
            "reasoning": "지급 조건을 지지한다.",
            "conditions_considered": ["보험기간 중 사망"],
            "source_reference_ids": ["chunk-1"], "confidence": "high",
        },
        {
            "claim_id": "c2", "assessment": "conditional",
            "reasoning": "면책 조건 적용 여부가 필요하다.",
            "conditions_considered": ["고의 사고"],
            "source_reference_ids": ["chunk-1"], "confidence": "medium",
        },
        {
            "claim_id": "c3", "assessment": "unclear",
            "reasoning": "면책 예외 사실관계가 부족하다.",
            "exceptions_considered": ["자유로운 의사결정 불가"],
            "source_reference_ids": ["chunk-1"], "confidence": "low",
        },
    ]
    result = validate_model_judgment(
        raw(
            claim_assessments=assessments,
            supporting_claim_ids=["c1"],
            limiting_claim_ids=["c2", "c3"],
        ),
        request(claims=claims),
        model_id="model-a",
    )
    assert [item.assessment for item in result.claim_assessments] == [
        "supports", "conditional", "unclear",
    ]
