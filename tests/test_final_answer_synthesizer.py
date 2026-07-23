from types import SimpleNamespace

import pytest

from insurance_rag.core import final_answer_synthesizer as synthesizer
from insurance_rag.core.final_answer_validator import (
    FinalAnswerValidationError,
    validate_final_answer,
)
from insurance_rag.schemas.final_answer import FinalAnswerSynthesisRequest
from insurance_rag.schemas.question_risk import QuestionRiskGateResult
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from test_claim_matrix_service import claim, model, request


def risk(*, allow=True, missing=False):
    return QuestionRiskGateResult(
        risk_level="medium",
        categories=["policy_interpretation"],
        allow_retrieval=True,
        allow_claim_extraction=True,
        allow_final_answer=allow,
        requires_disclaimer=True,
        requires_human_review=not allow,
        requires_additional_information=missing,
        missing_information=["사고 경위"] if missing else [],
    )


def synthesis_request(*, models=None, risk_result=None, claim_value=None):
    claim_value = claim_value or claim(
        conditions=["보험기간 중 사고"],
        exceptions=["고의 사고"],
    )
    matrix_request = request(
        models or [model("model1"), model("model2")],
        claims=[claim_value],
        risk=risk_result,
    )
    return FinalAnswerSynthesisRequest(
        original_question="보험금 지급 조건은?",
        risk=risk_result or risk(),
        claims=[claim_value],
        judgments=matrix_request.judgment_batch,
        claim_matrix=build_claim_matrix(matrix_request),
    )


def raw_answer(**updates):
    value = {
        "status": "completed",
        "answer": (
            "제공된 Claim에서 지급 조건이 확인됩니다. "
            "실제 적용 여부는 구체적 사실에 따라 달라질 수 있습니다."
        ),
        "key_points": ["검증된 Claim에 지급 조건이 있습니다."],
        "applicable_conditions": ["보험기간 중 사고"],
        "important_exceptions": ["고의 사고"],
        "missing_information": [],
        "sources": [{"claim_id": "c1"}],
        "model_agreement": "high",
        "requires_disclaimer": False,
        "requires_human_review": False,
        "warnings": [],
    }
    value.update(updates)
    return value


def test_valid_answer_preserves_claim_sources_and_risk():
    result = validate_final_answer(raw_answer(), synthesis_request())
    assert result.sources[0].claim_id == "c1"
    assert result.requires_disclaimer
    assert result.model_agreement == "high"


def test_unknown_source_is_rejected_and_unverified_condition_is_discarded():
    with pytest.raises(FinalAnswerValidationError, match="source metadata"):
        validate_final_answer(
            raw_answer(sources=[{"claim_id": "c1", "article": "없는 조항"}]),
            synthesis_request(),
        )
    result = validate_final_answer(
        raw_answer(
            applicable_conditions=["모델이 만든 조건"],
            important_exceptions=["모델이 만든 예외"],
        ),
        synthesis_request(),
    )
    assert result.applicable_conditions == ["보험기간 중 사고"]
    assert result.important_exceptions == ["고의 사고"]


def test_definitive_insurance_statement_is_rejected():
    with pytest.raises(FinalAnswerValidationError, match="definitive"):
        validate_final_answer(
            raw_answer(answer="보험금은 무조건 지급됩니다."),
            synthesis_request(),
        )


def test_risk_disallowing_final_answer_forces_limited_and_review():
    limited_risk = risk(allow=False)
    result = validate_final_answer(
        raw_answer(),
        synthesis_request(risk_result=limited_risk),
    )
    assert result.status == "limited"
    assert result.requires_human_review
    assert "risk_gate_disallows_final_answer" in result.warnings


def test_one_model_forces_limited_answer():
    result = validate_final_answer(
        raw_answer(),
        synthesis_request(models=[model("model1")]),
    )
    assert result.status == "limited"
    assert result.model_agreement == "indeterminate"
    assert "insufficient_independent_models" in result.warnings


def test_low_agreement_discloses_conflict_and_forces_review():
    result = validate_final_answer(
        raw_answer(),
        synthesis_request(models=[
            model("model1"),
            model(
                "model2",
                assessment="contradicts",
                disposition="not_supported",
            ),
        ]),
    )
    assert result.model_agreement == "low"
    assert "모델 간 주요 불일치" in result.answer
    assert result.requires_human_review


def test_verified_sources_and_missing_information_are_added_deterministically():
    missing_risk = risk(missing=True)
    result = validate_final_answer(
        raw_answer(sources=[], missing_information=[]),
        synthesis_request(risk_result=missing_risk),
    )
    assert [source.claim_id for source in result.sources] == ["c1"]
    assert "사고 경위" in result.missing_information


@pytest.mark.asyncio
async def test_synthesizer_uses_verified_fallback_after_invalid_json(monkeypatch):
    calls = []

    class Responses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="not json")

    monkeypatch.setattr(
        synthesizer, "create_llm_client",
        lambda: SimpleNamespace(responses=Responses()),
    )
    monkeypatch.setattr(synthesizer, "final_answer_model", lambda: "model-final")
    result = await synthesizer.synthesize_final_answer(synthesis_request())
    assert result.status == "limited"
    assert len(calls) == 1
    assert result.key_points
    assert result.sources[0].claim_id == "c1"
    assert "final_synthesis_fallback" in result.warnings
