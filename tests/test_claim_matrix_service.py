from __future__ import annotations

import pytest
from pydantic import ValidationError

from insurance_rag.schemas.claim_matrix import ClaimMatrixRequest
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.question_risk import QuestionRiskGateResult
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from test_independent_judgment_schema import judgment


def claim(claim_id="c1", *, conditions=None, exceptions=None, legal_effect=None):
    return InsuranceEvidenceClaim(
        claim_id=claim_id,
        claim_type="payment_condition",
        statement="조건을 충족하면 지급한다.",
        conditions=conditions or [],
        exceptions=exceptions or [],
        legal_effect=legal_effect,
        source_references=[ClaimSourceReference(
            chunk_id=f"chunk-{claim_id}", supporting_quote="지급한다."
        )],
        relevance="direct",
        confidence="high",
    )


def model(
    model_id,
    *,
    assessment="supports",
    conditions=None,
    exceptions=None,
    sources=None,
    confidence="high",
    disposition="supported",
    missing=None,
    review=False,
    claim_id="c1",
    sub_question_id=None,
):
    suffix = model_id[-1] if model_id[-1].isdigit() else str(ord(model_id[-1]) - 96)
    judgment_id = f"j{suffix}" if not sub_question_id else f"{sub_question_id}-j{suffix}"
    return judgment(
        judgment_id=judgment_id,
        model_id=model_id,
        sub_question_id=sub_question_id,
        disposition=disposition,
        claim_assessments=[{
            "claim_id": claim_id,
            "assessment": assessment,
            "reasoning": "판단 근거",
            "conditions_considered": conditions or [],
            "exceptions_considered": exceptions or [],
            "source_reference_ids": sources or ["chunk-c1"],
            "confidence": confidence,
        }],
        supporting_claim_ids=[claim_id] if assessment == "supports" else [],
        conflicting_claim_ids=[claim_id] if assessment == "contradicts" else [],
        missing_information=missing or [],
        requires_human_review=review,
        confidence=confidence,
    )


def batch(models, *, failures=None, sub_question_id=None):
    failures = failures or []
    return IndependentJudgmentBatchResult(
        original_question="질문",
        sub_question_id=sub_question_id,
        judgments=models,
        requested_model_count=len(models) + len(failures),
        completed_model_count=len(models),
        failed_model_count=len(failures),
        sufficient_for_comparison=len(models) >= 2,
        failures=failures,
    )


def request(models, **kwargs):
    return ClaimMatrixRequest(
        original_question="질문",
        claims=kwargs.pop("claims", [claim()]),
        judgment_batch=batch(
            models,
            failures=kwargs.pop("failures", None),
            sub_question_id=kwargs.get("sub_question_id"),
        ),
        **kwargs,
    )


def types(result):
    return {item.disagreement_type for item in result.disagreements}


def test_unanimous_matrix_has_high_agreement():
    result = build_claim_matrix(request([model("model1"), model("model2")]))
    assert result.rows[0].agreement == "unanimous"
    assert result.overall_agreement == "high"
    assert result.disagreements == []


def test_direct_conflict_and_legal_effect_are_high():
    result = build_claim_matrix(request(
        [model("model1"), model("model2", assessment="contradicts",
                                disposition="not_supported")],
        claims=[claim(legal_effect="보험금 지급 의무")],
    ))
    assert {"claim_assessment_conflict", "legal_effect_difference",
            "disposition_conflict"} <= types(result)
    assert result.rows[0].agreement == "conflicted"
    assert result.overall_agreement == "low"
    assert result.requires_human_review


def test_conditions_exceptions_evidence_and_confidence_are_compared():
    result = build_claim_matrix(request([
        model("model1", conditions=["  가입 기간. "], exceptions=["고의"],
              sources=["s1"], confidence="high"),
        model("model2", conditions=["가입 기간"], exceptions=["중과실"],
              sources=["s2"], confidence="low"),
    ]))
    assert "condition_interpretation_difference" not in types(result)
    assert {"exception_interpretation_difference", "evidence_usage_difference",
            "confidence_gap"} <= types(result)


def test_omission_and_model_failure_use_distinct_cells():
    result = build_claim_matrix(request(
        [model("model1"), model("model2", claim_id="c2")],
        claims=[claim(), claim("c2")],
        failures=[{
            "model_id": "model3", "error_type": "RuntimeError",
            "message": "failed", "retry_count": 1,
        }],
    ))
    assert [cell.assessment for cell in result.rows[0].cells] == [
        "supports", "not_assessed", "model_failed"
    ]
    assert "claim_omission" in types(result)


def test_disposition_missing_information_and_review_differences():
    result = build_claim_matrix(request([
        model("model1", disposition="partially_supported", missing=["나이"]),
        model("model2", disposition="insufficient_information", review=True),
    ]))
    assert {"disposition_conflict", "missing_information_difference",
            "human_review_difference"} <= types(result)
    assert result.requires_human_review


def test_risk_constraint_violation_is_critical():
    risk = QuestionRiskGateResult(
        risk_level="high",
        categories=["individual_claim_determination"],
        allow_retrieval=True,
        allow_claim_extraction=True,
        allow_final_answer=False,
        requires_disclaimer=True,
        requires_human_review=True,
        requires_additional_information=False,
    )
    result = build_claim_matrix(request(
        [model("model1"), model("model2")], risk=risk
    ))
    violations = [
        item for item in result.disagreements
        if item.disagreement_type == "risk_constraint_violation"
    ]
    assert len(violations) == 2
    assert all(item.severity == "critical" for item in violations)


def test_one_model_is_insufficient_but_matrix_is_built():
    result = build_claim_matrix(request([model("model1")]))
    assert result.comparison_status == "insufficient_models"
    assert result.overall_agreement == "indeterminate"
    assert result.rows[0].agreement == "insufficient"


def test_three_models_with_two_agree_is_mixed():
    result = build_claim_matrix(request([
        model("model1"), model("model2"),
        model("model3", assessment="conditional"),
    ]))
    assert result.rows[0].agreement == "mixed"


def test_compound_disagreement_ids_are_prefixed():
    result = build_claim_matrix(request(
        [
            model("model1", sub_question_id="q2", claim_id="q2-c1"),
            model("model2", assessment="contradicts",
                  disposition="not_supported", sub_question_id="q2",
                  claim_id="q2-c1"),
        ],
        claims=[claim("q2-c1")],
        sub_question_id="q2",
    ))
    assert all(item.disagreement_id.startswith("q2-d")
               for item in result.disagreements)


def test_unknown_claim_and_misaligned_subquestion_are_rejected():
    with pytest.raises(ValidationError, match="unknown claim_id"):
        request([model("model1", claim_id="missing")])
    with pytest.raises(ValidationError, match="sub_question_id"):
        ClaimMatrixRequest(
            original_question="질문",
            sub_question_id="q1",
            claims=[claim()],
            judgment_batch=batch([model("model1")]),
        )


def test_duplicate_model_ids_are_rejected():
    with pytest.raises(ValidationError, match="model_id"):
        request([model("model1"), model("model1")])
