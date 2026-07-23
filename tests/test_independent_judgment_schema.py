import pytest
from pydantic import ValidationError

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
    IndependentJudgmentRequest,
    IndependentModelJudgment,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceEvidenceClaim,
)


def claim(claim_id="c1"):
    return InsuranceEvidenceClaim(
        claim_id=claim_id,
        claim_type="benefit_eligibility",
        statement="보험기간 중 사망하면 사망보험금이 지급된다.",
        source_references=[ClaimSourceReference(
            chunk_id="chunk-1",
            supporting_quote="보험기간 중 사망하면 사망보험금을 지급한다.",
        )],
        relevance="direct",
        confidence="high",
    )


def request(**changes):
    values = {
        "original_question": "사망 시 어떤 급부가 있나요?",
        "normalized_question": "사망 시 급부",
        "risk_level": "medium",
        "risk_categories": ["policy_interpretation"],
        "allow_final_answer": True,
        "requires_disclaimer": True,
        "requires_human_review": False,
        "requires_additional_information": False,
        "claims": [claim()],
    }
    values.update(changes)
    return IndependentJudgmentRequest(**values)


def judgment(**changes):
    values = {
        "judgment_id": "j1",
        "model_id": "model-a",
        "disposition": "supported",
        "conclusion": "제공된 Claim은 일반적인 지급 조건을 지지한다.",
        "claim_assessments": [{
            "claim_id": "c1",
            "assessment": "supports",
            "reasoning": "Claim에 지급 조건이 명시되어 있다.",
            "source_reference_ids": ["chunk-1"],
            "confidence": "high",
        }],
        "supporting_claim_ids": ["c1"],
        "requires_disclaimer": True,
        "requires_human_review": False,
        "confidence": "high",
    }
    values.update(changes)
    return IndependentModelJudgment(**values)


def test_request_requires_claims_unique_ids_and_valid_limits():
    with pytest.raises(ValidationError):
        request(claims=[])
    with pytest.raises(ValidationError, match="unique"):
        request(claims=[claim(), claim()])
    with pytest.raises(ValidationError):
        request(sub_question_id="q6")
    with pytest.raises(ValidationError):
        request(max_judgments=6)


def test_judgment_id_and_human_review_consistency():
    with pytest.raises(ValidationError):
        judgment(judgment_id="model-j1")
    with pytest.raises(ValidationError, match="requires human review"):
        judgment(
            disposition="requires_human_review",
            requires_human_review=False,
        )


def test_batch_counts_and_comparison_threshold():
    item = judgment()
    with pytest.raises(ValidationError, match="completed count"):
        IndependentJudgmentBatchResult(
            original_question="질문",
            judgments=[item],
            requested_model_count=1,
            completed_model_count=0,
            failed_model_count=1,
            sufficient_for_comparison=False,
            failures=[{
                "model_id": "model-b", "error_type": "Error",
                "message": "failed", "retry_count": 1,
            }],
        )
    batch = IndependentJudgmentBatchResult(
        original_question="질문",
        judgments=[item],
        requested_model_count=1,
        completed_model_count=1,
        failed_model_count=0,
        sufficient_for_comparison=False,
    )
    assert not batch.sufficient_for_comparison
