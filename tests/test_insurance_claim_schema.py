import pytest
from pydantic import ValidationError

from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceClaimExtractionRequest,
    InsuranceClaimExtractionResult,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.insurance_retrieval import RetrievedInsuranceEvidence


def evidence(text: str = "보험금을 지급한다.") -> RetrievedInsuranceEvidence:
    return RetrievedInsuranceEvidence(rank=1, chunk_id="chunk-1", text=text)


def test_request_rejects_blank_questions_blank_evidence_and_invalid_max_claims():
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionRequest(
            original_question=" ", normalized_question="질문", evidence=[evidence()]
        )
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionRequest(
            original_question="질문", normalized_question="질문",
            sub_question_id="q6", evidence=[evidence()],
        )
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionRequest(
            original_question="질문", normalized_question="질문", evidence=[evidence(" ")]
        )
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionRequest(
            original_question="질문", normalized_question="질문",
            evidence=[evidence()], max_claims=31,
        )


def test_claim_schema_cleans_lists_and_requires_source():
    claim = InsuranceEvidenceClaim(
        claim_id="c1", claim_type="benefit_eligibility",
        statement="보험금을 지급한다.", conditions=[" 사망한 경우 ", "", "사망한 경우"],
        exceptions=[], source_references=[
            ClaimSourceReference(chunk_id="chunk-1", supporting_quote="보험금을 지급한다.")
        ], relevance="direct", confidence="high",
    )
    assert claim.conditions == ["사망한 경우"]
    source = ClaimSourceReference(
        chunk_id=" ", article="", supporting_quote="보험금을 지급한다."
    )
    assert source.chunk_id is None and source.article is None
    with pytest.raises(ValidationError):
        InsuranceEvidenceClaim(
            claim_id="c1", claim_type="unknown", statement="주장",
            source_references=[], relevance="direct", confidence="high",
        )


def test_result_requires_count_ids_and_insufficiency_consistency():
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionResult(
            original_question="질문", claims=[], claim_count=1,
            sufficient_evidence=False, insufficiency_reason="no_evidence",
        )
    with pytest.raises(ValidationError):
        InsuranceClaimExtractionResult(
            original_question="질문", claims=[], claim_count=0,
            sufficient_evidence=False,
        )
