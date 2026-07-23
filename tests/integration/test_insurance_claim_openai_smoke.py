import os

import pytest
from dotenv import load_dotenv

from insurance_rag.core.insurance_claim_extractor import extract_insurance_claims
from insurance_rag.schemas.insurance_claim import InsuranceClaimExtractionRequest
from insurance_rag.schemas.insurance_retrieval import RetrievedInsuranceEvidence

pytestmark = [pytest.mark.integration, pytest.mark.openai]
load_dotenv(override=False)


@pytest.mark.asyncio
async def test_real_openai_claim_extraction_smoke():
    if os.getenv("RUN_OPENAI_SMOKE_TESTS") != "1":
        pytest.skip("set RUN_OPENAI_SMOKE_TESTS=1 to enable the paid smoke test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    if not (
        os.getenv("INSURANCE_CLAIM_EXTRACTOR_MODEL")
        or os.getenv("OPENAI_MODEL")
    ):
        pytest.skip("claim extraction model is not configured")

    evidence_text = (
        "제3조: 피보험자가 보험기간 중 사망한 경우 "
        "보험수익자에게 사망보험금을 지급한다."
    )
    result = await extract_insurance_claims(InsuranceClaimExtractionRequest(
        original_question="보험기간 중 사망하면 어떤 급부가 있나요?",
        normalized_question="보험기간 중 사망 시 급부",
        evidence=[RetrievedInsuranceEvidence(
            rank=1,
            document_id="synthetic-terms",
            chunk_id="synthetic-chunk-1",
            article="제3조",
            page_start=1,
            page_end=1,
            text=evidence_text,
            source_file="synthetic-terms.txt",
        )],
        max_claims=3,
    ))

    assert result.claims
    assert result.claim_count == len(result.claims)
    assert all(claim.source_references for claim in result.claims)
    assert any(
        claim.claim_type == "benefit_eligibility" for claim in result.claims
    )
