import os

import pytest
from dotenv import load_dotenv

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
    IndependentJudgmentRequest,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceEvidenceClaim,
)
from insurance_rag.services.independent_judgment_service import (
    generate_independent_judgments,
)

pytestmark = [pytest.mark.integration, pytest.mark.openai]
load_dotenv(override=False)


@pytest.mark.asyncio
async def test_real_openai_independent_judgments_smoke():
    if os.getenv("RUN_OPENAI_SMOKE_TESTS") != "1":
        pytest.skip("set RUN_OPENAI_SMOKE_TESTS=1 to enable the paid smoke test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    model_ids = [
        os.getenv("INDEPENDENT_JUDGMENT_MODEL_A"),
        os.getenv("INDEPENDENT_JUDGMENT_MODEL_B"),
    ]
    if not all(model_ids):
        pytest.skip("two independent judgment models are not configured")

    claim = InsuranceEvidenceClaim(
        claim_id="c1",
        claim_type="benefit_eligibility",
        statement="보험기간 중 사망하면 사망보험금이 지급된다.",
        conditions=["보험기간 중 사망"],
        legal_effect="보험금 지급",
        source_references=[ClaimSourceReference(
            chunk_id="synthetic-chunk-1",
            article="제3조",
            supporting_quote=(
                "피보험자가 보험기간 중 사망한 경우 "
                "보험수익자에게 사망보험금을 지급한다."
            ),
        )],
        relevance="direct",
        confidence="high",
    )
    result = await generate_independent_judgments(
        IndependentJudgmentRequest(
            original_question="보험기간 중 사망하면 어떤 급부가 있나요?",
            normalized_question="보험기간 중 사망 시 급부",
            requested_action="interpretation",
            risk_level="medium",
            risk_categories=["policy_interpretation"],
            allow_final_answer=True,
            requires_disclaimer=True,
            requires_human_review=False,
            requires_additional_information=False,
            claims=[claim],
            max_judgments=2,
        ),
        model_configs=[
            IndependentJudgmentModelConfig(model_id=model_ids[0]),
            IndependentJudgmentModelConfig(model_id=model_ids[1]),
        ],
    )
    assert result.completed_model_count == 2
    assert result.failed_model_count == 0
    assert result.sufficient_for_comparison
