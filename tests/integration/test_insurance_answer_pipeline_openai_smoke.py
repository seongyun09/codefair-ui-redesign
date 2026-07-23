import os

import pytest
from dotenv import load_dotenv

from insurance_rag.core.final_answer_synthesizer import synthesize_final_answer
from insurance_rag.schemas.claim_matrix import ClaimMatrixRequest
from insurance_rag.schemas.final_answer import FinalAnswerSynthesisRequest
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
    IndependentJudgmentRequest,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.question_risk import QuestionRiskGateResult
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from insurance_rag.services.independent_judgment_service import (
    generate_independent_judgments,
)

pytestmark = [pytest.mark.integration, pytest.mark.openai]
load_dotenv(override=False)


@pytest.mark.asyncio
async def test_real_openai_insurance_answer_pipeline_smoke():
    if os.getenv("RUN_OPENAI_SMOKE_TESTS") != "1":
        pytest.skip("set RUN_OPENAI_SMOKE_TESTS=1 to enable the paid smoke test")
    required = (
        "OPENAI_API_KEY",
        "INDEPENDENT_JUDGMENT_MODEL_A",
        "INDEPENDENT_JUDGMENT_MODEL_B",
        "FINAL_ANSWER_MODEL",
    )
    if any(not os.getenv(name) for name in required):
        pytest.skip("required OpenAI smoke configuration is missing")

    claim = InsuranceEvidenceClaim(
        claim_id="c1",
        claim_type="payment_condition",
        statement="보험기간 중 약정한 사고가 발생하면 보험금을 지급한다.",
        conditions=["보험기간 중 약정한 사고 발생"],
        exceptions=["면책 사유에 해당하는 경우"],
        legal_effect="약정 보험금 지급",
        source_references=[ClaimSourceReference(
            chunk_id="synthetic-final-answer-1",
            article="제3조",
            supporting_quote=(
                "보험기간 중 약정한 사고가 발생하고 면책 사유에 "
                "해당하지 않는 경우 보험금을 지급한다."
            ),
        )],
        relevance="direct",
        confidence="high",
    )
    risk = QuestionRiskGateResult(
        risk_level="medium",
        categories=["policy_interpretation"],
        allow_retrieval=True,
        allow_claim_extraction=True,
        allow_final_answer=True,
        requires_disclaimer=True,
        requires_human_review=False,
        requires_additional_information=False,
    )
    judgment_request = IndependentJudgmentRequest(
        original_question="일반적인 보험금 지급 조건은 무엇인가요?",
        normalized_question="일반적인 보험금 지급 조건",
        requested_action="interpretation",
        risk_level=risk.risk_level,
        risk_categories=risk.categories,
        allow_final_answer=risk.allow_final_answer,
        requires_disclaimer=risk.requires_disclaimer,
        requires_human_review=risk.requires_human_review,
        requires_additional_information=risk.requires_additional_information,
        claims=[claim],
        max_judgments=2,
    )
    batch = await generate_independent_judgments(
        judgment_request,
        model_configs=[
            IndependentJudgmentModelConfig(
                model_id=os.environ["INDEPENDENT_JUDGMENT_MODEL_A"]
            ),
            IndependentJudgmentModelConfig(
                model_id=os.environ["INDEPENDENT_JUDGMENT_MODEL_B"]
            ),
        ],
    )
    matrix = build_claim_matrix(ClaimMatrixRequest(
        original_question=judgment_request.original_question,
        claims=[claim],
        judgment_batch=batch,
        risk=risk,
    ))
    answer = await synthesize_final_answer(FinalAnswerSynthesisRequest(
        original_question=judgment_request.original_question,
        risk=risk,
        claims=[claim],
        judgments=batch,
        claim_matrix=matrix,
    ))
    assert batch.completed_model_count == 2
    assert answer.sources[0].claim_id == "c1"
    assert answer.requires_disclaimer
