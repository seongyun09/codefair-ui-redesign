import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from insurance_rag.api import app
from insurance_rag.api_insurance import get_insurance_pipeline
from insurance_rag.core.final_answer_synthesizer import synthesize_final_answer
from insurance_rag.schemas.claim_matrix import ClaimMatrixRequest
from insurance_rag.schemas.final_answer import (
    FinalAnswerSynthesisRequest,
    InsuranceAnswerPipelineResult,
)
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
    IndependentJudgmentRequest,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceClaimExtractionResult,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.insurance_pipeline import InsuranceEvidencePipelineResult
from insurance_rag.schemas.question import QuestionAnalysis
from insurance_rag.schemas.question_risk import QuestionRiskGateResult
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from insurance_rag.services.independent_judgment_service import (
    generate_independent_judgments,
)

pytestmark = [pytest.mark.integration, pytest.mark.openai]
load_dotenv(override=False)


@pytest.mark.asyncio
async def test_real_openai_insurance_analysis_http_smoke():
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
        source_references=[ClaimSourceReference(
            chunk_id="synthetic-api-smoke-1",
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
    analysis = QuestionAnalysis(
        original_question="일반적인 보험금 지급 조건은 무엇인가요?",
        normalized_question="일반적인 보험금 지급 조건",
        main_intent="보험금 지급 조건 확인",
        is_compound=False,
        sub_questions=[{
            "id": "q1",
            "question": "일반적인 보험금 지급 조건은 무엇인가요?",
            "requested_action": "interpretation",
            "purpose": "지급 조건 확인",
        }],
    )
    claims = InsuranceClaimExtractionResult(
        original_question=analysis.original_question,
        claims=[claim],
        claim_count=1,
        sufficient_evidence=True,
    )
    evidence = InsuranceEvidencePipelineResult(
        status="completed",
        analysis=analysis,
        risk=risk,
        claims=claims,
    )

    async def synthetic_pipeline(question_request, **kwargs):
        request = IndependentJudgmentRequest(
            original_question=question_request.question,
            normalized_question=analysis.normalized_question,
            requested_action="interpretation",
            risk_level=risk.risk_level,
            risk_categories=risk.categories,
            allow_final_answer=risk.allow_final_answer,
            requires_disclaimer=risk.requires_disclaimer,
            requires_human_review=risk.requires_human_review,
            requires_additional_information=False,
            claims=[claim],
            max_judgments=2,
        )
        batch = await generate_independent_judgments(
            request,
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
            original_question=request.original_question,
            claims=[claim],
            judgment_batch=batch,
            risk=risk,
        ))
        final = await synthesize_final_answer(FinalAnswerSynthesisRequest(
            original_question=request.original_question,
            risk=risk,
            claims=[claim],
            judgments=batch,
            claim_matrix=matrix,
        ))
        return InsuranceAnswerPipelineResult(
            status="completed" if final.status == "completed" else "limited",
            evidence_pipeline=evidence,
            independent_judgments=batch,
            claim_matrix=matrix,
            final_answer=final,
        )

    app.dependency_overrides[get_insurance_pipeline] = lambda: synthetic_pipeline
    try:
        response = TestClient(app).post(
            "/insurance/analyze",
            json={"question": analysis.original_question},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"completed", "limited"}
    assert body["answer"]
    assert body["sources"][0]["claim_id"] == "c1"
    assert body["model_agreement"] in {
        "high", "moderate", "low", "indeterminate"
    }
    serialized = response.text.casefold()
    assert "api_key" not in serialized
    assert "raw_response" not in serialized
