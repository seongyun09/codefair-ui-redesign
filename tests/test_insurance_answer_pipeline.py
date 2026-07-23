import pytest

from insurance_rag.schemas.claim_matrix import (
    ClaimMatrixRequest,
    InsuranceClaimMatrixPipelineResult,
)
from insurance_rag.schemas.final_answer import FinalInsuranceAnswer
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentPipelineResult,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services import insurance_answer_pipeline as pipeline
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from test_claim_matrix_service import batch, model
from test_insurance_judgment_pipeline import evidence


def answer(status="completed", agreement="high"):
    return FinalInsuranceAnswer(
        status=status,
        answer="검증된 Claim 범위에서 조건이 확인됩니다.",
        model_agreement=agreement,
        requires_disclaimer=True,
        requires_human_review=status != "completed",
    )


def compared_simple(*, one_model=False):
    evidence_result = evidence()
    models = [model("model1")]
    if not one_model:
        models.append(model("model2"))
    judgment_batch = batch(models)
    judgments = IndependentJudgmentPipelineResult(
        status="completed",
        evidence=evidence_result,
        single_result=judgment_batch,
    )
    matrix = build_claim_matrix(ClaimMatrixRequest(
        original_question=evidence_result.analysis.original_question,
        claims=evidence_result.claims.claims,
        judgment_batch=judgment_batch,
        risk=evidence_result.risk,
    ))
    return InsuranceClaimMatrixPipelineResult(
        judgments=judgments,
        single_result=matrix,
    )


@pytest.mark.asyncio
async def test_normal_simple_question_returns_final_answer(monkeypatch):
    async def fake_compared(*args, **kwargs):
        return compared_simple()

    async def fake_synthesis(request):
        assert request.claims[0].claim_id == "c1"
        return answer()

    monkeypatch.setattr(pipeline, "build_insurance_claim_matrices", fake_compared)
    monkeypatch.setattr(pipeline, "synthesize_final_answer", fake_synthesis)
    result = await pipeline.build_final_insurance_answer(
        QuestionRequest(question="지급 조건은?")
    )
    assert result.status == "completed"
    assert result.final_answer.status == "completed"
    assert result.claim_matrix.rows[0].claim_id == "c1"


@pytest.mark.asyncio
async def test_blocked_question_makes_zero_synthesis_calls(monkeypatch):
    blocked_evidence = evidence(blocked=True)
    compared = InsuranceClaimMatrixPipelineResult(
        judgments=IndependentJudgmentPipelineResult(
            status="blocked", evidence=blocked_evidence
        )
    )

    async def fake_compared(*args, **kwargs):
        return compared

    async def forbidden(*args, **kwargs):
        raise AssertionError("synthesis must not run")

    monkeypatch.setattr(pipeline, "build_insurance_claim_matrices", fake_compared)
    monkeypatch.setattr(pipeline, "synthesize_final_answer", forbidden)
    result = await pipeline.build_final_insurance_answer(
        QuestionRequest(question="차단 질문")
    )
    assert result.status == "blocked"
    assert result.stopped_at == "risk_gate"
    assert result.final_answer is None


@pytest.mark.asyncio
async def test_single_model_answer_is_limited(monkeypatch):
    async def fake_compared(*args, **kwargs):
        return compared_simple(one_model=True)

    async def fake_synthesis(request):
        assert request.judgments.completed_model_count == 1
        return answer("limited", "indeterminate")

    monkeypatch.setattr(pipeline, "build_insurance_claim_matrices", fake_compared)
    monkeypatch.setattr(pipeline, "synthesize_final_answer", fake_synthesis)
    result = await pipeline.build_final_insurance_answer(
        QuestionRequest(question="지급 조건은?")
    )
    assert result.status == "limited"


@pytest.mark.asyncio
async def test_insufficient_evidence_makes_zero_synthesis_calls(monkeypatch):
    insufficient = evidence(sufficient=False)
    compared = InsuranceClaimMatrixPipelineResult(
        judgments=IndependentJudgmentPipelineResult(
            status="insufficient_evidence", evidence=insufficient
        )
    )

    async def fake_compared(*args, **kwargs):
        return compared

    async def forbidden(*args, **kwargs):
        raise AssertionError("synthesis must not run")

    monkeypatch.setattr(pipeline, "build_insurance_claim_matrices", fake_compared)
    monkeypatch.setattr(pipeline, "synthesize_final_answer", forbidden)
    result = await pipeline.build_final_insurance_answer(
        QuestionRequest(question="근거 없는 질문")
    )
    assert result.status == "insufficient_evidence"
    assert result.stopped_at == "claim_extraction"


@pytest.mark.asyncio
async def test_compound_answers_are_synthesized_separately(monkeypatch):
    evidence_result = evidence(compound=True)
    batches = []
    matrices = []
    for part in evidence_result.sub_question_results:
        claim_id = part.claims.claims[0].claim_id
        current_batch = batch([
            model("model1", sub_question_id=part.sub_question_id, claim_id=claim_id),
            model("model2", sub_question_id=part.sub_question_id, claim_id=claim_id),
        ], sub_question_id=part.sub_question_id)
        batches.append(current_batch)
        matrices.append(build_claim_matrix(ClaimMatrixRequest(
            original_question=evidence_result.analysis.original_question,
            sub_question_id=part.sub_question_id,
            claims=part.claims.claims,
            judgment_batch=current_batch,
            risk=evidence_result.risk,
        )))
    compared = InsuranceClaimMatrixPipelineResult(
        judgments=IndependentJudgmentPipelineResult(
            status="completed",
            evidence=evidence_result,
            sub_question_results=batches,
        ),
        sub_question_results=matrices,
    )
    seen = []

    async def fake_compared(*args, **kwargs):
        return compared

    async def fake_synthesis(request):
        seen.append((request.sub_question_id, request.claims[0].claim_id))
        return answer()

    monkeypatch.setattr(pipeline, "build_insurance_claim_matrices", fake_compared)
    monkeypatch.setattr(pipeline, "synthesize_final_answer", fake_synthesis)
    result = await pipeline.build_final_insurance_answer(
        QuestionRequest(question="지급 조건과 면책은?")
    )
    assert seen == [("q1", "q1-c1"), ("q2", "q2-c1")]
    assert result.status == "completed"
