import pytest

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentPipelineResult,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services import insurance_claim_matrix_pipeline as pipeline
from test_claim_matrix_service import batch, model
from test_insurance_judgment_pipeline import evidence


@pytest.mark.asyncio
async def test_blocked_judgments_make_zero_matrix_calls(monkeypatch):
    result = IndependentJudgmentPipelineResult(
        status="blocked",
        evidence=evidence(blocked=True),
    )

    async def fake_judgments(*args, **kwargs):
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("matrix must not be built")

    monkeypatch.setattr(
        pipeline, "build_independent_insurance_judgments", fake_judgments
    )
    monkeypatch.setattr(pipeline, "build_claim_matrix", forbidden)
    output = await pipeline.build_insurance_claim_matrices(
        QuestionRequest(question="차단 질문")
    )
    assert output.judgments.status == "blocked"
    assert output.single_result is None
    assert output.sub_question_results == []


@pytest.mark.asyncio
async def test_zero_successful_judgments_make_zero_matrix_calls(monkeypatch):
    empty_batch = batch([], failures=[{
        "model_id": "model1",
        "error_type": "RuntimeError",
        "message": "failed",
        "retry_count": 1,
    }])
    result = IndependentJudgmentPipelineResult(
        status="partial",
        evidence=evidence(),
        single_result=empty_batch,
    )

    async def fake_judgments(*args, **kwargs):
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("matrix must not be built")

    monkeypatch.setattr(
        pipeline, "build_independent_insurance_judgments", fake_judgments
    )
    monkeypatch.setattr(pipeline, "build_claim_matrix", forbidden)
    output = await pipeline.build_insurance_claim_matrices(
        QuestionRequest(question="지급 조건은?")
    )
    assert output.single_result is None


@pytest.mark.asyncio
async def test_compound_pipeline_keeps_claim_matrices_separate(monkeypatch):
    evidence_result = evidence(compound=True)
    batches = []
    for sub in evidence_result.sub_question_results:
        claim_id = sub.claims.claims[0].claim_id
        batches.append(batch([
            model("model1", sub_question_id=sub.sub_question_id, claim_id=claim_id),
            model(
                "model2", sub_question_id=sub.sub_question_id, claim_id=claim_id,
                assessment="contradicts", disposition="not_supported",
            ),
        ], sub_question_id=sub.sub_question_id))
    judgment_result = IndependentJudgmentPipelineResult(
        status="completed",
        evidence=evidence_result,
        sub_question_results=batches,
    )

    async def fake_judgments(*args, **kwargs):
        return judgment_result

    monkeypatch.setattr(
        pipeline, "build_independent_insurance_judgments", fake_judgments
    )
    output = await pipeline.build_insurance_claim_matrices(
        QuestionRequest(question="지급 조건과 면책 조건은?")
    )
    assert [item.sub_question_id for item in output.sub_question_results] == [
        "q1", "q2"
    ]
    assert [
        item.rows[0].claim_id for item in output.sub_question_results
    ] == ["q1-c1", "q2-c1"]
    assert all(
        disagreement.disagreement_id.startswith(f"{matrix.sub_question_id}-d")
        for matrix in output.sub_question_results
        for disagreement in matrix.disagreements
    )
