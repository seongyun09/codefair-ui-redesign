import pytest

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
    IndependentJudgmentModelConfig,
)
from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceClaimExtractionResult,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.insurance_pipeline import (
    InsuranceEvidencePipelineResult,
    SubQuestionEvidenceResult,
)
from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalResult
from insurance_rag.schemas.question import QuestionAnalysis, QuestionRequest
from insurance_rag.schemas.question_risk import QuestionRiskGateResult
from insurance_rag.services import insurance_judgment_pipeline as pipeline
from test_independent_judgment_schema import judgment


def analysis(*, compound=False):
    questions = [{
        "id": "q1", "question": "지급 조건은?",
        "requested_action": "interpretation", "purpose": "지급 조건 확인",
    }]
    if compound:
        questions.append({
            "id": "q2", "question": "면책 조건은?",
            "requested_action": "interpretation", "purpose": "면책 확인",
        })
    return QuestionAnalysis(
        original_question="지급 조건과 면책 조건은?",
        normalized_question="지급 및 면책 조건",
        main_intent="약관 확인",
        is_compound=compound,
        sub_questions=questions,
    )


def risk(*, blocked=False):
    return QuestionRiskGateResult(
        risk_level="blocked" if blocked else "medium",
        categories=["fraud_or_evasion"] if blocked else ["policy_interpretation"],
        allow_retrieval=not blocked,
        allow_claim_extraction=not blocked,
        allow_final_answer=not blocked,
        requires_disclaimer=not blocked,
        requires_human_review=blocked,
        requires_additional_information=False,
        blocked_reason="fraud_or_evasion" if blocked else None,
    )


def claims(claim_id="c1", *, sufficient=True):
    values = [InsuranceEvidenceClaim(
        claim_id=claim_id,
        claim_type="payment_condition",
        statement="조건을 충족하면 보험금을 지급한다.",
        source_references=[ClaimSourceReference(
            chunk_id=f"chunk-{claim_id}",
            supporting_quote="조건을 충족하면 보험금을 지급한다.",
        )],
        relevance="direct", confidence="high",
    )] if sufficient else []
    return InsuranceClaimExtractionResult(
        original_question="질문",
        claims=values,
        claim_count=len(values),
        sufficient_evidence=sufficient,
        insufficiency_reason=None if sufficient else "no_relevant_claims",
    )


def retrieval():
    return InsuranceRetrievalResult(
        original_question="질문", search_query="질문",
        evidence=[], result_count=0, sufficient_evidence=True,
    )


def evidence(*, compound=False, blocked=False, sufficient=True):
    base_analysis = analysis(compound=compound)
    if compound:
        sub_results = [
            SubQuestionEvidenceResult(
                sub_question_id=item.id,
                question=item.question,
                retrieval=retrieval(),
                claims=claims(f"{item.id}-c1", sufficient=sufficient),
                status="completed" if sufficient else "insufficient_evidence",
            )
            for item in base_analysis.sub_questions
        ]
        return InsuranceEvidencePipelineResult(
            status="completed" if sufficient else "insufficient_evidence",
            analysis=base_analysis,
            risk=risk(blocked=blocked),
            sub_question_results=sub_results,
        )
    return InsuranceEvidencePipelineResult(
        status="blocked" if blocked else (
            "completed" if sufficient else "insufficient_evidence"
        ),
        analysis=base_analysis,
        risk=risk(blocked=blocked),
        claims=None if blocked else claims(sufficient=sufficient),
        stopped_at="risk_gate" if blocked else None,
        reason="fraud_or_evasion" if blocked else None,
    )


def batch(request, *, failed=0):
    items = [
        judgment(
            judgment_id=(
                f"{request.sub_question_id}-j1"
                if request.sub_question_id else "j1"
            ),
            model_id="model-a",
            sub_question_id=request.sub_question_id,
            claim_assessments=[{
                "claim_id": request.claims[0].claim_id,
                "assessment": "supports",
                "reasoning": "Claim이 조건을 지지한다.",
                "source_reference_ids": [
                    request.claims[0].source_references[0].chunk_id
                ],
                "confidence": "high",
            }],
            supporting_claim_ids=[request.claims[0].claim_id],
        )
    ]
    failures = [{
        "model_id": "model-b", "error_type": "RuntimeError",
        "message": "model judgment failed after one retry", "retry_count": 1,
    }] if failed else []
    return IndependentJudgmentBatchResult(
        original_question=request.original_question,
        sub_question_id=request.sub_question_id,
        judgments=items,
        requested_model_count=1 + failed,
        completed_model_count=1,
        failed_model_count=failed,
        sufficient_for_comparison=False,
        failures=failures,
        warnings=["insufficient_independent_models"],
    )


def configs():
    return [IndependentJudgmentModelConfig(model_id="model-a")]


@pytest.mark.asyncio
async def test_simple_evidence_runs_independent_judgment(monkeypatch):
    async def fake_evidence(*args, **kwargs):
        return evidence()

    async def fake_judgments(request, **kwargs):
        assert request.claims[0].claim_id == "c1"
        assert request.requires_disclaimer
        return batch(request)

    monkeypatch.setattr(pipeline, "build_insurance_evidence_claims", fake_evidence)
    monkeypatch.setattr(pipeline, "generate_independent_judgments", fake_judgments)
    result = await pipeline.build_independent_insurance_judgments(
        QuestionRequest(question="지급 조건은?"), model_configs=configs()
    )
    assert result.status == "completed"
    assert result.single_result.judgments[0].judgment_id == "j1"


@pytest.mark.asyncio
async def test_blocked_or_insufficient_evidence_makes_zero_model_calls(monkeypatch):
    calls = 0

    async def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError

    monkeypatch.setattr(pipeline, "generate_independent_judgments", forbidden)
    for value, expected in [
        (evidence(blocked=True), "blocked"),
        (evidence(sufficient=False), "insufficient_evidence"),
    ]:
        async def fake_evidence(*args, current=value, **kwargs):
            return current

        monkeypatch.setattr(
            pipeline, "build_insurance_evidence_claims", fake_evidence
        )
        result = await pipeline.build_independent_insurance_judgments(
            QuestionRequest(question="지급 조건은?"), model_configs=configs()
        )
        assert result.status == expected
    assert calls == 0


@pytest.mark.asyncio
async def test_compound_claims_are_never_mixed(monkeypatch):
    async def fake_evidence(*args, **kwargs):
        return evidence(compound=True)

    seen = []

    async def fake_judgments(request, **kwargs):
        seen.append((
            request.sub_question_id,
            [item.claim_id for item in request.claims],
        ))
        return batch(request)

    monkeypatch.setattr(pipeline, "build_insurance_evidence_claims", fake_evidence)
    monkeypatch.setattr(pipeline, "generate_independent_judgments", fake_judgments)
    result = await pipeline.build_independent_insurance_judgments(
        QuestionRequest(question="지급 조건과 면책 조건은?"),
        model_configs=configs(),
    )
    assert seen == [("q1", ["q1-c1"]), ("q2", ["q2-c1"])]
    assert [
        item.judgments[0].judgment_id for item in result.sub_question_results
    ] == ["q1-j1", "q2-j1"]


@pytest.mark.asyncio
async def test_model_partial_failure_is_preserved(monkeypatch):
    async def fake_evidence(*args, **kwargs):
        return evidence()

    async def fake_judgments(request, **kwargs):
        return batch(request, failed=1)

    monkeypatch.setattr(pipeline, "build_insurance_evidence_claims", fake_evidence)
    monkeypatch.setattr(pipeline, "generate_independent_judgments", fake_judgments)
    result = await pipeline.build_independent_insurance_judgments(
        QuestionRequest(question="지급 조건은?"),
        model_configs=[
            IndependentJudgmentModelConfig(model_id="model-a"),
            IndependentJudgmentModelConfig(model_id="model-b"),
        ],
    )
    assert result.status == "partial"
    assert result.single_result.completed_model_count == 1
    assert result.single_result.failed_model_count == 1
