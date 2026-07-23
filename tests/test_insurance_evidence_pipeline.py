from copy import deepcopy

import pytest

from insurance_rag.schemas.insurance_claim import (
    ClaimSourceReference,
    InsuranceClaimExtractionResult,
    InsuranceEvidenceClaim,
)
from insurance_rag.schemas.insurance_retrieval import (
    InsuranceRetrievalResult,
    RetrievedInsuranceEvidence,
)
from insurance_rag.schemas.question import QuestionAnalysis, QuestionRequest
from insurance_rag.services import insurance_evidence_pipeline as pipeline


def analysis(*, compound=False, question="보험수익자는 누구인가요?"):
    sub_questions = [{
        "id": "q1", "question": question,
        "requested_action": "fact_lookup", "purpose": "근거 확인",
    }]
    if compound:
        sub_questions.append({
            "id": "q2", "question": "면책 사유는 무엇인가요?",
            "requested_action": "interpretation", "purpose": "면책 확인",
        })
    return QuestionAnalysis(
        original_question=question,
        normalized_question=question,
        main_intent="보험 약관 확인",
        is_compound=compound,
        sub_questions=sub_questions,
    )


def retrieval(*, evidence=True, sufficient=True, warning=None):
    items = [RetrievedInsuranceEvidence(
        rank=1, chunk_id="chunk-1", text="보험수익자는 지정된 사람이다."
    )] if evidence else []
    return InsuranceRetrievalResult(
        original_question="질문",
        search_query="질문",
        evidence=items,
        result_count=len(items),
        sufficient_evidence=sufficient,
        insufficiency_reason=None if sufficient else (
            "low_relevance" if items else "no_results"
        ),
        warnings=[warning] if warning else [],
    )


def claims(*, sufficient=True, prefix=""):
    items = [InsuranceEvidenceClaim(
        claim_id=f"{prefix}c1",
        claim_type="definition",
        statement="보험수익자는 지정된 사람이다.",
        source_references=[ClaimSourceReference(
            chunk_id="chunk-1", supporting_quote="보험수익자는 지정된 사람이다."
        )],
        relevance="direct", confidence="high",
    )] if sufficient else []
    return InsuranceClaimExtractionResult(
        original_question="질문",
        claims=items,
        claim_count=len(items),
        sufficient_evidence=sufficient,
        insufficiency_reason=None if sufficient else "no_relevant_claims",
    )


def install_analysis(monkeypatch, value):
    async def fake_analyze(request):
        return value

    monkeypatch.setattr(pipeline, "analyze_question", fake_analyze)


@pytest.mark.asyncio
async def test_simple_pipeline_completes_and_merges_warnings(monkeypatch):
    install_analysis(monkeypatch, analysis())

    async def fake_search(*args, **kwargs):
        return retrieval(warning="version_filter_relaxed")

    async def fake_extract(request):
        assert request.sub_question_id is None
        return claims()

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", fake_extract)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험수익자는 누구인가요?"),
        vector_store_id="vs-test",
    )
    assert result.status == "completed"
    assert result.retrieval and result.claims
    assert result.warnings == ["version_filter_relaxed"]


@pytest.mark.asyncio
async def test_risk_block_prevents_retrieval_and_extraction(monkeypatch):
    install_analysis(
        monkeypatch, analysis(question="보험사에 들키지 않고 고지의무를 피하려면?")
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr(pipeline, "search_insurance_documents", forbidden)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", forbidden)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험사에 들키지 않고 고지의무를 피하려면?"),
        vector_store_id="vs-test",
    )
    assert result.status == "blocked"
    assert result.stopped_at == "risk_gate"


@pytest.mark.asyncio
async def test_empty_retrieval_prevents_claim_extraction(monkeypatch):
    install_analysis(monkeypatch, analysis())

    async def fake_search(*args, **kwargs):
        return retrieval(evidence=False, sufficient=False)

    async def forbidden(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", forbidden)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험수익자는 누구인가요?"),
        vector_store_id="vs-test",
    )
    assert result.status == "insufficient_evidence"
    assert result.stopped_at == "retrieval"


@pytest.mark.asyncio
async def test_partial_evidence_is_passed_to_extraction(monkeypatch):
    install_analysis(monkeypatch, analysis())

    async def fake_search(*args, **kwargs):
        return retrieval(evidence=True, sufficient=False)

    async def fake_extract(request):
        assert request.retrieval_insufficiency_reason == "low_relevance"
        return claims(sufficient=False)

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", fake_extract)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험수익자는 누구인가요?"),
        vector_store_id="vs-test",
    )
    assert result.status == "insufficient_evidence"
    assert result.stopped_at == "claim_extraction"


@pytest.mark.asyncio
async def test_high_risk_searches_but_preserves_information_requirement(monkeypatch):
    value = analysis(question="내 경우 보험금이 무조건 나오나요?")
    install_analysis(monkeypatch, value)

    async def fake_search(*args, **kwargs):
        return retrieval()

    async def fake_extract(request):
        return claims()

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", fake_extract)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="내 경우 보험금이 무조건 나오나요?"),
        vector_store_id="vs-test",
    )
    assert result.status == "needs_information"
    assert not result.risk.allow_final_answer


@pytest.mark.asyncio
async def test_compound_questions_receive_prefixed_ids(monkeypatch):
    install_analysis(monkeypatch, analysis(compound=True))
    seen = []

    async def fake_search(*args, **kwargs):
        return retrieval()

    async def fake_extract(request):
        seen.append(request.sub_question_id)
        return claims(prefix=f"{request.sub_question_id}-")

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", fake_extract)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험수익자와 면책 사유를 알려줘"),
        vector_store_id="vs-test",
    )
    assert result.status == "completed"
    assert seen == ["q1", "q2"]
    assert [
        item.claims.claims[0].claim_id for item in result.sub_question_results
    ] == ["q1-c1", "q2-c1"]


@pytest.mark.asyncio
async def test_one_failed_sub_question_continues_and_returns_partial(monkeypatch):
    install_analysis(monkeypatch, analysis(compound=True))
    call_count = 0

    async def fake_search(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("upstream secret detail")
        return retrieval()

    async def fake_extract(request):
        return claims(prefix=f"{request.sub_question_id}-")

    monkeypatch.setattr(pipeline, "search_insurance_documents", fake_search)
    monkeypatch.setattr(pipeline, "extract_insurance_claims", fake_extract)
    result = await pipeline.build_insurance_evidence_claims(
        QuestionRequest(question="보험수익자와 면책 사유를 알려줘"),
        vector_store_id="vs-test",
    )
    assert call_count == 2
    assert result.status == "partial"
    assert [item.status for item in result.sub_question_results] == [
        "failed", "completed",
    ]
    assert "upstream secret detail" not in result.sub_question_results[0].reason


@pytest.mark.asyncio
async def test_analysis_failure_has_stage_specific_error(monkeypatch):
    async def failure(request):
        raise RuntimeError("private input")

    monkeypatch.setattr(pipeline, "analyze_question", failure)
    with pytest.raises(
        pipeline.InsuranceEvidencePipelineError, match="question analysis failed"
    ) as caught:
        await pipeline.build_insurance_evidence_claims(
            QuestionRequest(question="보험수익자는 누구인가요?"),
            vector_store_id="vs-test",
        )
    assert "private input" not in str(caught.value)
