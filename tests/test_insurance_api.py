import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from insurance_rag.api import app
from insurance_rag.api_insurance import (
    _RESPONSE_CACHE,
    get_analysis_timeout_seconds,
    get_insurance_pipeline,
)
from insurance_rag.core.insurance_retriever import InsuranceRetrievalError
from insurance_rag.schemas.final_answer import (
    FinalAnswerSource,
    InsuranceAnswerPipelineResult,
    SubQuestionFinalAnswer,
)
from test_insurance_answer_pipeline import answer, compared_simple
from test_insurance_judgment_pipeline import evidence

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    app.dependency_overrides.clear()
    _RESPONSE_CACHE.clear()
    yield
    app.dependency_overrides.clear()
    _RESPONSE_CACHE.clear()


def completed_result(*, one_model=False):
    compared = compared_simple(one_model=one_model)
    final = answer(
        "limited" if one_model else "completed",
        "indeterminate" if one_model else "high",
    )
    final.sources = [FinalAnswerSource(claim_id="c1")]
    return InsuranceAnswerPipelineResult(
        status="limited" if one_model else "completed",
        evidence_pipeline=compared.judgments.evidence,
        independent_judgments=compared.judgments.single_result,
        claim_matrix=compared.single_result,
        final_answer=final,
        warnings=(
            ["insufficient_independent_models"] if one_model else []
        ),
    )


def use_result(result):
    async def fake_pipeline(*args, **kwargs):
        return result

    app.dependency_overrides[get_insurance_pipeline] = lambda: fake_pipeline


def test_anonymous_completed_response_is_cached(monkeypatch):
    calls = 0

    async def fake_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        return completed_result()

    monkeypatch.setenv("INSURANCE_RESPONSE_CACHE_TTL_SECONDS", "900")
    app.dependency_overrides[get_insurance_pipeline] = lambda: fake_pipeline
    first = client.post("/insurance/analyze", json={"question": "캐시 질문"})
    second = client.post("/insurance/analyze", json={"question": "캐시 질문"})
    assert first.status_code == second.status_code == 200
    assert calls == 1
    assert first.json()["request_id"] != second.json()["request_id"]
    assert second.json()["processing_time_ms"] == 0


def test_personal_context_bypasses_response_cache(monkeypatch):
    calls = 0

    async def fake_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        return completed_result()

    monkeypatch.setenv("INSURANCE_RESPONSE_CACHE_TTL_SECONDS", "900")
    app.dependency_overrides[get_insurance_pipeline] = lambda: fake_pipeline
    payload = {
        "question": "개인 맥락 질문",
        "user_context": {"contract": "private"},
    }
    assert client.post("/insurance/analyze", json=payload).status_code == 200
    assert client.post("/insurance/analyze", json=payload).status_code == 200
    assert calls == 2


@pytest.mark.parametrize("question", ["정상 질문", "가" * 4000])
def test_valid_request_returns_completed_response(question):
    use_result(completed_result())
    response = client.post(
        "/insurance/analyze",
        json={"question": question, "user_context": {"contract_active": True}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["request_id"]
    assert body["answer"]
    assert body["sources"][0]["claim_id"] == "c1"
    assert isinstance(body["processing_time_ms"], int)
    assert "evidence_pipeline" not in body
    assert "independent_judgments" not in body


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": " "},
    {"question": "한"},
    {"question": "가" * 4001},
])
def test_invalid_question_returns_422(payload):
    response = client.post("/insurance/analyze", json=payload)
    assert response.status_code == 422


def test_missing_context_and_malformed_json():
    use_result(completed_result())
    assert client.post(
        "/insurance/analyze", json={"question": "정상 질문"}
    ).status_code == 200
    response = client.post(
        "/insurance/analyze",
        content="{broken",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_invalid_vector_store_id_returns_safe_400():
    response = client.post(
        "/insurance/analyze",
        json={"question": "정상 질문", "vector_store_id": "invalid"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_vector_store_id"


def test_blocked_and_insufficient_results_remain_http_200():
    blocked_evidence = evidence(blocked=True)
    use_result(InsuranceAnswerPipelineResult(
        status="blocked",
        evidence_pipeline=blocked_evidence,
        stopped_at="risk_gate",
        reason="fraud_or_evasion",
    ))
    blocked = client.post(
        "/insurance/analyze", json={"question": "차단 질문"}
    )
    assert blocked.status_code == 200
    assert blocked.json()["status"] == "blocked"
    assert blocked.json()["answer"] is None
    assert blocked.json()["stopped_at"] == "risk_gate"
    assert blocked.json()["requires_human_review"]

    use_result(InsuranceAnswerPipelineResult(
        status="insufficient_evidence",
        evidence_pipeline=evidence(sufficient=False),
        stopped_at="claim_extraction",
        reason="insufficient_evidence",
        warnings=["independent_judgment_skipped_no_claims"],
    ))
    insufficient = client.post(
        "/insurance/analyze", json={"question": "근거 부족 질문"}
    )
    assert insufficient.status_code == 200
    assert insufficient.json()["status"] == "insufficient_evidence"
    assert insufficient.json()["answer"] is None


def test_one_model_partial_result_is_limited():
    use_result(completed_result(one_model=True))
    response = client.post(
        "/insurance/analyze", json={"question": "정상 질문"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "limited"
    assert response.json()["model_agreement"] == "indeterminate"
    assert response.json()["requires_human_review"]


def test_compound_sub_answers_are_kept_separate():
    evidence_result = evidence(compound=True)
    compared = compared_simple()
    sub_results = []
    for index, part in enumerate(evidence_result.sub_question_results, 1):
        batch_value = compared.judgments.single_result.model_copy(update={
            "sub_question_id": part.sub_question_id,
        })
        matrix_value = compared.single_result.model_copy(update={
            "sub_question_id": part.sub_question_id,
            "rows": [
                compared.single_result.rows[0].model_copy(update={
                    "claim_id": part.claims.claims[0].claim_id
                })
            ],
        })
        sub_results.append(SubQuestionFinalAnswer(
            sub_question_id=part.sub_question_id,
            question=part.question,
            independent_judgments=batch_value,
            claim_matrix=matrix_value,
            final_answer=answer(),
        ))
    use_result(InsuranceAnswerPipelineResult(
        status="completed",
        evidence_pipeline=evidence_result,
        sub_question_results=sub_results,
    ))
    response = client.post(
        "/insurance/analyze", json={"question": "지급 조건과 면책은?"}
    )
    assert response.status_code == 200
    assert [
        item["sub_question_id"] for item in response.json()["sub_answers"]
    ] == ["q1", "q2"]
    assert response.json()["answer"] is None


def test_timeout_external_and_internal_errors_are_safe():
    async def slow(*args, **kwargs):
        await asyncio.sleep(0.1)

    app.dependency_overrides[get_insurance_pipeline] = lambda: slow
    app.dependency_overrides[get_analysis_timeout_seconds] = lambda: 0.001
    timeout = client.post(
        "/insurance/analyze", json={"question": "시간 초과 질문"}
    )
    assert timeout.status_code == 504
    assert timeout.json()["error_code"] == "analysis_timeout"
    assert timeout.json()["retryable"]

    async def external(*args, **kwargs):
        raise InsuranceRetrievalError("secret upstream detail")

    app.dependency_overrides[get_insurance_pipeline] = lambda: external
    external_response = client.post(
        "/insurance/analyze", json={"question": "외부 오류 질문"}
    )
    assert external_response.status_code == 503
    assert external_response.json()["error_code"] == "external_service_unavailable"
    assert "secret upstream detail" not in external_response.text

    async def internal(*args, **kwargs):
        raise ValueError("private stack detail")

    app.dependency_overrides[get_insurance_pipeline] = lambda: internal
    internal_response = client.post(
        "/insurance/analyze", json={"question": "내부 오류 질문"}
    )
    assert internal_response.status_code == 500
    assert internal_response.json()["error_code"] == "internal_error"
    assert "private stack detail" not in internal_response.text
    assert "traceback" not in internal_response.text.casefold()


def test_configuration_error_returns_503():
    async def unconfigured(*args, **kwargs):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    app.dependency_overrides[get_insurance_pipeline] = lambda: unconfigured
    response = client.post(
        "/insurance/analyze", json={"question": "설정 확인 질문"}
    )
    assert response.status_code == 503
    assert response.json()["error_code"] == "service_not_configured"
    assert not response.json()["retryable"]


def test_response_does_not_expose_internal_or_sensitive_fields():
    use_result(completed_result())
    response = client.post(
        "/insurance/analyze",
        json={
            "question": "정상 질문",
            "user_context": {
                "resident_registration_number": "secret-personal-value",
                "policy_number": "secret-policy-value",
            },
            "include_debug": True,
        },
    )
    serialized = json.dumps(response.json()).casefold()
    for forbidden in (
        "api_key", "system_prompt", "raw_response", "stack_trace", "traceback",
        "full_document_text", "full_claim_reasoning", "secret-personal-value",
        "secret-policy-value",
    ):
        assert forbidden not in serialized


def test_openapi_and_readiness_are_registered():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    operation = response.json()["paths"]["/insurance/analyze"]["post"]
    assert "requestBody" in operation
    assert "200" in operation["responses"]
    assert "503" in operation["responses"]
    readiness = client.get("/insurance/readiness")
    assert readiness.status_code == 200
    assert "openai_configured" in readiness.json()
    assert "judgment_models_configured" in readiness.json()
