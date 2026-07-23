from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from openai import APIError

from insurance_rag.core.independent_judgment_validator import (
    IndependentJudgmentConfigurationError,
)
from insurance_rag.core.insurance_retriever import InsuranceRetrievalError
from insurance_rag.schemas.insurance_api import (
    InsuranceAnalysisErrorResponse,
    InsuranceAnalysisRequest,
    InsuranceAnalysisResponse,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services.insurance_answer_pipeline import (
    build_final_insurance_answer,
)
from insurance_rag.services.insurance_api_mapper import (
    to_insurance_analysis_response,
)
from insurance_rag.services.insurance_evidence_pipeline import (
    InsuranceEvidencePipelineError,
)

logger = logging.getLogger(__name__)
router = APIRouter()
_VECTOR_STORE_ID = re.compile(r"^vs_[A-Za-z0-9_-]+$")
_RESPONSE_CACHE: OrderedDict[str, tuple[float, InsuranceAnalysisResponse]] = (
    OrderedDict()
)
_RESPONSE_CACHE_MAX_ITEMS = 128


def get_insurance_pipeline() -> Callable:
    return build_final_insurance_answer


def get_analysis_timeout_seconds() -> float:
    load_dotenv(override=True)
    try:
        value = float(os.getenv("INSURANCE_ANALYSIS_TIMEOUT_SECONDS", "90"))
    except ValueError:
        return 90.0
    return value if 0 < value <= 600 else 90.0


def _cache_ttl_seconds() -> float:
    try:
        value = float(os.getenv("INSURANCE_RESPONSE_CACHE_TTL_SECONDS", "900"))
    except ValueError:
        return 900.0
    return value if 0 <= value <= 3600 else 900.0


def _cache_key(
    request: InsuranceAnalysisRequest, *, pipeline: Callable
) -> str | None:
    if request.user_context or request.include_debug:
        return None
    return json.dumps(
        {
            "question": request.question,
            "vector_store_id": request.vector_store_id,
            "pipeline": id(pipeline),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _cached_response(key: str | None) -> InsuranceAnalysisResponse | None:
    if key is None or _cache_ttl_seconds() <= 0:
        return None
    cached = _RESPONSE_CACHE.get(key)
    if cached is None:
        return None
    expires_at, response = cached
    if expires_at <= time.monotonic():
        _RESPONSE_CACHE.pop(key, None)
        return None
    _RESPONSE_CACHE.move_to_end(key)
    return response.model_copy(deep=True)


def _store_cached_response(
    key: str | None, response: InsuranceAnalysisResponse
) -> None:
    ttl = _cache_ttl_seconds()
    if (
        key is None
        or ttl <= 0
        or response.status not in {"completed", "limited", "needs_information"}
    ):
        return
    _RESPONSE_CACHE[key] = (
        time.monotonic() + ttl,
        response.model_copy(deep=True),
    )
    _RESPONSE_CACHE.move_to_end(key)
    while len(_RESPONSE_CACHE) > _RESPONSE_CACHE_MAX_ITEMS:
        _RESPONSE_CACHE.popitem(last=False)


def _error(
    request_id: str,
    *,
    status_code: int,
    error_code: str,
    message: str,
    retryable: bool,
) -> JSONResponse:
    body = InsuranceAnalysisErrorResponse(
        request_id=request_id,
        error_code=error_code,
        message=message,
        retryable=retryable,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _causes(exc: Exception):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_external_error(exc: Exception) -> bool:
    return any(
        isinstance(item, (APIError, InsuranceRetrievalError))
        for item in _causes(exc)
    )


def _is_configuration_error(exc: Exception) -> bool:
    if isinstance(exc, IndependentJudgmentConfigurationError):
        return True
    return any(
        isinstance(item, (InsuranceEvidencePipelineError, RuntimeError))
        and "configured" in str(item).casefold()
        for item in _causes(exc)
    )


@router.post(
    "/insurance/analyze",
    response_model=InsuranceAnalysisResponse,
    responses={
        400: {"model": InsuranceAnalysisErrorResponse},
        503: {"model": InsuranceAnalysisErrorResponse},
        504: {"model": InsuranceAnalysisErrorResponse},
        500: {"model": InsuranceAnalysisErrorResponse},
    },
    tags=["insurance"],
)
async def analyze_insurance(
    request: InsuranceAnalysisRequest,
    pipeline: Callable = Depends(get_insurance_pipeline),
    timeout_seconds: float = Depends(get_analysis_timeout_seconds),
):
    request_id = str(uuid4())
    started = time.perf_counter()
    cache_key = _cache_key(request, pipeline=pipeline)
    if cached := _cached_response(cache_key):
        cached.request_id = request_id
        cached.processing_time_ms = 0
        return cached
    if (
        request.vector_store_id is not None
        and not _VECTOR_STORE_ID.fullmatch(request.vector_store_id)
    ):
        return _error(
            request_id,
            status_code=400,
            error_code="invalid_vector_store_id",
            message="vector_store_id 형식이 올바르지 않습니다.",
            retryable=False,
        )
    try:
        result = await asyncio.wait_for(
            pipeline(
                QuestionRequest(
                    question=request.question,
                    user_context=request.user_context,
                ),
                vector_store_id=request.vector_store_id,
            ),
            timeout=timeout_seconds,
        )
        elapsed = round((time.perf_counter() - started) * 1000)
        response = to_insurance_analysis_response(
            request_id=request_id,
            pipeline_result=result,
            processing_time_ms=elapsed,
        )
        _store_cached_response(cache_key, response)
        completed_models = (
            result.independent_judgments.completed_model_count
            if result.independent_judgments else sum(
                item.independent_judgments.completed_model_count
                for item in result.sub_question_results
            )
        )
        logger.info(
            "insurance API analysis completed",
            extra={
                "request_id": request_id,
                "endpoint": "/insurance/analyze",
                "status": response.status,
                "stopped_at": response.stopped_at,
                "processing_time_ms": elapsed,
                "question_length": len(request.question),
                "completed_model_count": completed_models,
                "requires_human_review": response.requires_human_review,
            },
        )
        return response
    except TimeoutError:
        logger.warning(
            "insurance API analysis timed out",
            extra={
                "request_id": request_id,
                "endpoint": "/insurance/analyze",
                "timeout_seconds": timeout_seconds,
                "question_length": len(request.question),
            },
        )
        return _error(
            request_id,
            status_code=504,
            error_code="analysis_timeout",
            message="보험 분석 시간이 초과되었습니다.",
            retryable=True,
        )
    except Exception as exc:
        logger.exception(
            "insurance API analysis failed",
            extra={
                "request_id": request_id,
                "endpoint": "/insurance/analyze",
                "error_class": type(exc).__name__,
                "question_length": len(request.question),
            },
        )
        if _is_configuration_error(exc):
            return _error(
                request_id,
                status_code=503,
                error_code="service_not_configured",
                message="보험 분석 서비스 설정이 준비되지 않았습니다.",
                retryable=False,
            )
        if _is_external_error(exc):
            return _error(
                request_id,
                status_code=503,
                error_code="external_service_unavailable",
                message="외부 분석 서비스를 일시적으로 사용할 수 없습니다.",
                retryable=True,
            )
        return _error(
            request_id,
            status_code=500,
            error_code="internal_error",
            message="보험 분석 중 내부 오류가 발생했습니다.",
            retryable=False,
        )


@router.get("/insurance/readiness", tags=["insurance"])
def insurance_readiness() -> dict:
    load_dotenv(override=True)
    judgment_count = sum(
        bool(os.getenv(name))
        for name in (
            "INDEPENDENT_JUDGMENT_MODEL_A",
            "INDEPENDENT_JUDGMENT_MODEL_B",
            "INDEPENDENT_JUDGMENT_MODEL_C",
        )
    )
    ready = all((
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_VECTOR_STORE_ID"),
        judgment_count >= 2,
        os.getenv("FINAL_ANSWER_MODEL") or os.getenv("OPENAI_MODEL"),
    ))
    return {
        "status": "ready" if ready else "not_ready",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "vector_store_configured": bool(os.getenv("OPENAI_VECTOR_STORE_ID")),
        "judgment_models_configured": judgment_count,
        "final_answer_model_configured": bool(
            os.getenv("FINAL_ANSWER_MODEL") or os.getenv("OPENAI_MODEL")
        ),
    }
