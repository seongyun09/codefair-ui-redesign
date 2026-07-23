from __future__ import annotations

import asyncio
import json
import logging
import os

from pydantic import ValidationError

from insurance_rag.core.final_answer_prompt_builder import build_final_answer_input
from insurance_rag.core.final_answer_validator import (
    FinalAnswerValidationError,
    validate_final_answer,
)
from insurance_rag.core.llm_client import (
    create_llm_client,
    final_answer_model,
)
from insurance_rag.schemas.final_answer import (
    FinalAnswerSynthesisRequest,
    FinalInsuranceAnswer,
)

logger = logging.getLogger(__name__)


class FinalAnswerSynthesisError(RuntimeError):
    pass


def _errors(exc: Exception) -> list[str]:
    if isinstance(exc, FinalAnswerValidationError):
        return [f"FinalAnswerValidationError: {exc}"]
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors()[:8]
        ]
    if isinstance(exc, json.JSONDecodeError):
        return ["JSON parsing failed"]
    return [f"{type(exc).__name__}: synthesis validation or API call failed"]


async def synthesize_final_answer(
    request: FinalAnswerSynthesisRequest,
) -> FinalInsuranceAnswer:
    """Make one evidence-bound synthesis and fall back without rerunning analysis."""
    try:
        client = create_llm_client()
        response = await asyncio.wait_for(
            client.responses.create(
                model=final_answer_model(),
                max_output_tokens=4000,
                input=build_final_answer_input(request),
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "low"},
            ),
            timeout=float(os.getenv("FINAL_SYNTHESIS_TIMEOUT_SECONDS", "12")),
        )
        payload = json.loads(response.output_text)
        if not isinstance(payload, dict):
            raise ValueError("model output must be a JSON object")
        return validate_final_answer(payload, request)
    except RuntimeError as exc:
        if "configured" in str(exc):
            raise
        errors = _errors(exc)
    except Exception as exc:
        errors = _errors(exc)
    logger.warning(
        "final answer synthesis failed; using verified fallback",
        extra={
            "stage": "final_answer_synthesis",
            "error_class": errors[0].split(":", 1)[0],
            "claim_count": len(request.claims),
            "sub_question_id": request.sub_question_id,
        },
    )
    claims = request.claims
    key_points = [claim.statement for claim in claims if claim.relevance == "direct"]
    if not key_points:
        key_points = [claim.statement for claim in claims[:3]]
    conditions = list(dict.fromkeys(
        value for claim in claims for value in [*claim.conditions, *claim.applies_when]
    ))
    exceptions = list(dict.fromkeys(
        value for claim in claims
        for value in [*claim.exceptions, *claim.does_not_apply_when]
    ))
    return FinalInsuranceAnswer(
        status="limited",
        answer=(
            "최종 문장 생성은 완료되지 않았지만, 앞 단계에서 검증된 약관 근거와 "
            "AI 판단 비교 결과를 아래에 그대로 제공합니다."
        ),
        key_points=key_points[:6],
        applicable_conditions=conditions[:8],
        important_exceptions=exceptions[:8],
        missing_information=request.risk.missing_information,
        sources=[{"claim_id": claim.claim_id} for claim in claims],
        model_agreement=request.claim_matrix.overall_agreement,
        requires_disclaimer=True,
        requires_human_review=(
            request.risk.requires_human_review
            or request.claim_matrix.requires_human_review
        ),
        warnings=["final_synthesis_fallback", *(errors or [])],
    )
