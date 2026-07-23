from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from insurance_rag.core.independent_judgment_prompt_builder import (
    build_independent_judgment_input,
)
from insurance_rag.core.independent_judgment_validator import (
    IndependentJudgmentConfigurationError,
    IndependentJudgmentError,
    validate_model_judgment,
)
from insurance_rag.core.llm_client import create_llm_client
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
    IndependentJudgmentRequest,
    IndependentModelJudgment,
)

logger = logging.getLogger(__name__)


class ModelJudgmentRunError(IndependentJudgmentError):
    pass


def _error_summary(exc: Exception) -> list[str]:
    if isinstance(exc, json.JSONDecodeError):
        return ["JSON parsing failed"]
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors()[:8]
        ]
    return [f"{type(exc).__name__}: judgment validation or API call failed"]


async def _request_model_judgment(
    request: IndependentJudgmentRequest,
    *,
    model_config: IndependentJudgmentModelConfig,
    validation_errors: list[str] | None = None,
) -> str:
    if model_config.provider != "openai":
        raise IndependentJudgmentConfigurationError(
            f"unsupported model provider: {model_config.provider}"
        )
    try:
        client = create_llm_client()
    except RuntimeError as exc:
        raise IndependentJudgmentConfigurationError(
            "OpenAI client is not configured"
        ) from exc
    kwargs = {
        "model": model_config.model_id,
        "max_output_tokens": 5000,
        "input": build_independent_judgment_input(
            request, validation_errors=validation_errors
        ),
        "text": {"format": {"type": "json_object"}},
    }
    if model_config.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": model_config.reasoning_effort}
    response = await client.responses.create(**kwargs)
    return response.output_text


async def generate_single_model_judgment(
    request: IndependentJudgmentRequest,
    *,
    model_config: IndependentJudgmentModelConfig,
) -> IndependentModelJudgment:
    """Generate one evidence-bound judgment without seeing any peer output."""
    if not model_config.enabled:
        raise IndependentJudgmentConfigurationError("model is disabled")
    errors = None
    for attempt in range(2):
        try:
            raw = await _request_model_judgment(
                request,
                model_config=model_config,
                validation_errors=errors,
            )
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("model output must be a JSON object")
            return validate_model_judgment(
                payload, request, model_id=model_config.model_id
            )
        except IndependentJudgmentConfigurationError:
            raise
        except Exception as exc:
            errors = _error_summary(exc)
            logger.warning(
                "independent judgment attempt failed",
                extra={
                    "stage": "model_judgment",
                    "model_id": model_config.model_id[:80],
                    "retry_count": attempt,
                    "error_class": type(exc).__name__,
                    "claim_count": len(request.claims),
                    "sub_question_id": request.sub_question_id,
                },
            )
            if attempt == 1:
                raise ModelJudgmentRunError(
                    "model judgment failed after one retry"
                ) from exc
    raise AssertionError("unreachable")
