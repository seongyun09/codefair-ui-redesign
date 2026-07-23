from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from insurance_rag.core.llm_client import create_llm_client, question_analyzer_model
from insurance_rag.schemas.question import QuestionAnalysis, QuestionRequest

logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "question_analysis.txt"


class QuestionAnalysisError(Exception):
    """Raised when the model cannot produce a valid question analysis."""


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


async def _request_question_analysis(
    *, request: QuestionRequest, retry: bool = False
) -> str:
    client = create_llm_client()
    model = question_analyzer_model()
    user_payload: dict[str, Any] = {
        "question": request.question,
        "user_context": request.user_context,
    }
    if retry:
        user_payload["retry_instruction"] = (
            "이전 출력이 JSON 형식 또는 지정 스키마를 충족하지 않았다. "
            "설명문 없이 지정된 JSON 스키마만 다시 출력하라."
        )

    schema_instruction = json.dumps(
        QuestionAnalysis.model_json_schema(), ensure_ascii=False
    )

    response = await client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        max_output_tokens=1500,
        input=[
            {
                "role": "system",
                "content": (
                    f"{_load_system_prompt()}\n\n"
                    "다음 JSON Schema의 필드명과 구조를 정확히 준수하라:\n"
                    f"{schema_instruction}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        text={
            "format": {
                "type": "json_object",
            }
        },
    )
    return response.output_text


def _validate_response(raw_response: str, request: QuestionRequest) -> QuestionAnalysis:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    payload["original_question"] = request.question
    analysis = QuestionAnalysis.model_validate(payload)
    context_values = list(request.user_context.values())
    for item in analysis.known_information:
        if item.source == "user_context" and item.value not in context_values:
            raise ValueError("user_context known information was not provided by the user")
        if (
            item.source == "question"
            and item.value in context_values
            and str(item.value) not in request.question
        ):
            item.source = "user_context"
    return analysis


async def analyze_question(request: QuestionRequest) -> QuestionAnalysis:
    """Analyze and structure a question without answering it."""
    model = "unconfigured"
    try:
        model = question_analyzer_model()
    except RuntimeError:
        pass

    for attempt in range(2):
        try:
            raw_response = await _request_question_analysis(
                request=request, retry=bool(attempt)
            )
            return _validate_response(raw_response, request)
        except Exception as exc:
            error_type = type(exc).__name__
            validation_type = (
                "pydantic_validation" if isinstance(exc, ValidationError) else error_type
            )
            logger.warning(
                "question analysis failed",
                extra={
                    "failure_stage": "llm_call_or_validation",
                    "request_id": None,
                    "error_type": error_type,
                    "validation_error_type": validation_type,
                    "retry": attempt == 1,
                    "model": model,
                    "question_length": len(request.question),
                    "context_key_count": len(request.user_context),
                },
            )
            if attempt == 1:
                raise QuestionAnalysisError(
                    "question analysis failed after one retry"
                ) from exc

    raise AssertionError("unreachable")
