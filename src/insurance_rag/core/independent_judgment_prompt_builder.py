from __future__ import annotations

import json
from pathlib import Path

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentRequest,
    IndependentModelJudgment,
)

PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "independent_insurance_judgment.txt"
)


def build_independent_judgment_input(
    request: IndependentJudgmentRequest,
    *,
    validation_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    question = {
        "original_question": request.original_question,
        "normalized_question": request.normalized_question,
        "requested_action": request.requested_action,
        "sub_question_id": request.sub_question_id,
    }
    risk = {
        "risk_level": request.risk_level,
        "categories": request.risk_categories,
        "allow_final_answer": request.allow_final_answer,
        "requires_disclaimer": request.requires_disclaimer,
        "requires_human_review": request.requires_human_review,
        "requires_additional_information": request.requires_additional_information,
        "missing_information": request.missing_information,
        "warnings": request.warnings,
    }
    claims = [claim.model_dump(exclude_none=True) for claim in request.claims]
    payload = (
        "<BEGIN_QUESTION>\n"
        f"{json.dumps(question, ensure_ascii=False)}\n"
        "<END_QUESTION>\n\n"
        "<BEGIN_RISK_RESULT>\n"
        f"{json.dumps(risk, ensure_ascii=False)}\n"
        "<END_RISK_RESULT>\n\n"
        "<BEGIN_CLAIMS>\n"
        f"{json.dumps(claims, ensure_ascii=False)}\n"
        "<END_CLAIMS>"
    )
    if validation_errors:
        payload += (
            "\n\n<BEGIN_RETRY_ERRORS>\n"
            f"{json.dumps(validation_errors[:8], ensure_ascii=False)}\n"
            "<END_RETRY_ERRORS>"
        )
    system = (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\n다음 JSON Schema를 정확히 준수하라:\n"
        + json.dumps(
            IndependentModelJudgment.model_json_schema(), ensure_ascii=False
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": payload},
    ]
