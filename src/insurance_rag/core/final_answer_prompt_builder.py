from __future__ import annotations

import json
from pathlib import Path

from insurance_rag.schemas.final_answer import (
    FinalAnswerSynthesisRequest,
    FinalInsuranceAnswer,
)

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "insurance_final_answer.txt"


def build_final_answer_input(
    request: FinalAnswerSynthesisRequest,
    *,
    validation_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": {
            "original_question": request.original_question,
            "sub_question_id": request.sub_question_id,
        },
        "risk": request.risk.model_dump(exclude_none=True),
        "claims": [claim.model_dump(exclude_none=True) for claim in request.claims],
        "independent_judgments": request.judgments.model_dump(exclude_none=True),
        "claim_matrix": request.claim_matrix.model_dump(exclude_none=True),
    }
    if validation_errors:
        payload["retry_validation_errors"] = validation_errors[:8]
    system = (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\n다음 JSON Schema를 정확히 준수하라:\n"
        + json.dumps(FinalInsuranceAnswer.model_json_schema(), ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
