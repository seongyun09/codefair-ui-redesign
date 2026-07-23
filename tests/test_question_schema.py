from __future__ import annotations

import pytest
from pydantic import ValidationError

from insurance_rag.schemas.question import QuestionAnalysis, QuestionRequest


def analysis_payload(**overrides):
    payload = {
        "original_question": "면책기간이 무엇인가요?",
        "normalized_question": "면책기간의 정의를 알고 싶다.",
        "main_intent": "면책기간 정의 확인",
        "assumptions": [],
        "known_information": [],
        "missing_information": [],
        "is_compound": False,
        "sub_questions": [
            {
                "id": "q1",
                "question": "면책기간이 무엇인가?",
                "requested_action": "definition",
                "purpose": "보험 용어 정의 확인",
                "depends_on_information": [],
            }
        ],
        "analysis_warnings": [],
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("question", [" ", " 가 ", "x" * 4001])
def test_question_request_rejects_invalid_length(question):
    with pytest.raises(ValidationError):
        QuestionRequest(question=question)


def test_question_request_strips_and_uses_independent_context_defaults():
    first = QuestionRequest(question="  질문입니다  ")
    second = QuestionRequest(question="다른 질문")
    first.user_context["age"] = 40
    assert first.question == "질문입니다"
    assert second.user_context == {}


@pytest.mark.parametrize(
    "sub_questions",
    [
        [],
        [
            {
                "id": f"q{index}",
                "question": "유효한 질문인가?",
                "requested_action": "definition",
                "purpose": "검증 목적",
                "depends_on_information": [],
            }
            for index in range(1, 7)
        ],
    ],
)
def test_sub_question_count_limits(sub_questions):
    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(
            analysis_payload(sub_questions=sub_questions, is_compound=True)
        )


@pytest.mark.parametrize(
    "sub_questions",
    [
        [{"id": "q6", "question": "질문인가?", "requested_action": "definition", "purpose": "검증 목적", "depends_on_information": []}],
        [
            {"id": "q1", "question": "첫 질문인가?", "requested_action": "definition", "purpose": "검증 목적", "depends_on_information": []},
            {"id": "q1", "question": "둘 질문인가?", "requested_action": "definition", "purpose": "검증 목적", "depends_on_information": []},
        ],
        [
            {"id": "q1", "question": "첫 질문인가?", "requested_action": "definition", "purpose": "검증 목적", "depends_on_information": []},
            {"id": "q3", "question": "셋 질문인가?", "requested_action": "definition", "purpose": "검증 목적", "depends_on_information": []},
        ],
    ],
)
def test_sub_question_ids_must_be_valid_and_consecutive(sub_questions):
    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(
            analysis_payload(sub_questions=sub_questions, is_compound=True)
        )


def test_non_compound_question_has_exactly_one_sub_question():
    question = analysis_payload()["sub_questions"][0]
    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(
            analysis_payload(sub_questions=[question, {**question, "id": "q2"}])
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assumptions", ["전제"] * 8),
        ("analysis_warnings", ["경고"] * 6),
    ],
)
def test_top_level_list_limits(field, value):
    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(analysis_payload(**{field: value}))


def test_dependency_limit_and_requested_action_are_validated():
    question = analysis_payload()["sub_questions"][0]
    for change in (
        {"depends_on_information": [str(index) for index in range(8)]},
        {"requested_action": "answer"},
    ):
        with pytest.raises(ValidationError):
            QuestionAnalysis.model_validate(
                analysis_payload(sub_questions=[{**question, **change}])
            )


def test_known_information_wins_over_duplicate_missing_information():
    result = QuestionAnalysis.model_validate(
        analysis_payload(
            known_information=[{"name": " 나이 ", "value": 40, "source": "user_context"}],
            missing_information=[
                {"name": "나이", "reason": "판단에 필요한 정보이다"},
                {"name": "건강 상태", "reason": "가입 가능성 판단에 필요하다"},
            ],
        )
    )
    assert [item.name for item in result.missing_information] == ["건강 상태"]
