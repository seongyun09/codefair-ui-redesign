from fastapi.testclient import TestClient

from insurance_rag.api import app

client = TestClient(app)


def test_insurance_frontend_is_served_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "하나의 AI 답변, 그대로 믿어도 될까요?" in response.text
    assert "AI들의 판단 차이를 보여줘" in response.text
    assert "/static/style.css" in response.text
    assert "/static/script.js" in response.text
    assert "새 대화" in response.text
    assert 'id="chatList"' in response.text
    assert 'id="sourceDrawer"' in response.text
    assert "등록된 약관은 어떤 보험인가요?" in response.text
    assert "현재 등록된 약관 문서의 보험상품명" in response.text


def test_frontend_script_calls_structured_insurance_api_with_conversation_history():
    response = client.get("/static/script.js")
    assert response.status_code == 200
    assert 'fetch("/insurance/analyze"' in response.text
    assert "localStorage" not in response.text
    assert "chatList" in response.text
    assert "/api/chat" not in response.text
    assert "sub_answers" in response.text
    assert "conversationHistory" in response.text
    assert "conversation_history" in response.text
    assert 'input.value = ""' in response.text
    assert 'messages.insertAdjacentHTML("beforeend"' in response.text
    assert "progressSteps" in response.text
    assert "followUpQuestions" in response.text
    assert "다시 시도" in response.text
    assert "근거 보기" in response.text
    assert "informationGuidance" in response.text
    assert "상품명 추가" in response.text
    assert "입력하신 정보의 문제가 아닙니다" in response.text


def test_file_manager_remains_available_on_separate_path():
    response = client.get("/files/manage")
    assert response.status_code == 200
    assert "Vector Store 파일 업로드" in response.text
