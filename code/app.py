# -*- coding: utf-8 -*-
"""
Nova — Flask 기반 AI 챗봇 (Claude / ChatGPT / Gemini 스타일 UI)

실행:
    pip install flask
    python app.py
    → 브라우저에서 http://localhost:5000 접속

기본은 '데모 모드'라서 API 키 없이 바로 동작합니다.
실제 AI(예: Claude API)를 연결하려면 아래 USE_REAL_API 부분을 참고하세요.
"""

import random
import time
from datetime import datetime

from flask import Flask, Response, render_template, request, stream_with_context

app = Flask(__name__)

BOT_NAME = "Nova"

# ──────────────────────────────────────────────────────────────
# 실제 AI API 연결하기 (선택)
#
#   1) pip install anthropic
#   2) 터미널에서 환경변수 설정:  export ANTHROPIC_API_KEY="sk-..."
#      (Windows PowerShell:  $env:ANTHROPIC_API_KEY="sk-...")
#   3) 아래 USE_REAL_API 를 True 로 변경
#
#   OpenAI를 쓰고 싶다면 stream_real_reply 안의 코드를
#   openai 라이브러리 방식으로 바꾸면 됩니다. (README.md 참고)
# ──────────────────────────────────────────────────────────────
USE_REAL_API = False


def stream_real_reply(message, history):
    """실제 LLM API(Anthropic Claude)를 스트리밍으로 호출하는 예시."""
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용

    messages = [
        {"role": h["role"], "content": h["content"]}
        for h in history[-20:]  # 최근 20개 메시지만 컨텍스트로 사용
        if h.get("role") in ("user", "assistant") and h.get("content")
    ]
    messages.append({"role": "user", "content": message})

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"당신은 {BOT_NAME}라는 이름의 친절한 한국어 AI 어시스턴트입니다.",
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ──────────────────────────────────────────────────────────────
# 데모 봇: API 키 없이도 동작하는 규칙 기반 응답
# ──────────────────────────────────────────────────────────────

FUN_FACTS = [
    "문어는 심장이 **3개**나 있어요. 두 개는 아가미로, 하나는 온몸으로 피를 보내죠. 🐙",
    "꿀은 거의 **상하지 않아요**. 고대 이집트 무덤에서 발견된 3,000년 된 꿀도 먹을 수 있는 상태였대요. 🍯",
    "식물학적으로 **바나나는 베리(장과)** 지만, 딸기는 베리가 아니에요. 🍌",
    "우주에서는 눈물이 흘러내리지 않고 **눈 주위에 공처럼 맺혀요**. 중력이 없기 때문이죠. 🚀",
    "고래상어의 몸 무늬는 사람의 지문처럼 **개체마다 전부 달라요**. 🦈",
]

MENUS = [
    ("김치찌개", "뜨끈한 국물 한 숟갈이면 하루 피로가 싹 풀리죠"),
    ("연어 초밥", "가볍지만 은근히 든든하게 채우고 싶은 날"),
    ("삼겹살", "불판 위 지글지글 소리만으로도 이미 힐링"),
    ("크림 파스타", "오늘은 조금 느끼해도 괜찮은 날"),
    ("쌀국수", "따뜻한 국물에 라임 한 조각, 고수는 취향껏"),
    ("치킨", "고민될 땐 언제나 정답"),
]

GREETINGS = [
    "안녕하세요! 만나서 반가워요. 😊 무엇을 도와드릴까요?",
    "안녕하세요! 오늘 하루는 어떠셨나요? 편하게 말 걸어 주세요.",
    "반갑습니다! 궁금한 것이 있다면 무엇이든 물어봐 주세요. ✨",
]


def demo_reply(message):
    """키워드 기반의 간단한 데모 응답 생성."""
    m = message.lower().strip()

    def has(*keywords):
        return any(k in m for k in keywords)

    if has("소개", "이름", "누구", "정체"):
        return (
            f"안녕하세요! 저는 **{BOT_NAME}**, Flask와 바닐라 JavaScript로 만들어진 챗봇이에요. ✨\n\n"
            "지금은 미리 준비된 답변으로 대화하는 **데모 모드**로 움직이고 있어요. "
            "`app.py`에서 실제 AI API(Claude, GPT 등)를 연결하면 진짜 언어 모델과 대화할 수 있답니다.\n\n"
            "이런 걸 시험해 보세요:\n"
            "- \"파이썬 코드 보여줘\"\n"
            "- \"재미있는 사실 알려줘\"\n"
            "- \"저녁 메뉴 추천해줘\""
        )

    if has("기능", "뭘 할", "무엇을 할", "할 수 있", "도움말"):
        return (
            "이 챗봇 앱이 보여주는 기능은 이런 것들이에요:\n\n"
            "1. **실시간 스트리밍** — 글자가 타자 치듯 흘러나와요\n"
            "2. **마크다운 렌더링** — 굵게, `코드`, 목록, 코드 블록까지\n"
            "3. **대화 기록 저장** — 브라우저(localStorage)에 대화가 남아요\n"
            "4. **다크 모드** — 사이드바 아래에서 전환할 수 있어요\n"
            "5. **응답 중지 / 다시 생성 / 복사** 버튼\n\n"
            "실제 AI를 연결하는 방법은 `README.md`에 정리해 뒀어요!"
        )

    if has("코드", "파이썬", "python", "코딩", "프로그래밍"):
        return (
            "물론이죠! 파이썬으로 피보나치 수열을 출력하는 예제예요:\n\n"
            "```python\n"
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        print(a, end=\" \")\n"
            "        a, b = b, a + b\n"
            "\n"
            "fibonacci(10)  # 0 1 1 2 3 5 8 13 21 34\n"
            "```\n\n"
            "코드 블록도 이렇게 렌더링된답니다. 다른 예제가 필요하면 말씀해 주세요!"
        )

    if has("사실", "fact", "신기한"):
        return f"오늘의 재미있는 사실! 💡\n\n{random.choice(FUN_FACTS)}\n\n또 듣고 싶으면 한 번 더 물어봐 주세요."

    if has("메뉴", "저녁", "점심", "아침", "먹을", "뭐 먹", "배고"):
        name, reason = random.choice(MENUS)
        return f"오늘은 **{name}** 어떠세요? 🍽️\n\n{reason}! 다른 메뉴가 궁금하면 한 번 더 물어봐 주세요."

    if has("날씨"):
        return (
            "데모 모드라서 실시간 날씨는 아직 볼 수 없어요. 😅\n\n"
            "`app.py`에 날씨 API나 실제 AI API를 연결하면 가능해져요. "
            "그때까지는… 창밖을 한번 봐 주시는 걸 추천드려요!"
        )

    if has("시간", "몇 시", "날짜", "며칠"):
        now = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        return f"지금은 **{now}**이에요. ⏰ (서버 컴퓨터 기준 시간이에요.)"

    if has("고마", "감사", "thank"):
        return "천만에요! 도움이 되었다니 기뻐요. 😊 더 궁금한 게 있으면 언제든지요."

    if has("안녕", "하이", "헬로", "hello", "hi", "반가"):
        return random.choice(GREETINGS)

    # 기본 응답
    snippet = message.strip()[:40]
    return random.choice(
        [
            (
                f"\"{snippet}\" 에 대한 이야기군요!\n\n"
                "지금 저는 **데모 모드**라서 미리 준비된 답변만 할 수 있어요. "
                "`app.py`에서 실제 AI API를 연결하면 어떤 질문이든 제대로 답할 수 있게 돼요.\n\n"
                "지금 바로 해볼 수 있는 것들:\n"
                "- \"네 소개를 해줘\"\n"
                "- \"파이썬 코드 보여줘\"\n"
                "- \"저녁 메뉴 추천해줘\""
            ),
            (
                f"흥미로운 주제네요 — \"{snippet}\"!\n\n"
                "다만 저는 아직 **데모 모드**라 정해진 답변 몇 가지만 알고 있어요. 😅 "
                "진짜 AI 두뇌를 달아주려면 `README.md`의 *실제 API 연결하기* 부분을 봐 주세요.\n\n"
                "그동안 \"재미있는 사실 알려줘\" 같은 건 지금도 잘한답니다!"
            ),
        ]
    )


def stream_demo_reply(message):
    """데모 응답을 타자 치듯 조금씩 흘려보내는 제너레이터."""
    reply = demo_reply(message)
    chunk_size = 2
    for i in range(0, len(reply), chunk_size):
        yield reply[i : i + chunk_size]
        time.sleep(0.012)  # 타이핑 속도 (숫자를 줄이면 빨라져요)


# ──────────────────────────────────────────────────────────────
# 라우트
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", bot_name=BOT_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return {"error": "message가 비어 있어요."}, 400

    def generate():
        try:
            if USE_REAL_API:
                yield from stream_real_reply(message, history)
            else:
                yield from stream_demo_reply(message)
        except Exception as e:  # API 키 누락 등 오류를 채팅창에 그대로 안내
            yield f"\n\n⚠️ 오류가 발생했어요: `{e}`\n\nAPI 키 설정과 라이브러리 설치를 확인해 주세요."

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # 프록시 버퍼링 방지
    }
    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers=headers,
    )


if __name__ == "__main__":
    # 포트를 바꾸려면 port=5000 값을 수정하세요.
    app.run(debug=True, host="127.0.0.1", port=5000)
