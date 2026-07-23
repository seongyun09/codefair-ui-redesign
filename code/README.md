# Nova — Flask AI 챗봇

Claude · ChatGPT · Gemini의 화면 구성을 참고해 만든 챗봇 웹 앱입니다.
Python(Flask) 백엔드 + 순수 HTML/CSS/JavaScript 프론트엔드로 되어 있어요.

## 주요 기능

- **실시간 스트리밍 응답** — 글자가 타자 치듯 흘러나오고, 생성 중 ■ 버튼으로 중지 가능
- **마크다운 렌더링** — 굵게, 목록, 인라인 코드, 코드 블록 지원
- **대화 기록** — 사이드바에 대화가 저장되고(브라우저 localStorage), 삭제 가능
- **다시 생성 / 복사** — 마지막 답변 재생성, 답변 복사 버튼
- **다크 모드** — 사이드바 하단에서 전환, 시스템 설정 자동 감지
- **시간대별 인사말** — 아침/오후/저녁/밤에 따라 첫 화면 인사가 바뀜
- **모바일 반응형** — 좁은 화면에서는 사이드바가 서랍(햄버거 메뉴)으로 변신

## 실행 방법

```bash
pip install flask
python app.py
```

브라우저에서 **http://localhost:5000** 접속.

기본은 **데모 모드**라 API 키 없이 바로 대화할 수 있어요.
(미리 준비된 규칙 기반 답변으로 동작합니다.)

## 실제 AI API 연결하기

`app.py` 상단의 `USE_REAL_API = False`를 `True`로 바꾸고, 아래 중 하나를 설정하세요.

### Claude (Anthropic) — 기본 예시 코드 포함

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
python app.py
```

`stream_real_reply()` 함수가 이미 Claude 스트리밍 방식으로 작성되어 있습니다.

### OpenAI (GPT)로 바꾸고 싶다면

`stream_real_reply()` 내용을 아래처럼 교체하세요.

```python
def stream_real_reply(message, history):
    from openai import OpenAI          # pip install openai
    client = OpenAI()                  # OPENAI_API_KEY 환경변수 사용
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-20:]]
    msgs.append({"role": "user", "content": message})
    stream = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

## 폴더 구조

```
nova-chatbot/
├── app.py               # Flask 서버 + 데모 봇 + API 연결 지점
├── requirements.txt
├── templates/
│   └── index.html       # 화면 구조
└── static/
    ├── style.css        # 라이트/다크 테마, 레이아웃, 애니메이션
    └── script.js        # 대화 저장, 스트리밍 수신, 마크다운 렌더러
```

## 커스터마이징 포인트

| 바꾸고 싶은 것 | 위치 |
|---|---|
| 챗봇 이름 | `app.py`의 `BOT_NAME` |
| 데모 답변 내용 | `app.py`의 `demo_reply()` |
| 타이핑 속도 | `app.py`의 `stream_demo_reply()` 안 `time.sleep(0.012)` |
| 색상 테마 | `static/style.css` 맨 위 `:root` / `[data-theme="dark"]` 변수 |
| 첫 화면 추천 질문 | `templates/index.html`의 `.chip` 버튼들 |
| 포트 번호 | `app.py` 맨 아래 `app.run(..., port=5000)` |
