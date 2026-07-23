# 로컬 개발 환경 점검

이 프로젝트는 Windows PowerShell과 cmd에서 모두 실행할 수 있다. 시스템 PATH에
Python이나 Git이 없어도 프로젝트의 `.venv`와 GitHub Desktop 내장 Git을 사용할
수 있다.

## 자동 점검

PowerShell:

```powershell
cd C:\git\codefair
.\.venv\Scripts\python.exe .\scripts\check_local_environment.py
```

cmd:

```cmd
cd /d C:\git\codefair
.venv\Scripts\python.exe scripts\check_local_environment.py
```

점검 스크립트는 API 키나 환경변수 값을 출력하지 않고 설정 여부만 표시한다.

## 전체 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

기본 실행에서는 실제 OpenAI smoke test가 skip된다.

## 보험 분석 API

로컬 서버:

```powershell
.\.venv\Scripts\python.exe -m uvicorn insurance_rag.api:app --host 127.0.0.1 --port 8000
```

보험 질문 화면은 `http://127.0.0.1:8000/`에서 연다. 대화 내용은 브라우저에
저장하지 않는다. API 문서는 `http://127.0.0.1:8000/docs`, OpenAPI JSON은
`http://127.0.0.1:8000/openapi.json`에서 확인한다. 보험 분석은
`POST http://127.0.0.1:8000/insurance/analyze`, 준비 상태는
`GET http://127.0.0.1:8000/insurance/readiness`로 제공한다. 기존 Vector Store
관리 화면은 `http://127.0.0.1:8000/files/manage`에서 사용할 수 있다.

## OpenAI smoke test

실제 호출에는 비용이 발생하므로 사용자가 직접 opt-in 해야 한다. `.env` 또는 현재
터미널에 API 키와 모델 ID를 설정한 뒤 실행한다.

PowerShell:

```powershell
$env:RUN_OPENAI_SMOKE_TESTS="1"
$env:INSURANCE_CLAIM_EXTRACTOR_MODEL="<model-id>"
$env:INDEPENDENT_JUDGMENT_MODEL_A="<model-a>"
$env:INDEPENDENT_JUDGMENT_MODEL_B="<model-b>"
$env:FINAL_ANSWER_MODEL="<model-final>"
.\.venv\Scripts\python.exe -m pytest .\tests\integration -v
```

cmd:

```cmd
set RUN_OPENAI_SMOKE_TESTS=1
set INSURANCE_CLAIM_EXTRACTOR_MODEL=<model-id>
set INDEPENDENT_JUDGMENT_MODEL_A=<model-a>
set INDEPENDENT_JUDGMENT_MODEL_B=<model-b>
set FINAL_ANSWER_MODEL=<model-final>
.venv\Scripts\python.exe -m pytest tests\integration -v
```

API 키는 명령 기록에 직접 입력하지 말고 `.env` 또는 안전한 비밀 관리 기능으로
설정한다.

## GitHub 게시

현재 브랜치는 GitHub Desktop의 `Publish branch` 또는 `Push origin`으로 게시할 수
있다. 명령줄 Git이 PATH에 등록된 환경에서는 다음 명령을 사용할 수 있다.

```powershell
git push -u origin feature/independent-insurance-judgments
```

force push는 사용하지 않는다.
