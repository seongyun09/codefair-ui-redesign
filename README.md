# OpenAI Vector Store 파일 업로더

지정된 OpenAI Vector Store에 파일을 업로드하는 단일 페이지 FastAPI 사이트입니다.

대상 Vector Store는 `OPENAI_VECTOR_STORE_ID` 환경변수로 설정합니다.

## 실행

채팅에 노출된 API 키는 폐기하고 새 키를 발급한 뒤 환경변수로 설정하세요. 키를 코드나 `.env.example`에 넣지 마세요.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
$env:OPENAI_API_KEY="새로 발급한 키"
$env:UPLOAD_PASSWORD="안전한 관리 비밀번호"
python -m uvicorn insurance_rag.api:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

## Codespaces

GitHub 저장소의 **Settings → Secrets and variables → Codespaces**에 새 `OPENAI_API_KEY`를 등록한 후 Codespace를 생성하세요. 서버는 자동으로 시작되고 포트 8000으로 열립니다.

## 동작

1. 서버가 파일을 OpenAI Files API에 `assistants` 목적으로 업로드합니다.
2. 생성된 File ID를 지정된 Vector Store에 연결합니다.
3. 인덱싱 완료까지 기다린 뒤 파일 ID와 상태를 화면에 표시합니다.

파일당 사이트 제한은 50MB이고 한 번에 최대 20개를 처리합니다. API 키는 서버 환경변수에서만 읽으며 브라우저로 보내지 않습니다.

## 보험 문서 배치 적재

`data/source/insurance/` 아래의 `.jsonl` 및 JSONL 형식 `.txt` 파일은 재귀적으로 발견하여 문서별로 검증하고 적재할 수 있습니다. 동일 내용과 이미 완료된 파일은 건너뛰며, 결과는 `data/vector_store_manifest.json`에 누적됩니다. ZIP은 자동 처리하지 않으므로 안전하게 수동 압축 해제한 뒤 입력 디렉터리에 넣습니다.

```python
from insurance_rag import ingest_insurance_documents

result = await ingest_insurance_documents(
    "data/source/insurance",
    vector_store_id="vs_...",
)
```

회사와 상품 코드는 각각 `data/config/company_codes.json`, `data/config/product_codes.json`에 JSON 문자열 매핑으로 둘 수 있습니다. 매핑이 없으면 충돌 가능성이 낮은 결정적 코드가 생성되고 manifest에 기록됩니다.

업로드 API는 `X-Upload-Password` 헤더를 검사합니다. 기본 비밀번호는 `0651`이며, 운영 환경에서는 `UPLOAD_PASSWORD` 환경변수로 변경할 수 있습니다.

## Render로 공개 배포

저장소 루트의 `render.yaml`을 이용하면 Render Blueprint로 배포할 수 있습니다.

1. [Render Dashboard](https://dashboard.render.com/)에 GitHub로 로그인합니다.
2. **New → Blueprint**를 선택합니다.
3. 비공개 저장소 `thqudgns/codefair`에 대한 접근을 허용하고 저장소를 선택합니다.
4. Blueprint가 요구하는 `OPENAI_API_KEY`에 새로 발급한 키를 입력합니다.
5. 배포가 끝나면 Render가 제공하는 `https://...onrender.com` 주소를 공유합니다.

`OPENAI_VECTOR_STORE_ID`와 업로드 비밀번호는 Blueprint에 설정되어 있습니다. 채팅에 노출된 기존 API 키는 사용하지 마세요.
