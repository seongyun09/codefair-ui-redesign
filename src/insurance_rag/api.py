from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from insurance_rag.api_insurance import router as insurance_router

VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "")
FRONTEND_ROOT = Path(__file__).parents[2] / "code"
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILES = 20
UPLOAD_PASSWORD = os.getenv("UPLOAD_PASSWORD", "")

app = FastAPI(title="Codefair Insurance Analysis", version="1.0.0")
app.include_router(insurance_router)
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_ROOT / "static"),
    name="static",
)


class FileAttributes(BaseModel):
    attributes: dict[str, str | int | bool]


def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="서버에 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def require_upload_password(x_upload_password: str | None = Header(default=None)) -> None:
    if not UPLOAD_PASSWORD:
        raise HTTPException(status_code=503, detail="서버에 UPLOAD_PASSWORD가 설정되지 않았습니다.")
    if not x_upload_password or not hmac.compare_digest(x_upload_password, UPLOAD_PASSWORD):
        raise HTTPException(status_code=401, detail="업로드 비밀번호가 올바르지 않습니다.")


async def upload_one(client: OpenAI, upload: UploadFile) -> dict:
    if not upload.filename:
        raise ValueError("파일 이름이 없습니다.")
    content = await upload.read(MAX_FILE_BYTES + 1)
    if not content:
        raise ValueError(f"{upload.filename}: 빈 파일입니다.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"{upload.filename}: 50MB 제한을 초과했습니다.")
    created = client.files.create(
        file=(upload.filename, content, upload.content_type or "application/octet-stream"),
        purpose="assistants",
    )
    try:
        attached = client.vector_stores.files.create_and_poll(
            vector_store_id=VECTOR_STORE_ID,
            file_id=created.id,
        )
    except Exception:
        client.files.delete(created.id)
        raise
    return {
        "filename": upload.filename,
        "file_id": created.id,
        "vector_store_id": VECTOR_STORE_ID,
        "status": attached.status,
        "bytes": len(content),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vector_store_id": VECTOR_STORE_ID or None, "api_key_configured": bool(os.getenv("OPENAI_API_KEY")), "vector_store_configured": bool(VECTOR_STORE_ID)}


@app.post("/upload", dependencies=[Depends(require_upload_password)])
async def upload_files(files: Annotated[list[UploadFile], File(...)]) -> dict:
    if not files or len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"한 번에 1~{MAX_FILES}개 파일을 선택하세요.")
    client = openai_client()
    completed, failed = [], []
    for upload in files:
        try:
            completed.append(await upload_one(client, upload))
        except Exception as exc:
            failed.append({"filename": upload.filename, "error": str(exc)})
        finally:
            await upload.close()
    return {"uploaded": completed, "failed": failed, "vector_store_id": VECTOR_STORE_ID}


@app.get("/files", dependencies=[Depends(require_upload_password)])
def list_files() -> dict:
    client = openai_client()
    page = client.vector_stores.files.list(vector_store_id=VECTOR_STORE_ID, limit=100, order="desc")
    files = []
    for item in page.data:
        filename, source_bytes = item.id, None
        try:
            source = client.files.retrieve(item.id)
            filename = source.filename
            source_bytes = source.bytes
        except Exception:
            pass
        files.append({
            "file_id": item.id,
            "filename": filename,
            "status": item.status,
            "usage_bytes": getattr(item, "usage_bytes", None),
            "source_bytes": source_bytes,
            "created_at": datetime.fromtimestamp(item.created_at, tz=timezone.utc).isoformat(),
            "attributes": getattr(item, "attributes", None) or {},
        })
    return {"files": files, "has_more": page.has_more, "vector_store_id": VECTOR_STORE_ID}


@app.patch("/files/{file_id}/attributes", dependencies=[Depends(require_upload_password)])
def update_file_attributes(file_id: str, body: FileAttributes) -> dict:
    if len(body.attributes) > 16:
        raise HTTPException(status_code=400, detail="속성은 최대 16개까지 저장할 수 있습니다.")
    cleaned = {str(k)[:64]: v[:512] if isinstance(v, str) else v for k, v in body.attributes.items() if str(k).strip()}
    updated = openai_client().vector_stores.files.update(
        vector_store_id=VECTOR_STORE_ID,
        file_id=file_id,
        attributes=cleaned,
    )
    return {"file_id": updated.id, "attributes": updated.attributes or {}}


@app.delete("/files/{file_id}", dependencies=[Depends(require_upload_password)])
def delete_file(file_id: str) -> dict:
    client = openai_client()
    client.vector_stores.files.delete(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
    deleted_source = False
    try:
        deleted_source = bool(client.files.delete(file_id).deleted)
    except Exception:
        pass
    return {"file_id": file_id, "removed_from_vector_store": True, "deleted_source": deleted_source}


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vector Store 파일 관리</title><style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#f3f6fb;color:#172033}.wrap{max-width:1000px;margin:36px auto;padding:20px}.card{background:#fff;border:1px solid #dde4ef;border-radius:18px;padding:28px;box-shadow:0 12px 35px #15213b12;margin-bottom:20px}h1,h2{margin:0 0 8px}.sub{color:#667085;margin:0 0 20px}.password,.search,.meta{width:100%;padding:11px;border:1px solid #b8c3d4;border-radius:9px;font:inherit}.password{margin:10px 0 18px}.drop{display:block;border:2px dashed #aebbd0;border-radius:14px;padding:30px;text-align:center;background:#f8faff;cursor:pointer}.drop:hover{border-color:#2768e8;background:#f1f6ff}input[type=file]{display:none}.selected{margin:14px 0;color:#475467;white-space:pre-wrap}.btn{padding:11px 15px;border:0;border-radius:9px;background:#1769e0;color:#fff;font:700 14px system-ui;cursor:pointer}.wide{width:100%}.btn:disabled{opacity:.55;cursor:wait}.danger{background:#b42318}.light{background:#eaf1ff;color:#174ea6}.result{margin-top:16px;padding:14px;border-radius:10px;background:#101828;color:#d9f7e5;white-space:pre-wrap;min-height:48px;line-height:1.5}.id{font-family:ui-monospace,monospace;font-size:12px;color:#667085;word-break:break-all}.toolbar{display:grid;grid-template-columns:1fr auto;gap:10px;margin:16px 0}.file-card{border:1px solid #dfe5ee;border-radius:12px;padding:16px;margin-top:12px}.file-head{display:flex;gap:12px;justify-content:space-between;align-items:start}.filename{font-weight:750;word-break:break-all}.status{font-size:12px;background:#e8f7ee;color:#067647;padding:4px 8px;border-radius:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.grid .full{grid-column:1/-1}.actions{display:flex;gap:8px;margin-top:10px}.empty{text-align:center;color:#667085;padding:30px}.warn{margin-top:14px;color:#9a3412;font-size:13px}@media(max-width:650px){.grid{grid-template-columns:1fr}.grid .full{grid-column:auto}.toolbar{grid-template-columns:1fr}.file-head{display:block}.status{display:inline-block;margin-top:8px}}</style></head><body><main class="wrap">
<section class="card"><h1>Vector Store 파일 업로드</h1><p class="sub">업로드와 파일 관리를 한 화면에서 할 수 있습니다.</p><p class="id">대상: __VECTOR_STORE_ID__</p><input id="password" class="password" type="password" inputmode="numeric" placeholder="관리 비밀번호" required><form id="form"><label class="drop" for="uploadFiles"><b>파일을 선택하세요</b><br><small>한 번에 최대 20개 · 파일당 최대 50MB</small></label><input id="uploadFiles" name="files" type="file" multiple required><div id="selected" class="selected">선택된 파일 없음</div><button id="submit" class="btn wide">업로드</button></form><div id="result" class="result">준비됨</div><p class="warn">삭제하면 Vector Store와 OpenAI 원본 파일에서 모두 제거되며 복구할 수 없습니다.</p></section>
<section class="card"><h2>파일 관리</h2><p class="sub">파일명 검색, 분류와 태그 저장, 삭제가 가능합니다.</p><div class="toolbar"><input id="search" class="search" placeholder="파일명·보험사·분류·태그 검색"><button id="refresh" class="btn light">목록 불러오기</button></div><div id="fileList"><div class="empty">비밀번호를 입력하고 목록 불러오기를 누르세요.</div></div></section>
</main><script>
const uploadFiles=document.querySelector('#uploadFiles'),password=document.querySelector('#password'),selected=document.querySelector('#selected'),form=document.querySelector('#form'),result=document.querySelector('#result'),submit=document.querySelector('#submit'),fileList=document.querySelector('#fileList'),search=document.querySelector('#search'),refresh=document.querySelector('#refresh');let allFiles=[];
const headers=()=>({'X-Upload-Password':password.value});const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const size=n=>n==null?'크기 정보 없음':n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KB':(n/1048576).toFixed(1)+' MB';
async function api(url,opt={}){opt.headers={...headers(),...(opt.headers||{})};let r=await fetch(url,opt),x=await r.json().catch(()=>({detail:'응답을 읽지 못했습니다.'}));if(!r.ok)throw Error(x.detail||JSON.stringify(x));return x}
uploadFiles.onchange=()=>selected.textContent=uploadFiles.files.length?[...uploadFiles.files].map((f,i)=>`${i+1}. ${f.name} (${Math.ceil(f.size/1024)} KB)`).join(String.fromCharCode(10)):'선택된 파일 없음';
form.onsubmit=async e=>{e.preventDefault();submit.disabled=true;result.textContent='업로드 및 인덱싱 중...';try{let x=await api('/upload',{method:'POST',body:new FormData(form)}),ok=x.uploaded.map(f=>`✓ ${f.filename} — ${f.file_id} · ${f.status}`),bad=x.failed.map(f=>`✗ ${f.filename} — ${f.error}`);result.textContent=[...ok,...bad].join(String.fromCharCode(10,10))||'처리된 파일이 없습니다.';await loadFiles()}catch(err){result.textContent='오류: '+err.message}finally{submit.disabled=false}};
function render(){let q=search.value.trim().toLowerCase(),shown=allFiles.filter(f=>(f.filename+' '+Object.values(f.attributes||{}).join(' ')).toLowerCase().includes(q));fileList.innerHTML=shown.length?shown.map(f=>{let a=f.attributes||{};return `<article class="file-card" data-id="${esc(f.file_id)}"><div class="file-head"><div><div class="filename">${esc(f.filename)}</div><div class="id">${esc(f.file_id)} · ${size(f.source_bytes||f.usage_bytes)}</div></div><span class="status">${esc(f.status)}</span></div><div class="grid"><input class="meta company" placeholder="보험사" value="${esc(a.company||'')}"><input class="meta category" placeholder="문서 분류 (약관, 설명서 등)" value="${esc(a.category||'')}"><input class="meta tags full" placeholder="태그 (쉼표로 구분)" value="${esc(a.tags||'')}"><input class="meta note full" placeholder="메모" value="${esc(a.note||'')}"></div><div class="actions"><button class="btn save">분류 저장</button><button class="btn danger delete">완전 삭제</button></div></article>`}).join(''):'<div class="empty">표시할 파일이 없습니다.</div>'}
async function loadFiles(){refresh.disabled=true;fileList.innerHTML='<div class="empty">목록을 불러오는 중...</div>';try{let x=await api('/files');allFiles=x.files;render()}catch(e){fileList.innerHTML=`<div class="empty">오류: ${esc(e.message)}</div>`}finally{refresh.disabled=false}}
refresh.onclick=loadFiles;search.oninput=render;fileList.onclick=async e=>{let card=e.target.closest('.file-card');if(!card)return;let id=card.dataset.id;if(e.target.classList.contains('save')){e.target.disabled=true;try{let attributes={company:card.querySelector('.company').value.trim(),category:card.querySelector('.category').value.trim(),tags:card.querySelector('.tags').value.trim(),note:card.querySelector('.note').value.trim()};Object.keys(attributes).forEach(k=>!attributes[k]&&delete attributes[k]);await api(`/files/${encodeURIComponent(id)}/attributes`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({attributes})});result.textContent='분류 정보를 저장했습니다.';await loadFiles()}catch(err){result.textContent='오류: '+err.message}finally{e.target.disabled=false}}if(e.target.classList.contains('delete')){let f=card.querySelector('.filename').textContent;if(!confirm(`“${f}” 파일을 완전히 삭제할까요?\n삭제 후 복구할 수 없습니다.`))return;e.target.disabled=true;try{await api(`/files/${encodeURIComponent(id)}`,{method:'DELETE'});result.textContent=`${f} 파일을 삭제했습니다.`;await loadFiles()}catch(err){result.textContent='오류: '+err.message;e.target.disabled=false}}};
</script></body></html>""".replace("__VECTOR_STORE_ID__", VECTOR_STORE_ID)


@app.get("/files/manage", response_class=HTMLResponse)
def file_manager() -> str:
    return PAGE


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "templates" / "index.html")
