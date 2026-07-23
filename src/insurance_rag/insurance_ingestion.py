from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from insurance_rag.core.llm_client import create_llm_client


MAX_CHUNK_CHARS = 6000
TARGET_CHUNK_CHARS = 3500
CHUNK_OVERLAP_CHARS = 300


class PageLocation(BaseModel):
    model_config = ConfigDict(extra="allow")
    part: str | None = None
    chapter: str | None = None
    article: str | None = None
    title: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_range(self) -> PageLocation:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class ChunkClassification(BaseModel):
    model_config = ConfigDict(extra="allow")
    domain: str | None = None
    category: str | None = None
    topic: str | None = None
    subtopics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    legal_effects: list[str] = Field(default_factory=list)


class ChunkRetrieval(BaseModel):
    model_config = ConfigDict(extra="allow")
    keywords: list[str] = Field(default_factory=list)
    query_aliases: list[str] = Field(default_factory=list)
    importance: str | None = None


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_name: str
    source_type: str = "official"
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_url: str | None = None

    @model_validator(mode="after")
    def valid_range(self) -> SourceReference:
        if self.page_end < self.page_start:
            raise ValueError("source.page_end must be greater than or equal to source.page_start")
        return self


class InsuranceChunk(BaseModel):
    """Minimum contract for a source JSONL record."""

    model_config = ConfigDict(extra="allow")
    chunk_id: str
    document_id: str
    company: str
    product: str
    product_type: str
    document_type: str
    document_version: str
    location: PageLocation
    classification: ChunkClassification = Field(default_factory=ChunkClassification)
    retrieval: ChunkRetrieval = Field(default_factory=ChunkRetrieval)
    source: SourceReference
    text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)

    @field_validator("chunk_id", "document_id", "company", "product", "product_type", "document_type", "document_version", "text", "embedding_text")
    @classmethod
    def strip_required(cls, value: str) -> str:
        if not (normalized := value.strip()):
            raise ValueError("required string must not be blank")
        return normalized

    @model_validator(mode="after")
    def pages_match(self) -> InsuranceChunk:
        if (self.location.page_start, self.location.page_end) != (self.source.page_start, self.source.page_end):
            raise ValueError("source and location page ranges must match")
        return self


InsuranceDocumentChunk = InsuranceChunk


class VectorStoreFileAttributes(BaseModel):
    model_config = ConfigDict(extra="allow")
    document_id: str | None = None
    company: str | None = None
    product: str | None = None
    product_type: str | None = None
    document_type: str | None = None
    document_version: str | None = None
    company_code: str | None = None
    product_code: str | None = None
    source_type: str | None = None
    language: str | None = None
    schema_version: str | None = None
    active: bool | None = None
    effective_from: int | None = None
    effective_to: int | None = None

    @model_validator(mode="after")
    def scalar_values_only(self) -> VectorStoreFileAttributes:
        for key, value in self.model_dump(exclude_none=True).items():
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"attribute {key!r} must be a scalar value")
        return self

    def for_openai(self) -> dict[str, str | int | bool]:
        return {k: v for k, v in self.model_dump(exclude_none=True).items()}


class DocumentStatistics(BaseModel):
    source_chunk_count: int
    generated_chunk_count: int
    split_chunk_count: int = 0
    invalid_record_count: int = 0
    duplicate_chunk_count: int = 0
    source_size_bytes: int
    generated_size_bytes: int


class VectorStoreUploadResult(BaseModel):
    source_file: str
    generated_file: str
    document_id: str
    document_version: str
    source_sha256: str
    generated_sha256: str
    file_id: str
    vector_store_id: str
    status: str
    attributes: dict[str, str | int | bool]
    statistics: DocumentStatistics


class SkippedDocumentResult(BaseModel):
    source_file: str
    document_id: str | None = None
    reason: Literal["duplicate_content", "already_completed", "unsupported_extension", "excluded_management_file"]
    source_sha256: str


class FailedDocumentResult(BaseModel):
    source_file: str
    document_id: str | None = None
    stage: str
    error_type: str
    message: str


class BatchVectorStoreUploadResult(BaseModel):
    source_directory: str
    discovered_file_count: int
    processed_file_count: int
    uploaded_file_count: int
    skipped_file_count: int
    failed_file_count: int
    uploaded: list[VectorStoreUploadResult] = Field(default_factory=list)
    skipped: list[SkippedDocumentResult] = Field(default_factory=list)
    failed: list[FailedDocumentResult] = Field(default_factory=list)


IDENTITY_FIELDS = (
    "document_id", "company", "product", "product_type", "document_type", "document_version"
)
EXCLUDED_NAMES = {"vector_store_manifest.json", "document.json"}
GENERATED_DIRECTORY_NAME = "vector_store_ready"
MANIFEST_NAME = "vector_store_manifest.json"
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def calculate_sha256(file_path: str | Path) -> str:
    path = Path(file_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_sha256 = calculate_sha256


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _is_excluded(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in EXCLUDED_NAMES
        or name.endswith((".tmp", ".bak", "~"))
        or GENERATED_DIRECTORY_NAME in (part.lower() for part in path.parts)
        or "search_ready" in name
        or "vector_store_ready" in name
    )


def discover_insurance_source_files(source_directory: str | Path) -> list[Path]:
    root = Path(source_directory)
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    files = (
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jsonl", ".txt"}
        and not _is_hidden(path, root)
        and not _is_excluded(path)
    )
    return sorted(files, key=lambda path: path.as_posix().casefold())


def load_insurance_chunks(file_path: str | Path) -> list[InsuranceDocumentChunk]:
    return _read_chunks(Path(file_path))[0]


def _read_chunks(path: Path) -> tuple[list[InsuranceChunk], int, int]:
    chunks: list[InsuranceChunk] = []
    seen: set[str] = set()
    duplicates = 0
    invalid = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                chunk = InsuranceChunk.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                invalid += 1
                raise ValueError(f"line {line_number}: invalid_record: {exc}") from exc
            if chunk.chunk_id in seen:
                duplicates += 1
                raise ValueError(f"line {line_number}: duplicate chunk_id {chunk.chunk_id!r}")
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    if not chunks:
        raise ValueError("source file contains no JSON records")
    first = chunks[0]
    expected = tuple(getattr(first, field) for field in IDENTITY_FIELDS) + (first.source.file_name,)
    for index, chunk in enumerate(chunks[1:], 2):
        actual = tuple(getattr(chunk, field) for field in IDENTITY_FIELDS) + (chunk.source.file_name,)
        if actual != expected:
            raise ValueError(f"line {index}: mixed_document_identity")
    return chunks, invalid, duplicates


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if ascii_slug:
        return ascii_slug
    # Preserve determinism for scripts that cannot be transliterated without an extra dependency.
    return "u-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"mapping must be a JSON string map: {path}")
    return value


def _data_root(source: Path) -> Path:
    for parent in (source.parent, *source.parents):
        if parent.name == "source" and parent.parent.name == "data":
            return parent.parent
    return Path(os.getenv("INSURANCE_DATA_DIRECTORY", "data"))


def _manifest_path(source: Path) -> Path:
    configured = os.getenv("INSURANCE_MANIFEST_PATH")
    return Path(configured) if configured else _data_root(source) / MANIFEST_NAME


def _read_manifest(path: Path, vector_store_id: str) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.setdefault("documents", [])
        return manifest
    return {"schema_version": "1.0", "vector_store": {"name": "insurance", "vector_store_id": vector_store_id}, "documents": []}


@contextmanager
def _manifest_lock(path: Path):
    """Serialize manifest read-modify-write across threads and processes."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock, lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _upsert_document(manifest: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any] | None:
    documents = manifest.setdefault("documents", [])
    for index, current in enumerate(documents):
        if current.get("document_id") == entry["document_id"] and current.get("document_version") == entry["document_version"]:
            documents[index] = entry
            return current
    documents.append(entry)
    return None


def _same_document(manifest: dict[str, Any], document_id: str, version: str) -> dict[str, Any] | None:
    return next((item for item in manifest.get("documents", []) if item.get("document_id") == document_id and item.get("document_version") == version), None)


def _check_code_collision(manifest: dict[str, Any], *, field: str, value_field: str, code: str, value: str) -> None:
    for item in manifest.get("documents", []):
        if item.get(field) == code and item.get(value_field) not in (None, value):
            raise ValueError(f"{field} collision: {code!r} represents both {item.get(value_field)!r} and {value!r}")


_SEMANTIC_BOUNDARY = re.compile(r"(?m)(?=^(?:\d+[.)]|[①-⑳]|[가-힣][.)]|[⑴-⒇])\s*)")


def _split_text(value: str) -> list[str]:
    if len(value) <= MAX_CHUNK_CHARS:
        return [value]
    units = [unit.strip() for unit in _SEMANTIC_BOUNDARY.split(value) if unit.strip()]
    if len(units) == 1:
        units = [unit.strip() for unit in re.split(r"\n\s*\n", value) if unit.strip()]
    if len(units) == 1:
        units = [unit.strip() for unit in re.split(r"(?<=[.!?。])\s+", value) if unit.strip()]
    parts: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > MAX_CHUNK_CHARS:
            if current:
                parts.append(current)
                current = ""
            start = 0
            while start < len(unit):
                end = min(start + TARGET_CHUNK_CHARS, len(unit))
                if end < len(unit):
                    boundary = max(unit.rfind(" ", start, end), unit.rfind("\n", start, end))
                    if boundary > start:
                        end = boundary
                parts.append(unit[start:end].strip())
                start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
            continue
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > TARGET_CHUNK_CHARS:
            parts.append(current)
            current = f"{current[-CHUNK_OVERLAP_CHARS:].lstrip()}\n\n{unit}".strip()
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_insurance_chunk(chunk: InsuranceDocumentChunk) -> list[InsuranceDocumentChunk]:
    basis = chunk.embedding_text if len(chunk.embedding_text) >= len(chunk.text) else chunk.text
    segments = _split_text(basis)
    if len(segments) == 1:
        return [chunk]
    return [chunk.model_copy(update={"chunk_id": f"{chunk.chunk_id}-p{index:03d}", "text": segment, "embedding_text": segment}) for index, segment in enumerate(segments, 1)]


def _render_chunk(chunk: InsuranceDocumentChunk) -> str:
    location, classification = chunk.location, chunk.classification
    keywords = [*chunk.retrieval.keywords, *chunk.retrieval.query_aliases]
    return "\n".join([
        "===== DOCUMENT CHUNK START =====", "", f"chunk_id: {chunk.chunk_id}", f"document_id: {chunk.document_id}", "",
        f"[회사] {chunk.company}", f"[상품] {chunk.product}", f"[상품유형] {chunk.product_type}",
        f"[문서유형] {chunk.document_type}", f"[문서버전] {chunk.document_version}", f"[편/관] {location.part or ''}",
        f"[장] {location.chapter or ''}", f"[조항] {location.article or ''}", f"[제목] {location.title or ''}",
        f"[분류] {classification.category or ''}", f"[주제] {classification.topic or ''}",
        f"[키워드] {', '.join(dict.fromkeys(value for value in keywords if value.strip()))}", "", "[출처]",
        f"파일: {chunk.source.file_name}", f"페이지: {chunk.source.page_start}" + (f"-{chunk.source.page_end}" if chunk.source.page_end != chunk.source.page_start else ""),
        "", "[검색 문맥]", chunk.embedding_text, "", "[원문]", chunk.text, "", "===== DOCUMENT CHUNK END =====",
    ])


def _openai_client():
    return create_llm_client()


async def upload_insurance_document(
    source_path: str | Path,
    *,
    vector_store_id: str,
    attributes: VectorStoreFileAttributes | None = None,
    force: bool = False,
) -> VectorStoreUploadResult:
    source = Path(source_path)
    if source.suffix.lower() not in {".jsonl", ".txt"}:
        raise ValueError("only .jsonl and JSONL-formatted .txt files are supported")
    source_hash = _sha256(source)
    chunks, invalid_count, duplicate_count = _read_chunks(source)
    identity = chunks[0]
    manifest_path = _manifest_path(source)
    with _manifest_lock(manifest_path):
        manifest = _read_manifest(manifest_path, vector_store_id)
    existing = next((item for item in manifest["documents"] if item.get("source_sha256") == source_hash and item.get("openai", {}).get("status") == "completed"), None)
    if existing and not force:
        raise FileExistsError("already_completed")

    data_root = _data_root(source)
    company_codes = _load_mapping(data_root / "config" / "company_codes.json")
    product_codes = _load_mapping(data_root / "config" / "product_codes.json")
    company_code = company_codes.get(identity.company) or _slug(identity.company)
    product_code = product_codes.get(identity.product) or _slug(identity.product)
    document_code = _slug(identity.document_type)
    _check_code_collision(manifest, field="company_code", value_field="company", code=company_code, value=identity.company)
    _check_code_collision(manifest, field="product_code", value_field="product", code=product_code, value=identity.product)
    document_id_code = _slug(identity.document_id)
    generated = data_root / GENERATED_DIRECTORY_NAME / f"{company_code}__{product_code}__{document_code}__{_slug(identity.document_version)}__{document_id_code}.txt"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated_chunks = [part for chunk in chunks for part in split_insurance_chunk(chunk)]
    sections = [_render_chunk(chunk) for chunk in generated_chunks]
    generated.write_text("\n\n---\n\n".join(sections) + "\n", encoding="utf-8")
    generated_hash = _sha256(generated)
    attribute_values = attributes.model_dump(exclude_none=True) if attributes else {}
    attribute_values.update({
        "document_id": identity.document_id, "company": identity.company, "product": identity.product,
        "product_type": identity.product_type, "document_type": identity.document_type,
        "document_version": identity.document_version, "company_code": company_code, "product_code": product_code,
        "language": "ko", "schema_version": "1",
    })
    source_types = {chunk.source.source_type for chunk in chunks}
    if len(source_types) == 1:
        attribute_values["source_type"] = source_types.pop()
    attribute_values.setdefault("active", True)
    final_attributes = VectorStoreFileAttributes.model_validate(attribute_values).for_openai()
    client = _openai_client()
    with generated.open("rb") as stream:
        created = await client.files.create(file=stream, purpose="assistants")
    try:
        attached = await client.vector_stores.files.create_and_poll(
            vector_store_id=vector_store_id, file_id=created.id, attributes=final_attributes
        )
    except Exception:
        await client.files.delete(created.id)
        raise
    status = str(attached.status)
    if status != "completed":
        try:
            await client.files.delete(created.id)
        finally:
            raise RuntimeError(f"vector store indexing did not complete: status={status}")
    stats = DocumentStatistics(
        source_chunk_count=len(chunks), generated_chunk_count=len(generated_chunks), split_chunk_count=len(generated_chunks) - len(chunks),
        invalid_record_count=invalid_count, duplicate_chunk_count=duplicate_count,
        source_size_bytes=source.stat().st_size, generated_size_bytes=generated.stat().st_size,
    )
    entry = {
        "document_id": identity.document_id, "document_version": identity.document_version,
        "company": identity.company, "product": identity.product, "product_type": identity.product_type,
        "document_type": identity.document_type, "source_file": str(source), "source_sha256": source_hash,
        "generated_file": str(generated), "generated_sha256": generated_hash,
        "company_code": company_code, "product_code": product_code,
        "codes_auto_generated": {"company": identity.company not in company_codes, "product": identity.product not in product_codes},
        "statistics": stats.model_dump(), "source_chunk_count": stats.source_chunk_count,
        "generated_chunk_count": stats.generated_chunk_count, "split_chunk_count": stats.split_chunk_count,
        "attributes": final_attributes, "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "openai": {"file_id": created.id, "vector_store_id": vector_store_id, "status": status},
    }
    with _manifest_lock(manifest_path):
        manifest = _read_manifest(manifest_path, vector_store_id)
        _check_code_collision(manifest, field="company_code", value_field="company", code=company_code, value=identity.company)
        _check_code_collision(manifest, field="product_code", value_field="product", code=product_code, value=identity.product)
        previous = _upsert_document(manifest, entry)
        manifest["vector_store"]["vector_store_id"] = vector_store_id
        _write_manifest(manifest_path, manifest)
    if previous and previous.get("openai", {}).get("file_id") != created.id and status == "completed":
        old_id = previous["openai"].get("file_id")
        if old_id:
            try:
                await client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=old_id)
            except Exception:
                pass
    return VectorStoreUploadResult(
        source_file=str(source), generated_file=str(generated), document_id=identity.document_id,
        document_version=identity.document_version, source_sha256=source_hash,
        generated_sha256=generated_hash, file_id=created.id, vector_store_id=vector_store_id,
        status=status, attributes=final_attributes, statistics=stats,
    )


async def ingest_insurance_documents(
    source_directory: str | Path,
    *,
    vector_store_id: str,
    force: bool = False,
    fail_fast: bool = False,
) -> BatchVectorStoreUploadResult:
    root = Path(source_directory)
    files = discover_insurance_source_files(root)
    result = BatchVectorStoreUploadResult(source_directory=str(root), discovered_file_count=len(files), processed_file_count=0, uploaded_file_count=0, skipped_file_count=0, failed_file_count=0)
    seen_hashes: set[str] = set()
    for source in files:
        source_hash = _sha256(source)
        if source_hash in seen_hashes:
            result.skipped.append(SkippedDocumentResult(source_file=str(source), reason="duplicate_content", source_sha256=source_hash))
            continue
        seen_hashes.add(source_hash)
        try:
            uploaded = await upload_insurance_document(source, vector_store_id=vector_store_id, force=force)
            result.uploaded.append(uploaded)
        except FileExistsError as exc:
            document_id = None
            try:
                document_id = _read_chunks(source)[0][0].document_id
            except Exception:
                pass
            result.skipped.append(SkippedDocumentResult(source_file=str(source), document_id=document_id, reason="already_completed", source_sha256=source_hash))
        except Exception as exc:
            document_id = None
            identity = None
            try:
                identity = _read_chunks(source)[0][0]
                document_id = identity.document_id
            except Exception:
                pass
            result.failed.append(FailedDocumentResult(source_file=str(source), document_id=document_id, stage="validation_or_upload", error_type=type(exc).__name__, message=str(exc)[:1000]))
            manifest_path = _manifest_path(source)
            failure = {"source_file": str(source), "source_sha256": source_hash, "openai": {"status": "failed"}, "error_type": type(exc).__name__, "message": str(exc)[:1000]}
            with _manifest_lock(manifest_path):
                manifest = _read_manifest(manifest_path, vector_store_id)
                if identity:
                    failure.update({"document_id": identity.document_id, "document_version": identity.document_version})
                    current = _same_document(manifest, identity.document_id, identity.document_version)
                    if current is None:
                        _upsert_document(manifest, failure)
                    else:
                        manifest.setdefault("failures", []).append(failure)
                else:
                    manifest.setdefault("failures", []).append(failure)
                _write_manifest(manifest_path, manifest)
            if fail_fast:
                break
    result.uploaded_file_count = len(result.uploaded)
    result.skipped_file_count = len(result.skipped)
    result.failed_file_count = len(result.failed)
    result.processed_file_count = result.uploaded_file_count + result.failed_file_count
    return result
