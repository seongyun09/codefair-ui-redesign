import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from insurance_rag import insurance_ingestion as ingestion


def write_document(path: Path, document_id: str, version: str = "v1", count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"chunk_id": f"{document_id}-{i}", "document_id": document_id, "company": f"Company {document_id}",
         "product": f"Product {document_id}", "product_type": "life", "document_type": "terms",
         "document_version": version,
         "location": {"part": "terms", "chapter": "chapter", "article": f"Article {i}", "title": "Title", "page_start": i + 1, "page_end": i + 1},
         "classification": {"category": "benefit", "topic": "payment"},
         "retrieval": {"keywords": ["payment"]},
         "source": {"file_name": path.name, "source_type": "official", "page_start": i + 1, "page_end": i + 1},
         "text": f"content {i}", "embedding_text": f"context content {i}"}
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")


class FakeClient:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.files = SimpleNamespace(create=self.create, delete=self.delete)
        self.vector_stores = SimpleNamespace(files=SimpleNamespace(create_and_poll=self.attach, delete=self.detach))
        self.status = "completed"

    async def create(self, file, purpose):
        item = SimpleNamespace(id=f"file-{len(self.created) + 1}")
        self.created.append(item.id)
        return item

    async def delete(self, file_id):
        self.deleted.append(file_id)

    async def attach(self, **kwargs):
        return SimpleNamespace(status=self.status)

    async def detach(self, **kwargs):
        self.deleted.append(kwargs["file_id"])


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ingestion, "_openai_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_empty_directory(tmp_path, fake_client):
    result = await ingestion.ingest_insurance_documents(tmp_path, vector_store_id="vs-test")
    assert result.discovered_file_count == result.processed_file_count == 0


@pytest.mark.asyncio
async def test_multiple_files_and_incremental_addition(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    for name in ("a", "b", "c"):
        write_document(source / f"{name}.jsonl", name)
    first = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert first.uploaded_file_count == 3
    generated_text = Path(first.uploaded[0].generated_file).read_text(encoding="utf-8")
    assert "===== DOCUMENT CHUNK START =====" in generated_text
    assert "[조항]" in generated_text and "[원문]" in generated_text and "페이지:" in generated_text
    assert not generated_text.lstrip().startswith("{")
    assert first.uploaded[0].attributes["company_code"]
    assert first.uploaded[0].attributes["schema_version"] == "1"
    assert len(json.loads((tmp_path / "data" / "vector_store_manifest.json").read_text())["documents"]) == 3
    write_document(source / "d.txt", "d")
    second = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert second.uploaded_file_count == 1
    assert second.skipped_file_count == 3


@pytest.mark.asyncio
async def test_bad_file_is_isolated_and_duplicate_content_is_skipped(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    write_document(source / "chunks.txt", "a")
    (source / "chunks(1).txt").write_bytes((source / "chunks.txt").read_bytes())
    (source / "bad.jsonl").write_text("not json", encoding="utf-8")
    result = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert result.uploaded_file_count == 1
    assert result.skipped[0].reason == "duplicate_content"
    assert result.failed_file_count == 1


@pytest.mark.asyncio
async def test_changed_document_replaces_attachment_but_versions_coexist(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    path = source / "a.jsonl"
    write_document(path, "a", "2504", 1)
    await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    write_document(path, "a", "2504", 2)
    changed = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert changed.uploaded_file_count == 1
    assert "file-1" in fake_client.deleted
    write_document(source / "a-2510.jsonl", "a", "2510", 1)
    await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    manifest = json.loads((tmp_path / "data" / "vector_store_manifest.json").read_text())
    assert {item["document_version"] for item in manifest["documents"]} == {"2504", "2510"}


@pytest.mark.asyncio
async def test_non_completed_openai_status_is_failed_and_source_file_is_cleaned_up(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    write_document(source / "a.jsonl", "a")
    fake_client.status = "failed"
    result = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert result.uploaded_file_count == 0
    assert result.failed_file_count == 1
    assert fake_client.deleted == ["file-1"]


@pytest.mark.asyncio
async def test_document_id_makes_generated_path_unique(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    write_document(source / "a.jsonl", "a")
    write_document(source / "b.jsonl", "b")
    result = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert len({item.generated_file for item in result.uploaded}) == 2


@pytest.mark.asyncio
async def test_failed_changed_upload_preserves_completed_manifest_entry(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    path = source / "a.jsonl"
    write_document(path, "a", count=1)
    await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    write_document(path, "a", count=2)
    fake_client.status = "failed"
    result = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    manifest = json.loads((tmp_path / "data" / "vector_store_manifest.json").read_text())
    assert result.failed_file_count == 1
    assert manifest["documents"][0]["openai"] == {"file_id": "file-1", "vector_store_id": "vs-test", "status": "completed"}
    assert manifest["failures"][0]["openai"]["status"] == "failed"


@pytest.mark.asyncio
async def test_force_reuploads_and_fail_fast_stops_after_first_error(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    write_document(source / "good.jsonl", "good")
    await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    forced = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test", force=True)
    assert forced.uploaded_file_count == 1

    isolated = tmp_path / "other" / "data" / "source" / "insurance"
    isolated.mkdir(parents=True)
    (isolated / "a-bad.jsonl").write_text("not json", encoding="utf-8")
    write_document(isolated / "b-good.jsonl", "later")
    stopped = await ingestion.ingest_insurance_documents(isolated, vector_store_id="vs-test", fail_fast=True)
    assert stopped.failed_file_count == 1
    assert stopped.uploaded_file_count == 0


def test_manifest_lock_prevents_concurrent_lost_updates(tmp_path):
    path = tmp_path / "manifest.json"

    def add_document(number):
        with ingestion._manifest_lock(path):
            manifest = ingestion._read_manifest(path, "vs-test")
            ingestion._upsert_document(manifest, {"document_id": f"doc-{number}", "document_version": "v1"})
            ingestion._write_manifest(path, manifest)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add_document, range(20)))
    manifest = json.loads(path.read_text())
    assert len(manifest["documents"]) == 20


@pytest.mark.asyncio
async def test_mapping_code_collision_is_rejected(tmp_path, fake_client):
    source = tmp_path / "data" / "source" / "insurance"
    write_document(source / "a.jsonl", "a")
    write_document(source / "b.jsonl", "b")
    config = tmp_path / "data" / "config"
    config.mkdir(parents=True)
    (config / "company_codes.json").write_text(json.dumps({"Company a": "same", "Company b": "same"}), encoding="utf-8")
    result = await ingestion.ingest_insurance_documents(source, vector_store_id="vs-test")
    assert result.uploaded_file_count == 1
    assert result.failed_file_count == 1
    assert "collision" in result.failed[0].message
