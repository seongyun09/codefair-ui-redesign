import json

import pytest

from insurance_rag.insurance_ingestion import InsuranceDocumentChunk, load_insurance_chunks, split_insurance_chunk


def record(**changes):
    value = {
        "chunk_id": "doc-c001", "document_id": "doc", "company": "회사", "product": "상품",
        "product_type": "종신보험", "document_type": "insurance_terms", "document_version": "2504",
        "location": {"part": "약관", "chapter": "제1장", "article": "제5조", "title": "면책", "page_start": 10, "page_end": 10},
        "classification": {"category": "보험금 지급", "topic": "면책"},
        "retrieval": {"keywords": ["면책"], "query_aliases": ["지급 제외"]},
        "source": {"file_name": "terms.pdf", "source_type": "official", "page_start": 10, "page_end": 10},
        "text": "약관 원문", "embedding_text": "면책 검색 문맥 약관 원문",
    }
    value.update(changes)
    return value


def test_loader_reports_line_and_required_fields(tmp_path):
    path = tmp_path / "bad.txt"
    bad = record()
    del bad["embedding_text"]
    path.write_text(json.dumps(record()) + "\n" + json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?s)line 2.*embedding_text"):
        load_insurance_chunks(path)


@pytest.mark.parametrize("change", [
    {"location": {"page_start": 11, "page_end": 10}},
    {"source": {"file_name": "terms.pdf", "page_start": 9, "page_end": 9}},
])
def test_page_validation(change):
    with pytest.raises(ValueError):
        InsuranceDocumentChunk.model_validate(record(**change))


def test_mixed_identity_uses_stable_error_code(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(record()) + "\n" + json.dumps(record(chunk_id="other", company="다른회사")), encoding="utf-8")
    with pytest.raises(ValueError, match="mixed_document_identity"):
        load_insurance_chunks(path)


def test_long_chunk_splits_on_numbered_units_and_short_chunk_does_not():
    short = InsuranceDocumentChunk.model_validate(record())
    assert split_insurance_chunk(short) == [short]
    long_text = "\n".join(f"{number}. " + ("독립 보장 내용입니다. " * 250) for number in range(1, 4))
    long = InsuranceDocumentChunk.model_validate(record(text=long_text, embedding_text=long_text))
    parts = split_insurance_chunk(long)
    assert len(parts) > 1
    assert [item.chunk_id for item in parts] == [f"doc-c001-p{i:03d}" for i in range(1, len(parts) + 1)]
