from types import SimpleNamespace

import pytest

from insurance_rag.core.insurance_evidence_parser import parse_insurance_evidence
from insurance_rag.core.insurance_filter_builder import build_insurance_filters
from insurance_rag.core.insurance_query_builder import build_insurance_search_query
from insurance_rag.core.insurance_retriever import InsuranceRetrievalError, search_insurance_documents
from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalRequest


TEXT = """===== DOCUMENT CHUNK START =====
chunk_id: doc-c1
document_id: doc

[회사] 삼성생명
[상품] 유니버설종신보험
[문서버전] 2504
[편/관] 보장형계약 약관
[장] 제2관 보험금의 지급
[조항] 제5조
[제목] 보험금을 지급하지 않는 사유
[출처]
파일: terms.pdf
페이지: 41-42

[원문]
고의로 사고를 일으킨 경우 보험금을 지급하지 않습니다.
===== DOCUMENT CHUNK END ====="""


def request(**changes):
    values = {"original_question": "보험금이 지급되지 않는 경우는?", "normalized_question": "보험금이 지급되지 않는 경우", "company_code": "samsunglife", "product_code": "universal", "document_version": "2504"}
    values.update(changes)
    return InsuranceRetrievalRequest(**values)


class SearchClient:
    def __init__(self, responses, error=False):
        self.responses = list(responses)
        self.calls = []
        self.error = error
        self.vector_stores = SimpleNamespace(search=self.search)

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError("secret upstream response")
        return SimpleNamespace(data=self.responses.pop(0))


def item(score=.9, text=TEXT, chunk_id=None):
    if chunk_id:
        text = text.replace("doc-c1", chunk_id)
    return SimpleNamespace(score=score, content=[SimpleNamespace(text=text)])


def test_query_adds_insurance_synonyms():
    query = build_insurance_search_query(request())
    assert "면책" in query and "지급 제외" in query


def test_filter_contains_known_values_and_omits_version_when_missing():
    filters, warnings = build_insurance_filters(request())
    keys = {entry["key"] for entry in filters["filters"]}
    assert {"company_code", "product_code", "document_type", "document_version"} <= keys
    filters, _ = build_insurance_filters(request(document_version=None))
    keys = {entry["key"] for entry in filters["filters"]}
    assert "document_version" not in keys and "active" in keys


def test_parser_extracts_article_pages_and_original_text():
    evidence = parse_insurance_evidence(TEXT, rank=1, score=.8)
    assert evidence.article == "제5조"
    assert (evidence.page_start, evidence.page_end) == (41, 42)
    assert evidence.source_file == "terms.pdf"


@pytest.mark.asyncio
async def test_search_deduplicates_by_chunk_id_and_keeps_best_score():
    client = SearchClient([[item(.5), item(.9)]])
    result = await search_insurance_documents(request(), vector_store_id="vs-test", client=client)
    assert result.result_count == 1
    assert result.evidence[0].score == .9
    assert result.sufficient_evidence


@pytest.mark.asyncio
async def test_no_results_relaxes_version_then_product_and_reports_warning():
    client = SearchClient([[], [item(.8)]])
    result = await search_insurance_documents(request(), vector_store_id="vs-test", client=client)
    assert result.result_count == 1
    assert "version_filter_relaxed" in result.warnings
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_no_results_and_low_relevance():
    empty = await search_insurance_documents(request(document_version=None, product_code=None), vector_store_id="vs", client=SearchClient([[], []]))
    assert not empty.sufficient_evidence and empty.insufficiency_reason == "no_results"
    low = await search_insurance_documents(request(), vector_store_id="vs", client=SearchClient([[item(.1)]]))
    assert not low.sufficient_evidence and low.insufficiency_reason == "low_relevance"


@pytest.mark.asyncio
async def test_malformed_result_is_ignored_and_openai_error_is_limited():
    malformed = await search_insurance_documents(request(), vector_store_id="vs", client=SearchClient([[SimpleNamespace(score=.8, content=[])], [], []]))
    assert malformed.result_count == 0
    with pytest.raises(InsuranceRetrievalError, match="RuntimeError") as caught:
        await search_insurance_documents(request(), vector_store_id="vs", client=SearchClient([], error=True))
    assert "secret upstream response" not in str(caught.value)
