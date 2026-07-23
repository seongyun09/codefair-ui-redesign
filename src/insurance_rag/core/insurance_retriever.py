from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from insurance_rag.core.insurance_evidence_parser import parse_insurance_evidence
from insurance_rag.core.insurance_filter_builder import build_insurance_filters, relax_filter
from insurance_rag.core.insurance_query_builder import build_insurance_search_query
from insurance_rag.core.llm_client import create_llm_client
from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalRequest, InsuranceRetrievalResult, RetrievedInsuranceEvidence
from insurance_rag.services.insurance_code_registry import InsuranceCodeRegistry

logger = logging.getLogger(__name__)
MIN_RELEVANCE_SCORE = 0.35


class InsuranceRetrievalError(Exception):
    pass


def _content_text(item: Any) -> str:
    blocks = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
    values = []
    for block in blocks:
        value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if value:
            values.append(value)
    return "\n".join(values)


def _item_value(item: Any, key: str, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _duplicate(left: RetrievedInsuranceEvidence, right: RetrievedInsuranceEvidence) -> bool:
    if left.chunk_id and left.chunk_id == right.chunk_id:
        return True
    if left.document_id and left.document_id == right.document_id and left.article and left.article == right.article and (left.page_start, left.page_end) == (right.page_start, right.page_end):
        return True
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    a, b = normalize(left.text), normalize(right.text)
    return bool(a and b and SequenceMatcher(None, a, b).ratio() >= 0.96)


def _deduplicate(evidence: list[RetrievedInsuranceEvidence], top_k: int) -> list[RetrievedInsuranceEvidence]:
    ordered = sorted(evidence, key=lambda item: item.score if item.score is not None else -1, reverse=True)
    kept: list[RetrievedInsuranceEvidence] = []
    for item in ordered:
        if not any(_duplicate(item, previous) for previous in kept):
            kept.append(item)
    for rank, item in enumerate(kept[:top_k], 1):
        item.rank = rank
    return kept[:top_k]


async def _search(client, vector_store_id: str, query: str, filters: dict[str, object] | None, top_k: int):
    last_error = None
    for attempt in range(2):
        try:
            kwargs = {"vector_store_id": vector_store_id, "query": query, "max_num_results": top_k}
            if filters:
                kwargs["filters"] = filters
            return await client.vector_stores.search(**kwargs)
        except Exception as exc:
            last_error = exc
            logger.warning("insurance retrieval attempt failed", extra={"stage": "vector_store_search", "vector_store_id": vector_store_id, "filter_keys": _filter_keys(filters), "retry_count": attempt, "error_class": type(exc).__name__})
            if attempt == 0:
                await asyncio.sleep(0)
    raise InsuranceRetrievalError(f"vector store search failed after retry: {type(last_error).__name__}") from last_error


def _filter_keys(filters: dict[str, object] | None) -> list[str]:
    if not filters:
        return []
    if filters.get("type") == "and":
        return [str(item.get("key")) for item in filters.get("filters", [])]
    return [str(filters.get("key"))]


async def search_insurance_documents(request: InsuranceRetrievalRequest, *, vector_store_id: str, client=None, registry: InsuranceCodeRegistry | None = None) -> InsuranceRetrievalResult:
    query = build_insurance_search_query(request)
    filters, warnings = build_insurance_filters(request, registry)
    client = client or create_llm_client()
    applied = filters
    page = await _search(client, vector_store_id, query, filters, request.top_k)
    raw = list(getattr(page, "data", page if isinstance(page, list) else []))
    if not raw and request.document_version:
        filters = relax_filter(filters, "document_version")
        warnings.append("version_filter_relaxed")
        page = await _search(client, vector_store_id, query, filters, request.top_k)
        raw = list(getattr(page, "data", page if isinstance(page, list) else []))
        applied = filters
    if not raw and request.product_code or (not raw and request.product):
        relaxed = relax_filter(filters, "product_code")
        if relaxed != filters:
            filters = relaxed
            warnings.append("product_filter_relaxed")
            page = await _search(client, vector_store_id, query, filters, request.top_k)
            raw = list(getattr(page, "data", page if isinstance(page, list) else []))
            applied = filters
    parsed = []
    for index, item in enumerate(raw, 1):
        evidence = parse_insurance_evidence(_content_text(item), rank=index, score=_item_value(item, "score"))
        if evidence:
            parsed.append(evidence)
    evidence = _deduplicate(parsed, request.top_k)
    sufficient = bool(evidence) and max((item.score or 0) for item in evidence) >= MIN_RELEVANCE_SCORE
    reason = None if sufficient else ("no_results" if not evidence else "low_relevance")
    logger.info("insurance retrieval completed", extra={"stage": "complete", "vector_store_id": vector_store_id, "filter_keys": _filter_keys(applied), "result_count": len(evidence), "retry_count": 0})
    return InsuranceRetrievalResult(original_question=request.original_question, search_query=query, applied_filters=applied, evidence=evidence, result_count=len(evidence), sufficient_evidence=sufficient, insufficiency_reason=reason, warnings=list(dict.fromkeys(warnings)))
