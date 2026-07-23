from __future__ import annotations

import re

from insurance_rag.schemas.insurance_retrieval import RetrievedInsuranceEvidence


class InsuranceEvidenceParseError(Exception):
    pass


def _value(text: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}\s*(.*?)\s*$", text)
    return match.group(1).strip() or None if match else None


def parse_insurance_evidence(text: str, *, rank: int, score: float | None = None) -> RetrievedInsuranceEvidence | None:
    if not text or not text.strip():
        return None
    original = re.search(r"(?s)\[원문\]\s*\n(.*?)(?:\n\s*===== DOCUMENT CHUNK END =====|$)", text)
    evidence_text = original.group(1).strip() if original else text.strip()
    if not evidence_text:
        return None
    page = _value(text, "페이지:")
    page_start = page_end = None
    if page and (match := re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", page)):
        page_start = int(match.group(1))
        page_end = int(match.group(2) or match.group(1))
    return RetrievedInsuranceEvidence(
        rank=rank, score=score, chunk_id=_value(text, "chunk_id:"), document_id=_value(text, "document_id:"),
        company=_value(text, "[회사]"), product=_value(text, "[상품]"), document_version=_value(text, "[문서버전]"),
        part=_value(text, "[편/관]"), chapter=_value(text, "[장]"), article=_value(text, "[조항]"), title=_value(text, "[제목]"),
        page_start=page_start, page_end=page_end, text=evidence_text, source_file=_value(text, "파일:"),
    )
