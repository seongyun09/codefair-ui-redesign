from insurance_rag.core.evidence_text_normalizer import (
    normalize_evidence_text,
    quote_occurs_in_evidence,
)


def test_normalizes_whitespace_and_unicode_quotes():
    assert normalize_evidence_text("  “보험금”을   지급한다. ") == '"보험금"을 지급한다.'


def test_pdf_midword_line_break_matches_quote():
    evidence = "보험기간 중 사망한 경우 사망보험\n금을 지급한다."
    assert quote_occurs_in_evidence("사망보험금을 지급한다.", evidence)


def test_absent_quote_does_not_match():
    assert not quote_occurs_in_evidence("보험료를 면제한다.", "보험금을 지급한다.")
