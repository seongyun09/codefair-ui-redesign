from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize_evidence_text(value: str) -> str:
    """Normalize PDF presentation artifacts without changing wording."""
    value = unicodedata.normalize("NFKC", value).translate(_QUOTES)
    # A line break in the middle of a word is a common PDF extraction artifact.
    value = re.sub(r"(?<=[0-9A-Za-z가-힣])\s*\r?\n\s*(?=[0-9A-Za-z가-힣])", "", value)
    return re.sub(r"\s+", " ", value).strip()


def quote_occurs_in_evidence(quote: str, evidence_text: str) -> bool:
    normalized_quote = normalize_evidence_text(quote)
    normalized_evidence = normalize_evidence_text(evidence_text)
    if normalized_quote in normalized_evidence:
        return True
    # PDF line wrapping can be ambiguous in Korean, where spaces are not part of
    # the underlying token. This remains exact after whitespace removal.
    compact = lambda text: re.sub(r"\s+", "", text)
    return bool(normalized_quote) and compact(normalized_quote) in compact(normalized_evidence)
