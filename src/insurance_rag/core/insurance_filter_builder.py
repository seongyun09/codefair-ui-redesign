from __future__ import annotations

from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalRequest
from insurance_rag.services.insurance_code_registry import InsuranceCodeRegistry


class InsuranceFilterError(Exception):
    pass


def build_insurance_filters(request: InsuranceRetrievalRequest, registry: InsuranceCodeRegistry | None = None) -> tuple[dict[str, object] | None, list[str]]:
    registry = registry or InsuranceCodeRegistry()
    warnings: list[str] = []
    company_code = request.company_code or registry.resolve_company(request.company)
    product_code = request.product_code or registry.resolve_product(request.product)
    if request.company and not company_code:
        warnings.append("unmapped_company_code")
    if request.product and not product_code:
        warnings.append("unmapped_product_code")
    if not company_code:
        warnings.append("missing_company_filter")
    if not product_code:
        warnings.append("missing_product_filter")
    values = {
        "company_code": company_code, "product_code": product_code,
        "product_type": request.product_type, "document_type": request.document_type,
        "document_version": request.document_version,
    }
    filters = [{"type": "eq", "key": key, "value": value} for key, value in values.items() if value]
    if not request.document_version:
        filters.append({"type": "eq", "key": "active", "value": True})
    if not filters:
        return None, warnings
    return (filters[0] if len(filters) == 1 else {"type": "and", "filters": filters}), warnings


def relax_filter(filters: dict[str, object] | None, key: str) -> dict[str, object] | None:
    if not filters:
        return None
    if filters.get("type") != "and":
        return None if filters.get("key") == key else filters
    remaining = [item for item in filters["filters"] if item.get("key") != key]
    if not remaining:
        return None
    return remaining[0] if len(remaining) == 1 else {"type": "and", "filters": remaining}
