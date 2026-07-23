from __future__ import annotations

from insurance_rag.schemas.insurance_retrieval import InsuranceRetrievalRequest
from insurance_rag.schemas.question import QuestionAnalysis


INSURANCE_SYNONYMS = {
    "지급하지 않는": ["지급 제외", "면책", "보험금을 지급하지 않는 사유"],
    "지급되지 않는": ["지급 제외", "면책", "보험금을 지급하지 않는 사유"],
    "해지": ["계약 해지", "해약", "효력 상실"],
    "보험료 미납": ["납입 연체", "납입최고", "독촉", "계약 해지"],
    "보험금 청구": ["청구 절차", "구비서류", "지급 절차"],
}


def build_insurance_search_query(request: InsuranceRetrievalRequest) -> str:
    terms = [request.normalized_question, *request.search_topics, *request.keywords]
    if request.requested_action:
        terms.append(request.requested_action)
    joined = " ".join(terms)
    for trigger, synonyms in INSURANCE_SYNONYMS.items():
        if trigger in joined:
            terms.extend(synonyms)
    return " ".join(dict.fromkeys(term.strip() for term in terms if term and term.strip()))


def retrieval_request_from_analysis(analysis: QuestionAnalysis, **overrides) -> InsuranceRetrievalRequest:
    known = {item.name.strip().casefold(): str(item.value) for item in analysis.known_information if item.value is not None}
    lookup = lambda *names: next((known[name.casefold()] for name in names if name.casefold() in known), None)
    values = {
        "original_question": analysis.original_question, "normalized_question": analysis.normalized_question,
        "company": lookup("company", "보험회사", "보험사"), "product": lookup("product", "보험상품", "상품"),
        "product_type": lookup("product_type", "상품유형"), "document_version": lookup("document_version", "문서버전", "가입버전"),
        "requested_action": analysis.sub_questions[0].requested_action if analysis.sub_questions else None,
        "search_topics": [analysis.main_intent, *(item.question for item in analysis.sub_questions)],
    }
    values.update(overrides)
    return InsuranceRetrievalRequest.model_validate(values)
