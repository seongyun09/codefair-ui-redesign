from __future__ import annotations

import asyncio
import argparse
import json
import os
from time import perf_counter

from dotenv import load_dotenv

from insurance_rag.core.insurance_claim_extractor import extract_insurance_claims
from insurance_rag.core.insurance_query_builder import retrieval_request_from_analysis
from insurance_rag.core.insurance_retriever import search_insurance_documents
from insurance_rag.core.question_analyzer import analyze_question
from insurance_rag.core.question_risk_gate import evaluate_question_risk
from insurance_rag.schemas.independent_judgment import IndependentJudgmentRequest
from insurance_rag.schemas.insurance_claim import InsuranceClaimExtractionRequest
from insurance_rag.schemas.insurance_retrieval import (
    InsuranceRetrievalResult,
    RetrievedInsuranceEvidence,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services.independent_judgment_service import (
    generate_independent_judgments,
)

SYNTHETIC_QUESTION = "보험기간 중 사망 시 사망보험금 지급 조건은 무엇인가요?"


async def benchmark(*, use_vector_store: bool = False) -> dict[str, object]:
    load_dotenv(override=False)
    if os.getenv("RUN_OPENAI_SMOKE_TESTS") != "1":
        raise RuntimeError("RUN_OPENAI_SMOKE_TESTS=1 is required")

    timings: dict[str, float] = {}
    total_started = perf_counter()

    started = perf_counter()
    analysis = await analyze_question(QuestionRequest(question=SYNTHETIC_QUESTION))
    timings["question_analysis_seconds"] = perf_counter() - started

    started = perf_counter()
    risk = evaluate_question_risk(analysis)
    timings["risk_gate_seconds"] = perf_counter() - started
    if not risk.allow_retrieval:
        return {
            "status": "blocked",
            "timings": timings,
            "total_seconds": perf_counter() - total_started,
        }

    sub_question = analysis.sub_questions[0]
    started = perf_counter()
    if use_vector_store:
        vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
        if not vector_store_id:
            raise RuntimeError("OPENAI_VECTOR_STORE_ID is required")
        retrieval_request = retrieval_request_from_analysis(
            analysis,
            normalized_question=sub_question.question,
            requested_action=sub_question.requested_action,
            search_topics=[analysis.main_intent, sub_question.question],
        )
        retrieval = await search_insurance_documents(
            retrieval_request, vector_store_id=vector_store_id
        )
        timings["retrieval_seconds"] = perf_counter() - started
        retrieval_mode = "configured_vector_store"
    else:
        synthetic_text = (
            "제3조: 피보험자가 보험기간 중 사망한 경우 "
            "보험수익자에게 사망보험금을 지급한다."
        )
        retrieval = InsuranceRetrievalResult(
            original_question=analysis.original_question,
            search_query=analysis.normalized_question,
            evidence=[RetrievedInsuranceEvidence(
                rank=1,
                document_id="synthetic-terms",
                chunk_id="synthetic-chunk-1",
                article="제3조",
                page_start=1,
                page_end=1,
                text=synthetic_text,
                source_file="synthetic-terms.txt",
            )],
            result_count=1,
            sufficient_evidence=True,
        )
        timings["synthetic_retrieval_setup_seconds"] = perf_counter() - started
        retrieval_mode = "synthetic_evidence_no_vector_store_call"
    if not retrieval.evidence:
        return {
            "status": "insufficient_evidence",
            "retrieval_mode": retrieval_mode,
            "evidence_count": 0,
            "retrieval_insufficiency_reason": retrieval.insufficiency_reason,
            "retrieval_warning_count": len(retrieval.warnings),
            "timings": {
                key: round(value, 3) for key, value in timings.items()
            },
            "total_seconds": round(perf_counter() - total_started, 3),
        }

    extraction_request = InsuranceClaimExtractionRequest.from_retrieval(
        retrieval,
        normalized_question=sub_question.question,
        requested_action=sub_question.requested_action,
    )
    started = perf_counter()
    claims = await extract_insurance_claims(extraction_request)
    timings["claim_extraction_seconds"] = perf_counter() - started
    if not claims.claims:
        return {
            "status": "insufficient_claims",
            "evidence_count": retrieval.result_count,
            "claim_count": 0,
            "timings": timings,
            "total_seconds": perf_counter() - total_started,
        }

    judgment_request = IndependentJudgmentRequest(
        original_question=analysis.original_question,
        normalized_question=analysis.normalized_question,
        requested_action=sub_question.requested_action,
        risk_level=risk.risk_level,
        risk_categories=risk.categories,
        allow_final_answer=risk.allow_final_answer,
        requires_disclaimer=risk.requires_disclaimer,
        requires_human_review=risk.requires_human_review,
        requires_additional_information=risk.requires_additional_information,
        missing_information=risk.missing_information,
        warnings=[*retrieval.warnings, *claims.warnings],
        claims=claims.claims,
        max_judgments=2,
    )
    started = perf_counter()
    judgments = await generate_independent_judgments(judgment_request)
    timings["independent_judgments_seconds"] = perf_counter() - started
    return {
        "status": "completed",
        "retrieval_mode": retrieval_mode,
        "risk_level": risk.risk_level,
        "evidence_count": retrieval.result_count,
        "retrieval_sufficient_evidence": retrieval.sufficient_evidence,
        "retrieval_warning_count": len(retrieval.warnings),
        "claim_count": claims.claim_count,
        "completed_judgment_count": judgments.completed_model_count,
        "failed_judgment_count": judgments.failed_model_count,
        "sufficient_for_comparison": judgments.sufficient_for_comparison,
        "timings": {
            key: round(value, 3) for key, value in timings.items()
        },
        "total_seconds": round(perf_counter() - total_started, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-vector-store",
        action="store_true",
        help="Use the configured private Vector Store instead of synthetic evidence.",
    )
    args = parser.parse_args()
    print(json.dumps(
        asyncio.run(benchmark(use_vector_store=args.use_vector_store)),
        ensure_ascii=False,
        indent=2,
    ))
