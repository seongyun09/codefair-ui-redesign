from __future__ import annotations

import asyncio
import logging
import os

from insurance_rag.core.insurance_claim_extractor import extract_insurance_claims
from insurance_rag.core.insurance_query_builder import retrieval_request_from_analysis
from insurance_rag.core.insurance_retriever import search_insurance_documents
from insurance_rag.core.question_analyzer import analyze_question
from insurance_rag.core.question_risk_gate import evaluate_question_risk
from insurance_rag.schemas.insurance_claim import InsuranceClaimExtractionRequest
from insurance_rag.schemas.insurance_pipeline import (
    InsuranceEvidencePipelineResult,
    SubQuestionEvidenceResult,
)
from insurance_rag.schemas.question import QuestionAnalysis, QuestionRequest, SubQuestion

logger = logging.getLogger(__name__)


class InsuranceEvidencePipelineError(Exception):
    pass


def _warnings(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(
        value for group in groups for value in group if value
    ))


async def _run_sub_question(
    analysis: QuestionAnalysis,
    sub_question: SubQuestion,
    *,
    vector_store_id: str,
    include_sub_question_id: bool,
) -> SubQuestionEvidenceResult:
    try:
        retrieval_request = retrieval_request_from_analysis(
            analysis,
            normalized_question=sub_question.question,
            requested_action=sub_question.requested_action,
            search_topics=[analysis.main_intent, sub_question.question],
        )
        retrieval = await asyncio.wait_for(
            search_insurance_documents(
                retrieval_request, vector_store_id=vector_store_id
            ),
            timeout=float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "8")),
        )
        if not retrieval.evidence:
            return SubQuestionEvidenceResult(
                sub_question_id=sub_question.id,
                question=sub_question.question,
                retrieval=retrieval,
                status="insufficient_evidence",
                stopped_at="retrieval",
                reason=retrieval.insufficiency_reason or "no_results",
                warnings=retrieval.warnings,
            )
        extraction_request = InsuranceClaimExtractionRequest.from_retrieval(
            retrieval,
            normalized_question=sub_question.question,
            sub_question_id=sub_question.id if include_sub_question_id else None,
            requested_action=sub_question.requested_action,
        )
        claims = await asyncio.wait_for(
            extract_insurance_claims(extraction_request),
            timeout=float(os.getenv("CLAIM_EXTRACTION_TIMEOUT_SECONDS", "12")),
        )
        status = "completed" if claims.sufficient_evidence else "insufficient_evidence"
        return SubQuestionEvidenceResult(
            sub_question_id=sub_question.id,
            question=sub_question.question,
            retrieval=retrieval,
            claims=claims,
            status=status,
            stopped_at=None if status == "completed" else "claim_extraction",
            reason=claims.insufficiency_reason,
            warnings=_warnings(retrieval.warnings, claims.warnings),
        )
    except TimeoutError as exc:
        stage = "claim_extraction" if "extraction_request" in locals() else "retrieval"
        logger.warning(
            "insurance evidence stage timed out",
            extra={"stage": stage, "sub_question_id": sub_question.id},
        )
        return SubQuestionEvidenceResult(
            sub_question_id=sub_question.id,
            question=sub_question.question,
            status="failed",
            stopped_at=stage,
            reason=f"{stage}_timeout",
            warnings=[f"{stage}_timeout"],
        )
    except Exception as exc:
        logger.warning(
            "insurance evidence sub-question failed",
            extra={
                "stage": "retrieval_or_claim_extraction",
                "error_class": type(exc).__name__,
                "sub_question_id": sub_question.id,
            },
        )
        return SubQuestionEvidenceResult(
            sub_question_id=sub_question.id,
            question=sub_question.question,
            status="failed",
            reason=type(exc).__name__,
        )


def _compound_status(
    results: list[SubQuestionEvidenceResult], *, needs_information: bool
) -> str:
    statuses = {item.status for item in results}
    if statuses == {"completed"}:
        return "needs_information" if needs_information else "completed"
    if statuses == {"failed"}:
        return "failed"
    if statuses <= {"insufficient_evidence"}:
        return "insufficient_evidence"
    return "partial"


async def build_insurance_evidence_claims(
    question_request: QuestionRequest,
    *,
    vector_store_id: str | None = None,
) -> InsuranceEvidencePipelineResult:
    """Build verified evidence claims without generating a final answer."""
    try:
        analysis = await analyze_question(question_request)
    except Exception as exc:
        raise InsuranceEvidencePipelineError(
            f"question analysis failed: {type(exc).__name__}"
        ) from exc
    risk = evaluate_question_risk(analysis)
    if not risk.allow_retrieval:
        return InsuranceEvidencePipelineResult(
            status="blocked",
            analysis=analysis,
            risk=risk,
            stopped_at="risk_gate",
            reason=risk.blocked_reason,
            warnings=risk.warnings,
        )

    store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")
    if not store_id:
        raise InsuranceEvidencePipelineError(
            "OPENAI_VECTOR_STORE_ID is not configured"
        )

    if analysis.is_compound:
        results = list(await asyncio.gather(*(
            _run_sub_question(
                analysis, sub_question, vector_store_id=store_id,
                include_sub_question_id=True,
            )
            for sub_question in analysis.sub_questions
        )))
        warnings = _warnings(
            risk.warnings, *(item.warnings for item in results)
        )
        status = _compound_status(
            results, needs_information=risk.requires_additional_information
        )
        return InsuranceEvidencePipelineResult(
            status=status,
            analysis=analysis,
            risk=risk,
            sub_question_results=results,
            reason="one_or_more_sub_questions_failed"
            if status in {"partial", "failed"} else None,
            warnings=warnings,
        )

    result = await _run_sub_question(
        analysis, analysis.sub_questions[0],
        vector_store_id=store_id, include_sub_question_id=False,
    )
    status = result.status
    if status == "completed" and risk.requires_additional_information:
        status = "needs_information"
    return InsuranceEvidencePipelineResult(
        status=status,
        analysis=analysis,
        risk=risk,
        retrieval=result.retrieval,
        claims=result.claims,
        stopped_at=result.stopped_at,
        reason=result.reason,
        warnings=_warnings(risk.warnings, result.warnings),
    )
