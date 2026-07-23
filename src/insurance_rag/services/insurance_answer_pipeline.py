from __future__ import annotations

import asyncio

from insurance_rag.schemas.final_answer import (
    FinalAnswerSynthesisRequest,
    InsuranceAnswerPipelineResult,
    SubQuestionFinalAnswer,
)
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.core.final_answer_synthesizer import synthesize_final_answer
from insurance_rag.services.insurance_claim_matrix_pipeline import (
    build_insurance_claim_matrices,
)


async def build_final_insurance_answer(
    question_request: QuestionRequest,
    *,
    vector_store_id: str | None = None,
    model_configs: list[IndependentJudgmentModelConfig] | None = None,
) -> InsuranceAnswerPipelineResult:
    compared = await build_insurance_claim_matrices(
        question_request,
        vector_store_id=vector_store_id,
        model_configs=model_configs,
    )
    judgments = compared.judgments
    evidence = judgments.evidence
    if judgments.status == "blocked":
        return InsuranceAnswerPipelineResult(
            status="blocked",
            evidence_pipeline=evidence,
            stopped_at="risk_gate",
            reason=evidence.reason or evidence.risk.blocked_reason,
            warnings=compared.warnings,
        )
    if judgments.status in {"insufficient_evidence", "failed"}:
        return InsuranceAnswerPipelineResult(
            status=(
                "insufficient_evidence"
                if judgments.status == "insufficient_evidence" else "failed"
            ),
            evidence_pipeline=evidence,
            stopped_at=(
                "claim_extraction"
                if judgments.status == "insufficient_evidence"
                else "independent_judgments"
            ),
            reason=judgments.status,
            warnings=compared.warnings,
        )

    if evidence.analysis.is_compound:
        batches = {
            item.sub_question_id: item for item in judgments.sub_question_results
        }
        matrices = {
            item.sub_question_id: item for item in compared.sub_question_results
        }
        evidence_parts = {
            item.sub_question_id: item for item in evidence.sub_question_results
        }
        questions = {
            item.id: item.question for item in evidence.analysis.sub_questions
        }
        synthesis_requests = []
        warnings = list(compared.warnings)
        for sub_question_id in questions:
            batch = batches.get(sub_question_id)
            matrix = matrices.get(sub_question_id)
            part = evidence_parts.get(sub_question_id)
            if (
                batch is None or not batch.judgments or matrix is None
                or part is None or part.claims is None or not part.claims.claims
            ):
                warnings.append(f"{sub_question_id}:final_answer_skipped")
                continue
            synthesis_requests.append((
                sub_question_id,
                questions[sub_question_id],
                batch,
                matrix,
                FinalAnswerSynthesisRequest(
                original_question=questions[sub_question_id],
                sub_question_id=sub_question_id,
                risk=evidence.risk,
                claims=part.claims.claims,
                judgments=batch,
                claim_matrix=matrix,
                ),
            ))
        answers = await asyncio.gather(*(
            synthesize_final_answer(item[4]) for item in synthesis_requests
        ))
        results = []
        for item, answer in zip(synthesis_requests, answers):
            sub_question_id, question, batch, matrix, _ = item
            results.append(SubQuestionFinalAnswer(
                sub_question_id=sub_question_id,
                question=question,
                independent_judgments=batch,
                claim_matrix=matrix,
                final_answer=answer,
            ))
        if not results:
            status = "failed"
        elif len(results) != len(questions):
            status = "partial"
        elif any(item.final_answer.status != "completed" for item in results):
            status = "limited"
        else:
            status = "completed"
        return InsuranceAnswerPipelineResult(
            status=status,
            evidence_pipeline=evidence,
            sub_question_results=results,
            warnings=list(dict.fromkeys(warnings)),
        )

    batch = judgments.single_result
    matrix = compared.single_result
    claims_result = evidence.claims
    if (
        batch is None or not batch.judgments or matrix is None
        or claims_result is None or not claims_result.claims
    ):
        return InsuranceAnswerPipelineResult(
            status="failed",
            evidence_pipeline=evidence,
            stopped_at="independent_judgments",
            reason="no_completed_judgments",
            warnings=compared.warnings,
        )
    answer = await synthesize_final_answer(FinalAnswerSynthesisRequest(
        original_question=evidence.analysis.original_question,
        risk=evidence.risk,
        claims=claims_result.claims,
        judgments=batch,
        claim_matrix=matrix,
    ))
    return InsuranceAnswerPipelineResult(
        status="completed" if answer.status == "completed" else "limited",
        evidence_pipeline=evidence,
        independent_judgments=batch,
        claim_matrix=matrix,
        final_answer=answer,
        warnings=list(dict.fromkeys([*compared.warnings, *answer.warnings])),
    )
