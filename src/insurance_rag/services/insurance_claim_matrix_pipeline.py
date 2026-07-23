from __future__ import annotations

from insurance_rag.schemas.claim_matrix import (
    ClaimMatrixRequest,
    InsuranceClaimMatrixPipelineResult,
)
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services.claim_matrix_service import build_claim_matrix
from insurance_rag.services.insurance_judgment_pipeline import (
    build_independent_insurance_judgments,
)


async def build_insurance_claim_matrices(
    question_request: QuestionRequest,
    *,
    vector_store_id: str | None = None,
    model_configs: list[IndependentJudgmentModelConfig] | None = None,
) -> InsuranceClaimMatrixPipelineResult:
    judgments = await build_independent_insurance_judgments(
        question_request,
        vector_store_id=vector_store_id,
        model_configs=model_configs,
    )
    warnings = list(judgments.warnings)
    if judgments.status in {"blocked", "failed", "insufficient_evidence"}:
        return InsuranceClaimMatrixPipelineResult(
            judgments=judgments, warnings=warnings
        )
    if judgments.evidence.analysis.is_compound:
        results = []
        claims_by_id = {
            item.sub_question_id: item.claims.claims
            for item in judgments.evidence.sub_question_results
            if item.claims is not None
        }
        for batch in judgments.sub_question_results:
            claims = claims_by_id.get(batch.sub_question_id, [])
            if not claims or not batch.judgments:
                continue
            results.append(build_claim_matrix(ClaimMatrixRequest(
                original_question=judgments.evidence.analysis.original_question,
                sub_question_id=batch.sub_question_id,
                claims=claims,
                judgment_batch=batch,
                risk=judgments.evidence.risk,
            )))
        return InsuranceClaimMatrixPipelineResult(
            judgments=judgments,
            sub_question_results=results,
            warnings=warnings,
        )
    batch = judgments.single_result
    claims_result = judgments.evidence.claims
    if (
        batch is None
        or not batch.judgments
        or claims_result is None
        or not claims_result.claims
    ):
        return InsuranceClaimMatrixPipelineResult(
            judgments=judgments, warnings=warnings
        )
    matrix = build_claim_matrix(ClaimMatrixRequest(
        original_question=judgments.evidence.analysis.original_question,
        claims=claims_result.claims,
        judgment_batch=batch,
        risk=judgments.evidence.risk,
    ))
    return InsuranceClaimMatrixPipelineResult(
        judgments=judgments,
        single_result=matrix,
        warnings=list(dict.fromkeys([*warnings, *matrix.warnings])),
    )
