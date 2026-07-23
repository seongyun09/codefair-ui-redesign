from __future__ import annotations

import asyncio

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
    IndependentJudgmentPipelineResult,
    IndependentJudgmentRequest,
)
from insurance_rag.schemas.insurance_pipeline import (
    InsuranceEvidencePipelineResult,
    SubQuestionEvidenceResult,
)
from insurance_rag.schemas.question import QuestionRequest
from insurance_rag.services.independent_judgment_service import (
    generate_independent_judgments,
)
from insurance_rag.services.insurance_evidence_pipeline import (
    build_insurance_evidence_claims,
)


def _request_from_evidence(
    evidence: InsuranceEvidencePipelineResult,
    *,
    sub_result: SubQuestionEvidenceResult | None = None,
    max_judgments: int = 3,
) -> IndependentJudgmentRequest:
    claims_result = sub_result.claims if sub_result else evidence.claims
    if claims_result is None or not claims_result.claims:
        raise ValueError("verified claims are required for independent judgments")
    sub_question_id = sub_result.sub_question_id if sub_result else None
    sub_question = next(
        (
            item for item in evidence.analysis.sub_questions
            if item.id == sub_question_id
        ),
        evidence.analysis.sub_questions[0],
    )
    return IndependentJudgmentRequest(
        original_question=evidence.analysis.original_question,
        normalized_question=(
            sub_result.question if sub_result
            else evidence.analysis.normalized_question
        ),
        requested_action=sub_question.requested_action,
        risk_level=evidence.risk.risk_level,
        risk_categories=evidence.risk.categories,
        allow_final_answer=evidence.risk.allow_final_answer,
        requires_disclaimer=evidence.risk.requires_disclaimer,
        requires_human_review=evidence.risk.requires_human_review,
        requires_additional_information=(
            evidence.risk.requires_additional_information
        ),
        missing_information=evidence.risk.missing_information,
        warnings=[*evidence.warnings, *claims_result.warnings],
        claims=claims_result.claims,
        sub_question_id=sub_question_id,
        max_judgments=max_judgments,
    )


async def build_independent_insurance_judgments(
    question_request: QuestionRequest,
    *,
    vector_store_id: str | None = None,
    model_configs: list[IndependentJudgmentModelConfig] | None = None,
) -> IndependentJudgmentPipelineResult:
    """Run independent judgments after evidence extraction; never compare them."""
    evidence = await build_insurance_evidence_claims(
        question_request, vector_store_id=vector_store_id
    )
    if evidence.status == "blocked":
        return IndependentJudgmentPipelineResult(
            status="blocked", evidence=evidence, warnings=evidence.warnings
        )

    max_judgments = max(1, len(model_configs)) if model_configs is not None else 2
    if evidence.status == "failed":
        return IndependentJudgmentPipelineResult(
            status="failed", evidence=evidence, warnings=evidence.warnings
        )
    if evidence.analysis.is_compound:
        requests = []
        warnings = list(evidence.warnings)
        for sub_result in evidence.sub_question_results:
            if (
                sub_result.claims is None
                or not sub_result.claims.sufficient_evidence
                or not sub_result.claims.claims
            ):
                warnings.append(
                    f"{sub_result.sub_question_id}:insufficient_evidence"
                )
                continue
            requests.append(_request_from_evidence(
                evidence, sub_result=sub_result,
                max_judgments=max_judgments,
            ))
        batches = list(await asyncio.gather(*(
            generate_independent_judgments(
                request, model_configs=model_configs
            )
            for request in requests
        )))
        if not batches:
            status = "insufficient_evidence"
        elif len(batches) != len(evidence.sub_question_results):
            status = "partial"
        elif any(batch.failed_model_count for batch in batches):
            status = "partial"
        else:
            status = "completed"
        return IndependentJudgmentPipelineResult(
            status=status,
            evidence=evidence,
            sub_question_results=batches,
            warnings=list(dict.fromkeys(warnings)),
        )

    if (
        evidence.claims is None
        or not evidence.claims.sufficient_evidence
        or not evidence.claims.claims
    ):
        return IndependentJudgmentPipelineResult(
            status="insufficient_evidence",
            evidence=evidence,
            warnings=list(dict.fromkeys([
                *evidence.warnings, "independent_judgment_skipped_no_claims",
            ])),
        )
    request = _request_from_evidence(
        evidence, max_judgments=max_judgments
    )
    batch = await generate_independent_judgments(
        request, model_configs=model_configs
    )
    return IndependentJudgmentPipelineResult(
        status="partial" if batch.failed_model_count else "completed",
        evidence=evidence,
        single_result=batch,
        warnings=list(dict.fromkeys([*evidence.warnings, *batch.warnings])),
    )
