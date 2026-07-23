from __future__ import annotations

from insurance_rag.schemas.final_answer import (
    FinalInsuranceAnswer,
    InsuranceAnswerPipelineResult,
)
from insurance_rag.schemas.insurance_api import (
    ApiSourceReference,
    InsuranceAnalysisResponse,
    InsuranceSubAnswer,
)
from insurance_rag.schemas.insurance_claim import InsuranceEvidenceClaim


def _source_lookup(
    claims: list[InsuranceEvidenceClaim],
) -> dict[str, InsuranceEvidenceClaim]:
    return {claim.claim_id: claim for claim in claims}


def _api_sources(
    answer: FinalInsuranceAnswer,
    claims: list[InsuranceEvidenceClaim],
) -> list[ApiSourceReference]:
    lookup = _source_lookup(claims)
    output = []
    for source in answer.sources:
        claim = lookup.get(source.claim_id)
        if claim is None:
            continue
        matched = next((
            reference for reference in claim.source_references
            if all(
                getattr(source, field) is None
                or getattr(source, field) == getattr(reference, field)
                for field in (
                    "document_id", "article", "title", "page_start",
                    "page_end", "source_file",
                )
            )
        ), claim.source_references[0])
        output.append(ApiSourceReference(
            claim_id=claim.claim_id,
            document_id=matched.document_id,
            company=matched.company,
            product=matched.product,
            document_type=None,
            document_version=matched.document_version,
            article=matched.article,
            title=matched.title,
            page_start=matched.page_start,
            page_end=matched.page_end,
            source_file=matched.source_file,
        ))
    return output


def to_insurance_analysis_response(
    *,
    request_id: str,
    pipeline_result: InsuranceAnswerPipelineResult,
    processing_time_ms: int,
) -> InsuranceAnalysisResponse:
    evidence = pipeline_result.evidence_pipeline
    final = pipeline_result.final_answer
    status = final.status if final is not None else pipeline_result.status
    if evidence.risk.requires_additional_information and status not in {
        "blocked", "insufficient_evidence", "failed",
    }:
        status = "needs_information"

    claims = evidence.claims.claims if evidence.claims is not None else []
    sub_answers = []
    sub_claims = {
        item.sub_question_id: item.claims.claims
        for item in evidence.sub_question_results
        if item.claims is not None
    }
    for item in pipeline_result.sub_question_results:
        answer = item.final_answer
        sub_answers.append(InsuranceSubAnswer(
            sub_question_id=item.sub_question_id,
            question=item.question,
            status=answer.status,
            answer=answer.answer,
            sources=_api_sources(
                answer, sub_claims.get(item.sub_question_id, [])
            ),
            model_agreement=answer.model_agreement,
            disagreements=[
                disagreement.description
                for disagreement in item.claim_matrix.disagreements
            ],
            requires_human_review=answer.requires_human_review,
            warnings=answer.warnings,
        ))

    return InsuranceAnalysisResponse(
        request_id=request_id,
        status=status,
        answer=final.answer if final else None,
        key_points=final.key_points if final else [],
        applicable_conditions=final.applicable_conditions if final else [],
        important_exceptions=final.important_exceptions if final else [],
        missing_information=(
            final.missing_information
            if final else evidence.risk.missing_information
        ),
        sources=_api_sources(final, claims) if final else [],
        sub_answers=sub_answers,
        model_agreement=final.model_agreement if final else None,
        disagreements=[
            disagreement.description
            for disagreement in (
                pipeline_result.claim_matrix.disagreements
                if pipeline_result.claim_matrix is not None else []
            )
        ],
        requires_disclaimer=(
            final.requires_disclaimer
            if final else evidence.risk.requires_disclaimer
        ),
        requires_human_review=(
            final.requires_human_review
            if final else evidence.risk.requires_human_review
        ),
        stopped_at=pipeline_result.stopped_at,
        reason_code=pipeline_result.reason,
        warnings=pipeline_result.warnings,
        processing_time_ms=processing_time_ms,
    )
