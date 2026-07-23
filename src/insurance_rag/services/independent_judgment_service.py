from __future__ import annotations

import asyncio
import os

from insurance_rag.core.independent_judgment_runner import (
    generate_single_model_judgment,
)
from insurance_rag.core.independent_judgment_validator import (
    IndependentJudgmentConfigurationError,
)
from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentBatchResult,
    IndependentJudgmentModelConfig,
    IndependentJudgmentRequest,
    ModelJudgmentFailure,
)

MAX_CONCURRENT_JUDGMENTS = 3


def configured_judgment_models() -> list[IndependentJudgmentModelConfig]:
    return [
        IndependentJudgmentModelConfig(model_id=value)
        for name in (
            "INDEPENDENT_JUDGMENT_MODEL_A",
            "INDEPENDENT_JUDGMENT_MODEL_B",
            "INDEPENDENT_JUDGMENT_MODEL_C",
        )
        if (value := os.getenv(name))
    ]


async def generate_independent_judgments(
    request: IndependentJudgmentRequest,
    *,
    model_configs: list[IndependentJudgmentModelConfig] | None = None,
) -> IndependentJudgmentBatchResult:
    configs = [
        config for config in (
            model_configs if model_configs is not None
            else configured_judgment_models()
        )
        if config.enabled
    ][:request.max_judgments]
    if not configs:
        raise IndependentJudgmentConfigurationError(
            "no independent judgment models are configured"
        )
    model_ids = [config.model_id for config in configs]
    if len(model_ids) != len(set(model_ids)):
        raise IndependentJudgmentConfigurationError(
            "independent judgment model IDs must be unique"
        )
    if any(config.provider != "openai" for config in configs):
        raise IndependentJudgmentConfigurationError(
            "only the configured OpenAI provider is currently supported"
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGMENTS)

    async def run(config):
        async with semaphore:
            return await generate_single_model_judgment(
                request, model_config=config
            )

    outputs = await asyncio.gather(
        *(run(config) for config in configs), return_exceptions=True
    )
    judgments = []
    failures = []
    for index, (config, output) in enumerate(zip(configs, outputs), 1):
        if isinstance(output, asyncio.CancelledError):
            raise output
        if isinstance(output, Exception):
            cause = output.__cause__ or output
            failures.append(ModelJudgmentFailure(
                model_id=config.model_id,
                error_type=type(cause).__name__,
                message="model judgment failed after one retry",
                retry_count=1,
            ))
            continue
        prefix = f"{request.sub_question_id}-" if request.sub_question_id else ""
        output.judgment_id = f"{prefix}j{index}"
        judgments.append(output)

    warnings = []
    if len(judgments) < 2:
        warnings.append("insufficient_independent_models")
    return IndependentJudgmentBatchResult(
        original_question=request.original_question,
        sub_question_id=request.sub_question_id,
        judgments=judgments,
        requested_model_count=len(configs),
        completed_model_count=len(judgments),
        failed_model_count=len(failures),
        sufficient_for_comparison=len(judgments) >= 2,
        failures=failures,
        warnings=warnings,
    )
