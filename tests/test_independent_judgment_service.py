import asyncio

import pytest

from insurance_rag.schemas.independent_judgment import (
    IndependentJudgmentModelConfig,
)
from insurance_rag.services import independent_judgment_service as service
from test_independent_judgment_schema import judgment, request


def configs(count=2):
    return [
        IndependentJudgmentModelConfig(model_id=f"model-{index}")
        for index in range(1, count + 1)
    ]


@pytest.mark.asyncio
async def test_models_start_independently_and_results_are_collected(monkeypatch):
    started = []
    all_started = asyncio.Event()
    request_snapshots = []

    async def fake_generate(value, *, model_config):
        started.append(model_config.model_id)
        request_snapshots.append(value.model_dump_json())
        if len(started) == 2:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return judgment(model_id=model_config.model_id)

    monkeypatch.setattr(service, "generate_single_model_judgment", fake_generate)
    result = await service.generate_independent_judgments(
        request(), model_configs=configs()
    )
    assert started == ["model-1", "model-2"]
    assert request_snapshots[0] == request_snapshots[1]
    assert [item.judgment_id for item in result.judgments] == ["j1", "j2"]
    assert result.sufficient_for_comparison


@pytest.mark.asyncio
async def test_one_model_failure_keeps_success(monkeypatch):
    async def fake_generate(value, *, model_config):
        if model_config.model_id == "model-2":
            raise RuntimeError("private upstream output")
        return judgment(model_id=model_config.model_id)

    monkeypatch.setattr(service, "generate_single_model_judgment", fake_generate)
    result = await service.generate_independent_judgments(
        request(), model_configs=configs()
    )
    assert result.completed_model_count == 1
    assert result.failed_model_count == 1
    assert result.judgments[0].judgment_id == "j1"
    assert not result.sufficient_for_comparison
    assert "private upstream output" not in result.failures[0].message
    assert "insufficient_independent_models" in result.warnings


@pytest.mark.asyncio
async def test_one_configured_model_is_not_sufficient_for_comparison(monkeypatch):
    async def fake_generate(value, *, model_config):
        return judgment(model_id=model_config.model_id)

    monkeypatch.setattr(service, "generate_single_model_judgment", fake_generate)
    result = await service.generate_independent_judgments(
        request(sub_question_id="q1"), model_configs=configs(1)
    )
    assert result.judgments[0].judgment_id == "q1-j1"
    assert not result.sufficient_for_comparison


@pytest.mark.asyncio
async def test_missing_or_duplicate_model_configuration_is_rejected():
    with pytest.raises(service.IndependentJudgmentConfigurationError):
        await service.generate_independent_judgments(
            request(), model_configs=[]
        )
    with pytest.raises(service.IndependentJudgmentConfigurationError):
        await service.generate_independent_judgments(
            request(), model_configs=[configs(1)[0], configs(1)[0]]
        )
