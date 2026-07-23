from __future__ import annotations

import os

from openai import AsyncOpenAI
from dotenv import load_dotenv


def _load_local_environment() -> None:
    # A project-local .env is used for local development only; hosted
    # environments normally do not contain this ignored file.
    load_dotenv(override=True)


def create_llm_client() -> AsyncOpenAI:
    """Create the shared async LLM client from environment configuration."""
    _load_local_environment()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return AsyncOpenAI(api_key=api_key)


def question_analyzer_model() -> str:
    _load_local_environment()
    model = os.getenv("QUESTION_ANALYZER_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("QUESTION_ANALYZER_MODEL or OPENAI_MODEL is not configured")
    return model


def insurance_claim_extractor_model() -> str:
    _load_local_environment()
    model = os.getenv("INSURANCE_CLAIM_EXTRACTOR_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError(
            "INSURANCE_CLAIM_EXTRACTOR_MODEL or OPENAI_MODEL is not configured"
        )
    return model


def final_answer_model() -> str:
    _load_local_environment()
    model = os.getenv("FINAL_ANSWER_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("FINAL_ANSWER_MODEL or OPENAI_MODEL is not configured")
    return model
