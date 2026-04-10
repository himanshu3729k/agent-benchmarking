"""Evaluation utilities for benchmark outputs.

Includes:
- Exact match metric (normalized substring match)
- LLM-as-a-judge structured scoring via Pydantic AI
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from langfuse.decorators import observe
from pydantic import BaseModel, Field
from pydantic_ai import Agent


logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for robust comparison.

    Steps:
    - lowercase
    - collapse non-alphanumeric characters into spaces
    - squeeze repeated whitespace
    """

    lowered = text.lower().strip()
    alnum_spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", alnum_spaced).strip()


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Return True when normalized strings match by substring containment."""

    normalized_prediction = _normalize_text(prediction)
    normalized_ground_truth = _normalize_text(ground_truth)
    if not normalized_prediction or not normalized_ground_truth:
        return False
    return (
        normalized_prediction in normalized_ground_truth
        or normalized_ground_truth in normalized_prediction
    )


class JudgeResult(BaseModel):
    """Strict schema returned by LLM judge."""

    rationale: str = Field(description="Concise justification of the assigned score.")
    score: int = Field(ge=1, le=5, description="Integer quality score from 1 (poor) to 5 (excellent).")


@lru_cache(maxsize=4)
def _build_judge_agent(model_name: str) -> Agent[None, JudgeResult]:
    """Create and cache an LLM-as-a-judge agent per model."""

    return Agent[None, JudgeResult](
        model_name,
        result_type=JudgeResult,
        system_prompt=(
            "You are an objective evaluator for QA predictions.\n"
            "Given question, prediction, and ground truth:\n"
            "- Score 1-5 for factual correctness and reasoning quality.\n"
            "- 5: fully correct and well-grounded.\n"
            "- 3: partially correct or missing key detail.\n"
            "- 1: incorrect or unsupported.\n"
            "Return strict JSON with fields: rationale, score."
        ),
    )


@observe(name="run_llm_judge")
def llm_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    model_name: str | None = None,
) -> JudgeResult:
    """Evaluate model output quality using a second LLM."""

    selected_model = model_name or os.getenv("JUDGE_MODEL", "openai:gpt-4o-mini")
    prompt = (
        "Evaluate the prediction against the reference.\n\n"
        f"Question:\n{question}\n\n"
        f"Prediction:\n{prediction}\n\n"
        f"Ground Truth:\n{ground_truth}"
    )

    logger.debug("Running judge with model=%s", selected_model)
    result = _build_judge_agent(selected_model).run_sync(prompt)
    return result.output
