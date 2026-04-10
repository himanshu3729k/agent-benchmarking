"""Celery worker for asynchronous pairwise research execution."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from evaluation.evaluator import JudgeResult, exact_match_score, llm_judge
from src.agent import AgentAnswer, run_qa_experiment
from src.database import EvaluationResult, get_engine


logger = logging.getLogger(__name__)
task_logger = get_task_logger(__name__)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:gpt-4o-mini")

celery_app = Celery("benchmark_worker", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((TimeoutError, RuntimeError, ValueError)),
)
def _run_agent_with_backoff(question: str, context: str, architecture: str) -> tuple[AgentAnswer, float]:
    """Run QA experiment with exponential backoff to handle rate limits."""
    return run_qa_experiment(question=question, context=context, architecture=architecture)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((TimeoutError, RuntimeError, ValueError)),
)
def _run_judge_with_backoff(question: str, prediction: str, ground_truth: str) -> JudgeResult:
    """Run judge agent with exponential backoff."""
    return llm_judge(
        question=question,
        prediction=prediction,
        ground_truth=ground_truth,
        model_name=JUDGE_MODEL,
    )


def _update_task_status(task_id: str, status_value: str) -> None:
    """Best-effort status update for the `tasks` table."""
    engine = get_engine()
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE tasks SET status = :status, updated_at = NOW() "
                    "WHERE id = :task_id"
                ),
                {"status": status_value, "task_id": task_id},
            )
    finally:
        engine.dispose()


def _save_evaluation(
    *,
    task_id: str,
    architecture_type: str,
    category: str,
    question: str,
    expected_answer: str,
    agent_prediction: str,
    latency_ms: float,
    exact_match: bool,
    judge_score: int,
    judge_rationale: str,
) -> None:
    """Persist benchmark output into `evaluation_results` with research metadata."""
    engine = get_engine()
    # We use a unique sub-task ID so we can save both baseline and agentic results
    # under the same parent task execution.
    eval_id = f"{task_id}_{architecture_type}_{uuid.uuid4().hex[:6]}"
    
    try:
        with Session(engine) as session:
            row = EvaluationResult(
                task_id=eval_id,
                architecture_type=architecture_type,
                category=category,
                model_name=os.getenv("PRIMARY_QA_MODEL", "openai:gpt-4o-mini"),
                question=question,
                expected_answer=expected_answer,
                agent_prediction=agent_prediction,
                latency_ms=latency_ms,
                exact_match_score=exact_match,
                judge_score=judge_score,
                judge_rationale=judge_rationale,
            )
            session.add(row)
            session.commit()
    finally:
        engine.dispose()


@celery_app.task(name="process_benchmark_query", bind=True, max_retries=0)
def process_benchmark_query(
    self: Any,
    task_id: str,
    question: str,
    context: str,
    ground_truth: str,
    category: str,
) -> dict[str, Any]:
    """Process a single benchmark query using BOTH architectures for comparison."""
    del self
    task_logger.info("Starting pairwise execution for task_id=%s category=%s", task_id, category)
    results_summary = {}

    try:
        # Loop over our two experimental conditions
        for architecture in ["baseline", "agentic"]:
            task_logger.info("Running architecture=%s for task_id=%s", architecture, task_id)
            
            # 1. Run the specific architecture
            answer_payload, latency = _run_agent_with_backoff(
                question=question, 
                context=context, 
                architecture=architecture
            )
            prediction = answer_payload.answer
            
            # 2. Evaluate the output
            exact_match = exact_match_score(prediction=prediction, ground_truth=ground_truth)
            judge_payload = _run_judge_with_backoff(
                question=question,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            # 3. Save the specific result
            _save_evaluation(
                task_id=task_id,
                architecture_type=architecture,
                category=category,
                question=question,
                expected_answer=ground_truth,
                agent_prediction=prediction,
                latency_ms=latency,
                exact_match=exact_match,
                judge_score=judge_payload.score,
                judge_rationale=judge_payload.rationale,
            )
            
            # Store for the Celery return payload
            results_summary[architecture] = {
                "judge_score": judge_payload.score,
                "latency_ms": latency
            }

        _update_task_status(task_id=task_id, status_value="COMPLETED")
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "results": results_summary
        }

    except RetryError as exc:
        task_logger.exception("Retries exhausted for task_id=%s: %s", task_id, exc)
        _update_task_status(task_id=task_id, status_value="FAILED")
        raise
    except Exception as exc:
        task_logger.exception("Unexpected task failure task_id=%s: %s", task_id, exc)
        _update_task_status(task_id=task_id, status_value="FAILED")
        raise