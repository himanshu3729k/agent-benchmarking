"""Pydantic AI agent implementing a Baseline vs. Agentic Reflection experiment."""

from __future__ import annotations
import logging
import os
import time
from functools import lru_cache
from langfuse.decorators import observe
from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

class AgentAnswer(BaseModel):
    """Strict response schema for financial QA."""
    answer: str = Field(description="The final concise answer to the financial question.")
    reflection: str | None = Field(None, description="The internal critique of the initial draft (Agentic mode only).")

@lru_cache(maxsize=4)
def _get_agent(model_name: str, mode: str) -> Agent[None, AgentAnswer]:
    """Factory to build agents for specific experimental conditions."""
    
    if mode == "baseline":
        system_prompt = (
            "You are a professional financial analyst. Answer the question precisely based ONLY on the context. "
            "Return answer='Not answerable' if information is missing."
        )
    else:
        system_prompt = (
            "You are a meticulous financial auditor. \n"
            "PROCESS:\n"
            "1. Generate a draft answer.\n"
            "2. Critically reflect on the draft: Check for numerical errors, alignment with the text, and potential hallucinations.\n"
            "3. Provide the final corrected answer.\n"
            "Return answer='Not answerable' if information is missing."
        )

    return Agent[None, AgentAnswer](
        model_name,
        result_type=AgentAnswer,
        system_prompt=system_prompt,
    )

@observe(name="run_qa_experiment")
def run_qa_experiment(question: str, context: str, architecture: str = "baseline") -> tuple[AgentAnswer, float]:
    """Runs the experiment and returns the answer plus latency for analysis."""
    
    model_name = os.getenv("PRIMARY_QA_MODEL", "openai:gpt-4o-mini")
    agent = _get_agent(model_name, mode=architecture)
    
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    
    start_time = time.perf_counter()
    result = agent.run_sync(prompt)
    latency = (time.perf_counter() - start_time) * 1000
    
    return result.output, latency