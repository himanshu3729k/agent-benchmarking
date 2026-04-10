"""Database configuration and ORM models for research-grade benchmark persistence."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/benchmark")

class Base(DeclarativeBase):
    """Base SQLAlchemy declarative model."""

class EvaluationResult(Base):
    """Stores comparative outputs for research analysis."""
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    
    # Research Metadata
    architecture_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False) # 'baseline' vs 'agentic'
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False) # 'Numerical', 'Logical', 'Extraction'
    model_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    
    # Input Data
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Model Outputs
    agent_prediction: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False) # For cost-benefit analysis
    
    # Evaluation Metrics
    exact_match_score: Mapped[bool] = mapped_column(Boolean, nullable=False)
    judge_score: Mapped[int] = mapped_column(Integer, nullable=False)
    judge_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def init_db_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)