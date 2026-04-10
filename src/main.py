"""FastAPI entrypoint for the AI agent benchmarking pipeline.

This module provides:
- typed application settings from environment variables
- startup/shutdown lifecycle management
- lightweight dependency health checks (Postgres + Redis)
- initial benchmark API surface for orchestration
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.database import init_db_tables
from src.worker import process_benchmark_query

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Agent Benchmarking API"
    app_version: str = "0.1.0"
    app_env: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/benchmark"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _to_async_database_url(url: str) -> str:
    """Convert sync Postgres URL to asyncpg URL for FastAPI engine."""

    if "+asyncpg" in url:
        return url
    if "+psycopg2" in url:
        return url.replace("+psycopg2", "+asyncpg")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


class HealthResponse(BaseModel):
    """Public health payload used by infra probes."""

    status: str
    environment: str
    timestamp_utc: str
    dependencies: dict[str, bool]


class BenchmarkStartRequest(BaseModel):
    """Benchmark trigger request for batch metadata or single-query enqueue."""

    dataset_size: int = Field(default=100, ge=1, le=100_000)
    chunk_size: int = Field(default=25, ge=1, le=1_000)
    question: str | None = Field(default=None)
    context: str | None = Field(default=None)
    ground_truth: str | None = Field(default=None)
    model_name: str = Field(default="openai:gpt-4o-mini") # Kept so run_benchmark doesn't crash
    category: str = Field(default="General Finance", description="Research slice category.") # NEW: Added for research
    task_id: str | None = Field(default=None)


class BenchmarkStartResponse(BaseModel):
    """Response returned after benchmark orchestration request is accepted."""

    message: str
    accepted: bool
    requested_dataset_size: int
    requested_chunk_size: int
    task_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and close shared app resources."""
    app.state.db_engine = create_async_engine(
        _to_async_database_url(settings.database_url),
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        future=True,
    )
    app.state.redis = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)

    logger.info("Starting API in %s environment", settings.app_env)
    init_db_tables()
    yield
    logger.info("Shutting down API")

    await app.state.redis.close()
    await app.state.db_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


async def _check_postgres(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.warning("Postgres health check failed: %s", exc)
        return False


async def _check_redis(redis_client: aioredis.Redis) -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    db_ok = await _check_postgres(app.state.db_engine)
    redis_ok = await _check_redis(app.state.redis)

    return HealthResponse(
        status="ok" if db_ok and redis_ok else "degraded",
        environment=settings.app_env,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dependencies={"postgres": db_ok, "redis": redis_ok},
    )


@app.post(
    "/benchmark/start",
    response_model=BenchmarkStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["benchmark"],
)
async def start_benchmark(payload: BenchmarkStartRequest) -> BenchmarkStartResponse:
    """Kick off a benchmark request and enqueue a Celery task when query payload is provided."""

    if payload.chunk_size > payload.dataset_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_size must be less than or equal to dataset_size",
        )

    task_identifier: str | None = None
    if payload.question and payload.context and payload.ground_truth:
        task_identifier = payload.task_id or str(uuid.uuid4())
        
        try:
            # NEW: We now pass `category` to the Celery worker, and removed `model_name`
            process_benchmark_query.delay(
                task_id=task_identifier,
                question=payload.question,
                context=payload.context,
                ground_truth=payload.ground_truth,
                category=payload.category,
            )
            logger.info("Enqueued benchmark task_id=%s category=%s", task_identifier, payload.category)
        except Exception as e:
            logger.error("Failed to enqueue task: %s", e)
            raise HTTPException(status_code=500, detail="Failed to enqueue to Celery")

    return BenchmarkStartResponse(
        message="Benchmark request accepted.",
        accepted=True,
        requested_dataset_size=payload.dataset_size,
        requested_chunk_size=payload.chunk_size,
        task_id=task_identifier,
    )


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    return {"service": settings.app_name, "version": settings.app_version}