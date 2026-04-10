"""Analyze benchmark results and generate research metrics."""

import os
import pandas as pd
from sqlalchemy import create_engine

# Note: Connect to localhost port 5432 because you are running this from your host machine, not inside Docker.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/benchmark")

def main():
    print("Connecting to database...")
    engine = create_engine(DATABASE_URL)
    
    # Load all results into a pandas DataFrame
    query = "SELECT * FROM evaluation_results"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No data found! Ensure the Celery workers have finished running.")
        return

    print("\n" + "="*50)
    print("🔬 OVERALL ARCHITECTURE COMPARISON")
    print("="*50)
    
    # Calculate overarching metrics
    overall_stats = df.groupby("architecture_type").agg(
        total_queries=("id", "count"),
        avg_judge_score=("judge_score", "mean"),
        exact_match_rate=("exact_match_score", "mean"),
        avg_latency_ms=("latency_ms", "mean")
    ).round(3)
    
    # Convert EM to percentage
    overall_stats["exact_match_rate"] = (overall_stats["exact_match_rate"] * 100).astype(str) + "%"
    print(overall_stats.to_markdown())

    print("\n" + "="*50)
    print("📊 SLICE ANALYSIS: PERFORMANCE BY CATEGORY")
    print("="*50)
    
    # Calculate metrics broken down by category (The "Slice" analysis)
    slice_stats = df.pivot_table(
        index="category", 
        columns="architecture_type", 
        values="judge_score", 
        aggfunc="mean"
    ).round(2)
    
    print(slice_stats.to_markdown())
    
    print("\n" + "="*50)
    print("⏱️ TRADE-OFF ANALYSIS")
    print("="*50)
    
    baseline_lat = df[df['architecture_type'] == 'baseline']['latency_ms'].mean()
    agentic_lat = df[df['architecture_type'] == 'agentic']['latency_ms'].mean()
    latency_diff = agentic_lat - baseline_lat
    
    baseline_score = df[df['architecture_type'] == 'baseline']['judge_score'].mean()
    agentic_score = df[df['architecture_type'] == 'agentic']['judge_score'].mean()
    score_diff = agentic_score - baseline_score
    
    print(f"Agentic Reflection added {latency_diff / 1000:.2f} seconds of latency on average.")
    if score_diff > 0:
        print(f"However, it improved the average LLM Judge score by {score_diff:.2f} points out of 5.")
    else:
        print(f"Surprisingly, it did NOT improve the average judge score (Difference: {score_diff:.2f}).")

if __name__ == "__main__":
    main()