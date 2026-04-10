"""Download and preprocess a Finance subset (FinQA) for research-grade benchmarking."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

FINQA_DEV_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json"

def _download_json(url: str) -> list[dict[str, Any]]:
    """Download the raw JSON dataset."""
    request = Request(url, headers={"User-Agent": "finance-research-benchmark/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))

def _format_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    return "\n".join([" | ".join(row) for row in table])

def _extract_examples(payload: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    
    for item in payload:
        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        table_str = _format_table(item.get("table", []))
        
        context = f"TEXT:\n{pre_text}\n\nTABLE:\n{table_str}\n\nTEXT:\n{post_text}"
        category = "Numerical Reasoning" if item.get("table") else "Financial Extraction"
        
        # THE FIX: FinQA nests questions and answers inside a 'qa' dictionary
        qa_dict = item.get("qa", {})
        question = str(qa_dict.get("question", "")).strip()
        answer = str(qa_dict.get("answer", "")).strip()
        
        # Only save if we successfully extracted actual text
        if question and answer:
            examples.append({
                "question": question,
                "context": context.strip(),
                "answer": answer,
                "category": category,
                "id": item.get("id", "")
            })
        
        if len(examples) >= limit:
            break
            
    return examples

def main() -> None:
    parser = argparse.ArgumentParser(description="Download FinQA subset for research benchmarking.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of items to sample.")
    args = parser.parse_args()

    print(f"Fetching finance data from {FINQA_DEV_URL}...")
    raw_data = _download_json(FINQA_DEV_URL)
    subset = _extract_examples(raw_data, limit=args.limit)

    output_path = Path("data") / "finance_dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Research data ready: {len(subset)} VALID financial queries saved to {output_path}")

if __name__ == "__main__":
    main()