#!/usr/bin/env python3
"""Run standard MTEB evaluation on datasets that overlap with MITE tasks.

This script evaluates embedding models on the MTEB versions of the same
datasets that MITE re-frames as interaction tasks:
  - SICK-R  (STS)
  - FEVER   (Retrieval)
  - FiQA    (Retrieval)
  - SummEval (Summarization)

Results are saved to JSON for later comparison with MITE scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default models ──────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "jinaai/jina-embeddings-v2-base-en",
    "thenlper/gte-base",
]

# ── MTEB datasets to evaluate ──────────────────────────────────────────────

MTEB_TASKS = [
    "SICK-R",
    "FEVER",
    "FiQA2018",
    "SummEval",
]

# Mapping from MTEB task name to the metric key we extract from results.
METRIC_KEYS = {
    "SICK-R": "cosine_spearman",
    "FEVER": "ndcg_at_10",
    "FiQA2018": "ndcg_at_10",
    "SummEval": "cosine_spearman",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation on datasets overlapping with MITE tasks.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model names to evaluate (default: 10 popular models).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write results JSON (default: results/).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=MTEB_TASKS,
        choices=MTEB_TASKS,
        help="MTEB tasks to evaluate (default: all four).",
    )
    return parser.parse_args()


def run_mteb_for_model(
    model_name: str,
    tasks: list[str],
) -> dict[str, dict[str, float]]:
    """Run MTEB evaluation for a single model on the specified tasks.

    Returns a dict mapping task_name -> {metric_name: score, ...}.
    """
    try:
        import mteb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "mteb and sentence-transformers are required. "
            "Install with: pip install mteb sentence-transformers"
        ) from exc

    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)

    task_objects = mteb.get_tasks(tasks=tasks)

    evaluation = mteb.MTEB(tasks=task_objects)

    logger.info("Running MTEB evaluation on %d tasks ...", len(tasks))
    results = evaluation.run(
        model,
        output_folder=None,
        verbosity=1,
    )

    # Parse results into a clean dict.
    parsed: dict[str, dict[str, float]] = {}
    for task_result in results:
        task_name = task_result.task_name
        # MTEB results have nested structure; extract the test split scores.
        scores = task_result.scores
        if "test" in scores:
            split_scores = scores["test"]
        elif "validation" in scores:
            split_scores = scores["validation"]
        else:
            # Take the first available split.
            first_split = next(iter(scores.keys()))
            split_scores = scores[first_split]

        # split_scores is a list of dicts (one per language/subset).
        # Take the first one.
        if isinstance(split_scores, list) and len(split_scores) > 0:
            metric_dict = split_scores[0]
        else:
            metric_dict = split_scores

        parsed[task_name] = {}
        if isinstance(metric_dict, dict):
            for k, v in metric_dict.items():
                if isinstance(v, (int, float)):
                    parsed[task_name][k] = float(v)

    return parsed


def extract_primary_score(task_name: str, metrics: dict[str, float]) -> float | None:
    """Extract the primary metric score for a given MTEB task."""
    key = METRIC_KEYS.get(task_name)
    if key and key in metrics:
        return metrics[key]
    # Fallback: try main_score.
    if "main_score" in metrics:
        return metrics["main_score"]
    return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info("Evaluating model: %s", model_name)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            task_metrics = run_mteb_for_model(model_name, args.tasks)
        except Exception:
            logger.exception("Failed to evaluate %s", model_name)
            continue
        elapsed = time.time() - t0

        # Build per-task primary scores.
        primary_scores: dict[str, float | None] = {}
        for task_name in args.tasks:
            if task_name in task_metrics:
                primary_scores[task_name] = extract_primary_score(
                    task_name, task_metrics[task_name]
                )
            else:
                primary_scores[task_name] = None

        all_results[model_name] = {
            "primary_scores": primary_scores,
            "all_metrics": task_metrics,
            "elapsed_seconds": round(elapsed, 1),
        }

        # Print progress.
        logger.info(
            "  %s done in %.1fs: %s",
            model_name,
            elapsed,
            {k: f"{v:.4f}" if v is not None else "N/A" for k, v in primary_scores.items()},
        )

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MTEB BASELINE RESULTS")
    print("=" * 80)

    header = f"{'Model':<45}"
    for task_name in args.tasks:
        header += f"  {task_name:<12}"
    header += "  Avg"
    print(header)
    print("-" * len(header))

    for model_name, result in all_results.items():
        row = f"{model_name:<45}"
        scores = []
        for task_name in args.tasks:
            score = result["primary_scores"].get(task_name)
            if score is not None:
                row += f"  {score:<12.4f}"
                scores.append(score)
            else:
                row += f"  {'N/A':<12}"
        if scores:
            avg = sum(scores) / len(scores)
            row += f"  {avg:.4f}"
        print(row)

    # ── Save to JSON ────────────────────────────────────────────────────
    output_path = output_dir / "mteb_baseline.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
