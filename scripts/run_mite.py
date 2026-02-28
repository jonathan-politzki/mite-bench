#!/usr/bin/env python3
"""Run MITE interaction evaluation on all tasks.

Main evaluation script for the MITE benchmark. Evaluates one or more
embedding models on MITE's interaction tasks and produces a results
JSON plus a summary table.

Example usage:
    # Run all default models on all tasks
    python scripts/run_mite.py

    # Run specific models on specific tasks
    python scripts/run_mite.py \
        --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 \
        --tasks SICKREntailment FEVERInteraction

    # Custom batch size and output directory
    python scripts/run_mite.py --batch-size 128 --output-dir results/run1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MITE interaction evaluation on all tasks.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate (default: 10 popular models).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help=(
            "Task names to evaluate, or 'all' for every task. "
            "Use the task_name attribute (e.g. SICKREntailment, FEVERInteraction)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64).",
    )
    return parser.parse_args()


def get_task_instances(requested: list[str]) -> list:
    """Resolve requested task names to instantiated task objects."""
    from mite.tasks import ALL_TASKS

    if requested == ["all"] or requested == ["ALL"]:
        return [cls() for cls in ALL_TASKS]

    # Build a lookup from task_name -> class.
    name_to_cls = {}
    for cls in ALL_TASKS:
        instance = cls()
        name_to_cls[instance.task_name] = cls
        # Also allow matching on the class name.
        name_to_cls[cls.__name__] = cls

    instances = []
    for name in requested:
        if name in name_to_cls:
            instances.append(name_to_cls[name]())
        else:
            available = sorted(name_to_cls.keys())
            logger.error(
                "Unknown task %r. Available tasks: %s", name, ", ".join(available)
            )
            sys.exit(1)
    return instances


def create_model(model_name: str):
    """Create a model wrapper for the given model name.

    Tries the mite.models wrappers first, falls back to
    sentence-transformers directly.
    """
    try:
        from mite.models import SentenceTransformerModel

        return SentenceTransformerModel(model_name)
    except ImportError:
        pass

    # Fallback: wrap sentence-transformers directly.
    from sentence_transformers import SentenceTransformer
    import numpy as np

    class _STWrapper:
        """Minimal wrapper around SentenceTransformer for MITE compatibility."""

        def __init__(self, name: str) -> None:
            self.name = name
            self._model = SentenceTransformer(name, trust_remote_code=True)

        def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
            """Row-wise cosine similarity."""
            norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
            norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
            return np.sum(norm_a * norm_b, axis=1)

    return _STWrapper(model_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve tasks ───────────────────────────────────────────────────
    tasks = get_task_instances(args.tasks)
    task_names = [t.task_name for t in tasks]
    logger.info("Tasks to evaluate: %s", task_names)
    logger.info("Models to evaluate: %d models", len(args.models))

    # ── Pre-load all task data ──────────────────────────────────────────
    logger.info("Loading task data ...")
    for task in tasks:
        t0 = time.time()
        task.load_data()
        n_pairs = len(task.get_pairs())
        logger.info(
            "  %s: %d pairs loaded in %.1fs", task.task_name, n_pairs, time.time() - t0
        )

    # ── Evaluate ────────────────────────────────────────────────────────
    # results[model_name][task_name] = {primary_metric, primary_score, metrics, elapsed}
    all_results: dict[str, dict[str, dict]] = {}
    total_t0 = time.time()

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("=" * 60)

        model_t0 = time.time()
        try:
            model = create_model(model_name)
        except Exception:
            logger.exception("Failed to load model %s, skipping.", model_name)
            continue

        model_results: dict[str, dict] = {}

        for task in tasks:
            logger.info("  Evaluating %s ...", task.task_name)
            task_t0 = time.time()

            try:
                result = task.evaluate(model)
                elapsed = time.time() - task_t0

                model_results[task.task_name] = {
                    **result.to_dict(),
                    "elapsed_seconds": round(elapsed, 2),
                }

                logger.info(
                    "    %s = %.4f  (%.1fs)",
                    result.primary_metric,
                    result.primary_score,
                    elapsed,
                )
            except Exception:
                logger.exception("    FAILED on %s", task.task_name)
                model_results[task.task_name] = {
                    "task_name": task.task_name,
                    "model_name": model_name,
                    "primary_metric": task.primary_metric,
                    "primary_score": None,
                    "metrics": {},
                    "error": True,
                    "elapsed_seconds": round(time.time() - task_t0, 2),
                }

        all_results[model_name] = model_results
        logger.info(
            "  Model total: %.1fs", time.time() - model_t0
        )

    total_elapsed = time.time() - total_t0

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("MITE EVALUATION RESULTS")
    print("=" * 100)

    # Header.
    col_width = max(14, max((len(t) for t in task_names), default=14))
    header = f"{'Model':<45}"
    for tn in task_names:
        header += f"  {tn:<{col_width}}"
    header += "  Avg"
    print(header)
    print("-" * len(header))

    # Rows.
    for model_name, model_results in all_results.items():
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        row = f"{short_name:<45}"
        scores: list[float] = []
        for tn in task_names:
            tr = model_results.get(tn, {})
            score = tr.get("primary_score")
            if score is not None:
                row += f"  {score:<{col_width}.4f}"
                scores.append(score)
            else:
                row += f"  {'ERR':<{col_width}}"
        if scores:
            avg = sum(scores) / len(scores)
            row += f"  {avg:.4f}"
        print(row)

    print(f"\nTotal evaluation time: {total_elapsed:.1f}s")
    print(f"Models evaluated: {len(all_results)}")
    print(f"Tasks evaluated: {len(task_names)}")

    # ── Timing breakdown ────────────────────────────────────────────────
    print("\nTiming breakdown (seconds):")
    timing_header = f"{'Model':<45}"
    for tn in task_names:
        timing_header += f"  {tn:<{col_width}}"
    timing_header += "  Total"
    print(timing_header)
    print("-" * len(timing_header))

    for model_name, model_results in all_results.items():
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        row = f"{short_name:<45}"
        total_model = 0.0
        for tn in task_names:
            tr = model_results.get(tn, {})
            elapsed = tr.get("elapsed_seconds", 0.0)
            total_model += elapsed
            row += f"  {elapsed:<{col_width}.1f}"
        row += f"  {total_model:.1f}"
        print(row)

    # ── Save results ────────────────────────────────────────────────────
    output_path = output_dir / "mite_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
