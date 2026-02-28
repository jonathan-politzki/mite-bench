#!/usr/bin/env python3
"""Compare MTEB vs MITE rankings across models.

Loads results from both benchmarks, computes rank correlations for each
MITE task against its MTEB counterpart, and generates comparison tables
and scatter plots.

Example usage:
    # Using saved result files
    python scripts/compare_rankings.py

    # With custom paths
    python scripts/compare_rankings.py \
        --mteb-results results/mteb_baseline.json \
        --mite-results results/mite_results.json

    # Use hardcoded leaderboard scores (no MTEB run needed)
    python scripts/compare_rankings.py --use-leaderboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Approximate MTEB leaderboard scores for the default 10 models ──────────
# These are approximate values from the public MTEB leaderboard so the
# comparison script works without running MTEB locally.

MTEB_LEADERBOARD_APPROX: dict[str, dict[str, float]] = {
    "all-MiniLM-L6-v2": {
        "SICK-R": 0.804,
        "FEVER": 0.654,
        "FiQA": 0.317,
        "SummEval": 0.309,
        "avg": 0.521,
    },
    "BAAI/bge-small-en-v1.5": {
        "SICK-R": 0.798,
        "FEVER": 0.740,
        "FiQA": 0.362,
        "SummEval": 0.313,
        "avg": 0.553,
    },
    "BAAI/bge-base-en-v1.5": {
        "SICK-R": 0.810,
        "FEVER": 0.773,
        "FiQA": 0.402,
        "SummEval": 0.310,
        "avg": 0.574,
    },
    "intfloat/e5-small-v2": {
        "SICK-R": 0.802,
        "FEVER": 0.702,
        "FiQA": 0.348,
        "SummEval": 0.301,
        "avg": 0.538,
    },
    "intfloat/e5-base-v2": {
        "SICK-R": 0.813,
        "FEVER": 0.748,
        "FiQA": 0.380,
        "SummEval": 0.306,
        "avg": 0.562,
    },
    "intfloat/e5-large-v2": {
        "SICK-R": 0.829,
        "FEVER": 0.782,
        "FiQA": 0.423,
        "SummEval": 0.315,
        "avg": 0.587,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "SICK-R": 0.808,
        "FEVER": 0.671,
        "FiQA": 0.339,
        "SummEval": 0.307,
        "avg": 0.531,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "SICK-R": 0.801,
        "FEVER": 0.753,
        "FiQA": 0.394,
        "SummEval": 0.305,
        "avg": 0.563,
    },
    "jinaai/jina-embeddings-v2-base-en": {
        "SICK-R": 0.800,
        "FEVER": 0.717,
        "FiQA": 0.367,
        "SummEval": 0.303,
        "avg": 0.547,
    },
    "thenlper/gte-base": {
        "SICK-R": 0.818,
        "FEVER": 0.735,
        "FiQA": 0.381,
        "SummEval": 0.311,
        "avg": 0.561,
    },
}

# ── Mapping from MITE task names to MTEB dataset/metric ────────────────────

MITE_TO_MTEB_MAP = {
    "SICKREntailment": {"mteb_dataset": "SICK-R", "leaderboard_key": "SICK-R"},
    "FEVERInteraction": {"mteb_dataset": "FEVER", "leaderboard_key": "FEVER"},
    "ClimateFEVERInteraction": {"mteb_dataset": "FEVER", "leaderboard_key": "FEVER"},
    "SciFActInteraction": {"mteb_dataset": "FEVER", "leaderboard_key": "FEVER"},
    "FiQAAnswerQuality": {"mteb_dataset": "FiQA2018", "leaderboard_key": "FiQA"},
    "CQADupstackAnswerQuality": {"mteb_dataset": "FiQA2018", "leaderboard_key": "FiQA"},
    "SummEvalQuality": {"mteb_dataset": "SummEval", "leaderboard_key": "SummEval"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MTEB vs MITE rankings across models.",
    )
    parser.add_argument(
        "--mteb-results",
        type=str,
        default="results/mteb_baseline.json",
        help="Path to MTEB results JSON from run_mteb_baseline.py.",
    )
    parser.add_argument(
        "--mite-results",
        type=str,
        default="results/mite_results.json",
        help="Path to MITE results JSON from run_mite.py.",
    )
    parser.add_argument(
        "--use-leaderboard",
        action="store_true",
        help="Use hardcoded MTEB leaderboard scores instead of local results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save comparison figures (default: results/figures/).",
    )
    return parser.parse_args()


def load_mteb_scores(
    results_path: str, use_leaderboard: bool
) -> dict[str, dict[str, float]]:
    """Load MTEB scores: either from a local results file or the hardcoded leaderboard.

    Returns a dict: model_name -> {dataset_name: score, ...}
    """
    if use_leaderboard:
        logger.info("Using hardcoded MTEB leaderboard scores.")
        return MTEB_LEADERBOARD_APPROX

    path = Path(results_path)
    if not path.exists():
        logger.warning(
            "MTEB results file %s not found. Falling back to leaderboard scores.",
            results_path,
        )
        return MTEB_LEADERBOARD_APPROX

    logger.info("Loading MTEB results from %s", results_path)
    with open(path) as f:
        raw = json.load(f)

    # Normalize: the run_mteb_baseline.py output has
    # model_name -> {primary_scores: {task: score}, ...}
    scores: dict[str, dict[str, float]] = {}
    for model_name, data in raw.items():
        primary = data.get("primary_scores", {})
        scores[model_name] = {}
        for task_name, score in primary.items():
            if score is not None:
                # Map MTEB task names to our short keys.
                short = task_name.replace("2018", "")
                scores[model_name][short] = score
                # Also keep the original name.
                scores[model_name][task_name] = score
    return scores


def load_mite_scores(results_path: str) -> dict[str, dict[str, float]]:
    """Load MITE results.

    Returns a dict: model_name -> {task_name: primary_score, ...}
    """
    path = Path(results_path)
    if not path.exists():
        logger.error("MITE results file %s not found.", results_path)
        sys.exit(1)

    logger.info("Loading MITE results from %s", results_path)
    with open(path) as f:
        raw = json.load(f)

    scores: dict[str, dict[str, float]] = {}
    for model_name, tasks in raw.items():
        scores[model_name] = {}
        for task_name, task_data in tasks.items():
            score = task_data.get("primary_score")
            if score is not None:
                scores[model_name][task_name] = score
    return scores


def compute_rank_correlation(
    mteb_scores: list[float], mite_scores: list[float]
) -> tuple[float, float]:
    """Spearman rank correlation and p-value between two score lists."""
    if len(mteb_scores) < 3:
        return 0.0, 1.0
    rho, p = stats.spearmanr(mteb_scores, mite_scores)
    return float(rho), float(p)


def find_biggest_divergences(
    models: list[str],
    mteb_scores: list[float],
    mite_scores: list[float],
    top_k: int = 3,
) -> list[tuple[str, int, int, int]]:
    """Find models whose rank changes the most between MTEB and MITE.

    Returns a list of (model_name, mteb_rank, mite_rank, rank_change).
    """
    n = len(models)
    # Compute ranks (1-based, higher score = better rank = rank 1).
    mteb_order = np.argsort(-np.array(mteb_scores))
    mite_order = np.argsort(-np.array(mite_scores))

    mteb_ranks = np.empty(n, dtype=int)
    mite_ranks = np.empty(n, dtype=int)
    for rank, idx in enumerate(mteb_order):
        mteb_ranks[idx] = rank + 1
    for rank, idx in enumerate(mite_order):
        mite_ranks[idx] = rank + 1

    divergences = []
    for i in range(n):
        change = abs(int(mteb_ranks[i]) - int(mite_ranks[i]))
        divergences.append(
            (models[i], int(mteb_ranks[i]), int(mite_ranks[i]), change)
        )

    divergences.sort(key=lambda x: x[3], reverse=True)
    return divergences[:top_k]


def make_scatter_plots(
    comparison_data: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate scatter plots of MTEB rank vs MITE rank for each task."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping scatter plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = len(comparison_data)
    if n_tasks == 0:
        logger.warning("No comparison data; skipping plots.")
        return

    fig, axes = plt.subplots(
        1, n_tasks, figsize=(5 * n_tasks, 5), squeeze=False
    )

    for idx, (task_name, data) in enumerate(comparison_data.items()):
        ax = axes[0, idx]
        models = data["models"]
        mteb = np.array(data["mteb_scores"])
        mite = np.array(data["mite_scores"])
        n = len(models)

        if n < 2:
            ax.set_title(f"{task_name}\n(insufficient data)")
            continue

        # Compute ranks.
        mteb_ranks = np.empty(n, dtype=int)
        mite_ranks = np.empty(n, dtype=int)
        for rank, i in enumerate(np.argsort(-mteb)):
            mteb_ranks[i] = rank + 1
        for rank, i in enumerate(np.argsort(-mite)):
            mite_ranks[i] = rank + 1

        ax.scatter(mteb_ranks, mite_ranks, s=50, c="#2c7bb6", zorder=5)

        # Identity line.
        lim = [0.5, n + 0.5]
        ax.plot(lim, lim, "--", color="#999999", linewidth=1, zorder=1)

        # Label points.
        for i, model in enumerate(models):
            short = model.split("/")[-1] if "/" in model else model
            # Truncate long names.
            if len(short) > 20:
                short = short[:18] + ".."
            ax.annotate(
                short,
                (mteb_ranks[i], mite_ranks[i]),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

        rho = data.get("spearman", 0.0)
        ax.set_title(f"{task_name}\nrho = {rho:.3f}", fontsize=10)
        ax.set_xlabel("MTEB Rank", fontsize=9)
        ax.set_ylabel("MITE Rank", fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = output_dir / "mteb_vs_mite_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter plot saved to %s", out_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scores ─────────────────────────────────────────────────────
    mteb_scores = load_mteb_scores(args.mteb_results, args.use_leaderboard)
    mite_scores = load_mite_scores(args.mite_results)

    # Find common models.
    mteb_models = set(mteb_scores.keys())
    mite_models = set(mite_scores.keys())
    common_models = sorted(mteb_models & mite_models)

    if len(common_models) == 0:
        logger.error(
            "No overlapping models between MTEB and MITE results.\n"
            "MTEB models: %s\nMITE models: %s",
            sorted(mteb_models),
            sorted(mite_models),
        )
        sys.exit(1)

    logger.info("Found %d common models.", len(common_models))

    # ── Get MITE task names from the results ────────────────────────────
    all_mite_tasks: set[str] = set()
    for model_tasks in mite_scores.values():
        all_mite_tasks.update(model_tasks.keys())
    mite_task_names = sorted(all_mite_tasks)

    # ── Per-task comparison ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MTEB vs MITE RANKING COMPARISON")
    print("=" * 80)

    comparison_data: dict[str, dict] = {}

    header = f"{'MITE Task':<28}{'MTEB Dataset':<16}{'Spearman':<12}{'p-value':<12}{'N models':<10}"
    print(header)
    print("-" * len(header))

    for mite_task in mite_task_names:
        mapping = MITE_TO_MTEB_MAP.get(mite_task)
        if mapping is None:
            logger.warning(
                "No MTEB mapping for MITE task %r; skipping.", mite_task
            )
            continue

        leaderboard_key = mapping["leaderboard_key"]

        # Collect paired scores.
        models_with_both: list[str] = []
        mteb_vals: list[float] = []
        mite_vals: list[float] = []

        for model in common_models:
            mite_score = mite_scores.get(model, {}).get(mite_task)
            mteb_model_scores = mteb_scores.get(model, {})
            mteb_score = mteb_model_scores.get(leaderboard_key)

            if mite_score is not None and mteb_score is not None:
                models_with_both.append(model)
                mteb_vals.append(mteb_score)
                mite_vals.append(mite_score)

        if len(models_with_both) < 3:
            print(
                f"{mite_task:<28}{leaderboard_key:<16}{'N/A':<12}{'N/A':<12}{len(models_with_both):<10}"
            )
            continue

        rho, p = compute_rank_correlation(mteb_vals, mite_vals)

        comparison_data[mite_task] = {
            "models": models_with_both,
            "mteb_scores": mteb_vals,
            "mite_scores": mite_vals,
            "spearman": rho,
            "p_value": p,
            "mteb_dataset": leaderboard_key,
        }

        print(
            f"{mite_task:<28}{leaderboard_key:<16}{rho:<12.4f}{p:<12.4f}{len(models_with_both):<10}"
        )

    # ── Average correlation ─────────────────────────────────────────────
    if comparison_data:
        rhos = [d["spearman"] for d in comparison_data.values()]
        print("-" * len(header))
        avg_rho = np.mean(rhos)
        print(f"{'Average':<28}{'':<16}{avg_rho:<12.4f}")

    # ── Biggest divergences per task ────────────────────────────────────
    print("\n" + "=" * 80)
    print("BIGGEST RANK DIVERGENCES (models that rank very differently)")
    print("=" * 80)

    for mite_task, data in comparison_data.items():
        print(f"\n{mite_task}:")
        divergences = find_biggest_divergences(
            data["models"], data["mteb_scores"], data["mite_scores"]
        )
        for model, mteb_rank, mite_rank, change in divergences:
            short = model.split("/")[-1] if "/" in model else model
            direction = "up" if mite_rank < mteb_rank else "down"
            print(
                f"  {short:<35} MTEB #{mteb_rank} -> MITE #{mite_rank}  "
                f"({direction} {change} places)"
            )

    # ── Overall model rankings ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL MODEL RANKINGS")
    print("=" * 80)

    # Compute average MITE score per model.
    model_avg_mite: dict[str, float] = {}
    model_avg_mteb: dict[str, float] = {}

    for model in common_models:
        mite_task_scores = [
            mite_scores[model][t]
            for t in mite_task_names
            if t in mite_scores.get(model, {}) and mite_scores[model][t] is not None
        ]
        if mite_task_scores:
            model_avg_mite[model] = np.mean(mite_task_scores)

        mteb_model_data = mteb_scores.get(model, {})
        if "avg" in mteb_model_data:
            model_avg_mteb[model] = mteb_model_data["avg"]
        else:
            # Compute avg from available tasks.
            vals = [v for k, v in mteb_model_data.items() if isinstance(v, (int, float))]
            if vals:
                model_avg_mteb[model] = np.mean(vals)

    # Sort by MITE avg.
    sorted_by_mite = sorted(model_avg_mite.items(), key=lambda x: x[1], reverse=True)

    row_header = f"{'Rank':<6}{'Model':<40}{'MITE Avg':<12}{'MTEB Avg':<12}"
    print(row_header)
    print("-" * len(row_header))
    for rank, (model, mite_avg) in enumerate(sorted_by_mite, 1):
        short = model.split("/")[-1] if "/" in model else model
        mteb_avg = model_avg_mteb.get(model)
        mteb_str = f"{mteb_avg:.4f}" if mteb_avg is not None else "N/A"
        print(f"{rank:<6}{short:<40}{mite_avg:<12.4f}{mteb_str:<12}")

    # ── Scatter plots ───────────────────────────────────────────────────
    make_scatter_plots(comparison_data, output_dir)

    # ── Save comparison data ────────────────────────────────────────────
    save_path = output_dir.parent / "mteb_vs_mite_comparison.json"
    serializable = {}
    for task, data in comparison_data.items():
        serializable[task] = {
            "models": data["models"],
            "mteb_scores": data["mteb_scores"],
            "mite_scores": data["mite_scores"],
            "spearman": data["spearman"],
            "p_value": data["p_value"],
            "mteb_dataset": data["mteb_dataset"],
        }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Comparison data saved to %s", save_path)


if __name__ == "__main__":
    main()
