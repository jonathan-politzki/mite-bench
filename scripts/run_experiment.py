#!/usr/bin/env python3
"""Run the core MITE experiment: evaluate models on SICK-R and FEVER,
compare MTEB similarity rankings vs MITE interaction rankings.

This is the script that proves the thesis.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mite.tasks import SICKREntailmentTask, FEVERInteractionTask
from mite.models import SentenceTransformerModel

# Models spanning the MTEB leaderboard — small to large, different architectures
MODELS = [
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


def run_task(task, models, task_label):
    """Run a single task across all models, return results dict."""
    print(f"\n{'='*70}")
    print(f"  TASK: {task_label}")
    print(f"{'='*70}")

    print("Loading data...")
    task.load_data()
    print(f"  {len(task.data)} pairs loaded")

    results = {}
    for model_name in models:
        print(f"\n  Evaluating: {model_name}")
        t0 = time.time()
        try:
            model = SentenceTransformerModel(model_name, trust_remote_code=True)
            result = task.evaluate(model)
            elapsed = time.time() - t0
            results[model_name] = result
            print(f"    Primary ({result.primary_metric}): {result.primary_score:.4f}")
            for k, v in sorted(result.metrics.items()):
                if k != result.primary_metric and isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
            print(f"    Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[model_name] = None

    return results


def print_ranking_comparison(sick_results):
    """Print MTEB vs MITE ranking comparison for SICK-R."""
    print(f"\n{'='*70}")
    print("  RANKING COMPARISON: MTEB (Similarity) vs MITE (Interaction)")
    print(f"{'='*70}")

    # Collect scores
    rows = []
    for model_name, result in sick_results.items():
        if result is None:
            continue
        mteb_score = result.metrics.get("mteb_spearman", float("nan"))
        mite_score = result.primary_score
        rows.append((model_name, mteb_score, mite_score))

    # Rank by MTEB (higher = better)
    rows_mteb = sorted(rows, key=lambda x: -x[1])
    mteb_ranks = {name: i + 1 for i, (name, _, _) in enumerate(rows_mteb)}

    # Rank by MITE (higher = better)
    rows_mite = sorted(rows, key=lambda x: -x[2])
    mite_ranks = {name: i + 1 for i, (name, _, _) in enumerate(rows_mite)}

    # Print table
    print(f"\n{'Model':<45} {'MTEB Score':>10} {'MTEB Rank':>10} {'MITE Score':>10} {'MITE Rank':>10} {'Delta':>7}")
    print("-" * 95)
    for name, mteb_s, mite_s in sorted(rows, key=lambda x: mteb_ranks[x[0]]):
        mr = mteb_ranks[name]
        ir = mite_ranks[name]
        delta = mr - ir
        arrow = "+" if delta > 0 else "" if delta == 0 else ""
        print(f"{name:<45} {mteb_s:>10.4f} {mr:>10} {mite_s:>10.4f} {ir:>10} {arrow}{delta:>6}")

    # Compute rank correlation
    from scipy.stats import spearmanr
    mteb_r = [mteb_ranks[n] for n, _, _ in rows]
    mite_r = [mite_ranks[n] for n, _, _ in rows]
    rho, pval = spearmanr(mteb_r, mite_r)
    print(f"\nSpearman rank correlation (MTEB vs MITE): rho={rho:.3f}, p={pval:.3f}")
    print(f"Number of models: {len(rows)}")

    if abs(rho) < 0.5:
        print("\n*** LOW CORRELATION: MTEB rankings DO NOT predict interaction task performance ***")
    elif abs(rho) < 0.7:
        print("\n*** MODERATE CORRELATION: MTEB rankings weakly predict interaction task performance ***")
    else:
        print("\n*** HIGH CORRELATION: MTEB rankings predict interaction task performance ***")

    return rho, pval


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Phase 1: SICK-R (fastest, most compelling)
    sick_task = SICKREntailmentTask()
    sick_results = run_task(sick_task, MODELS, "SICK-R: Entailment vs Similarity")

    # Print the key comparison
    rho, pval = print_ranking_comparison(sick_results)

    # Print per-class similarity analysis (the smoking gun)
    print(f"\n{'='*70}")
    print("  SMOKING GUN: Per-Class Similarity Analysis")
    print(f"{'='*70}")
    print(f"\n{'Model':<45} {'Sim(E)':>8} {'Sim(N)':>8} {'Sim(C)':>8} {'C > N?':>7}")
    print("-" * 80)
    for name, result in sick_results.items():
        if result is None:
            continue
        m = result.metrics
        sim_e = m.get("mean_sim_entailment", 0)
        sim_n = m.get("mean_sim_neutral", 0)
        sim_c = m.get("mean_sim_contradiction", 0)
        c_gt_n = "YES" if sim_c > sim_n else "no"
        print(f"{name:<45} {sim_e:>8.4f} {sim_n:>8.4f} {sim_c:>8.4f} {c_gt_n:>7}")

    # Save all results
    all_results = {}
    for name, result in sick_results.items():
        if result is not None:
            all_results[name] = {
                "task": result.task_name,
                "primary_metric": result.primary_metric,
                "primary_score": result.primary_score,
                "metrics": result.metrics,
            }

    results_file = output_dir / "sick_r_experiment.json"
    with open(results_file, "w") as f:
        json.dump({
            "task": "SICKREntailment",
            "rank_correlation": {"rho": rho, "p_value": pval},
            "models": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
