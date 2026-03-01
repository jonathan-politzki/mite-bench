#!/usr/bin/env python3
"""Run FEVER and SummEval experiments across multiple models."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mite.tasks import FEVERInteractionTask, SummEvalInteractionTask
from mite.models import SentenceTransformerModel

MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "thenlper/gte-base",
]


def run_task(task, models, task_label):
    """Run a single task across all models."""
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
            import traceback
            traceback.print_exc()
            results[model_name] = None

    return results


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # --- FEVER ---
    print("\n" + "=" * 70)
    print("  PHASE 1: FEVER Claim Verification")
    print("=" * 70)
    fever_task = FEVERInteractionTask()
    fever_task._max_pairs = 1000  # Cap for CPU speed
    fever_results = run_task(fever_task, MODELS, "FEVER: Support vs Refute")

    # Print FEVER analysis
    print(f"\n{'Model':<45} {'Sep Score':>10} {'AUROC':>8} {'Sim(S)':>8} {'Sim(R)':>8}")
    print("-" * 85)
    for name, r in fever_results.items():
        if r is None:
            continue
        m = r.metrics
        print(f"{name:<45} {m.get('separation_score',0):>10.4f} {m.get('auroc',0):>8.4f} "
              f"{m.get('mean_sim_supports',0):>8.4f} {m.get('mean_sim_refutes',0):>8.4f}")

    # Save FEVER
    fever_out = {}
    for name, r in fever_results.items():
        if r is not None:
            fever_out[name] = {"primary_score": r.primary_score, "metrics": r.metrics}
    with open(output_dir / "fever_experiment.json", "w") as f:
        json.dump({"task": "FEVERInteraction", "models": fever_out}, f, indent=2, default=str)
    print(f"\nFEVER results saved to {output_dir / 'fever_experiment.json'}")

    # --- SummEval ---
    print("\n" + "=" * 70)
    print("  PHASE 2: SummEval Summary Quality")
    print("=" * 70)
    summ_task = SummEvalInteractionTask()
    summ_results = run_task(summ_task, MODELS, "SummEval: Summary Quality")

    # Print SummEval analysis
    print(f"\n{'Model':<45} {'Mean Spearman':>14} {'Pairwise Acc':>13} {'Global Sp':>10}")
    print("-" * 85)
    for name, r in summ_results.items():
        if r is None:
            continue
        m = r.metrics
        print(f"{name:<45} {m.get('mean_spearman',0):>14.4f} "
              f"{m.get('pairwise_accuracy',0):>13.4f} {m.get('global_spearman',0):>10.4f}")

    # Save SummEval
    summ_out = {}
    for name, r in summ_results.items():
        if r is not None:
            summ_out[name] = {"primary_score": r.primary_score, "metrics": r.metrics}
    with open(output_dir / "summeval_experiment.json", "w") as f:
        json.dump({"task": "SummEvalQuality", "models": summ_out}, f, indent=2, default=str)
    print(f"\nSummEval results saved to {output_dir / 'summeval_experiment.json'}")

    # --- Cross-task ranking comparison ---
    print(f"\n{'='*70}")
    print("  CROSS-TASK RANKING ANALYSIS")
    print(f"{'='*70}")

    # Load SICK-R results
    sick_file = output_dir / "sick_r_experiment.json"
    if sick_file.exists():
        with open(sick_file) as f:
            sick_data = json.load(f)

        from scipy.stats import spearmanr

        # For each model, get MTEB score (from SICK-R mteb_spearman) and MITE scores
        all_models = set()
        for name in fever_results:
            if fever_results[name] is not None:
                all_models.add(name)

        print(f"\n{'Model':<40} {'MTEB(STS)':>10} {'MITE SICK':>10} {'MITE FEVER':>11} {'MITE Summ':>10}")
        print("-" * 85)
        mteb_scores = []
        mite_sick_scores = []
        mite_fever_scores = []
        mite_summ_scores = []
        model_list = []

        for name in sorted(all_models):
            sick_model = sick_data["models"].get(name)
            fever_r = fever_results.get(name)
            summ_r = summ_results.get(name)

            if sick_model is None or fever_r is None or summ_r is None:
                continue

            mteb = sick_model["metrics"]["mteb_spearman"]
            mite_s = sick_model["primary_score"]
            mite_f = fever_r.primary_score
            mite_sm = summ_r.primary_score

            model_list.append(name)
            mteb_scores.append(mteb)
            mite_sick_scores.append(mite_s)
            mite_fever_scores.append(mite_f)
            mite_summ_scores.append(mite_sm)

            short = name.split("/")[-1]
            print(f"{short:<40} {mteb:>10.4f} {mite_s:>10.4f} {mite_f:>11.4f} {mite_sm:>10.4f}")

        # Compute rank correlations
        if len(model_list) >= 5:
            print(f"\nRank correlations (MTEB STS rank vs MITE rank):")
            for label, scores in [
                ("SICK-R Entailment", mite_sick_scores),
                ("FEVER Claim Verif", mite_fever_scores),
                ("SummEval Quality", mite_summ_scores),
            ]:
                rho, p = spearmanr(mteb_scores, scores)
                print(f"  {label:<25} rho={rho:>7.3f}  p={p:.3f}")

            # Average MITE score across tasks (normalize each to [0,1] first)
            def normalize(arr):
                arr = np.array(arr)
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-9:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)

            avg_mite = (normalize(mite_sick_scores) + normalize(mite_fever_scores) + normalize(mite_summ_scores)) / 3
            rho, p = spearmanr(mteb_scores, avg_mite)
            print(f"  {'Average MITE':<25} rho={rho:>7.3f}  p={p:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
