#!/usr/bin/env python3
"""Fast FEVER + SummEval experiment: 3 models, smaller samples."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mite.tasks import FEVERInteractionTask, SummEvalInteractionTask
from mite.models import SentenceTransformerModel

# Just 3 representative models: small, medium, best-MTEB
MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
]


def run_task(task, models, task_label):
    print(f"\n{'='*60}")
    print(f"  TASK: {task_label}")
    print(f"{'='*60}")

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

    # --- FEVER (cap at 1000 pairs for speed) ---
    print("\n" + "=" * 60)
    print("  PHASE 1: FEVER Claim Verification")
    print("=" * 60)
    fever_task = FEVERInteractionTask()
    fever_task._max_pairs = 1000  # Cap for speed
    fever_results = run_task(fever_task, MODELS, "FEVER: Support vs Refute")

    # Print FEVER summary
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
    print(f"\nFEVER results -> {output_dir / 'fever_experiment.json'}")

    # --- SummEval ---
    print("\n" + "=" * 60)
    print("  PHASE 2: SummEval Summary Quality")
    print("=" * 60)
    summ_task = SummEvalInteractionTask()
    summ_results = run_task(summ_task, MODELS, "SummEval: Summary Quality")

    # Print SummEval summary
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
    print(f"\nSummEval results -> {output_dir / 'summeval_experiment.json'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
