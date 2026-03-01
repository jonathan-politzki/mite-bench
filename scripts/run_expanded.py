#!/usr/bin/env python3
"""Expanded experiment: 15+ models spanning weak-to-strong MTEB range.

The problem with our initial run: all 9 models had MTEB STS in 0.77-0.80.
That's a 4% window of a 50% range. We need models from 0.40 to 0.85+
to properly test whether MTEB rank predicts interaction performance.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from mite.tasks import SICKREntailmentTask, SummEvalInteractionTask
from mite.models import SentenceTransformerModel

# Models spanning the full MTEB quality range
MODELS = [
    # --- Weak (expected MTEB STS < 0.70) ---
    "average_word_embeddings_glove.6B.300d",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/all-distilroberta-v1",

    # --- Medium (our original set, MTEB STS 0.77-0.80) ---
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "thenlper/gte-base",

    # --- Strong (MTEB STS > 0.80) ---
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-large",
    "intfloat/multilingual-e5-large",
]


def run_sick_r(models):
    """Run SICK-R entailment experiment on all models."""
    print("\n" + "=" * 70)
    print("  SICK-R ENTAILMENT (expanded model set)")
    print("=" * 70)

    task = SICKREntailmentTask()
    print("Loading data...")
    task.load_data()
    print(f"  {len(task.data)} pairs loaded")

    results = {}
    for model_name in models:
        print(f"\n  [{len(results)+1}/{len(models)}] {model_name}")
        t0 = time.time()
        try:
            model = SentenceTransformerModel(model_name, trust_remote_code=True)
            result = task.evaluate(model)
            elapsed = time.time() - t0
            results[model_name] = result
            m = result.metrics
            print(f"    MTEB Spearman:  {m.get('mteb_spearman', 0):.4f}")
            print(f"    Separation:     {result.primary_score:.4f}")
            print(f"    Sim(Ent/Contra): {m.get('mean_sim_entailment',0):.3f} / {m.get('mean_sim_contradiction',0):.3f}")
            print(f"    Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None

    return results


def run_summeval(models):
    """Run SummEval experiment on all models."""
    print("\n" + "=" * 70)
    print("  SUMMEVAL QUALITY (expanded model set)")
    print("=" * 70)

    task = SummEvalInteractionTask()
    print("Loading data...")
    task.load_data()
    print(f"  {len(task.data)} pairs loaded")

    results = {}
    for model_name in models:
        print(f"\n  [{len(results)+1}/{len(models)}] {model_name}")
        t0 = time.time()
        try:
            model = SentenceTransformerModel(model_name, trust_remote_code=True)
            result = task.evaluate(model)
            elapsed = time.time() - t0
            results[model_name] = result
            m = result.metrics
            print(f"    Mean Spearman:    {m.get('mean_spearman', 0):.4f}")
            print(f"    Pairwise Acc:     {m.get('pairwise_accuracy', 0):.4f}")
            print(f"    Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None

    return results


def analyze_and_save(sick_results, summ_results, output_dir):
    """Compute rank correlations and save everything."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- SICK-R Analysis ---
    print("\n" + "=" * 70)
    print("  SICK-R RANK CORRELATION ANALYSIS")
    print("=" * 70)

    mteb_scores, mite_scores, names = [], [], []
    for name, r in sick_results.items():
        if r is None:
            continue
        mteb = r.metrics.get("mteb_spearman", 0)
        mite = r.primary_score
        if mteb > 0:
            names.append(name)
            mteb_scores.append(mteb)
            mite_scores.append(mite)

    print(f"\n{'Model':<45} {'MTEB STS':>10} {'MITE Sep':>10}")
    print("-" * 68)
    # Sort by MTEB score
    order = np.argsort(mteb_scores)
    for i in order:
        short = names[i].split("/")[-1]
        print(f"{short:<45} {mteb_scores[i]:>10.4f} {mite_scores[i]:>10.4f}")

    rho, p = spearmanr(mteb_scores, mite_scores)
    print(f"\nMTEB range: {min(mteb_scores):.4f} - {max(mteb_scores):.4f} (spread: {max(mteb_scores)-min(mteb_scores):.4f})")
    print(f"Models: {len(names)}")
    print(f"Spearman rho = {rho:.4f}, p = {p:.6f}")
    if p < 0.01:
        print("*** HIGHLY SIGNIFICANT (p < 0.01) ***")
    elif p < 0.05:
        print("** SIGNIFICANT (p < 0.05) **")
    else:
        print(f"  Not significant (p = {p:.3f})")

    # Save SICK-R
    sick_out = {"task": "SICKREntailment", "rank_correlation": {"rho": rho, "p_value": p}, "models": {}}
    for name, r in sick_results.items():
        if r is not None:
            sick_out["models"][name] = {
                "task": r.task_name,
                "primary_metric": r.primary_metric,
                "primary_score": r.primary_score,
                "metrics": r.metrics,
            }
    with open(output_dir / "sick_r_experiment.json", "w") as f:
        json.dump(sick_out, f, indent=2, default=str)

    # --- SummEval Analysis ---
    if summ_results:
        print("\n" + "=" * 70)
        print("  SUMMEVAL RANK CORRELATION ANALYSIS")
        print("=" * 70)

        # Match models between SICK (which has mteb_spearman) and SummEval
        mteb_s, summ_s, names_s = [], [], []
        for name, r in summ_results.items():
            if r is None:
                continue
            sick_r = sick_results.get(name)
            if sick_r is None:
                continue
            mteb = sick_r.metrics.get("mteb_spearman", 0)
            if mteb > 0:
                names_s.append(name)
                mteb_s.append(mteb)
                summ_s.append(r.primary_score)

        print(f"\n{'Model':<45} {'MTEB STS':>10} {'SummEval':>10}")
        print("-" * 68)
        order = np.argsort(mteb_s)
        for i in order:
            short = names_s[i].split("/")[-1]
            print(f"{short:<45} {mteb_s[i]:>10.4f} {summ_s[i]:>10.4f}")

        if len(names_s) >= 5:
            rho_s, p_s = spearmanr(mteb_s, summ_s)
            print(f"\nModels: {len(names_s)}")
            print(f"Spearman rho = {rho_s:.4f}, p = {p_s:.6f}")
        else:
            print(f"\nOnly {len(names_s)} models — need >= 5 for correlation")

        summ_out = {"task": "SummEvalQuality", "models": {}}
        for name, r in summ_results.items():
            if r is not None:
                summ_out["models"][name] = {"primary_score": r.primary_score, "metrics": r.metrics}
        with open(output_dir / "summeval_experiment.json", "w") as f:
            json.dump(summ_out, f, indent=2, default=str)

    print("\nDone!")


def main():
    sick_results = run_sick_r(MODELS)
    summ_results = run_summeval(MODELS)
    analyze_and_save(sick_results, summ_results, "results")


if __name__ == "__main__":
    main()
