"""SICK-R Entailment Interaction Task.

The smoking-gun experiment: SICK-R has BOTH similarity scores (1-5) AND
entailment labels (ENTAILMENT / CONTRADICTION / NEUTRAL) on the same 9,927
sentence pairs.  MTEB evaluates Spearman correlation with relatedness scores.
MITE evaluates whether cosine similarity can separate entailment classes --
a fundamentally different (interaction-level) question.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mite.evaluation import separation_score as _cohens_d
from mite.tasks.base import MITETask, TaskResult

logger = logging.getLogger(__name__)


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-averaged F1 over all unique classes."""
    classes = sorted(set(y_true) | set(y_pred))
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ── Label normalisation ──────────────────────────────────────────────────

# yangwang825/sick uses integer labels: 0=contradiction, 1=neutral, 2=entailment
_INT_LABEL_MAP = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

_STR_LABEL_MAP = {
    "ENTAILMENT": "ENTAILMENT",
    "entailment": "ENTAILMENT",
    "A_ENTAILS_B": "ENTAILMENT",
    "CONTRADICTION": "CONTRADICTION",
    "contradiction": "CONTRADICTION",
    "NEUTRAL": "NEUTRAL",
    "neutral": "NEUTRAL",
}


def _normalise_label(raw) -> str | None:
    if isinstance(raw, (int, float)):
        return _INT_LABEL_MAP.get(int(raw))
    return _STR_LABEL_MAP.get(str(raw).strip())


# ── Task implementation ──────────────────────────────────────────────────


class SICKREntailmentTask(MITETask):
    """Evaluate entailment separation on the SICK dataset.

    MTEB uses SICK-R for Semantic Textual Similarity (Spearman with
    relatedness).  MITE asks: can cosine similarity *separate* entailment
    from contradiction?  A model that merely captures topical similarity
    will score high on STS but poorly here.
    """

    task_name = "SICKREntailment"
    task_type = "entailment"
    description = (
        "Entailment-class separation on the SICK dataset. "
        "Tests whether cosine similarity distinguishes entailment from "
        "contradiction on the same pairs MTEB uses for STS."
    )
    mteb_dataset = "mteb/sickr-sts"
    mteb_task_type = "STS"
    primary_metric = "separation_score"

    def __init__(self) -> None:
        super().__init__()
        self._relatedness: list[float] = []

    # ── Data loading ──────────────────────────────────────────────────

    def load_data(self) -> None:
        """Load SICK with both entailment labels and relatedness scores.

        Strategy: load yangwang825/sick (has entailment labels) and
        mteb/sickr-sts (has relatedness scores), then join on sentence text.
        """
        from datasets import load_dataset

        # 1. Load entailment labels from yangwang825/sick
        logger.info("Loading entailment labels from yangwang825/sick...")
        ds_ent = load_dataset("yangwang825/sick", split="test")

        # 2. Load relatedness scores from mteb/sickr-sts
        logger.info("Loading relatedness scores from mteb/sickr-sts...")
        ds_sim = load_dataset("mteb/sickr-sts", split="test")

        # 3. Build relatedness lookup by sentence pair
        sim_lookup: dict[tuple[str, str], float] = {}
        for row in ds_sim:
            key = (row["sentence1"].strip(), row["sentence2"].strip())
            sim_lookup[key] = row["score"]

        # 4. Join: iterate entailment dataset, match relatedness scores
        records: list[dict[str, Any]] = []
        relatedness_scores: list[float] = []

        for row in ds_ent:
            lbl = _normalise_label(row["label"])
            if lbl is None:
                continue
            sent_a = str(row["text1"])
            sent_b = str(row["text2"])
            records.append({
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "entailment_label": lbl,
            })
            # Match relatedness score
            key = (sent_a.strip(), sent_b.strip())
            rel = sim_lookup.get(key, float("nan"))
            relatedness_scores.append(rel)

        self.data = records
        self._relatedness = relatedness_scores
        self._is_loaded = True

        n_matched = sum(1 for r in relatedness_scores if not np.isnan(r))
        from collections import Counter
        label_counts = Counter(r["entailment_label"] for r in records)
        logger.info(
            "Loaded %d SICK pairs (E=%d / N=%d / C=%d), %d with relatedness scores",
            len(records),
            label_counts["ENTAILMENT"],
            label_counts["NEUTRAL"],
            label_counts["CONTRADICTION"],
            n_matched,
        )

    # ── Pair interface ────────────────────────────────────────────────

    def get_pairs(self) -> list[tuple[str, str, str]]:
        """Return (sentence_a, sentence_b, entailment_label) tuples."""
        self.ensure_loaded()
        return [
            (r["sentence_a"], r["sentence_b"], r["entailment_label"])
            for r in self.data
        ]

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate an embedding model on SICK entailment separation.

        Steps
        -----
        1. Encode sentence_A and sentence_B.
        2. Compute row-wise cosine similarities.
        3. Compute separation score (Cohen's d) between ENTAILMENT and
           CONTRADICTION similarity distributions.
        4. Find optimal thresholds to classify E/N/C and compute macro F1.
        5. Report per-class mean similarity.
        6. Optionally compute MTEB-style Spearman with relatedness scores.
        """
        self.ensure_loaded()

        emb_a, emb_b, labels = self.encode_pairs(model)
        sims = self.cosine_similarities(emb_a, emb_b)

        # Group similarities by class
        label_list: list[str] = labels  # type: ignore[assignment]
        class_sims: dict[str, np.ndarray] = {}
        for cls in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION"):
            mask = np.array([l == cls for l in label_list])
            class_sims[cls] = sims[mask]

        # --- Separation score (primary metric) ---
        sep = _cohens_d(class_sims["ENTAILMENT"], class_sims["CONTRADICTION"])

        # --- Per-class mean similarity ---
        mean_sims = {
            cls: float(np.mean(arr)) if len(arr) > 0 else 0.0
            for cls, arr in class_sims.items()
        }

        # --- Threshold-based classification using dev portion ---
        # Use first 20% as dev to find thresholds, evaluate on remaining 80%
        n = len(sims)
        n_dev = max(1, n // 5)
        idx = np.random.RandomState(42).permutation(n)
        dev_idx, test_idx = idx[:n_dev], idx[n_dev:]

        best_f1 = 0.0
        best_thresholds = (0.5, 0.3)
        # Grid search over threshold pairs
        for t_high in np.arange(0.2, 0.9, 0.05):
            for t_low in np.arange(0.0, t_high, 0.05):
                preds_dev = []
                for i in dev_idx:
                    s = sims[i]
                    if s >= t_high:
                        preds_dev.append("ENTAILMENT")
                    elif s <= t_low:
                        preds_dev.append("CONTRADICTION")
                    else:
                        preds_dev.append("NEUTRAL")
                true_dev = [label_list[i] for i in dev_idx]
                f1 = _macro_f1(true_dev, preds_dev)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = (float(t_high), float(t_low))

        # Evaluate on test portion with best thresholds
        t_high, t_low = best_thresholds
        preds_test = []
        for i in test_idx:
            s = sims[i]
            if s >= t_high:
                preds_test.append("ENTAILMENT")
            elif s <= t_low:
                preds_test.append("CONTRADICTION")
            else:
                preds_test.append("NEUTRAL")
        true_test = [label_list[i] for i in test_idx]
        macro_f1 = _macro_f1(true_test, preds_test)

        # --- MTEB-style Spearman with relatedness (if available) ---
        relatedness = np.array(self._relatedness)
        valid_mask = ~np.isnan(relatedness)
        mteb_spearman = float("nan")
        if np.sum(valid_mask) > 10:
            rho, _ = spearmanr(sims[valid_mask], relatedness[valid_mask])
            mteb_spearman = float(rho)

        # --- Build result ---
        model_name = getattr(model, "model_name", getattr(model, "name", str(type(model).__name__)))
        metrics = {
            "separation_score": sep,
            "macro_f1": macro_f1,
            "threshold_high": t_high,
            "threshold_low": t_low,
            "mean_sim_entailment": mean_sims.get("ENTAILMENT", 0.0),
            "mean_sim_neutral": mean_sims.get("NEUTRAL", 0.0),
            "mean_sim_contradiction": mean_sims.get("CONTRADICTION", 0.0),
            "mteb_spearman": mteb_spearman,
            "n_entailment": len(class_sims["ENTAILMENT"]),
            "n_neutral": len(class_sims["NEUTRAL"]),
            "n_contradiction": len(class_sims["CONTRADICTION"]),
        }

        return TaskResult(
            task_name=self.task_name,
            model_name=model_name,
            primary_metric=self.primary_metric,
            primary_score=sep,
            metrics=metrics,
            predictions=[
                {"index": int(i), "similarity": float(sims[i]), "true_label": label_list[i]}
                for i in range(n)
            ],
        )

    def mteb_metric_name(self) -> str:
        """MTEB evaluates SICK-R with Spearman correlation on relatedness."""
        return "cos_sim_spearman"
