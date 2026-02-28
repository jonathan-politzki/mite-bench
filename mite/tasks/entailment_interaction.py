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

from mite.tasks.base import MITETask, TaskResult

logger = logging.getLogger(__name__)

# ── Evaluation helpers ────────────────────────────────────────────────────

def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


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

_ENTAILMENT_MAP = {
    "ENTAILMENT": "ENTAILMENT",
    "entailment": "ENTAILMENT",
    "A": "ENTAILMENT",
    "CONTRADICTION": "CONTRADICTION",
    "contradiction": "CONTRADICTION",
    "C": "CONTRADICTION",
    "NEUTRAL": "NEUTRAL",
    "neutral": "NEUTRAL",
    "B": "NEUTRAL",
}


def _normalise_label(raw: str) -> str | None:
    return _ENTAILMENT_MAP.get(str(raw).strip())


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
        """Load SICK with entailment labels.

        We try several HuggingFace dataset ids until one succeeds.  The
        original SICK dataset contains entailment labels; the MTEB version
        may only have relatedness scores.
        """
        from datasets import load_dataset

        dataset = None
        sources = [
            ("sentence-transformers/sick", None),
            ("mariannefelice/SICK", None),
            ("sick", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying dataset %s (config=%s)", ds_name, config)
                dataset = load_dataset(ds_name, config, trust_remote_code=True)
                # Quick check: does it have an entailment label column?
                sample = next(iter(dataset[list(dataset.keys())[0]]))
                has_entailment = any(
                    k in sample
                    for k in ("label", "entailment_label", "entailment_judgment")
                )
                if has_entailment:
                    break
                logger.info("Dataset %s loaded but no entailment labels found, trying next.", ds_name)
                dataset = None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError(
                "Could not load SICK dataset with entailment labels. "
                "Install `datasets` and ensure network access."
            )

        # Use the test split if available, otherwise fall back to the full dataset
        if "test" in dataset:
            split = dataset["test"]
        elif "default" in dataset:
            split = dataset["default"]
        else:
            split = dataset[list(dataset.keys())[0]]

        # Determine column names ─────────────────────────────────────
        columns = set(split.column_names)
        sent_a_col = next(
            (c for c in ("sentence_A", "sentence1", "sent_1", "premise") if c in columns),
            None,
        )
        sent_b_col = next(
            (c for c in ("sentence_B", "sentence2", "sent_2", "hypothesis") if c in columns),
            None,
        )
        label_col = next(
            (c for c in ("entailment_label", "entailment_judgment", "label") if c in columns),
            None,
        )
        relatedness_col = next(
            (c for c in ("relatedness_score", "score", "label_score", "relatedness") if c in columns),
            None,
        )

        if sent_a_col is None or sent_b_col is None or label_col is None:
            raise RuntimeError(
                f"Could not identify required columns. Available: {columns}"
            )

        # Build internal data list ────────────────────────────────────
        records: list[dict[str, Any]] = []
        relatedness_scores: list[float] = []

        for row in split:
            lbl = _normalise_label(row[label_col])
            if lbl is None:
                continue
            records.append(
                {
                    "sentence_a": str(row[sent_a_col]),
                    "sentence_b": str(row[sent_b_col]),
                    "entailment_label": lbl,
                }
            )
            if relatedness_col and row.get(relatedness_col) is not None:
                relatedness_scores.append(float(row[relatedness_col]))
            else:
                relatedness_scores.append(float("nan"))

        self.data = records
        self._relatedness = relatedness_scores
        self._is_loaded = True
        logger.info(
            "Loaded %d SICK pairs (%s / %s / %s)",
            len(records),
            sum(1 for r in records if r["entailment_label"] == "ENTAILMENT"),
            sum(1 for r in records if r["entailment_label"] == "NEUTRAL"),
            sum(1 for r in records if r["entailment_label"] == "CONTRADICTION"),
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
