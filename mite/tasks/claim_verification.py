"""Claim Verification Interaction Tasks: FEVER, ClimateFEVER, SciFact.

Three related tasks that test whether an embedding model can distinguish
claims that evidence SUPPORTS from claims that evidence REFUTES.  Both
SUPPORTS and REFUTES pairs are topically relevant (high retrieval score),
so a model that only captures topical similarity will fail to separate them.
This is the core interaction signal that MTEB retrieval metrics miss.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
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
    va = np.var(group_a, ddof=1)
    vb = np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve (binary labels: 1=positive, 0=negative).

    Uses the Mann-Whitney U statistic formulation for efficiency.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count how often a positive score exceeds a negative score
    n_pos, n_neg = len(pos), len(neg)
    # Sort-based O(n log n) implementation
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(-all_scores)  # descending
    sorted_labels = all_labels[order]
    # Accumulate
    tp = 0.0
    fp = 0.0
    auc = 0.0
    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return float(auc / (n_pos * n_neg)) if (n_pos * n_neg) > 0 else 0.5


# ── Label normalisation ──────────────────────────────────────────────────

_LABEL_MAP = {
    "SUPPORTS": "SUPPORTS",
    "supports": "SUPPORTS",
    "SUPPORT": "SUPPORTS",
    "support": "SUPPORTS",
    "REFUTES": "REFUTES",
    "refutes": "REFUTES",
    "REFUTE": "REFUTES",
    "refute": "REFUTES",
    "CONTRADICT": "REFUTES",
    "contradict": "REFUTES",
    0: "SUPPORTS",
    1: "REFUTES",
}


def _normalise_claim_label(raw: Any) -> str | None:
    """Map raw label to SUPPORTS / REFUTES, or None if ambiguous."""
    if isinstance(raw, (int, float, np.integer)):
        return _LABEL_MAP.get(int(raw))
    return _LABEL_MAP.get(str(raw).strip().upper())


# ── Abstract base for claim verification ─────────────────────────────────


class _ClaimVerificationBase(MITETask):
    """Shared evaluation logic for FEVER-family claim verification tasks."""

    task_type = "claim_verification"
    primary_metric = "separation_score"
    _max_pairs: int = 5000

    @abstractmethod
    def load_data(self) -> None: ...

    def get_pairs(self) -> list[tuple[str, str, str]]:
        """Return (claim, evidence, label) tuples where label is SUPPORTS or REFUTES."""
        self.ensure_loaded()
        return [
            (r["claim"], r["evidence"], r["label"])
            for r in self.data
        ]

    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate model on claim-evidence interaction.

        For each (claim, evidence) pair, compute cosine similarity.
        SUPPORTS pairs should have higher similarity than REFUTES pairs
        if the model captures the interaction (not just topical relevance).

        Metrics
        -------
        - separation_score: Cohen's d between SUPPORTS and REFUTES sims
        - auroc: AUROC treating SUPPORTS=1, REFUTES=0
        - mean_sim_supports / mean_sim_refutes
        - n_supports / n_refutes
        """
        self.ensure_loaded()

        emb_a, emb_b, labels = self.encode_pairs(model)
        sims = self.cosine_similarities(emb_a, emb_b)

        label_list: list[str] = labels  # type: ignore[assignment]
        supports_mask = np.array([l == "SUPPORTS" for l in label_list])
        refutes_mask = np.array([l == "REFUTES" for l in label_list])

        sims_supports = sims[supports_mask]
        sims_refutes = sims[refutes_mask]

        sep = _cohens_d(sims_supports, sims_refutes)

        # AUROC: SUPPORTS=1, REFUTES=0
        binary_labels = np.zeros(len(sims))
        binary_labels[supports_mask] = 1
        valid = supports_mask | refutes_mask
        auroc = _auroc(binary_labels[valid].astype(int), sims[valid])

        model_name = getattr(model, "model_name", getattr(model, "name", str(type(model).__name__)))

        metrics = {
            "separation_score": sep,
            "auroc": auroc,
            "mean_sim_supports": float(np.mean(sims_supports)) if len(sims_supports) > 0 else 0.0,
            "mean_sim_refutes": float(np.mean(sims_refutes)) if len(sims_refutes) > 0 else 0.0,
            "std_sim_supports": float(np.std(sims_supports)) if len(sims_supports) > 0 else 0.0,
            "std_sim_refutes": float(np.std(sims_refutes)) if len(sims_refutes) > 0 else 0.0,
            "n_supports": int(np.sum(supports_mask)),
            "n_refutes": int(np.sum(refutes_mask)),
            "n_total": len(sims),
        }

        return TaskResult(
            task_name=self.task_name,
            model_name=model_name,
            primary_metric=self.primary_metric,
            primary_score=sep,
            metrics=metrics,
        )


# ── FEVER ─────────────────────────────────────────────────────────────────


class FEVERInteractionTask(_ClaimVerificationBase):
    """FEVER fact verification: does evidence support or refute a claim?

    MTEB uses FEVER for retrieval (find the right evidence passage).
    MITE asks: given the evidence, does cosine similarity distinguish
    SUPPORTS from REFUTES?
    """

    task_name = "FEVERInteraction"
    description = (
        "Claim-evidence interaction on FEVER. Tests whether cosine "
        "similarity separates supporting from refuting evidence."
    )
    mteb_dataset = "mteb/fever"
    mteb_task_type = "Retrieval"

    def load_data(self) -> None:
        from datasets import load_dataset

        dataset = None
        sources = [
            ("copenlu/fever_gold_evidence", None),
            ("fever/fever", "v1.0"),
            ("fever/fever", None),
            ("pietrolesci/fever", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying FEVER dataset %s (config=%s)", ds_name, config)
                if config:
                    dataset = load_dataset(ds_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError("Could not load any FEVER dataset variant.")

        # Prefer test split, then validation, then train
        for split_name in ("test", "labelled_test", "validation", "paper_test", "train"):
            if split_name in dataset:
                split = dataset[split_name]
                break
        else:
            split = dataset[list(dataset.keys())[0]]

        self._parse_fever_split(split)

    def _parse_fever_split(self, split: Any) -> None:
        """Parse a FEVER-format split into (claim, evidence, label) records."""
        columns = set(split.column_names)
        records: list[dict[str, str]] = []

        # Determine evidence column
        evidence_col = next(
            (c for c in ("evidence", "evidence_sentence", "evidence_text", "evidences") if c in columns),
            None,
        )
        claim_col = next(
            (c for c in ("claim",) if c in columns),
            None,
        )
        label_col = next(
            (c for c in ("label", "gold_label", "verdict") if c in columns),
            None,
        )

        if claim_col is None or label_col is None:
            raise RuntimeError(f"Missing required columns. Available: {columns}")

        for row in split:
            lbl = _normalise_claim_label(row[label_col])
            if lbl is None:
                continue

            # Extract evidence text
            evidence_text = ""
            if evidence_col and row.get(evidence_col):
                ev = row[evidence_col]
                if isinstance(ev, str):
                    evidence_text = ev
                elif isinstance(ev, list):
                    # May be a list of evidence sentences or a nested structure
                    parts = []
                    for item in ev:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            parts.append(str(item.get("text", item.get("evidence", str(item)))))
                        elif isinstance(item, (list, tuple)):
                            # Nested list: [[annotation_id, evidence_id, page, sent_id, text], ...]
                            for sub in item:
                                if isinstance(sub, (list, tuple)) and len(sub) >= 5:
                                    parts.append(str(sub[-1]))
                                elif isinstance(sub, str):
                                    parts.append(sub)
                    evidence_text = " ".join(parts)

            if not evidence_text.strip():
                continue

            records.append({
                "claim": str(row[claim_col]),
                "evidence": evidence_text.strip(),
                "label": lbl,
            })

            if len(records) >= self._max_pairs:
                break

        # Balance classes
        records = self._balance_and_sample(records)
        self.data = records
        self._is_loaded = True
        n_sup = sum(1 for r in records if r["label"] == "SUPPORTS")
        n_ref = sum(1 for r in records if r["label"] == "REFUTES")
        logger.info("Loaded %d FEVER pairs (SUPPORTS=%d, REFUTES=%d)", len(records), n_sup, n_ref)

    @staticmethod
    def _balance_and_sample(
        records: list[dict[str, str]], max_per_class: int = 2500
    ) -> list[dict[str, str]]:
        """Balance classes and cap total size."""
        rng = np.random.RandomState(42)
        by_class: dict[str, list[dict[str, str]]] = {"SUPPORTS": [], "REFUTES": []}
        for r in records:
            by_class.setdefault(r["label"], []).append(r)

        balanced: list[dict[str, str]] = []
        min_count = min(len(v) for v in by_class.values()) if by_class else 0
        target = min(min_count, max_per_class)

        for cls_records in by_class.values():
            idx = rng.choice(len(cls_records), size=min(target, len(cls_records)), replace=False)
            balanced.extend(cls_records[i] for i in idx)

        rng.shuffle(balanced)
        return balanced

    def mteb_metric_name(self) -> str:
        return "ndcg_at_10"


# ── ClimateFEVER ──────────────────────────────────────────────────────────


class ClimateFEVERInteractionTask(_ClaimVerificationBase):
    """ClimateFEVER: climate-science claim verification.

    Tests the same interaction signal as FEVER but in the climate domain.
    """

    task_name = "ClimateFEVERInteraction"
    description = (
        "Claim-evidence interaction on ClimateFEVER. Tests whether cosine "
        "similarity separates supporting from refuting climate-science evidence."
    )
    mteb_dataset = "mteb/climate-fever"
    mteb_task_type = "Retrieval"

    def load_data(self) -> None:
        from datasets import load_dataset

        dataset = None
        sources = [
            ("climate_fever", None),
            ("climatefever/climate_fever", None),
            ("alistairewj/climate-fever", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying ClimateFEVER dataset %s", ds_name)
                if config:
                    dataset = load_dataset(ds_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError("Could not load ClimateFEVER dataset.")

        # Use test split if available
        for split_name in ("test", "test_dataset", "validation", "train"):
            if split_name in dataset:
                split = dataset[split_name]
                break
        else:
            split = dataset[list(dataset.keys())[0]]

        self._parse_climate_fever(split)

    def _parse_climate_fever(self, split: Any) -> None:
        """Parse ClimateFEVER into (claim, evidence, label) records.

        ClimateFEVER typically has: claim, claim_label, evidences (list of dicts
        with evidence, evidence_label, article, etc.)
        """
        columns = set(split.column_names)
        records: list[dict[str, str]] = []

        for row in split:
            claim = str(row.get("claim", ""))
            if not claim.strip():
                continue

            # ClimateFEVER stores per-evidence labels in the evidences list
            evidences = row.get("evidences", [])
            if isinstance(evidences, list):
                for ev in evidences:
                    if isinstance(ev, dict):
                        ev_text = str(ev.get("evidence", ""))
                        ev_label = ev.get("evidence_label", ev.get("label", ""))
                        lbl = _normalise_claim_label(ev_label)
                        if lbl and ev_text.strip():
                            records.append({
                                "claim": claim,
                                "evidence": ev_text.strip(),
                                "label": lbl,
                            })
            else:
                # Fallback: claim-level label
                claim_label = row.get("claim_label", row.get("label", ""))
                lbl = _normalise_claim_label(claim_label)
                ev_text = str(row.get("evidence", ""))
                if lbl and ev_text.strip():
                    records.append({
                        "claim": claim,
                        "evidence": ev_text.strip(),
                        "label": lbl,
                    })

            if len(records) >= self._max_pairs * 2:
                break

        records = FEVERInteractionTask._balance_and_sample(records)
        self.data = records
        self._is_loaded = True
        n_sup = sum(1 for r in records if r["label"] == "SUPPORTS")
        n_ref = sum(1 for r in records if r["label"] == "REFUTES")
        logger.info("Loaded %d ClimateFEVER pairs (SUPPORTS=%d, REFUTES=%d)", len(records), n_sup, n_ref)

    def mteb_metric_name(self) -> str:
        return "ndcg_at_10"


# ── SciFact ───────────────────────────────────────────────────────────────


class SciFActInteractionTask(_ClaimVerificationBase):
    """SciFact: scientific claim verification against paper abstracts.

    Tests whether cosine similarity can separate claims that a scientific
    abstract supports from claims that it refutes.
    """

    task_name = "SciFActInteraction"
    description = (
        "Claim-evidence interaction on SciFact. Tests whether cosine "
        "similarity separates supporting from refuting scientific evidence."
    )
    mteb_dataset = "mteb/scifact"
    mteb_task_type = "Retrieval"

    def load_data(self) -> None:
        from datasets import load_dataset

        dataset = None
        sources = [
            ("allenai/scifact", "claims"),
            ("allenai/scifact", None),
            ("scifact", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying SciFact dataset %s (config=%s)", ds_name, config)
                if config:
                    dataset = load_dataset(ds_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError("Could not load SciFact dataset.")

        # SciFact is small, so use all available splits
        all_records: list[dict[str, str]] = []

        # Also try to load the corpus for evidence text
        corpus: dict[int, str] = {}
        try:
            corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
            for split_name in corpus_ds:
                for row in corpus_ds[split_name]:
                    doc_id = row.get("doc_id", row.get("id", ""))
                    title = str(row.get("title", ""))
                    abstract_sents = row.get("abstract", [])
                    if isinstance(abstract_sents, list):
                        abstract = " ".join(str(s) for s in abstract_sents)
                    else:
                        abstract = str(abstract_sents)
                    corpus[doc_id] = f"{title}. {abstract}".strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load SciFact corpus: %s", exc)

        for split_name in dataset:
            split = dataset[split_name]
            columns = set(split.column_names)

            for row in split:
                claim = str(row.get("claim", ""))
                if not claim.strip():
                    continue

                # SciFact has cited_doc_ids and evidence structure
                # evidence is typically {doc_id: [{sentences: [...], label: "SUPPORT"/"CONTRADICT"}]}
                evidence_dict = row.get("evidence", {})
                cited_docs = row.get("cited_doc_ids", [])

                if isinstance(evidence_dict, dict):
                    for doc_id_str, evidences in evidence_dict.items():
                        doc_id = int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str
                        for ev in (evidences if isinstance(evidences, list) else [evidences]):
                            if isinstance(ev, dict):
                                lbl_raw = ev.get("label", "")
                                lbl = _normalise_claim_label(lbl_raw)
                                if lbl is None:
                                    # SciFact uses SUPPORT/CONTRADICT
                                    if str(lbl_raw).upper() == "SUPPORT":
                                        lbl = "SUPPORTS"
                                    elif str(lbl_raw).upper() in ("CONTRADICT", "CONTRADICTION"):
                                        lbl = "REFUTES"
                                if lbl is None:
                                    continue
                                # Get evidence text
                                ev_text = corpus.get(doc_id, "")
                                if not ev_text:
                                    # Try sentence indices
                                    sents = ev.get("sentences", [])
                                    if isinstance(sents, list) and sents:
                                        ev_text = " ".join(str(s) for s in sents)
                                if ev_text.strip():
                                    all_records.append({
                                        "claim": claim,
                                        "evidence": ev_text.strip(),
                                        "label": lbl,
                                    })
                elif "evidence" in columns or "abstract" in columns:
                    # Flat format fallback
                    ev_text = str(row.get("evidence", row.get("abstract", "")))
                    lbl_raw = row.get("label", row.get("gold_label", ""))
                    lbl = _normalise_claim_label(lbl_raw)
                    if lbl is None and isinstance(lbl_raw, str):
                        if lbl_raw.upper() == "SUPPORT":
                            lbl = "SUPPORTS"
                        elif lbl_raw.upper() in ("CONTRADICT", "CONTRADICTION"):
                            lbl = "REFUTES"
                    if lbl and ev_text.strip():
                        all_records.append({
                            "claim": claim,
                            "evidence": ev_text.strip(),
                            "label": lbl,
                        })

        all_records = FEVERInteractionTask._balance_and_sample(all_records, max_per_class=2500)
        self.data = all_records
        self._is_loaded = True
        n_sup = sum(1 for r in all_records if r["label"] == "SUPPORTS")
        n_ref = sum(1 for r in all_records if r["label"] == "REFUTES")
        logger.info("Loaded %d SciFact pairs (SUPPORTS=%d, REFUTES=%d)", len(all_records), n_sup, n_ref)

    def mteb_metric_name(self) -> str:
        return "ndcg_at_10"
