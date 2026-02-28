"""Answer Quality Interaction Tasks: FiQA and CQADupstack.

These tasks test whether cosine similarity can predict answer *quality* --
not just whether an answer is topically relevant (which MTEB retrieval
measures), but whether it is the *best* answer to a question.

FiQA: Among answers retrieved for a financial question, which one best
      resolves it?  Uses graded relevance scores.

CQADupstack: Among candidate answers on StackExchange, which was accepted?
             Uses binary accepted/not-accepted labels.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mite.tasks.base import MITETask, TaskResult

logger = logging.getLogger(__name__)


# ── Evaluation helpers ────────────────────────────────────────────────────


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve (binary labels: 1=positive, 0=negative)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]
    tp = 0.0
    auc = 0.0
    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            auc += tp
    return float(auc / (n_pos * n_neg)) if (n_pos * n_neg) > 0 else 0.5


# ── FiQA ──────────────────────────────────────────────────────────────────


class FiQAInteractionTask(MITETask):
    """FiQA answer quality: does cosine similarity predict answer relevance?

    MTEB uses FiQA for retrieval (find relevant answers).  MITE asks:
    among the relevant answers to a question, does cosine similarity rank
    them by quality?  This requires understanding the *interaction* between
    question intent and answer substance, not just topical overlap.

    Primary metric: mean per-query Spearman correlation between cosine
    similarity ranking and relevance ranking.
    """

    task_name = "FiQAAnswerQuality"
    task_type = "answer_quality"
    description = (
        "Answer quality ranking on FiQA. Tests whether cosine similarity "
        "predicts graded answer relevance for financial questions."
    )
    mteb_dataset = "mteb/fiqa"
    mteb_task_type = "Retrieval"
    primary_metric = "mean_spearman"

    def __init__(self) -> None:
        super().__init__()
        # Grouped data: query_id -> list of (question, answer, relevance)
        self._grouped: dict[str, list[dict[str, Any]]] = {}

    def load_data(self) -> None:
        """Load FiQA in BEIR format: corpus, queries, qrels."""
        from datasets import load_dataset

        dataset = None
        sources = [
            ("mteb/fiqa", None),
            ("BeIR/fiqa", None),
            ("fiqa", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying FiQA dataset %s (config=%s)", ds_name, config)
                if config:
                    dataset = load_dataset(ds_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError("Could not load FiQA dataset.")

        # BEIR format has corpus, queries, and qrels splits/subsets
        available_splits = list(dataset.keys())
        logger.info("FiQA splits: %s", available_splits)

        # Try to parse BEIR-style format (corpus + queries + qrels)
        if self._try_load_beir_format(dataset):
            return

        # Fallback: try direct format with question/answer columns
        self._try_load_direct_format(dataset)

    def _try_load_beir_format(self, dataset: Any) -> bool:
        """Attempt to load from BEIR-style corpus/queries/qrels format."""
        try:
            # Load corpus
            corpus: dict[str, str] = {}
            if "corpus" in dataset:
                for row in dataset["corpus"]:
                    doc_id = str(row.get("_id", row.get("id", "")))
                    title = str(row.get("title", ""))
                    text = str(row.get("text", ""))
                    corpus[doc_id] = f"{title}. {text}".strip(". ") if title else text

            # Load queries
            queries: dict[str, str] = {}
            if "queries" in dataset:
                for row in dataset["queries"]:
                    q_id = str(row.get("_id", row.get("id", "")))
                    queries[q_id] = str(row.get("text", row.get("query", "")))

            if not corpus or not queries:
                return False

            # Load qrels from the test or default split
            qrels: dict[str, dict[str, int]] = defaultdict(dict)
            for split_name in ("test", "validation", "default"):
                if split_name in dataset:
                    split = dataset[split_name]
                    cols = set(split.column_names)
                    if "query-id" in cols and "corpus-id" in cols and "score" in cols:
                        for row in split:
                            q_id = str(row["query-id"])
                            c_id = str(row["corpus-id"])
                            score = int(row["score"])
                            if q_id in queries and c_id in corpus:
                                qrels[q_id][c_id] = score
                        break

            if not qrels:
                return False

            # Group by query: only keep queries with 2+ answers at different relevance levels
            grouped: dict[str, list[dict[str, Any]]] = {}
            for q_id, doc_scores in qrels.items():
                if len(doc_scores) < 2:
                    continue
                scores = set(doc_scores.values())
                if len(scores) < 2:
                    continue  # need variation in relevance for Spearman
                entries = []
                for c_id, rel in doc_scores.items():
                    entries.append({
                        "question": queries[q_id],
                        "answer": corpus[c_id],
                        "relevance": rel,
                        "query_id": q_id,
                    })
                grouped[q_id] = entries

            if not grouped:
                return False

            # Cap to ~5000 total pairs
            self._cap_grouped(grouped, max_total=5000)
            self._grouped = grouped

            all_records = []
            for entries in grouped.values():
                all_records.extend(entries)
            self.data = all_records
            self._is_loaded = True
            logger.info(
                "Loaded FiQA BEIR: %d queries, %d total pairs",
                len(grouped), len(all_records),
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning("BEIR parse failed: %s", exc)
            return False

    def _try_load_direct_format(self, dataset: Any) -> None:
        """Fallback: parse dataset with question/answer columns directly."""
        for split_name in ("test", "validation", "train"):
            if split_name not in dataset:
                continue
            split = dataset[split_name]
            columns = set(split.column_names)

            question_col = next(
                (c for c in ("question", "query", "text_a", "sentence1") if c in columns), None
            )
            answer_col = next(
                (c for c in ("answer", "text", "text_b", "sentence2") if c in columns), None
            )
            score_col = next(
                (c for c in ("score", "relevance", "label", "rating") if c in columns), None
            )

            if question_col is None or answer_col is None:
                continue

            # Group by question text
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for i, row in enumerate(split):
                q = str(row[question_col])
                a = str(row[answer_col])
                rel = float(row.get(score_col, 1)) if score_col else 1.0
                grouped[q].append({
                    "question": q,
                    "answer": a,
                    "relevance": rel,
                    "query_id": str(i),
                })

            # Filter to queries with 2+ answers at different relevance levels
            filtered = {
                q: entries for q, entries in grouped.items()
                if len(entries) >= 2 and len(set(e["relevance"] for e in entries)) >= 2
            }

            if filtered:
                self._cap_grouped(filtered, max_total=5000)
                self._grouped = filtered
                self.data = [e for entries in filtered.values() for e in entries]
                self._is_loaded = True
                logger.info(
                    "Loaded FiQA direct: %d queries, %d total pairs",
                    len(filtered), len(self.data),
                )
                return

        raise RuntimeError("Could not parse FiQA dataset in any format.")

    @staticmethod
    def _cap_grouped(
        grouped: dict[str, list[dict[str, Any]]], max_total: int = 5000
    ) -> None:
        """Remove queries to keep total pairs under max_total (in-place)."""
        total = sum(len(v) for v in grouped.values())
        if total <= max_total:
            return
        rng = np.random.RandomState(42)
        keys = list(grouped.keys())
        rng.shuffle(keys)
        kept = 0
        to_remove = []
        for k in keys:
            if kept + len(grouped[k]) > max_total:
                to_remove.append(k)
            else:
                kept += len(grouped[k])
        for k in to_remove:
            del grouped[k]

    def get_pairs(self) -> list[tuple[str, str, float]]:
        """Return (question, answer, relevance_score) tuples."""
        self.ensure_loaded()
        return [
            (r["question"], r["answer"], float(r["relevance"]))
            for r in self.data
        ]

    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate answer quality ranking.

        For each query with multiple graded answers:
        1. Compute cosine similarity between the query and each answer.
        2. Compute Spearman correlation between cosine ranking and
           relevance ranking.
        3. Average Spearman across queries.
        """
        self.ensure_loaded()

        # Encode all texts
        all_questions = [r["question"] for r in self.data]
        all_answers = [r["answer"] for r in self.data]
        emb_q: np.ndarray = model.encode(all_questions, batch_size=64)
        emb_a: np.ndarray = model.encode(all_answers, batch_size=64)
        all_sims = self.cosine_similarities(emb_q, emb_a)

        # Map each record to its index
        idx = 0
        record_to_idx: dict[int, int] = {}
        for i in range(len(self.data)):
            record_to_idx[i] = i

        # Per-query Spearman
        spearmans: list[float] = []
        pairwise_correct = 0
        pairwise_total = 0
        offset = 0

        for q_id, entries in self._grouped.items():
            n = len(entries)
            sims = all_sims[offset: offset + n]
            rels = np.array([e["relevance"] for e in entries])
            offset += n

            # Spearman
            if len(set(rels)) >= 2 and n >= 3:
                rho, _ = spearmanr(sims, rels)
                if not np.isnan(rho):
                    spearmans.append(float(rho))

            # Pairwise accuracy
            for i in range(n):
                for j in range(i + 1, n):
                    if rels[i] != rels[j]:
                        pairwise_total += 1
                        if (sims[i] > sims[j]) == (rels[i] > rels[j]):
                            pairwise_correct += 1

        mean_spearman = float(np.mean(spearmans)) if spearmans else 0.0
        pairwise_acc = pairwise_correct / pairwise_total if pairwise_total > 0 else 0.5

        model_name = getattr(model, "model_name", getattr(model, "name", str(type(model).__name__)))

        metrics = {
            "mean_spearman": mean_spearman,
            "median_spearman": float(np.median(spearmans)) if spearmans else 0.0,
            "pairwise_accuracy": pairwise_acc,
            "n_queries_evaluated": len(spearmans),
            "n_queries_total": len(self._grouped),
            "n_pairs_total": len(self.data),
            "n_pairwise_comparisons": pairwise_total,
        }

        return TaskResult(
            task_name=self.task_name,
            model_name=model_name,
            primary_metric=self.primary_metric,
            primary_score=mean_spearman,
            metrics=metrics,
        )

    def mteb_metric_name(self) -> str:
        return "ndcg_at_10"


# ── CQADupstack ──────────────────────────────────────────────────────────


class CQADupstackInteractionTask(MITETask):
    """CQADupstack answer acceptance: does cosine predict which answer was accepted?

    MTEB uses CQADupstack subsets for retrieval.  MITE asks: among candidate
    answers, can cosine similarity distinguish the accepted answer from
    non-accepted ones?  This is an interaction-level question about answer
    quality, not topical relevance.

    Primary metric: AUROC on accepted (1) vs non-accepted (0).
    """

    task_name = "CQADupstackAnswerQuality"
    task_type = "answer_quality"
    description = (
        "Answer acceptance prediction on CQADupstack. Tests whether cosine "
        "similarity predicts which StackExchange answer was accepted."
    )
    mteb_dataset = "mteb/cqadupstack"
    mteb_task_type = "Retrieval"
    primary_metric = "auroc"

    # Subsets to load (a representative sample of CQADupstack domains)
    SUBSETS = ("android", "programmers", "stats", "english", "physics")

    def load_data(self) -> None:
        """Load CQADupstack subsets in BEIR format."""
        from datasets import load_dataset

        all_records: list[dict[str, Any]] = []

        for subset in self.SUBSETS:
            for ds_name in [f"mteb/cqadupstack-{subset}", f"BeIR/cqadupstack/{subset}"]:
                try:
                    logger.info("Trying CQADupstack subset: %s", ds_name)
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                    records = self._parse_beir_subset(dataset, subset)
                    all_records.extend(records)
                    logger.info("Loaded %d pairs from %s", len(records), ds_name)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load %s: %s", ds_name, exc)
                    continue

        if not all_records:
            # Try loading as a single dataset with subsets
            try:
                for subset in self.SUBSETS:
                    try:
                        dataset = load_dataset("mteb/cqadupstack", subset, trust_remote_code=True)
                        records = self._parse_beir_subset(dataset, subset)
                        all_records.extend(records)
                        logger.info("Loaded %d pairs from mteb/cqadupstack/%s", len(records), subset)
                    except Exception:  # noqa: BLE001
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fallback loading failed: %s", exc)

        if not all_records:
            raise RuntimeError(
                "Could not load any CQADupstack subset. "
                "Tried subsets: " + ", ".join(self.SUBSETS)
            )

        # Balance and sample
        rng = np.random.RandomState(42)
        pos = [r for r in all_records if r["accepted"] == 1]
        neg = [r for r in all_records if r["accepted"] == 0]
        target = min(len(pos), len(neg), 2500)

        if len(pos) > target:
            idx = rng.choice(len(pos), size=target, replace=False)
            pos = [pos[i] for i in idx]
        if len(neg) > target:
            idx = rng.choice(len(neg), size=target, replace=False)
            neg = [neg[i] for i in idx]

        self.data = pos + neg
        rng.shuffle(self.data)
        self._is_loaded = True
        logger.info(
            "Loaded %d CQADupstack pairs (accepted=%d, not_accepted=%d)",
            len(self.data), len(pos), len(neg),
        )

    def _parse_beir_subset(
        self, dataset: Any, subset_name: str
    ) -> list[dict[str, Any]]:
        """Parse a BEIR-format CQADupstack subset."""
        available = list(dataset.keys())

        # Build corpus
        corpus: dict[str, str] = {}
        if "corpus" in dataset:
            for row in dataset["corpus"]:
                doc_id = str(row.get("_id", row.get("id", "")))
                title = str(row.get("title", ""))
                text = str(row.get("text", ""))
                corpus[doc_id] = f"{title}. {text}".strip(". ") if title else text

        # Build queries
        queries: dict[str, str] = {}
        if "queries" in dataset:
            for row in dataset["queries"]:
                q_id = str(row.get("_id", row.get("id", "")))
                queries[q_id] = str(row.get("text", row.get("query", "")))

        if not corpus or not queries:
            # Try direct format
            return self._parse_direct_format(dataset, subset_name)

        # Parse qrels: relevance 1+ = accepted, 0 = not accepted
        # In BEIR, score > 0 means relevant; we treat higher score as "accepted"
        records: list[dict[str, Any]] = []
        for split_name in ("test", "validation", "default"):
            if split_name not in dataset:
                continue
            split = dataset[split_name]
            cols = set(split.column_names)
            if "query-id" in cols and "corpus-id" in cols and "score" in cols:
                # Group by query to identify best answer
                query_docs: dict[str, list[tuple[str, int]]] = defaultdict(list)
                for row in split:
                    q_id = str(row["query-id"])
                    c_id = str(row["corpus-id"])
                    score = int(row["score"])
                    if q_id in queries and c_id in corpus:
                        query_docs[q_id].append((c_id, score))

                for q_id, doc_scores in query_docs.items():
                    if len(doc_scores) < 2:
                        continue
                    max_score = max(s for _, s in doc_scores)
                    for c_id, score in doc_scores:
                        records.append({
                            "question": queries[q_id],
                            "answer": corpus[c_id],
                            "accepted": 1 if score == max_score else 0,
                            "relevance_score": score,
                            "subset": subset_name,
                        })
                break

        return records

    def _parse_direct_format(
        self, dataset: Any, subset_name: str
    ) -> list[dict[str, Any]]:
        """Fallback parser for non-BEIR format datasets."""
        records: list[dict[str, Any]] = []
        for split_name in dataset:
            split = dataset[split_name]
            columns = set(split.column_names)

            q_col = next(
                (c for c in ("question", "query", "title") if c in columns), None
            )
            a_col = next(
                (c for c in ("answer", "text", "body") if c in columns), None
            )
            label_col = next(
                (c for c in ("label", "score", "accepted", "is_accepted") if c in columns), None
            )

            if q_col is None or a_col is None:
                continue

            for row in split:
                accepted = 0
                if label_col:
                    val = row[label_col]
                    accepted = 1 if (isinstance(val, (int, float)) and val > 0) or val is True else 0
                records.append({
                    "question": str(row[q_col]),
                    "answer": str(row[a_col]),
                    "accepted": accepted,
                    "relevance_score": accepted,
                    "subset": subset_name,
                })

        return records

    def get_pairs(self) -> list[tuple[str, str, int]]:
        """Return (question, answer, accepted) tuples."""
        self.ensure_loaded()
        return [
            (r["question"], r["answer"], int(r["accepted"]))
            for r in self.data
        ]

    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate answer acceptance prediction.

        Compute cosine similarity between questions and answers, then
        measure how well similarity predicts answer acceptance.
        """
        self.ensure_loaded()

        emb_a, emb_b, labels = self.encode_pairs(model)
        sims = self.cosine_similarities(emb_a, emb_b)

        labels_arr = np.array(labels, dtype=int)
        auroc = _auroc(labels_arr, sims)

        # Pairwise accuracy: within same question, does higher sim = accepted?
        # (approximated by global pairwise comparison)
        pos_sims = sims[labels_arr == 1]
        neg_sims = sims[labels_arr == 0]

        # Cohen's d separation
        sep = 0.0
        if len(pos_sims) > 1 and len(neg_sims) > 1:
            na, nb = len(pos_sims), len(neg_sims)
            va = np.var(pos_sims, ddof=1)
            vb = np.var(neg_sims, ddof=1)
            pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
            if pooled > 1e-12:
                sep = float((np.mean(pos_sims) - np.mean(neg_sims)) / pooled)

        model_name = getattr(model, "model_name", getattr(model, "name", str(type(model).__name__)))

        metrics = {
            "auroc": auroc,
            "separation_score": sep,
            "mean_sim_accepted": float(np.mean(pos_sims)) if len(pos_sims) > 0 else 0.0,
            "mean_sim_rejected": float(np.mean(neg_sims)) if len(neg_sims) > 0 else 0.0,
            "n_accepted": int(np.sum(labels_arr == 1)),
            "n_rejected": int(np.sum(labels_arr == 0)),
            "n_total": len(sims),
        }

        # Per-subset breakdown if available
        subsets = set(r.get("subset", "") for r in self.data)
        if len(subsets) > 1:
            for subset in sorted(subsets):
                if not subset:
                    continue
                mask = np.array([r.get("subset") == subset for r in self.data])
                sub_sims = sims[mask]
                sub_labels = labels_arr[mask]
                if len(sub_sims) > 10:
                    metrics[f"auroc_{subset}"] = _auroc(sub_labels, sub_sims)

        return TaskResult(
            task_name=self.task_name,
            model_name=model_name,
            primary_metric=self.primary_metric,
            primary_score=auroc,
            metrics=metrics,
        )

    def mteb_metric_name(self) -> str:
        return "ndcg_at_10"
