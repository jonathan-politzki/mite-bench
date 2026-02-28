"""SummEval Summary Quality Interaction Task.

SummEval contains 100 CNN/DailyMail articles, each with ~16 model-generated
summaries rated by humans on consistency, relevance, coherence, and fluency.

MTEB measures: cosine(source, summary) correlated with human scores.
MITE measures: can cosine similarity *rank* summaries by quality for the
same source document?  This is a per-document interaction question -- it
tests whether the embedding captures how well a summary serves its source,
not just topical overlap.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mite.tasks.base import MITETask, TaskResult

logger = logging.getLogger(__name__)


class SummEvalInteractionTask(MITETask):
    """SummEval summary quality ranking.

    For each source document with multiple summaries, test whether cosine
    similarity between source and summary predicts the human quality rating.

    Primary metric: mean per-source Spearman correlation between cosine
    similarity ranking and human quality ranking.
    """

    task_name = "SummEvalQuality"
    task_type = "summary_quality"
    description = (
        "Summary quality ranking on SummEval. Tests whether cosine "
        "similarity between source documents and summaries predicts "
        "human-judged summary quality."
    )
    mteb_dataset = "mteb/summeval"
    mteb_task_type = "Summarization"
    primary_metric = "mean_spearman"

    # Quality dimensions to average for the composite score
    QUALITY_DIMS = ("consistency", "relevance", "coherence", "fluency")

    def __init__(self) -> None:
        super().__init__()
        # Grouped: source_id -> list of {source, summary, quality, ...}
        self._grouped: dict[str, list[dict[str, Any]]] = {}

    def load_data(self) -> None:
        """Load SummEval dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = None
        sources = [
            ("mteb/summeval", None),
            ("Yale-LILY/summeval", None),
            ("summeval", None),
        ]

        for ds_name, config in sources:
            try:
                logger.info("Trying SummEval dataset: %s (config=%s)", ds_name, config)
                if config:
                    dataset = load_dataset(ds_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, trust_remote_code=True)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", ds_name, exc)
                dataset = None

        if dataset is None:
            raise RuntimeError("Could not load SummEval dataset.")

        # Use test split if available, otherwise any available split
        for split_name in ("test", "validation", "train"):
            if split_name in dataset:
                split = dataset[split_name]
                break
        else:
            split = dataset[list(dataset.keys())[0]]

        self._parse_summeval(split)

    def _parse_summeval(self, split: Any) -> None:
        """Parse SummEval into grouped records.

        SummEval format varies by source.  Common patterns:
        1. Each row is one (source, summary) pair with human scores.
        2. Each row is one source with multiple summaries and scores.
        """
        columns = set(split.column_names)
        logger.info("SummEval columns: %s", columns)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Detect format
        has_machine_summaries = "machine_summaries" in columns
        has_summary = any(c in columns for c in ("summary", "decoded", "text"))

        if has_machine_summaries:
            # Format: each row = one source article with multiple summaries
            self._parse_grouped_format(split, grouped)
        elif has_summary:
            # Format: each row = one (source, summary) pair
            self._parse_flat_format(split, grouped)
        else:
            # Try to detect from first row
            sample = next(iter(split))
            logger.info("SummEval sample keys: %s", list(sample.keys()))

            # Try human_summaries / machine_summaries pattern
            if any(k for k in sample if "summar" in k.lower()):
                self._parse_grouped_format(split, grouped)
            else:
                self._parse_flat_format(split, grouped)

        # Filter to sources with 2+ summaries at different quality levels
        filtered: dict[str, list[dict[str, Any]]] = {}
        for src_id, entries in grouped.items():
            if len(entries) < 2:
                continue
            qualities = [e["quality"] for e in entries]
            if max(qualities) - min(qualities) < 0.01:
                continue  # no variation
            filtered[src_id] = entries

        if not filtered:
            raise RuntimeError(
                "No source documents with multiple quality-varied summaries found."
            )

        self._grouped = filtered
        self.data = [e for entries in filtered.values() for e in entries]
        self._is_loaded = True
        logger.info(
            "Loaded SummEval: %d sources, %d total (source, summary) pairs",
            len(filtered), len(self.data),
        )

    def _parse_grouped_format(
        self, split: Any, grouped: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Parse format where each row contains multiple summaries."""
        columns = set(split.column_names)

        source_col = next(
            (c for c in ("text", "source", "article", "document") if c in columns),
            None,
        )
        summaries_col = next(
            (c for c in ("machine_summaries", "summaries", "decoded") if c in columns),
            None,
        )

        # Score columns — may be lists aligned with summaries
        score_cols = {}
        for dim in self.QUALITY_DIMS:
            for candidate in (dim, f"{dim}_scores", f"human_{dim}", f"expert_{dim}"):
                if candidate in columns:
                    score_cols[dim] = candidate
                    break

        for row_idx, row in enumerate(split):
            source_text = str(row.get(source_col, "")) if source_col else ""

            # Try ID column, fall back to row index
            src_id = str(row.get("id", row.get("article_id", row.get("doc_id", row_idx))))

            if not source_text.strip():
                continue

            summaries = row.get(summaries_col, []) if summaries_col else []
            if not isinstance(summaries, list):
                continue

            for s_idx, summary in enumerate(summaries):
                summary_text = str(summary) if not isinstance(summary, dict) else str(summary.get("text", summary))
                if not summary_text.strip():
                    continue

                # Compute quality score
                quality_scores: list[float] = []
                for dim, col_name in score_cols.items():
                    scores = row.get(col_name, [])
                    if isinstance(scores, list) and s_idx < len(scores):
                        val = scores[s_idx]
                        if isinstance(val, (int, float)):
                            quality_scores.append(float(val))
                        elif isinstance(val, list):
                            # Average of multiple annotators
                            quality_scores.append(float(np.mean([v for v in val if isinstance(v, (int, float))])))
                    elif isinstance(scores, (int, float)):
                        quality_scores.append(float(scores))

                quality = float(np.mean(quality_scores)) if quality_scores else 0.0

                grouped[src_id].append({
                    "source": source_text,
                    "summary": summary_text,
                    "quality": quality,
                    "source_id": src_id,
                    "summary_idx": s_idx,
                })

    def _parse_flat_format(
        self, split: Any, grouped: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Parse format where each row is one (source, summary) pair."""
        columns = set(split.column_names)

        source_col = next(
            (c for c in ("text", "source", "article", "document", "src") if c in columns),
            None,
        )
        summary_col = next(
            (c for c in ("summary", "decoded", "prediction", "hyp") if c in columns),
            None,
        )
        id_col = next(
            (c for c in ("id", "article_id", "doc_id", "source_id") if c in columns),
            None,
        )

        if source_col is None or summary_col is None:
            logger.warning("Could not identify source/summary columns from: %s", columns)
            return

        for row_idx, row in enumerate(split):
            source_text = str(row.get(source_col, ""))
            summary_text = str(row.get(summary_col, ""))
            if not source_text.strip() or not summary_text.strip():
                continue

            src_id = str(row.get(id_col, "")) if id_col else ""
            if not src_id:
                # Group by source text hash
                src_id = str(hash(source_text[:200]))

            # Compute quality score
            quality_scores: list[float] = []
            for dim in self.QUALITY_DIMS:
                for candidate in (dim, f"{dim}_score", f"human_{dim}", f"expert_{dim}"):
                    if candidate in columns:
                        val = row.get(candidate)
                        if isinstance(val, (int, float)):
                            quality_scores.append(float(val))
                        elif isinstance(val, list):
                            vals = [v for v in val if isinstance(v, (int, float))]
                            if vals:
                                quality_scores.append(float(np.mean(vals)))
                        break

            quality = float(np.mean(quality_scores)) if quality_scores else 0.0

            grouped[src_id].append({
                "source": source_text,
                "summary": summary_text,
                "quality": quality,
                "source_id": src_id,
                "summary_idx": row_idx,
            })

    def get_pairs(self) -> list[tuple[str, str, float]]:
        """Return (source_document, summary, quality_score) tuples."""
        self.ensure_loaded()
        return [
            (r["source"], r["summary"], float(r["quality"]))
            for r in self.data
        ]

    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate summary quality ranking.

        For each source document with multiple summaries:
        1. Compute cosine similarity between source and each summary.
        2. Compute Spearman correlation between cosine ranking and
           quality ranking.
        3. Compute pairwise accuracy: for pairs of summaries of the
           same source, does higher cosine = higher quality?
        4. Average metrics across source documents.
        """
        self.ensure_loaded()

        # Encode all source documents and summaries
        all_sources = [r["source"] for r in self.data]
        all_summaries = [r["summary"] for r in self.data]

        emb_src: np.ndarray = model.encode(all_sources, batch_size=32)
        emb_sum: np.ndarray = model.encode(all_summaries, batch_size=32)
        all_sims = self.cosine_similarities(emb_src, emb_sum)

        # Per-source evaluation
        spearmans: list[float] = []
        pairwise_correct = 0
        pairwise_total = 0
        offset = 0

        for src_id, entries in self._grouped.items():
            n = len(entries)
            sims = all_sims[offset: offset + n]
            qualities = np.array([e["quality"] for e in entries])
            offset += n

            # Spearman correlation
            if len(set(qualities)) >= 2 and n >= 3:
                rho, _ = spearmanr(sims, qualities)
                if not np.isnan(rho):
                    spearmans.append(float(rho))

            # Pairwise accuracy
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(qualities[i] - qualities[j]) > 0.01:
                        pairwise_total += 1
                        if (sims[i] > sims[j]) == (qualities[i] > qualities[j]):
                            pairwise_correct += 1

        mean_spearman = float(np.mean(spearmans)) if spearmans else 0.0
        median_spearman = float(np.median(spearmans)) if spearmans else 0.0
        pairwise_acc = pairwise_correct / pairwise_total if pairwise_total > 0 else 0.5

        # Global correlation (all pairs, ignoring source grouping)
        all_qualities = np.array([r["quality"] for r in self.data])
        global_rho = float("nan")
        if len(set(all_qualities)) >= 2:
            rho, _ = spearmanr(all_sims, all_qualities)
            if not np.isnan(rho):
                global_rho = float(rho)

        model_name = getattr(model, "model_name", getattr(model, "name", str(type(model).__name__)))

        metrics = {
            "mean_spearman": mean_spearman,
            "median_spearman": median_spearman,
            "pairwise_accuracy": pairwise_acc,
            "global_spearman": global_rho,
            "n_sources_evaluated": len(spearmans),
            "n_sources_total": len(self._grouped),
            "n_pairs_total": len(self.data),
            "n_pairwise_comparisons": pairwise_total,
            "mean_quality": float(np.mean(all_qualities)),
            "std_quality": float(np.std(all_qualities)),
        }

        return TaskResult(
            task_name=self.task_name,
            model_name=model_name,
            primary_metric=self.primary_metric,
            primary_score=mean_spearman,
            metrics=metrics,
        )

    def mteb_metric_name(self) -> str:
        """MTEB evaluates SummEval with Spearman correlation."""
        return "cos_sim_spearman"
