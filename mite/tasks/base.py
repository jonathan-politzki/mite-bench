"""Abstract base class for all MITE tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskResult:
    """Results from evaluating a model on a MITE task.

    Stores the primary metric used for ranking along with all computed
    metrics and optional per-example predictions for downstream analysis.
    """

    task_name: str
    model_name: str
    # Primary metric for ranking
    primary_metric: str
    primary_score: float
    # All metrics
    metrics: dict[str, float] = field(default_factory=dict)
    # Optional per-example predictions
    predictions: list[Any] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TaskResult(task={self.task_name!r}, model={self.model_name!r}, "
            f"{self.primary_metric}={self.primary_score:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for JSON export."""
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "metrics": self.metrics,
        }


class MITETask(ABC):
    """Abstract base class for MITE interaction tasks.

    Every MITE task re-frames an existing MTEB dataset as an *interaction*
    evaluation: instead of asking "which document is most relevant?", we ask
    "does the embedding capture the quality of the interaction between the
    two texts?"

    Subclasses must define the class-level metadata attributes and implement
    all abstract methods.
    """

    # ── Metadata (subclasses must override) ──────────────────────────────
    task_name: str = ""
    task_type: str = ""  # "claim_verification", "entailment", "answer_quality", "summary_quality"
    description: str = ""
    mteb_dataset: str = ""  # corresponding MTEB dataset name
    mteb_task_type: str = ""  # what MTEB uses this dataset for (e.g. "Retrieval", "STS")
    primary_metric: str = ""

    def __init__(self) -> None:
        self.data: Any = None
        self._is_loaded: bool = False

    # ── Abstract interface ───────────────────────────────────────────────

    @abstractmethod
    def load_data(self) -> None:
        """Load and preprocess the dataset (typically from HuggingFace).

        After this call ``self.data`` must be populated and
        ``self._is_loaded`` must be ``True``.
        """

    @abstractmethod
    def get_pairs(self) -> list[tuple[str, str, Any]]:
        """Return a list of ``(text_a, text_b, label)`` tuples.

        ``label`` semantics depend on the task type:
        * claim_verification  -> bool or int (supported / refuted)
        * entailment          -> float (relatedness score)
        * answer_quality      -> int or float (quality / relevance grade)
        * summary_quality     -> float (summary quality score)
        """

    @abstractmethod
    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate *model* on this task.

        Parameters
        ----------
        model:
            Any object that exposes an ``encode(texts, batch_size)`` method
            returning an ``np.ndarray`` of shape ``(n, dim)``.

        Returns
        -------
        TaskResult
            Populated result dataclass with all computed metrics.
        """

    @abstractmethod
    def mteb_metric_name(self) -> str:
        """Return the canonical MTEB metric name used for this dataset.

        This is needed so the comparison module can look up the
        corresponding MTEB leaderboard score for each model.
        """

    # ── Convenience helpers ──────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        """Call ``load_data`` if it has not been called yet."""
        if not self._is_loaded:
            self.load_data()

    def encode_pairs(
        self, model: Any, batch_size: int = 64
    ) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        """Encode all pairs with *model* and return (emb_a, emb_b, labels).

        This is a convenience method that subclasses can call inside
        ``evaluate`` to avoid duplicating the encode-and-collect logic.
        """
        self.ensure_loaded()
        pairs = self.get_pairs()
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]

        emb_a: np.ndarray = model.encode(texts_a, batch_size=batch_size)
        emb_b: np.ndarray = model.encode(texts_b, batch_size=batch_size)
        return emb_a, emb_b, labels

    @staticmethod
    def cosine_similarities(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """Row-wise cosine similarity between two embedding matrices."""
        # L2-normalise each row
        norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
        norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
        return np.sum(norm_a * norm_b, axis=1)
