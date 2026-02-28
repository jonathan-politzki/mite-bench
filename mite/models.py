"""Model wrappers for MITE evaluation.

Provides a uniform ``encode()`` interface over local (sentence-transformers)
and API-based (OpenAI, Voyage, Cohere) embedding models.  API wrappers are
optional -- they gracefully degrade when the corresponding SDK is not
installed.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from tqdm import tqdm


class ModelWrapper(ABC):
    """Base class for embedding models used in MITE evaluation."""

    model_name: str = ""

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode *texts* into embeddings.

        Parameters
        ----------
        texts : list of str
            Input texts to encode.
        batch_size : int
            Batch size for encoding (interpretation is model-specific).

        Returns
        -------
        np.ndarray of shape ``(len(texts), dim)``
        """

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """Row-wise cosine similarity between paired embedding matrices.

        Parameters
        ----------
        emb_a, emb_b : np.ndarray of shape ``(n, dim)``

        Returns
        -------
        np.ndarray of shape ``(n,)``
        """
        norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
        norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
        return np.sum(norm_a * norm_b, axis=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"


# ---------------------------------------------------------------------------
# Local: sentence-transformers
# ---------------------------------------------------------------------------


class SentenceTransformerModel(ModelWrapper):
    """Wraps any model loadable by the ``sentence-transformers`` library."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name_or_path
        self._model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 512,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# API: OpenAI
# ---------------------------------------------------------------------------


class OpenAIModel(ModelWrapper):
    """Wraps the OpenAI embeddings API.

    Requires the ``openai`` package and a valid ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI wrapper requires the `openai` package. "
                "Install it with: pip install 'mite-bench[api]'"
            ) from exc

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"OpenAI/{self.model_name}"):
            batch = texts[start : start + batch_size]
            # Replace empty strings (API rejects them)
            batch = [t if t.strip() else " " for t in batch]
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            batch_embs = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            all_embeddings.extend(batch_embs)
        emb = np.stack(all_embeddings, axis=0)
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        return emb


# ---------------------------------------------------------------------------
# API: Voyage AI
# ---------------------------------------------------------------------------


class VoyageModel(ModelWrapper):
    """Wraps the Voyage AI embeddings API.

    Requires the ``voyageai`` package and a valid ``VOYAGE_API_KEY`` env var.
    """

    def __init__(
        self,
        model_name: str = "voyage-3-large",
        api_key: str | None = None,
    ) -> None:
        try:
            import voyageai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Voyage wrapper requires the `voyageai` package. "
                "Install it with: pip install 'mite-bench[api]'"
            ) from exc

        self.model_name = model_name
        self._client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Voyage/{self.model_name}"):
            batch = texts[start : start + batch_size]
            batch = [t if t.strip() else " " for t in batch]
            result = self._client.embed(batch, model=self.model_name)
            batch_embs = [np.array(e, dtype=np.float32) for e in result.embeddings]
            all_embeddings.extend(batch_embs)
        emb = np.stack(all_embeddings, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        return emb


# ---------------------------------------------------------------------------
# API: Cohere
# ---------------------------------------------------------------------------


class CohereModel(ModelWrapper):
    """Wraps the Cohere embeddings API.

    Requires the ``cohere`` package and a valid ``CO_API_KEY`` env var.
    """

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        input_type: str = "search_document",
        api_key: str | None = None,
    ) -> None:
        try:
            import cohere  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Cohere wrapper requires the `cohere` package. "
                "Install it with: pip install 'mite-bench[api]'"
            ) from exc

        self.model_name = model_name
        self._input_type = input_type
        self._client = cohere.ClientV2(api_key=api_key or os.environ.get("CO_API_KEY"))

    def encode(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Cohere/{self.model_name}"):
            batch = texts[start : start + batch_size]
            batch = [t if t.strip() else " " for t in batch]
            response = self._client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self._input_type,
                embedding_types=["float"],
            )
            batch_embs = [
                np.array(e, dtype=np.float32) for e in response.embeddings.float_
            ]
            all_embeddings.extend(batch_embs)
        emb = np.stack(all_embeddings, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        return emb
