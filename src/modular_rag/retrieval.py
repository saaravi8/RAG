"""Default retrieval and reranking implementations."""

import math
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from .errors import ComponentContractError, OptionalDependencyError
from .models import SearchResult
from .ports import Embedder, VectorStore


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("{} must be a positive integer.".format(name))
    if value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return value


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        top_k = _positive_int(top_k, "top_k")
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, top_k=top_k, filters=filters)


class KeywordReranker:
    """Small local reranker showing where a model-based reranker can plug in."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> Sequence[SearchResult]:
        top_k = _positive_int(top_k, "top_k")
        query_terms = set(self._TOKEN.findall(query.lower()))

        def score(result: SearchResult):
            text_terms = set(self._TOKEN.findall(result.chunk.text.lower()))
            overlap = len(query_terms & text_terms)
            return (-overlap, -result.score, result.chunk.id)

        return tuple(sorted(results, key=score)[:top_k])


class CrossEncoderReranker:
    """Rerank retrieved chunks with a Sentence Transformers cross-encoder.

    A cross-encoder scores each query/chunk pair jointly, which is slower than
    vector search but typically gives a better final ordering. Keep retrieval
    broad and use this component only on the small candidate set supplied by
    :class:`RAGService`.

    Pass an already loaded ``model`` to share it across components or tests.
    The object only needs a Sentence Transformers-compatible ``predict``
    method. Returned ``SearchResult`` scores are the cross-encoder scores.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = False,
        model: Optional[Any] = None,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        batch_size = _positive_int(batch_size, "batch_size")
        if max_length is not None:
            max_length = _positive_int(max_length, "max_length")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.show_progress_bar = show_progress_bar
        self._model = model if model is not None else self._load_model()

        if not callable(getattr(self._model, "predict", None)):
            raise TypeError("model must provide a callable predict method.")

    def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> Sequence[SearchResult]:
        top_k = _positive_int(top_k, "top_k")

        candidates = tuple(results)
        if not candidates:
            return ()

        pairs = [(query, result.chunk.text) for result in candidates]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        scores = self._scores(raw_scores, expected=len(candidates))

        scored = tuple(
            SearchResult(result.chunk, score)
            for result, score in zip(candidates, scores)
        )
        return tuple(
            result
            for _, result in sorted(
                enumerate(scored),
                key=lambda item: (-item[1].score, item[0]),
            )[:top_k]
        )

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise OptionalDependencyError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install it with: pip install 'modular-rag[reranking]'"
            ) from exc

        options = {"device": self.device}
        if self.max_length is not None:
            options["max_length"] = self.max_length
        return CrossEncoder(self.model_name, **options)

    @staticmethod
    def _scores(raw_scores: Any, *, expected: int) -> Tuple[float, ...]:
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if isinstance(values, (str, bytes, bytearray)):
            raise ComponentContractError(
                "Cross-encoder returned malformed relevance scores."
            )
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise ComponentContractError(
                "Cross-encoder returned malformed relevance scores."
            ) from exc
        scores = []
        for value in raw_values:
            if isinstance(value, (bool, str, bytes, bytearray)):
                raise ComponentContractError(
                    "Cross-encoder returned malformed relevance scores."
                )
            try:
                scores.append(float(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ComponentContractError(
                    "Cross-encoder returned malformed relevance scores."
                ) from exc
        normalized_scores = tuple(scores)

        if len(normalized_scores) != expected:
            raise ComponentContractError(
                "Cross-encoder returned {} scores for {} candidates.".format(
                    len(normalized_scores), expected
                )
            )
        if any(not math.isfinite(score) for score in normalized_scores):
            raise ComponentContractError(
                "Cross-encoder returned non-finite relevance scores."
            )
        return normalized_scores
