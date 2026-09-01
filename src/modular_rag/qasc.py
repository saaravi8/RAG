"""Optional Query-Adaptive Semantic Chunking retrieval strategy."""

import hashlib
import math
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from rag_ingestion import Document

from .errors import ComponentContractError, VectorDimensionError
from .models import Chunk, SearchResult, SentenceSpan, Vector
from .ports import Embedder, SentenceSegmenter


@dataclass(frozen=True)
class QASCConfig:
    """Parameters for query-adaptive seed selection and window construction."""

    seed_percentile: float = 75.0
    window_radius: int = 3
    decay: float = 0.3
    gap_tolerance: int = 2
    chunk_threshold_factor: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.seed_percentile <= 100.0:
            raise ValueError("seed_percentile must be between 0 and 100.")
        if self.window_radius < 0:
            raise ValueError("window_radius cannot be negative.")
        if self.decay < 0.0:
            raise ValueError("decay cannot be negative.")
        if self.gap_tolerance < 0:
            raise ValueError("gap_tolerance cannot be negative.")
        if not 0.0 <= self.chunk_threshold_factor <= 1.0:
            raise ValueError("chunk_threshold_factor must be between 0 and 1.")


@dataclass(frozen=True)
class _SentenceDocument:
    text: str
    metadata: Mapping[str, Any]
    sentences: Tuple[SentenceSpan, ...]
    vectors: Tuple[Vector, ...]


@dataclass(frozen=True)
class _Window:
    start: int
    end: int
    seeds: Tuple[int, ...]
    score: float


class QASCRetriever:
    """Build query-specific sentence windows from an optional sentence index.

    Sentence embeddings are computed once when a document is indexed. At query
    time, QASC scores every eligible sentence against the query, selects seed
    sentences using an adaptive percentile, expands their context windows,
    filters by aggregate relevance, and merges overlapping nearby windows.
    """

    def __init__(
        self,
        segmenter: SentenceSegmenter,
        embedder: Embedder,
        *,
        config: Optional[QASCConfig] = None,
    ) -> None:
        self.segmenter = segmenter
        self.embedder = embedder
        self.config = config or QASCConfig()
        self._documents: Dict[str, _SentenceDocument] = {}
        self._dimension: Optional[int] = None
        self._lock = RLock()

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)

    def replace_document(self, document: Document) -> None:
        sentences = tuple(self.segmenter.segment(document))
        if not sentences:
            raise ComponentContractError(
                "Sentence segmenter returned no sentences for the document."
            )

        document_ids = {sentence.document_id for sentence in sentences}
        if len(document_ids) != 1:
            raise ComponentContractError(
                "All sentences from one document must share one document_id."
            )
        document_id = next(iter(document_ids))
        expected_indexes = tuple(range(len(sentences)))
        actual_indexes = tuple(sentence.index for sentence in sentences)
        if actual_indexes != expected_indexes:
            raise ComponentContractError(
                "Sentence indexes must be contiguous and start at zero."
            )

        embedded = tuple(
            self.embedder.embed_documents([sentence.text for sentence in sentences])
        )
        if len(embedded) != len(sentences):
            raise ComponentContractError(
                "Embedder returned {} vectors for {} sentences.".format(
                    len(embedded), len(sentences)
                )
            )
        vectors = tuple(tuple(float(value) for value in vector) for vector in embedded)
        dimensions = {len(vector) for vector in vectors}
        if not dimensions or dimensions == {0}:
            raise ComponentContractError("Sentence embeddings cannot be empty.")
        if len(dimensions) != 1:
            raise VectorDimensionError("Sentence embeddings have mixed dimensions.")
        dimension = next(iter(dimensions))
        metadata = dict(document.metadata)
        metadata.setdefault("document_id", document_id)
        metadata.setdefault("document_type", document.document_type)

        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise VectorDimensionError(
                    "Expected vectors of dimension {}, got {}.".format(
                        self._dimension, dimension
                    )
                )
            self._documents[document_id] = _SentenceDocument(
                document.text, metadata, sentences, vectors
            )
            self._dimension = dimension

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            removed = self._documents.pop(document_id, None)
            if not self._documents:
                self._dimension = None
            return len(removed.sentences) if removed is not None else 0

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vector = tuple(float(value) for value in self.embedder.embed_query(query))
        if not query_vector:
            raise ComponentContractError("Query embedding cannot be empty.")

        with self._lock:
            if self._dimension is not None and len(query_vector) != self._dimension:
                raise VectorDimensionError(
                    "Expected query dimension {}, got {}.".format(
                        self._dimension, len(query_vector)
                    )
                )
            documents = tuple(self._documents.items())

        results = []
        selected_filters = filters or {}
        for document_id, sentence_document in documents:
            if not self._matches(document_id, sentence_document, selected_filters):
                continue
            results.extend(
                self._document_results(query_vector, sentence_document)
            )

        return tuple(
            sorted(results, key=lambda result: (-result.score, result.chunk.id))[
                :top_k
            ]
        )

    def _document_results(
        self, query_vector: Vector, document: _SentenceDocument
    ) -> Sequence[SearchResult]:
        similarities = tuple(
            self._cosine(query_vector, vector) for vector in document.vectors
        )
        threshold = self._percentile(similarities, self.config.seed_percentile)
        seeds = tuple(
            index for index, score in enumerate(similarities) if score >= threshold
        )

        candidates = []
        last_index = len(document.sentences) - 1
        chunk_threshold = self.config.chunk_threshold_factor * threshold
        for seed in seeds:
            start = max(0, seed - self.config.window_radius)
            end = min(last_index, seed + self.config.window_radius)
            score = self._aggregate_score(start, end, seed, similarities)
            if score >= chunk_threshold:
                candidates.append(_Window(start, end, (seed,), score))

        windows = self._merge_windows(candidates, similarities)
        return tuple(
            SearchResult(self._chunk(document, window), window.score)
            for window in windows
        )

    def _merge_windows(
        self, candidates: Iterable[_Window], similarities: Sequence[float]
    ) -> Sequence[_Window]:
        ordered = sorted(candidates, key=lambda window: (window.start, window.end))
        if not ordered:
            return ()

        merged = []
        current = ordered[0]
        for candidate in ordered[1:]:
            if candidate.start - current.end <= self.config.gap_tolerance:
                seeds = tuple(sorted(set(current.seeds + candidate.seeds)))
                start = min(current.start, candidate.start)
                end = max(current.end, candidate.end)
                center = max(seeds, key=lambda seed: (similarities[seed], -seed))
                current = _Window(
                    start,
                    end,
                    seeds,
                    self._aggregate_score(start, end, center, similarities),
                )
            else:
                merged.append(current)
                current = candidate
        merged.append(current)
        return tuple(merged)

    def _aggregate_score(
        self,
        start: int,
        end: int,
        center: int,
        similarities: Sequence[float],
    ) -> float:
        weighted_score = 0.0
        total_weight = 0.0
        for index in range(start, end + 1):
            weight = math.exp(-self.config.decay * abs(index - center))
            weighted_score += weight * similarities[index]
            total_weight += weight
        return weighted_score / total_weight

    @staticmethod
    def _chunk(document: _SentenceDocument, window: _Window) -> Chunk:
        selected = document.sentences[window.start : window.end + 1]
        first = selected[0]
        last = selected[-1]
        text = document.text[first.start_char : last.end_char].strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        metadata = dict(document.metadata)
        metadata.update(first.metadata)
        metadata.update(
            {
                "chunking_method": "qasc",
                "document_id": first.document_id,
                "sentence_start": window.start,
                "sentence_end": window.end + 1,
                "seed_sentences": window.seeds,
                "char_start": first.start_char,
                "char_end": last.end_char,
                "qasc_score": window.score,
            }
        )
        return Chunk(
            id="{}:qasc:{}-{}:{}".format(
                first.document_id, window.start, window.end, digest
            ),
            document_id=first.document_id,
            text=text,
            index=window.start,
            metadata=metadata,
        )

    @staticmethod
    def _matches(
        document_id: str,
        document: _SentenceDocument,
        filters: Mapping[str, Any],
    ) -> bool:
        for key, expected in filters.items():
            actual = (
                document_id
                if key == "document_id"
                else document.metadata.get(key)
            )
            if actual != expected:
                return False
        return True

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    @staticmethod
    def _cosine(left: Vector, right: Vector) -> float:
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if not left_magnitude or not right_magnitude:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        return dot_product / (left_magnitude * right_magnitude)
