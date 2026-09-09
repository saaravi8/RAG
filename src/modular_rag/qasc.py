"""Optional Query-Adaptive Semantic Chunking retrieval strategy."""

import hashlib
import math
import uuid
from numbers import Real
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from rag_ingestion import Document

from .errors import ComponentContractError, VectorDimensionError
from .fingerprint import describe_component
from .models import Chunk, SearchResult, SentenceSpan, Vector, _immutable_metadata
from .ports import (
    Embedder,
    PreparedDocumentDeletion,
    PreparedDocumentReplacement,
    SentenceSegmenter,
)
from .transactions import IndexTransactionCoordinator


@dataclass(frozen=True)
class QASCConfig:
    """Parameters for query-adaptive seed selection and window construction."""

    seed_percentile: float = 75.0
    window_radius: int = 3
    decay: float = 0.3
    gap_tolerance: int = 2
    chunk_threshold_factor: float = 0.6

    def __post_init__(self) -> None:
        if not self._finite_number(self.seed_percentile):
            raise ValueError("seed_percentile must be a finite number.")
        if not 0.0 <= self.seed_percentile <= 100.0:
            raise ValueError("seed_percentile must be between 0 and 100.")
        if not self._non_negative_integer(self.window_radius):
            raise ValueError("window_radius must be a non-negative integer.")
        if not self._finite_number(self.decay) or self.decay < 0.0:
            raise ValueError("decay must be a finite non-negative number.")
        if not self._non_negative_integer(self.gap_tolerance):
            raise ValueError("gap_tolerance must be a non-negative integer.")
        if not self._finite_number(self.chunk_threshold_factor):
            raise ValueError("chunk_threshold_factor must be a finite number.")
        if not 0.0 <= self.chunk_threshold_factor <= 1.0:
            raise ValueError("chunk_threshold_factor must be between 0 and 1.")

    @staticmethod
    def _finite_number(value: Any) -> bool:
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    @staticmethod
    def _non_negative_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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


class _PreparedSentenceIndexReplacement:
    """Prevalidated per-document change used by coordinated transactions."""

    def __init__(
        self,
        retriever: "QASCRetriever",
        document_id: str,
        previous_document: Optional[_SentenceDocument],
        previous_dimension: Optional[int],
        candidate_document: Optional[_SentenceDocument],
        candidate_dimension: Optional[int],
        expected_generation: int,
        deleted_count: int = 0,
    ) -> None:
        self._retriever = retriever
        self._document_id = document_id
        self._previous_document = previous_document
        self._previous_dimension = previous_dimension
        self._candidate_document = candidate_document
        self._candidate_dimension = candidate_dimension
        self._expected_generation = expected_generation
        self._committed_generation: Optional[int] = None
        self._deleted_count = deleted_count
        self._committed = False
        self._rolled_back = False

    def commit(self) -> None:
        with self._retriever.transaction_coordinator.synchronized():
            with self._retriever._lock:
                if self._committed or self._rolled_back:
                    return
                if (
                    self._retriever._state_generation
                    != self._expected_generation
                ):
                    raise ComponentContractError(
                        "Prepared QASC change is stale."
                    )
                self._retriever._apply_document(
                    self._document_id, self._candidate_document
                )
                self._retriever._dimension = self._candidate_dimension
                self._retriever._state_generation += 1
                self._committed_generation = self._retriever._state_generation
                self._committed = True

    def rollback(self) -> None:
        with self._retriever.transaction_coordinator.synchronized():
            with self._retriever._lock:
                if self._rolled_back:
                    return
                if (
                    self._committed
                    and self._retriever._state_generation
                    == self._committed_generation
                ):
                    self._retriever._apply_document(
                        self._document_id, self._previous_document
                    )
                    self._retriever._dimension = self._previous_dimension
                    self._retriever._state_generation += 1
                self._rolled_back = True

    @property
    def deleted_count(self) -> int:
        return self._deleted_count


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
        transaction_coordinator: Optional[IndexTransactionCoordinator] = None,
    ) -> None:
        self.segmenter = segmenter
        self.embedder = embedder
        self.config = config or QASCConfig()
        self._documents: Dict[str, _SentenceDocument] = {}
        self._dimension: Optional[int] = None
        self._state_generation = 0
        self._lock = RLock()
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else IndexTransactionCoordinator()
        )
        self.index_state_id = uuid.uuid4().hex

    def fingerprint_components(self):
        segmenter, segmenter_reusable = describe_component(self.segmenter)
        embedder, embedder_reusable = describe_component(self.embedder)
        return {
            "algorithm": "query-adaptive-semantic-chunking",
            "algorithm_version": 1,
            "segmenter": segmenter,
            "embedder": embedder,
            "opaque": not segmenter_reusable or not embedder_reusable,
        }

    @property
    def document_count(self) -> int:
        with self.transaction_coordinator.synchronized():
            with self._lock:
                return len(self._documents)

    def document_state_token(self, document_id: str) -> Optional[str]:
        """Return the repository state token attached to a sentence document."""

        with self.transaction_coordinator.synchronized():
            with self._lock:
                document = self._documents.get(document_id)
                if document is None:
                    return None
                token = document.metadata.get("document_state_token")
                return token if isinstance(token, str) and token else None

    def replace_document(self, document: Document) -> None:
        with self.transaction_coordinator.synchronized():
            prepared = self.prepare_replace_document(document)
            prepared.commit()

    def prepare_replace_document(
        self, document: Document
    ) -> "PreparedDocumentReplacement":
        canonical_metadata = _immutable_metadata(document.metadata)
        segmenter_document = Document(
            document.text,
            document.document_type,
            canonical_metadata,
        )
        canonical_document_id = canonical_metadata.get("document_id")
        sentences = tuple(self.segmenter.segment(segmenter_document))
        if not sentences:
            raise ComponentContractError(
                "Sentence segmenter returned no sentences for the document."
            )
        if any(not isinstance(sentence, SentenceSpan) for sentence in sentences):
            raise ComponentContractError(
                "Sentence segmenter must return only SentenceSpan values."
            )

        document_ids = {sentence.document_id for sentence in sentences}
        if len(document_ids) != 1:
            raise ComponentContractError(
                "All sentences from one document must share one document_id."
            )
        document_id = next(iter(document_ids))
        if "document_id" in canonical_metadata:
            if (
                not isinstance(canonical_document_id, str)
                or not canonical_document_id.strip()
                or document_id != canonical_document_id
            ):
                raise ComponentContractError(
                    "Sentence document_id does not match the canonical document metadata."
                )
        expected_indexes = tuple(range(len(sentences)))
        actual_indexes = tuple(sentence.index for sentence in sentences)
        if actual_indexes != expected_indexes:
            raise ComponentContractError(
                "Sentence indexes must be contiguous and start at zero."
            )
        previous_end = 0
        for sentence in sentences:
            if sentence.start_char < previous_end:
                raise ComponentContractError(
                    "Sentence spans must be ordered and non-overlapping."
                )
            if sentence.end_char > len(document.text):
                raise ComponentContractError(
                    "Sentence spans must stay within the canonical document text."
                )
            if document.text[sentence.start_char : sentence.end_char] != sentence.text:
                raise ComponentContractError(
                    "Sentence text must match its canonical document slice."
                )
            previous_end = sentence.end_char

        embedded_result = self.embedder.embed_documents(
            [sentence.text for sentence in sentences]
        )
        try:
            embedded = tuple(embedded_result)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComponentContractError(
                "Embedder returned malformed sentence embeddings."
            ) from exc
        if len(embedded) != len(sentences):
            raise ComponentContractError(
                "Embedder returned {} vectors for {} sentences.".format(
                    len(embedded), len(sentences)
                )
            )
        vectors = tuple(
            self._normalize_vector(
                vector,
                empty_message="Sentence embeddings cannot be empty.",
                malformed_message="Sentence embeddings are malformed.",
                non_finite_message=(
                    "Sentence embeddings contain non-finite values."
                ),
            )
            for vector in embedded
        )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise VectorDimensionError("Sentence embeddings have mixed dimensions.")
        dimension = next(iter(dimensions))
        metadata = dict(canonical_metadata)
        metadata.setdefault("document_id", document_id)
        metadata.setdefault("document_type", document.document_type)
        metadata = _immutable_metadata(metadata)

        with self.transaction_coordinator.synchronized():
            with self._lock:
                previous_document = self._documents.get(document_id)
                previous_dimension = self._dimension
                if self._dimension is not None and dimension != self._dimension:
                    raise VectorDimensionError(
                        "Expected vectors of dimension {}, got {}.".format(
                            self._dimension, dimension
                        )
                    )
                candidate_document = _SentenceDocument(
                    document.text, metadata, sentences, vectors
                )
                return _PreparedSentenceIndexReplacement(
                    self,
                    document_id,
                    previous_document,
                    previous_dimension,
                    candidate_document,
                    dimension,
                    self._state_generation,
                )

    def delete_document(self, document_id: str) -> int:
        with self.transaction_coordinator.synchronized():
            prepared = self.prepare_delete_document(document_id)
            prepared.commit()
            return prepared.deleted_count

    def prepare_delete_document(
        self, document_id: str
    ) -> "PreparedDocumentDeletion":
        with self.transaction_coordinator.synchronized():
            with self._lock:
                removed = self._documents.get(document_id)
                other_document_count = len(self._documents) - int(
                    removed is not None
                )
                dimension = self._dimension if other_document_count else None
                return _PreparedSentenceIndexReplacement(
                    self,
                    document_id,
                    removed,
                    self._dimension,
                    None,
                    dimension,
                    self._state_generation,
                    deleted_count=(
                        len(removed.sentences) if removed is not None else 0
                    ),
                )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vector = self._normalize_vector(
            self.embedder.embed_query(query),
            empty_message="Query embedding cannot be empty.",
            malformed_message="Query embedding is malformed.",
            non_finite_message="Query embedding contains non-finite values.",
        )

        with self.transaction_coordinator.synchronized():
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

    def _apply_document(
        self,
        document_id: str,
        document: Optional[_SentenceDocument],
    ) -> None:
        """Publish one prevalidated sentence document while holding the lock."""

        if document is None:
            self._documents.pop(document_id, None)
        else:
            self._documents[document_id] = document

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
        seed_window_scores = self._seed_window_scores(similarities)

        candidates = []
        last_index = len(document.sentences) - 1
        chunk_threshold = self.config.chunk_threshold_factor * threshold
        for seed in seeds:
            start = max(0, seed - self.config.window_radius)
            end = min(last_index, seed + self.config.window_radius)
            score = seed_window_scores[seed]
            if score >= chunk_threshold:
                candidates.append(_Window(start, end, (seed,), score))

        windows = self._merge_windows(candidates, similarities)
        return tuple(
            SearchResult(self._chunk(document, window), window.score)
            for window in windows
        )

    def _seed_window_scores(
        self, similarities: Sequence[float]
    ) -> Tuple[float, ...]:
        """Score every fixed-radius seed window in linear time."""

        if not similarities:
            return ()
        radius = min(self.config.window_radius, len(similarities) - 1)
        if radius == 0:
            return tuple(similarities)
        alpha = math.exp(-self.config.decay)
        expired_weight = alpha ** (radius + 1)

        left_scores = []
        left_weights = []
        weighted_score = 0.0
        total_weight = 0.0
        for index, similarity in enumerate(similarities):
            weighted_score = similarity + alpha * weighted_score
            total_weight = 1.0 + alpha * total_weight
            expired = index - radius - 1
            if expired >= 0:
                weighted_score -= expired_weight * similarities[expired]
                total_weight -= expired_weight
            left_scores.append(weighted_score)
            left_weights.append(total_weight)

        right_scores = [0.0] * len(similarities)
        right_weights = [0.0] * len(similarities)
        weighted_score = 0.0
        total_weight = 0.0
        for index in range(len(similarities) - 1, -1, -1):
            weighted_score = similarities[index] + alpha * weighted_score
            total_weight = 1.0 + alpha * total_weight
            expired = index + radius + 1
            if expired < len(similarities):
                weighted_score -= expired_weight * similarities[expired]
                total_weight -= expired_weight
            right_scores[index] = weighted_score
            right_weights[index] = total_weight

        return tuple(
            (left_scores[index] + right_scores[index] - similarity)
            / (left_weights[index] + right_weights[index] - 1.0)
            for index, similarity in enumerate(similarities)
        )

    def _merge_windows(
        self, candidates: Iterable[_Window], similarities: Sequence[float]
    ) -> Sequence[_Window]:
        ordered = sorted(candidates, key=lambda window: (window.start, window.end))
        if not ordered:
            return ()

        merged = []
        current = ordered[0]
        current_start = current.start
        current_end = current.end
        current_seeds = set(current.seeds)
        current_was_merged = False

        def finish_current() -> _Window:
            seeds = tuple(sorted(current_seeds))
            if not current_was_merged:
                return current
            center = max(seeds, key=lambda seed: (similarities[seed], -seed))
            return _Window(
                current_start,
                current_end,
                seeds,
                self._aggregate_score(
                    current_start, current_end, center, similarities
                ),
            )

        for candidate in ordered[1:]:
            if candidate.start - current_end <= self.config.gap_tolerance:
                current_start = min(current_start, candidate.start)
                current_end = max(current_end, candidate.end)
                current_seeds.update(candidate.seeds)
                current_was_merged = True
            else:
                merged.append(finish_current())
                current = candidate
                current_start = current.start
                current_end = current.end
                current_seeds = set(current.seeds)
                current_was_merged = False
        merged.append(finish_current())
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
        metadata = dict(first.metadata)
        metadata.update(document.metadata)
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
    def _normalize_vector(
        vector: Any,
        *,
        empty_message: str,
        malformed_message: str,
        non_finite_message: str,
    ) -> Vector:
        if isinstance(vector, (str, bytes, bytearray)):
            raise ComponentContractError(malformed_message)
        try:
            values = tuple(vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComponentContractError(malformed_message) from exc
        if not values:
            raise ComponentContractError(empty_message)
        if any(
            isinstance(value, (bool, str, bytes, bytearray)) for value in values
        ):
            raise ComponentContractError(malformed_message)
        try:
            normalized = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComponentContractError(malformed_message) from exc
        if any(not math.isfinite(value) for value in normalized):
            raise ComponentContractError(non_finite_message)
        return normalized

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
        left_scale = max(abs(value) for value in left)
        right_scale = max(abs(value) for value in right)
        if not left_scale or not right_scale:
            return 0.0
        scaled_left = tuple(value / left_scale for value in left)
        scaled_right = tuple(value / right_scale for value in right)
        left_magnitude = math.sqrt(
            math.fsum(value * value for value in scaled_left)
        )
        right_magnitude = math.sqrt(
            math.fsum(value * value for value in scaled_right)
        )
        dot_product = math.fsum(
            a * b for a, b in zip(scaled_left, scaled_right)
        )
        return dot_product / (left_magnitude * right_magnitude)
