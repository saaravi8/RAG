"""Thread-safe in-memory vector storage for development and tests."""

import math
import uuid
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..errors import ComponentContractError, VectorDimensionError
from ..models import SearchResult, Vector, VectorRecord
from ..ports import PreparedDocumentDeletion, PreparedDocumentReplacement
from ..transactions import IndexTransactionCoordinator


class _PreparedVectorStoreReplacement:
    """Prevalidated per-document change used by coordinated transactions."""

    def __init__(
        self,
        store: "InMemoryVectorStore",
        document_id: str,
        previous_records: Tuple[VectorRecord, ...],
        previous_token: Optional[str],
        previous_dimension: Optional[int],
        candidate_records: Tuple[VectorRecord, ...],
        candidate_token: Optional[str],
        candidate_dimension: Optional[int],
        expected_generation: int,
        deleted_count: int = 0,
    ) -> None:
        self._store = store
        self._document_id = document_id
        self._previous_records = previous_records
        self._previous_token = previous_token
        self._previous_dimension = previous_dimension
        self._candidate_records = candidate_records
        self._candidate_token = candidate_token
        self._candidate_dimension = candidate_dimension
        self._expected_generation = expected_generation
        self._committed_generation: Optional[int] = None
        self._deleted_count = deleted_count
        self._committed = False
        self._rolled_back = False

    def commit(self) -> None:
        with self._store.transaction_coordinator.synchronized():
            with self._store._lock:
                if self._committed or self._rolled_back:
                    return
                if self._store._state_generation != self._expected_generation:
                    raise ComponentContractError(
                        "Prepared vector-store change is stale."
                    )
                self._store._apply_document_records(
                    self._document_id,
                    self._candidate_records,
                    self._candidate_token,
                )
                self._store._dimension = self._candidate_dimension
                self._store._state_generation += 1
                self._committed_generation = self._store._state_generation
                self._committed = True

    def rollback(self) -> None:
        with self._store.transaction_coordinator.synchronized():
            with self._store._lock:
                if self._rolled_back:
                    return
                if (
                    self._committed
                    and self._store._state_generation
                    == self._committed_generation
                ):
                    self._store._apply_document_records(
                        self._document_id,
                        self._previous_records,
                        self._previous_token,
                    )
                    self._store._dimension = self._previous_dimension
                    self._store._state_generation += 1
                self._rolled_back = True

    @property
    def deleted_count(self) -> int:
        return self._deleted_count


class InMemoryVectorStore:
    def __init__(
        self,
        *,
        transaction_coordinator: Optional[IndexTransactionCoordinator] = None,
    ) -> None:
        self._records: Dict[str, VectorRecord] = {}
        self._document_chunks: Dict[str, Tuple[str, ...]] = {}
        self._document_tokens: Dict[str, Optional[str]] = {}
        self._dimension: Optional[int] = None
        self._state_generation = 0
        self._lock = RLock()
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else IndexTransactionCoordinator()
        )
        self.index_state_id = uuid.uuid4().hex

    @property
    def count(self) -> int:
        with self.transaction_coordinator.synchronized():
            with self._lock:
                return len(self._records)

    def document_state_token(self, document_id: str) -> Optional[str]:
        """Return the common repository state token for one stored document."""

        with self.transaction_coordinator.synchronized():
            with self._lock:
                return self._document_tokens.get(document_id)

    def replace_document(
        self, document_id: str, records: Sequence[VectorRecord]
    ) -> None:
        with self.transaction_coordinator.synchronized():
            prepared = self.prepare_replace_document(document_id, records)
            prepared.commit()

    def prepare_replace_document(
        self, document_id: str, records: Sequence[VectorRecord]
    ) -> "PreparedDocumentReplacement":
        records = tuple(records)
        if any(record.chunk.document_id != document_id for record in records):
            raise ComponentContractError(
                "Every replacement record must match the supplied document_id."
            )
        chunk_ids = tuple(record.chunk.id for record in records)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ComponentContractError(
                "Replacement records must have unique chunk IDs."
            )
        vectors = tuple(
            self._normalize_vector(record.vector, label="Replacement vector")
            for record in records
        )
        normalized_records = tuple(
            VectorRecord(record.chunk, vector)
            for record, vector in zip(records, vectors)
        )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise VectorDimensionError("Replacement vectors have mixed dimensions.")

        with self.transaction_coordinator.synchronized():
            with self._lock:
                previous_chunk_ids = self._document_chunks.get(document_id, ())
                previous_records = tuple(
                    self._records[chunk_id] for chunk_id in previous_chunk_ids
                )
                previous_token = self._document_tokens.get(document_id)
                previous_dimension = self._dimension
                collisions = tuple(
                    chunk_id
                    for chunk_id in chunk_ids
                    if chunk_id in self._records
                    and self._records[chunk_id].chunk.document_id != document_id
                )
                if collisions:
                    raise ComponentContractError(
                        "Replacement chunk IDs cannot collide with another document."
                    )
                if dimensions:
                    dimension = next(iter(dimensions))
                    if self._dimension is not None and dimension != self._dimension:
                        raise VectorDimensionError(
                            "Expected vectors of dimension {}, got {}.".format(
                                self._dimension, dimension
                            )
                        )
                else:
                    other_document_count = len(self._document_chunks) - int(
                        document_id in self._document_chunks
                    )
                    dimension = self._dimension if other_document_count else None
                return _PreparedVectorStoreReplacement(
                    self,
                    document_id,
                    previous_records,
                    previous_token,
                    previous_dimension,
                    normalized_records,
                    self._state_token(normalized_records),
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
                previous_chunk_ids = self._document_chunks.get(document_id, ())
                previous_records = tuple(
                    self._records[chunk_id] for chunk_id in previous_chunk_ids
                )
                previous_token = self._document_tokens.get(document_id)
                other_document_count = len(self._document_chunks) - int(
                    document_id in self._document_chunks
                )
                dimension = self._dimension if other_document_count else None
                return _PreparedVectorStoreReplacement(
                    self,
                    document_id,
                    previous_records,
                    previous_token,
                    self._dimension,
                    (),
                    None,
                    dimension,
                    self._state_generation,
                    deleted_count=len(previous_records),
                )

    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query_vector = self._normalize_vector(query_vector, label="Query vector")

        with self.transaction_coordinator.synchronized():
            with self._lock:
                if self._dimension is not None and len(query_vector) != self._dimension:
                    raise VectorDimensionError(
                        "Expected query dimension {}, got {}.".format(
                            self._dimension, len(query_vector)
                        )
                    )
                records = tuple(self._records.values())

        results = [
            SearchResult(record.chunk, self._cosine(query_vector, record.vector))
            for record in records
            if self._matches(record, filters or {})
        ]
        return tuple(
            sorted(results, key=lambda result: (-result.score, result.chunk.id))[:top_k]
        )

    def _apply_document_records(
        self,
        document_id: str,
        records: Sequence[VectorRecord],
        state_token: Optional[str],
    ) -> None:
        """Publish one prevalidated document while the store lock is held."""

        for chunk_id in self._document_chunks.get(document_id, ()):
            del self._records[chunk_id]
        if records:
            chunk_ids = tuple(record.chunk.id for record in records)
            for record in records:
                self._records[record.chunk.id] = record
            self._document_chunks[document_id] = chunk_ids
            self._document_tokens[document_id] = state_token
        else:
            self._document_chunks.pop(document_id, None)
            self._document_tokens.pop(document_id, None)

    @staticmethod
    def _state_token(records: Sequence[VectorRecord]) -> Optional[str]:
        if not records:
            return None
        tokens = tuple(
            record.chunk.metadata.get("document_state_token") for record in records
        )
        if any(not isinstance(token, str) or not token for token in tokens):
            return None
        return tokens[0] if len(set(tokens)) == 1 else None

    @staticmethod
    def _normalize_vector(vector: Any, *, label: str) -> Vector:
        if isinstance(vector, (str, bytes, bytearray)):
            raise ComponentContractError("{} is malformed.".format(label))
        try:
            values = tuple(vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComponentContractError("{} is malformed.".format(label)) from exc
        if not values:
            raise ComponentContractError("{} cannot be empty.".format(label))
        if any(
            isinstance(value, (bool, str, bytes, bytearray)) for value in values
        ):
            raise ComponentContractError("{} is malformed.".format(label))
        try:
            normalized = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ComponentContractError("{} is malformed.".format(label)) from exc
        if any(not math.isfinite(value) for value in normalized):
            raise ComponentContractError(
                "{} contains non-finite values.".format(label)
            )
        return normalized

    @staticmethod
    def _matches(record: VectorRecord, filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            if key == "document_id":
                actual = record.chunk.document_id
            elif key == "chunk_id":
                actual = record.chunk.id
            else:
                actual = record.chunk.metadata.get(key)
            if actual != expected:
                return False
        return True

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
