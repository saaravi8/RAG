"""Thread-safe in-memory vector storage for development and tests."""

import math
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..errors import ComponentContractError, VectorDimensionError
from ..models import SearchResult, Vector, VectorRecord


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: Dict[str, VectorRecord] = {}
        self._dimension: Optional[int] = None
        self._lock = RLock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def replace_document(
        self, document_id: str, records: Sequence[VectorRecord]
    ) -> None:
        records = tuple(records)
        if any(record.chunk.document_id != document_id for record in records):
            raise ComponentContractError(
                "Every replacement record must match the supplied document_id."
            )
        dimensions = {len(record.vector) for record in records}
        if len(dimensions) > 1:
            raise VectorDimensionError("Replacement vectors have mixed dimensions.")

        with self._lock:
            if dimensions:
                dimension = next(iter(dimensions))
                if self._dimension is not None and dimension != self._dimension:
                    raise VectorDimensionError(
                        "Expected vectors of dimension {}, got {}.".format(
                            self._dimension, dimension
                        )
                    )
                self._dimension = dimension

            retained = {
                chunk_id: record
                for chunk_id, record in self._records.items()
                if record.chunk.document_id != document_id
            }
            retained.update({record.chunk.id: record for record in records})
            self._records = retained
            if not self._records:
                self._dimension = None

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            matching = [
                chunk_id
                for chunk_id, record in self._records.items()
                if record.chunk.document_id == document_id
            ]
            for chunk_id in matching:
                del self._records[chunk_id]
            if not self._records:
                self._dimension = None
            return len(matching)

    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

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
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if not left_magnitude or not right_magnitude:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        return dot_product / (left_magnitude * right_magnitude)
