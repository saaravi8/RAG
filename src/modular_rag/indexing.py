"""Application service for ingestion, chunking, embedding, and storage."""

from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from rag_ingestion import Document, DocumentSource

from .errors import ComponentContractError, RepositoryResourceLimitError
from .fingerprint import IndexFingerprint, build_index_fingerprint
from .models import Chunk, IndexReport, VectorRecord, _immutable_metadata
from .ports import (
    Chunker,
    CoordinatedPreparedDocumentReplacement,
    DocumentIndex,
    DocumentProcessor,
    Embedder,
    PreparedDocumentDeletion,
    PreparedDocumentReplacement,
    VectorStore,
)
from .transactions import IndexTransactionCoordinator

SourceLike = Union[str, Path, bytes, DocumentSource]
PreparedChangeFactory = Callable[[], CoordinatedPreparedDocumentReplacement]
ReportPreparedChangeFactory = Callable[
    [IndexReport], CoordinatedPreparedDocumentReplacement
]


class Indexer:
    def __init__(
        self,
        processor: DocumentProcessor,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        document_indexes: Sequence[DocumentIndex] = (),
        transaction_coordinator: Optional[IndexTransactionCoordinator] = None,
    ) -> None:
        self.processor = processor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.document_indexes = tuple(document_indexes)
        inherited_coordinator = getattr(store, "transaction_coordinator", None)
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else inherited_coordinator
            if isinstance(inherited_coordinator, IndexTransactionCoordinator)
            else IndexTransactionCoordinator()
        )
        self._write_lock = RLock()
        self._validate_atomic_participants()
        self._validate_transaction_coordinators()

    @property
    def index_fingerprint(self) -> IndexFingerprint:
        """Return the content-shaping identity used for incremental indexing."""

        return build_index_fingerprint(
            self.embedder,
            self.chunker,
            self.document_indexes,
            processor=self.processor,
        )

    @property
    def index_state_identity(self):
        """Identify the concrete store/index generation described by a manifest."""

        participants = (self.store,) + self.document_indexes
        identities = []
        for participant in participants:
            state_id = getattr(participant, "index_state_id", None)
            if not isinstance(state_id, str) or not state_id:
                return None
            identities.append(
                {
                    "type": "{}.{}".format(
                        type(participant).__module__,
                        type(participant).__qualname__,
                    ),
                    "state_id": state_id,
                }
            )
        return tuple(identities)

    def _validate_atomic_participants(self) -> None:
        """Reject unsafe multi-index compositions before their first write."""

        if not self.document_indexes:
            return
        participants = (("primary vector store", self.store),) + tuple(
            ("auxiliary document index {}".format(index), document_index)
            for index, document_index in enumerate(self.document_indexes)
        )
        unsupported = []
        for name, participant in participants:
            for operation in (
                "prepare_replace_document",
                "prepare_delete_document",
            ):
                if not callable(getattr(participant, operation, None)):
                    unsupported.append("{} ({})".format(name, operation))
        if unsupported:
            raise ComponentContractError(
                "Atomic multi-index operations require staged replacement and "
                "deletion on every participant; missing: {}.".format(
                    ", ".join(unsupported)
                )
            )

    def _validate_transaction_coordinators(self) -> None:
        if not self.document_indexes:
            return
        self._require_shared_transaction_coordinators()

    def _require_shared_transaction_coordinators(self) -> None:
        participants = (("primary vector store", self.store),) + tuple(
            ("auxiliary document index {}".format(index), document_index)
            for index, document_index in enumerate(self.document_indexes)
        )
        for name, participant in participants:
            coordinator = getattr(participant, "transaction_coordinator", None)
            if coordinator is not self.transaction_coordinator:
                raise ComponentContractError(
                    "{} must expose the Indexer's shared "
                    "IndexTransactionCoordinator for an atomic multi-participant "
                    "operation.".format(name)
                )

    def index(
        self,
        source: SourceLike,
        *,
        document_type: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> IndexReport:
        document = self.processor.process(
            source,
            document_type=document_type,
            name=name,
            metadata=metadata,
        )
        return self.index_document(document)

    def index_document(
        self,
        document: Document,
        *,
        max_chunks: Optional[int] = None,
        required_chunk_metadata: Optional[Mapping[str, Any]] = None,
        additional_preparations: Sequence[PreparedChangeFactory] = (),
        report_preparations: Sequence[ReportPreparedChangeFactory] = (),
    ) -> IndexReport:
        """Chunk, embed, and replace one canonical document.

        ``max_chunks`` rejects an oversized chunk batch before embedding or
        storage, allowing repository ingestion to enforce per-file bounds.
        ``required_chunk_metadata`` protects canonical metadata that a chunker
        must preserve exactly on every returned chunk.
        """

        with self._write_lock:
            return self._index_document_locked(
                document,
                max_chunks=max_chunks,
                required_chunk_metadata=required_chunk_metadata,
                additional_preparations=additional_preparations,
                report_preparations=report_preparations,
            )

    def _index_document_locked(
        self,
        document: Document,
        *,
        max_chunks: Optional[int],
        required_chunk_metadata: Optional[Mapping[str, Any]],
        additional_preparations: Sequence[PreparedChangeFactory],
        report_preparations: Sequence[ReportPreparedChangeFactory],
    ) -> IndexReport:
        if max_chunks is not None and (
            isinstance(max_chunks, bool) or not isinstance(max_chunks, int)
        ):
            raise TypeError("max_chunks must be an integer when supplied.")
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive when supplied.")
        if required_chunk_metadata is not None and not isinstance(
            required_chunk_metadata, Mapping
        ):
            raise TypeError("required_chunk_metadata must be a mapping when supplied.")
        protected_metadata = dict(required_chunk_metadata or {})
        canonical_metadata = _immutable_metadata(document.metadata)
        canonical_document = Document(
            document.text,
            document.document_type,
            canonical_metadata,
        )
        chunk_result = self.chunker.chunk(canonical_document)
        try:
            chunk_iterator = iter(chunk_result)
        except TypeError as exc:
            raise ComponentContractError(
                "Chunker returned a non-iterable result."
            ) from exc
        chunks = tuple(chunk_iterator)
        if not chunks:
            raise ComponentContractError(
                "Chunker returned no chunks for the document."
            )
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ComponentContractError(
                    "Chunker returned a non-Chunk item at position {}.".format(index)
                )
        if max_chunks is not None and len(chunks) > max_chunks:
            raise RepositoryResourceLimitError(
                "Document chunk count exceeds the configured limit."
            )

        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ComponentContractError(
                "All chunks from one document must share one document_id."
            )
        document_id = next(iter(document_ids))
        if (
            "document_id" in canonical_metadata
            and document_id != canonical_metadata["document_id"]
        ):
            raise ComponentContractError(
                "Chunk document_id does not match the canonical document metadata."
            )
        for index, chunk in enumerate(chunks):
            for key, expected in protected_metadata.items():
                if key not in chunk.metadata or chunk.metadata[key] != expected:
                    raise ComponentContractError(
                        "Chunk at position {} did not preserve required metadata "
                        "key {!r}.".format(index, key)
                    )

        vector_result = self.embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        try:
            vector_iterator = iter(vector_result)
        except TypeError as exc:
            raise ComponentContractError(
                "Embedder returned a non-iterable result."
            ) from exc
        vectors = tuple(vector_iterator)
        if len(vectors) != len(chunks):
            raise ComponentContractError(
                "Embedder returned {} vectors for {} chunks.".format(
                    len(vectors), len(chunks)
                )
            )

        records: Sequence[VectorRecord] = tuple(
            VectorRecord(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors)
        )
        report = IndexReport(document_id=document_id, chunk_count=len(records))
        prepared_factories = tuple(additional_preparations) + tuple(
            lambda prepare=prepare: prepare(report)
            for prepare in report_preparations
        )
        if self.document_indexes or prepared_factories:
            self._replace_document_atomically(
                document_id,
                records,
                Document(
                    document.text,
                    document.document_type,
                    canonical_metadata,
                ),
                additional_preparations=prepared_factories,
            )
        else:
            self.store.replace_document(document_id, records)
        return report

    def _replace_document_atomically(
        self,
        document_id: str,
        records: Sequence[VectorRecord],
        document: Document,
        *,
        additional_preparations: Sequence[PreparedChangeFactory] = (),
    ) -> None:
        self._require_shared_transaction_coordinators()
        canonical_metadata = _immutable_metadata(document.metadata)
        prepared = []
        try:
            primary_prepare = getattr(self.store, "prepare_replace_document")
            primary = primary_prepare(document_id, records)
            self._validate_prepared_replacement(primary, "primary vector store")
            prepared.append(primary)

            for index, document_index in enumerate(self.document_indexes):
                prepare = getattr(document_index, "prepare_replace_document")
                replacement = prepare(
                    Document(
                        document.text,
                        document.document_type,
                        canonical_metadata,
                    )
                )
                self._validate_prepared_replacement(
                    replacement,
                    "auxiliary document index {}".format(index),
                )
                prepared.append(replacement)

            for index, prepare in enumerate(additional_preparations):
                replacement = prepare()
                self._validate_prepared_replacement(
                    replacement, "additional transaction participant {}".format(index)
                )
                prepared.append(replacement)
                self._validate_dynamic_prepared_coordinator(
                    replacement,
                    "additional transaction participant {}".format(index),
                )

        except BaseException as error:
            if prepared:
                self._rollback_or_raise(
                    prepared, operation="replacement", cause=error
                )
            raise

        with self.transaction_coordinator.synchronized():
            try:
                self._commit_prepared(prepared)
            except BaseException as error:
                self._rollback_or_raise(
                    prepared, operation="replacement", cause=error
                )
                raise

    @staticmethod
    def _validate_prepared_replacement(
        replacement: PreparedDocumentReplacement, participant: str
    ) -> None:
        if not callable(getattr(replacement, "commit", None)) or not callable(
            getattr(replacement, "rollback", None)
        ):
            raise ComponentContractError(
                "{} returned an invalid prepared document replacement.".format(
                    participant
                )
            )

    def _validate_dynamic_prepared_coordinator(
        self,
        replacement: PreparedDocumentReplacement,
        participant: str,
    ) -> None:
        coordinator = getattr(replacement, "transaction_coordinator", None)
        if coordinator is not self.transaction_coordinator:
            raise ComponentContractError(
                "{} returned a prepared change that is not bound to the "
                "Indexer's shared IndexTransactionCoordinator.".format(participant)
            )

    @staticmethod
    def _rollback_prepared(
        prepared: Sequence[PreparedDocumentReplacement],
    ) -> Sequence[BaseException]:
        errors = []
        for replacement in reversed(prepared):
            try:
                replacement.rollback()
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    @staticmethod
    def _commit_prepared(
        prepared: Sequence[PreparedDocumentReplacement],
    ) -> None:
        for replacement in prepared:
            replacement.commit()

    @classmethod
    def _rollback_or_raise(
        cls,
        prepared: Sequence[PreparedDocumentReplacement],
        *,
        operation: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        rollback_errors = cls._rollback_prepared(prepared)
        if rollback_errors:
            raise ComponentContractError(
                "Atomic document {} failed and {} rollback(s) violated the "
                "non-raising rollback contract.".format(
                    operation, len(rollback_errors)
                )
            ) from cause

    def delete(
        self,
        document_id: str,
        *,
        additional_preparations: Sequence[PreparedChangeFactory] = (),
    ) -> int:
        if not isinstance(document_id, str):
            raise TypeError("document_id must be a string.")
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")
        with self._write_lock:
            if not self.document_indexes and not additional_preparations:
                return self.store.delete_document(document_id)
            return self._delete_document_atomically(
                document_id,
                additional_preparations=additional_preparations,
            )

    def _delete_document_atomically(
        self,
        document_id: str,
        *,
        additional_preparations: Sequence[PreparedChangeFactory],
    ) -> int:
        self._require_shared_transaction_coordinators()
        prepared = []
        try:
            primary_prepare = getattr(self.store, "prepare_delete_document", None)
            if not callable(primary_prepare):
                raise ComponentContractError(
                    "Atomic document deletion requires prepare_delete_document "
                    "on the primary vector store."
                )
            primary = primary_prepare(document_id)
            self._validate_prepared_deletion(primary, "primary vector store")
            prepared.append(primary)

            for index, document_index in enumerate(self.document_indexes):
                prepare = getattr(document_index, "prepare_delete_document")
                deletion = prepare(document_id)
                self._validate_prepared_deletion(
                    deletion, "auxiliary document index {}".format(index)
                )
                prepared.append(deletion)

            for index, prepare in enumerate(additional_preparations):
                replacement = prepare()
                self._validate_prepared_replacement(
                    replacement, "additional transaction participant {}".format(index)
                )
                prepared.append(replacement)
                self._validate_dynamic_prepared_coordinator(
                    replacement,
                    "additional transaction participant {}".format(index),
                )

        except BaseException as error:
            if prepared:
                self._rollback_or_raise(
                    prepared, operation="deletion", cause=error
                )
            raise

        with self.transaction_coordinator.synchronized():
            try:
                self._commit_prepared(prepared)
            except BaseException as error:
                self._rollback_or_raise(
                    prepared, operation="deletion", cause=error
                )
                raise
        return primary.deleted_count

    @classmethod
    def _validate_prepared_deletion(
        cls, deletion: PreparedDocumentDeletion, participant: str
    ) -> None:
        cls._validate_prepared_replacement(deletion, participant)
        deleted_count = getattr(deletion, "deleted_count", None)
        if (
            not isinstance(deleted_count, int)
            or isinstance(deleted_count, bool)
            or deleted_count < 0
        ):
            raise ComponentContractError(
                "{} returned an invalid prepared deletion count.".format(participant)
            )
