"""Application service for ingestion, chunking, embedding, and storage."""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from rag_ingestion import Document, DocumentSource

from .errors import ComponentContractError, RepositoryResourceLimitError
from .models import IndexReport, VectorRecord
from .ports import Chunker, DocumentIndex, DocumentProcessor, Embedder, VectorStore

SourceLike = Union[str, Path, bytes, DocumentSource]


class Indexer:
    def __init__(
        self,
        processor: DocumentProcessor,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        document_indexes: Sequence[DocumentIndex] = (),
    ) -> None:
        self.processor = processor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.document_indexes = tuple(document_indexes)

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
        self, document: Document, *, max_chunks: Optional[int] = None
    ) -> IndexReport:
        """Chunk, embed, and replace one canonical document.

        ``max_chunks`` rejects an oversized chunk batch before embedding or
        storage, allowing repository ingestion to enforce per-file bounds.
        """

        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive when supplied.")
        chunks = tuple(self.chunker.chunk(document))
        if not chunks:
            raise ComponentContractError("Chunker returned no chunks for the document.")
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

        vectors = tuple(self.embedder.embed_documents([chunk.text for chunk in chunks]))
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
        self.store.replace_document(document_id, records)
        for document_index in self.document_indexes:
            document_index.replace_document(document)
        return IndexReport(document_id=document_id, chunk_count=len(records))

    def delete(self, document_id: str) -> int:
        deleted = self.store.delete_document(document_id)
        for document_index in self.document_indexes:
            document_index.delete_document(document_id)
        return deleted
