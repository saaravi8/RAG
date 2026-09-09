"""Composition root for the runnable dependency-free example."""

from dataclasses import dataclass
from typing import Any, Optional

from rag_ingestion import create_default_pipeline

from .adapters import InMemoryVectorStore, SpacySentenceSegmenter
from .code import CodeChunker, RoutingChunker
from .chunking import WordWindowChunker
from .embedding import HashingEmbedder
from .generation import DemoExtractiveGenerator
from .indexing import Indexer
from .models import IndexReport, RAGResponse
from .ports import (
    DocumentProcessor,
    Embedder,
    Reranker,
    SentenceSegmenter,
)
from .qasc import QASCConfig, QASCRetriever
from .repository import (
    InMemoryRepositoryManifest,
    RepositoryIndexReport,
    RepositoryIndexer,
    RepositoryManifest,
    RepositoryPolicy,
)
from .retrieval import KeywordReranker, VectorRetriever
from .service import RAGService
from .transactions import IndexTransactionCoordinator


@dataclass
class RAGApplication:
    """Convenience container exposing both write and query sides."""

    indexer: Indexer
    rag: RAGService
    store: InMemoryVectorStore
    repository_indexer: Optional[RepositoryIndexer] = None

    def index(self, source: Any, **kwargs: Any) -> IndexReport:
        return self.indexer.index(source, **kwargs)

    def ask(self, question: str, **kwargs: Any) -> RAGResponse:
        return self.rag.ask(question, **kwargs)

    def index_repository(self, source: Any, **kwargs: Any) -> RepositoryIndexReport:
        """Index a local Git working tree when a repository indexer is configured."""

        if self.repository_indexer is None:
            raise RuntimeError("Repository indexing is not configured.")
        return self.repository_indexer.index_repository(source, **kwargs)


def build_demo_rag(
    *,
    max_words: int = 200,
    overlap_words: int = 30,
    dimensions: int = 256,
    processor: Optional[DocumentProcessor] = None,
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
    code_parser: Optional[Any] = None,
    code_max_lines: int = 200,
    code_overlap_lines: int = 20,
    repository_manifest: Optional[RepositoryManifest] = None,
    repository_policy: Optional[RepositoryPolicy] = None,
    enable_qasc: bool = False,
    qasc_segmenter: Optional[SentenceSegmenter] = None,
    qasc_config: Optional[QASCConfig] = None,
) -> RAGApplication:
    """Wire local components that make the complete pipeline runnable offline.

    The dependency-free keyword reranker remains the default. Supply any
    ``Reranker`` implementation to replace it without changing orchestration.
    QASC is opt-in because it maintains an additional sentence index. When it
    is enabled without a custom segmenter, the optional spaCy adapter is used.
    """

    selected_processor = (
        processor if processor is not None else create_default_pipeline()
    )
    chunker = RoutingChunker(
        WordWindowChunker(max_words, overlap_words),
        CodeChunker(
            max_lines=code_max_lines,
            overlap_lines=code_overlap_lines,
            parser=code_parser,
        ),
    )
    selected_embedder = (
        embedder if embedder is not None else HashingEmbedder(dimensions)
    )
    manifest_coordinator = getattr(
        repository_manifest, "transaction_coordinator", None
    )
    transaction_coordinator = (
        manifest_coordinator
        if isinstance(manifest_coordinator, IndexTransactionCoordinator)
        else IndexTransactionCoordinator()
    )
    store = InMemoryVectorStore(
        transaction_coordinator=transaction_coordinator
    )
    query_methods = {}
    document_indexes = ()
    if enable_qasc:
        segmenter = (
            qasc_segmenter
            if qasc_segmenter is not None
            else SpacySentenceSegmenter()
        )
        qasc = QASCRetriever(
            segmenter,
            selected_embedder,
            config=qasc_config,
            transaction_coordinator=transaction_coordinator,
        )
        query_methods["qasc"] = qasc
        document_indexes = (qasc,)
    elif qasc_segmenter is not None or qasc_config is not None:
        raise ValueError("Set enable_qasc=True to configure QASC.")

    indexer = Indexer(
        selected_processor,
        chunker,
        selected_embedder,
        store,
        document_indexes=document_indexes,
        transaction_coordinator=transaction_coordinator,
    )
    retriever = VectorRetriever(selected_embedder, store)
    rag = RAGService(
        retriever,
        DemoExtractiveGenerator(),
        reranker=reranker if reranker is not None else KeywordReranker(),
        query_methods=query_methods,
    )
    repository_indexer = RepositoryIndexer(
        indexer,
        manifest=(
            repository_manifest
            if repository_manifest is not None
            else InMemoryRepositoryManifest(
                transaction_coordinator=transaction_coordinator
            )
        ),
        policy=repository_policy,
    )
    return RAGApplication(
        indexer=indexer,
        rag=rag,
        store=store,
        repository_indexer=repository_indexer,
    )
