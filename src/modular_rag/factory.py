"""Composition root for the runnable dependency-free example."""

from dataclasses import dataclass
from typing import Any, Optional

from rag_ingestion import create_default_pipeline

from .adapters import InMemoryVectorStore, SpacySentenceSegmenter
from .chunking import WordWindowChunker
from .embedding import HashingEmbedder
from .generation import DemoExtractiveGenerator
from .indexing import Indexer
from .models import IndexReport, RAGResponse
from .ports import DocumentProcessor, Embedder, Reranker, SentenceSegmenter
from .qasc import QASCConfig, QASCRetriever
from .retrieval import KeywordReranker, VectorRetriever
from .service import RAGService


@dataclass
class RAGApplication:
    """Convenience container exposing both write and query sides."""

    indexer: Indexer
    rag: RAGService
    store: InMemoryVectorStore

    def index(self, source: Any, **kwargs: Any) -> IndexReport:
        return self.indexer.index(source, **kwargs)

    def ask(self, question: str, **kwargs: Any) -> RAGResponse:
        return self.rag.ask(question, **kwargs)


def build_demo_rag(
    *,
    max_words: int = 200,
    overlap_words: int = 30,
    dimensions: int = 256,
    processor: Optional[DocumentProcessor] = None,
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
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
    chunker = WordWindowChunker(max_words, overlap_words)
    selected_embedder = (
        embedder if embedder is not None else HashingEmbedder(dimensions)
    )
    store = InMemoryVectorStore()
    query_methods = {}
    document_indexes = ()
    if enable_qasc:
        segmenter = (
            qasc_segmenter
            if qasc_segmenter is not None
            else SpacySentenceSegmenter()
        )
        qasc = QASCRetriever(segmenter, selected_embedder, config=qasc_config)
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
    )
    retriever = VectorRetriever(selected_embedder, store)
    rag = RAGService(
        retriever,
        DemoExtractiveGenerator(),
        reranker=reranker if reranker is not None else KeywordReranker(),
        query_methods=query_methods,
    )
    return RAGApplication(indexer=indexer, rag=rag, store=store)
