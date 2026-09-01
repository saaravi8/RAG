"""Composition root for the runnable dependency-free example."""

from dataclasses import dataclass
from typing import Any, Optional

from rag_ingestion import create_default_pipeline

from .adapters import InMemoryVectorStore
from .chunking import WordWindowChunker
from .embedding import HashingEmbedder
from .generation import DemoExtractiveGenerator
from .indexing import Indexer
from .models import IndexReport, RAGResponse
from .ports import DocumentProcessor
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
) -> RAGApplication:
    """Wire local components that make the complete pipeline runnable offline."""

    selected_processor = (
        processor if processor is not None else create_default_pipeline()
    )
    chunker = WordWindowChunker(max_words, overlap_words)
    embedder = HashingEmbedder(dimensions)
    store = InMemoryVectorStore()
    indexer = Indexer(selected_processor, chunker, embedder, store)
    retriever = VectorRetriever(embedder, store)
    rag = RAGService(
        retriever,
        DemoExtractiveGenerator(),
        reranker=KeywordReranker(),
    )
    return RAGApplication(indexer=indexer, rag=rag, store=store)
