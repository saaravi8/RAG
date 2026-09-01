"""Public API for the modular RAG skeleton."""

from .adapters import InMemoryVectorStore, SpacySentenceSegmenter
from .chunking import WordWindowChunker
from .embedding import HashingEmbedder
from .errors import (
    ComponentContractError,
    OptionalDependencyError,
    RAGError,
    VectorDimensionError,
)
from .factory import RAGApplication, build_demo_rag
from .generation import DemoExtractiveGenerator
from .indexing import Indexer
from .models import (
    Chunk,
    Citation,
    IndexReport,
    RAGResponse,
    SearchResult,
    SentenceSpan,
    Vector,
    VectorRecord,
)
from .ports import (
    AnswerGenerator,
    Chunker,
    DocumentProcessor,
    Embedder,
    Reranker,
    Retriever,
    SentenceSegmenter,
    VectorStore,
)
from .retrieval import KeywordReranker, VectorRetriever
from .service import RAGService

__all__ = [
    "AnswerGenerator",
    "Chunk",
    "Chunker",
    "Citation",
    "ComponentContractError",
    "DemoExtractiveGenerator",
    "DocumentProcessor",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "IndexReport",
    "Indexer",
    "KeywordReranker",
    "OptionalDependencyError",
    "RAGApplication",
    "RAGError",
    "RAGResponse",
    "RAGService",
    "Reranker",
    "Retriever",
    "SearchResult",
    "SentenceSegmenter",
    "SentenceSpan",
    "SpacySentenceSegmenter",
    "Vector",
    "VectorDimensionError",
    "VectorRecord",
    "VectorRetriever",
    "VectorStore",
    "WordWindowChunker",
    "build_demo_rag",
]
