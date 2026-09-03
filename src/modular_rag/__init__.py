"""Public API for the modular RAG skeleton."""

from .adapters import InMemoryVectorStore, SpacySentenceSegmenter
from .chunking import WordWindowChunker
from .embedding import HashingEmbedder, SentenceTransformerEmbedder
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
    VerificationResult,
)
from .ports import (
    AnswerGenerator,
    AnswerVerifier,
    Chunker,
    DocumentIndex,
    DocumentProcessor,
    Embedder,
    Reranker,
    Retriever,
    SentenceSegmenter,
    VectorStore,
)
from .qasc import QASCConfig, QASCRetriever
from .retrieval import CrossEncoderReranker, KeywordReranker, VectorRetriever
from .service import RAGService

__all__ = [
    "AnswerGenerator",
    "AnswerVerifier",
    "Chunk",
    "Chunker",
    "Citation",
    "ComponentContractError",
    "CrossEncoderReranker",
    "DemoExtractiveGenerator",
    "DocumentIndex",
    "DocumentProcessor",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "IndexReport",
    "Indexer",
    "KeywordReranker",
    "OptionalDependencyError",
    "QASCConfig",
    "QASCRetriever",
    "RAGApplication",
    "RAGError",
    "RAGResponse",
    "RAGService",
    "Reranker",
    "Retriever",
    "SearchResult",
    "SentenceSegmenter",
    "SentenceSpan",
    "SentenceTransformerEmbedder",
    "SpacySentenceSegmenter",
    "Vector",
    "VectorDimensionError",
    "VectorRecord",
    "VectorRetriever",
    "VectorStore",
    "VerificationResult",
    "WordWindowChunker",
    "build_demo_rag",
]
