"""Public API for the modular RAG skeleton."""

from .adapters import InMemoryVectorStore, SpacySentenceSegmenter
from .chunking import WordWindowChunker
from .code import (
    BuiltinSyntaxParser,
    CodeChunker,
    RoutingChunker,
    SyntaxParseResult,
    SyntaxSpan,
    TreeSitterSyntaxParser,
)
from .embedding import HashingEmbedder, SentenceTransformerEmbedder
from .errors import (
    ComponentContractError,
    OptionalDependencyError,
    RAGError,
    RepositoryIngestionError,
    RepositoryResourceLimitError,
    UnsafeRepositoryError,
    VectorDimensionError,
)
from .factory import RAGApplication, build_demo_rag
from .fingerprint import INDEX_SCHEMA_VERSION, IndexFingerprint, build_index_fingerprint
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
from .repository import (
    DefaultSecretScanner,
    InMemoryRepositoryManifest,
    RepositoryIndexReport,
    RepositoryIndexer,
    RepositoryIssue,
    RepositoryLimits,
    RepositoryManifest,
    RepositoryManifestEntry,
    RepositoryPolicy,
    SecretScanResult,
    SecretScanner,
)
from .retrieval import CrossEncoderReranker, KeywordReranker, VectorRetriever
from .service import RAGService

__all__ = [
    "AnswerGenerator",
    "AnswerVerifier",
    "BuiltinSyntaxParser",
    "Chunk",
    "Chunker",
    "Citation",
    "ComponentContractError",
    "CodeChunker",
    "CrossEncoderReranker",
    "DemoExtractiveGenerator",
    "DocumentIndex",
    "DocumentProcessor",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "InMemoryRepositoryManifest",
    "INDEX_SCHEMA_VERSION",
    "IndexReport",
    "IndexFingerprint",
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
    "RepositoryIndexReport",
    "RepositoryIndexer",
    "RepositoryIngestionError",
    "RepositoryIssue",
    "RepositoryLimits",
    "RepositoryManifest",
    "RepositoryManifestEntry",
    "RepositoryPolicy",
    "RepositoryResourceLimitError",
    "RoutingChunker",
    "SearchResult",
    "SentenceSegmenter",
    "SentenceSpan",
    "SentenceTransformerEmbedder",
    "SecretScanResult",
    "SecretScanner",
    "SpacySentenceSegmenter",
    "SyntaxParseResult",
    "SyntaxSpan",
    "TreeSitterSyntaxParser",
    "UnsafeRepositoryError",
    "Vector",
    "VectorDimensionError",
    "VectorRecord",
    "VectorRetriever",
    "VectorStore",
    "VerificationResult",
    "WordWindowChunker",
    "build_demo_rag",
    "build_index_fingerprint",
    "DefaultSecretScanner",
]
