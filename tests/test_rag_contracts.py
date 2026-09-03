import unittest

from rag_ingestion import Document

from modular_rag import (
    AnswerVerifier,
    Chunk,
    ComponentContractError,
    HashingEmbedder,
    InMemoryVectorStore,
    Indexer,
    QASCConfig,
    RAGService,
    SearchResult,
    VectorDimensionError,
    VectorRecord,
    VerificationResult,
    WordWindowChunker,
    build_demo_rag,
)


def record(chunk_id, document_id, vector, **metadata):
    return VectorRecord(
        Chunk(chunk_id, document_id, chunk_id, 0, metadata),
        vector,
    )


class ChunkerContractTests(unittest.TestCase):
    def test_chunker_rejects_invalid_window_configuration(self):
        """Window size must be positive and overlap must leave forward progress."""

        invalid_parameters = (
            {"max_words": 0, "overlap_words": 0},
            {"max_words": 4, "overlap_words": -1},
            {"max_words": 4, "overlap_words": 4},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    WordWindowChunker(**parameters)

        self.assertEqual(WordWindowChunker(max_words=4, overlap_words=3).max_words, 4)

    def test_chunker_generates_stable_source_based_document_ids(self):
        """Documents without explicit IDs remain stable but distinct by provenance."""

        chunker = WordWindowChunker(max_words=3, overlap_words=1)
        first = chunker.chunk(
            Document("one two three four", "txt", {"source_name": "first.txt"})
        )
        repeated = chunker.chunk(
            Document("one two three four", "txt", {"source_name": "first.txt"})
        )
        other_source = chunker.chunk(
            Document("one two three four", "txt", {"source_name": "second.txt"})
        )

        self.assertTrue(first[0].document_id.startswith("doc_"))
        self.assertEqual(first[0].document_id, repeated[0].document_id)
        self.assertNotEqual(first[0].document_id, other_source[0].document_id)
        self.assertEqual([chunk.metadata["word_start"] for chunk in first], [0, 2])


class VectorStoreContractTests(unittest.TestCase):
    def test_rejected_replacements_preserve_existing_records(self):
        """Validation must finish before a bad replacement can damage stored data."""

        store = InMemoryVectorStore()
        store.replace_document("a", [record("a1", "a", (1.0, 0.0))])

        invalid_replacements = (
            (
                ComponentContractError,
                [record("wrong-document", "b", (1.0, 0.0))],
            ),
            (
                VectorDimensionError,
                [
                    record("mixed-2d", "a", (1.0, 0.0)),
                    record("mixed-3d", "a", (1.0, 0.0, 0.0)),
                ],
            ),
            (
                VectorDimensionError,
                [record("incompatible-3d", "a", (1.0, 0.0, 0.0))],
            ),
        )
        for error, replacement in invalid_replacements:
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    store.replace_document("a", replacement)
                self.assertEqual(store.count, 1)
                self.assertEqual(
                    store.search((1.0, 0.0), top_k=1)[0].chunk.id,
                    "a1",
                )

    def test_deleting_the_last_document_resets_vector_dimension(self):
        """An empty store may be reused with a different embedding model dimension."""

        store = InMemoryVectorStore()
        store.replace_document("a", [record("a1", "a", (1.0, 0.0))])

        self.assertEqual(store.delete_document("a"), 1)
        store.replace_document("b", [record("b1", "b", (1.0, 0.0, 0.0))])

        self.assertEqual(store.search((1.0, 0.0, 0.0), top_k=1)[0].chunk.id, "b1")

    def test_search_supports_reserved_filters_and_deterministic_ties(self):
        """Document/chunk filters are exact and equal scores sort by stable chunk ID."""

        store = InMemoryVectorStore()
        store.replace_document(
            "a",
            [
                record("a2", "a", (1.0, 0.0), tenant="acme"),
                record("a1", "a", (1.0, 0.0), tenant="acme"),
            ],
        )

        tied = store.search((1.0, 0.0), top_k=2)
        selected = store.search(
            (1.0, 0.0),
            top_k=2,
            filters={"document_id": "a", "chunk_id": "a2"},
        )

        self.assertEqual([result.chunk.id for result in tied], ["a1", "a2"])
        self.assertEqual([result.chunk.id for result in selected], ["a2"])

    def test_search_rejects_nonpositive_top_k(self):
        """Zero or negative limits are caller errors, not empty-search requests."""

        store = InMemoryVectorStore()
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k must be positive"):
                    store.search((1.0,), top_k=top_k)


class IndexerContractTests(unittest.TestCase):
    def test_indexer_rejects_an_empty_chunk_sequence(self):
        """Indexing an apparently successful document with no evidence must fail."""

        class EmptyChunker:
            def chunk(self, document):
                del document
                return ()

        indexer = Indexer(
            object(), EmptyChunker(), HashingEmbedder(8), InMemoryVectorStore()
        )

        with self.assertRaisesRegex(ComponentContractError, "returned no chunks"):
            indexer.index_document(Document("useful text", "txt"))

    def test_indexer_rejects_chunks_with_multiple_document_ids(self):
        """One replacement operation cannot safely contain multiple document owners."""

        class MixedChunker:
            def chunk(self, document):
                del document
                return (
                    Chunk("a:0", "a", "first", 0),
                    Chunk("b:0", "b", "second", 1),
                )

        indexer = Indexer(
            object(), MixedChunker(), HashingEmbedder(8), InMemoryVectorStore()
        )

        with self.assertRaisesRegex(ComponentContractError, "share one document_id"):
            indexer.index_document(Document("useful text", "txt"))

    def test_indexer_updates_and_deletes_auxiliary_document_indexes(self):
        """Optional query indexes must follow the primary vector-store lifecycle."""

        class RecordingDocumentIndex:
            def __init__(self):
                self.replaced = []
                self.deleted = []

            def replace_document(self, document):
                self.replaced.append(document)

            def delete_document(self, document_id):
                self.deleted.append(document_id)
                return 1

        auxiliary = RecordingDocumentIndex()
        store = InMemoryVectorStore()
        indexer = Indexer(
            object(),
            WordWindowChunker(max_words=10, overlap_words=0),
            HashingEmbedder(8),
            store,
            document_indexes=(auxiliary,),
        )
        document = Document("one short chunk", "txt", {"document_id": "guide"})

        report = indexer.index_document(document)
        deleted = indexer.delete("guide")

        self.assertEqual((report.document_id, report.chunk_count), ("guide", 1))
        self.assertEqual(auxiliary.replaced, [document])
        self.assertEqual(deleted, 1)
        self.assertEqual(auxiliary.deleted, ["guide"])
        self.assertEqual(store.count, 0)


class EmbedderContractTests(unittest.TestCase):
    def test_hashing_embedder_rejects_nonpositive_dimensions(self):
        """Every embedding must have at least one coordinate."""

        for dimensions in (0, -1):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
                    HashingEmbedder(dimensions)


class AnswerVerifierContractTests(unittest.TestCase):
    def test_structural_verifier_receives_the_answer_and_exact_contexts(self):
        """Verifier adapters need no inheritance and receive generation evidence unchanged."""

        class RecordingVerifier:
            def __init__(self):
                self.calls = []

            def verify(self, question, answer, contexts):
                self.calls.append((question, answer, tuple(contexts)))
                return VerificationResult(True, "supported by the supplied context")

        contexts = (
            SearchResult(Chunk("c1", "guide", "supporting evidence", 0), 0.9),
        )
        verifier: AnswerVerifier = RecordingVerifier()

        result = verifier.verify("question", "answer", contexts)

        self.assertEqual(verifier.calls, [("question", "answer", contexts)])
        self.assertTrue(result.supported)
        self.assertEqual(result.reason, "supported by the supplied context")

    def test_verification_result_validates_fields_and_snapshots_metadata(self):
        """Verifier output has an unambiguous verdict and stable top-level diagnostics."""

        metadata = {"provider": "example"}
        result = VerificationResult(False, "claim is unsupported", metadata)
        metadata["provider"] = "changed"

        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "claim is unsupported")
        self.assertEqual(result.metadata, {"provider": "example"})

        invalid_fields = (
            ({"supported": 1}, TypeError, "supported"),
            ({"supported": True, "reason": None}, TypeError, "reason"),
        )
        for arguments, error, message in invalid_fields:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(error, message):
                    VerificationResult(**arguments)


class RAGServiceContractTests(unittest.TestCase):
    class EmptyRetriever:
        def retrieve(self, query, *, top_k, filters=None):
            del query, top_k, filters
            return ()

    class Generator:
        def generate(self, question, contexts):
            del question, contexts
            return "answer"

    def test_service_rejects_invalid_default_limits(self):
        """Defaults must be positive and fetch enough candidates for final selection."""

        invalid_limits = (
            {"default_top_k": 0, "default_fetch_k": 1},
            {"default_top_k": 1, "default_fetch_k": 0},
            {"default_top_k": 3, "default_fetch_k": 2},
        )
        for limits in invalid_limits:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    RAGService(self.EmptyRetriever(), self.Generator(), **limits)

    def test_service_rejects_invalid_requests_and_method_registrations(self):
        """Bad questions, limits, and ambiguous method names fail explicitly."""

        retriever = self.EmptyRetriever()
        service = RAGService(retriever, self.Generator())

        invalid_requests = (
            (ValueError, {"question": "   "}),
            (ValueError, {"question": "valid", "top_k": 0}),
            (TypeError, {"question": "valid", "method": 7}),
            (ValueError, {"question": "valid", "method": "  "}),
        )
        for error, request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(error):
                    service.ask(**request)

        with self.assertRaisesRegex(ValueError, "reserved"):
            RAGService(
                retriever,
                self.Generator(),
                query_methods={" STANDARD ": retriever},
            )
        with self.assertRaisesRegex(ValueError, "Duplicate query method"):
            RAGService(
                retriever,
                self.Generator(),
                query_methods={"qasc": retriever, " QASC ": retriever},
            )

    def test_reranking_fetches_more_candidates_than_it_returns(self):
        """The service separates retrieval breadth from the final response limit."""

        class RecordingRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, query, *, top_k, filters=None):
                self.calls.append((query, top_k, filters))
                return tuple(
                    SearchResult(
                        Chunk(
                            "c{}".format(index),
                            "guide",
                            "text {}".format(index),
                            index,
                        ),
                        1.0 - index / 10.0,
                    )
                    for index in range(top_k)
                )

        class RecordingReranker:
            def __init__(self):
                self.calls = []

            def rerank(self, query, results, *, top_k):
                self.calls.append((query, tuple(results), top_k))
                return tuple(reversed(results))[:top_k]

        retriever = RecordingRetriever()
        reranker = RecordingReranker()
        service = RAGService(
            retriever,
            self.Generator(),
            reranker=reranker,
            default_top_k=2,
            default_fetch_k=4,
        )

        response = service.ask("question", filters={"tenant": "acme"})

        self.assertEqual(retriever.calls, [("question", 4, {"tenant": "acme"})])
        self.assertEqual(reranker.calls[0][2], 2)
        self.assertEqual(len(reranker.calls[0][1]), 4)
        self.assertEqual([result.chunk.id for result in response.results], ["c3", "c2"])

    def test_citations_normalize_excerpts_and_apply_source_fallbacks(self):
        """Citation display remains readable and traceable with sparse metadata."""

        class Retriever:
            def retrieve(self, query, *, top_k, filters=None):
                del query, filters
                results = (
                    SearchResult(
                        Chunk(
                            "c1",
                            "guide",
                            "  first\n\n  evidence  ",
                            0,
                            {"source_path": "/docs/guide.txt"},
                        ),
                        0.9,
                    ),
                    SearchResult(Chunk("c2", "fallback", "second", 0), 0.8),
                )
                return results[:top_k]

        response = RAGService(Retriever(), self.Generator()).ask(
            "question", top_k=2
        )

        self.assertEqual([citation.number for citation in response.citations], [1, 2])
        self.assertEqual(
            [citation.source for citation in response.citations],
            ["/docs/guide.txt", "fallback"],
        )
        self.assertEqual(response.citations[0].excerpt, "first evidence")


class FactoryContractTests(unittest.TestCase):
    def test_qasc_configuration_requires_explicit_enablement(self):
        """Supplying optional QASC parts cannot silently change indexing cost."""

        class Segmenter:
            def segment(self, document):
                del document
                return ()

        optional_arguments = (
            {"qasc_segmenter": Segmenter()},
            {"qasc_config": QASCConfig()},
        )
        for arguments in optional_arguments:
            with self.subTest(arguments=tuple(arguments)):
                with self.assertRaisesRegex(ValueError, "enable_qasc=True"):
                    build_demo_rag(**arguments)


if __name__ == "__main__":
    unittest.main()
