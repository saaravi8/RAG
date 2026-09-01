import unittest

from rag_ingestion import Document, DocumentSource, create_default_pipeline

from modular_rag import (
    Chunk,
    ComponentContractError,
    HashingEmbedder,
    InMemoryVectorStore,
    Indexer,
    RAGService,
    SearchResult,
    VectorDimensionError,
    VectorRecord,
    WordWindowChunker,
    build_demo_rag,
)


class ChunkerTests(unittest.TestCase):
    def test_word_windows_overlap_and_keep_stable_document_id(self):
        document = Document(
            "one two three four five six seven",
            "txt",
            {"document_id": "guide"},
        )
        chunker = WordWindowChunker(max_words=4, overlap_words=1)

        chunks = chunker.chunk(document)

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three four",
            "four five six seven",
        ])
        self.assertTrue(all(chunk.document_id == "guide" for chunk in chunks))
        self.assertEqual(chunks[1].metadata["word_start"], 3)

    def test_blank_document_returns_no_chunks(self):
        self.assertEqual(WordWindowChunker().chunk(Document("  ", "txt")), ())

    def test_demo_factory_accepts_a_pipeline_with_a_new_format(self):
        processor = create_default_pipeline()

        @processor.register_loader("note")
        def load_note(source):
            return Document(source.read_text(), source.document_type)

        app = build_demo_rag(processor=processor)

        report = app.index(
            DocumentSource.from_text(
                "A newly supported document.",
                name="example.note",
                metadata={"document_id": "note"},
            )
        )

        self.assertEqual(report.document_id, "note")


class InMemoryVectorStoreTests(unittest.TestCase):
    def record(self, chunk_id, document_id, vector, **metadata):
        return VectorRecord(
            Chunk(chunk_id, document_id, chunk_id, 0, metadata),
            vector,
        )

    def test_search_filters_and_replaces_a_document_atomically(self):
        store = InMemoryVectorStore()
        store.replace_document(
            "a",
            [
                self.record("a1", "a", (1.0, 0.0), tenant="one"),
                self.record("a2", "a", (0.0, 1.0), tenant="one"),
            ],
        )
        store.replace_document(
            "b", [self.record("b1", "b", (1.0, 0.0), tenant="two")]
        )

        filtered = store.search((1.0, 0.0), top_k=5, filters={"tenant": "one"})
        self.assertEqual([result.chunk.id for result in filtered], ["a1", "a2"])

        store.replace_document(
            "a", [self.record("a3", "a", (1.0, 0.0), tenant="one")]
        )
        self.assertEqual(store.count, 2)
        self.assertEqual(store.delete_document("a"), 1)

    def test_vector_dimensions_are_validated(self):
        store = InMemoryVectorStore()
        store.replace_document("a", [self.record("a1", "a", (1.0, 0.0))])

        with self.assertRaises(VectorDimensionError):
            store.search((1.0,), top_k=1)


class IndexerTests(unittest.TestCase):
    def test_component_contract_detects_embedding_count_mismatch(self):
        class Processor:
            def process(self, source, **kwargs):
                return Document("some useful text", "txt", {"document_id": "x"})

        class BrokenEmbedder:
            def embed_documents(self, texts):
                return ()

            def embed_query(self, text):
                return (1.0,)

        indexer = Indexer(
            Processor(),
            WordWindowChunker(),
            BrokenEmbedder(),
            InMemoryVectorStore(),
        )

        with self.assertRaises(ComponentContractError):
            indexer.index(b"ignored", document_type="txt")


class EndToEndRAGTests(unittest.TestCase):
    def test_demo_pipeline_indexes_retrieves_answers_and_cites(self):
        app = build_demo_rag(max_words=20, overlap_words=3)
        report = app.index(
            DocumentSource.from_text(
                "The application database is PostgreSQL. "
                "Redis is used only for temporary caching.",
                name="architecture.txt",
                metadata={"document_id": "architecture", "tenant": "acme"},
            )
        )

        response = app.ask(
            "Which application database is used?", filters={"tenant": "acme"}
        )

        self.assertEqual(report.document_id, "architecture")
        self.assertIn("PostgreSQL", response.answer)
        self.assertEqual(response.citations[0].source, "architecture.txt")
        self.assertEqual(response.citations[0].document_id, "architecture")

    def test_filters_enforce_document_visibility(self):
        app = build_demo_rag()
        app.index(
            DocumentSource.from_text(
                "Private launch code is blue.",
                name="secret.txt",
                metadata={"document_id": "secret", "tenant": "one"},
            )
        )

        response = app.ask("What is the launch code?", filters={"tenant": "two"})

        self.assertEqual(response.citations, ())
        self.assertIn("not have enough", response.answer)

    def test_service_accepts_completely_custom_retriever_and_generator(self):
        chunk = Chunk("c", "d", "custom evidence", 0)

        class Retriever:
            def retrieve(self, query, *, top_k, filters=None):
                return [SearchResult(chunk, 0.9)]

        class Generator:
            def generate(self, question, contexts):
                return "custom answer"

        response = RAGService(Retriever(), Generator()).ask("question")

        self.assertEqual(response.answer, "custom answer")
        self.assertEqual(response.results[0].chunk.id, "c")

    def test_hashing_embedder_is_deterministic(self):
        embedder = HashingEmbedder(32)
        self.assertEqual(
            embedder.embed_query("repeatable text"),
            embedder.embed_query("repeatable text"),
        )


if __name__ == "__main__":
    unittest.main()
