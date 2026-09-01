import unittest

from rag_ingestion import Document, DocumentSource, create_default_pipeline

from modular_rag import (
    Chunk,
    ComponentContractError,
    HashingEmbedder,
    InMemoryVectorStore,
    Indexer,
    QASCConfig,
    QASCRetriever,
    RAGService,
    SearchResult,
    SentenceSpan,
    VectorDimensionError,
    VectorRecord,
    WordWindowChunker,
    build_demo_rag,
)


class PipeSentenceSegmenter:
    """Dependency-free sentence test double using ``|`` as a boundary."""

    def segment(self, document):
        document_id = str(document.metadata["document_id"])
        sentences = []
        cursor = 0
        for index, text in enumerate(document.text.split("|")):
            start = cursor
            end = start + len(text)
            metadata = dict(document.metadata)
            metadata["sentence_index"] = index
            sentences.append(
                SentenceSpan(
                    id="{}:sentence:{}".format(document_id, index),
                    document_id=document_id,
                    text=text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata,
                )
            )
            cursor = end + 1
        return tuple(sentences)


class MappedEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        return tuple(self.vectors[text] for text in texts)

    def embed_query(self, text):
        return self.vectors[text]


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


class FactoryIntegrationTests(unittest.TestCase):
    def test_demo_factory_accepts_a_pipeline_with_a_new_format(self):
        """The composition root must honor an injected ingestion pipeline."""

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

    def test_replacement_removes_old_chunks_without_touching_other_documents(self):
        """Sequential replacement proves document isolation and removal semantics."""

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
        self.assertEqual(
            [
                result.chunk.id
                for result in store.search((1.0, 0.0), top_k=5)
            ],
            ["a3", "b1"],
        )
        self.assertEqual(store.delete_document("a"), 1)
        self.assertEqual(
            [
                result.chunk.id
                for result in store.search((1.0, 0.0), top_k=5)
            ],
            ["b1"],
        )

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
        self.assertEqual(report.chunk_count, 1)
        self.assertIn("PostgreSQL", response.answer)
        self.assertIn("[1]", response.answer)
        self.assertEqual(len(response.results), 1)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source, "architecture.txt")
        self.assertEqual(response.citations[0].document_id, "architecture")
        self.assertIn("PostgreSQL", response.citations[0].excerpt)

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

    def test_reindex_and_delete_remove_stale_evidence_end_to_end(self):
        """The application lifecycle cannot retrieve superseded or deleted chunks."""

        app = build_demo_rag(max_words=10, overlap_words=0)
        app.index(
            DocumentSource.from_text(
                "The obsolete status is red.",
                name="status.txt",
                metadata={"document_id": "status"},
            )
        )
        app.index(
            DocumentSource.from_text(
                "The current status is green.",
                name="status.txt",
                metadata={"document_id": "status"},
            )
        )

        response = app.ask("What is the obsolete status?")

        self.assertEqual(app.store.count, 1)
        self.assertNotIn("obsolete status is red", response.citations[0].excerpt)
        self.assertIn("current status is green", response.citations[0].excerpt)
        self.assertEqual(app.indexer.delete("status"), 1)
        self.assertEqual(app.ask("What is the status?").citations, ())

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

    def test_query_method_flag_selects_a_registered_retriever(self):
        standard_chunk = Chunk("standard", "d", "standard evidence", 0)
        qasc_chunk = Chunk("qasc", "d", "adaptive evidence", 0)

        class FixedRetriever:
            def __init__(self, chunk):
                self.chunk = chunk

            def retrieve(self, query, *, top_k, filters=None):
                return [SearchResult(self.chunk, 0.9)]

        class Generator:
            def generate(self, question, contexts):
                return contexts[0].chunk.text

        rag = RAGService(
            FixedRetriever(standard_chunk),
            Generator(),
            query_methods={"qasc": FixedRetriever(qasc_chunk)},
        )

        self.assertEqual(rag.ask("question").answer, "standard evidence")
        self.assertEqual(
            rag.ask("question", method="QASC").answer, "adaptive evidence"
        )
        self.assertEqual(rag.available_methods, ("standard", "qasc"))

        with self.assertRaisesRegex(ValueError, "Unknown query method"):
            rag.ask("question", method="missing")

    def test_demo_factory_only_indexes_qasc_when_enabled(self):
        standard_app = build_demo_rag()
        self.assertEqual(standard_app.rag.available_methods, ("standard",))
        self.assertEqual(standard_app.indexer.document_indexes, ())

        app = build_demo_rag(
            enable_qasc=True,
            qasc_segmenter=PipeSentenceSegmenter(),
            qasc_config=QASCConfig(window_radius=0, gap_tolerance=1),
        )
        app.index(
            DocumentSource.from_text(
                "Mercury is a planet|Venus has a dense atmosphere|Mars is red",
                name="planets.txt",
                metadata={"document_id": "planets"},
            )
        )

        response = app.ask("Which planet has a dense atmosphere?", method="qasc")

        self.assertEqual(app.rag.available_methods, ("standard", "qasc"))
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].chunk.metadata["chunking_method"], "qasc")

        app.indexer.delete("planets")
        self.assertEqual(
            app.ask("Which planet has a dense atmosphere?", method="qasc").results,
            (),
        )

    def test_hashing_embedder_is_deterministic_normalized_and_input_sensitive(self):
        """A constant or malformed vector must not satisfy determinism by itself."""

        embedder = HashingEmbedder(32)
        first = embedder.embed_query("repeatable text")
        second = embedder.embed_query("repeatable text")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertNotEqual(first, embedder.embed_query("different tokens"))
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)
        self.assertEqual(embedder.embed_query(""), (0.0,) * 32)


class QASCRetrieverTests(unittest.TestCase):
    def test_selects_seeds_merges_nearby_windows_and_applies_filters(self):
        vectors = {
            "question": (1.0, 0.0),
            "zero": (0.0, 1.0),
            "one": (0.0, 1.0),
            "relevant alpha": (0.8, 0.6),
            "relevant beta": (0.9, 0.435889894),
            "four": (0.0, 1.0),
            "five": (0.0, 1.0),
        }
        retriever = QASCRetriever(
            PipeSentenceSegmenter(),
            MappedEmbedder(vectors),
            config=QASCConfig(
                seed_percentile=75,
                window_radius=0,
                gap_tolerance=1,
                chunk_threshold_factor=0.6,
            ),
        )
        retriever.replace_document(
            Document(
                "zero|one|relevant alpha|relevant beta|four|five",
                "txt",
                {"document_id": "guide", "tenant": "acme"},
            )
        )

        hidden = retriever.retrieve(
            "question", top_k=5, filters={"tenant": "other"}
        )
        results = retriever.retrieve(
            "question", top_k=5, filters={"tenant": "acme"}
        )

        self.assertEqual(hidden, ())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.text, "relevant alpha|relevant beta")
        self.assertEqual(results[0].chunk.metadata["sentence_start"], 2)
        self.assertEqual(results[0].chunk.metadata["sentence_end"], 4)
        self.assertEqual(results[0].chunk.metadata["seed_sentences"], (2, 3))

    def test_expands_a_seed_with_its_surrounding_context(self):
        vectors = {
            "question": (1.0, 0.0),
            "left context": (0.2, 0.979795897),
            "answer seed": (1.0, 0.0),
            "right context": (0.2, 0.979795897),
            "noise": (0.0, 1.0),
        }
        retriever = QASCRetriever(
            PipeSentenceSegmenter(),
            MappedEmbedder(vectors),
            config=QASCConfig(seed_percentile=75, window_radius=1),
        )
        retriever.replace_document(
            Document(
                "left context|answer seed|right context|noise",
                "txt",
                {"document_id": "guide"},
            )
        )

        results = retriever.retrieve("question", top_k=1)

        self.assertEqual(
            results[0].chunk.text,
            "left context|answer seed|right context",
        )

    def test_delete_removes_the_optional_sentence_index(self):
        vectors = {"question": (1.0, 0.0), "relevant": (1.0, 0.0)}
        retriever = QASCRetriever(
            PipeSentenceSegmenter(), MappedEmbedder(vectors)
        )
        retriever.replace_document(
            Document("relevant", "txt", {"document_id": "guide"})
        )

        self.assertEqual(retriever.document_count, 1)
        self.assertEqual(retriever.delete_document("guide"), 1)
        self.assertEqual(retriever.document_count, 0)
        self.assertEqual(retriever.retrieve("question", top_k=1), ())


if __name__ == "__main__":
    unittest.main()
