import math
import unittest

from rag_ingestion import Document

from modular_rag import (
    Citation,
    Chunk,
    ComponentContractError,
    InMemoryVectorStore,
    QASCRetriever,
    SearchResult,
    SentenceSpan,
    VectorDimensionError,
    VectorRecord,
)


def record(chunk_id, document_id, vector):
    return VectorRecord(
        Chunk(chunk_id, document_id, chunk_id, 0),
        vector,
    )


def unsafe_record(chunk_id, document_id, vector):
    """Bypass model validation to verify the storage trust boundary itself."""

    candidate = record(chunk_id, document_id, (1.0,))
    object.__setattr__(candidate, "vector", vector)
    return candidate


def spans(document_id="guide"):
    return (
        SentenceSpan(
            "{}:0".format(document_id),
            document_id,
            "alpha",
            0,
            0,
            5,
        ),
        SentenceSpan(
            "{}:1".format(document_id),
            document_id,
            "beta",
            1,
            6,
            10,
        ),
    )


class FixedSegmenter:
    def segment(self, document):
        return spans(str(document.metadata.get("document_id", "guide")))


class MutableEmbedder:
    def __init__(self, document_vectors, query_vector=(1.0, 0.0)):
        self.document_vectors = document_vectors
        self.query_vector = query_vector

    def embed_documents(self, texts):
        del texts
        return self.document_vectors

    def embed_query(self, text):
        del text
        return self.query_vector


class InMemoryVectorValidationTests(unittest.TestCase):
    def test_document_state_token_requires_present_consistent_chunk_metadata(self):
        """Direct writes need one common token to identify repository state."""

        store = InMemoryVectorStore()
        token = "a" * 64
        store.replace_document(
            "guide",
            (
                VectorRecord(
                    Chunk(
                        "one",
                        "guide",
                        "one",
                        0,
                        {"document_state_token": token},
                    ),
                    (1.0, 0.0),
                ),
                VectorRecord(
                    Chunk(
                        "two",
                        "guide",
                        "two",
                        1,
                        {"document_state_token": token},
                    ),
                    (0.0, 1.0),
                ),
            ),
        )
        self.assertEqual(store.document_state_token("guide"), token)

        store.replace_document(
            "guide", (record("manual", "guide", (1.0, 0.0)),)
        )
        self.assertIsNone(store.document_state_token("guide"))
        self.assertIsNone(store.document_state_token("missing"))

    def test_many_replacements_and_token_reads_do_not_scan_the_global_record_map(self):
        """Per-document ownership makes writes and unchanged checks locally bounded."""

        class NoIterationDict(dict):
            def __iter__(self):
                raise AssertionError("global record iteration is not allowed")

            def items(self):
                raise AssertionError("global record iteration is not allowed")

            def values(self):
                raise AssertionError("global record iteration is not allowed")

            def copy(self):
                raise AssertionError("global record copying is not allowed")

        store = InMemoryVectorStore()
        store._records = NoIterationDict()
        record_map = store._records
        document_count = 200

        for index in range(document_count):
            document_id = "document-{}".format(index)
            token = "{:064x}".format(index + 1)
            store.replace_document(
                document_id,
                (
                    VectorRecord(
                        Chunk(
                            "{}:0".format(document_id),
                            document_id,
                            "content",
                            0,
                            {"document_state_token": token},
                        ),
                        (1.0, 0.0),
                    ),
                ),
            )

        for index in range(document_count):
            self.assertEqual(
                store.document_state_token("document-{}".format(index)),
                "{:064x}".format(index + 1),
            )
        for index in range(0, document_count, 2):
            self.assertEqual(
                store.delete_document("document-{}".format(index)), 1
            )
            self.assertIsNone(
                store.document_state_token("document-{}".format(index))
            )
        self.assertIs(store._records, record_map)
        self.assertEqual(store.count, document_count // 2)

    def test_replacement_rejects_duplicate_and_cross_document_chunk_ids(self):
        """Chunk identity collisions cannot silently discard stored evidence."""

        store = InMemoryVectorStore()
        store.replace_document(
            "existing", [record("shared", "existing", (1.0, 0.0))]
        )

        with self.assertRaisesRegex(ComponentContractError, "unique chunk IDs"):
            store.replace_document(
                "new",
                [
                    record("duplicate", "new", (1.0, 0.0)),
                    record("duplicate", "new", (0.0, 1.0)),
                ],
            )
        with self.assertRaisesRegex(ComponentContractError, "another document"):
            store.replace_document(
                "new", [record("shared", "new", (0.0, 1.0))]
            )

        self.assertEqual(store.count, 1)
        result = store.search((1.0, 0.0), top_k=1)[0]
        self.assertEqual(
            (result.chunk.id, result.chunk.document_id),
            ("shared", "existing"),
        )

    def test_competing_prepares_reject_the_stale_store_commit(self):
        """A prepared replacement cannot overwrite a newer committed candidate."""

        store = InMemoryVectorStore()
        store.replace_document("guide", [record("base", "guide", (1.0, 0.0))])
        first = store.prepare_replace_document(
            "guide", [record("first", "guide", (1.0, 0.0))]
        )
        stale = store.prepare_replace_document(
            "guide", [record("stale", "guide", (1.0, 0.0))]
        )

        first.commit()
        with self.assertRaisesRegex(ComponentContractError, "stale"):
            stale.commit()
        stale.rollback()
        stale.rollback()

        self.assertEqual(store.count, 1)
        self.assertEqual(store.search((1.0, 0.0), top_k=1)[0].chunk.id, "first")

    def test_store_rollback_does_not_clobber_an_intervening_write_or_delete(self):
        """Rollback is safe once a later state generation owns visibility."""

        store = InMemoryVectorStore()
        store.replace_document("guide", [record("base", "guide", (1.0, 0.0))])
        prepared = store.prepare_replace_document(
            "guide", [record("candidate", "guide", (1.0, 0.0))]
        )
        prepared.commit()
        store.replace_document("other", [record("newer", "other", (1.0, 0.0))])
        self.assertEqual(store.delete_document("guide"), 1)

        prepared.rollback()
        prepared.rollback()

        self.assertEqual(store.count, 1)
        self.assertEqual(store.search((1.0, 0.0), top_k=1)[0].chunk.id, "newer")

    def test_replacement_rejects_empty_non_finite_and_malformed_vectors(self):
        """Every stored coordinate sequence is safe before state can change."""

        store = InMemoryVectorStore()
        store.replace_document("guide", [record("valid", "guide", (1.0, 0.0))])
        invalid_vectors = (
            ((), "cannot be empty"),
            ((math.nan, 0.0), "non-finite"),
            ((math.inf, 0.0), "non-finite"),
            ((-math.inf, 0.0), "non-finite"),
            ((object(), 0.0), "malformed"),
            ((True, 0.0), "malformed"),
            (("1.0", 0.0), "malformed"),
            ("12", "malformed"),
            (object(), "malformed"),
        )

        for vector, message in invalid_vectors:
            with self.subTest(vector=vector, message=message):
                with self.assertRaisesRegex(ComponentContractError, message):
                    store.replace_document(
                        "guide", [unsafe_record("invalid", "guide", vector)]
                    )
                self.assertEqual(store.count, 1)
                self.assertEqual(
                    store.search((1.0, 0.0), top_k=1)[0].chunk.id,
                    "valid",
                )

    def test_replacement_rejects_mixed_dimensions_and_accepts_valid_vectors(self):
        """A replacement is rectangular, while ordinary finite vectors still work."""

        store = InMemoryVectorStore()
        with self.assertRaisesRegex(VectorDimensionError, "mixed dimensions"):
            store.replace_document(
                "guide",
                [
                    record("two-dimensional", "guide", (1.0, 0.0)),
                    record("three-dimensional", "guide", (1.0, 0.0, 0.0)),
                ],
            )

        store.replace_document(
            "guide",
            [
                record("x", "guide", (1.0, 0.0)),
                record("y", "guide", (0.0, 1.0)),
            ],
        )
        self.assertEqual(
            [result.chunk.id for result in store.search((1, 0), top_k=2)],
            ["x", "y"],
        )

    def test_search_rejects_empty_non_finite_malformed_and_wrong_dimension_vectors(self):
        """Invalid queries never reach cosine scoring or return misleading results."""

        store = InMemoryVectorStore()
        store.replace_document("guide", [record("valid", "guide", (1.0, 0.0))])
        invalid_queries = (
            ((), ComponentContractError, "cannot be empty"),
            ((math.nan, 0.0), ComponentContractError, "non-finite"),
            ((math.inf, 0.0), ComponentContractError, "non-finite"),
            ((object(), 0.0), ComponentContractError, "malformed"),
            (("1.0", 0.0), ComponentContractError, "malformed"),
            ((1.0,), VectorDimensionError, "Expected query dimension"),
        )

        for vector, error, message in invalid_queries:
            with self.subTest(vector=vector, error=error.__name__):
                with self.assertRaisesRegex(error, message):
                    store.search(vector, top_k=1)

    def test_search_top_k_requires_a_non_boolean_integer(self):
        """Direct store callers cannot pass booleans or lossy numeric limits."""

        store = InMemoryVectorStore()
        store.replace_document("guide", [record("valid", "guide", (1.0, 0.0))])

        for value in (True, False, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    store.search((1.0, 0.0), top_k=value)

    def test_store_cosine_stays_finite_for_extreme_finite_coordinates(self):
        """Finite inputs near the float limit must not overflow into a NaN score."""

        extreme = (1e308, 1e308)
        store = InMemoryVectorStore()
        store.replace_document("guide", [record("extreme", "guide", extreme)])

        result = store.search(extreme, top_k=1)[0]

        self.assertTrue(math.isfinite(result.score))
        self.assertAlmostEqual(result.score, 1.0)


class QASCVectorValidationTests(unittest.TestCase):
    @staticmethod
    def _version(retriever, document_id="guide"):
        return retriever.retrieve(
            "question", top_k=1, filters={"document_id": document_id}
        )[0].chunk.metadata["version"]

    def test_competing_prepares_reject_the_stale_qasc_commit(self):
        """A QASC candidate cannot replace a sentence index published after prepare."""

        embedder = MutableEmbedder(((1.0, 0.0), (0.0, 1.0)))
        retriever = QASCRetriever(FixedSegmenter(), embedder)
        retriever.replace_document(
            Document("alpha beta", "txt", {"document_id": "guide", "version": "base"})
        )
        first = retriever.prepare_replace_document(
            Document("alpha beta", "txt", {"document_id": "guide", "version": "first"})
        )
        stale = retriever.prepare_replace_document(
            Document("alpha beta", "txt", {"document_id": "guide", "version": "stale"})
        )

        first.commit()
        with self.assertRaisesRegex(ComponentContractError, "stale"):
            stale.commit()
        stale.rollback()
        stale.rollback()

        self.assertEqual(retriever.document_count, 1)
        self.assertEqual(self._version(retriever), "first")

    def test_qasc_rollback_does_not_clobber_an_intervening_write_or_delete(self):
        """QASC rollback cannot resurrect an index after a newer state takes over."""

        embedder = MutableEmbedder(((1.0, 0.0), (0.0, 1.0)))
        retriever = QASCRetriever(FixedSegmenter(), embedder)
        retriever.replace_document(
            Document("alpha beta", "txt", {"document_id": "guide", "version": "base"})
        )
        prepared = retriever.prepare_replace_document(
            Document("alpha beta", "txt", {"document_id": "guide", "version": "candidate"})
        )
        prepared.commit()
        retriever.replace_document(
            Document("alpha beta", "txt", {"document_id": "other", "version": "newer"})
        )
        self.assertEqual(retriever.delete_document("guide"), 2)

        prepared.rollback()
        prepared.rollback()

        self.assertEqual(retriever.document_count, 1)
        self.assertEqual(self._version(retriever, "other"), "newer")

    def test_sentence_index_rejects_empty_non_finite_and_malformed_vectors(self):
        """QASC validates every sentence embedding before publishing an index."""

        invalid_batches = (
            (((), (1.0, 0.0)), "cannot be empty"),
            (((math.nan, 0.0), (1.0, 0.0)), "non-finite"),
            (((math.inf, 0.0), (1.0, 0.0)), "non-finite"),
            (((-math.inf, 0.0), (1.0, 0.0)), "non-finite"),
            (((object(), 0.0), (1.0, 0.0)), "malformed"),
            (((True, 0.0), (1.0, 0.0)), "malformed"),
            ((("1.0", 0.0), (1.0, 0.0)), "malformed"),
            (("12", (1.0, 0.0)), "malformed"),
            ((1.0, (1.0, 0.0)), "malformed"),
            (None, "malformed"),
        )

        for vectors, message in invalid_batches:
            with self.subTest(vectors=vectors, message=message):
                embedder = MutableEmbedder(((1.0, 0.0), (0.0, 1.0)))
                retriever = QASCRetriever(FixedSegmenter(), embedder)
                retriever.replace_document(Document("alpha beta", "txt"))
                embedder.document_vectors = vectors
                with self.assertRaisesRegex(ComponentContractError, message):
                    retriever.replace_document(Document("alpha beta", "txt"))
                self.assertEqual(retriever.document_count, 1)
                self.assertTrue(retriever.retrieve("question", top_k=1))

    def test_sentence_index_rejects_mixed_dimensions_and_accepts_valid_vectors(self):
        """QASC requires a rectangular batch but preserves valid finite behavior."""

        embedder = MutableEmbedder(((1.0, 0.0), (1.0, 0.0, 0.0)))
        retriever = QASCRetriever(FixedSegmenter(), embedder)
        with self.assertRaisesRegex(VectorDimensionError, "mixed dimensions"):
            retriever.replace_document(Document("alpha beta", "txt"))

        embedder.document_vectors = ((1, 0), (0, 1))
        retriever.replace_document(Document("alpha beta", "txt"))
        self.assertEqual(retriever.document_count, 1)
        self.assertTrue(retriever.retrieve("question", top_k=1))

    def test_query_rejects_empty_non_finite_malformed_and_wrong_dimension_vectors(self):
        """QASC query vectors satisfy the same numeric contract as its index."""

        embedder = MutableEmbedder(((1.0, 0.0), (0.0, 1.0)))
        retriever = QASCRetriever(FixedSegmenter(), embedder)
        retriever.replace_document(Document("alpha beta", "txt"))
        invalid_queries = (
            ((), ComponentContractError, "cannot be empty"),
            ((math.nan, 0.0), ComponentContractError, "non-finite"),
            ((math.inf, 0.0), ComponentContractError, "non-finite"),
            ((object(), 0.0), ComponentContractError, "malformed"),
            (("1.0", 0.0), ComponentContractError, "malformed"),
            ((1.0,), VectorDimensionError, "Expected query dimension"),
        )

        for vector, error, message in invalid_queries:
            with self.subTest(vector=vector, error=error.__name__):
                embedder.query_vector = vector
                with self.assertRaisesRegex(error, message):
                    retriever.retrieve("question", top_k=1)

    def test_qasc_scores_stay_finite_for_extreme_finite_coordinates(self):
        """QASC supports finite float-limit vectors without NaN propagation."""

        extreme = (1e308, 1e308)
        embedder = MutableEmbedder((extreme, extreme), query_vector=extreme)
        retriever = QASCRetriever(FixedSegmenter(), embedder)
        retriever.replace_document(Document("alpha beta", "txt"))

        results = retriever.retrieve("question", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(math.isfinite(results[0].score))
        self.assertAlmostEqual(results[0].score, 1.0)


class NumericDomainModelTests(unittest.TestCase):
    def test_vector_record_rejects_non_finite_and_non_numeric_coordinates(self):
        """Unsafe vectors fail before they can reach any storage adapter."""

        chunk = Chunk("chunk", "guide", "evidence", 0)
        invalid = (
            ((math.nan,), ValueError),
            ((math.inf,), ValueError),
            ((-math.inf,), ValueError),
            ((True,), TypeError),
            (("1.0",), TypeError),
            ("12", TypeError),
            (object(), TypeError),
        )
        for vector, error in invalid:
            with self.subTest(vector=vector):
                with self.assertRaises(error):
                    VectorRecord(chunk, vector)

    def test_result_and_citation_scores_must_be_finite_numbers(self):
        """Ranking and citation models cannot carry corrupt score values."""

        chunk = Chunk("chunk", "guide", "evidence", 0)
        invalid = (
            (math.nan, ValueError),
            (math.inf, ValueError),
            (-math.inf, ValueError),
            (True, TypeError),
            ("0.5", TypeError),
        )
        for score, error in invalid:
            with self.subTest(model="SearchResult", score=score):
                with self.assertRaises(error):
                    SearchResult(chunk, score)
            with self.subTest(model="Citation", score=score):
                with self.assertRaises(error):
                    Citation(1, "chunk", "guide", "guide", "evidence", score)

        result = SearchResult(chunk, -0.25)
        citation = Citation(
            1, "chunk", "guide", "guide", "evidence", -0.25
        )
        self.assertEqual((result.score, citation.score), (-0.25, -0.25))


if __name__ == "__main__":
    unittest.main()
