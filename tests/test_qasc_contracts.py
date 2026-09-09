import unittest

import modular_rag.qasc as qasc_module
from rag_ingestion import Document

from modular_rag import (
    ComponentContractError,
    QASCConfig,
    QASCRetriever,
    SentenceSpan,
    VectorDimensionError,
)


def sentence(document_id, index, text):
    start = index * 10
    return SentenceSpan(
        id="{}:{}".format(document_id, index),
        document_id=document_id,
        text=text,
        index=index,
        start_char=start,
        end_char=start + len(text),
        metadata={"document_id": document_id, "sentence_index": index},
    )


class FixedSegmenter:
    def __init__(self, sentences):
        self.sentences = tuple(sentences)

    def segment(self, document):
        del document
        return self.sentences


class MutableEmbedder:
    def __init__(self, document_vectors, query_vector=(1.0, 0.0)):
        self.document_vectors = tuple(document_vectors)
        self.query_vector = query_vector

    def embed_documents(self, texts):
        del texts
        return self.document_vectors

    def embed_query(self, text):
        del text
        return self.query_vector


class QASCConfigTests(unittest.TestCase):
    def test_config_rejects_values_outside_each_supported_domain(self):
        """Every QASC tuning parameter has an explicit safe mathematical domain."""

        invalid_parameters = (
            {"seed_percentile": -0.1},
            {"seed_percentile": 100.1},
            {"window_radius": -1},
            {"decay": -0.1},
            {"gap_tolerance": -1},
            {"chunk_threshold_factor": -0.1},
            {"seed_percentile": float("nan")},
            {"seed_percentile": True},
            {"window_radius": 1.5},
            {"window_radius": False},
            {"decay": float("inf")},
            {"decay": True},
            {"gap_tolerance": 1.5},
            {"gap_tolerance": True},
            {"chunk_threshold_factor": float("nan")},
            {"chunk_threshold_factor": False},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    QASCConfig(**parameters)

        self.assertEqual(QASCConfig(seed_percentile=0).seed_percentile, 0)
        self.assertEqual(QASCConfig(seed_percentile=100).seed_percentile, 100)


class QASCRetrieverContractTests(unittest.TestCase):
    def test_state_token_is_read_only_and_remains_canonical_in_result_metadata(self):
        """Sentence metadata cannot replace the repository document ownership token."""

        span = SentenceSpan(
            "guide:0",
            "guide",
            "relevant",
            0,
            0,
            8,
            metadata={"document_state_token": "forged"},
        )
        retriever = QASCRetriever(
            FixedSegmenter((span,)),
            MutableEmbedder(((1.0, 0.0),)),
            config=QASCConfig(window_radius=0),
        )
        token = "a" * 64

        retriever.replace_document(
            Document(
                "relevant",
                "txt",
                {"document_id": "guide", "document_state_token": token},
            )
        )
        result = retriever.retrieve("question", top_k=1)[0]

        self.assertEqual(retriever.document_state_token("guide"), token)
        self.assertEqual(result.chunk.metadata["document_state_token"], token)
        self.assertIsNone(retriever.document_state_token("missing"))

        retriever.replace_document(
            Document("relevant", "txt", {"document_id": "guide"})
        )
        self.assertIsNone(retriever.document_state_token("guide"))

    def test_replacement_rejects_empty_mixed_or_noncontiguous_sentence_spans(self):
        """A sentence index requires one ordered, nonempty document sequence."""

        invalid_sequences = (
            ((), "returned no sentences"),
            (
                (sentence("a", 0, "first"), sentence("b", 1, "second")),
                "share one document_id",
            ),
            (
                (sentence("a", 0, "first"), sentence("a", 2, "third")),
                "contiguous and start at zero",
            ),
        )
        for sentences, message in invalid_sequences:
            with self.subTest(message=message):
                retriever = QASCRetriever(
                    FixedSegmenter(sentences),
                    MutableEmbedder(((1.0, 0.0),) * len(sentences)),
                )
                with self.assertRaisesRegex(ComponentContractError, message):
                    retriever.replace_document(Document("text", "txt"))

    def test_segmenter_output_must_match_the_canonical_document(self):
        """Malformed spans fail before embedding or replacing valid evidence."""

        class MutableSegmenter:
            spans = (sentence("guide", 0, "relevant"),)

            def segment(self, document):
                del document
                return self.spans

        class CountingEmbedder(MutableEmbedder):
            def __init__(self):
                super().__init__(((1.0, 0.0),))
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                return super().embed_documents(texts)

        segmenter = MutableSegmenter()
        embedder = CountingEmbedder()
        retriever = QASCRetriever(
            segmenter,
            embedder,
            config=QASCConfig(window_radius=0),
        )
        retriever.replace_document(
            Document("relevant", "txt", {"document_id": "guide"})
        )
        baseline_calls = embedder.calls

        invalid = (
            ((object(),), "SentenceSpan"),
            ((sentence("forged", 0, "relevant"),), "canonical document"),
            (
                (SentenceSpan("guide:0", "guide", "fabricated", 0, 0, 11),),
                "within the canonical",
            ),
            (
                (SentenceSpan("guide:0", "guide", "fake", 0, 0, 4),),
                "match its canonical",
            ),
            (
                (
                    SentenceSpan("guide:0", "guide", "alpha", 0, 0, 5),
                    SentenceSpan("guide:1", "guide", "ha", 1, 3, 5),
                ),
                "ordered and non-overlapping",
            ),
        )
        for spans, message in invalid:
            with self.subTest(message=message):
                segmenter.spans = spans
                with self.assertRaisesRegex(ComponentContractError, message):
                    retriever.replace_document(
                        Document(
                            "alpha beta",
                            "txt",
                            {"document_id": "guide"},
                        )
                    )
                self.assertEqual(embedder.calls, baseline_calls)
                self.assertEqual(retriever.document_count, 1)
                self.assertEqual(
                    retriever.retrieve("question", top_k=1)[0].chunk.text,
                    "relevant",
                )

    def test_document_metadata_is_deeply_snapshotted_when_qasc_publishes(self):
        """Nested caller mutations cannot change later QASC evidence."""

        nested = {"label": "before"}
        retriever = QASCRetriever(
            FixedSegmenter((sentence("guide", 0, "relevant"),)),
            MutableEmbedder(((1.0, 0.0),)),
            config=QASCConfig(window_radius=0),
        )
        retriever.replace_document(
            Document(
                "relevant",
                "txt",
                {"document_id": "guide", "custom": nested},
            )
        )

        nested["label"] = "after"

        result = retriever.retrieve("question", top_k=1)[0]
        self.assertEqual(result.chunk.metadata["custom"]["label"], "before")

    def test_replacement_rejects_invalid_sentence_embedding_batches(self):
        """Sentence count and vector dimensions must form a rectangular index."""

        sentences = (sentence("guide", 0, "first"), sentence("guide", 1, "second"))
        invalid_batches = (
            (ComponentContractError, ((1.0, 0.0),)),
            (ComponentContractError, ((), ())),
            (VectorDimensionError, ((1.0, 0.0), (1.0, 0.0, 0.0))),
        )
        for error, vectors in invalid_batches:
            with self.subTest(error=error.__name__, vectors=vectors):
                retriever = QASCRetriever(
                    FixedSegmenter(sentences), MutableEmbedder(vectors)
                )
                with self.assertRaises(error):
                    retriever.replace_document(
                        Document("first     second", "txt")
                    )

    def test_embedder_type_error_is_not_mislabeled_as_malformed_output(self):
        """An implementation error remains distinguishable from a bad sequence."""

        class FailingEmbedder(MutableEmbedder):
            def embed_documents(self, texts):
                del texts
                raise TypeError("provider implementation failed")

        retriever = QASCRetriever(
            FixedSegmenter((sentence("guide", 0, "first"),)),
            FailingEmbedder(()),
        )

        with self.assertRaisesRegex(TypeError, "provider implementation failed"):
            retriever.replace_document(Document("first", "txt"))

    def test_rejected_replacement_preserves_the_previous_sentence_index(self):
        """A failed re-index cannot erase evidence from the last valid version."""

        spans = (sentence("guide", 0, "relevant"),)
        embedder = MutableEmbedder(((1.0, 0.0),))
        retriever = QASCRetriever(
            FixedSegmenter(spans),
            embedder,
            config=QASCConfig(window_radius=0),
        )
        retriever.replace_document(Document("relevant", "txt"))

        embedder.document_vectors = ()
        with self.assertRaisesRegex(ComponentContractError, "returned 0 vectors"):
            retriever.replace_document(Document("relevant", "txt"))

        results = retriever.retrieve("question", top_k=1)
        self.assertEqual(retriever.document_count, 1)
        self.assertEqual(results[0].chunk.text, "relevant")

    def test_query_rejects_empty_text_nonpositive_limits_and_bad_vectors(self):
        """Invalid query inputs fail before scoring or returning misleading results."""

        spans = (sentence("guide", 0, "relevant"),)
        embedder = MutableEmbedder(((1.0, 0.0),))
        retriever = QASCRetriever(FixedSegmenter(spans), embedder)
        retriever.replace_document(Document("relevant", "txt"))

        with self.assertRaisesRegex(ValueError, "query cannot be empty"):
            retriever.retrieve("  ", top_k=1)
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            retriever.retrieve("question", top_k=0)
        for query in (None, 7, object()):
            with self.subTest(query=query):
                with self.assertRaisesRegex(TypeError, "query must be a string"):
                    retriever.retrieve(query, top_k=1)
        for top_k in (True, False, 1.0, "1"):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(TypeError, "top_k must be an integer"):
                    retriever.retrieve("question", top_k=top_k)

        embedder.query_vector = ()
        with self.assertRaisesRegex(
            ComponentContractError, "Query embedding cannot be empty"
        ):
            retriever.retrieve("question", top_k=1)

        embedder.query_vector = (1.0, 0.0, 0.0)
        with self.assertRaises(VectorDimensionError):
            retriever.retrieve("question", top_k=1)

    def test_many_document_replacements_keep_one_global_document_map(self):
        """QASC publishes each prepared document without copying all peers."""

        class Segmenter:
            def segment(self, document):
                document_id = document.metadata["document_id"]
                return (sentence(document_id, 0, "content"),)

        retriever = QASCRetriever(
            Segmenter(), MutableEmbedder(((1.0, 0.0),))
        )
        document_map = retriever._documents

        for index in range(200):
            retriever.replace_document(
                Document(
                    "content",
                    "txt",
                    {"document_id": "document-{}".format(index)},
                )
            )
        for index in range(0, 200, 2):
            self.assertEqual(
                retriever.delete_document("document-{}".format(index)), 1
            )

        self.assertIs(retriever._documents, document_map)
        self.assertEqual(retriever.document_count, 100)

    def test_deleting_the_last_document_allows_a_new_embedding_dimension(self):
        """A fully emptied optional index can follow an embedding-model migration."""

        spans = (sentence("guide", 0, "relevant"),)
        embedder = MutableEmbedder(((1.0, 0.0),))
        retriever = QASCRetriever(FixedSegmenter(spans), embedder)
        retriever.replace_document(Document("relevant", "txt"))

        self.assertEqual(retriever.delete_document("guide"), 1)
        embedder.document_vectors = ((1.0, 0.0, 0.0),)
        embedder.query_vector = (1.0, 0.0, 0.0)
        retriever.replace_document(Document("relevant", "txt"))

        self.assertEqual(
            retriever.retrieve("question", top_k=1)[0].chunk.text, "relevant"
        )


class QASCWindowMergePerformanceTests(unittest.TestCase):
    class CountingRetriever(QASCRetriever):
        def __init__(self, gap_tolerance=1):
            super().__init__(
                object(),
                object(),
                config=QASCConfig(gap_tolerance=gap_tolerance),
            )
            self.aggregate_calls = 0
            self.scored_positions = 0

        def _aggregate_score(self, start, end, center, similarities):
            del center, similarities
            self.aggregate_calls += 1
            self.scored_positions += end - start + 1
            return 1.0

    def test_seed_window_recurrence_matches_direct_aggregation(self):
        """Linear precomputation preserves fixed-radius decay-weighted scores."""

        similarities = (0.2, -0.4, 0.8, 0.1, 1.0, -0.2, 0.5)
        for radius, decay in ((0, 0.3), (1, 0.0), (3, 0.3), (20, 2.0)):
            with self.subTest(radius=radius, decay=decay):
                retriever = QASCRetriever(
                    FixedSegmenter(()),
                    MutableEmbedder(()),
                    config=QASCConfig(window_radius=radius, decay=decay),
                )
                actual = retriever._seed_window_scores(similarities)
                expected = tuple(
                    retriever._aggregate_score(
                        max(0, center - radius),
                        min(len(similarities) - 1, center + radius),
                        center,
                        similarities,
                    )
                    for center in range(len(similarities))
                )

                for observed, direct in zip(actual, expected):
                    self.assertAlmostEqual(observed, direct, places=12)

    def test_adjacent_candidates_score_only_the_final_merged_window(self):
        """Merge work grows with the final window instead of every prefix."""

        size = 1_000
        retriever = self.CountingRetriever()
        candidates = tuple(
            qasc_module._Window(index, index, (index,), 1.0)
            for index in range(size)
        )

        windows = retriever._merge_windows(candidates, (1.0,) * size)

        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start, windows[0].end), (0, size - 1))
        self.assertEqual(len(windows[0].seeds), size)
        self.assertEqual(retriever.aggregate_calls, 1)
        self.assertEqual(retriever.scored_positions, size)

    def test_large_radius_query_avoids_per_seed_window_rescoring(self):
        """All-seed, document-wide windows retain a linear aggregation bound."""

        size = 1_000
        text = "x " * size
        sentence_spans = tuple(
            SentenceSpan(
                "guide:{}".format(index),
                "guide",
                "x",
                index,
                index * 2,
                index * 2 + 1,
            )
            for index in range(size)
        )

        class Segmenter:
            def segment(self, document):
                del document
                return sentence_spans

        class Embedder:
            def embed_documents(self, texts):
                return ((1.0,),) * len(texts)

            def embed_query(self, query):
                del query
                return (1.0,)

        class MeasuredRetriever(QASCRetriever):
            aggregate_calls = 0
            scored_positions = 0

            def _aggregate_score(self, start, end, center, similarities):
                self.aggregate_calls += 1
                self.scored_positions += end - start + 1
                return super()._aggregate_score(start, end, center, similarities)

        retriever = MeasuredRetriever(
            Segmenter(),
            Embedder(),
            config=QASCConfig(
                seed_percentile=0,
                window_radius=size,
                gap_tolerance=0,
                chunk_threshold_factor=0,
            ),
        )
        retriever.replace_document(Document(text, "txt", {"document_id": "guide"}))

        results = retriever.retrieve("question", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(retriever.aggregate_calls, 1)
        self.assertEqual(retriever.scored_positions, size)

    def test_separate_groups_preserve_single_scores_and_score_merged_groups_once(self):
        """Optimization preserves group boundaries and existing singleton scores."""

        retriever = self.CountingRetriever(gap_tolerance=1)
        candidates = (
            qasc_module._Window(0, 0, (0,), 0.2),
            qasc_module._Window(1, 1, (1,), 0.3),
            qasc_module._Window(10, 10, (10,), 0.42),
            qasc_module._Window(20, 20, (20,), 0.5),
            qasc_module._Window(21, 21, (21,), 0.6),
            qasc_module._Window(22, 22, (22,), 0.7),
        )

        windows = retriever._merge_windows(candidates, (1.0,) * 23)

        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(0, 1), (10, 10), (20, 22)],
        )
        self.assertEqual(windows[1].score, 0.42)
        self.assertEqual(retriever.aggregate_calls, 2)
        self.assertEqual(retriever.scored_positions, 5)


if __name__ == "__main__":
    unittest.main()
