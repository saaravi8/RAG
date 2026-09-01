import unittest

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
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    QASCConfig(**parameters)

        self.assertEqual(QASCConfig(seed_percentile=0).seed_percentile, 0)
        self.assertEqual(QASCConfig(seed_percentile=100).seed_percentile, 100)


class QASCRetrieverContractTests(unittest.TestCase):
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
                    retriever.replace_document(Document("text", "txt"))

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
            retriever.replace_document(Document("broken replacement", "txt"))

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

        embedder.query_vector = ()
        with self.assertRaisesRegex(
            ComponentContractError, "Query embedding cannot be empty"
        ):
            retriever.retrieve("question", top_k=1)

        embedder.query_vector = (1.0, 0.0, 0.0)
        with self.assertRaises(VectorDimensionError):
            retriever.retrieve("question", top_k=1)

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


if __name__ == "__main__":
    unittest.main()
