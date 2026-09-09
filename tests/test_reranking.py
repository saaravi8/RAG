import math
import unittest

from modular_rag import (
    Chunk,
    ComponentContractError,
    CrossEncoderReranker,
    SearchResult,
    VectorRecord,
    build_demo_rag,
)
from modular_rag.relevance import ScoreThresholdRelevancePolicy


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def result(chunk_id, text, score):
    return SearchResult(Chunk(chunk_id, "guide", text, 0), score)


class OverflowingFloat:
    def __float__(self):
        raise OverflowError("too large")


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_scores_query_chunk_pairs_and_returns_the_best_results(self):
        model = FakeCrossEncoder([0.2, 0.9, -0.1])
        reranker = CrossEncoderReranker(model=model, batch_size=8)
        candidates = (
            result("first", "first passage", 0.95),
            result("second", "second passage", 0.80),
            result("third", "third passage", 0.70),
        )

        ranked = reranker.rerank("question", candidates, top_k=2)

        self.assertEqual([item.chunk.id for item in ranked], ["second", "first"])
        self.assertEqual([item.score for item in ranked], [0.9, 0.2])
        self.assertEqual(
            model.calls[0][0],
            [
                ("question", "first passage"),
                ("question", "second passage"),
                ("question", "third passage"),
            ],
        )
        self.assertEqual(model.calls[0][1]["batch_size"], 8)
        self.assertTrue(model.calls[0][1]["convert_to_numpy"])

    def test_equal_scores_preserve_retrieval_order(self):
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.5, 0.5]))
        candidates = (
            result("z", "first", 0.9),
            result("a", "second", 0.8),
        )

        ranked = reranker.rerank("question", candidates, top_k=2)

        self.assertEqual([item.chunk.id for item in ranked], ["z", "a"])

    def test_empty_candidates_skip_model_inference(self):
        model = FakeCrossEncoder([])
        reranker = CrossEncoderReranker(model=model)

        self.assertEqual(reranker.rerank("question", (), top_k=3), ())
        self.assertEqual(model.calls, [])

    def test_rejects_invalid_configuration_and_limits(self):
        model = FakeCrossEncoder([])

        with self.assertRaisesRegex(ValueError, "batch_size"):
            CrossEncoderReranker(model=model, batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_length"):
            CrossEncoderReranker(model=model, max_length=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            CrossEncoderReranker(model=model).rerank("question", (), top_k=0)
        with self.assertRaisesRegex(TypeError, "predict"):
            CrossEncoderReranker(model=object())

        invalid_integer_options = (
            {"batch_size": True},
            {"batch_size": 2.0},
            {"max_length": False},
            {"max_length": 128.0},
        )
        for options in invalid_integer_options:
            with self.subTest(options=options):
                with self.assertRaises(TypeError):
                    CrossEncoderReranker(model=model, **options)

        reranker = CrossEncoderReranker(model=model)
        for top_k in (True, 1.0):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(TypeError, "top_k"):
                    reranker.rerank("question", (), top_k=top_k)

    def test_rejects_malformed_model_scores(self):
        candidate = (result("first", "first passage", 0.9),)
        invalid_outputs = (
            ([], "0 scores for 1 candidates"),
            ([[0.2]], "malformed relevance scores"),
            ([True], "malformed relevance scores"),
            (["0.2"], "malformed relevance scores"),
            ([b"0.2"], "malformed relevance scores"),
            ([OverflowingFloat()], "malformed relevance scores"),
            ([math.nan], "non-finite relevance scores"),
        )

        for scores, message in invalid_outputs:
            with self.subTest(scores=scores):
                reranker = CrossEncoderReranker(model=FakeCrossEncoder(scores))
                with self.assertRaisesRegex(ComponentContractError, message):
                    reranker.rerank("question", candidate, top_k=1)

    def test_factory_accepts_an_injected_reranker(self):
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([]))

        app = build_demo_rag(reranker=reranker)

        self.assertIs(app.rag.reranker, reranker)

    def test_factory_supports_negative_logits_with_a_calibrated_policy(self):
        """Factory wiring preserves an explicitly accepted retrieval score domain."""

        class FixedEmbedder:
            def embed_documents(self, texts):
                return tuple((-1.0, 0.0) for _ in texts)

            def embed_query(self, text):
                del text
                return (1.0, 0.0)

        reranker = CrossEncoderReranker(model=FakeCrossEncoder([-0.4]))
        application = build_demo_rag(
            embedder=FixedEmbedder(),
            reranker=reranker,
            relevance_policy=ScoreThresholdRelevancePolicy(-1.1),
        )
        application.store.replace_document(
            "guide",
            (
                VectorRecord(
                    Chunk("c", "guide", "calibrated evidence", 0),
                    (-1.0, 0.0),
                ),
            ),
        )

        response = application.ask("question", top_k=1)

        self.assertFalse(response.abstained)
        self.assertEqual(response.results[0].score, -0.4)
        self.assertEqual(response.citations[0].score, -0.4)


if __name__ == "__main__":
    unittest.main()
