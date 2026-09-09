import unittest

from modular_rag import (
    Chunk,
    ComponentContractError,
    RAGResponse,
    RAGService,
    SearchResult,
    build_demo_rag,
)
from modular_rag.relevance import ScoreThresholdRelevancePolicy


class FixedRetriever:
    def __init__(self, results):
        self.results = tuple(results)

    def retrieve(self, query, *, top_k, filters=None):
        del query, filters
        return self.results[:top_k]


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, contexts):
        self.calls.append((question, tuple(contexts)))
        return "generated answer"


class RecordingReranker:
    def __init__(self, final_score):
        self.final_score = final_score
        self.calls = []

    def rerank(self, query, results, *, top_k):
        self.calls.append((query, tuple(results), top_k))
        return tuple(
            SearchResult(item.chunk, self.final_score) for item in results[:top_k]
        )


def result(chunk_id, score):
    return SearchResult(
        Chunk(chunk_id, "guide", "evidence from {}".format(chunk_id), 0),
        score,
    )


class RAGServiceAbstentionTests(unittest.TestCase):
    def test_empty_retrieval_abstains_before_generation(self):
        """No retrieved evidence must return a service-owned abstention."""

        generator = RecordingGenerator()
        service = RAGService(FixedRetriever(()), generator)

        response = service.ask("What has been indexed?")

        self.assertEqual(response.answer, RAGService.ABSTENTION_ANSWER)
        self.assertEqual(response.results, ())
        self.assertEqual(response.citations, ())
        self.assertTrue(response.abstained)
        self.assertEqual(response.abstention_reason, "no_relevant_evidence")
        self.assertEqual(generator.calls, [])

    def test_all_zero_scores_at_default_boundary_abstain_before_generation(self):
        """The default floor is exclusive, so neutral scores are not evidence."""

        generator = RecordingGenerator()
        reranker = RecordingReranker(-0.1)
        service = RAGService(
            FixedRetriever((result("first", 0.0), result("second", 0.0))),
            generator,
            reranker=reranker,
        )

        response = service.ask("What is not covered by these documents?")

        self.assertEqual(response.answer, RAGService.ABSTENTION_ANSWER)
        self.assertEqual(response.results, ())
        self.assertEqual(response.citations, ())
        self.assertTrue(response.abstained)
        self.assertEqual(reranker.calls, [])
        self.assertEqual(generator.calls, [])

    def test_only_relevant_results_reach_generation_and_citations(self):
        """The safety gate removes nonpositive evidence without hiding a good hit."""

        generator = RecordingGenerator()
        relevant = result("relevant", 0.8)
        service = RAGService(
            FixedRetriever(
                (relevant, result("neutral", 0.0), result("negative", -0.2))
            ),
            generator,
        )

        response = service.ask("What does the guide cover?")

        self.assertEqual(response.answer, "generated answer")
        self.assertFalse(response.abstained)
        self.assertIsNone(response.abstention_reason)
        self.assertEqual(generator.calls, [(response.question, (relevant,))])
        self.assertEqual(response.results, (relevant,))
        self.assertEqual(
            [citation.chunk_id for citation in response.citations], ["relevant"]
        )
        self.assertEqual(response.citations[0].number, 1)

    def test_negative_reranker_score_does_not_reuse_retrieval_threshold(self):
        """A relevant retrieval hit remains usable in a different score domain."""

        generator = RecordingGenerator()
        reranker = RecordingReranker(-0.1)
        retrieved = result("relevant", 0.8)
        service = RAGService(
            FixedRetriever((retrieved,)),
            generator,
            reranker=reranker,
            default_top_k=1,
            default_fetch_k=3,
        )

        response = service.ask("What does the guide cover?", top_k=1)

        self.assertEqual(reranker.calls, [(response.question, (retrieved,), 1)])
        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(response.results[0].score, -0.1)
        self.assertEqual(response.citations[0].score, -0.1)
        self.assertEqual(generator.calls, [(response.question, response.results)])

    def test_method_specific_threshold_supports_backend_calibration(self):
        """Different query methods can opt into score ranges calibrated for them."""

        generator = RecordingGenerator()
        negative_but_accepted = result("qasc-result", -0.25)
        service = RAGService(
            FixedRetriever((negative_but_accepted,)),
            generator,
            relevance_policy=ScoreThresholdRelevancePolicy(
                method_minimum_scores={"QASC": -0.5}
            ),
            query_methods={"qasc": FixedRetriever((negative_but_accepted,))},
        )

        standard = service.ask("question")
        qasc = service.ask("question", method="QASC")

        self.assertEqual(standard.answer, RAGService.ABSTENTION_ANSWER)
        self.assertEqual(qasc.results, (negative_but_accepted,))
        self.assertEqual(qasc.answer, "generated answer")
        self.assertEqual(generator.calls, [("question", (negative_but_accepted,))])

    def test_invalid_policy_thresholds_fail_during_configuration(self):
        """Non-numeric, boolean, and non-finite floors cannot define relevance."""

        invalid_thresholds = (
            (True, TypeError),
            ("not-a-score", TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
        )
        for threshold, error in invalid_thresholds:
            with self.subTest(threshold=threshold):
                with self.assertRaises(error):
                    ScoreThresholdRelevancePolicy(threshold)

    def test_factory_exposes_the_relevance_policy(self):
        """Applications can calibrate evidence without rebuilding the service."""

        policy = ScoreThresholdRelevancePolicy(0.25)

        application = build_demo_rag(relevance_policy=policy)

        self.assertIs(application.rag.relevance_policy, policy)

    def test_policy_cannot_invent_or_duplicate_evidence(self):
        """An injected policy may filter candidates but cannot create evidence."""

        candidate = result("candidate", 0.8)

        class InvalidPolicy:
            def __init__(self, selected):
                self.selected = selected

            def select_relevant(self, question, results, *, method):
                del question, results, method
                return self.selected

        invalid_selections = (
            (result("invented", 0.9),),
            (candidate, candidate),
            (object(),),
        )
        for selected in invalid_selections:
            with self.subTest(selected=selected):
                service = RAGService(
                    FixedRetriever((candidate,)),
                    RecordingGenerator(),
                    relevance_policy=InvalidPolicy(selected),
                )
                with self.assertRaises(ComponentContractError):
                    service.ask("question")

    def test_response_requires_consistent_structured_abstention_state(self):
        """Callers can trust the flag and reason without parsing answer text."""

        with self.assertRaisesRegex(TypeError, "requires a string reason"):
            RAGResponse("question", "answer", (), (), abstained=True)
        with self.assertRaisesRegex(ValueError, "cannot have"):
            RAGResponse(
                "question",
                "answer",
                (),
                (),
                abstention_reason="unexpected",
            )

    def test_reranker_cannot_invent_duplicate_or_excess_evidence(self):
        """Ranking may rescore and reorder candidates but cannot expand them."""

        candidate = result("candidate", 0.8)

        class InvalidReranker:
            def __init__(self, selected):
                self.selected = selected

            def rerank(self, query, results, *, top_k):
                del query, results, top_k
                return self.selected

        invalid_selections = (
            ((result("invented", 0.9),), 1),
            ((candidate, candidate), 2),
            ((candidate, candidate, candidate), 2),
        )
        for selected, top_k in invalid_selections:
            with self.subTest(selected=selected, top_k=top_k):
                service = RAGService(
                    FixedRetriever((candidate,)),
                    RecordingGenerator(),
                    reranker=InvalidReranker(selected),
                    default_top_k=top_k,
                    default_fetch_k=top_k,
                )
                with self.assertRaises(ComponentContractError):
                    service.ask("question")

    def test_retriever_cannot_supply_duplicate_chunk_evidence(self):
        """Duplicate source identities cannot inflate generation or citations."""

        duplicate = result("duplicate", 0.8)
        service = RAGService(
            FixedRetriever((duplicate, duplicate)),
            RecordingGenerator(),
        )

        with self.assertRaisesRegex(ComponentContractError, "duplicate"):
            service.ask("question", top_k=2)


if __name__ == "__main__":
    unittest.main()
