import json
import math
import re
import unittest
from collections.abc import Sequence
from pathlib import Path

from modular_rag import (
    AnswerVerificationPolicy,
    Chunk,
    RAGService,
    SearchResult,
    VerificationResult,
)
from modular_rag.evaluation import (
    EvaluationCase,
    EvaluationObservation,
    evaluate_observations,
)


class TokenOverlapRetriever:
    """Deterministic test backend; it is not a semantic quality claim."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, documents):
        self.documents = tuple(documents)

    def retrieve(self, query, *, top_k, filters=None):
        del filters
        query_terms = set(self._TOKEN.findall(query.lower()))
        results = []
        for document in self.documents:
            terms = set(self._TOKEN.findall(document["text"].lower()))
            score = float(len(query_terms & terms))
            results.append(
                SearchResult(
                    Chunk(
                        document["document_id"] + ":0",
                        document["document_id"],
                        document["text"],
                        0,
                    ),
                    score,
                )
            )
        return tuple(
            sorted(results, key=lambda result: (-result.score, result.chunk.id))[
                :top_k
            ]
        )


class ExtractiveFixtureGenerator:
    def generate(self, question, contexts):
        del question
        return contexts[0].chunk.text


class UnsupportedFixtureGenerator:
    def generate(self, question, contexts):
        del question, contexts
        return "This generated statement is absent from the fixture evidence."


class ExactContextVerifier:
    def verify(self, question, answer, contexts):
        del question
        supported = any(answer == result.chunk.text for result in contexts)
        return VerificationResult(supported, "deterministic exact-context check")


class EvaluationHarnessTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        path = Path(__file__).parent / "fixtures" / "rag_evaluation.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_unique_nonempty_strings(self, values, *, label):
        self.assertGreater(len(values), 0, "{} cannot be empty".format(label))
        for index, value in enumerate(values):
            with self.subTest(label=label, index=index):
                self.assertIsInstance(value, str)
                self.assertTrue(value.strip())
        self.assertEqual(
            len(values),
            len(set(values)),
            "{} must be unique".format(label),
        )

    @staticmethod
    def _metric_isolation_inputs():
        cases = (
            EvaluationCase("retrieval", ("d1", "d2")),
            EvaluationCase("false-abstention", ("d3",)),
            EvaluationCase("unknown-a", ()),
            EvaluationCase("unknown-b", ()),
        )
        observations = (
            EvaluationObservation("retrieval", ("d1", "d2"), False, True),
            EvaluationObservation("false-abstention", ("d3",), True),
            EvaluationObservation("unknown-a", (), True),
            EvaluationObservation("unknown-b", (), True),
        )
        return cases, observations

    def test_fixture_has_unique_nonempty_identity_and_content_fields(self):
        """Corpus identities and user-facing text remain explicit and unambiguous."""

        fixture = self._fixture()
        documents = fixture["documents"]
        cases = fixture["cases"]

        self._assert_unique_nonempty_strings(
            [item["document_id"] for item in documents],
            label="document IDs",
        )
        self._assert_unique_nonempty_strings(
            [item["text"] for item in documents],
            label="document texts",
        )
        self._assert_unique_nonempty_strings(
            [item["case_id"] for item in cases],
            label="case IDs",
        )
        self._assert_unique_nonempty_strings(
            [item["question"] for item in cases],
            label="case questions",
        )

    def test_fixture_relevant_ids_reference_declared_documents(self):
        """Every relevance judgment points to a document present in the corpus."""

        fixture = self._fixture()
        document_ids = {
            item["document_id"] for item in fixture["documents"]
        }

        for item in fixture["cases"]:
            with self.subTest(case_id=item["case_id"]):
                relevant_ids = item["relevant_document_ids"]
                self.assertIsInstance(relevant_ids, list)
                EvaluationCase(item["case_id"], relevant_ids)
                self.assertLessEqual(set(relevant_ids), document_ids)

    def test_bilingual_document_code_and_no_answer_smoke_corpus(self):
        """Required CI measures deterministic coverage across core use cases."""

        fixture = self._fixture()
        service = RAGService(
            TokenOverlapRetriever(fixture["documents"]),
            ExtractiveFixtureGenerator(),
            answer_verifier=ExactContextVerifier(),
            verification_policy=AnswerVerificationPolicy("report"),
            default_top_k=3,
        )
        cases = tuple(
            EvaluationCase(
                item["case_id"], tuple(item["relevant_document_ids"])
            )
            for item in fixture["cases"]
        )

        def evaluate_once():
            observations = []
            for item, case in zip(fixture["cases"], cases):
                response = service.ask(item["question"], top_k=3)
                retrieved_ids = tuple(
                    result.chunk.document_id for result in response.results
                )
                observations.append(
                    EvaluationObservation(
                        case.case_id,
                        retrieved_ids,
                        response.abstained,
                        answer_supported=(
                            None
                            if response.abstained
                            else response.verification.supported
                        ),
                    )
                )
            return evaluate_observations(cases, observations, top_k=3)

        first = evaluate_once()
        second = evaluate_once()

        self.assertEqual(first, second)
        self.assertEqual((first.case_count, first.answerable_count), (7, 5))
        self.assertEqual(first.unanswerable_count, 2)
        self.assertEqual(first.recall_at_k, 1.0)
        self.assertEqual(first.mean_reciprocal_rank, 1.0)
        self.assertEqual(first.ndcg_at_k, 1.0)
        self.assertEqual(first.abstention_precision, 1.0)
        self.assertEqual(first.abstention_recall, 1.0)
        self.assertEqual(first.unsupported_answer_rate, 0.0)
        self.assertEqual((first.answered_count, first.checked_answer_count), (5, 5))

    def test_unsupported_generation_is_measured_end_to_end(self):
        """A retrieved answer can be relevant while its generated text is unsupported."""

        fixture = self._fixture()
        item = fixture["cases"][0]
        case = EvaluationCase(item["case_id"], item["relevant_document_ids"])
        service = RAGService(
            TokenOverlapRetriever(fixture["documents"]),
            UnsupportedFixtureGenerator(),
            answer_verifier=ExactContextVerifier(),
            verification_policy=AnswerVerificationPolicy("report"),
            default_top_k=3,
        )

        response = service.ask(item["question"], top_k=3)
        observation = EvaluationObservation(
            case.case_id,
            tuple(result.chunk.document_id for result in response.results),
            response.abstained,
            answer_supported=response.verification.supported,
        )
        report = evaluate_observations((case,), (observation,), top_k=3)

        self.assertFalse(response.abstained)
        self.assertFalse(response.verification.supported)
        self.assertEqual(report.recall_at_k, 1.0)
        self.assertEqual(report.unsupported_answer_rate, 1.0)
        self.assertEqual((report.answered_count, report.checked_answer_count), (1, 1))

    def test_metrics_expose_missed_relevance_and_unsupported_answers(self):
        """An imperfect result set changes each metric in a predictable way."""

        cases = (
            EvaluationCase("multi", ("d1", "d2")),
            EvaluationCase("missed", ("d3",)),
            EvaluationCase("unknown", ()),
        )
        observations = (
            EvaluationObservation(
                "multi", ("noise", "d1", "d2"), False, True
            ),
            EvaluationObservation("missed", ("noise",), False, False),
            EvaluationObservation("unknown", (), True, None),
        )

        report = evaluate_observations(cases, observations, top_k=3)

        ideal = 1.0 + 1.0 / math.log2(3)
        observed = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertEqual(report.recall_at_k, 0.5)
        self.assertEqual(report.mean_reciprocal_rank, 0.25)
        self.assertAlmostEqual(report.ndcg_at_k, (observed / ideal) / 2)
        self.assertEqual(report.abstention_precision, 1.0)
        self.assertEqual(report.abstention_recall, 1.0)
        self.assertEqual(report.unsupported_answer_rate, 0.5)

    def test_retrieval_perturbation_does_not_change_safety_metrics(self):
        """Changing only ranked evidence affects retrieval metrics, not safety."""

        cases, observations = self._metric_isolation_inputs()
        baseline = evaluate_observations(cases, observations, top_k=2)
        perturbed = evaluate_observations(
            cases,
            (
                EvaluationObservation(
                    "retrieval", ("noise", "d1"), False, True
                ),
                *observations[1:],
            ),
            top_k=2,
        )

        self.assertLess(perturbed.recall_at_k, baseline.recall_at_k)
        self.assertLess(
            perturbed.mean_reciprocal_rank, baseline.mean_reciprocal_rank
        )
        self.assertLess(perturbed.ndcg_at_k, baseline.ndcg_at_k)
        self.assertEqual(
            (
                perturbed.abstention_precision,
                perturbed.abstention_recall,
                perturbed.unsupported_answer_rate,
            ),
            (
                baseline.abstention_precision,
                baseline.abstention_recall,
                baseline.unsupported_answer_rate,
            ),
        )

    def test_abstention_perturbation_does_not_change_retrieval_or_support_metrics(self):
        """Changing one no-answer outcome affects abstention metrics only."""

        cases, observations = self._metric_isolation_inputs()
        baseline = evaluate_observations(cases, observations, top_k=2)
        perturbed = evaluate_observations(
            cases,
            (
                *observations[:3],
                EvaluationObservation("unknown-b", (), False, True),
            ),
            top_k=2,
        )

        self.assertNotEqual(
            perturbed.abstention_precision, baseline.abstention_precision
        )
        self.assertNotEqual(perturbed.abstention_recall, baseline.abstention_recall)
        self.assertEqual(
            (
                perturbed.recall_at_k,
                perturbed.mean_reciprocal_rank,
                perturbed.ndcg_at_k,
                perturbed.unsupported_answer_rate,
            ),
            (
                baseline.recall_at_k,
                baseline.mean_reciprocal_rank,
                baseline.ndcg_at_k,
                baseline.unsupported_answer_rate,
            ),
        )

    def test_support_perturbation_does_not_change_retrieval_or_abstention_metrics(self):
        """Changing one verifier verdict affects unsupported-answer rate only."""

        cases, observations = self._metric_isolation_inputs()
        baseline = evaluate_observations(cases, observations, top_k=2)
        perturbed = evaluate_observations(
            cases,
            (
                EvaluationObservation("retrieval", ("d1", "d2"), False, False),
                *observations[1:],
            ),
            top_k=2,
        )

        self.assertGreater(
            perturbed.unsupported_answer_rate, baseline.unsupported_answer_rate
        )
        self.assertEqual(
            (
                perturbed.recall_at_k,
                perturbed.mean_reciprocal_rank,
                perturbed.ndcg_at_k,
                perturbed.abstention_precision,
                perturbed.abstention_recall,
            ),
            (
                baseline.recall_at_k,
                baseline.mean_reciprocal_rank,
                baseline.ndcg_at_k,
                baseline.abstention_precision,
                baseline.abstention_recall,
            ),
        )

    def test_document_id_collections_require_non_text_sequences(self):
        """Mappings, sets, generators, and text-like inputs cannot imply rank."""

        invalid_factories = (
            ("string", lambda: "doc"),
            ("bytes", lambda: b"doc"),
            ("bytearray", lambda: bytearray(b"doc")),
            ("mapping", lambda: {"doc": True}),
            ("set", lambda: {"doc"}),
            ("generator", lambda: (value for value in ("doc",))),
        )
        constructors = (
            ("case", lambda values: EvaluationCase("case", values)),
            (
                "observation",
                lambda values: EvaluationObservation(
                    "case", values, False, True
                ),
            ),
        )

        for constructor_name, constructor in constructors:
            for value_name, value_factory in invalid_factories:
                with self.subTest(
                    constructor=constructor_name, value=value_name
                ):
                    with self.assertRaisesRegex(TypeError, "sequence"):
                        constructor(value_factory())

    def test_document_id_sequences_accept_lists_and_tuples_and_snapshot_lists(self):
        """Ordered mutable inputs are accepted but cannot mutate frozen records."""

        relevant_ids = ["relevant"]
        retrieved_ids = ["retrieved"]
        case = EvaluationCase("case", relevant_ids)
        observation = EvaluationObservation(
            "case", retrieved_ids, False, True
        )
        tuple_case = EvaluationCase("tuple-case", ("relevant",))
        tuple_observation = EvaluationObservation(
            "tuple-case", ("retrieved",), False, True
        )

        relevant_ids.append("later")
        retrieved_ids.append("later")

        self.assertEqual(case.relevant_document_ids, ("relevant",))
        self.assertEqual(observation.retrieved_document_ids, ("retrieved",))
        self.assertEqual(tuple_case.relevant_document_ids, ("relevant",))
        self.assertEqual(tuple_observation.retrieved_document_ids, ("retrieved",))

    def test_evaluation_inputs_require_non_text_sequences(self):
        """Unordered, lazy, and text-like inputs cannot define case evaluation order."""

        case = EvaluationCase("case", ("document",))
        observation = EvaluationObservation(
            "case", ("document",), False, True
        )
        invalid_factories = (
            ("set", lambda item: {item}),
            ("mapping", lambda item: {item: True}),
            ("generator", lambda item: (value for value in (item,))),
            ("string", lambda item: item.case_id),
            ("bytes", lambda item: item.case_id.encode("utf-8")),
            ("bytearray", lambda item: bytearray(item.case_id, "utf-8")),
        )

        for input_name, valid_other, invalid_item in (
            ("cases", (observation,), case),
            ("observations", (case,), observation),
        ):
            for value_name, value_factory in invalid_factories:
                with self.subTest(input=input_name, value=value_name):
                    invalid = value_factory(invalid_item)
                    arguments = {
                        "cases": invalid if input_name == "cases" else valid_other,
                        "observations": (
                            invalid
                            if input_name == "observations"
                            else valid_other
                        ),
                    }
                    with self.assertRaisesRegex(
                        TypeError,
                        r"{} must be a non-text sequence".format(input_name),
                    ):
                        evaluate_observations(top_k=1, **arguments)

    def test_evaluation_accepts_ordered_sequences_and_preserves_rank(self):
        """A list/tuple pair keeps the rank-two hit used by MRR and nDCG."""

        cases = [EvaluationCase("case", ("relevant",))]
        observations = [
            EvaluationObservation(
                "case", ("noise", "relevant"), False, True
            )
        ]

        report = evaluate_observations(cases, tuple(observations), top_k=2)

        self.assertEqual(report.recall_at_k, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 0.5)
        self.assertAlmostEqual(report.ndcg_at_k, 1.0 / math.log2(3))

    def test_evaluation_does_not_mask_sequence_iteration_type_errors(self):
        """A Sequence implementation's own TypeError remains diagnosable to callers."""

        class BrokenSequence(Sequence):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise TypeError("fixture sequence failure")

        with self.assertRaisesRegex(TypeError, "fixture sequence failure"):
            evaluate_observations(BrokenSequence(), (), top_k=1)

    def test_metric_inputs_reject_ambiguous_case_sets_and_limits(self):
        """Bad fixtures fail rather than producing plausible but invalid scores."""

        case = EvaluationCase("case", ("doc",))
        observation = EvaluationObservation("case", ("doc",), False, True)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_observations((case,), (observation,), top_k=0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_observations((case, case), (observation,), top_k=1)
        with self.assertRaisesRegex(ValueError, "matching IDs"):
            evaluate_observations(
                (case,),
                (EvaluationObservation("different", (), True),),
                top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_observations((), (), top_k=1)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            EvaluationObservation("case", ("doc", "doc"), False, True)
        with self.assertRaisesRegex(ValueError, "requires"):
            EvaluationObservation("case", ("doc",), False, None)
        with self.assertRaisesRegex(ValueError, "cannot label"):
            EvaluationObservation("case", (), True, True)


if __name__ == "__main__":
    unittest.main()
