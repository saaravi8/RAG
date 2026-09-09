import unittest

from modular_rag import (
    AnswerVerificationPolicy,
    Chunk,
    ComponentContractError,
    RAGService,
    SearchResult,
    VerificationResult,
    build_demo_rag,
)


class FixedRetriever:
    def __init__(self, events):
        self.events = events
        self.result = SearchResult(
            Chunk("chunk", "guide", "supported evidence", 0), 0.8
        )

    def retrieve(self, query, *, top_k, filters=None):
        del query, top_k, filters
        self.events.append("retrieve")
        return (self.result,)


class RecordingGenerator:
    def __init__(self, events):
        self.events = events

    def generate(self, question, contexts):
        del question, contexts
        self.events.append("generate")
        return "generated answer"


class RecordingVerifier:
    def __init__(self, events, result):
        self.events = events
        self.result = result
        self.calls = []

    def verify(self, question, answer, contexts):
        self.events.append("verify")
        self.calls.append((question, answer, tuple(contexts)))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class AnswerVerificationIntegrationTests(unittest.TestCase):
    @staticmethod
    def _service(policy, verifier_result):
        events = []
        retriever = FixedRetriever(events)
        verifier = RecordingVerifier(events, verifier_result)
        service = RAGService(
            retriever,
            RecordingGenerator(events),
            answer_verifier=verifier,
            verification_policy=policy,
        )
        return service, retriever, verifier, events

    def test_disabled_verification_preserves_existing_behavior(self):
        """The default policy adds no verifier call or response-state change."""

        service, _, verifier, events = self._service(
            AnswerVerificationPolicy(),
            VerificationResult(True),
        )

        response = service.ask("question")

        self.assertEqual(events, ["retrieve", "generate"])
        self.assertEqual(verifier.calls, [])
        self.assertIsNone(response.verification)
        self.assertFalse(response.abstained)

    def test_report_mode_attaches_supported_or_unsupported_verdict(self):
        """Report mode exposes the verdict without replacing generated output."""

        for supported in (True, False):
            with self.subTest(supported=supported):
                verdict = VerificationResult(supported, "checked")
                service, retriever, verifier, events = self._service(
                    AnswerVerificationPolicy("report"), verdict
                )

                response = service.ask("question")

                self.assertEqual(events, ["retrieve", "generate", "verify"])
                self.assertEqual(response.answer, "generated answer")
                self.assertEqual(response.verification, verdict)
                self.assertFalse(response.abstained)
                self.assertEqual(
                    verifier.calls,
                    [("question", "generated answer", (retriever.result,))],
                )

    def test_enforce_mode_turns_unsupported_output_into_safe_abstention(self):
        """Unsupported generated text is never returned as an accepted answer."""

        verdict = VerificationResult(False, "missing support")
        service, retriever, _, _ = self._service(
            AnswerVerificationPolicy("enforce"), verdict
        )

        response = service.ask("question")

        self.assertTrue(response.abstained)
        self.assertEqual(response.abstention_reason, "unsupported_answer")
        self.assertEqual(response.answer, RAGService.UNSUPPORTED_ANSWER)
        self.assertEqual(response.verification, verdict)
        self.assertEqual(response.results, (retriever.result,))
        self.assertEqual(response.citations[0].chunk_id, "chunk")

    def test_enforce_mode_returns_a_supported_verified_answer(self):
        """Enforcement accepts a positive verdict without changing the answer."""

        verdict = VerificationResult(True, "supported")
        service, _, _, _ = self._service(
            AnswerVerificationPolicy("enforce"), verdict
        )

        response = service.ask("question")

        self.assertFalse(response.abstained)
        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(response.verification, verdict)

    def test_pre_generation_abstention_skips_the_verifier(self):
        """Answer verification never runs when the evidence gate stops generation."""

        events = []

        class EmptyRetriever:
            def retrieve(self, query, *, top_k, filters=None):
                del query, top_k, filters
                events.append("retrieve")
                return ()

        verifier = RecordingVerifier(events, VerificationResult(True))
        service = RAGService(
            EmptyRetriever(),
            RecordingGenerator(events),
            answer_verifier=verifier,
            verification_policy=AnswerVerificationPolicy("enforce"),
        )

        response = service.ask("question")

        self.assertTrue(response.abstained)
        self.assertEqual(events, ["retrieve"])
        self.assertEqual(verifier.calls, [])

    def test_verifier_error_policy_is_explicit(self):
        """Verifier outages either propagate or fail closed as configured."""

        service, _, _, _ = self._service(
            AnswerVerificationPolicy("enforce", "raise"),
            RuntimeError("verifier unavailable"),
        )
        with self.assertRaisesRegex(RuntimeError, "verifier unavailable"):
            service.ask("question")

        service, _, _, _ = self._service(
            AnswerVerificationPolicy("report", "abstain"),
            RuntimeError("verifier unavailable"),
        )
        response = service.ask("question")
        self.assertTrue(response.abstained)
        self.assertEqual(response.abstention_reason, "verification_error")
        self.assertEqual(response.answer, RAGService.UNSUPPORTED_ANSWER)
        self.assertEqual(
            response.verification.metadata["status"], "verification_error"
        )

    def test_verifier_metadata_cannot_trigger_internal_error_control_flow(self):
        """Adapter diagnostics remain data even when they resemble service metadata."""

        verdict = VerificationResult(
            True,
            "supported",
            {"status": "verification_error"},
        )
        service, _, _, _ = self._service(
            AnswerVerificationPolicy("report", "abstain"), verdict
        )

        response = service.ask("question")

        self.assertFalse(response.abstained)
        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(response.verification, verdict)

    def test_untrusted_components_cannot_spoof_returned_evidence(self):
        """Generator/verifier mutations are isolated from citations and response data."""

        source_metadata = {
            "source_name": "trusted.txt",
            "nested": {"labels": ["trusted"]},
        }
        source_result = SearchResult(
            Chunk("trusted", "guide", "supported evidence", 0, source_metadata),
            0.8,
        )
        verdict = VerificationResult(
            True,
            "supported",
            {"provider": {"labels": ["trusted"]}},
        )

        class Retriever:
            def retrieve(self, query, *, top_k, filters=None):
                del query, top_k, filters
                return (source_result,)

        class SpoofingGenerator:
            def generate(self, question, contexts):
                del question
                object.__setattr__(contexts[0], "score", -99.0)
                object.__setattr__(contexts[0].chunk, "id", "generator-spoof")
                object.__setattr__(
                    contexts[0].chunk,
                    "metadata",
                    {"source_name": "generator-spoof.txt"},
                )
                return "generated answer"

        class SpoofingVerifier:
            def __init__(self):
                self.observed = None

            def verify(self, question, answer, contexts):
                del question, answer
                self.observed = (
                    contexts[0].chunk.id,
                    contexts[0].chunk.metadata["source_name"],
                    contexts[0].score,
                )
                object.__setattr__(contexts[0], "score", -100.0)
                object.__setattr__(contexts[0].chunk, "id", "verifier-spoof")
                return verdict

        verifier = SpoofingVerifier()
        response = RAGService(
            Retriever(),
            SpoofingGenerator(),
            answer_verifier=verifier,
            verification_policy=AnswerVerificationPolicy("report"),
        ).ask("question")

        self.assertEqual(verifier.observed, ("trusted", "trusted.txt", 0.8))
        self.assertEqual(response.results[0].chunk.id, "trusted")
        self.assertEqual(response.results[0].score, 0.8)
        self.assertEqual(response.citations[0].chunk_id, "trusted")
        self.assertEqual(response.citations[0].source, "trusted.txt")
        self.assertEqual(
            response.results[0].chunk.metadata["nested"]["labels"],
            ("trusted",),
        )

        object.__setattr__(source_result.chunk, "id", "late-source-spoof")
        object.__setattr__(verdict, "reason", "late-verdict-spoof")
        source_metadata["nested"]["labels"].append("late-spoof")

        self.assertEqual(response.results[0].chunk.id, "trusted")
        self.assertEqual(response.verification.reason, "supported")
        self.assertEqual(
            response.verification.metadata["provider"]["labels"],
            ("trusted",),
        )

    def test_enabled_verifier_must_return_the_declared_result(self):
        """Malformed verifier adapters fail instead of being treated as support."""

        service, _, _, _ = self._service(
            AnswerVerificationPolicy("report"), object()
        )
        with self.assertRaisesRegex(ComponentContractError, "VerificationResult"):
            service.ask("question")

        service, _, _, _ = self._service(
            AnswerVerificationPolicy("report", "abstain"), object()
        )
        response = service.ask("question")
        self.assertTrue(response.abstained)
        self.assertEqual(response.abstention_reason, "verification_error")

    def test_factory_accepts_verifier_and_policy(self):
        """The composition root exposes post-generation verification controls."""

        events = []
        verifier = RecordingVerifier(events, VerificationResult(True))
        policy = AnswerVerificationPolicy("report")

        application = build_demo_rag(
            answer_verifier=verifier,
            verification_policy=policy,
        )

        self.assertIs(application.rag.answer_verifier, verifier)
        self.assertIs(application.rag.verification_policy, policy)

    def test_policy_configuration_rejects_unknown_values(self):
        """Typos cannot silently disable or weaken verification behavior."""

        invalid = (
            (("unknown", "raise"), ValueError),
            (("report", "ignore"), ValueError),
            ((True, "raise"), TypeError),
            (("report", False), TypeError),
        )
        for arguments, error in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    AnswerVerificationPolicy(*arguments)

        with self.assertRaisesRegex(ValueError, "answer_verifier"):
            RAGService(
                FixedRetriever([]),
                RecordingGenerator([]),
                verification_policy=AnswerVerificationPolicy("report"),
            )


if __name__ == "__main__":
    unittest.main()
