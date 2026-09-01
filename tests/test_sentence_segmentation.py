import importlib.util
import unittest

from rag_ingestion import Document

from modular_rag import ComponentContractError, SentenceSpan, SpacySentenceSegmenter


class FakeSentence:
    def __init__(self, text, start_char, end_char):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class FakeDoc:
    def __init__(self, sentences):
        self.sents = sentences


class FakeNLP:
    def __call__(self, text):
        return FakeDoc(
            (
                FakeSentence(text[0:15], 0, 15),
                FakeSentence(text[16:32], 16, 32),
            )
        )


class SentenceSegmentationContractTests(unittest.TestCase):
    def test_spacy_adapter_returns_provider_neutral_spans(self):
        document = Document(
            "First sentence. Second sentence.",
            "txt",
            {"document_id": "guide", "source_name": "guide.txt"},
        )

        sentences = SpacySentenceSegmenter(nlp=FakeNLP()).segment(document)

        self.assertEqual([sentence.text for sentence in sentences], [
            "First sentence.",
            "Second sentence.",
        ])
        self.assertTrue(all(isinstance(sentence, SentenceSpan) for sentence in sentences))
        self.assertEqual(sentences[0].document_id, "guide")
        self.assertEqual(sentences[1].start_char, 16)
        self.assertEqual(sentences[1].metadata["source_name"], "guide.txt")
        self.assertEqual(sentences[1].metadata["sentence_segmenter"], "spacy")

    def test_spacy_adapter_trims_spans_skips_blanks_and_generates_a_stable_id(self):
        """Provider whitespace cannot corrupt offsets or create empty sentences."""

        class WhitespaceNLP:
            def __call__(self, text):
                del text
                return FakeDoc(
                    (
                        FakeSentence("  First. ", 0, 9),
                        FakeSentence("   ", 9, 12),
                    )
                )

        document = Document("  First.    ", "txt", {"source_name": "guide.txt"})

        first = SpacySentenceSegmenter(nlp=WhitespaceNLP()).segment(document)
        second = SpacySentenceSegmenter(nlp=WhitespaceNLP()).segment(document)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].text, "First.")
        self.assertEqual((first[0].start_char, first[0].end_char), (2, 8))
        self.assertTrue(first[0].document_id.startswith("doc_"))
        self.assertEqual(first[0].document_id, second[0].document_id)

    def test_spacy_adapter_rejects_a_non_callable_pipeline(self):
        """Injection fails immediately when the supplied adapter is unusable."""

        with self.assertRaisesRegex(TypeError, "nlp must be a callable"):
            SpacySentenceSegmenter(nlp=object())

    def test_spacy_adapter_reports_missing_sentence_boundaries(self):
        """A spaCy pipeline without a parser or sentencizer gets an actionable error."""

        class BoundarylessDoc:
            @property
            def sents(self):
                raise ValueError("sentence boundaries unset")

        class BoundarylessNLP:
            def __call__(self, text):
                del text
                return BoundarylessDoc()

        with self.assertRaisesRegex(
            ComponentContractError, "did not set sentence boundaries"
        ):
            SpacySentenceSegmenter(nlp=BoundarylessNLP()).segment(
                Document("Sentence.", "txt")
            )

    @unittest.skipUnless(importlib.util.find_spec("spacy"), "spaCy is optional")
    def test_real_spacy_sentencizer_identifies_boundaries(self):
        document = Document(
            "First sentence. Second sentence! Is this third?",
            "txt",
            {"document_id": "guide"},
        )

        sentences = SpacySentenceSegmenter().segment(document)

        self.assertEqual([sentence.text for sentence in sentences], [
            "First sentence.",
            "Second sentence!",
            "Is this third?",
        ])


if __name__ == "__main__":
    unittest.main()
