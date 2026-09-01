import importlib.util
import unittest

from rag_ingestion import Document

from modular_rag import SentenceSpan, SpacySentenceSegmenter


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
