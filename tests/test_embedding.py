import math
import unittest

from modular_rag import SentenceTransformerEmbedder, build_demo_rag


class FakeSentenceTransformer:
    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]

    def get_sentence_embedding_dimension(self):
        return 2


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_applies_e5_retrieval_prefixes_and_encoding_options(self):
        model = FakeSentenceTransformer()
        embedder = SentenceTransformerEmbedder(
            model=model,
            batch_size=8,
            max_seq_length=384,
        )

        documents = embedder.embed_documents(["first", "second"])
        query = embedder.embed_query("question")

        self.assertEqual(model.calls[0][0], ["passage: first", "passage: second"])
        self.assertEqual(model.calls[1][0], ["query: question"])
        self.assertEqual(model.calls[0][1]["batch_size"], 8)
        self.assertTrue(model.calls[0][1]["normalize_embeddings"])
        self.assertTrue(model.calls[0][1]["convert_to_numpy"])
        self.assertEqual(model.max_seq_length, 384)
        self.assertEqual(embedder.dimensions, 2)
        self.assertEqual(len(documents), 2)
        self.assertEqual(query, (15.0, 1.0))

    def test_empty_document_batch_does_not_call_the_model(self):
        model = FakeSentenceTransformer()
        embedder = SentenceTransformerEmbedder(model=model)

        self.assertEqual(embedder.embed_documents([]), ())
        self.assertEqual(model.calls, [])

    def test_rejects_invalid_batch_and_sequence_lengths(self):
        model = FakeSentenceTransformer()

        with self.assertRaisesRegex(ValueError, "batch_size"):
            SentenceTransformerEmbedder(model=model, batch_size=0)
        with self.assertRaisesRegex(ValueError, "exceeds the model limit"):
            SentenceTransformerEmbedder(model=model, max_seq_length=513)

    def test_factory_uses_an_injected_embedder(self):
        embedder = SentenceTransformerEmbedder(model=FakeSentenceTransformer())

        app = build_demo_rag(embedder=embedder)

        self.assertIs(app.indexer.embedder, embedder)

    def test_rejects_non_finite_model_output(self):
        class BrokenModel(FakeSentenceTransformer):
            def encode(self, texts, **kwargs):
                del texts, kwargs
                return [[math.nan, 1.0]]

        embedder = SentenceTransformerEmbedder(model=BrokenModel())

        with self.assertRaisesRegex(TypeError, "non-finite"):
            embedder.embed_query("question")


if __name__ == "__main__":
    unittest.main()
