import math
import tempfile
import unittest
from unittest.mock import patch

from rag_ingestion import create_default_pipeline

from modular_rag import (
    HashingEmbedder,
    QASCConfig,
    QASCRetriever,
    SentenceTransformerEmbedder,
    WordWindowChunker,
    build_demo_rag,
    build_index_fingerprint,
)


class _FakeSentenceTransformer:
    max_seq_length = 512

    def encode(self, texts, **kwargs):
        del kwargs
        return [[float(index + 1), 0.0, 1.0] for index, _ in enumerate(texts)]

    def get_sentence_embedding_dimension(self):
        return 3


class _FingerprintableSegmenter:
    def segment(self, document):
        del document
        return ()

    def fingerprint_components(self):
        return {"algorithm": "test-segmenter", "version": 1}


class FingerprintTests(unittest.TestCase):
    def test_equal_configurations_have_equal_reusable_fingerprints(self):
        """Equivalent composition roots produce one stable cache identity."""

        first = build_demo_rag().indexer.index_fingerprint
        second = build_demo_rag().indexer.index_fingerprint

        self.assertTrue(first.reusable)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.components, second.components)

    def test_content_shaping_configuration_changes_digest(self):
        """Embedding, chunking, QASC, and schema changes invalidate old vectors."""

        baseline = build_index_fingerprint(
            HashingEmbedder(16),
            WordWindowChunker(20, 2),
            processor=create_default_pipeline(),
        ).digest
        alternatives = (
            build_index_fingerprint(
                HashingEmbedder(32),
                WordWindowChunker(20, 2),
                processor=create_default_pipeline(),
            ).digest,
            build_index_fingerprint(
                HashingEmbedder(16),
                WordWindowChunker(21, 2),
                processor=create_default_pipeline(),
            ).digest,
            build_index_fingerprint(
                HashingEmbedder(16),
                WordWindowChunker(20, 2),
                processor=create_default_pipeline(),
                schema_version=2,
            ).digest,
        )

        self.assertTrue(all(candidate != baseline for candidate in alternatives))

    def test_sentence_transformer_semantics_are_part_of_the_digest(self):
        """Model, revision, prefixes, normalization, and dimensions are explicit."""

        def fingerprint(**kwargs):
            embedder = SentenceTransformerEmbedder(
                model=_FakeSentenceTransformer(),
                **kwargs,
            )
            return build_index_fingerprint(
                embedder,
                WordWindowChunker(),
                processor=create_default_pipeline(),
            )

        baseline = fingerprint(
            model_name="model-a",
            revision="one",
            query_prefix="query: ",
            document_prefix="passage: ",
            normalize_embeddings=True,
        )
        changed = (
            fingerprint(model_name="model-b", revision="one"),
            fingerprint(model_name="model-a", revision="two"),
            fingerprint(model_name="model-a", revision="one", query_prefix="q: "),
            fingerprint(
                model_name="model-a", revision="one", document_prefix="doc: "
            ),
            fingerprint(
                model_name="model-a", revision="one", normalize_embeddings=False
            ),
        )

        self.assertFalse(baseline.reusable)
        self.assertTrue(all(item.digest != baseline.digest for item in changed))
        configuration = baseline.components["embedder"]["configuration"]
        self.assertEqual(configuration["dimensions"], 3)

    def test_local_dense_model_path_disables_reuse_without_an_artifact_digest(self):
        """A path name alone cannot prove that local model bytes are unchanged."""

        with tempfile.TemporaryDirectory() as model_directory:
            with patch.object(
                SentenceTransformerEmbedder,
                "_load_model",
                return_value=_FakeSentenceTransformer(),
            ):
                embedder = SentenceTransformerEmbedder(
                    model_name=model_directory,
                    revision="a" * 40,
                )

            identity = build_index_fingerprint(
                embedder,
                WordWindowChunker(),
                processor=create_default_pipeline(),
            )

        self.assertFalse(identity.reusable)
        configuration = identity.components["embedder"]["configuration"]
        self.assertTrue(configuration["local_model_path"])

    def test_qasc_query_configuration_does_not_change_the_index_fingerprint(self):
        """Query-time window tuning does not invalidate the sentence index."""

        embedder = HashingEmbedder(16)
        segmenter = _FingerprintableSegmenter()
        first = QASCRetriever(
            segmenter,
            embedder,
            config=QASCConfig(window_radius=2),
        )
        second = QASCRetriever(
            segmenter,
            embedder,
            config=QASCConfig(window_radius=4),
        )

        first_digest = build_index_fingerprint(
            embedder, WordWindowChunker(), (first,)
        ).digest
        second_digest = build_index_fingerprint(
            embedder, WordWindowChunker(), (second,)
        ).digest

        self.assertEqual(first_digest, second_digest)

    def test_opaque_components_disable_reuse(self):
        """Unknown custom behavior is hashed for diagnostics but never trusted."""

        class OpaqueChunker:
            def chunk(self, document):
                del document
                return ()

        fingerprint = build_index_fingerprint(
            HashingEmbedder(),
            OpaqueChunker(),
            processor=create_default_pipeline(),
        )

        self.assertFalse(fingerprint.reusable)
        self.assertTrue(fingerprint.components["chunker"]["opaque"])

    def test_non_finite_custom_fingerprint_values_disable_reuse(self):
        """A non-canonical custom identity cannot authorize cache reuse."""

        class UnsafeChunker:
            def chunk(self, document):
                del document
                return ()

            def fingerprint_components(self):
                return {"threshold": math.nan}

        fingerprint = build_index_fingerprint(
            HashingEmbedder(),
            UnsafeChunker(),
            processor=create_default_pipeline(),
        )

        self.assertFalse(fingerprint.reusable)

    def test_non_string_or_colliding_mapping_keys_disable_reuse(self):
        """String normalization cannot hide distinct custom configuration keys."""

        class CollidingChunker:
            def chunk(self, document):
                del document
                return ()

            def fingerprint_components(self):
                return {1: "numeric", "1": "text"}

        fingerprint = build_index_fingerprint(
            HashingEmbedder(),
            CollidingChunker(),
            processor=create_default_pipeline(),
        )

        self.assertFalse(fingerprint.reusable)


if __name__ == "__main__":
    unittest.main()
