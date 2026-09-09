# Stabilization commit guide

## Canonical model validation

Models reject malformed identities, positions, vectors and nonfinite scores. Nested metadata becomes an immutable snapshot, including mutable byte buffers. Domain-model tests cover invalid fields and evidence isolation.

## Component fingerprints

Content-shaping adapters expose deterministic configuration descriptions. Opaque adapters disable safe reuse. Embedding and chunking settings are validated. Embedding and code-parser tests cover malformed provider values and bounded source spans; index fingerprint integration follows in the next change.

## Coordinated indexing

Primary storage and QASC share prepare/commit/rollback ownership. Invalid vectors, chunks and sentence spans fail before replacing evidence. Regression tests inject participant failures and concurrent replacement, and check fingerprints. Custom multi-index adapters must implement coordinated transactions.
