# Stabilization commit guide

## Canonical model validation

Models reject malformed identities, positions, vectors and nonfinite scores. Nested metadata becomes an immutable snapshot, including mutable byte buffers. Domain-model tests cover invalid fields and evidence isolation.
