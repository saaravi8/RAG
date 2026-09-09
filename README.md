# Modular RAG skeleton

[![CI](https://github.com/saaravi8/RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/saaravi8/RAG/actions/workflows/ci.yml)

A small, runnable retrieval-augmented generation foundation designed so each
piece can be replaced independently. The core contains no model-provider,
database, or web-framework dependency. PDF text extraction uses `pypdf`.

The included hashing embedder, in-memory vector store, and extractive answer
generator are intentionally development-only components. They let the complete
pipeline run offline while real adapters are being built.

## Architecture

```text
WRITE SIDE

file / bytes / upload -> loader registry -> cleaner chain      rag_ingestion
local Git tree -> safe discovery -> secret screening           RepositoryIndexer
        |                              |
        +---------------+--------------+
                        v
                canonical Document
        |
        v
RoutingChunker -> Embedder -> VectorStore                         Indexer


QUERY SIDE

question + method -> Retriever -> relevance gate -> optional Reranker
                                             |                |
                                             +---- evidence ---+
                                                              v
                                      AnswerGenerator -> optional Verifier
                                                              |
                                                              v
                                  answer + citations + status       RAGService
```

Every named component is a protocol in
[`ports.py`](src/modular_rag/ports.py). Core services receive implementations
through their constructors; changing a provider does not change orchestration.

## Run the example

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=24,<27"
python -m pip install -e .
python examples/basic.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The editable install includes required ingestion dependencies and makes both
packages importable without setting `PYTHONPATH` manually. The pip upgrade is
required because older bundled versions can misread `pyproject.toml` project
metadata and produce an `UNKNOWN-0.0.0` package.

Or use the composition root directly:

```python
from rag_ingestion import DocumentSource
from modular_rag import build_demo_rag

app = build_demo_rag()

app.index(
    DocumentSource.from_text(
        "PostgreSQL stores the application data.",
        name="architecture.txt",
        metadata={
            "document_id": "architecture",
            "tenant_id": "acme",
        },
    )
)

response = app.ask(
    "Where is application data stored?",
    filters={"tenant_id": "acme"},
)

print(response.answer)
for citation in response.citations:
    print(citation.source, citation.excerpt)
```

Metadata filters are applied inside the vector store. Tenant or access-control
filters should always be supplied by trusted application code, never copied
directly from model output.

## Use production embeddings

For a local multilingual baseline, use
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base).
It is MIT-licensed, emits 768-dimensional vectors, and balances retrieval
quality against memory and inference cost. The adapter supplies E5's required
`query: ` and `passage: ` prefixes and normalizes vectors for cosine search.

Install the optional model runtime:

```bash
pip install -e ".[embeddings]"
```

Then inject the embedder into the existing composition root:

```python
from modular_rag import SentenceTransformerEmbedder, build_demo_rag

embedder = SentenceTransformerEmbedder(
    model_name="intfloat/multilingual-e5-base",
    batch_size=32,
    max_seq_length=512,
)
app = build_demo_rag(
    embedder=embedder,
    max_words=200,
    overlap_words=30,
)
```

The model's first construction downloads its weights. Pin or pre-download the
model artifact in production (use `revision` with a Hub commit hash, or pass a
local model path with `local_files_only=True`) so an upstream model update
cannot silently change all new vectors. Changing the model, prefixes,
normalization, or output dimensions requires re-embedding the entire index;
query vectors and stored document vectors must always use the same
configuration. Automatic unchanged-file reuse is deliberately stricter: the
built-in dense adapter authorizes it only for a provider-loaded model pinned to
a 40-64 character hexadecimal revision. Injected models and local model paths
remain fingerprint-opaque until the application supplies an artifact identity
that is derived from the actual model bytes.

Parameters worth tuning, in order:

1. **Retrieval quality on your own data.** Build a small evaluation set of real
   queries and relevant chunk IDs, then compare recall@k and nDCG@k. Public
   leaderboards are a shortlist, not a final decision.
2. **Language and domain.** Keep the multilingual default for mixed-language or
   cross-language retrieval. An English-only model may be faster if the corpus
   and every query are English. Legal, medical, or code corpora need their own
   evaluation set.
3. **Input length and chunking.** E5-base truncates inputs after 512 tokens.
   Keep chunks below that limit, including the `passage: ` prefix. Smaller
   `max_seq_length` values improve throughput when the relevant evidence is
   short.
4. **Vector dimensions and index size.** This model uses 768 floats: about 3 KB
   per vector in float32 before index overhead. More dimensions usually cost
   more RAM, storage, and search time; do not truncate a model unless it was
   trained to support dimension truncation.
5. **Batch size and device.** Increase `batch_size` until GPU or unified memory
   is efficiently used without out-of-memory errors. CPU deployments normally
   need smaller batches and should be benchmarked with realistic chunk lengths.
6. **Similarity and normalization.** The included stores use cosine similarity,
   and the adapter enables L2 normalization. Keep this setting identical at
   indexing and query time.
7. **Operational constraints.** Measure p95 query latency, indexing throughput,
   model download size, licensing, and whether documents are allowed to leave
   your environment. A hosted API can simplify operations but adds network,
   privacy, availability, and per-token cost considerations.

Useful model-size alternatives are `intfloat/multilingual-e5-small` (384
dimensions, faster and smaller) and `intfloat/multilingual-e5-large` (1024
dimensions, better quality but heavier). Choose between them from evaluation
results and your latency/memory budget, not dimensions alone.

## Add cross-encoder reranking

Dense retrieval is fast enough to search the full index, while a cross-encoder
can improve the final ordering by jointly scoring each question and candidate
chunk. The two stages stay independent: `RAGService` asks the retriever for
`default_fetch_k` candidates, then asks the configured `Reranker` for the best
`default_top_k` results.

Install the optional local model runtime:

```bash
pip install -e ".[reranking]"
```

Then inject the included adapter through the composition root:

```python
from modular_rag import CrossEncoderReranker, build_demo_rag

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    batch_size=32,
)
app = build_demo_rag(reranker=reranker)

# The service retrieves 20 candidates by default and returns the best 5.
response = app.ask("How does the system authenticate users?")
```

The default model is a compact English passage reranker. Select and evaluate a
multilingual or domain-specific cross-encoder when the corpus requires it.
Model scores replace the vector-store scores in the final `SearchResult` and
citation objects. To use a hosted provider instead, implement the small
`Reranker.rerank(...)` protocol and inject that adapter in the same place.

## Prepare answer verification

`AnswerVerifier` defines a provider-neutral boundary for checking whether a
generated answer is supported by the exact retrieved contexts used during
generation. Implementations return a `VerificationResult` with a boolean
support verdict, a human-readable reason, and optional provider diagnostics.

```python
from modular_rag import VerificationResult


class ProviderAnswerVerifier:
    def verify(self, question, answer, contexts):
        verdict = self.client.verify(
            question=question,
            answer=answer,
            contexts=[result.chunk.text for result in contexts],
        )
        return VerificationResult(
            supported=verdict.supported,
            reason=verdict.reason,
            metadata={"provider": "example"},
        )
```

The contract stays provider-neutral. `RAGService` can run it after generation
in report or enforcement mode; the policy and failure behavior are described
under [Verify generated answers](#verify-generated-answers).

## Index a local code repository

Repository ingestion discovers a bounded local Git working tree, screens each
candidate before indexing, and routes source/configuration files through
line- and syntax-aware chunking. It never clones repositories, follows
symlinks, imports modules, or executes repository code.

```python
from modular_rag import RepositoryLimits, RepositoryPolicy, build_demo_rag

app = build_demo_rag(
    code_max_lines=160,
    code_overlap_lines=20,
    repository_policy=RepositoryPolicy(
        limits=RepositoryLimits(
            max_files=5_000,
            max_discovered_entries=25_000,
            max_file_bytes=1_000_000,
            max_total_bytes=50_000_000,
            max_issues=500,
        )
    ),
)

report = app.index_repository(
    "/absolute/path/to/local/repository",
    repository_id="payments-service",
)

response = app.ask(
    "Where is authentication configured?",
    filters={"repository_id": "payments-service"},
)
```

`repository_id` is a stable application identity, not a path. Reuse it when
refreshing the same repository. A complete refresh deletes indexed files that
no longer exist; if discovery stops at a resource limit, previously indexed
files that were not observed are preserved.

Successful manifest entries also record a deterministic index fingerprint and
the file's indexed chunk count. On a later scan, an identical raw file is
reported as `unchanged` and skips chunking and embedding only when the processor,
chunker, embedder, auxiliary indexes, repository policy, secret scanner, and
index schema all have the same reusable identity, the manifest describes the
same physical index generation, the stored chunk count is positive, and the
recorded Git commit still matches `HEAD`. Model name and revision, prefixes,
normalization, output dimension, window sizes, parser/segmenter settings, and
security policy are included. A legacy manifest, detached store, opaque custom
component, or changed commit safely disables reuse and forces reprocessing.
The commit check is intentionally conservative: a new `HEAD` currently
reindexes unchanged files so chunk provenance cannot become stale.

Custom content-shaping components can opt into incremental reuse by exposing a
deterministic `fingerprint_components()` mapping. Change that mapping whenever
the component's output semantics change. Persist the manifest with the exact
vector/index namespace it describes; a matching content fingerprint does not
prove that an unrelated or freshly empty store contains those vectors.
Fingerprint mappings must use unique string keys and finite canonical values;
otherwise they remain diagnostic but cannot authorize reuse.

Each successfully indexed repository document also receives a random
`document_state_token`. The same token is stored in its manifest entry and in
the primary and auxiliary index metadata. Reuse requires every participant to
report that token, which detects accidental direct replacement, deletion, or a
manifest detached from its physical index. This is an integrity and cache-state
identity check, not authentication: a trusted writer that knows the token can
copy or forge it, so it must not be treated as a credential or proof of source
authenticity.

The default policy:

- respects `.gitignore` rules and excludes VCS metadata, dependency trees,
  caches, build output, generated directories, lockfiles, binaries, symlinks,
  and unknown file types;
- rejects conventional credential files and content under `.ssh/` or
  `secrets/` directories;
- redacts high-confidence private keys, cloud access keys, credential
  assignments, and credential-bearing URLs before embedding;
- attaches repository, path, language, line, symbol, parser, trust, redaction,
  and suspected prompt-injection metadata to chunks; and
- applies hard limits to discovery depth, file/byte/line/chunk counts, errors,
  metadata length, total directory entries visited, and retained diagnostic
  details.

`max_discovered_entries` counts every directory entry visited, including
ignored, excluded, and non-regular entries, so a tree full of ignored files
cannot bypass the traversal bound. Reaching the configured number is allowed;
the snapshot becomes incomplete only when a one-entry lookahead proves that
more work remains. Each directory's `.gitignore` is loaded through the bounded,
no-follow reader before that directory's entries are classified, so ignore
behavior does not depend on `os.scandir()` order at the cap. `max_issues` caps
the combined stored `skips` and `failures` details, while the aggregate
`skipped` and `failed` counts remain truthful. A terminal limit marker is
retained when possible so callers can distinguish a bounded snapshot from a
complete one.

Secret detection is defense in depth, not a guarantee. Only index repositories
you are authorized to read, keep retrieval filters in trusted application
code, and never place secrets in source control. Repository content always has
`content_trust="untrusted_repository"`; a prompt-injection flag is a warning
for downstream policy and does not make the text trusted.

The dependency-free baseline uses Python's AST and conservative declaration
patterns, then falls back to bounded line windows. For richer multi-language
parsing, install the optional Tree-sitter adapter:

```bash
python -m pip install -e ".[code]"
```

Repository ingestion does not download grammars. Provision them separately or
inject a trusted parser factory. The default manifest is in memory; inject a
persistent `RepositoryManifest` in production. A custom in-process manifest
must provide staged replacement, a repository-scoped `synchronized_scan(...)`
lease shared by every `RepositoryIndexer` instance, and the same
`IndexTransactionCoordinator` as its indexer. Only repositories with a real
`.git` directory are accepted currently, so linked Git worktrees are not yet
supported. `commit_sha` identifies `HEAD`, while `snapshot_kind="working_tree"`
warns that indexed content may also include uncommitted files.

The built-in in-memory adapters update repository state incrementally:
`InMemoryVectorStore` tracks chunk IDs and state tokens per document, QASC
updates one entry in its stable document map, and `InMemoryRepositoryManifest`
stages one file entry or deletion without copying the full repository snapshot.
These optimizations are specific to the built-in adapters; a custom manifest
that only implements full-snapshot staging remains correct but may copy all
entries for each file publication.

Current repository-indexing boundaries remain:

- chunks from one file are embedded together, but files are not combined into
  cross-file embedding batches;
- a changed `HEAD` reindexes byte-identical files so stored commit provenance
  remains current;
- the routing chunker's fingerprint includes both its prose and code/config
  branches, so changing an inactive branch still conservatively invalidates
  files routed through the other branch; and
- stores, manifest data, state identities, transaction locks, and scan leases
  supplied by the demo adapters are in-memory and process-local. They provide
  neither restart persistence nor cross-process/crash atomicity.

## Opt into QASC per query

[Query-Adaptive Semantic Chunking (QASC)](https://arxiv.org/abs/2605.22834)
constructs query-specific sentence windows around relevant seed sentences. It
is disabled by default: standard indexing and `app.ask(...)` do not build or
use the additional sentence index.

Install the optional sentence-segmentation dependency and enable QASC at the
composition root:

```bash
pip install -e ".[spacy]"
```

```python
from modular_rag import QASCConfig, build_demo_rag

app = build_demo_rag(
    enable_qasc=True,
    qasc_config=QASCConfig(
        seed_percentile=75,
        window_radius=3,
        decay=0.3,
        gap_tolerance=2,
        chunk_threshold_factor=0.6,
    ),
)

# Index once into both the standard chunk index and the optional QASC
# sentence index.
app.index(source)

standard_response = app.ask("How does the system authenticate users?")
qasc_response = app.ask(
    "How does the system authenticate users?",
    method="qasc",
)
```

The query flag is a general strategy selector, not a QASC-specific branch.
Register any retriever under its own name when constructing `RAGService`:

```python
rag = RAGService(
    standard_retriever,
    generator,
    query_methods={
        "qasc": qasc_retriever,
        "hybrid": hybrid_retriever,
    },
)

response = rag.ask(question, method="hybrid")
```

`rag.available_methods` reports the enabled choices. `"standard"` is always
present and remains the default. If another strategy needs its own document
index, register that component through `Indexer(..., document_indexes=(...))`
as well as under `query_methods`.

QASC seed-window scoring uses a linear recurrence per document, and adjacent
candidate windows are merged before their final score is aggregated. The
regression suite verifies that a 1,000-sentence document-wide window requires
one final aggregation pass over 1,000 positions, rather than repeatedly
rescoring every growing prefix or every seed window.

### Calibrate relevance and abstention

`RAGService` applies a relevance policy to retrieval candidates before optional
reranking and generation. Its default policy keeps only results with finite
retrieval scores strictly greater than `0.0`. If no results survive, the
service returns `RAGService.ABSTENTION_ANSWER` with empty results and citations,
and does not call the reranker, answer generator, or answer verifier. The
response also sets `abstained=True` and provides a stable
`abstention_reason`, so callers do not need to compare user-facing text.

Score ranges are backend-specific, so inject calibrated thresholds instead of
assuming that one value fits every retrieval method. `RelevancePolicy` and
`ScoreThresholdRelevancePolicy` are both public imports from `modular_rag`:

```python
from modular_rag import RAGService, ScoreThresholdRelevancePolicy

rag = RAGService(
    standard_retriever,
    generator,
    query_methods={"qasc": qasc_retriever},
    relevance_policy=ScoreThresholdRelevancePolicy(
        minimum_score=0.15,
        method_minimum_scores={"qasc": -0.25},
    ),
)
```

Threshold comparison is exclusive: a score must be greater than its selected
floor, not equal to it. Method names are normalized for matching. Accepted
candidates may then receive scores from a different domain during reranking;
those scores are preserved and are not compared with the retrieval threshold.
Applications with richer relevance signals can inject their own
`RelevancePolicy`. A separate reranker-stage acceptance gate needs its own
calibration rather than reusing retrieval thresholds. Choose every threshold
from an evaluation set for the actual backend and corpus instead of copying the
example values.

`build_demo_rag(relevance_policy=...)` exposes the same injection point at the
composition root. A policy may filter or reorder retrieved candidates, but it
cannot invent or duplicate evidence. Rerankers may rescore and reorder the
survivors but likewise cannot introduce new chunks.

### Coordinate multi-index replacement

When an `Indexer` has one or more auxiliary document indexes, every write
participant (the primary vector store and each auxiliary index) must implement
both `prepare_replace_document(...)` and `prepare_delete_document(...)`.
Preparation must leave query-visible state unchanged and return a handle with
`commit()` and non-raising, idempotent
`rollback()` operations. The indexer prepares every participant before the
first commit and rolls prepared participants back in reverse order if
preparation or commit fails, including process-control exceptions such as
`KeyboardInterrupt`. Unsupported multi-index compositions are rejected when the
indexer is constructed.

Prepared commits reject stale handles before mutation. Rollback restores the
prior snapshot only while that handle's committed candidate is still current;
if another effective write or delete has taken ownership, rollback leaves the
newer state intact. The in-memory store and QASC use monotonic state generations
with per-document mutation, so the same ownership checks cover deletions
without copying all unrelated documents.

The dependency-free composition shares one `IndexTransactionCoordinator`
between the primary store, QASC, repository manifest, and indexer. Participating
reads wait while a replacement or deletion commits, so they observe the
complete old state or complete new state rather than an intermediate mix.
Repository ingestion also stages each manifest change with the corresponding
file replacement or deletion; a manifest publication failure rolls the indexes
back to the same prior file version. The manifest-owned scan lease serializes
peer repository indexers refreshing the same repository identity, preventing
one full-snapshot writer from publishing a stale manifest over another.

This stricter protocol is required only for coordinated multi-index operations
or repository-manifest updates. A primary-only `Indexer` remains compatible
with the original `VectorStore` methods. The in-memory coordinator provides
single-process isolation, not process-crash durability or cross-process
transactions. A repository scan commits consistently one file at a time; it is
not an all-or-nothing transaction for the entire repository. Production
backends should provide an equivalent database transaction or versioned active
generation, and direct writers must participate in that same mechanism.

This is a compatibility break for custom transactional adapters: an adapter
that previously exposed staged replace/delete methods but no coordinator
ownership can no longer join a coordinated composition. Every configured
`TransactionalVectorStore` and `TransactionalDocumentIndex` must expose the
exact same `IndexTransactionCoordinator`; dynamically supplied transaction
handles must satisfy `CoordinatedPreparedDocumentReplacement` and carry that
same coordinator. The protocols and coordinator are public imports:

```python
from modular_rag import (
    CoordinatedPreparedDocumentReplacement,
    IndexTransactionCoordinator,
    PreparedDocumentDeletion,
    PreparedDocumentReplacement,
    TransactionalDocumentIndex,
    TransactionalVectorStore,
)
```

Primary-only applications that call the original direct `replace_document`,
`delete_document`, and `search` methods need no transactional retrofit unless
they add an auxiliary index or repository-manifest participant.

### Reject unsafe vectors at adapter boundaries

`VectorRecord`, `SearchResult`, and `Citation` reject malformed or non-finite
numeric data at construction. The in-memory vector store validates both
replacement and query vectors again before mutation or cosine scoring. QASC
applies the same rule to sentence-embedding batches and query embeddings.
Vectors must be iterable, nonempty, numeric,
finite, rectangular within a replacement batch, and dimension-compatible with
the existing index. Malformed values, `NaN`, and positive or negative infinity
are rejected; a rejected replacement preserves the last valid state. Cosine
scoring scales coordinates before computing norms and dot products so extreme
but finite values do not overflow into `NaN` scores. Replacement chunk IDs must
also be unique and cannot collide with records owned by another document.

### Immutable metadata and serialization

`SentenceSpan`, `Chunk`, `Citation`, and `VerificationResult` take a deep
snapshot of metadata at construction. The supported domain is `None`, built-in
booleans, integers, floats, strings, and bytes; mappings must have built-in
string keys; lists/tuples become tuples; sets become frozensets; and mutable
`bytearray`/`memoryview` inputs become bytes. Arbitrary custom objects are
rejected because copying them cannot guarantee isolation. Nested mappings are
read-only, so adapter or caller mutation cannot rewrite evidence already stored
in a response.

`dict(model.metadata)` materializes only the top-level mapping. JSON and other
wire formats need an explicit recursive conversion policy, especially for
bytes and sets. For example:

```python
from collections.abc import Mapping
import json


def materialize_metadata(value):
    if isinstance(value, Mapping):
        return {key: materialize_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [materialize_metadata(item) for item in value]
    if isinstance(value, frozenset):
        return [
            materialize_metadata(item)
            for item in sorted(value, key=repr)
        ]
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    return value


payload = json.dumps(
    materialize_metadata(response.results[0].chunk.metadata),
    allow_nan=False,
)
```

The bytes representation and set ordering are application-level schema choices;
use a tagged format that the receiving side can decode, and decide explicitly
how to handle non-finite metadata floats when the target format forbids them.

### Verify generated answers

Answer verification is optional and runs after generation. Inject an
`AnswerVerifier` together with `AnswerVerificationPolicy(mode=...)`:

- `disabled` preserves the original behavior and does not call the verifier.
- `report` returns the generated answer with its `VerificationResult`.
- `enforce` replaces unsupported output with a structured abstention while
  preserving the evidence and verdict for inspection.

Verifier exceptions either propagate with `on_error="raise"` or fail closed
with `on_error="abstain"`. Both choices are explicit; verifier failure is never
silently treated as support.

### Measure retrieval and abstention

`evaluate_observations(...)` calculates recall@k, mean reciprocal rank, nDCG,
abstention precision/recall, and unsupported-answer rate from fixed cases. It
rejects duplicate ranked IDs and incomplete or contradictory answer labels;
the report exposes answered and verifier-checked denominators so a zero safety
rate cannot look meaningful when no answer was evaluated. The required smoke
corpus in `tests/fixtures/rag_evaluation.json` covers English, Hebrew, prose,
code, and unanswerable questions with a deterministic local retriever and an
independent exact-context verifier. It validates the evaluation and
orchestration path, not the quality of a production embedding model. Real E5
and cross-encoder measurements should remain an explicit optional suite and
must report which weights were actually available.

## Packages

```text
src/
├── rag_ingestion/
│   ├── models.py          DocumentSource and canonical Document
│   ├── registry.py        Dynamic loader and cleaner registries
│   └── pipeline.py        Load -> clean orchestration
└── modular_rag/
    ├── ports.py           Replaceable component contracts, including verification
    ├── models.py          Sentence, chunk, vector, citation, verification, response models
    ├── transactions.py    Shared in-process index transaction coordinator
    ├── fingerprint.py     Deterministic index identities for safe incremental reuse
    ├── evaluation.py      Retrieval and answer-safety metrics
    ├── embedding.py       Demo hashing and optional local dense embeddings
    ├── chunking.py        Default overlapping word chunker
    ├── code.py            Non-executing syntax-aware repository chunking
    ├── indexing.py        Ingest -> chunk -> embed -> store
    ├── repository.py      Safe repository discovery, screening, and manifests
    ├── retrieval.py       Vector retrieval and optional reranking
    ├── qasc.py            Optional query-adaptive semantic chunking
    ├── relevance.py       Pre-generation evidence selection policies
    ├── verification.py    Post-generation verification policy
    ├── generation.py      Offline demo generator
    ├── service.py         Retrieve -> relevance gate -> rerank -> answer -> cite
    ├── factory.py         Dependency-free composition root
    └── adapters/
        ├── memory.py      In-memory vector store
        └── spacy_sentences.py  Optional spaCy sentence identification
```

## Replace a component

Implement only the relevant protocol and inject it at the composition root.
For example, a production embedder needs these two operations:

```python
class ProductionEmbedder:
    def embed_documents(self, texts):
        # Call a local model or embedding API in batches.
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)
```

Wire custom components explicitly:

```python
from modular_rag import Indexer, RAGService, VectorRetriever
from rag_ingestion import create_default_pipeline

processor = create_default_pipeline()
chunker = MyStructureAwareChunker()
embedder = ProductionEmbedder()
store = PgVectorStore(connection_pool)

indexer = Indexer(processor, chunker, embedder, store)
retriever = VectorRetriever(embedder, store)
rag = RAGService(
    retriever,
    MyAnswerGenerator(model_client),
    reranker=MyReranker(),
)
```

Suggested adapter modules as the project grows:

```text
adapters/
├── embeddings/
│   ├── local.py
│   └── provider_api.py
├── generators/
│   └── provider_api.py
├── loaders/
│   ├── docling.py
│   └── markitdown.py
└── stores/
    ├── pgvector.py
    └── qdrant.py
```

Do not put provider SDK objects into domain models. Translate provider results
to `Vector`, `SearchResult`, and strings at the adapter boundary.

## Add document formats and cleaners

The ingestion package supports PDF, plain text, Markdown, JSON, CSV, and HTML
files by default. Pass a file path directly to the application; its suffix
selects the loader:

```python
from modular_rag import build_demo_rag

app = build_demo_rag()
app.index(
    "documents/handbook.pdf",
    metadata={"document_id": "handbook", "tenant_id": "acme"},
)
app.index("documents/notes.txt")
app.index("documents/products.csv")
app.index("documents/help.html")
```

PDF ingestion extracts embedded text, labels it by page, and records the page
count and any pages without extractable text. Scanned/image-only PDFs need an
OCR-capable custom loader. CSV rows are rendered with their column labels, and
HTML ingestion keeps visible text while excluding scripts and styles.

The exact active extensions are available from
`processor.supported_document_types`. New handlers are ordinary functions:

```python
from modular_rag import build_demo_rag
from rag_ingestion import Document, create_default_pipeline

processor = create_default_pipeline()

@processor.register_loader("docx")
def load_docx(source):
    parsed = my_docx_parser(source.read_bytes())
    return Document(
        text=parsed.text,
        document_type=source.document_type,
        metadata={"heading_count": parsed.heading_count},
    )

@processor.register_cleaner("docx")
def clean_docx(document):
    return document.with_text(remove_repeated_headers(document.text))

# Pass this configured beginning of the pipeline into the application.
app = build_demo_rag(processor=processor)
```

Registering the loader is what makes a format supported. The filename suffix
is normalized, so `.PDF`, `pdf`, and ` pdf ` all select the same handler. A
loader must return the canonical `Document`; zero or more cleaners can then be
registered for that format.

## Identify sentences with spaCy

Sentence identification is an optional stage after loading and cleaning. Install
the adapter dependency and use it explicitly:

```bash
pip install -e ".[spacy]"
```

```python
from modular_rag import SpacySentenceSegmenter
from rag_ingestion import create_default_pipeline

processor = create_default_pipeline()
document = processor.process("guide.html")
sentences = SpacySentenceSegmenter(language="en").segment(document)
```

The default adapter uses spaCy's rule-based `sentencizer`, so no language-model
download is required. Each `SentenceSpan` contains stable identity, text,
document identity, character offsets, sequence index, and inherited source
metadata.

Downstream code should depend on the small `SentenceSegmenter` protocol rather
than on spaCy. A replacement only needs to implement:

```python
class CustomSentenceSegmenter:
    def segment(self, document):
        return sentence_spans
```

## Important semantics

- Supply a stable `document_id` when indexing. Re-indexing that ID replaces its
  previous chunks, preventing stale results in the successfully updated store.
  See the multi-index replacement limits above before treating this as a
  database-level atomicity or isolation guarantee.
- Preserve `source_name`, page numbers, headings, and bounding boxes in chunk
  metadata when a parser provides them.
- Scores come from the latest ranking stage (the retriever or, when enabled,
  the reranker). Do not treat scores from different backends as directly
  comparable.
- The default filters use exact metadata equality. A production store must
  implement equivalent filtering before results are returned.
- Treat all repository text as untrusted input even after secret redaction.
  Enforce tenant/repository filters outside the model and inspect
  `prompt_injection_suspected` according to application policy.
- Persist the repository manifest alongside a production vector store so
  refreshes can remove deleted files without confusing transient failures with
  intentional deletion. Keep the manifest and its vector/index namespace as one
  operational unit; do not reuse a manifest against a different store. Ensure
  every refresh process participates in one database transaction or equivalent
  repository-scoped lease; the built-in lock is single-process only.
- `DemoExtractiveGenerator`, `HashingEmbedder`, and `KeywordReranker` prove the
  wiring only; they are not substitutes for production models.

## Test

No test-runner dependency is required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

See [`tests/README.md`](tests/README.md) for the behavioral and parameter
rationale behind the refined contract tests.

At commit `d508311`, the local suite runs 256 tests: 254 pass, while two
optional integration tests skip when Tree-sitter or spaCy extras are not
installed.

Every pull request and every push to `main` installs the package, runs the
dependency-consistency and source-compilation gates, runs the tests, and
executes the end-to-end example on the oldest and newest tested Python
versions. A separate Python 3.9 and 3.14 matrix installs the optional spaCy
extra and runs the sentence-segmentation suite, exercising both declared spaCy
dependency branches. The packaging gate builds a wheel from the checked-out
revision, installs it, checks its dependencies and metadata, imports both public
packages from outside the checkout, and constructs the demo application. These
are required CI gates, not a claim that a particular remote CI run has passed.
Dependabot checks the Python and GitHub Actions dependencies weekly and groups
related updates into small pull requests.
