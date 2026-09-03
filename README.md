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

file / bytes / upload
        |
        v
loader registry -> cleaner chain       rag_ingestion
        |
        v
canonical Document
        |
        v
     Chunker -> Embedder -> VectorStore             Indexer


QUERY SIDE

question + method -> Retriever strategy -> optional Reranker -> AnswerGenerator
                           |                              |
                           +-------- evidence ------------+
                                                          |
                                                          v
                                            answer + citations       RAGService
```

Every named component is a protocol in
[`ports.py`](src/modular_rag/ports.py). Core services receive implementations
through their constructors; changing a provider does not change orchestration.

## Run the example

```bash
PYTHONPATH=src python3 examples/basic.py
```

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
configuration.

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

The contract is intentionally separate from `RAGService` in this first slice:
answers are not verified or blocked yet, so existing applications behave
exactly as before. A follow-up can inject the verifier after generation and
add explicit report or enforcement policies without coupling the core to a
model provider.

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
    ├── embedding.py       Demo hashing and optional local dense embeddings
    ├── chunking.py        Default overlapping word chunker
    ├── indexing.py        Ingest -> chunk -> embed -> store
    ├── retrieval.py       Vector retrieval and optional reranking
    ├── qasc.py            Optional query-adaptive semantic chunking
    ├── generation.py      Offline demo generator
    ├── service.py         Retrieve -> rerank -> answer -> cite
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

- Supply a stable `document_id` when indexing. Re-indexing that ID atomically
  replaces its previous chunks, preventing stale retrieval results.
- Preserve `source_name`, page numbers, headings, and bounding boxes in chunk
  metadata when a parser provides them.
- Scores come from the latest ranking stage (the retriever or, when enabled,
  the reranker). Do not treat scores from different backends as directly
  comparable.
- The default filters use exact metadata equality. A production store must
  implement equivalent filtering before results are returned.
- `DemoExtractiveGenerator`, `HashingEmbedder`, and `KeywordReranker` prove the
  wiring only; they are not substitutes for production models.

## Test

No test-runner dependency is required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

See [`tests/README.md`](tests/README.md) for the behavioral and parameter
rationale behind the refined contract tests.

Every pull request and every push to `main` installs the package, runs the
tests, and executes the end-to-end example on the oldest and newest supported
Python versions. A separate Python 3.9 job installs the optional spaCy extra and
runs the sentence-segmentation suite, so the real adapter integration is not
left permanently skipped in CI. Dependabot checks the Python and GitHub Actions
dependencies weekly and groups related updates into small pull requests.
