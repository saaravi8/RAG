# Modular RAG skeleton

A small, runnable retrieval-augmented generation foundation designed so each
piece can be replaced independently. The core contains no model-provider,
database, web-framework, or document-parser dependency.

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

question -> Retriever -> optional Reranker -> AnswerGenerator
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

## Packages

```text
src/
├── rag_ingestion/
│   ├── models.py          DocumentSource and canonical Document
│   ├── registry.py        Dynamic loader and cleaner registries
│   └── pipeline.py        Load -> clean orchestration
└── modular_rag/
    ├── ports.py           Replaceable component contracts
    ├── models.py          Sentence, chunk, vector, citation, response models
    ├── chunking.py        Default overlapping word chunker
    ├── indexing.py        Ingest -> chunk -> embed -> store
    ├── retrieval.py       Vector retrieval and optional reranking
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

The ingestion package supports `txt`, `md`, `markdown`, `json`, `csv`, `htm`,
and `html` by default. The active list is always available from
`processor.supported_document_types`. New handlers are ordinary functions:

```python
from modular_rag import build_demo_rag
from rag_ingestion import Document, create_default_pipeline

processor = create_default_pipeline()

@processor.register_loader("pdf")
def load_pdf(source):
    parsed = my_pdf_parser(source.read_bytes())
    return Document(
        text=parsed.text,
        document_type=source.document_type,
        metadata={"page_count": parsed.page_count},
    )

@processor.register_cleaner("pdf")
def clean_pdf(document):
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
- Retrieval scores are store-specific. Do not treat scores from different
  backends as directly comparable.
- The default filters use exact metadata equality. A production store must
  implement equivalent filtering before results are returned.
- `DemoExtractiveGenerator` and `HashingEmbedder` prove the wiring only; they
  are not substitutes for production embedding and generation models.

## Test

No test-runner dependency is required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
