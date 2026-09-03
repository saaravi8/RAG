# Test rationale

The suite uses parameters that isolate a contract boundary or make ordering
deterministic. They are not production retrieval-quality targets. This document
records the rationale for tests added or materially refined during the contract
test expansion.

## Built-in ingestion handlers

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_pdf_loader_reports_pages_and_empty_pages` | PDF routing, page accounting, document metadata, and source provenance | A one-page blank PDF is generated in memory with `pypdf`; it deterministically exercises the empty-page path without checking parser-specific text layout, while title and author verify useful metadata extraction. |
| `test_pdf_loader_extracts_text_with_a_page_label` | Embedded PDF text is extracted and retains its human-readable page reference | A minimal one-page Helvetica PDF contains a single unique phrase, avoiding layout ambiguity while exercising the real `pypdf` parser. |
| `test_default_pipeline_ingests_supported_file_paths` | PDF, TXT, CSV, and HTML all work through the filesystem entry point | One minimal file per requested format verifies suffix routing, extracted text, and `source_path` provenance without involving retrieval ranking. |
| `test_text_normalization_handles_unicode_cr_and_outer_blank_lines` | NFC normalization, classic-Mac newlines, trailing spaces, and outer blank removal | A decomposed `e` + acute accent proves NFC composition; `\r` isolates CR conversion from the already-covered CRLF case; two outer CRs prove blank trimming without changing internal prose. |
| `test_json_loader_preserves_unicode_and_reports_non_object_roots` | Readable Unicode and correct root metadata for JSON arrays | A two-item array is the smallest non-object root containing both Unicode text and a nested object; `42` is a stable scalar with no formatting ambiguity. |
| `test_json_loader_rejects_malformed_input` | Invalid structured data is not silently indexed | `{"broken":` is valid enough to enter JSON parsing but necessarily incomplete, producing the standard `JSONDecodeError`. |
| `test_csv_loader_handles_empty_input` | Empty input has zero rows and columns | `b""` is the actual empty-file boundary; no fabricated header or row should appear. |
| `test_csv_loader_handles_bom_blank_headers_and_uneven_rows` | BOM decoding, fallback headers, and short/long rows | One blank header triggers `column_2`; a four-cell row against three headers triggers `column_4`; a two-cell row proves missing-cell padding. The final expectation ends at `role:` because global normalization intentionally removes trailing whitespace. |
| `test_html_loader_preserves_entities_and_block_boundaries` | Visible entities and paragraph/break structure while skipped elements stay hidden | `&amp;` exercises entity decoding, `<br/>` the self-closing block path, and `template`/`svg` two separate skipped-content categories. |

## Document pipeline

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_cleaner_prepend_and_replace_control_the_type_specific_chain` | Exact semantics of `prepend=True` and `replace=True` | Appending `A`, prepending `B`, then replacing with `C` makes order observable as `0BA` and replacement observable as `0C`; single characters avoid text-normalization effects. |
| `test_loader_must_preserve_the_resolved_document_type` | A dispatched loader cannot relabel its result | Registered `txt` returning `pdf` is the smallest unambiguous mismatch and verifies both sides of the error message. |
| `test_cleaner_must_return_a_document` | Cleaner output obeys the canonical model contract | A plain string is callable output but not a `Document`, isolating result validation from invocation failure. |
| `test_cleaner_must_preserve_the_document_type` | Cleaners cannot change routing identity | `txt` to `pdf` is a clear type mutation while leaving the text valid. |
| `test_source_arguments_reject_ambiguous_or_unsupported_inputs` | Provenance cannot be supplied twice, and invalid source classes fail early | A `DocumentSource` plus metadata tests duplicate provenance; a path plus `name` tests conflicting filesystem identity; integer `42` is neither bytes, path, nor `DocumentSource`. |

## Sentence segmentation

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_spacy_adapter_trims_spans_skips_blanks_and_generates_a_stable_id` | Correct offsets after trimming, blank-span removal, and deterministic fallback IDs | Two leading and one trailing spaces make the expected span exactly `[2, 8)`; a second all-space span must disappear; repeating the same `source_name` must reproduce the ID. |
| `test_spacy_adapter_rejects_a_non_callable_pipeline` | Invalid dependency injection fails at construction | `object()` is deliberately non-callable and has no spaCy-like accidental behavior. |
| `test_spacy_adapter_reports_missing_sentence_boundaries` | Boundaryless spaCy pipelines return the project-specific actionable error | The fake `sents` property raises the same `ValueError` shape as spaCy without requiring the optional package. |

## Existing modular RAG tests refined

| Test | Refinement | Parameter rationale |
| --- | --- | --- |
| `test_demo_factory_accepts_a_pipeline_with_a_new_format` | Moved from `ChunkerTests` to `FactoryIntegrationTests` and documented | No runtime parameter changed; the move fixes ownership because the test exercises composition and injection, not chunk boundaries. |
| `test_replacement_removes_old_chunks_without_touching_other_documents` | Replaced the unsupported “atomic” claim with assertions for stale-chunk removal and document isolation | Existing 2D vectors and `top_k=5` were retained: 2D makes cosine order obvious, and five is safely above the three possible records so truncation cannot hide stale data. |
| `test_demo_pipeline_indexes_retrieves_answers_and_cites` | Added chunk-count, result-count, citation-count, marker, and excerpt checks | Existing `max_words=20` and overlap `3` were retained so both short sentences form one evidence chunk; this isolates wiring and citation fidelity from ranking across chunks. |
| `test_reindex_and_delete_remove_stale_evidence_end_to_end` | Added application-level replacement and deletion coverage | `max_words=10` fits each five-word version in exactly one chunk; overlap `0` removes overlap as a confounder; the same `document_id` is required to exercise replacement. |
| `test_hashing_embedder_is_deterministic_normalized_and_input_sensitive` | Strengthened same-input equality with dimension, norm, different-input, and empty-input assertions | Dimension `32` was retained because it is small and fast while giving the two chosen token sets ample room to avoid a contrived collision; unit norm is the cosine-search contract, while empty text must remain the zero vector. |

## Core contract tests

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_chunker_rejects_invalid_window_configuration` | Constructor boundaries and guaranteed forward progress | `max_words=0` is the nonpositive boundary; overlap `-1` is the first negative value; overlap `4` equals a four-word window and would make the step zero; overlap `3` proves the largest valid value. |
| `test_chunker_generates_stable_source_based_document_ids` | Fallback IDs are deterministic and provenance-sensitive | Four words with window `3` and overlap `1` must start at words `0` and `2`; identical text under two source names proves provenance participates in identity. |
| `test_rejected_replacements_preserve_existing_records` | Store validation occurs before mutation | A valid 2D record is challenged by a wrong owner, a mixed 2D/3D batch, and an all-3D incompatible replacement; querying 2D afterward proves the original survived each failure. |
| `test_deleting_the_last_document_resets_vector_dimension` | Empty stores can migrate embedding dimensions | The change from 2D to 3D is the smallest visible dimension migration and occurs only after deletion of the sole document. |
| `test_search_supports_reserved_filters_and_deterministic_ties` | Reserved ID filters and stable tie-breaking | Two equal unit vectors force a score tie; reverse insertion order proves sorting uses chunk IDs; `top_k=2` includes both, and combined document/chunk filters select exactly one. |
| `test_search_rejects_nonpositive_top_k` | Invalid result limits fail explicitly | `0` tests the empty boundary and `-1` the negative domain. |
| `test_indexer_rejects_an_empty_chunk_sequence` | A processed document cannot report success with no evidence | An eight-dimensional embedder is a cheap valid collaborator but is intentionally never called; the empty tuple isolates the chunker contract. |
| `test_indexer_rejects_chunks_with_multiple_document_ids` | One replacement cannot mix document owners | Two chunks are the minimum sequence that can disagree; IDs `a` and `b` make the mismatch direct. |
| `test_indexer_updates_and_deletes_auxiliary_document_indexes` | Optional indexes follow primary replacement/deletion lifecycle | Window `10` with overlap `0` forces the three-word document into one chunk; dimension `8` keeps hashing cheap while remaining a valid nonempty vector. |
| `test_hashing_embedder_rejects_nonpositive_dimensions` | Embeddings always have at least one coordinate | `0` is the empty-vector boundary and `-1` covers the negative domain. |
| `test_structural_verifier_receives_the_answer_and_exact_contexts` | Third-party answer verifiers satisfy the protocol structurally and receive the generated answer plus its evidence unchanged | One result is the smallest nonempty evidence set; distinct `question` and `answer` strings make argument order observable, while score `0.9` is data forwarded to the adapter rather than a verification threshold. |
| `test_verification_result_validates_fields_and_snapshots_metadata` | Verification verdicts remain unambiguous and their top-level diagnostics cannot change through the caller's original mapping | `False` exercises an unsupported verdict; integer `1` must not masquerade as a boolean, `None` is the smallest invalid reason, and a single metadata key is sufficient to prove defensive copying. |
| `test_service_rejects_invalid_default_limits` | Service defaults are positive and fetch covers final selection | `(0,1)` invalidates final count, `(1,0)` invalidates fetch count, and `(3,2)` uses positive values but violates `fetch >= final`. |
| `test_service_rejects_invalid_requests_and_method_registrations` | Questions, per-call limits, method types, reserved names, and normalized duplicates are validated | Whitespace isolates empty text; `top_k=0` is the boundary; integer `7` is non-string; padded/case-varied method names prove validation happens after normalization. |
| `test_reranking_fetches_more_candidates_than_it_returns` | Candidate breadth and final response size remain separate | Fetch `4` and final `2` are deliberately different and small; reversing four candidates makes it observable that reranking saw all four before two were returned. |
| `test_citations_normalize_excerpts_and_apply_source_fallbacks` | Citation numbering, readable excerpts, and provenance fallback | Two results are the minimum for numbering; scores `0.9` and `0.8` keep their order deterministic but are not acceptance thresholds; missing source metadata on the second result forces document-ID fallback. |
| `test_qasc_configuration_requires_explicit_enablement` | Optional indexing cost cannot be activated accidentally | Segmenter-only and config-only cases cover both optional arguments independently while leaving `enable_qasc` at its explicit default `False`. |

## Cross-encoder reranking

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_scores_query_chunk_pairs_and_returns_the_best_results` | The adapter sends the correct pairs, honors inference settings, ranks by model relevance, and exposes those final scores | Three candidates make the reordering visible; `top_k=2` proves truncation happens after all candidate scores are considered; batch size `8` is distinct from both counts so argument forwarding is observable. |
| `test_equal_scores_preserve_retrieval_order` | Model ties remain deterministic without inventing a secondary relevance signal | Two equal `0.5` scores are the minimum tie; reverse-alphabetical IDs prove the original retrieval order, rather than chunk identity, resolves it. |
| `test_empty_candidates_skip_model_inference` | Empty retrieval results do not trigger unnecessary model calls | An empty tuple is the exact no-evidence boundary while a positive limit keeps validation separate. |
| `test_rejects_invalid_configuration_and_limits` | Model injection and all positive-size boundaries fail early | Zero is the lower boundary for batch, sequence, and result sizes; `object()` isolates the required `predict` capability. |
| `test_rejects_malformed_model_scores` | Provider output has one finite scalar per candidate | One candidate paired with zero scores, a nested score, and NaN independently exercises count, shape, and finiteness validation. |
| `test_factory_accepts_an_injected_reranker` | The composition root keeps the reranker replaceable | Identity comparison proves the supplied adapter reaches `RAGService` unchanged without requiring model inference. |

## QASC contracts

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_config_rejects_values_outside_each_supported_domain` | Every QASC parameter respects its mathematical domain | Percentiles `-0.1` and `100.1` sit just outside both inclusive limits; `-1` is the first invalid integer for radii/gaps; `-0.1` is a small invalid negative float for decay/factor; `0` and `100` prove valid percentile boundaries. |
| `test_replacement_rejects_empty_mixed_or_noncontiguous_sentence_spans` | Sentence batches are nonempty, single-owner, zero-based, and contiguous | Empty, two-owner, and index sequence `(0,2)` isolate the three distinct contract failures with the smallest inputs. |
| `test_replacement_rejects_invalid_sentence_embedding_batches` | Embedding batches match sentence count and have one nonzero dimension | Two sentences are paired with one vector, two empty vectors, and mixed 2D/3D vectors to isolate count, emptiness, and rectangularity. |
| `test_rejected_replacement_preserves_the_previous_sentence_index` | Invalid re-indexing preserves valid QASC evidence | Radius `0` makes one sentence produce exactly one window; changing the embedder batch to empty triggers validation without changing stored query vectors. |
| `test_query_rejects_empty_text_nonpositive_limits_and_bad_vectors` | Query validation precedes scoring | Whitespace tests semantic emptiness, `top_k=0` the lower boundary, `()` an empty embedding, and 3D against stored 2D the smallest clear mismatch. |
| `test_deleting_the_last_document_allows_a_new_embedding_dimension` | A fully empty QASC index supports model migration | The same one-sentence document moves from 2D to 3D only after deletion, separating dimension reset from replacement behavior. |

## CI parameter change

The `spacy-adapter` job installs `.[spacy]` and runs only
`test_sentence_segmentation.py`. Python 3.9 is used because it is the project's
oldest supported interpreter and `pyproject.toml` already declares a dedicated
compatible spaCy range for it. The ten-minute timeout matches the core test job
and provides installation headroom without permitting a hung optional job.
