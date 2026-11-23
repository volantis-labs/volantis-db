# Changelog

All notable changes to Volantis DB are recorded here. Versions follow
semantic versioning, with the 0.x rule that a new public surface is a minor
release and a compatible correction is a patch release.

## 0.1.0 - Unreleased

First release of Volantis DB.

### Added

- In-process store with vector, lexical (BM25) and hybrid search in one
  document ID space, write-ahead-log recovery, immutable searchable
  generations, typed filters, document revisions, cancellation, deadlines,
  retention, physical purge, memory budgets, health and query diagnostics.
- Namespaces, snapshots and maintenance.
- Property graph storage with an openCypher 9 subset and search extensions
  (`volantis-db-cypher`, graph-enabled C ABI and bindings). Text written on
  a graph node is in the full-text index from the same write.
- Local text embedding and retrieval on Apple silicon (`volantis-db-text`):
  pinned default model `volantis-labs/volantis-db-v98-e5-small` at
  `5909e5a5985f1c727766f7b50a713cfbe043256c` (query tower v98-8-r1,
  `query: `, 96 tokens; document tower `intfloat/multilingual-e5-small`,
  `passage: `, 512 tokens; 384 dimensions; FP16 on MLX only; a
  347,613,712-byte download, about 348 MB), verified cached download,
  offline mode, ingest, dense, lexical and hybrid queries, and an MLX cache
  limit as an open option. Target languages: en, es, fr, de, pt, it, ru, zh,
  ja, ko, hi, ar, id, vi. Dutch queries lose quality: Belebele nld nDCG@10
  0.764 against 0.929 with e5 as the query tower. Stores embedded with the
  earlier default (`volantis-labs/volantis-db-clean71-arctic-m-v2`) do not
  open under the new default, because the embedding epoch differs; rebuild
  them or open them with the old pin.
- Embedded namespaces in the unified `Store`: text upserts are chunked and
  embedded by the engine, with 128-bit ids, revisions, typed attributes,
  filters on every query leg, cascade deletes, atomic batches, backup and
  Cypher `ze.vector_search` / `ze.hybrid_search` with query text.
- Background embedding for embedded namespaces. The schedule
  `embeddingSchedule` (Node; C `ZeEmbeddedNamespaceSpecV4`; Rust
  `DocumentSchedule`) is `immediate` (default: embed inside the upsert),
  `background` (commit text, attributes and the lexical index at once; a
  worker of the writable handle embeds later) or `paused` (`background` with
  the worker stopped at open). `pauseEmbedding`, `resumeEmbedding` and
  `embeddingStatus` (paused, in flight, running, pending count, last error)
  control the worker; `queryEmbeddedReport` returns the hits with the store
  generation, the eligible pending-row count and a state per hit. A pending
  row is a lexical hit at once and a dense hit after it is embedded. Pending
  rows survive close and reopen and are embedded later. C ABI:
  `ze_namespace_embedding_pause`, `ze_namespace_embedding_resume`,
  `ze_namespace_embedding_status`, `ze_query_embedded_report`.
- Model cache folder and offline switch for embedded namespace opens: Node
  `openNamespace` / `openNamespaceAsync` options and `namespaceBatch`
  participants take `modelCacheDir` and `offline`, with the same meaning as
  `TextStore.openModel`; C `ZeEmbeddedNamespaceSpecV3`; Rust
  `ModelOpenOptions`. An explicit folder wins over `ZE_MODEL_CACHE`. With
  `offline` (or `ZE_MODEL_OFFLINE=1`) a cache miss returns
  `ZE_ERR_NOT_FOUND` and makes no network connection.
- Embedded namespaces store an upsert with omitted or empty text as a row
  with zero chunks and no vector, with no model call. It is readable and
  filterable, never a dense or neighbour hit, and never pending. A space is
  real text.
- `volantis-db-text` and the text-enabled C ABI build for Intel macOS and
  Windows x64 without MLX or Core ML; MLX and Core ML are linked on macOS
  arm64 only. Intel macOS runs both BERT towers of
  the default model on a pure-Rust fp32 CPU runtime (`cpu-fp32`) from the
  same FP16 bundle and with the same store epoch, so stores move between
  Apple silicon and Intel Macs; the darwin-x64 Node prebuild carries text.
  On Windows every text-store open and embedded namespace open returns
  `ZE_ERR_UNSUPPORTED` (Rust: `RuntimeError::UnsupportedBackend`) before any
  download or store change.
- Language APIs over one native engine: Rust crates, a versioned C ABI
  (`volantis_db.h`), Python (`volantis-db`), Node.js and TypeScript
  (`volantis-db`), and Swift (`VolantisDB` package).
- Platforms: macOS arm64 and x86_64; Windows x64 for the engine, C ABI and
  Node package. Model-backed text runs on macOS only: MLX on arm64, the CPU
  runtime on x86_64.

### Known limitations

- Windows x64 has no text runtime. The Windows Node prebuild and C ABI
  return `ZE_ERR_UNSUPPORTED` for every text-store and embedded namespace
  open.
- The Python wheel on PyPI is macOS arm64 only. A source build on Intel
  macOS has no text API ("text API is available on macOS arm64 only").
- The C SDK archives and the graph XCFramework are macOS arm64. The
  universal `VolantisDB` XCFramework (arm64 and x86_64) is text-free, so
  Swift and C consumers on Intel Macs have no text API. On Intel Macs, text
  ships in the Node package and in Rust source builds.
- The Python and Swift bindings do not expose `embeddingSchedule`, the
  embedding worker controls, `modelCacheDir` or `offline` yet.
- Stores embedded with the earlier default model
  (`volantis-labs/volantis-db-clean71-arctic-m-v2`) open only with that old
  pin. Rebuild them to use the new default.
- A sealed segment that holds a pending or empty-text row has no ANN graph
  index and is searched by exact scan until a merge makes every row
  vector-backed (empty-text rows keep their segment on exact scan). Results
  are correct; the cost is linear in that segment.
