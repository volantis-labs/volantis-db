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
- Language APIs over one native engine: Rust crates, a versioned C ABI
  (`volantis_db.h`), Python (`volantis-db`), Node.js and TypeScript
  (`volantis-db`), and Swift (`VolantisDB` package).
- Platforms: macOS arm64 and x86_64; Windows x64 for the engine, C ABI and
  Node package.
