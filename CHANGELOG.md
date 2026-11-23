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
  pinned default model
  `volantis-labs/volantis-db-clean71-arctic-m-v2`, verified cached
  download, offline mode, ingest, dense, lexical and hybrid queries, and an
  MLX cache limit as an open option.
- Embedded namespaces in the unified `Store`: text upserts are chunked and
  embedded by the engine, with 128-bit ids, revisions, typed attributes,
  filters on every query leg, cascade deletes, atomic batches, backup and
  Cypher `ze.vector_search` / `ze.hybrid_search` with query text.
- Language APIs over one native engine: Rust crates, a versioned C ABI
  (`volantis_db.h`), Python (`volantis-db`), Node.js and TypeScript
  (`volantis-db`), and Swift (`VolantisDB` package).
- Platforms: macOS arm64 and x86_64; Windows x64 for the engine, C ABI and
  Node package.
