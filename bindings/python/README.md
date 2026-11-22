# Zeppelin Embed for Python

Python 3.11+ bindings for the Zeppelin Embed C ABI. Platform wheels include the
native library, so a wheel installation does not require a separate Rust build.

```bash
python -m pip install zeppelin-embed
```

```python
from pathlib import Path

import numpy as np
import zeppelin_embed as ze

vectors = np.asarray([[0.0, 0.5, 1.0, 1.5]], dtype=np.float32)
with ze.open(Path("example-store")) as store:
    store.ingest([1], vectors, texts=["first document"])
    result = store.query(vector=vectors[0], text="first", k=1)
    print(result.hits[0].doc_id)
```

## Conditional writes and filtered deletion

Pass an `UpsertDocument` to `store.upsert()` or a `DeleteTarget` to
`store.delete()` to check the target's live revision before writing.
`expected_revision` is an unsigned 64-bit integer for an exact match, or
`None` to require that the id has no live document. Plain `StoredDocument`
entries and bare ids carry no revision condition and can share a batch
with conditional entries.

```python
with ze.open_namespace("example-store", "drafts", ze.NamespaceSpec()) as store:
    store.upsert([ze.StoredDocument(7, revision=1, text="draft")])
    store.upsert([
        ze.UpsertDocument(7, revision=2, expected_revision=1, text="edited draft"),
        ze.UpsertDocument(8, expected_revision=None, timestamp=10, text="new draft"),
    ])
    store.delete([ze.DeleteTarget(7, expected_revision=2)])
    result = store.delete_where(ze.Filter.eq("ts", 10))
    print(result.deleted, result.generation)
```

Both conditional writes return a `MutationReport`. The writer checks every
condition against the latest committed state before applying any entry. On
`ze.RevisionConflict` (`ZE_ERR_REVISION_CONFLICT`), the entire batch leaves
the store unchanged. The exception's `conflict` gives the first failed
entry's zero-based `index`, integer `doc_id`, `expected_revision`, and
`current_revision` (`None` when no live document exists). The normal upsert
revision rules still apply. Reusing a revision is idempotent, and recreating a
deleted id via upsert requires a revision above the deleted revision.

`store.delete_where(filter)` takes a required structured `Filter`, using the
same builders as `scan` and `count`. It atomically deletes every matching
live document and physically removes the matching documents' bytes from
store files before returning. If interrupted, the next writable open
finishes any pending removal. `DeleteWhereReport` gives the number
`deleted` and the resulting `generation`. With zero matches, the generation
is unchanged. A malformed filter raises `ze.InvalidArgument`. These
operations use core store handles. Text stores use `delete_text()` for
document deletion.

## Store verification

Call `ze.verify(path)` with a string or a string-valued path-like object. Pass
the store directory itself, such as `Path(root) / name` for a namespace.
Verification reads the manifest, its referenced segments, the write-ahead log
(WAL), and any pending purge intent. It opens no store handle, takes no store
lock, and modifies no files. Run it before reopening after an unclean shutdown.
A concurrent writer can cause the report to reflect a write in progress.

```python
report = ze.verify(Path("example-store"))
if report.ok:
    print(f"Checked {report.segments_checked} segments and "
          f"{report.wal_records_checked} WAL records")
for finding in report.findings:
    print(finding.kind.name, finding.file, finding.offset, finding.detail)
```

`VerifyReport.generation` is the decoded manifest generation, or zero when
there is no decoded manifest. `segments_checked` counts the segments referenced
by the manifest. `wal_records_checked` counts WAL records that passed checksum
and sequence validation. `ok` is true exactly when the `findings` tuple is empty.
Each `VerifyFinding` has a `VerifyFindingKind`, a relative file name, an optional
byte offset (`None` when unavailable), and decoder detail.

Damage is returned in the report. An invalid path argument raises
`ze.InvalidArgument` (`ZE_ERR_INVALID_ARGUMENT`), a missing path raises
`ze.NotFound` (`ZE_ERR_NOT_FOUND`), and a non-directory or unlistable path
raises `ze.Io` (`ZE_ERR_IO`). Graph-object checks require a graph-enabled
native library.

## Attribute-ordered scans and grouped counts

Pass an `AttributeScanOrder` to `scan()` or `iter_documents()` to sort by a
declared `U64`, `I64`, or `F64` attribute. Both directions break ties by
ascending document ID and put missing values and NaNs last. Floating-point
values sort numerically, with `-0.0` and `0.0` treated as equal.
The existing `"storage"`, `"timestamp_ascending"`, and
`"timestamp_descending"` orders are also available.

```python
order = ze.AttributeScanOrder(attribute_id=1, direction="descending")
page = store.scan(order=order, limit=100)
if page.cursor is not None:
    next_page = store.scan(order=order, cursor=page.cursor, limit=100)

for document in store.iter_documents(order=order):
    print(document.doc_id)
```

Reuse the same order, attribute, and direction when continuing a scan.
A different order or an undeclared or unsupported ordering attribute raises
`ze.InvalidArgument` (`ZE_ERR_INVALID_ARGUMENT`). Use a timestamp order for
attribute ID 0. A write between pages makes the cursor stale and raises
`ze.ScanStale` (`ZE_ERR_SCAN_STALE`). Restart without a cursor.

Pass `group_by` to `count()` to group matching live documents by a declared
`U64`, `I64`, `DICTIONARY_STRING`, or `RAW_STRING` attribute. The optional
structured `filter` and half-open `timestamp_range` apply to the whole count.

```python
result = store.count(group_by=ze.CountGroupBy(attribute_id=2, limit=100))
for group in result.groups:
    print(group.value, group.count)
print(result.count, result.missing_count, result.generation)
```

`GroupedCountResult` gives the total `count`, the pinned `generation`, and a
tuple of `CountGroup` values sorted by ascending numeric or UTF-8 byte order.
Each group holds an `int` or `str` value and its count. Documents with a null
grouping attribute contribute to `missing_count`. The group counts plus
`missing_count` equal `count`. The limit defaults to 1,024 and accepts
1 through 65,536 distinct values. Exceeding it raises `ze.BudgetExceeded`
(`ZE_ERR_BUDGET_EXCEEDED`) for the entire call. Unknown attribute IDs,
unsupported types, and invalid limits raise `ze.InvalidArgument`.
Omitting `group_by` returns the scalar `CountResult`.

## Native library loading

The loader uses `ZEPPELIN_EMBED_LIBRARY` when it is set. Otherwise, an installed
wheel loads its bundled library from `zeppelin_embed/.dylibs`. Source-tree use
also recognizes the workspace's debug and release Cargo outputs for development.
An invalid override fails loudly and does not fall back to another library.

The arm64 wheel (the only published wheel, including text) requires macOS 13
or newer on Apple silicon. Other platforms are not part of this PyPI release.

## Graph support

The Python wheel ships without graph support. The `Store` methods
`enable_graph()`, `graph_apply()`, `graph_query()`, `cypher()`, `get_nodes()`,
`get_relationships()`, and `graph_resources()` raise
`zeppelin_embed.UnsupportedBuild` (`ZE_ERR_GRAPH_UNSUPPORTED_BUILD`).

## Text API (macOS arm64)

Apple silicon wheels include the text feature and MLX's `mlx.metallib` beside
its native library. Other builds raise `ze.TextUnavailable` with
"text API is available on macOS arm64 only" from text entry points.
No separate MLX install or shader-library path is needed.

MLX keeps a process-global buffer cache, capped at 4 GiB by default during text
ingest. On an Apple M3 Max with 128 GiB, a full FiQA ingest (57,638 documents)
with the release FP16 default model peaked at about 7.3 GB process footprint
(about 112 GB before the cap), with max RSS about 4.8 GB (ZE-528). Pass `mlx_cache_limit_bytes` as an unsigned integer keyword
argument to `open_text` or `open_model`.
The first explicit limit overrides `ZE_MLX_CACHE_LIMIT_BYTES` at initialization.
Zero disables caching. After initialization, omitting the limit reuses the
process setting. A different explicit limit returns `ZE_ERR_MLX_CACHE_LIMIT_CONFLICT`.

```python
import zeppelin_embed as ze

with ze.open_model("text-store") as store:  # downloads the default model on first use
    store.ingest_text([7], ["the bronze zeppelin"])
    for legs in ("dense", "lexical", "hybrid"):
        result = store.query_text("bronze zeppelin", k=1, legs=legs, tier="exact")
        print(result.hits, result.diagnostics)
    print(store.health())
    store.maintain(wall_time_ns=10_000_000, bytes=1_000_000)
    store.delete_text([7])  # removes every chunk belonging to document 7
```

To download and verify a model from Hugging Face, call
`ze.open_model("text-store", repo=repo, revision=commit_sha,
manifest_sha256=manifest_sha256, cache_dir=None, offline=False, progress=None)`.
Omit all three pins (`ze.open_model("text-store")`) to use
`ze.DEFAULT_MODEL_REF`: `volantis-labs/volantis-db-clean71-arctic-m-v2` at
`d64b817c4021a3f1b9bded3152d9afdf9ac813e7`. Explicit pins must be supplied together;
a partial set is an error.
The native resolver honors `ZE_MODEL_CACHE`, `ZE_MODEL_OFFLINE`, and
`ZE_MODEL_ENDPOINT`. The first open downloads and verifies the pinned model;
subsequent opens reuse the verified cache. `cache_dir` overrides `ZE_MODEL_CACHE`
and the default `~/Library/Caches/zeppelin-embed/models`.
`progress(done, total)` runs synchronously with byte counts. Intermediate reports
require both at least 64 KiB and at least 100 ms since the last report; each
verified file also reports completion. Initial `(0, 0)` means fetching the
manifest. Callbacks must not reenter model resolution; callback exceptions
propagate after resolution.
`offline=True` (or `ZE_MODEL_OFFLINE=1`) prohibits network access and raises
`ze.NotFound` on a cache miss. `ze.open_text("text-store", "model.zem")` opens a
preinstalled bundle. See the [model card](https://huggingface.co/volantis-labs/volantis-db-clean71-arctic-m-v2/blob/d64b817c4021a3f1b9bded3152d9afdf9ac813e7/README.md)
and [root text API guide](../../README.md#text-api-macos-apple-silicon) for model
limits, download size, and measured performance.

`ingest_text` and `query_text` accept `cancel_token=ze.CancelToken()` or
`deadline_ns` (a relative monotonic budget), mutually exclusive. Cancellation
raises `ze.Cancelled`; expiry raises `ze.Timeout`. Ingest may leave acknowledged
partial writes. Its report includes generation, documents, chunks, tokens, batches,
and worker completion. Set `chunk_max` and `chunk_overlap` to split long documents
into token windows. Queries return stored text, core diagnostics, backend facts,
and stage timings when the native runtime enables them. Text `maintain` executes
one budgeted native maintenance call.

Explicit `store.observed_compute_units()` loads the CoreML compute plan on demand and returns planned placement counts (not an execution trace), or `None` without a CoreML query model.

## Development

The package is built from the repository because the wheel build compiles
`crates/zeppelin-embed-ffi` with the locked Rust 1.93 dependency graph:

```bash
python -m pip wheel --no-deps ./bindings/python -w wheelhouse
```

Run all binding checks on macOS arm64, outside a process sandbox (MLX uses Metal):

```bash
cargo build --locked -p zeppelin-embed-ffi --features text
ZEPPELIN_EMBED_LIBRARY="$PWD/target/debug/libzeppelin_embed_ffi.dylib" \
  python -m pytest bindings/python/tests
```

The text tests use the committed synthetic bundle
`crates/zeppelin-embed-text/tests/fixtures/ze115/gte.zem` and a local HTTP server
for model download, verified cache reuse, and offline behavior. The built-in default is pinned to the published Hugging Face model. The ignored
Rust test `real_default_model_downloads_and_answers_queries` verifies a real
download and dense, lexical and hybrid retrieval outside the sandbox.

Build and verify the installed wheel in a fresh venv and temporary directory:

```bash
python -m pip wheel --no-deps ./bindings/python -w wheelhouse
python bindings/python/tests/verify_wheel.py wheelhouse/zeppelin_embed-*.whl
```

The verifier clears source-library overrides, copies the synthetic fixture into
the temporary directory, and tests ingest plus all three query legs using only
the installed package and its colocated shader library.

## Releasing

Manually dispatching `python-release.yml` with release tag `v<version>` builds
and verifies the macOS arm64 wheel, then uploads it with PyPI Trusted
Publishing. Configure
the PyPI publisher with owner `zepdb`, repository `zeppelin-embed`, workflow
`python-release.yml`, and environment `pypi`.

### Prepare lexical queries after open

Call `store.warm_lexical()` before exact or `last_as_prefix=True` queries to
prepare the lexical assembly and prefix vocabulary, including record-only
namespaces. Repeated calls reuse preparation; mutations invalidate it.
Optional `cancel_token` and `deadline_ns` controls are mutually exclusive.
A cancelled call may retain a complete assembly and can be retried.
