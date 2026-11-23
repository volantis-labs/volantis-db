# Volantis DB for Python

Python 3.11+ bindings for the Volantis DB C ABI. Platform wheels include the
native library, so a wheel installation does not require a separate Rust build.

```bash
python -m pip install volantis-db
```

```python
from pathlib import Path

import numpy as np
import volantis_db as ze

vectors = np.asarray([[0.0, 0.5, 1.0, 1.5]], dtype=np.float32)
with ze.open(Path("example-store")) as store:
    store.ingest([1], vectors, texts=["first document"])
    result = store.query(vector=vectors[0], text="first", k=1)
    print(result.hits[0].doc_id)
```

## Query filters and snippets

`store.query(filter=...)` accepts the same structured `ze.Filter` builders as
`scan` and `count`. The filter applies to every query leg before ranking and
fusion. When supplied together, `filter` and `eligible_ids` must both match.
Malformed filters, unknown attributes, and incompatible values raise
`ze.InvalidArgument` (`ZE_ERR_INVALID_ARGUMENT`).

```python
result = store.query(
    text="harbour cafe", filter=ze.Filter.range_("ts", gte=10), snippet_bytes=120
)
for hit in result.hits:
    if hit.snippet is not None:
        snippet = hit.snippet
        excerpt = snippet.text.encode("utf-8")
        print(snippet.text)
        for highlight in snippet.highlights:
            print(excerpt[highlight.start:highlight.end].decode("utf-8"))
```

`snippet_bytes` requires a lexical leg and a positive integer that fits the
native `size_t`. Invalid requests raise `ze.InvalidArgument`. The excerpt
starts at a matched token and can exceed the byte budget by up to three bytes
to finish a UTF-8 character. Only matches fully inside the excerpt are
highlighted. `truncated_start` and `truncated_end` mark the cuts.

`QueryHit.snippet` is a `Snippet`, or `None` when snippets were omitted or a
hybrid hit has no matching stored text. Each `SnippetHighlight` gives
half-open UTF-8 byte offsets `start` and `end` in the excerpt. The snippet
and each highlight also give absolute half-open `source_byte_start` and
`source_byte_end` offsets in the original stored text. Slice UTF-8 encoded
bytes to recover either range.

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

## Namespace tokenizers and transactions

Set `NamespaceSpec(tokenizer_profile=ze.TokenizerProfile.VOICE)` when opening a
namespace with `ze.open_namespace(root, name, spec)`. The profiles are
`TEXT_DEFAULT`, `CODE`, and `VOICE`. The equivalent strings are
`"text_default"`, `"code"`, and `"voice"`. Code preserves stopwords and skips
stemming. Voice normalizes number words so a query for `"25"` can match
`"twenty five"`. Omission uses the declared embedding epoch's tokenizer or the
text default. These profiles work for record-only namespaces and core builds.
The profile is persisted and a different profile on reopen raises
`ze.EpochMismatch`. An explicit profile must agree with any supplied epoch.

`ze.namespace_batch(root, participants)` commits changes across 2 to 128
existing namespaces under one root in a single durable transaction. Close
each participant's writer and snapshot views before calling. Each
`NamespaceMutation` supplies the namespace name and exact existing spec,
including its tokenizer. Operations run in this order for each namespace:
upserts, explicit deletes, then a structured `delete_where` filter. Use
`StoredDocument` or `UpsertDocument` for upserts. Conditions are checked
before that participant's deletes. A failed revision condition publishes none
of the batch and surfaces as `ze.Io` with the C ABI's revision failure message.

```python
generations = ze.namespace_batch(root, [
    ze.NamespaceMutation("notes", notes_spec, upserts=[ze.StoredDocument(7, text="draft")]),
    ze.NamespaceMutation("archive", archive_spec, delete_where=ze.Filter.eq("ts", 10)),
])
```

`ze.namespace_batch_live(root, participants)` takes the same fields on
`LiveNamespaceMutation` plus a `store=` keyword for each namespace's writable
handle in this process. The handles remain usable after success. Both
functions return one generation per participant in input order. Existing
readers keep their snapshots. An I/O error during commit can leave the
outcome indeterminate. Close any live participant handles and reopen the
namespaces to check the outcome before retrying.

### Cascades

`ze.declare_cascade(root, participants, declaration)` persists an ownership
rule. Pass `CascadeParticipant(name, spec)` entries with the exact existing
specs, then a `CascadeDeclaration(parent_index, child_index, attribute_id)`.
The indices refer to the participant list. The child attribute must be
`AttributeType.ID128`. Write its parent ID as an unsigned 128-bit Python
integer in an `AttributeValue`. Null values have no parent. Exact
redeclaration is idempotent, and a cycle raises `ze.CascadeCycle`
(`ZE_ERR_CASCADE_CYCLE`). The rule is used by `delete_cascade`.

```python
ze.declare_cascade(root, [
    ze.CascadeParticipant("notes", notes_spec),
    ze.CascadeParticipant("chunks", chunks_spec),
], ze.CascadeDeclaration(parent_index=0, child_index=1, attribute_id=1))

generations = ze.delete_cascade(root, [
    ze.CascadeDeleteParticipant("notes", notes_spec, deletes=[7]),
    ze.CascadeDeleteParticipant("chunks", chunks_spec),
])
```

`delete_cascade` atomically deletes explicit IDs and all their transitive
dependants. Pass 1 to 128 participants, including every namespace reachable
through the declared rules. Each `CascadeDeleteParticipant` takes optional
explicit IDs. The return value is a list of generations in participant order.
Close writers and snapshot views before declaring a rule or cascading a
delete. After an I/O error during commit, reopen before retrying. Cascade
deletion is logical and may leave deleted bytes in retained store files.

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

## ID128 attributes

Declare an `AttributeType.ID128` column for unsigned 128-bit identifiers.
Pass a Python `int` from 0 through `2**128 - 1` to `AttributeValue`. Reads
return the same integer type. A UUID can be converted with `UUID(...).int`.

```python
from uuid import UUID

import volantis_db as ze

parent = UUID("fedcba98-7654-3210-0123-456789abcdef").int
kind = ze.AttributeType.ID128
spec = ze.NamespaceSpec(attributes=(ze.AttributeDefinition(1, "parent", kind),))
with ze.open_namespace("example-store", "children", spec) as store:
    store.upsert([ze.StoredDocument(1, attributes=(ze.AttributeValue(1, kind, parent),))])
    page = store.scan(filter=ze.Filter.eq("parent", parent))
    print(page.documents[0].attributes)
```

Use the column name or ID with `Filter.eq`, `Filter.not_eq`, `Filter.in_`,
and `Filter.not_in`. Nullable columns also support `Filter.is_null`.
Malformed or out-of-range values raise `ze.InvalidArgument`. The engine
rejects range predicates, grouped counts and attribute ordering on ID128 columns.

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

## Schema readback and sealed segment merging

Declare namespace columns with `NamespaceSpec(attributes=...)` and read them
back with `store.schema()`. It returns a tuple of `AttributeDefinition` values
in native schema order, with each column's ID, UTF-8 name, type, and nullability.
The implicit `ts` column is excluded, and a schema with no user columns returns
an empty tuple. `AttributeType.ID128` is available for schema definitions.
Reading the schema of a closed store raises `ze.Closed`.

Call `store.merge()` during idle time on a writable store to consolidate small,
compatible sealed scan segments. Each atomic batch takes at most 16 segments
and 8 MiB of input files. The call holds the writer slot until it returns and
leaves active rows unsealed. It returns a `GenerationReport` with the last publication
generation, or the current generation when no merge is due. Pass
`cancel_token=token` to cancel cooperatively. Cancellation raises `ze.Cancelled`
and preserves any batches already published.

## Native library loading

The loader uses `VOLANTIS_DB_LIBRARY` when it is set. Otherwise, an installed
wheel loads its bundled library from `volantis_db/.dylibs`. Source-tree use
also recognizes the workspace's debug and release Cargo outputs for development.
An invalid override fails loudly and does not fall back to another library.

The arm64 wheel (the only published wheel, including text) requires macOS 13
or newer on Apple silicon. Other platforms are not part of this PyPI release.

## Graph support

The Python wheel ships without graph support. The `Store` methods
`enable_graph()`, `graph_apply()`, `graph_query()`, `cypher()`, `get_nodes()`,
`get_relationships()`, and `graph_resources()` raise
`volantis_db.UnsupportedBuild` (`ZE_ERR_GRAPH_UNSUPPORTED_BUILD`).

## Text API (macOS arm64)

Apple silicon wheels include the text feature and MLX's `mlx.metallib` beside
its native library. Other builds raise `ze.TextUnavailable` with
"text API is available on macOS arm64 only" from text entry points.
No separate MLX install or shader-library path is needed.

MLX keeps a process-global buffer cache, capped at 4 GiB by default during text
ingest. On an Apple M3 Max with 128 GiB, a full FiQA ingest (57,638 documents)
with the earlier FP16 clean71/Arctic default peaked at about 7.3 GB process footprint
(about 112 GB before the cap), with max RSS about 4.8 GB (ZE-528). Pass `mlx_cache_limit_bytes` as an unsigned integer keyword
argument to `open_text` or `open_model`.
The first explicit limit overrides `ZE_MLX_CACHE_LIMIT_BYTES` at initialization.
Zero disables caching. After initialization, omitting the limit reuses the
process setting. A different explicit limit returns `ZE_ERR_MLX_CACHE_LIMIT_CONFLICT`.

```python
import volantis_db as ze

with ze.open_model("text-store") as store:  # downloads the default model on first use
    store.ingest_text([7], ["the bronze airship"])
    for legs in ("dense", "lexical", "hybrid"):
        result = store.query_text("bronze airship", k=1, legs=legs, tier="exact")
        print(result.hits, result.diagnostics)
    print(store.health())
    store.maintain(wall_time_ns=10_000_000, bytes=1_000_000)
    store.delete_text([7])  # removes every chunk belonging to document 7
```

To download and verify a model from Hugging Face, call
`ze.open_model("text-store", repo=repo, revision=commit_sha,
manifest_sha256=manifest_sha256, cache_dir=None, offline=False, progress=None)`.
Omit all three pins (`ze.open_model("text-store")`) to use
`ze.DEFAULT_MODEL_REF`: `volantis-labs/volantis-db-v98-e5-small` at
`5909e5a5985f1c727766f7b50a713cfbe043256c`: the v98-8-r1 query tower and the `intfloat/multilingual-e5-small` document tower, 384 dimensions, FP16 MLX only, 347,613,712 bytes, about 348 MB. Dutch queries lose quality (Belebele nld nDCG@10 0.764 against 0.929). Stores embedded with the earlier clean71/Arctic default do not open under this default.
Explicit pins must be supplied together;
a partial set is an error.
The native resolver honors `ZE_MODEL_CACHE`, `ZE_MODEL_OFFLINE`, and
`ZE_MODEL_ENDPOINT`. The first open downloads and verifies the pinned model;
subsequent opens reuse the verified cache. `cache_dir` overrides `ZE_MODEL_CACHE`
and the default `~/Library/Caches/volantis-db/models`.
`progress(done, total)` runs synchronously with byte counts. Intermediate reports
require both at least 64 KiB and at least 100 ms since the last report; each
verified file also reports completion. Initial `(0, 0)` means fetching the
manifest. Callbacks must not reenter model resolution; callback exceptions
propagate after resolution.
`offline=True` (or `ZE_MODEL_OFFLINE=1`) prohibits network access and raises
`ze.NotFound` on a cache miss. `ze.open_text("text-store", "model.zem")` opens a
preinstalled bundle. See the [model card](https://huggingface.co/volantis-labs/volantis-db-v98-e5-small/blob/5909e5a5985f1c727766f7b50a713cfbe043256c/README.md)
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

### Embedded namespaces

Pass a pinned `EmbeddedModelRef(repo, revision, manifest_sha256)` as
`NamespaceSpec.embedded_model`. The model sets the vector dimensions. Use the
same pins when reopening the namespace. The resolver honors `ZE_MODEL_CACHE`,
`ZE_MODEL_OFFLINE`, and `ZE_MODEL_ENDPOINT`. Opening an embedded namespace and
calling the chunk query methods require a macOS arm64 text build. Other builds
raise `ze.TextUnavailable`.

```python
import volantis_db as ze

model = ze.EmbeddedModelRef(**ze.DEFAULT_MODEL_REF)
spec = ze.NamespaceSpec(
    attributes=(ze.AttributeDefinition(1, "meeting", ze.AttributeType.U64),),
    embedded_model=model,
)
meeting = ze.AttributeValue(1, ze.AttributeType.U64, 42)
with ze.open_namespace("db", "notes", spec) as notes:
    notes.upsert([
        ze.StoredDocument(7, text="The fixtures arrive on Tuesday.", attributes=(meeting,)),
        ze.StoredDocument(9, text="The shipment is due Tuesday.", attributes=(meeting,)),
    ])
    filter = ze.Filter.eq("meeting", 42)
    hits = notes.query_embedded("delivery date", leg="hybrid", k=5, filter=filter)
    neighbours = notes.nearest_chunks(ze.NeighbourSource(7, chunk=0), k=5)
    groups = notes.nearest_chunks_batch([7, 9, 9], threshold=-0.25, filter=filter)
```

`upsert` accepts one or more source documents with text, optional metadata and
schema attributes. The model embeds chunks. Supplying a vector raises
`ze.InvalidArgument`. Replacing or deleting the parent row retires all of its
chunks.

`query_embedded` returns a tuple of `EmbeddedHit` values. Use `leg="dense"`
for exact vector search (the default), `"lexical"` for BM25, or `"hybrid"` for
reciprocal rank fusion. The limit `k` is 1–4096 and defaults to 10. Each hit
contains the parent ID, revision, timestamp and attributes, plus its score,
chunk text, zero-based chunk number and half-open UTF-8 byte offsets. Slice
`source.encode("utf-8")[hit.byte_start:hit.byte_end]` for the original bytes.

`nearest_chunks` uses persisted vectors. Pass a document ID to use all its
chunks or a `NeighbourSource` to select one. Each `ChunkNeighbour` contains
`doc_id`, `chunk`, and the negative squared L2 distance to the closest selected
source chunk. Zero is an exact match. Ties sort by document ID, then chunk.
Source chunks are excluded unless `include_source=True`. Set `k` to 1–4096
to cap the results. A threshold alone returns every score strictly above it.
Omitting both returns up to 10 hits. `nearest_chunks_batch` returns one tuple
per source in input order, preserving duplicate sources and empty groups.
An unknown source ID or chunk raises `ze.NotFound` for the entire call.

Both query types accept a structured `filter`, a half-open `timestamp_range`,
and `eligible_ids` to restrict candidates before ranking. An empty eligible
list selects no candidates. Use either `cancel_token` or a relative
`deadline_ns`. Stopped queries raise a typed error without returning partial
results. Querying a regular namespace with either method raises
`ze.MissingEmbedder`.

## Development

The package is built from the repository because the wheel build compiles
`crates/volantis-db-ffi` with the locked Rust 1.93 dependency graph:

```bash
python -m pip wheel --no-deps ./bindings/python -w wheelhouse
```

Run all binding checks on macOS arm64, outside a process sandbox (MLX uses Metal):

```bash
cargo build --locked -p volantis-db-ffi --features text
VOLANTIS_DB_LIBRARY="$PWD/target/debug/libvolantis_db_ffi.dylib" \
  python -m pytest bindings/python/tests
```

The text tests use the committed synthetic bundle
`crates/volantis-db-text/tests/fixtures/ze115/gte.zem` and a local HTTP server
for model download, verified cache reuse, and offline behavior. The built-in default is pinned to the published Hugging Face model. The ignored
Rust test `real_default_model_downloads_and_answers_queries` verifies a real
download and dense, lexical and hybrid retrieval outside the sandbox.

Build and verify the installed wheel in a fresh venv and temporary directory:

```bash
python -m pip wheel --no-deps ./bindings/python -w wheelhouse
python bindings/python/tests/verify_wheel.py wheelhouse/volantis_db-*.whl
```

The verifier clears source-library overrides, copies the synthetic fixture into
the temporary directory, and tests ingest plus all three query legs using only
the installed package and its colocated shader library.

## Releasing

Manually dispatching `python-release.yml` with release tag `v<version>` builds
and verifies the macOS arm64 wheel, then uploads it with PyPI Trusted
Publishing. Configure
the PyPI publisher with owner `volantis-labs`, repository `volantis-db`, workflow
`python-release.yml`, and environment `pypi`.

### Prepare lexical queries after open

Call `store.warm_lexical()` before exact or `last_as_prefix=True` queries to
prepare the lexical assembly and prefix vocabulary, including record-only
namespaces. Repeated calls reuse preparation; mutations invalidate it.
Optional `cancel_token` and `deadline_ns` controls are mutually exclusive.
A cancelled call may retain a complete assembly and can be retried.

### Required API check

The binding suite checks its public API against the
[E17 API table](../fixtures/required_api_v1.json). Graph and text rows apply to
builds that ship those capabilities. Python 0.1.0 has the owner-approved graph
exclusion. The temporary E17 gap list is empty and CI rejects new entries.
