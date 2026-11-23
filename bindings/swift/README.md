# ZeppelinEmbed for Swift

`ZeppelinStore` is the async Swift actor over the frozen Zeppelin Embed C ABI.
The package targets macOS 14 or newer on Apple silicon and Intel. Release
consumers use the checksum-pinned XCFramework binary target in the root
`Package.swift`:

```swift
.package(url: "https://github.com/zepdb/zeppelin-embed", from: "0.8.0")
```

Local source builds set `ZE_USE_LOCAL_FFI=1` after building the release FFI
archive. Release validation can set `ZE_USE_LOCAL_XCFRAMEWORK=1` to test the
artifact in `target/xcframework` before uploading it.

To include the deterministic mid-query cancellation test in a local core run,
build the test-support archive and enable the matching Swift test flag:

```sh
cargo build -p zeppelin-embed-ffi --release --features swift-query-test-support
ZE_USE_LOCAL_FFI=1 swift test --disable-sandbox --jobs 4 -Xswiftc -DZE_SWIFT_QUERY_TEST_SUPPORT
```

The cancellation trigger runs after the native exact scan scores its first
row. Suites linked to release artifacts skip this test.

Store databases in Application Support. `OpenOptions.excludeFromBackup`
defaults to `true`, which applies `NSURLIsExcludedFromBackupKey` to the store
directory. Do not put a store in iCloud Drive, Dropbox, or another synchronized
folder: the engine owns its rename, lock, and durability protocol.

The host owns maintenance scheduling. The package does not register or schedule
background work.

### Write documents with attributes

Use `try await store.upsert(documents)` to store `IngestDocument.attributes`.
`ZeppelinStore.ingest(_:)` accepts attribute-free batches. If any document
carries attributes, ingest throws `ZeppelinIngestError.attributesRequireUpsert`
before writing any document in the batch.

### Scan in attribute order

Pass an `AttributeScanOrder` to order a page by a declared `u64`, `i64`, or
`f64` attribute. A scan accepts 1 through 1,048,576 documents per page
and defaults to 1,024.

```swift
let order = AttributeScanOrder(attributeID: 1, direction: .descending)
let page = try await store.scan(limit: 50, order: order)
if let cursor = page.next {
    let next = try await store.scan(after: cursor, limit: 50, order: order)
    print(next.documents)
}
```

Equal values sort by ascending document ID. Floating-point values sort
numerically with signed zeros treated as equal. Missing and NaN values sort
last in both directions. An unsupported or undeclared attribute, or a cursor
used with a different order, attribute, or direction, throws
`ZeppelinError.invalidArgument`. If a write invalidates a cursor, the next
page throws `ZeppelinError.scanStale`. Restart the scan without a cursor.

### Count by attribute

Use `count(groupBy:filter:timestampRange:)` to group matching live documents
by a declared `u64`, `i64`, `dictionaryString`, or `rawString` attribute.
The filter and half-open timestamp range are optional.

```swift
let result = try await store.count(
    groupBy: CountGroupBy(attributeID: 2, limit: 100),
    timestampRange: 100..<200
)
for group in result.groups {
    print(group.value, group.count)
}
```

Group values are returned as `AttributeValue` in ascending order, using
numeric order for integers and byte order for strings. The result reports
one `generation`, and `count` includes both the group counts and `missingCount`
for documents with a null attribute. The group limit defaults to 1,024 and
accepts 1 through 65,536. Exceeding the limit throws
`ZeppelinError.budgetExceeded`. An unsupported attribute or an out-of-range
limit throws `ZeppelinError.invalidArgument`.

### Filter structured queries and return snippets

`ZeppelinStore.query` accepts the same typed `Filter` used by scans. A
`timestampRange` includes its lower bound and excludes its upper bound.
The engine intersects both constraints with `eligibleIds` before ranking
and fusion. An empty ID list matches no documents.

```swift
let result = try await store.query(
    text: "harbour cafe",
    filter: .eq(1, .string("note")),
    timestampRange: 10..<20,
    options: QueryOptions(k: 10, snippetBytes: 120)
)
for hit in result.hits {
    guard let snippet = hit.snippet else { continue }
    let bytes = Array(snippet.text.utf8)
    let matches = snippet.highlights.map {
        String(decoding: bytes[$0.byteRange], as: UTF8.self)
    }
    print(snippet.text, matches)
}
```

The example uses a namespace whose attribute ID 1 stores a string. An unknown
attribute or a filter value of the wrong type throws `ZeppelinError.invalidArgument`.

A positive `snippetBytes` requests an excerpt for each hit whose text matched.
The engine uses the query analyzer, including prefix expansion, to find
highlights. Only matches fully inside the excerpt are highlighted. The budget
is in UTF-8 bytes and the excerpt can extend by up to three bytes to finish a
Unicode scalar. `truncatedStart` and `truncatedEnd` indicate where source text
was cut. Snippets require a lexical leg. A vector-only request with snippets
or a nonpositive byte budget throws `ZeppelinError.invalidArgument`.
Omit `snippetBytes` to skip excerpt generation.

`SnippetHighlight.byteRange` addresses the excerpt's UTF-8 bytes.
`Snippet.sourceByteRange` and `SnippetHighlight.sourceByteRange` address the
stored document's UTF-8 bytes. All ranges are half-open. A hybrid hit with no
matching text has a nil `snippet`.

### Read a store schema

```swift
let attributes = try await store.schema()
```

`schema()` returns `[AttributeDefinition]` in persisted schema order. Each
entry contains its ID, name, type (including `.id128`), and nullability.
The list contains user attributes and omits the built-in timestamp column.
A store without user attributes returns an empty list. A closed store throws
`ZeppelinError.closed`.

### Merge sealed segments

```swift
let report = try await store.merge()
print(report.generation)
```

Call `merge()` during application idle time on a writable store. It merges
compatible small sealed scan segments in atomic batches of up to 16 segments
and 8 MiB of input files each. Large and graph segments remain separate.
Active writes and the WAL stay unchanged. Call `seal()` first to include
active writes. The `GenerationReport` contains the final generation, which
stays the same if no compatible batch fits.

Pass a `ZeppelinCancellationToken` with `merge(cancellationToken:)` to request
cooperative cancellation. A cancelled call throws `ZeppelinError.cancelled`.
Batches published before cancellation remain committed.

### Prepare lexical queries after open

Use `try await store.warmLexical()` before exact or `lastAsPrefix` queries.
It prepares the lexical assembly and prefix vocabulary, including record-only
namespaces. Repeated calls reuse preparation; mutations invalidate it.
Optional `cancellationToken` and `deadlineNanoseconds` controls are mutually
exclusive. A cancelled call may retain a complete assembly and can be retried.

### Conditional writes and filter deletion

Use `upsert(conditional:)` and `delete(conditional:)` to check the live
revisions of documents before writing. Each entry accepts `.exactly(revision)`
to require a particular live revision, `.absent` to require a missing or
deleted document, or `nil` to skip the check for that entry.

```swift
// replacement is an IngestDocument with a higher revision than the live one.
let updated = try await store.upsert(conditional: [
    ConditionalUpsertDocument(document: replacement, expectedRevision: .exactly(previousRevision))
])
let deleted = try await store.delete(conditional: [
    DeleteTarget(id: replacement.id, expectedRevision: .exactly(replacement.revision))
])
```

The writer checks all conditions against the same committed state before
applying the batch. If any condition fails, the whole batch stays unchanged.
Catch `RevisionConflict` to read the `.revisionConflict` code, the zero-based
batch `index`, the document `id`, its `expectedRevision`, and its
`currentRevision` (nil when missing or deleted). Successful writes return
`MutationReport` with the sequence and generation. Upserts still follow the
usual revision rules, including the revision floor left by a delete.

`deleteWhere(_:)` takes a typed `Filter`. It atomically deletes all live
matches and physically removes their bytes from the store before returning.
If interrupted, the next writable open finishes the removal. The report
contains `deleted` and `generation`. Zero matches leave the generation
unchanged. An invalid filter throws `ZeppelinError.invalidArgument`, and a
read-only handle throws `ZeppelinError.accessMode`. The call runs to completion
without a cancellation token.

```swift
// Attribute 1 is a string column in this store's schema.
let report = try await store.deleteWhere(.eq(1, .string("expired")))
print(report.deleted, report.generation)
```

### Verify a store

Use `ZeppelinStore.verify(at:)` with a local file URL for the store directory.
For a namespace, pass its directory under the root. Verification is read-only,
requires no open handle, and can run after an unclean shutdown. It checks
the manifest, referenced segments, WAL replay, and any pending purge intent.
The graph-enabled build also checks graph objects and inventory. A concurrent
writer can cause verification to report a write in progress.

```swift
let report = try await ZeppelinStore.verify(at: root.appendingPathComponent("notes"))
for finding in report.findings {
    print(finding.kind, finding.file, finding.offset as Any, finding.detail)
}
```

`VerifyReport` includes the manifest `generation` (zero if unavailable),
`segmentsChecked`, and `walRecordsChecked`. Its `ok` property is true when
`findings` is empty. Each `VerifyFinding` has a typed `VerifyFindingKind`, a
relative file name, an optional byte offset, and decoder detail.

If the walk cannot start, the call throws `ZeppelinError`. A missing directory
throws `.notFound`, a non-directory or unlistable path throws `.io`, and an
invalid path or a URL that does not refer to the local filesystem throws
`.invalidArgument`.

### Graph

Graph methods use the same `ZeppelinStore` handle: `enableGraph`, `graphApply`,
`graphQuery`, `cypher`, `getNodes`, `getRelationships`, and `graphResources`.
Call `enableGraph()` before graph queries on a store created with ordinary
`open(at:options:)`. `GraphNodeID` is an alias for `DocumentID`. A document node created with
`GraphNodeImage(id:timestamp:attributes:metadata:)` uses the appended V2 ABI;
create requires a caller ID, and put uses its existing node ID.
Text on a graph node is searchable with a lexical query or Cypher `ze.text_search`
as soon as the write returns, including when the store assigns the node ID.
A later graph put replaces the indexed text, and deleting the node removes it.
Vector and hybrid queries use document vectors.

To create a graph-enabled store with immutable relationship rules, use the
creation-only `open(at:relationshipTypes:maxResidentBytes:readerDrainTimeoutMilliseconds:)`
overload with a new path.

```swift
let store = try await ZeppelinStore.open(at: path, relationshipTypes: [
    GraphRelationshipType(type: "DEPENDS_ON", onDelete: .cascade),
    GraphRelationshipType(type: "REFERENCES", onDelete: .restrict),
])

try await store.graphSetMaintenancePolicy(GraphMaintenancePolicy(autoReclaim: false))
var report = try await store.graphMaintain()
while !report.cycleComplete {
    report = try await store.graphMaintain()
}
print(report.generation, report.removedBytes)
```

Rules apply to incoming edges from child to parent. `.restrict` refuses deletion
while a surviving child references the parent. `.cascade` deletes dependent
children transitively in the same mutation. The rules apply to batch mutations
and Cypher deletion. The creation overload accepts up to 16,384 unique rules
and 8 MiB of encoded declarations. Its resident-byte limit is 1 byte through
256 MiB and defaults to 256 MiB. The reader drain timeout defaults to 5,000 ms.
Use ordinary `open(at:options:)` to reopen the store. Its rules persist across
reopens.

`graphSetMaintenancePolicy(_:)` configures the current writer. The default
policy enables automatic maintenance with a 64 MiB threshold. The threshold
must be at least 1 MiB. Apply a custom policy again after reopening.
`graphMaintain(interruption:)` runs one bounded step and returns the last
published generation, step counters, and `cycleComplete`. Pass
`.cancellation(token)` or `.deadlineNanoseconds(ns)` to interrupt the step.
Both maintenance methods require a writer and can throw `ZeppelinError.busy`
when another writer operation is in progress. On a core-only build, the
creation overload and both maintenance methods throw
`ZeppelinError.graphUnsupportedBuild`.

For local graph builds, link one full archive:

```sh
CARGO_BUILD_JOBS=4 cargo build -p zeppelin-embed-ffi --release --features graph-bindings-test-support,swift-query-test-support
CLANG_MODULE_CACHE_PATH=/tmp/ze-swift-clang ZE_USE_LOCAL_FFI=1 ZE_ENABLE_GRAPH=1 swift build --disable-sandbox --jobs 4
CLANG_MODULE_CACHE_PATH=/tmp/ze-swift-clang ZE_USE_LOCAL_FFI=1 ZE_ENABLE_GRAPH=1 swift test --disable-sandbox --jobs 4 -Xswiftc -DZE72_TEST_BRIDGE -Xswiftc -DZE_SWIFT_QUERY_TEST_SUPPORT
```

The separate graph package is retired. The default package resolves the
universal graph-free XCFramework on Apple silicon and Intel. Set
`ZE_ENABLE_GRAPH=1` when resolving and building the package to select the
checksum-pinned graph XCFramework and expose graph methods on `ZeppelinStore`.
The graph XCFramework is the full macOS 14+ Apple silicon artifact and includes
the model-backed text C ABI; the default universal, Intel and non-macOS artifacts
are text-free. Its dynamic `ZeppelinEmbedGraph.framework` carries
`Resources/mlx.metallib`, so SwiftPM consumers need no shader-copy step. For local
artifact validation, also set `ZE_USE_LOCAL_XCFRAMEWORK=1`; the package selects
`target/xcframework-graph-cypher/ZeppelinEmbedGraph.xcframework`.

### Embedded text namespaces

Use the full graph + text XCFramework (`ZE_ENABLE_GRAPH=1`) on macOS 14+
arm64. Core and graph-only builds throw `ZeppelinError.unsupported` for
embedded namespace operations.

`NamespaceSpec(attributes:embeddedModel:)` accepts a pinned `TextModel`.
The model supplies the vector space and embedding epoch. Supplying an
explicit vector space or epoch as well is an error. Namespace open resolves
the model through the standard model cache, and reopen uses the same
model pins. Send text with an empty vector to `ZeppelinStore.upsert`.

```swift
let spec = NamespaceSpec(
    attributes: [AttributeDefinition(id: 1, name: "meeting", type: .u64, nullable: false)],
    embeddedModel: .default
)
let notes = try await ZeppelinStore.openNamespace(root: rootURL, name: "notes", spec: spec)
let noteID = DocumentID(high: 0, low: 1)
_ = try await notes.upsert([
    IngestDocument(id: noteID, revision: 1, timestamp: 10, vector: [],
                   text: transcript, attributes: [1: .u64(7)])
])
let hits = try await notes.queryEmbedded("delivery dates", leg: .hybrid, k: 5,
                                        filter: .eq(1, .u64(7)))
```

`queryEmbedded` ranks chunks using exact squared L2 with `.dense`, BM25
with `.lexical`, or reciprocal rank fusion with `.hybrid`. The limit
`k` is 1–4096. Each `EmbeddedHit` includes the document ID, revision,
timestamp, attributes, zero-based chunk, text and score. The half-open
`byteStart..<byteEnd` range indexes the source's UTF-8 bytes.

```swift
let source = NeighbourSource(id: noteID, chunk: 0)
let nearest = try await notes.nearestChunks(source, options: NeighbourOptions(k: 5))
let groups = try await notes.nearestChunksBatch(
    [NeighbourSource(id: noteID), source],
    options: NeighbourOptions(threshold: -0.25, filter: .eq(1, .u64(7)))
)
```

Nearest-chunk search uses persisted document vectors. Omit `chunk` to
score each candidate against the closest chunk in the whole source.
Each `ChunkNeighbour` has an ID, zero-based chunk and negative squared L2
score, with zero for an exact match. Ties sort by ID and chunk. Source
document chunks are excluded unless `includeSource` is true. Batch results
preserve input order, duplicate sources and empty groups. An unknown
source ID or chunk throws `ZeppelinError.notFound`.

`NeighbourOptions.k` limits results to 1–4096 per source. With only a
`threshold`, the result contains every score strictly above it. Omitting
both returns up to 10 hits per source. All chunk searches support a
`filter`, half-open `timestampRange` and `eligibleIds` before ranking.
An empty eligibility list selects zero hits. Supply either a
`cancellationToken` or a nonzero relative `deadlineNanoseconds`.
Cancellation and timeout throw typed errors without partial results.

### TextStore

`TextStore` is available from the release with `ZE_ENABLE_GRAPH=1` on macOS 14+
arm64, or with a locally built full XCFramework
(`ZE_USE_LOCAL_XCFRAMEWORK=1 ZE_ENABLE_GRAPH=1`). It is not available with
`ZE_USE_LOCAL_FFI=1`. It adds no binary target or second copy of core.
The dynamic framework includes `Resources/mlx.metallib`; preserve that resource
when distributing your app. The default universal artifact has no text API.

```sh
ZE_ENABLE_GRAPH=1 swift build
```

MLX keeps a process-global buffer cache, capped at 4 GiB by default during text
ingest. On an Apple M3 Max with 128 GiB, a full FiQA ingest (57,638 documents)
with the release FP16 default model peaked at about 7.3 GB process footprint
(about 112 GB before the cap), with max RSS about 4.8 GB (ZE-528). Pass `mlxCacheLimitBytes` as a `UInt64` to either
`TextStore.open` overload.
The first explicit limit overrides `ZE_MLX_CACHE_LIMIT_BYTES` at initialization.
Zero disables caching. After initialization, omitting the limit reuses the
process setting. A different explicit limit returns `ZE_ERR_MLX_CACHE_LIMIT_CONFLICT`.

```swift
import Foundation
import ZeppelinEmbed

let storeURL = URL.applicationSupportDirectory.appending(path: "ZeppelinText")
let store = try await TextStore.open(at: storeURL, model: .default)
let report = try await store.ingest([
    TextDocument(id: DocumentID(high: 0, low: 1), revision: 1, text: "hello world")
])
for legs in [TextLegs.dense, .lexical, .hybrid] {
    let result = try await store.query("hello", legs: legs, k: 5)
    print(result.hits)
}
let health = try await store.health()
// Run more slices if the returned status is .budgetExhausted.
let maintenance = try await store.maintain(
    wallTimeNanoseconds: 25_000_000, byteBudget: 64 * 1024 * 1024)
_ = try await store.delete([DocumentID(high: 0, low: 1)])
try await store.close()
```

To download a pinned model, use
`TextStore.open(at:model:cacheDirectory:offline:options:progress:)` with a
`TextModel(repo:revision:manifestSHA256:)`. Omit `model` to use `TextModel.default`:
`volantis-labs/volantis-db-clean71-arctic-m-v2` at
`d64b817c4021a3f1b9bded3152d9afdf9ac813e7`. The revision is a 40-hex commit and
the manifest pin is a 64-hex SHA-256. The call runs on a detached task, like
`ZeppelinStore`. The optional `@Sendable` progress closure receives
`completedBytes` and `totalBytes` on that task's thread; total is zero while
fetching the manifest. Intermediate reports require both at least 64 KiB and
at least 100 ms since the last report; each verified file also reports completion.
Dispatch UI updates to the main actor; do not reenter model resolution from the
callback.

The first open downloads and verifies the pinned model; subsequent opens reuse
the verified cache. `cacheDirectory` overrides `ZE_MODEL_CACHE` and the default
`~/Library/Caches/zeppelin-embed/models`. `offline: true` or `ZE_MODEL_OFFLINE=1`
prohibits networking; an offline miss throws `ZeppelinError.notFound`.
`ZE_MODEL_ENDPOINT` overrides the download endpoint.
`TextStore.open(at: storeURL, bundle: bundleURL)` opens a preinstalled bundle.
See the [model card](https://huggingface.co/volantis-labs/volantis-db-clean71-arctic-m-v2/blob/d64b817c4021a3f1b9bded3152d9afdf9ac813e7/README.md)
and [root text API guide](../../README.md#text-api-macos-apple-silicon) for model
limits, download size, and measured performance.

Ingest and query use V2 calls. Ordinary calls bridge Task cancellation to a
native cancel token. A nonzero `deadlineNanoseconds` instead selects the native
relative deadline; the C ABI makes cancellation tokens and deadlines mutually
exclusive. Cancellation and expiry are cooperative; ingest can acknowledge
partial writes before interruption. `tier: nil` leaves tier selection to the
engine. Text caller IDs must fit in 96 bits. `TextIngestOptions` exposes bounded
batching and optional token-window chunking.

`try await store.observedComputeUnits()` explicitly loads CoreML compute-plan
placement counts (`cpu`, `gpu`, `neuralEngine`, `other`), or returns `nil` when
unavailable. These are planned placements, not an execution trace. Query never
loads the compute plan.

Local full-artifact tests and installed-package qualification:

```sh
scripts/xcframework/build.sh --artifact graph-cypher
scripts/macos-sdk/build.sh --artifact graph-cypher
CLANG_MODULE_CACHE_PATH=/tmp/ze509-clang ZE_USE_LOCAL_XCFRAMEWORK=1 ZE_ENABLE_GRAPH=1 swift test --disable-sandbox --jobs 4 --filter TextStoreTests
python3 scripts/release/check-installed-text.py --artifact-root target --output /tmp/ze509-installed-text --disable-swift-sandbox
```

The installed check also needs the full C SDK release artifact under
`target/macos-sdk-graph-cypher`; it exercises the Swift API outside the checkout,
including the existing missing-metallib negative control. The Swift HTTP test
uses a local Python fixture server and needs permission to listen on loopback.
