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

### Graph

Graph methods use the same `ZeppelinStore` handle: `enableGraph`, `graphApply`,
`graphQuery`, `cypher`, `getNodes`, `getRelationships`, and `graphResources`.
Call `enableGraph()` explicitly before graph queries. `GraphNodeID` is an alias for `DocumentID`. A document node created with
`GraphNodeImage(id:timestamp:attributes:metadata:)` uses the appended V2 ABI;
create requires a caller ID, and put uses its existing node ID.
Text on a graph node is searchable with a lexical query or Cypher `ze.text_search`
as soon as the write returns, including when the store assigns the node ID.
A later graph put replaces the indexed text, and deleting the node removes it.
Vector and hybrid queries use document vectors.

For local graph builds, link one full archive:

```sh
CARGO_BUILD_JOBS=4 cargo build -p zeppelin-embed-ffi --release --features graph-bindings-test-support
CLANG_MODULE_CACHE_PATH=/tmp/ze-swift-clang ZE_USE_LOCAL_FFI=1 ZE_ENABLE_GRAPH=1 swift build --disable-sandbox --jobs 4
CLANG_MODULE_CACHE_PATH=/tmp/ze-swift-clang ZE_USE_LOCAL_FFI=1 ZE_ENABLE_GRAPH=1 swift test --disable-sandbox --jobs 4 -Xswiftc -DZE72_TEST_BRIDGE
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
