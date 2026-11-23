# Volantis DB for Node.js

`volantis-db` provides in-process vector, text and hybrid search for
macOS on Apple silicon and Intel, and for Windows x64. The package uses stable
Node-API and includes its native engine, so it does not need a separate dynamic
library at runtime.

```bash
npm install volantis-db
```

```js
const { Store } = require('volantis-db');

const store = new Store('my-index');
store.ingest(
  [{
    id: 1n,
    vector: new Float32Array([0.9, 0.1]),
    text: 'harbour meeting notes',
    metadata: Uint8Array.of(0, 255),
  }],
  2,
);

console.log(store.search(new Float32Array([0.8, 0.2]), 1));
console.log(store.state()); // 'open'
console.log(store.stats().activeRowCount);
console.log(store.get([1n], { text: true, metadata: true }).documents);
store.close();
```

Applications supply document and query vectors. Document IDs are unsigned
128-bit `bigint` values. Close each store when it is no longer needed; the
native finalizer also closes an open handle if the JavaScript object is
collected.

`store.state()` returns the engine lifecycle state as `'open'`, `'closing'`,
or `'closed'` while the handle is live. `store.stats()` reads the open store's
exact resource counters. Both calls are synchronous. After `store.close()`
releases the handle, either call throws `VolantisError` with
`code === 'ZE_ERR_CLOSED'`.

The `StoreStats` counters are `bigint` values. Byte fields report memory,
mappings, segment capacity, the in-memory WAL, caches and temporary space.
Count fields cover active rows, tombstones, open files, active queries and
snapshot leases. `physicalFootprint` is the process footprint in bytes on
macOS and is `null` when unavailable.
`Store.ingest(documents, dimension)` writes one atomic batch. Every document
requires a `Float32Array` vector of that dimension. Supply an optional `text`
string for lexical search and optional `metadata` bytes as a `Uint8Array` or
Node `Buffer`. An empty string carries no text. Read both fields with
`store.get(ids, { text: true, metadata: true })`.

Use `Store.upsert` to write schema attributes. An `attributes` property on
an ingest document or a wrong type for `text` or `metadata` throws a
`TypeError` with code `ERR_INVALID_ARG_TYPE`.

The engine stores all 128 bits of an ID and every API returns it unchanged,
so the whole UUID range fits. `uuidToId` reads a UUID's 32 hex digits as one
big-endian integer and `idToUuid` formats an ID back as a lowercase UUID; the
pair is exact for every 128-bit value:

```js
const { uuidToId, idToUuid } = require('volantis-db');

const id = uuidToId('123e4567-e89b-12d3-a456-426614174000');
store.upsert([{ id, text: 'meeting notes' }]);
const hit = store.query({ text: 'meeting', k: 1 }).hits[0];
console.log(idToUuid(hit.id)); // '123e4567-e89b-12d3-a456-426614174000'
```

`uuidToId` accepts only the `8-4-4-4-12` hex form (either case) and throws a
`TypeError` for anything else; `idToUuid` throws a `RangeError` for a bigint
outside `0n..2n ** 128n - 1n`.

Namespaces add typed records without changing the existing vector API:

```js
const { openNamespace, listNamespaces } = require('volantis-db');

const store = openNamespace('my-database', 'documents', {
  attributes: [{ id: 1, name: 'category', type: 'dictionaryString' }],
  vectorSpace: { dimensions: 2 },
});
store.upsert([{
  id: 1n,
  vector: new Float32Array([0.9, 0.1]),
  text: 'stored text',
  attributes: [{ id: 1, type: 'string', value: 'example' }],
}]);
console.log(store.get([1n]));
console.log(store.scan({ limit: 100 }));
console.log(listNamespaces('my-database'));
store.close();
```

Each document has a caller-chosen `revision` (`1n` when omitted) that only
moves forward per id: a higher revision replaces the document, the same
revision is an idempotent retry that writes nothing, and a lower one throws
`ZE_ERR_STALE_REVISION`. A deleted id keeps its last revision as that floor.
`get` returns the live revision.

An upsert or delete entry can carry `expectedRevision`, which makes the whole
call a compare-and-set: a `bigint` requires a live document at exactly that
revision, and `null` requires that no live document exists (never written, or
deleted). The single writer checks every condition against the latest
committed state, atomically with the write. If any condition fails, the call
writes nothing and throws `ZE_ERR_REVISION_CONFLICT` with a `conflict` that
names the failed entry:

```js
const head = store.get([headId]).documents[0];
try {
  store.upsert([
    { id: headId, revision: head.revision + 1n, expectedRevision: head.revision },
    { id: segmentId, expectedRevision: null },
  ]);
  store.delete([{ id: draftId, expectedRevision: 3n }]);
} catch (error) {
  if (error.code !== 'ZE_ERR_REVISION_CONFLICT') throw error;
  // error.conflict: { index, id, expectedRevision, currentRevision }
}
```

Omit `vectorSpace` for a record-only namespace. Scan cursors are opaque and
must be passed back unchanged; a cursor invalidated by a write throws
`ZE_ERR_SCAN_STALE`.

A scan can also order by a declared `u64`, `i64` or `f64` attribute:

```js
const page = store.scan({
  limit: 50,
  order: { attributeId: 1, direction: 'descending' },
});
const next = store.scan({
  limit: 50,
  order: { attributeId: 1, direction: 'descending' },
  cursor: page.cursor,
});
```

Equal values break ties by ascending document id, so pages are stable.
`f64` compares numerically and `-0` equals `+0`. A document whose value is
missing or `NaN` sorts after every document with a value, in both
directions. A cursor works only with the order that issued it; any other
order, attribute or direction throws `ZE_ERR_INVALID_ARGUMENT`. Ordering by an
undeclared attribute, by attribute 0 (the timestamp), or by a `bool` or string
attribute throws `ZE_ERR_INVALID_ARGUMENT`.

A later release can add attributes without re-ingesting. Declare the new
attribute as nullable and open the namespace for writing; documents written
before it read it as null, exactly like a document written without it
(`isNull` matches them; `eq`, `in`, `range` and `exists` do not):

```js
const store = openNamespace('my-database', 'documents', {
  attributes: [
    { id: 1, name: 'category', type: 'dictionaryString' },
    { id: 2, name: 'language', type: 'dictionaryString', nullable: true },
  ],
  vectorSpace: { dimensions: 2 },
});
```

Attributes match by `id`. Every stored attribute must be declared unchanged.
Removing or changing one, adding a non-nullable one, or adding one on a
`readOnly` open throws `ZE_ERR_SCHEMA_MISMATCH` naming the attribute. Once a
release adds an attribute, earlier releases can no longer open the namespace
with their shorter declaration.

One `upsert`, `ingest` or `delete` call is one atomic batch. If the process
or the machine stops at any point during the call, the next writable open
shows every document of the batch or none of them, and every batch whose call
returned is still there. Put documents that must change together (a note head
and its body, a transcript version and its head pointer) in one call.

A batch is atomic only inside one namespace. Each namespace has its own
write-ahead log, so an `upsert` to `notes` and an `upsert` to `segments` are
two independent batches, and a crash between them can keep the first without
the second. Keep documents that must change together in one namespace, or
make the second write repairable from the first (for example, write the
dependent documents first and the pointer that makes them live last).

A read-only open never repairs: if the last write was cut mid-record, it
throws `ZE_ERR_CORRUPT` until a writable open has cut that record off.

Writes land in an active segment backed by the write-ahead log. `seal()`
turns the active segment into an immutable segment and absorbs the log
records it covers, so opening the store again reads the sealed segment
instead of replaying those writes. It then truncates the log to its
header, so `wal.ze` holds only the writes since the last seal and stays
bounded under `autoSealRows`; a crash during that step loses nothing.
Seal after a bulk load, when the
application is idle, or let the store do it with `autoSealRows`:

```js
const store = openNamespace('my-database', 'notes', { attributes: [] }, {
  autoSealRows: 2048,
  autoMerge: true,
});
store.seal(); // { generation }; a no-op when nothing is unsealed
```

With `autoSealRows`, the store seals once at open and again before the write
that follows that many written documents since the last seal. A smaller value
keeps each write cheaper, because a write copies the active segment; a larger
value makes fewer sealed segments. `autoSealRows` is disabled by default and
must be a positive safe integer. It counts documents (including revisions and
deleted IDs), not bytes or calls; a batch may exceed the threshold before the
next write seals it.

`autoMerge` defaults to `false`. With `true`, open seals and merges existing
writes, and every subsequent automatic or explicit seal runs `merge()`. Pair
both options as above so the application need not schedule maintenance. Both
require a writable store. Merging starts when at least two compatible small
scan segments fit the native batch limits below; there is no additional timer
or configurable segment-count threshold. It drops dead revisions in selected
segments, including stored text. Live retained data still consumes memory and
disk; these options do not implement retention or cap total store size.

Automatic maintenance is synchronous and adds latency to the triggering open,
seal or write. Errors propagate before the pending write; already completed
maintenance remains committed. `seal()` returns the final merge generation.
With automatic merging disabled, call `store.merge()` during application idle
time to combine small sealed segments and reduce first-query cost after reopen. This synchronous call leaves active writes and the WAL alone;
call `seal()` first to include the current writes. It publishes each replacement
before removing the inputs, in batches of at most 16 segments and 8 MiB of input
files. Decoded working memory exceeds those input bytes. Large segments and graph
segments stay separate. The result is `{ generation }`, unchanged if no batch
fits; a failure can follow already committed batches.
With an open snapshot, explicit and automatic merges still publish replacements,
but retain their input files. Close the snapshots and reopen the writable store
to reclaim those retired files. Snapshot reads and cursors keep their original
generation throughout sealing and merging.

`store.maintain({ wallTimeNs, bytes })` runs due tier transitions. Both fields
are required unsigned 64-bit bigints. `wallTimeNs` sets the
monotonic time budget in nanoseconds and `bytes` sets the graph work budget.
The report contains `status` (0 complete, 1 budget exhausted) and three bigint
counters, `graphsBuilt`, `bytesConsumed` and `checkpointsResumed`. A zero budget
in either field returns budget exhausted with all counters at zero.

`await store.maintainAsync({ wallTimeNs, bytes })` uses the same budgets and
report. It copies the budget at the call and runs on a native worker in the
store's async write queue. Both calls require a writable handle and have no
cancellation option. Invalid budgets and read-only handles produce
`VolantisError` with `ZE_ERR_INVALID_ARGUMENT` and `ZE_ERR_ACCESS_MODE`,
respectively. For the sealed merge previously available as `maintainAsync()`,
call `await store.mergeAsync()`.

`deleteWhere` deletes every document that matches a filter, in one mutation,
and removes their bytes from disk. It takes the same `Filter` as `scan` and
`count`, and the filter is required:

```js
const { deleted, generation } = store.deleteWhere({
  op: 'eq',
  attributeId: 1,
  values: [{ id: 1, type: 'u64', value: 42n }],
});
```

No other write can land between finding the matches and deleting them, and
readers see all of the matched documents or none of them. When the call
returns, no byte of a deleted document remains in any store file: each
affected sealed segment and the write-ahead log are rewritten without it, so
there is no separate compaction step. If the process stops during the call,
either nothing was deleted or the next writable open finishes the removal.
`delete(ids)` only hides documents; their bytes stay on disk.

`count` returns the live documents that match an optional `filter` and
`timestampRange`. With `groupBy` it also returns one count per value of a
`u64`, `i64`, `dictionaryString` or `rawString` attribute, all from one
generation:

```js
store.count({ groupBy: { attributeId: 1 } });
// { count: 5n, generation: 7n, missingCount: 1n,
//   groups: [{ value: 'home', count: 1n }, { value: 'work', count: 3n }] }
```

Groups are in ascending value order: numeric for integers, byte order for
strings. Documents whose attribute is null are not a group; they are in
`missingCount`, so the group counts plus `missingCount` equal `count`.
`groupBy.limit` (default 1024, at most 65536) bounds the number of distinct
values; more values throw `ZE_ERR_BUDGET_EXCEEDED` and no group is dropped.
Other attribute types throw `ZE_ERR_INVALID_ARGUMENT`. There is no timestamp
bucketing: to count per day, store a day number as an `i64` attribute and
group by it.

`openSnapshot()` pins an in-place read-only view of the current generation:

```js
const view = store.openSnapshot();
try {
  const first = view.scan({ limit: 100 });
  // Writes, seals and logical deletes on store do not change view or its cursors.
  const next = first.cursor ? view.scan({ limit: 100, cursor: first.cursor }) : null;
} finally {
  view.close();
}
```

The source must be writable. The view shares active data and sealed mappings;
opening it creates no files. All its reads use the pinned generation, and its
mutation methods return `ZE_ERR_ACCESS_MODE`. It remains readable after the
source closes. Close every view explicitly: while any is open, `purge()` and
`deleteWhere()` return `ZE_ERR_STORE_BUSY` before mutation. Retired segment paths
remain until the next writable open (or a later physical purge). If the source
closes first, its writer lock stays held until the views close, preventing a new
writer from reclaiming their files.

Use `openSnapshot()` on the existing namespace handle to pin it. Opening another
writable `openNamespace()` is a second writer and returns `ZE_ERR_STORE_BUSY`.
This is distinct from `ZE_ERR_BUSY`, which means concurrent FFI writer calls on
one handle. Snapshot admission uses the existing handle and waits for core
admission locks; it does not attempt either writer-lock admission path.

`snapshot(target)` writes a consistent copy of the store at one generation
into a directory, for a backup or an export, while the application keeps
writing. It runs on a worker thread and resolves to `{ generation }`; writes
made while it runs are not in the snapshot. The target must not exist or must
be an empty directory, its parent must exist, and it must not be inside the
store. A failed snapshot never creates the target. The snapshot is an ordinary
store: open it, read-only or read-write, with the same namespace spec to
restore that state.

```js
const { generation } = await store.snapshot('/Volumes/Backup/notes-2026-09-24');
const restored = openNamespace('/Volumes/Backup', 'notes-2026-09-24',
  { attributes: [] }, { readOnly: true });
```

A crash during a snapshot can leave a hidden `.<name>.snapshot-*.tmp`
directory beside the target. It is never a snapshot and can be deleted.

`verify` checks a store directory end to end without opening or changing it:
the manifest, every segment's checksums and index regions, and the
write-ahead log through the same replay recovery runs. Run it after an unclean
shutdown, before reopening, or from diagnostics. Damage comes back as
findings rather than an exception:

```js
const { verify } = require('volantis-db');

const report = verify(path.join('my-database', 'notes'));
if (!report.ok) {
  for (const finding of report.findings) {
    // { kind: 'walRecordCorrupt', file: 'wal.ze', offset: 40n, detail: '...' }
    console.error(finding.kind, finding.file, finding.offset, finding.detail);
  }
}
```

Every finding is damage: the store will not open, or would lose data. A torn
log tail is damage too, because recovery refuses it. The package also installs
a `volantis-verify <store-directory>` command that prints the same report as
JSON and exits 0 when the store is clean, 1 when it found damage, and 2 when
it could not run.

`query` runs one structured query. `text` selects the lexical leg, `vector`
selects the vector leg, and both together run hybrid fusion; a request with
neither is refused.

```js
const { CancellationToken } = require('volantis-db');

// Lexical, with the last term treated as a type-ahead prefix.
store.query({ text: 'harb', k: 10, lastAsPrefix: true });

// Hybrid, with an explicit fusion weight.
const result = store.query({
  text: 'harbour lights',
  vector: new Float32Array([0.9, 0.1]),
  k: 10,
  alpha: 0.75,
});
console.log(result.mode, result.fusion.effectiveAlpha, result.hits);
```

Each hit carries `id`, `score`, and the per-leg `lexicalBm25` and
`vectorSquaredL2` when that leg ran; `revision` is absent on a fused hit,
which carries identity only. The result carries the queried `generation`, the
`mode` that ran, the `approximate`, `exactRescore`, and `budgetExhausted`
flags, and a `fusion` report when both legs ran.

Set `snippetBytes` to also get an excerpt of each hit's stored text with the
matched terms marked. The engine marks them with the same analyzer and the
same scored terms the query used, so stemmed, accent-folded, and
prefix-expanded forms are marked exactly as they matched. Without
`snippetBytes` the query reads no extra text.

```js
const { hits } = store.query({ text: 'harbour cafe', k: 10, snippetBytes: 120 });
for (const { snippet } of hits) {
  if (snippet === undefined) continue; // a hybrid hit the text did not match
  const marked = snippet.highlights.map(({ start, end }) => snippet.text.slice(start, end));
  const prefix = snippet.truncatedStart ? '…' : '';
  const suffix = snippet.truncatedEnd ? '…' : '';
  console.log(prefix + snippet.text + suffix, marked);
}
```

`highlights` are ascending, non-overlapping `{ start, end }` ranges in UTF-16
code units into `snippet.text`, so `String.prototype.slice` takes them
directly. The excerpt starts at a matched word and covers `snippetBytes` bytes
of UTF-8, plus at most 3 to finish a character; the window with the most
distinct matches wins. The engine adds no ellipsis; `truncatedStart` and
`truncatedEnd` say where the excerpt was cut. A document has one text field, so
a hit has at most one `snippet`. It is absent only on a hybrid hit whose text
contains no query term. The query operators are terms and a trailing prefix;
a quoted phrase is not matched as a phrase, so each of its terms is marked.

A query stops early on either a `deadlineNs` or a `cancelToken`, never both.
A `CancellationToken` owns an engine handle, so close it:

```js
const token = new CancellationToken();
try {
  const hits = store.query({ text: 'harbour', k: 10, cancelToken: token });
  // token.cancel() from elsewhere asks any query holding it to stop.
} finally {
  token.close();
}
```

A lexical query works on a record-only namespace. A hybrid query there throws
`ZE_ERR_NO_VECTOR_SPACE`.

The package supports Node.js 18 or newer on macOS arm64 and macOS x64, and
Node.js with Node-API 8 or later on Windows x64. Unsupported platforms fail
during installation and report a clear error if the package is loaded directly.

macOS ships one binary per architecture and no per-runtime variant. The addon
is a bundle built with `-undefined dynamic_lookup`, so its Node-API symbols
resolve from whichever host process loads it and the same file works in Node
and in Electron.

On Windows the loader picks the binary from `process.platform`,
`process.arch`, and `process.versions.electron`, with no try-each fallback.
Under Electron it accepts only the majors this package ships a binary for,
currently 44; any other runtime is refused by name with
`UnsupportedRuntimeError`. The Windows addon links the Visual C++ runtime, so
the machine needs the Visual C++ redistributable.

## Electron

The Node package workflow qualifies Electron 44.4.1 on macOS arm64 and x64
using a minimal electron-builder app with an ASAR archive. It runs the addon
in the main process, in a utility process, and in two utility processes using
different stores. Node tests also exercise worker threads, garbage collection,
and process exit with an unclosed store.

Declare `volantis-db` as a production dependency and include this in
your electron-builder configuration:

```json
{
  "asar": true,
  "asarUnpack": ["node_modules/volantis-db/prebuilds/**/*.node"],
  "npmRebuild": false
}
```

The unpack pattern is relative to the app directory. It keeps the native files
in `app.asar.unpacked`; Electron redirects their ASAR paths automatically, so
keep using `require('volantis-db')`. The published prebuilds need no
rebuild for macOS Electron. `npmRebuild: false` is appropriate for this minimal
app; applications with other native dependencies must handle those dependencies'
rebuild requirements separately. Windows requires the shipped Electron-major
variant described above.

Call `utilityProcess.fork` after `app.whenReady()`. In the utility entry point:

```js
const { openNamespace } = require('volantis-db');
const store = openNamespace('/absolute/writable/data/path', 'notes', {});
try {
  store.upsert([{ id: 1n, text: 'harbour lights' }]);
  console.log(store.query({ text: 'harbour', k: 1 }));
} finally {
  store.close();
}
```

Keep stores outside the signed app bundle. Each store has one writer; different
processes must use different stores. Loading the addon twice is tested by
removing both its JavaScript and native entries from the CommonJS cache and
asserting two native initializations, independent live stores, and persisted
results after close and reopen. Ordinary repeated `require` also works.

The macOS arm64 addon (including MLX text) targets macOS 13.0, the Metal 3.0
floor for MLX's shaders; the Intel addon targets macOS 11.0. The Electron
44.4.1 app bundle also declares macOS 13.0 as its minimum, so on arm64 the
addon requires macOS 13.0 or later. CI executes on macos-14,
not every older OS. The addon performs no runtime network requests
except `TextStore.openModel`, which downloads a pinned model unless `offline`
is set; installation and CI may download packages and Electron.

### Signing scope

CI uses only ad-hoc signing (`mac.identity: "-"`), with
`mac.hardenedRuntime: true` and `mac.notarize: false`. It verifies the app,
utility helper, and unpacked addon with `codesign --verify --strict`, checks
that their signatures are ad-hoc and carry the runtime flag, and checks the
whole bundle with `--deep --strict` before launching it.

The fixture grants `com.apple.security.cs.allow-jit` for Electron/V8 and
`com.apple.security.cs.disable-library-validation` because ad-hoc code has no
shared Team ID for library validation. It grants no unsigned-executable-memory
exception. These are host signing settings, not an addon JIT requirement.
Hardened runtime remains enabled with those explicit exceptions.

This proves packaged loading and signature integrity under the tested ad-hoc
configuration. It does not authenticate a publisher, prove Gatekeeper trust,
provide notarization or stapling, or prove loading with library validation
enabled. A future Developer ID qualification must sign all nested code with
one real identity, remove the ad-hoc library-validation exception and retest,
then separately notarize, staple, and assess distribution. No certificate or
notarization credentials are required for this CI fixture.

Volantis DB is licensed under GPL-3.0-only.

### Read-only diagnostics shell

The package installs `volantis-shell` (Node 18+ and the packaged native addon).
Run it against an extracted, quiescent **copy** of a store. Every open is
read-only: the shell has no mutation commands or writable mode, never repairs a
WAL, and does not create missing stores. A truncated WAL may prevent reads;
`verify` works without opening the store and reports the damage.

```sh
volantis-shell /diagnostics/root namespaces
volantis-shell /diagnostics/root/docs schema
volantis-shell /diagnostics/root/docs get 18446744073709551617
volantis-shell /diagnostics/root/docs scan '{"order":{"attributeId":1,"direction":"descending"},"limit":10,"filter":{"op":"exists","attributeId":1}}'
volantis-shell /diagnostics/root/docs count '{"filter":{"op":"eq","attributeId":1,"values":[{"id":1,"type":"u64","value":{"$bigint":"42"}}]}}'
volantis-shell /diagnostics/root/docs query '{"text":"hello world","k":10}'
volantis-shell /diagnostics/root/docs dump '{"limit":100}' > documents.jsonl
volantis-shell /diagnostics/root/docs verify
```

From this checkout, substitute `node bindings/node/bin/volantis-shell.js` for
`volantis-shell`. `--help` lists the commands. `namespaces` takes a namespace
root; all other commands take a store directory. Namespace listings include
attribute ids, names, types and nullability (the built-in timestamp is omitted).

Request objects use the Node API's `ScanRequest`, `CountRequest` and
`QueryRequest` shapes. Represent bigint inputs as `{"$bigint":"123"}` to avoid
JSON number rounding. `get` takes an unsigned decimal 128-bit id. `scan` returns
one page; `dump` follows all pages, includes every stored field, and treats
`limit` as page size. Its optional filter and order apply to the whole dump.
JSON output encodes bigint values as decimal strings, vectors and metadata as
arrays, and non-finite floats as `"NaN"`, `"Infinity"`, or `"-Infinity"`.
Dump output is for inspection, not an import format. Errors go to stderr;
exit codes are 0 for success, 1 for verify findings, and 2 for usage/open errors.

The same inspection capabilities are available programmatically:

```js
const { openInspection } = require('volantis-db');
const store = openInspection('/diagnostics/root/docs');
try {
  console.log(store.schema());
  console.log(store.count());
} finally {
  store.close();
}
```

`openInspection` reads the persisted identity and schema without requiring the
original application's namespace spec. It retains tokenizer compatibility
checks and rejects writes. Ordinary `Store` and `openNamespace` epoch/schema
validation is unchanged. This shell inspects document stores, not graph stores.
## Graph documents and Cypher

Graph methods use the same `Store` handle as document methods. A node's ID is
its document ID. Call `enableGraph()` explicitly to opt into graph storage;
opening a Store or checking `Store.graphSupported()` does not enable it.
Graph-free builds expose the methods and throw `ZE_ERR_GRAPH_UNSUPPORTED_BUILD`.

```js
const { Store } = require('volantis-db');
const store = new Store('/path/to/store', { maxResidentBytes: 268435456n });
try {
  store.enableGraph();
  const result = store.graphApply([
    { kind: 'node', operation: 'create', namespace: 'notes', key: 'note-1',
      revision: 1n, id: 1n, labels: ['Note'], text: 'Meeting notes',
      timestamp: 10n, metadata: new Uint8Array([1]),
      properties: { title: 'Planning', done: false } },
    { kind: 'node', operation: 'create', namespace: 'folders', key: 'work',
      revision: 1n, labels: ['Folder'], properties: { name: 'Work' } },
    { kind: 'relationship', operation: 'create', namespace: 'filing', key: 'note-1/work',
      revision: 1n, type: 'IN_FOLDER', source: { local: 0 }, target: { local: 1 } },
  ]);
  console.log(result.receipts);
  console.log(store.cypher('MATCH (n:Note) WHERE n.title IN $titles RETURN n',
    { titles: ['Planning'] }).rows);
  console.log(store.graphGetNodes([1n, 999n, 1n], { text: true }));
  console.log(store.graphQuery({root: 0, operators: [['scanNodes', 0, 'Note']],
    expressions: [], parameters: [], searches: [], eagerSearches: []}).rows);
} finally { store.close(); }
// Reopen with new Store(path), or new Store(path, {readOnly: true}).
```

Node items can also carry a finite `Float32Array` vector and the same
`attributes` array accepted by `upsert`. Vectors must match the Store's vector
space; create a namespace with `openNamespace` to declare that space/schema.
`id` selects a document ID on create only. Getters preserve input order,
duplicates, and nulls for missing IDs, and copy vectors into independent arrays.
`graphResources()` returns four bigint allocation counters.

`graphApplyAsync`, `cypherAsync`, `graphQueryAsync`, `graphGetNodesAsync`, and
`graphGetRelationshipsAsync` accept an options object containing `signal`.
Requests copy nested buffers before queuing work. Cancellation waits for the
worker to settle; completion may win an abort race. Sync/async graph calls also
accept `cancelToken` or a relative `deadlineNs`; use one interruption source.
Structured plans use tagged tuples in the declaration order in `index.d.ts`.

Creation-only relationship policies use `new Store(path, {relationshipTypes:
[{type: 'IN_FOLDER', onDelete: 'restrict'}], maxResidentBytes: 268435456n})`.
This synchronous creation form accepts only relationship types, resident limit,
and reader drain timeout. Graph maintenance uses `graphMaintain()` /
`graphMaintainAsync()`. Configure it explicitly with `graphSetMaintenancePolicy`
and `{autoReclaim, reclaimAfterBytes}`; document `maintain()` retains its behavior.

Mutations are keyed by `(namespace, key)` with positive unsigned 64-bit bigint
revisions. Exact retries report `Replayed`. `put` replaces the full image and
requires `expectedId`; `delete` also requires `expectedId` (node deletion accepts
`detach: true`); `recreate` requires `expectedDeletionRevision`. Endpoints are
node IDs (`bigint`) or `{ local: index }` references into the same batch. At most
16,384 items and 8 MiB of encoded graph input are permitted. Before staging,
the engine also checks a batch-derived bound on preparation attributable to
the submitted items against the 32 MiB preparation allowance. An oversized
estimate returns a distinct `Limit` message without staging the batch.
Existing records, catalogs and tree paths remain subject to the existing
in-flight storage limits: a batch within the caps can still be refused by
store-dependent limits. Omitted image fields are empty/absent, not patches.

Query results contain `columns`, `rows`, `receipts`, `disposition`,
`admittedGeneration`, `changedGeneration`, and `generation` (changed when known,
otherwise admitted). Values are null, boolean, I64 bigint, F64 number, string,
node/relationship objects with a `kind` discriminator, or lists. Properties use
scalar values or homogeneous scalar lists. Parameters accept scalars, arrays
and numeric typed arrays, including mixed scalar types, nulls and nested lists. Lists are limited to 16 levels and
524,288 total elements across parameters. Execution budgets also apply: the
current engine expression arena has 1,024 cells shared by parameter list
elements and expression temporaries; larger inputs can fail during execution.
For example, use
`store.cypher('MATCH (n) WHERE n.id IN $ids RETURN n.id', { ids: ['a', 'b'] })`. Pass a
bigint for an integer; ordinary JavaScript numbers are F64. IDs are unsigned
128-bit bigints. Read a node’s `text` with `ze.stored_text(n)`. After a graph write returns,
`store.query({text})` and Cypher `ze.text_search` can find that node, including
when the store assigned its ID. A later `put` replaces the indexed text, and
a node delete removes it.

`maxRows` accepts integers 0..65,536; omitted or zero selects 1,024. Exceeding the
cap throws `ZE_ERR_BUDGET_EXCEEDED` without truncation. Raising it permits larger results,
subject to the engine's separate memory/work limits. Errors preserve settlement
dispositions, including `NotCommitted`, `Committed`, `Replayed`, and
`Indeterminate`; do not blindly retry a write because it threw. Invalid JS inputs
are rejected before calling the engine.

The [documented Cypher profile](https://github.com/volantis-labs/volantis-db/blob/main/crates/volantis-db-cypher/README.md)
defines supported statements and functions. This binding does not add syntax,
graph search or vector inputs. The example uses synchronous
calls; use the async API with `AbortSignal` for potentially long operations.

The normal Node suite includes `test/bounded-soak.test.mjs`: eight compressed
meeting hours (one transcript per second and one note edit per five seconds),
then eight hours of note edits with a fixed live corpus. It samples process RSS,
WAL and total store bytes hourly, without explicit maintenance or forced GC.
The first half of the fixed-corpus phase warms allocator and merge state;
the second half must grow by less than 64 MiB from that warmed baseline.
Run `ZE_LONG_SOAK=1 node --test test/bounded-soak.test.mjs` for 48 hours per phase
(up to 30 minutes instead of the normal three-minute deadline). The workload
uses `commitTier: 'none'` to measure engine behavior without per-write fsync;
it does not qualify power-loss durability or weeks of real-time use.

### Prepare lexical queries after open

Call `await store.warmLexicalAsync({ signal })` after opening a namespace to
prepare its lexical assembly and prefix vocabulary before the first search.
`store.warmLexical()` is the synchronous alternative. Both return no value.
This also works for record-only namespaces and empty stores. Repeated calls
reuse the current caches; mutations invalidate preparation, so warm again
after writes when first-query latency matters. Prefer the async method to
keep Wallflower's event loop free.

## TextStore (macOS arm64)

Requires macOS 13.0 or later on Apple silicon. `TextStore` embeds text locally and searches dense, lexical or hybrid legs. The
arm64 prebuild includes MLX and its `mlx.metallib` beside the addon; keep both
files together when repackaging. Intel macOS and Windows builds throw a typed
`VolantisError` with code `ZE_ERR_UNSUPPORTED` and the message
`text API is available on macOS arm64 only`.

Electron apps must `asarUnpack` both `mlx.metallib` and `volantis_db.node`,
because MLX loads the metallib from the directory of the native binary.

MLX keeps a process-global buffer cache, capped at 4 GiB by default during text
ingest. On an Apple M3 Max with 128 GiB, a full FiQA ingest (57,638 documents)
with the release FP16 default model peaked at about 7.3 GB process footprint
(about 112 GB before the cap), with max RSS about 4.8 GB (ZE-528). Pass `mlxCacheLimitBytes` as an unsigned bigint in
`TextStore.openModel({path, mlxCacheLimitBytes})` or in the third argument to
`TextStore.open(path, bundlePath, {mlxCacheLimitBytes})` and `openAsync`.
The first explicit limit overrides `ZE_MLX_CACHE_LIMIT_BYTES` at initialization.
Zero disables caching. After initialization, omitting the limit reuses the
process setting. A different explicit limit returns `ZE_ERR_MLX_CACHE_LIMIT_CONFLICT`.

```js
const { TextStore } = require('volantis-db');

async function main() {
  const store = await TextStore.openModel({ path: './notes' }); // default model
  try {
    // Caller ids are unsigned bigints smaller than 2^96.
    await store.ingestAsync([
      { id: 1n, revision: 1n, text: 'Harbour lights at dusk' },
    ]);
    for (const legs of ['dense', 'lexical', 'hybrid']) {
      const result = await store.queryAsync({ text: 'harbour', k: 5, legs });
      console.log(result.hits);
    }
    await store.deleteAsync([1n]);
  } finally {
    store.close();
  }
}
main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
```

`open`, `ingest`, `query`, `delete`, `maintain` and `health` also have synchronous
forms. Async forms run native work off the JS thread. `query` defaults to dense;
`legs` accepts `dense`, `lexical`, or `hybrid`, and `tier` accepts `auto`, `exact`,
`scan`, or `graph`. `deadlineNs` is a relative monotonic deadline, mutually
exclusive with `signal`. In-flight AbortSignal cancellation requires an async
call. Cancellation of ingest may leave acknowledged partial writes.

Both query forms return a `TextQueryResult` with `hits`, the store's
`embeddingEpoch` and `tokenizerEpoch`, and `bigint` counts for `queryTokens`
(nonpadding model tokens) and `embeddingCalls` (completed query embeddings).
Lexical queries and queries with `k: 0` return zero for both counts.

- `diagnostics` contains the pinned `generation`, the executed `mode`
  (`vector`, `lexical`, or `hybrid`), `approximate`, `exactRescore`,
  `budgetExhausted`, and `bigint` counters `dimsTouched`, `bytesRead`,
  `docsEvaluated`, and `postingsDecoded`. The leg epochs and `fusion` report
  are present when the engine reports them. `diagnostics` is omitted for `k: 0`.
- `timings` is present in timing-enabled builds. Its `bigint` nanosecond spans
  are `tokenizationNs`, `lexicalAnalysisNs`, `embeddingQueueNs`,
  `embeddingEvaluationNs`, `embeddingNormalizationNs`, `retrievalNs`,
  `materializationNs`, and `endToEndNs`. The spans can overlap.
- `backend` is present when a model runtime participates. It reports
  `runtimeName`, `gpu`, and `requestedComputeUnits` (1 CPU, 2 CPU and GPU,
  3 CPU and Neural Engine, 4 all). Fixed-width models also report
  `sequenceLength` as a `bigint`. When CoreML placement has been cached,
  `observedComputeUnits` contains `cpuOps`, `gpuOps`, and `neuralEngineOps`
  as `bigint` counts from the plan.

Ingest rejects overlong documents unless `chunkMax` is set; `chunkOverlap`
controls overlap. Re-ingest with a higher revision replaces the previous chunks.
Both `ingest` and `ingestAsync` take `maintenanceWallTimeNs` and
`maintenanceBytes` as unsigned 64-bit `bigint` options. Each defaults to `0n`.
Either zero disables ingest maintenance. When both are positive, they bound each
maintenance slice and ingest runs those slices to completion. Values outside the
unsigned 64-bit `bigint` range throw or reject with `RangeError`
(`ERR_OUT_OF_RANGE`).

Explicitly call `store.observedComputeUnits()` or `await store.observedComputeUnitsAsync()` to load the CoreML compute plan on demand: four `bigint` counts describe planned placement, not an execution trace; returns `null` without a CoreML query model.

To download and cache a model, use the asynchronous `TextStore.openModel` with
`{path, repo, revision, manifestSha256, cacheDir?, offline?, onProgress?}`.
`revision` must be a pinned 40-hex commit and `manifestSha256` the manifest's
64-hex SHA-256. Omit all three pins (`TextStore.openModel({path})`) to use
`TextStore.defaultModel`: `volantis-labs/volantis-db-clean71-arctic-m-v2` at
`5833eb696dbc41def2421a7fb97d58f92db94fab`. A partial set of pins is an error. `onProgress({done, total})` receives
bigint byte counts on the JS thread. Intermediate reports require both at least
64 KiB and at least 100 ms since the last report; each verified file also reports
completion. `(0n, 0n)` means fetching the manifest.

The first open downloads and verifies the pinned model; subsequent opens reuse
the verified cache. `cacheDir` overrides `ZE_MODEL_CACHE` and the default
`~/Library/Caches/volantis-db/models`. `offline: true` or `ZE_MODEL_OFFLINE=1`
prohibits networking; a cache miss throws `ZE_ERR_NOT_FOUND`. `ZE_MODEL_ENDPOINT`
overrides the download endpoint. `TextStore.openAsync(path, bundlePath)` opens a
preinstalled bundle. See the [model card](https://huggingface.co/volantis-labs/volantis-db-clean71-arctic-m-v2/blob/5833eb696dbc41def2421a7fb97d58f92db94fab/README.md)
and [root text API guide](../../README.md#text-api-macos-apple-silicon) for model
limits, download size, and measured performance.

### Embedded namespaces

`openNamespace` and `openNamespaceAsync` accept an `embeddedModel` with
`repo`, `revision` and `manifestSha256`. Use the same pinned spec when reopening
the namespace or naming it in a namespace batch. The model supplies the vector
dimensions. A multi-row upsert batches new and changed text across rows and
chunks before committing. Rows eligible for same-text vector reuse skip inference.
Overlapping opens with the same model pins and query backend share one runtime.
Each namespace keeps its own rows and generations. A snapshot retains the model
after its writer closes. Closing the final namespace model owner stops its worker
and releases unused MLX cache buffers. A cold reopen starts a new runtime, and
the first write that needs the document tower pays its loading cost for paired models.
A text-enabled build and the resolved model are required. The fourth argument
can set the process-wide ceiling for reusable MLX buffers on the first embedded
open. Embedded namespaces and TextStore share the [first-opener rules](#textstore-macos-arm64).

The same fourth argument takes `modelCacheDir` and `offline`, with the meaning of
`TextStore.openModel` `cacheDir` and `offline`. An explicit folder wins over
`ZE_MODEL_CACHE` and `~/Library/Caches/volantis-db/models`; an omitted field or an
empty string uses them. `offline: true` or `ZE_MODEL_OFFLINE=1` prohibits network
access: a cache miss throws `ZE_ERR_NOT_FOUND` and a damaged entry throws
`ZE_ERR_CORRUPT`. Each open verifies its own folder, also while another handle
already shares the pinned model. The folder and the offline switch are not part
of the namespace epoch. `namespaceBatch` and `namespaceBatchLive` read the same
two fields from each embedded participant. Plain and vector namespaces reject them.

```js
const models = join(app.getPath('userData'), 'models');
const notes = openNamespace(root, 'notes', spec, {modelCacheDir: models, offline: true});
namespaceBatch(root, [
  {name: 'notes', spec, modelCacheDir: models, offline: true, upserts: [{id: noteId, text: transcript}]},
  {name: 'tags', spec: tagSpec, upserts: [tag]},
]);
```

```js
const spec = {
  embeddedModel: modelRef,
  attributes: [{ id: 1, name: 'meeting', type: 'id128', nullable: false }]
};
const notes = await openNamespaceAsync(root, 'notes', spec, {mlxCacheLimitBytes: 536870912n});
const meeting = { id: 1, type: 'id128', value: meetingId };
await notes.upsertAsync([{ id: noteId, revision: 1n, text: transcript, attributes: [meeting] }]);
const hits = await notes.queryEmbeddedAsync({
  text: 'delivery dates', leg: 'hybrid', k: 5,
  filter: { op: 'eq', attributeId: 1, values: [meeting] }
});
```

The synchronous form is `notes.queryEmbedded(request)`. Choose `dense`,
`lexical` or `hybrid`. Dense search uses the query tower and exact squared L2.
Lexical search uses BM25. Hybrid combines both rankings by reciprocal rank.
The Store filter AST, timestamp range and eligible document IDs restrict both
legs before ranking. Hits contain `id`, `revision`, `timestamp`, `attributes`,
`score`, zero-based `chunk`, `text`, and half-open UTF-8 offsets `byteStart`
and `byteEnd`. Slice `Buffer.from(transcript)` to use these byte offsets.

After `enableGraph()`, use `CALL ze.vector_search($text, 5, 'exact') YIELD node
RETURN node` or `CALL ze.hybrid_search($text, 5, 'exact') YIELD node RETURN node`.
Cypher returns one result per document and uses its first chunk as the document
vector and its whole source for lexical search. Chunk search scores each chunk.
A text vector argument requires a registered embedder and otherwise reports
`ZE_ERR_MISSING_EMBEDDER`.

Each source revision occupies one Store row. The namespace epoch carries the
pinned model declaration. The row metadata envelope stores the caller metadata
followed by chunk byte ranges and little-endian float vectors. Normal Store
reads expose the caller metadata. Replacing or deleting the row retires all of
its chunks. A higher revision of a live ID with byte-identical UTF-8 source and
the same embedding and tokenizer epoch reuses every stored chunk vector and
source span. The new revision supplies the timestamp, attributes and caller
metadata. A new ID or changed source runs the document encoder. Store verification covers the persisted row bytes.

C callers pass a size-versioned `ZeEmbeddedNamespaceSpec` through its
`ZeNamespaceSpec` prefix with `has_vector_space = 2`. Pass a
`ZeEmbeddedQueryRequest` to `ze_query_embedded` with space for `k` chunk ranks.
The returned `ZeGetResult` owns the parallel hit rows, text and attributes.
Release that result with `ze_get_result_free`.

#### Neighbours of stored chunks

On an embedded Store namespace, use a document ID for the whole source or
`{id, chunk}` for one zero-based source chunk. These calls use persisted
document-tower vectors, including on a snapshot or inspection handle. The
standalone `TextStore` has a separate API.

```js
const hits = notes.nearestChunks({id: noteId, chunk: 0}, {k: 5});
const groups = await notes.nearestChunksBatchAsync([noteId, otherId], {
  threshold: -0.25,
  filter: {op: 'not', children: [{op: 'eq', attributeId: 1, values: [meeting]}]},
  signal
});
```

Each hit contains `{id, chunk, score}`. The score is negative squared L2
distance. Zero is an exact match. For a whole source document, a candidate
gets its highest score across all source chunks and appears once. Equal scores
sort by ascending candidate ID and chunk number. Source-document chunks are
excluded by default. Set `includeSource: true` to include them.

`filter`, `timestampRange` and `eligibleIds` constrain candidates before
ranking. Supply `k` (1–4096) for a maximum per source. With only a threshold,
all scores strictly above it are returned. Omitting both returns up to 10 hits.
`nearestChunksBatch` and its async form preserve input order, duplicate
sources and empty groups. Supply one of `deadlineNs`, `cancelToken` or
(async only) `signal`. A stopped batch returns a typed error with no partial
results. An unknown source ID or chunk produces `ZE_ERR_NOT_FOUND`.

C callers pass one or more `ZeNeighbourSource` entries to `ze_nearest_chunks`.
The result contains a flat array grouped by zero-based input `source_index`.
Release it with `ze_neighbour_result_free`. Set `k = 0` for an omitted limit.

### Required API check

The binding suite checks its public API against the
[E17 API table](../fixtures/required_api_v1.json). Graph and text rows apply to
builds that ship those capabilities. Python 0.1.0 has the owner-approved graph
exclusion. The temporary E17 gap list is empty and CI rejects new entries.
