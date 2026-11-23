import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { ABI_VERSION, Store, VolantisError } = require('..');

test('ingests and searches caller-supplied vectors through the C ABI', () => {
  const directory = mkdtempSync(join(tmpdir(), 'zeppelin-node-test-'));
  const store = new Store(join(directory, 'index'));
  try {
    const mutation = store.ingest(
      [
        { id: 1n, vector: new Float32Array([0.9, 0.1, 0.05, 0]) },
        { id: (1n << 80n) + 2n, vector: new Float32Array([0.1, 0.9, 0.05, 0]) },
      ],
      4,
    );
    assert.equal(ABI_VERSION, 1);
    assert.equal(typeof mutation.generation, 'bigint');

    const hits = store.search(new Float32Array([0.88, 0.12, 0.07, 0.02]), 2);
    assert.equal(hits.length, 2);
    assert.equal(hits[0].id, 1n);
    assert.equal(typeof hits[0].score, 'number');
  } finally {
    store.close();
    rmSync(directory, { force: true, recursive: true });
  }
});

function openIngestStore(t) {
  const directory = mkdtempSync(join(tmpdir(), 'zeppelin-node-ingest-'));
  const store = new Store(join(directory, 'index'));
  t.after(() => {
    store.close();
    rmSync(directory, { force: true, recursive: true });
  });
  return store;
}

test('indexes and returns text supplied to Store.ingest', (t) => {
  const store = openIngestStore(t);
  const text = 'Harbour café 🚀';
  store.ingest([{ id: 1n, vector: new Float32Array([1, 0]), text }], 2);
  store.seal();

  assert.equal(store.get([1n], { text: true }).documents[0].text, text);
  assert.deepEqual(store.query({ text: 'harbour', k: 2 }).hits.map(({ id }) => id), [1n]);
});

test('round-trips metadata supplied to Store.ingest', (t) => {
  const store = openIngestStore(t);
  const bytes = Uint8Array.of(10, 0, 255, 13);
  store.ingest(
    [
      { id: 1n, vector: new Float32Array([1, 0]), metadata: bytes.subarray(1, 3) },
      { id: 2n, vector: new Float32Array([0, 1]), metadata: Buffer.from([42, 128, 0]) },
    ],
    2,
  );

  const { documents } = store.get([1n, 2n], { metadata: true });
  assert.deepEqual(documents.map(({ metadata }) => metadata), [
    Uint8Array.of(0, 255),
    Uint8Array.of(42, 128, 0),
  ]);
});

test('rejects non-string text in Store.ingest', (t) => {
  const store = openIngestStore(t);
  assert.throws(
    () => store.ingest([{ id: 1n, vector: new Float32Array([1, 0]), text: 42 }], 2),
    { name: 'TypeError', code: 'ERR_INVALID_ARG_TYPE', message: /text/ },
  );
});

test('rejects non-byte-array metadata in Store.ingest', (t) => {
  const store = openIngestStore(t);
  assert.throws(
    () => store.ingest(
      [{ id: 1n, vector: new Float32Array([1, 0]), metadata: new Uint16Array([42]) }],
      2,
    ),
    { name: 'TypeError', code: 'ERR_INVALID_ARG_TYPE', message: /metadata/ },
  );
});

test('rejects attributes before writing a Store.ingest batch', (t) => {
  const store = openIngestStore(t);
  const vector = new Float32Array([1, 0]);
  for (const attributes of [[{ id: 1, type: 'u64', value: 5n }], [], undefined]) {
    assert.throws(
      () => store.ingest([{ id: 1n, vector }, { id: 2n, vector, attributes }], 2),
      { name: 'TypeError', code: 'ERR_INVALID_ARG_TYPE', message: /attributes.*upsert/ },
    );
  }
  assert.deepEqual(store.get([1n, 2n]).documents, [null, null]);
});

test('surfaces C ABI failures as VolantisError instances', () => {
  const directory = mkdtempSync(join(tmpdir(), 'zeppelin-node-error-'));
  const store = new Store(join(directory, 'index'));
  try {
    assert.throws(
      () =>
        store.ingest(
          [{ id: 1n, vector: new Float32Array([0.9, 0.1, 0.05, 0]) }],
          3,
        ),
      (error) =>
        error instanceof VolantisError &&
        error.code === 'ZE_ERR_DIMENSION_MISMATCH' &&
        error.errorCode === 18,
    );
  } finally {
    store.close();
    rmSync(directory, { force: true, recursive: true });
  }
});

test('releases the handle and reports use after close', () => {
  const directory = mkdtempSync(join(tmpdir(), 'zeppelin-node-close-'));
  const store = new Store(join(directory, 'index'));
  store.close();
  try {
    assert.throws(
      () => store.search(new Float32Array([1, 0]), 1),
      (error) =>
        error instanceof VolantisError && error.code === 'ZE_ERR_CLOSED',
    );
  } finally {
    rmSync(directory, { force: true, recursive: true });
  }
});

// Names are obtained dynamically from ze_error_code_name by the addon.
// Pin the appended C value and the JavaScript error representation.
test('graph epoch transition error retains its append-only name and value', () => {
  const header = readFileSync(new URL('../../../crates/volantis-db-ffi/include/volantis_db.h', import.meta.url), 'utf8');
  assert.match(header, /ZE_ERR_GRAPH_EPOCH_TRANSITION = 60,/);
  const error = new VolantisError('graph epoch transition refused', 'ZE_ERR_GRAPH_EPOCH_TRANSITION', 60);
  assert.equal(error.code, 'ZE_ERR_GRAPH_EPOCH_TRANSITION');
  assert.equal(error.errorCode, 60);
});
