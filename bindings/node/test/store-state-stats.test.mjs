import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const { Store, ZeppelinError } = createRequire(import.meta.url)('..');

function isClosedError(error) {
  return error instanceof ZeppelinError &&
    error.code === 'ZE_ERR_CLOSED' && error.errorCode === 3;
}

test('Store.state reports the lifecycle state and rejects a closed handle', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'zeppelin-state-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const store = new Store(root);
  try {
    assert.equal(store.state(), 'open');
  } finally {
    store.close();
  }
  assert.throws(() => store.state(), isClosedError);
});

test('Store.stats reports exact counters and rejects a closed handle', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'zeppelin-stats-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const store = new Store(root);
  try {
    const { physicalFootprint, ...counters } = store.stats();
    assert.deepEqual(Object.keys(counters).sort(), [
      'activeQueries', 'activeRowCount', 'activeSegmentBytes',
      'activeSnapshotLeases', 'cacheBytes', 'mappedBytes',
      'mappedResidentBytes', 'openFiles', 'queryPoolBytes', 'residentOwnedBytes',
      'segmentBytes', 'temporaryBytes', 'tombstoneBytes', 'tombstoneCount',
      'walBytes',
    ]);
    for (const [name, value] of Object.entries(counters)) {
      assert.equal(typeof value, 'bigint', name);
      assert.ok(value >= 0n, name);
    }
    if (process.platform === 'darwin') {
      assert.equal(typeof physicalFootprint, 'bigint');
      assert.ok(physicalFootprint > 0n);
    } else {
      assert.equal(physicalFootprint, null);
    }
    assert.equal(counters.activeRowCount, 0n);
    store.ingest([
      { id: 1n, vector: new Float32Array([1, 0]) },
      { id: 2n, vector: new Float32Array([0, 1]) },
    ], 2);
    assert.equal(store.stats().activeRowCount, 2n);
    store.seal();
    const sealed = store.stats();
    assert.equal(sealed.activeRowCount, 0n);
    assert.ok(sealed.segmentBytes > 0n);
  } finally {
    store.close();
  }
  assert.throws(() => store.stats(), isClosedError);
});
