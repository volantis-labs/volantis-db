import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const {openNamespace, openNamespaceAsync, TextStore} = createRequire(import.meta.url)('../..');

// Waits for the background worker with a bounded deadline.
async function drained(writer, deadline = Date.now() + 60_000) {
  for (;;) {
    const status = writer.embeddingStatus();
    if (status.pending === 0n && !status.inFlight) return;
    assert.equal(status.lastErrorCode, 0);
    assert.ok(Date.now() < deadline, 'background embedding did not drain');
    await new Promise(resolve => setTimeout(resolve, 20));
  }
}
const root = mkdtempSync(join(tmpdir(), 'ze599-'));
const writers = [];
try {
  const revision = '0123456789abcdef0123456789abcdef01234567';
  process.env.ZE_MODEL_CACHE = join(root, 'cache');
  const entry = join(process.env.ZE_MODEL_CACHE, 'fixture%2Ftiny', revision);
  mkdirSync(entry, {recursive: true});
  const bytes = readFileSync(new URL('../../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem', import.meta.url));
  const sha = data => createHash('sha256').update(data).digest('hex');
  const manifest = Buffer.from(JSON.stringify({format: 1, name: 'tiny', bundle: 'tiny.zem', query_coreml: null,
    files: [{path: 'tiny.zem', size: bytes.length, sha256: sha(bytes)}]}));
  const bundle = join(entry, 'tiny.zem');
  writeFileSync(bundle, bytes);
  writeFileSync(join(entry, 'volantis-model.json'), manifest);
  const spec = {embeddedModel: {repo: 'fixture/tiny', revision, manifestSha256: sha(manifest)}};
  const conflict = {code: 'ZE_ERR_MLX_CACHE_LIMIT_CONFLICT', errorCode: 63};
  const opts = mlxCacheLimitBytes => ({mlxCacheLimitBytes});
  const text = limit => TextStore.open(join(root, 'text'), bundle, opts(limit));
  const mode = process.argv[2];
  if (mode === 'deferred') {
    for (const [name, open] of [['sync', openNamespace], ['async', openNamespaceAsync]]) {
      const writer = await open(root, name, {...spec, embeddingSchedule: 'paused'});
      writers.push(writer);
      const id = 60001n;
      const text = 'fictitious mauve sprocket';
      writer.upsert([{id, revision: 7n, text}]);
      assert.equal(writer.get([id]).documents[0].revision, 7n);
      const query = {text, leg: 'dense', eligibleIds: [id], k: 1};
      assert.deepEqual(writer.queryEmbedded(query).map(hit => hit.id), []);
      // The pending row is a lexical hit at once and counts as pending.
      const report = writer.queryEmbeddedReport({...query, leg: 'lexical'});
      assert.deepEqual(report.hits.map(hit => [hit.id, hit.state]), [[id, 'pending']]);
      assert.equal(report.pendingRows, 1n);
      const status = writer.embeddingStatus();
      assert.equal(status.paused, true);
      assert.equal(status.pending, 1n);
      writer.resumeEmbedding();
      await drained(writer);
      assert.deepEqual(writer.queryEmbedded(query).map(hit => hit.id), [id]);
      assert.equal(writer.queryEmbeddedReport(query).pendingRows, 0n);
    }
  } else if (mode === 'deferred-reopen') {
    const id = 60001n;
    const query = {text: 'mauve', leg: 'lexical', eligibleIds: [id], k: 1};
    let writer = openNamespace(root, 'reopen', {...spec, embeddingSchedule: 'paused'});
    writer.upsert([{id, revision: 7n, text: 'fictitious mauve sprocket'}]);
    writer.close();
    writer = openNamespace(root, 'reopen', {...spec, embeddingSchedule: 'paused'});
    const report = writer.queryEmbeddedReport(query);
    assert.deepEqual(report.hits.map(hit => [hit.id, hit.state]), [[id, 'pending']]);
    assert.equal(report.pendingRows, 1n);
    writer.close();
    writer = openNamespace(root, 'reopen', {...spec, embeddingSchedule: 'background'});
    writers.push(writer);
    await drained(writer);
    const dense = writer.queryEmbeddedReport({...query, leg: 'dense'});
    assert.deepEqual(dense.hits.map(hit => [hit.id, hit.state]), [[id, 'ready']]);
  } else if (mode === 'race') {
    const results = await Promise.allSettled(['toy-a', 'toy-b'].map((name, i) =>
      openNamespaceAsync(root, name, spec, opts(BigInt(i)))));
    const won = results.filter(r => r.status === 'fulfilled');
    writers.push(...won.map(r => r.value));
    assert.equal(won.length, 1);
    const loser = results.find(r => r.status === 'rejected').reason;
    assert.equal(loser.code, conflict.code);
    assert.equal(loser.errorCode, conflict.errorCode);
  } else if (mode === 'validation') {
    for (const value of [123, -1n, 1n << 64n]) {
      const error = /mlxCacheLimitBytes must be an unsigned 64-bit bigint/;
      assert.throws(() => openNamespace(root, 'toy-a', spec, opts(value)), error);
      await assert.rejects(openNamespaceAsync(root, 'toy-b', spec, opts(value)), error);
    }
    for (const plain of [{}, {vectorSpace: {dimensions: 2}}]) {
      assert.throws(() => openNamespace(root, 'toy-a', plain, opts(0n)), /embeddedModel/);
      await assert.rejects(openNamespaceAsync(root, 'toy-b', plain, opts(0n)), /embeddedModel/);
    }
  } else if (mode === 'invalid-env') {
    assert.throws(() => openNamespace(root, 'toy-a', spec), /ZE_MLX_CACHE_LIMIT_BYTES/);
  } else {
    const implicit = mode === 'implicit';
    const limit = implicit ? BigInt(process.env.ZE_MLX_CACHE_LIMIT_BYTES ?? 4294967296) : 0n;
    const other = limit === 0n ? 8192n : 0n;
    if (mode === 'text-first') {
      text(limit).close();
      assert.throws(() => openNamespace(root, 'toy-a', spec, opts(other)), conflict);
    }
    const writer = openNamespace(root, 'toy-a', spec, implicit ? undefined : opts(limit));
    writers.push(writer);
    text(limit).close();
    assert.throws(() => text(other), conflict);
    (await openNamespaceAsync(root, 'toy-b', spec, opts(limit))).close();
    openNamespace(root, 'toy-b', spec, opts(undefined)).close();
    await assert.rejects(openNamespaceAsync(root, 'toy-b', spec, opts(other)), conflict);
    writer.upsert([{id: 42n, revision: 1n, text: 'bronze zeppelin'}]);
    assert.equal(writer.queryEmbedded({text: 'bronze', leg: 'dense', k: 1})[0].id, 42n);
    writers.pop().close();
    assert.throws(() => openNamespace(root, 'toy-b', spec, opts(other)), conflict);
  }
} finally {
  for (const writer of writers) writer.close();
  rmSync(root, {recursive: true, force: true});
}
