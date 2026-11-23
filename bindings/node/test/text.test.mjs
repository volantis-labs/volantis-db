import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
const { TextStore, VolantisError } = createRequire(import.meta.url)('..');
const bundle = fileURLToPath(new URL('../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem', import.meta.url));

test('TextStore exposes the text API', () => {
  assert.equal(typeof TextStore, 'function');
  for (const name of ['open', 'openAsync', 'openModel']) assert.equal(typeof TextStore[name], 'function');
  for (const name of ['ingest', 'ingestAsync', 'query', 'queryAsync', 'delete', 'maintain', 'health', 'observedComputeUnits', 'observedComputeUnitsAsync', 'close']) {
    assert.equal(typeof TextStore.prototype[name], 'function', name);
  }
});

test('text dense lexical hybrid ingest and shorter revision', {skip: process.platform !== 'darwin' || process.arch !== 'arm64'}, async () => {
  const root = mkdtempSync(join(tmpdir(), 'ze510-'));
  const store = await TextStore.openAsync(join(root, 'store'), bundle);
  try {
    const report = await store.ingestAsync([{id: 1n, revision: 1n, text: 'bronze zeppelin bronze zeppelin bronze zeppelin bronze zeppelin'}], {chunkMax: 4, chunkOverlap: 0});
    assert.ok(report.chunks > 1n);
    for (const legs of ['dense', 'lexical', 'hybrid']) {
      const result = await store.queryAsync({text: 'bronze', k: 5, legs});
      assert.ok(result.hits.length > 0, legs);
      assert.equal(result.hits[0].id, 1n);
      assert.equal(result.hits[0].revision, 1n);
    }
    store.ingest([{id: 1n, revision: 2n, text: 'bronze'}], {chunkMax: 4, chunkOverlap: 0});
    const result = store.query({text: 'bronze', k: 10, legs: 'lexical'});
    assert.equal(result.hits.length, 1);
    assert.equal(result.hits[0].revision, 2n);
    assert.equal(result.hits[0].chunk, 0);
  } finally { store.close(); rmSync(root, {recursive: true, force: true}); }
});

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
const supported = process.platform === 'darwin' && process.arch === 'arm64';
const id = 1n;
async function fixture(run) {
  const root = mkdtempSync(join(tmpdir(), 'ze510-text-'));
  let store;
  try { store = TextStore.open(join(root, 'store'), bundle); await run(store, root); }
  finally {
    try { store?.close(); } catch (error) { if (error.code !== 'ZE_ERR_CLOSED') throw error; }
    rmSync(root, {recursive: true, force: true});
  }
}

test('text query returns the v2 outcome in sync and async calls', {skip: !supported}, async () => {
  const previous = process.env.ZE_QUERY_COREML;
  process.env.ZE_QUERY_COREML = 'off';
  try {
    await fixture(async store => {
      const {generation} = store.ingest([{id, text: 'bronze zeppelin'}]);
      for (const method of ['query', 'queryAsync']) {
        for (const legs of ['lexical', 'dense', 'hybrid']) {
          const result = await store[method]({text: 'bronze', legs, tier: 'exact'});
          assert.equal(result.hits[0].id, id);
          assert.equal(typeof result.embeddingEpoch, 'bigint');
          assert.equal(typeof result.tokenizerEpoch, 'bigint');
          const {diagnostics, timings, backend, queryTokens, embeddingCalls} = result;
          assert.equal(diagnostics?.generation, generation);
          assert.equal(diagnostics.mode, legs === 'dense' ? 'vector' : legs);
          assert.equal(diagnostics.approximate, false);
          assert.equal(diagnostics.budgetExhausted, false);
          assert.equal(typeof diagnostics.exactRescore, 'boolean');
          for (const counter of ['dimsTouched', 'bytesRead', 'docsEvaluated', 'postingsDecoded']) {
            assert.equal(typeof diagnostics[counter], 'bigint', counter);
          }
          if (legs !== 'dense') assert.equal(diagnostics.tokenizerEpoch, result.tokenizerEpoch);
          if (legs === 'hybrid') {
            assert.ok(['convex', 'reciprocalRank'].includes(diagnostics.fusion?.method));
            assert.equal(typeof diagnostics.fusion.rounds, 'bigint');
            assert.ok(Number.isFinite(diagnostics.fusion.effectiveAlpha));
          } else {
            assert.equal(diagnostics.fusion, undefined);
          }
          if (legs === 'lexical') {
            assert.equal(backend, undefined);
            assert.equal(queryTokens, 0n);
            assert.equal(embeddingCalls, 0n);
          } else {
            assert.equal(backend?.runtimeName, 'mlx-c');
            assert.equal(typeof backend.gpu, 'boolean');
            assert.equal(backend.requestedComputeUnits, backend.gpu ? 2 : 1);
            assert.equal(backend.observedComputeUnits, undefined);
            assert.equal(backend.sequenceLength, undefined);
            assert.equal(typeof queryTokens, 'bigint');
            assert.ok(queryTokens > 0n);
            assert.equal(embeddingCalls, 1n);
          }
          if (process.env.ZE_TEST_QUERY_TIMING === '1') assert.ok(timings);
          if (timings !== undefined) {
            for (const stage of ['tokenizationNs', 'lexicalAnalysisNs', 'embeddingQueueNs',
              'embeddingEvaluationNs', 'embeddingNormalizationNs', 'retrievalNs',
              'materializationNs', 'endToEndNs']) {
              assert.equal(typeof timings[stage], 'bigint', stage);
            }
            if (legs === 'lexical') assert.equal(timings.embeddingEvaluationNs, 0n);
          }
        }
        const empty = await store[method]({text: 'bronze', k: 0});
        assert.deepEqual(empty.hits, []);
        assert.equal(empty.diagnostics, undefined);
        assert.equal(empty.backend, undefined);
        assert.equal(empty.queryTokens, 0n);
        assert.equal(empty.embeddingCalls, 0n);
      }
      const options = {text: 'bronze', legs: 'lexical', signal: AbortSignal.abort()};
      const cancelled = error => error instanceof VolantisError &&
        error.code === 'ZE_ERR_CANCELLED' && error.errorCode === 12;
      assert.throws(() => store.query(options), cancelled);
      await assert.rejects(store.queryAsync(options), cancelled);
    });
  } finally {
    if (previous === undefined) delete process.env.ZE_QUERY_COREML;
    else process.env.ZE_QUERY_COREML = previous;
  }
});

test('text ingest accepts and validates maintenance budgets in sync and async calls', {skip: !supported}, () =>
  fixture(async store => {
    const docs = [{id, text: 'bronze zeppelin'}];
    const before = store.health().generation;
    for (const field of ['maintenanceWallTimeNs', 'maintenanceBytes']) {
      const invalid = error => error instanceof RangeError &&
        error.code === 'ERR_OUT_OF_RANGE' && error.message.includes(field);
      for (const value of [-1n, 1n << 64n, 256]) {
        assert.throws(() => store.ingest(docs, {[field]: value}), invalid);
        await assert.rejects(store.ingestAsync(docs, {[field]: value}), invalid);
      }
    }
    assert.equal(store.health().generation, before);
    const budgets = {maintenanceWallTimeNs: 100000000n, maintenanceBytes: 67108864n};
    const sync = store.ingest(docs, budgets);
    assert.equal(sync.documents, 1n);
    const async = await store.ingestAsync([{...docs[0], revision: 2n}], budgets);
    assert.equal(async.documents, 1n);
    assert.ok(async.generation > sync.generation);
  }));

test('text observed compute units loads the tiny CoreML plan (ZE-521)', {skip: !supported}, async t => {
  const [major, minor] = execFileSync('sw_vers', ['-productVersion'], {encoding: 'utf8'}).trim().split('.').map(Number);
  if (major < 14 || (major === 14 && minor < 4)) {
    t.skip('MLComputePlan requires macOS 14.4 or later');
    return;
  }
  const root = mkdtempSync(join(tmpdir(), 'ze521-compute-'));
  const previous = process.env.ZE_QUERY_COREML;
  try {
    const model = fileURLToPath(new URL('../../../crates/volantis-db-text/tests/fixtures/ze115/tiny.mlmodel', import.meta.url));
    execFileSync('xcrun', ['coremlcompiler', 'compile', model, root]);
    process.env.ZE_QUERY_COREML = join(root, 'tiny.mlmodelc');
    const computeBundle = fileURLToPath(new URL('../../../crates/volantis-db-ffi/tests/fixtures/compute_units.zem', import.meta.url));
    // Each variant must load the plan on a fresh store.
    for (const method of ['observedComputeUnits', 'observedComputeUnitsAsync']) {
      const store = TextStore.open(join(root, method), computeBundle);
      try {
        const counts = await store[method]();
        assert.notEqual(counts, null, method);
        assert.deepEqual(Object.keys(counts).sort(), ['cpuOps', 'gpuOps', 'neuralEngineOps', 'otherOps']);
        assert.equal(counts.cpuOps + counts.gpuOps + counts.neuralEngineOps + counts.otherOps, 1n);
      } finally { store.close(); }
    }
  } finally {
    if (previous === undefined) delete process.env.ZE_QUERY_COREML; else process.env.ZE_QUERY_COREML = previous;
    rmSync(root, {recursive: true, force: true});
  }
});

test('text observed compute units is absent without CoreML (ZE-521)', {skip: !supported}, async () => {
  const previous = process.env.ZE_QUERY_COREML;
  process.env.ZE_QUERY_COREML = 'off';
  try {
    await fixture(async store => {
      assert.equal(store.observedComputeUnits(), null);
      assert.equal(await store.observedComputeUnitsAsync(), null);
      store.close();
      assert.throws(() => store.observedComputeUnits(), error => error.code === 'ZE_ERR_CLOSED');
      await assert.rejects(store.observedComputeUnitsAsync(), error => error.code === 'ZE_ERR_CLOSED');
    });
  } finally {
    if (previous === undefined) delete process.env.ZE_QUERY_COREML; else process.env.ZE_QUERY_COREML = previous;
  }
});

test('lexical and hybrid return zero hits on fresh and deleted stores (ZE-518)', {skip: !supported}, () => fixture(async store => {
  for (const legs of ['lexical', 'hybrid']) {
    assert.deepEqual(store.query({text: 'bronze', k: 5, legs}).hits, []);
    assert.deepEqual((await store.queryAsync({text: 'bronze', k: 5, legs})).hits, []);
  }
  store.ingest([{id, revision: 1n, text: 'bronze zeppelin'}]);
  const removed = await store.deleteAsync([id]);
  assert.equal(typeof removed.generation, 'bigint');
  for (const legs of ['lexical', 'hybrid']) {
    assert.deepEqual(store.query({text: 'bronze', k: 5, legs}).hits, []);
    assert.deepEqual((await store.queryAsync({text: 'bronze', k: 5, legs})).hits, []);
  }
}));

test('text health maintenance cancellation and worker lifetime', {skip: !supported}, () => fixture(async store => {
  assert.equal(typeof store.health().generation, 'bigint');
  assert.deepEqual(await store.healthAsync(), store.health());
  store.ingest([{id, revision: 1n, text: 'bronze zeppelin'}]);
  const maintained = await store.maintainAsync({wallTimeNs: 100000000n, bytes: 67108864n});
  assert.ok([0, 1].includes(maintained.status));
  assert.equal(typeof maintained.graphsBuilt, 'bigint');
  assert.ok([0, 1].includes(store.maintain({wallTimeNs: 100000000n, bytes: 67108864n}).status));
  assert.equal(store.health().unresolvedFaultCount, 0n);
  const c = new AbortController(); c.abort();
  assert.throws(() => store.ingest([{id, revision: 2n, text: 'bronze'}], {signal: c.signal}), error => error.name === 'VolantisError' && error.code === 'ZE_ERR_INVALID_ARGUMENT' && error.errorCode === 1);
  const cancelled = error => error.code === 'ZE_ERR_CANCELLED';
  await assert.rejects(store.queryAsync({text: 'bronze', legs: 'lexical', signal: c.signal}), cancelled);
  assert.throws(() => store.query({text: 'bronze', legs: 'lexical', signal: c.signal}), cancelled);
  await assert.rejects(store.ingestAsync([{id, revision: 2n, text: 'bronze'}], {signal: c.signal}), cancelled);
  await assert.rejects(store.queryAsync({text: 'bronze', deadlineNs: 1n}), error => error.code === 'ZE_ERR_TIMEOUT');
  const pending = store.queryAsync({text: 'bronze', legs: 'lexical'});
  store.close();
  await pending.catch(error => assert.equal(error.code, 'ZE_ERR_CLOSED'));
  await assert.rejects(store.queryAsync({text: 'bronze'}), error => error.code === 'ZE_ERR_CLOSED');

}));

test('openModel downloads pinned local fixture with progress and opens offline cache', {skip: !supported}, async () => {
  const root = mkdtempSync(join(tmpdir(), 'ze510-model-'));
  const bytes = readFileSync(bundle);
  const sha = bytes => createHash('sha256').update(bytes).digest('hex');
  const manifest = Buffer.from(JSON.stringify({format: 1, name: 'tiny', bundle: 'tiny.zem', query_coreml: null,
    files: [{path: 'tiny.zem', size: bytes.length, sha256: sha(bytes)}]}));
  const requests = [];
  const server = createServer((request, response) => {
    requests.push(request.url);
    const body = request.url.endsWith('/volantis-model.json') ? manifest : bytes;
    response.writeHead(200, {'content-length': body.length}); response.end(body);
  });
  const previous = [process.env.ZE_MODEL_ENDPOINT, process.env.ZE_MODEL_ALLOW_HTTP];
  let store;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    process.env.ZE_MODEL_ENDPOINT = `http://127.0.0.1:${server.address().port}`;
    process.env.ZE_MODEL_ALLOW_HTTP = '1';
    const options = {path: join(root, 'store'), repo: 'fixture/tiny', revision: '0123456789abcdef0123456789abcdef01234567',
      manifestSha256: sha(manifest), cacheDir: join(root, 'cache')};
    const events = [];
    store = await TextStore.openModel({...options, onProgress: event => events.push(event)});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, 2, 'local JS HTTP server serviced the native download without blocking');
    assert.deepEqual(events[0], {done: 0n, total: 0n});
    assert.deepEqual(events.at(-1), {done: BigInt(bytes.length), total: BigInt(bytes.length)});
    await store.ingestAsync([{id, revision: 1n, text: 'bronze zeppelin'}]);
    assert.equal((await store.queryAsync({text: 'bronze', legs: 'lexical'})).hits[0].id, id);
    store.close(); store = undefined;
    store = await TextStore.openModel({...options, offline: true});
    assert.equal(requests.length, 2);
    assert.equal(store.query({text: 'bronze', legs: 'lexical'}).hits[0].id, id);
    store.close(); store = undefined;
    const limit = BigInt(process.env.ZE_MLX_CACHE_LIMIT_BYTES ?? (4n << 30n));
    const other = {mlxCacheLimitBytes: limit ^ 1n};
    const conflict = error => error.code === 'ZE_ERR_MLX_CACHE_LIMIT_CONFLICT' && error.errorCode === 63;
    (await TextStore.openAsync(options.path, bundle, {mlxCacheLimitBytes: limit})).close();
    assert.throws(() => TextStore.open(options.path, bundle, other), conflict);
    await assert.rejects(TextStore.openAsync(options.path, bundle, other), conflict);
    await assert.rejects(TextStore.openModel({...options, ...other, offline: true}), conflict);
    await assert.rejects(TextStore.openModel({...options, path: join(root, 'miss'), cacheDir: join(root, 'missing'), offline: true}),
      error => error.code === 'ZE_ERR_NOT_FOUND' && /offline/i.test(error.message));
  } finally {
    store?.close(); await new Promise(resolve => server.close(resolve));
    for (const [i, key] of ['ZE_MODEL_ENDPOINT', 'ZE_MODEL_ALLOW_HTTP'].entries()) {
      if (previous[i] === undefined) delete process.env[key]; else process.env[key] = previous[i];
    }
    rmSync(root, {recursive: true, force: true});
  }
});

// Both macOS prebuilds carry text since WF-991; only other platforms are text-free.
test('text-free prebuild reports a typed platform error', {skip: process.platform === 'darwin'}, async () => {
  const unavailable = error => error.name === 'VolantisError' && error.code === 'ZE_ERR_UNSUPPORTED' && /text API is available on macOS only/.test(error.message);
  assert.throws(() => TextStore.open('unused', 'unused'), unavailable);
  await assert.rejects(TextStore.openAsync('unused', 'unused'), unavailable);
  await assert.rejects(TextStore.openModel({}), unavailable);
});

test('openModel offline miss is a typed not-found error without loading MLX', {skip: !supported}, async () => {
  const root = mkdtempSync(join(tmpdir(), 'ze510-offline-'));
  try {
    await assert.rejects(TextStore.openModel({path: join(root, 'store'), repo: 'fixture/tiny',
      revision: '0123456789abcdef0123456789abcdef01234567', manifestSha256: '00'.repeat(32),
      cacheDir: join(root, 'empty-cache'), offline: true}),
    error => error.name === 'VolantisError' && error.code === 'ZE_ERR_NOT_FOUND' && /offline/i.test(error.message));
  } finally { rmSync(root, {recursive: true, force: true}); }
});


test('default model matches Rust constant and requires complete pins', {skip: !supported}, async () => {
  const native = createRequire(import.meta.url)('../prebuilds/darwin-arm64/volantis_db.node');
  assert.deepEqual(TextStore.defaultModel, native.textDefaultModel());
  for (const [key, value] of Object.entries(TextStore.defaultModel)) {
    await assert.rejects(TextStore.openModel({path: '/unused', [key]: value, offline: true}), /all three/);
  }
  const root = mkdtempSync(join(tmpdir(), 'ze500-'));
  try {
    await assert.rejects(TextStore.openModel({path: join(root, 'store'), cacheDir: join(root, 'cache'), offline: true}),
      error => error.code === 'ZE_ERR_NOT_FOUND');
  } finally { rmSync(root, {recursive: true, force: true}); }
});

test('all text opens validate mlxCacheLimitBytes as an unsigned bigint', {skip: !supported}, async () => {
  for (const mlxCacheLimitBytes of [-1n, 1n << 64n, 256]) {
    const options = {mlxCacheLimitBytes};
    const invalid = /mlxCacheLimitBytes must be an unsigned 64-bit bigint/;
    assert.throws(() => TextStore.open('/unused', '/missing.zem', options), invalid);
    await assert.rejects(TextStore.openAsync('/unused', '/missing.zem', options), invalid);
    await assert.rejects(TextStore.openModel({path: '/unused', offline: true, ...options}), invalid);
  }
});

// WF-987: opt-in loopback rehearsal of a release folder before upload (WF987_RELEASE).
test('WF-987 release model downloads, verifies and reopens offline', {skip: !supported || !process.env.WF987_RELEASE}, async () => {
  const {serveRelease, useEndpoint, DOCUMENTS, QUERY} = await import('./fixtures/wf987-release-server.mjs');
  const server = await serveRelease();
  const restore = useEndpoint(server.url);
  const root = mkdtempSync(join(tmpdir(), 'wf987-model-'));
  const options = {path: join(root, 'store'), ...server.pin, cacheDir: join(root, 'cache')};
  const bytes = server.manifest.files.reduce((sum, file) => sum + BigInt(file.size), 0n);
  let store;
  try {
    const events = [];
    const started = performance.now();
    store = await TextStore.openModel({...options, onProgress: event => events.push(event)});
    const seconds = (performance.now() - started) / 1000;
    console.log(`WF-987 loopback download and verify: ${bytes} bytes in ${seconds.toFixed(1)} s`);
    assert.deepEqual(server.requests.map(url => url.split('/').at(-1)).sort(),
      [...server.manifest.files.map(file => file.path), 'volantis-model.json'].sort());
    assert.deepEqual(events.at(-1), {done: bytes, total: bytes});
    await store.ingestAsync(DOCUMENTS);
    const online = await store.queryAsync({text: QUERY, legs: 'dense', k: 2});
    assert.deepEqual(online.hits.map(hit => hit.id), DOCUMENTS.map(doc => doc.id));
    assert.ok(online.hits[0].score > online.hits[1].score);
    assert.match(online.backend.runtimeName, /mlx/i);
    store.close(); store = undefined;
    const requests = server.requests.length;
    store = await TextStore.openModel({...options, offline: true});
    assert.equal(server.requests.length, requests, 'offline reopen makes no request');
    const offline = store.query({text: QUERY, legs: 'dense', k: 2});
    assert.deepEqual(offline.hits.map(hit => [hit.id, hit.score]), online.hits.map(hit => [hit.id, hit.score]));
  } finally {
    store?.close(); await server.close(); restore();
    rmSync(root, {recursive: true, force: true});
  }
});
