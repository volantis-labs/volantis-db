import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';

const {openNamespace,openNamespaceAsync,verify} = createRequire(import.meta.url)('..');
const options = {maxResidentBytes: 268435456n};

test('Store and Cypher report a missing namespace embedder', () => {
  const root = mkdtempSync(join(tmpdir(),'ze551-'));
  const store = openNamespace(root,'notes',{vectorSpace:{dimensions:2}},options);
  try {
    store.enableGraph();
    const missing = {code:'ZE_ERR_MISSING_EMBEDDER',errorCode:62};
    assert.throws(() => store.queryEmbedded({text:'bronze',leg:'dense'}),missing);
    for (const call of ["vector_search('bronze',2,'exact')","hybrid_search('bronze',2,'exact')"]) {
      assert.throws(() => store.cypher('CALL ze.'+call+' YIELD node RETURN node'),missing);
    }
  } finally {store.close();rmSync(root,{recursive:true,force:true});}
});

test('embedded namespace resolves a pinned model and searches chunks and graph nodes',
  {skip:process.platform !== 'darwin' || process.arch !== 'arm64'}, async () => {
  const root = mkdtempSync(join(tmpdir(),'ze551-model-'));
  const original = process.env.ZE_MODEL_CACHE;
  let store;
  try {
    const bytes = readFileSync(new URL('../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem',import.meta.url));
    const sha = b => createHash('sha256').update(b).digest('hex');
    const manifest = Buffer.from(JSON.stringify({format:1,name:'tiny',bundle:'tiny.zem',query_coreml:null,
      files:[{path:'tiny.zem',size:bytes.length,sha256:sha(bytes)}]}));
    const revision = '0123456789abcdef0123456789abcdef01234567';
    process.env.ZE_MODEL_CACHE = join(root,'cache');
    const entry = join(process.env.ZE_MODEL_CACHE,'fixture%2Ftiny',revision);
    mkdirSync(entry,{recursive:true});
    writeFileSync(join(entry,'volantis-model.json'),manifest);
    writeFileSync(join(entry,'tiny.zem'),bytes);
    const spec = {embeddedModel:{repo:'fixture/tiny',revision,manifestSha256:sha(manifest)},
      attributes:[{id:1,name:'speaker',type:'u64',nullable:false}]};
    store = await openNamespaceAsync(root,'notes',spec,options);
    store.enableGraph();
    const id=(1n<<120n)+42n;
    const attributes=[{id:1,type:'u64',value:7n}];
    const source='bronze zeppelin '.repeat(600);
    await store.upsertAsync([{id,revision:3n,text:source,attributes}]);
    for (const leg of ['dense','lexical','hybrid']) {
      const hits=await store.queryEmbeddedAsync({text:'bronze',leg,k:2,eligibleIds:[id],filter:{op:'eq',attributeId:1,values:attributes}});
      assert.equal(hits.length,2);
      for (const hit of hits) {
        assert.equal(hit.id,id);
        assert.equal(hit.revision,3n);
        assert.deepEqual(hit.attributes,attributes);
        assert.equal(Buffer.from(source).subarray(hit.byteStart,hit.byteEnd).toString(),hit.text);
      }
    }
    const view = store.openSnapshot();
    try {
      assert.equal(view.queryEmbedded({text:'bronze',leg:'dense',k:1})[0].id,id);
      assert.deepEqual(view.nearestChunks(id), []);
    } finally { view.close(); }
    const backup = join(root,'backup');
    await store.backupAsync(backup);
    assert.deepEqual(verify(backup).findings,[]);
    for (const call of ["vector_search('bronze',1,'exact')","hybrid_search('bronze',1,'exact')"]) {
      assert.deepEqual(store.cypher('CALL ze.'+call+' YIELD node RETURN ze.node_id(node)').rows,[[id.toString(16).padStart(32,'0')]]);
    }

    const other = id + 1n;
    await store.upsertAsync([{id: other, revision: 1n, text: 'bronze zeppelin', attributes}]);
    const opts = {k: 2, eligibleIds: [id], filter: {op:'eq', attributeId:1, values:attributes}};
    const hits = store.nearestChunks({id: other, chunk: 0}, opts);
    assert.equal(hits.length, 2);
    assert.ok(hits.every(hit => hit.id === id && Number.isFinite(hit.score)));
    assert.ok(hits[0].score >= hits[1].score);
    assert.deepEqual(await store.nearestChunksBatchAsync([id, other, other], opts), [[], hits, hits]);
    assert.deepEqual(store.nearestChunksBatch([id, other], {threshold: 0}), [[], []]);
    assert.throws(() => store.nearestChunks(other + 1n), {code:'ZE_ERR_NOT_FOUND'});
  } finally {
    store?.close();
    if(original===undefined)delete process.env.ZE_MODEL_CACHE;else process.env.ZE_MODEL_CACHE=original;
    rmSync(root,{recursive:true,force:true});
  }
});

test('stored chunk neighbours expose single and cancellable batch calls', async () => {
  const root = mkdtempSync(join(tmpdir(), 'ze555-'));
  const store = openNamespace(root, 'notes', {vectorSpace: {dimensions: 2}}, options);
  try {
    assert.throws(() => store.nearestChunks(1n), {code: 'ZE_ERR_MISSING_EMBEDDER'});
    await assert.rejects(store.nearestChunksBatchAsync([1n, 2n], {signal: AbortSignal.abort()}),
      {code: 'ZE_ERR_CANCELLED'});
    assert.throws(() => store.nearestChunksBatch([1n], {k: 0}), RangeError);
  } finally {
    store.close();
    rmSync(root, {recursive: true, force: true});
  }
});

test('shared namespace model survives live batches and writer close',
  {skip: process.platform !== 'darwin' || process.arch !== 'arm64'}, () => {
  const {namespaceBatchLive} = createRequire(import.meta.url)('..');
  const root = mkdtempSync(join(tmpdir(), 'ze598-'));
  const originalCache = process.env.ZE_MODEL_CACHE;
  const writers = [];
  let view;
  try {
    process.env.ZE_MODEL_CACHE = join(root, 'cache');
    const bytes = readFileSync(new URL('../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem', import.meta.url));
    const sha = data => createHash('sha256').update(data).digest('hex');
    const manifest = Buffer.from(JSON.stringify({
      format: 1, name: 'tiny', bundle: 'tiny.zem', query_coreml: null,
      files: [{path: 'tiny.zem', size: bytes.length, sha256: sha(bytes)}],
    }));
    const [spec, other] = ['1', '2'].map(digit => {
      const revision = digit.repeat(40);
      const entry = join(process.env.ZE_MODEL_CACHE, 'fixture%2Ffake-ze598', revision);
      mkdirSync(entry, {recursive: true});
      writeFileSync(join(entry, 'volantis-model.json'), manifest);
      writeFileSync(join(entry, 'tiny.zem'), bytes);
      return {embeddedModel: {repo: 'fixture/fake-ze598', revision, manifestSha256: sha(manifest)}};
    });

    const ids = [59801n, 59802n];
    const texts = ['fixture copper kite', 'fixture violet spool'];
    const changes = ['fake_notes', 'fake_segments'].map((name, index) => {
      const store = openNamespace(root, name, spec, options);
      writers.push(store);
      return {store, name, spec, upserts: [{id: ids[index], revision: 1n, text: texts[index]}]};
    });
    const generations = namespaceBatchLive(root, changes);
    function checkRows() {
      for (const [index, writer] of writers.entries()) {
        const rows = writer.get(ids);
        assert.equal(rows.generation, generations[index]);
        assert.deepEqual(rows.documents.filter(Boolean).map(doc => doc.id), [ids[index]]);
        assert.equal(rows.documents[index].revision, 1n);
      }
    }
    checkRows();
    const rejected = changes.map((change, index) => ({
      ...change, spec: index === 1 ? other : spec,
      upserts: [{id: ids[index], revision: 2n, text: 'changed'}],
    }));
    assert.throws(() => namespaceBatchLive(root, rejected), {code: 'ZE_ERR_EPOCH_MISMATCH'});
    checkRows();

    const originalBackend = process.env.ZE_QUERY_COREML;
    const invalidBackend = join(root, 'invalid.mlmodelc');
    mkdirSync(invalidBackend);
    try {
      process.env.ZE_QUERY_COREML = invalidBackend;
      assert.throws(() => openNamespace(root, 'backend_check', spec, options));
    } finally {
      if (originalBackend === undefined) delete process.env.ZE_QUERY_COREML;
      else process.env.ZE_QUERY_COREML = originalBackend;
    }

    view = writers[1].openSnapshot();
    writers.pop().close();
    writers[0].upsert([{id: 59803n, text: 'fresh copper'}]);
    assert.equal(view.queryEmbedded({text: 'spool', leg: 'dense', k: 1})[0].id, ids[1]);
    view.close();
    view = undefined;
    writers.pop().close();

    writers.push(openNamespace(root, 'fake_notes', spec, options));
    assert.equal(writers[0].get([59803n]).documents[0].id, 59803n);
  } finally {
    view?.close();
    for (const writer of writers) writer.close();
    if (originalCache === undefined) delete process.env.ZE_MODEL_CACHE;
    else process.env.ZE_MODEL_CACHE = originalCache;
    rmSync(root, {recursive: true, force: true});
  }
});

const nativeMetal = process.platform === 'darwin' && process.arch === 'arm64';

function runCacheCase(mode, envValue) {
  const env = {...process.env, ZE_MODEL_OFFLINE: '1'};
  if (envValue === undefined) delete env.ZE_MLX_CACHE_LIMIT_BYTES;
  else env.ZE_MLX_CACHE_LIMIT_BYTES = envValue;
  const result = spawnSync(process.execPath, [fileURLToPath(new URL('./fixtures/namespace-cache.mjs', import.meta.url)), mode],
    {env, encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr);
}

test('embedded namespace open sets the process MLX cache limit', {skip: !nativeMetal}, () => {
  runCacheCase('namespace-first', '8192');
  runCacheCase('text-first', '8192');
  runCacheCase('implicit');
  runCacheCase('implicit', '8192');
  runCacheCase('invalid-env', 'invalid');
  runCacheCase('namespace-first', 'invalid');
});

test('namespace open validates the optional MLX byte count', {skip: !nativeMetal}, () => {
  runCacheCase('validation');
  runCacheCase('race');
});

test('WF-989 opens from the requested cache offline', {skip: !nativeMetal}, async () => {
  const {runCase} = await import('./fixtures/wf989-model-cache.mjs');
  runCase('open');
  runCase('mlx-zero');
  runCase('mlx-implicit', {ZE_MLX_CACHE_LIMIT_BYTES: '8192'});
});

test('deferred embedded namespace commits before inference', {skip: !nativeMetal}, () => {
  runCacheCase('deferred');
});

test('deferred paused writes survive reopen', {skip: !nativeMetal}, () => {
  runCacheCase('deferred-reopen');
});

// WF-987: opt-in loopback rehearsal of a release folder before upload (WF987_RELEASE).
test('WF-987 release model in an embedded namespace', {skip: !nativeMetal || !process.env.WF987_RELEASE}, async () => {
  const {serveRelease, useEndpoint, DOCUMENTS, QUERY} = await import('./fixtures/wf987-release-server.mjs');
  const server = await serveRelease();
  const restore = useEndpoint(server.url);
  const root = mkdtempSync(join(tmpdir(), 'wf987-namespace-'));
  const spec = {embeddedModel: server.pin};
  const modelCacheDir = join(root, 'cache');
  let store;
  try {
    // Online open downloads into the requested cache and verifies the pin.
    store = await openNamespaceAsync(join(root, 'store'), 'notes', spec, {...options, modelCacheDir, offline: false});
    store.close(); store = undefined;
    const requests = server.requests.length;
    assert.ok(requests >= 2);
    process.env.ZE_MODEL_OFFLINE = '1';
    store = openNamespace(join(root, 'store'), 'notes', spec, {...options, modelCacheDir, offline: true});
    await store.upsertAsync(DOCUMENTS);
    const hits = store.queryEmbedded({text: QUERY, leg: 'dense', k: 2});
    assert.deepEqual(hits.map(hit => hit.id), DOCUMENTS.map(doc => doc.id));
    assert.equal(server.requests.length, requests, 'offline open makes no request');
  } finally {
    store?.close(); await server.close(); restore();
    rmSync(root, {recursive: true, force: true});
  }
});

test('WF-988 empty embedded rows', {skip: !nativeMetal}, async () => {
  const root = mkdtempSync(join(tmpdir(), 'wf988-'));
  const original = process.env.ZE_MODEL_CACHE;
  let store;
  try {
    const bytes = readFileSync(new URL('../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem', import.meta.url));
    const real = readFileSync(new URL('../../../tests/fixtures/wf988/real-source.txt', import.meta.url), 'utf8');
    const sha = data => createHash('sha256').update(data).digest('hex');
    const manifest = Buffer.from(JSON.stringify({format: 1, name: 'tiny', bundle: 'tiny.zem', query_coreml: null,
      files: [{path: 'tiny.zem', size: bytes.length, sha256: sha(bytes)}]}));
    const revision = '0123456789abcdef0123456789abcdef01234567';
    process.env.ZE_MODEL_CACHE = join(root, 'cache');
    const entry = join(process.env.ZE_MODEL_CACHE, 'fixture%2Ftiny', revision);
    mkdirSync(entry, {recursive: true});
    writeFileSync(join(entry, 'volantis-model.json'), manifest);
    writeFileSync(join(entry, 'tiny.zem'), bytes);
    const spec = {embeddedModel: {repo: 'fixture/tiny', revision, manifestSha256: sha(manifest)},
      attributes: [{id: 1, name: 'fake_group', type: 'u64', nullable: false}]};
    store = await openNamespaceAsync(root, 'notes', spec, options);
    store.enableGraph();
    const attributes = [{id: 1, type: 'u64', value: 988n}];
    const filter = {op: 'eq', attributeId: 1, values: attributes};
    const metadata = Buffer.from('FAKE-988-meta');
    const row = (id, rev, text) => ({id, revision: rev, timestamp: 98800n, attributes, metadata,
      ...(text === undefined ? {} : {text})});
    const [omitted, zero, control] = [98801n, 98802n, 98803n];
    const fields = {vector: true, text: true, metadata: true, attributes: true};
    const assertEmpty = (id, rev) => {
      const [doc] = store.get([id], fields).documents;
      assert.equal(doc.revision, rev);
      assert.equal(doc.timestamp, 98800n);
      assert.deepEqual(doc.attributes, attributes);
      assert.deepEqual(Buffer.from(doc.metadata), metadata);
      assert.equal(doc.vector, undefined);
      assert.equal(doc.text, undefined);  // '' crosses the C ABI as absent text
      assert.equal(store.get([id], {vector: true}).documents[0].vector, undefined);
    };
    // Sync and async writes, with omitted text and the empty string.
    store.upsert([row(omitted, 1n), row(zero, 1n, '')]);
    await store.upsertAsync([row(omitted, 2n, ''), row(zero, 2n)]);
    assertEmpty(omitted, 2n);
    assertEmpty(zero, 2n);
    assert.deepEqual(store.scan({filter, fields}).documents.map(doc => doc.id), [omitted, zero]);

    // Populate the control and one empty id, then clear that id again.
    await store.upsertAsync([row(control, 1n, real), row(zero, 3n, real)]);
    const vector = store.get([control], {vector: true}).documents[0].vector;
    assert.ok(vector instanceof Float32Array && vector.length > 0);
    assert.ok(store.get([zero], {vector: true}).documents[0].vector instanceof Float32Array);
    store.upsert([row(zero, 4n, '')]);
    assertEmpty(zero, 4n);
    assert.deepEqual(store.scan({filter, fields}).documents.map(doc => doc.id).sort(), [omitted, zero, control]);

    // k exceeds the live vector count: only the control can rank.
    assert.deepEqual(store.search(vector, 64).map(hit => hit.id), [control]);
    for (const leg of ['dense', 'lexical', 'hybrid']) {
      const hits = store.queryEmbedded({text: 'FAKE-988-copper', leg, k: 64});
      assert.ok(hits.length > 0, leg);
      assert.ok(hits.every(hit => hit.id === control), leg);
      assert.deepEqual(store.queryEmbedded({text: 'FAKE-988-copper', leg, k: 64, eligibleIds: [omitted, zero]}), [], leg);
    }
    const id = value => value.toString(16).padStart(32, '0');
    for (const call of ["vector_search('FAKE-988-copper',64,'exact')", "hybrid_search('FAKE-988-copper',64,'exact')"]) {
      assert.deepEqual(store.cypher('CALL ze.' + call + ' YIELD node RETURN ze.node_id(node)').rows, [[id(control)]], call);
    }

    // Neighbours: empty candidates add nothing; an empty source is not found.
    assert.deepEqual(store.nearestChunks(control, {k: 64, eligibleIds: [omitted, zero]}), []);
    const own = store.nearestChunks(control, {k: 64, includeSource: true});
    assert.ok(own.length > 0 && own.every(hit => hit.id === control));
    for (const source of [omitted, {id: zero, chunk: 0}]) {
      assert.throws(() => store.nearestChunks(source), {code: 'ZE_ERR_NOT_FOUND'});
    }
    await assert.rejects(store.nearestChunksBatchAsync([control, omitted]), {code: 'ZE_ERR_NOT_FOUND'});
    const backup = join(root, 'backup');
    await store.backupAsync(backup);
    assert.deepEqual(verify(backup).findings, []);
  } finally {
    store?.close();
    if (original === undefined) delete process.env.ZE_MODEL_CACHE; else process.env.ZE_MODEL_CACHE = original;
    rmSync(root, {recursive: true, force: true});
  }
});

// WF-991: both macOS prebuilds carry text. arm64 runs MLX; x64 runs the fp32
// CPU runtime from the same bundle. On darwin-x64 a text-free addon fails here.
const textSupported = process.platform === 'darwin' &&
  createRequire(import.meta.url)(`../prebuilds/darwin-${process.arch}/volantis_db.node`).textSupported;

function pinnedCache(root, repo, files, revision = '9'.repeat(40)) {
  const sha = data => createHash('sha256').update(data).digest('hex');
  const [name] = Object.keys(files)[0].split('.');
  const manifest = Buffer.from(JSON.stringify({format: 1, name, bundle: `${name}.zem`, query_coreml: null,
    files: Object.entries(files).map(([path, data]) => ({path, size: data.length, sha256: sha(data)}))}));
  const entry = join(root, repo.replace('/', '%2F'), revision);
  mkdirSync(entry, {recursive: true});
  writeFileSync(join(entry, 'volantis-model.json'), manifest);
  for (const [path, data] of Object.entries(files)) writeFileSync(join(entry, path), data);
  return {repo, revision, manifestSha256: sha(manifest)};
}

test('wf991 portable pinned namespace', {skip: process.platform !== 'darwin'}, async () => {
  assert.equal(textSupported, true, `darwin-${process.arch} prebuild must carry text`);
  const root = mkdtempSync(join(tmpdir(), 'wf991-'));
  const modelCacheDir = join(root, 'cache');
  // WF-986 tiny BERT pair: the CPU runtime runs BERT towers only.
  const bytes = readFileSync(new URL('../../../crates/volantis-db-text/tests/fixtures/wf986/pair.zem', import.meta.url));
  const spec = {embeddedModel: pinnedCache(modelCacheDir, 'fixture/fake-wf991', {'tiny.zem': bytes})};
  const docs = [{id: 99101n, revision: 1n, text: 'Fake tulip ZQ986'}, {id: 99102n, revision: 1n, text: 'Balise fictive ZQ986'}];
  let store;
  try {
    store = await openNamespaceAsync(join(root, 'store'), 'notes', spec, {...options, modelCacheDir, offline: true});
    await store.upsertAsync(docs);
    const hits = store.queryEmbedded({text: 'Fake tulip', leg: 'dense', k: 2});
    assert.deepEqual(hits.map(hit => hit.id).sort(), docs.map(doc => doc.id));
    const vector = store.get([docs[0].id], {vector: true}).documents[0].vector;
    assert.ok(vector instanceof Float32Array && vector.length === 4);
    store.close(); store = undefined;
    store = openNamespace(join(root, 'store'), 'notes', spec, {...options, modelCacheDir, offline: true});
    assert.deepEqual(store.queryEmbedded({text: 'Fake tulip', leg: 'dense', k: 2}).map(hit => hit.id), hits.map(hit => hit.id));
    assert.deepEqual(store.get([docs[0].id], {vector: true}).documents[0].vector, vector);
  } finally { store?.close(); rmSync(root, {recursive: true, force: true}); }
});

// WF-991 opt-in smoke: the default model from a pre-filled local cache
// (WF991_MODEL_CACHE), offline. Under `arch -x86_64` this is the Intel path.
test('wf991 default model from a local cache', {skip: process.platform !== 'darwin' || !process.env.WF991_MODEL_CACHE}, async () => {
  const {TextStore} = createRequire(import.meta.url)('..');
  const root = mkdtempSync(join(tmpdir(), 'wf991-default-'));
  const spec = {embeddedModel: {...TextStore.defaultModel}};
  let store;
  try {
    store = await openNamespaceAsync(join(root, 'store'), 'notes', spec,
      {...options, modelCacheDir: process.env.WF991_MODEL_CACHE, offline: true});
    await store.upsertAsync([
      {id: 1n, revision: 1n, text: 'Paris is the capital and largest city of France.'},
      {id: 2n, revision: 1n, text: 'Bananas are yellow.'},
    ]);
    const hits = store.queryEmbedded({text: 'What is the capital of France?', leg: 'dense', k: 2});
    console.log(`wf991 default model ${process.platform}-${process.arch}: ${JSON.stringify(hits, (key, value) => typeof value === 'bigint' ? `${value}` : value)}`);
    assert.equal(hits[0].id, 1n);
  } finally { store?.close(); rmSync(root, {recursive: true, force: true}); }
});
