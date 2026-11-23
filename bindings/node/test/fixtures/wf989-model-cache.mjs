// WF-989: app-owned model cache and offline policy for embedded namespaces.
// Each mode runs in a fresh child: MLX policy and model environment are process-wide.
import assert from 'node:assert/strict';
import {fork, spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, truncateSync, writeFileSync} from 'node:fs';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const self = fileURLToPath(import.meta.url);
const inherited = ['ZE_MODEL_CACHE', 'ZE_MODEL_OFFLINE', 'ZE_MODEL_ENDPOINT', 'ZE_MODEL_ALLOW_HTTP',
  'HF_TOKEN', 'ZE_QUERY_COREML', 'ZE_TEXT_BUNDLE', 'ZE_MLX_CACHE_LIMIT_BYTES'];

/** Runs one mode in a child with every inherited model override removed. */
export function runCase(mode, extra = {}) {
  const env = {...process.env};
  for (const key of inherited) delete env[key];
  const result = spawnSync(process.execPath, [self, mode], {env: {...env, ...extra}, encoding: 'utf8'});
  assert.equal(result.status, 0, `${mode}: ${result.stderr}`);
}

const REVISION = '1'.repeat(40);
const IDS = [98901n, 98902n];
const TEXTS = ['fixture copper kite', 'fixture violet spool'];
const NAMES = ['fake_notes', 'fake_segments'];
const notFound = {code: 'ZE_ERR_NOT_FOUND'};
const corrupt = {code: 'ZE_ERR_CORRUPT'};
const conflict = {code: 'ZE_ERR_MLX_CACHE_LIMIT_CONFLICT'};

// Loopback endpoint in its own process: it counts accepted connections even
// while the caller's JS thread is blocked in a synchronous open.
async function endpointProcess() {
  let count = 0;
  const server = createServer((_request, response) => { response.writeHead(404); response.end(); });
  server.on('connection', () => { count += 1; });
  process.on('message', message => {
    if (message === 'count') process.send({count});
    else server.close(() => process.exit(0));
  });
  server.listen(0, '127.0.0.1', () => process.send({port: server.address().port}));
}

async function startEndpoint() {
  const child = fork(self, ['endpoint'], {stdio: ['ignore', 'inherit', 'inherit', 'ipc']});
  const reply = () => new Promise((resolve, reject) => {
    const exit = code => reject(new Error(`endpoint exited with ${code}`));
    child.once('exit', exit);
    child.once('message', message => { child.off('exit', exit); resolve(message); });
  });
  const {port} = await reply();
  return {
    url: `http://127.0.0.1:${port}`,
    async count() { const pending = reply(); child.send('count'); return (await pending).count; },
    close() { child.kill(); },
  };
}

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'wf989-'));
  const bytes = readFileSync(new URL('../../../../crates/volantis-db-text/tests/fixtures/ze115/gte.zem', import.meta.url));
  const sha = data => createHash('sha256').update(data).digest('hex');
  const manifest = Buffer.from(JSON.stringify({format: 1, name: 'tiny', bundle: 'tiny.zem', query_coreml: null,
    files: [{path: 'tiny.zem', size: bytes.length, sha256: sha(bytes)}]}));
  const cache = name => join(root, name);
  for (const name of ['fake-cache-a', 'fake-cache-b', 'fake-cache-damaged']) {
    const entry = join(cache(name), 'fixture%2Ffake-wf989', REVISION);
    mkdirSync(entry, {recursive: true});
    writeFileSync(join(entry, 'tiny.zem'), bytes);
    writeFileSync(join(entry, 'volantis-model.json'), manifest);
    if (name === 'fake-cache-damaged') truncateSync(join(entry, 'tiny.zem'), bytes.length - 1);
  }
  for (const name of ['fake-cache-missing', 'fake-cache-online', 'fake-cache-decoy']) mkdirSync(cache(name));
  return {
    root,
    store: join(root, 'store'),
    a: cache('fake-cache-a'),
    b: cache('fake-cache-b'),
    damaged: cache('fake-cache-damaged'),
    missing: cache('fake-cache-missing'),
    online: cache('fake-cache-online'),
    decoy: cache('fake-cache-decoy'),
    spec: {embeddedModel: {repo: 'fixture/fake-wf989', revision: REVISION, manifestSha256: sha(manifest)}},
  };
}

const policy = (modelCacheDir, offline = true) => ({modelCacheDir, offline});
const denseId = (store, index) => store.queryEmbedded({text: TEXTS[index], leg: 'dense', k: 1})[0]?.id;

async function openMode(api, f, writers, endpoint) {
  const {openNamespace, openNamespaceAsync} = api;
  const {store, spec} = f;
  process.env.ZE_MODEL_CACHE = f.decoy;
  process.env.ZE_MODEL_ENDPOINT = endpoint.url;
  process.env.ZE_MODEL_ALLOW_HTTP = '1';
  const notes = openNamespace(store, NAMES[0], spec, policy(f.a));
  writers.push(notes);
  notes.upsert([{id: IDS[0], revision: 1n, text: TEXTS[0]}]);
  assert.equal(denseId(notes, 0), IDS[0]);
  const segments = await openNamespaceAsync(store, NAMES[1], spec, policy(f.a));
  segments.upsert([{id: IDS[1], revision: 1n, text: TEXTS[1]}]);
  assert.equal(denseId(segments, 1), IDS[1]);
  segments.close();

  // The writer keeps the pinned session live; the peer root is still verified first.
  assert.throws(() => openNamespace(store, NAMES[1], spec, policy(f.missing)), notFound);
  await assert.rejects(openNamespaceAsync(store, NAMES[1], spec, policy(f.missing)), notFound);
  assert.throws(() => openNamespace(store, NAMES[1], spec, policy(f.damaged)), corrupt);
  await assert.rejects(openNamespaceAsync(store, NAMES[1], spec, policy(f.damaged)), corrupt);
  const peer = openNamespace(store, NAMES[1], spec, policy(f.b));
  assert.equal(denseId(peer, 1), IDS[1]);
  peer.close();
  (await openNamespaceAsync(store, NAMES[1], spec, policy(f.b))).close();
  assert.equal(await endpoint.count(), 0);

  // Omitted or empty directory: the environment root still applies.
  process.env.ZE_MODEL_CACHE = f.a;
  openNamespace(store, NAMES[1], spec).close();
  (await openNamespaceAsync(store, NAMES[1], spec, {offline: true})).close();
  openNamespace(store, NAMES[1], spec, policy('')).close();
  process.env.ZE_MODEL_CACHE = f.decoy;

  // Explicit offline false does not cancel ZE_MODEL_OFFLINE=1.
  process.env.ZE_MODEL_OFFLINE = '1';
  openNamespace(store, NAMES[1], spec, policy(f.b, false)).close();
  assert.throws(() => openNamespace(store, NAMES[1], spec, policy(f.missing, false)), notFound);
  delete process.env.ZE_MODEL_OFFLINE;
  assert.equal(await endpoint.count(), 0);

  // Control: an online miss does reach this endpoint, so zero above is meaningful.
  assert.throws(() => openNamespace(store, NAMES[1], spec, policy(f.online, false)), {code: 'ZE_ERR_IO'});
  assert.ok(await endpoint.count() > 0);

  for (const [options, message] of [
    [{modelCacheDir: 1}, /modelCacheDir must be a string/],
    [{modelCacheDir: undefined}, /modelCacheDir must be a string/],
    [{offline: 'yes'}, /offline must be a boolean/],
    [{offline: undefined}, /offline must be a boolean/],
  ]) {
    assert.throws(() => openNamespace(store, NAMES[1], spec, options), message);
    await assert.rejects(openNamespaceAsync(store, NAMES[1], spec, options), message);
  }
  for (const plain of [{}, {vectorSpace: {dimensions: 2}}]) {
    for (const options of [{modelCacheDir: f.a}, {offline: false}]) {
      assert.throws(() => openNamespace(store, 'plain', plain, options), /requires embeddedModel/);
      await assert.rejects(openNamespaceAsync(store, 'plain', plain, options), /requires embeddedModel/);
    }
  }
}

// A zero limit is explicit; an omitted limit accepts the configured policy.
async function mlxMode(api, f, writers, mode) {
  const {openNamespace, openNamespaceAsync} = api;
  const {store, spec} = f;
  process.env.ZE_MODEL_CACHE = f.decoy;
  process.env.ZE_MODEL_ENDPOINT = 'http://127.0.0.1:1';
  const limited = limit => ({...policy(f.a), mlxCacheLimitBytes: limit});
  if (mode === 'mlx-zero') {
    writers.push(openNamespace(store, NAMES[0], spec, limited(0n)));
    assert.throws(() => openNamespace(store, NAMES[1], spec, limited(8192n)), conflict);
    await assert.rejects(openNamespaceAsync(store, NAMES[1], spec, limited(8192n)), conflict);
    openNamespace(store, NAMES[1], spec, policy(f.b)).close();
  } else {
    assert.equal(process.env.ZE_MLX_CACHE_LIMIT_BYTES, '8192');
    writers.push(openNamespace(store, NAMES[0], spec, policy(f.a)));
    assert.throws(() => openNamespace(store, NAMES[1], spec, limited(0n)), conflict);
    (await openNamespaceAsync(store, NAMES[1], spec, limited(8192n))).close();
  }
}

async function batchMode(api, f, writers, live) {
  const {openNamespace, namespaceBatch, namespaceBatchLive} = api;
  const {store, spec} = f;
  // Seed through the legacy environment root, then make it unusable.
  process.env.ZE_MODEL_CACHE = f.a;
  for (const name of NAMES) writers.push(openNamespace(store, name, spec));
  const start = writers.map(writer => writer.get(IDS).generation);
  if (!live) while (writers.length) writers.pop().close();
  process.env.ZE_MODEL_CACHE = f.decoy;
  process.env.ZE_MODEL_ENDPOINT = 'http://127.0.0.1:1';
  process.env.ZE_MODEL_ALLOW_HTTP = '1';

  const participant = (index, modelCacheDir) => ({
    name: NAMES[index], spec, modelCacheDir, offline: true,
    upserts: [{id: IDS[index], revision: 1n, text: TEXTS[index]}],
    ...(live ? {store: writers[index]} : {}),
  });
  const run = participants => (live ? namespaceBatchLive : namespaceBatch)(store, participants);
  const check = (generations, present) => {
    for (const [index, name] of NAMES.entries()) {
      const writer = live ? writers[index] : openNamespace(store, name, spec, policy(f.b));
      try {
        const rows = writer.get(IDS);
        assert.equal(rows.generation, generations[index]);
        assert.deepEqual(rows.documents.filter(Boolean).map(doc => doc.id), present ? [IDS[index]] : []);
        if (present) assert.equal(denseId(writer, index), IDS[index]);
      } finally { if (!live) writer.close(); }
    }
  };
  assert.throws(() => run([participant(0, f.b), participant(1, f.missing)]), notFound);
  check(start, false);
  assert.throws(() => run([participant(0, f.b), participant(1, 1)]), /modelCacheDir must be a string/);
  assert.throws(() => run([participant(0, f.b), {...participant(1, f.b), offline: 1}]), /offline must be a boolean/);
  const generations = run([participant(0, f.b), participant(1, f.b)]);
  assert.deepEqual(generations, start.map(generation => generation + 1n));
  check(generations, true);
}

async function main(mode) {
  if (mode === 'endpoint') return endpointProcess();
  const api = createRequire(import.meta.url)('../..');
  const f = fixture();
  const writers = [];
  let endpoint;
  try {
    if (mode === 'open') {
      endpoint = await startEndpoint();
      await openMode(api, f, writers, endpoint);
    } else if (mode === 'mlx-zero' || mode === 'mlx-implicit') {
      await mlxMode(api, f, writers, mode);
    } else if (mode === 'batch' || mode === 'batch-live') {
      await batchMode(api, f, writers, mode === 'batch-live');
    } else {
      throw new Error(`unknown mode ${mode}`);
    }
  } finally {
    for (const writer of writers) writer.close();
    endpoint?.close();
    rmSync(f.root, {recursive: true, force: true});
  }
}

if (process.argv[1] === self) await main(process.argv[2]);
