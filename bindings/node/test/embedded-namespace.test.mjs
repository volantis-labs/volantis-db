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
    const bytes = readFileSync(new URL('../../../crates/zeppelin-embed-text/tests/fixtures/ze115/gte.zem',import.meta.url));
    const sha = b => createHash('sha256').update(b).digest('hex');
    const manifest = Buffer.from(JSON.stringify({format:1,name:'tiny',bundle:'tiny.zem',query_coreml:null,
      files:[{path:'tiny.zem',size:bytes.length,sha256:sha(bytes)}]}));
    const revision = '0123456789abcdef0123456789abcdef01234567';
    process.env.ZE_MODEL_CACHE = join(root,'cache');
    const entry = join(process.env.ZE_MODEL_CACHE,'fixture%2Ftiny',revision);
    mkdirSync(entry,{recursive:true});
    writeFileSync(join(entry,'zeppelin-model.json'),manifest);
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
