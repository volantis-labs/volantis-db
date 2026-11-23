import {
  ABI_VERSION,
  type GraphMutation,
  type GraphResult,
  type GraphValue,
  CancellationToken,
  Store,
  idToUuid,
  openNamespace,
  openNamespaceAsync,
  uuidToId,
  verify,
  type CountGroup,
  type CountResult,
  type Document,
  type MaintainReport,
  type QueryResult,
  type SearchHit,
  type StoreState,
  type StoreStats,
  type VerifyFindingKind,
  type VerifyReport,
} from 'volantis-db';

const documents: readonly Document[] = [
  { id: 1n, revision: 1n, vector: new Float32Array([1, 0]), text: 'harbour', metadata: Uint8Array.of(42) },
];
const store = new Store('typecheck-index');
const state: StoreState = store.state();
const stats: StoreStats = store.stats();
const activeRows: bigint = stats.activeRowCount;
const physicalFootprint: bigint | null = stats.physicalFootprint;
const generation: bigint = store.ingest(documents, 2).generation;
const attributedDocument = { id: 2n, vector: new Float32Array([1, 0]), attributes: [] };
// @ts-expect-error Attributes belong on upsert.
store.ingest([attributedDocument], 2);
const hits: SearchHit[] = store.search(new Float32Array([1, 0]), 1);

const records = openNamespace(
  'typecheck-records',
  'notes',
  { vectorSpace: { dimensions: 2 } },
  { autoSealRows: 2048, autoMerge: true },
);
const sealed: bigint = records.seal().generation;
const merged: bigint = records.merge().generation;
const maintained: MaintainReport = records.maintain({ wallTimeNs: 0n, bytes: 0n });
const maintaining: Promise<MaintainReport> = records.maintainAsync({ wallTimeNs: 0n, bytes: 0n });
const grouped = records.count({ groupBy: { attributeId: 1, limit: 16 } });
const groups: CountGroup[] = grouped.groups;
const missing: bigint = grouped.missingCount;
const plain: CountResult = records.count({ timestampRange: { start: 0n, end: 1n } });
const token = new CancellationToken();
const result: QueryResult = records.query({
  text: 'harbour',
  vector: new Float32Array([1, 0]),
  k: 5,
  lastAsPrefix: true,
  alpha: 0.5,
  deadlineNs: undefined,
  cancelToken: token,
});
const mode: 'vector' | 'lexical' | 'hybrid' = result.mode;
const lexicalScore: number | undefined = result.hits[0]?.lexicalBm25;
const effectiveAlpha: number | undefined = result.fusion?.effectiveAlpha;
const uuid: string = idToUuid(uuidToId('123e4567-e89b-12d3-a456-426614174000'));

const report: VerifyReport = verify('typecheck-records/notes');
const damage: VerifyFindingKind | undefined = report.findings[0]?.kind;
const offset: bigint | undefined = report.findings[0]?.offset;

console.log(ABI_VERSION, generation, sealed, merged, maintained, maintaining, groups, missing, plain, hits, mode, lexicalScore, effectiveAlpha, uuid);
console.log(report.ok, damage, offset);
token.close();
records.close();
store.close();

const graph = new Store('/unused/typecheck-graph', {maxResidentBytes: 268435456n});
const items: GraphMutation[] = [{ kind: 'node', operation: 'create', namespace: 'notes',
  key: 'first', revision: 1n, labels: ['Note'] }];
const graphReport: GraphResult = graph.graphApply(items);
const graphResult: GraphResult = graph.cypher('MATCH (n) RETURN n');
const values: readonly GraphValue[] = graphResult.rows[0];
void graphReport;
void values;
graph.close();

import { TextStore, TextQueryResult, TextHealthReport, TextIngestReport } from 'volantis-db';
async function textExample(bundle: string) {
  const text = await TextStore.openAsync('/tmp/text-store', bundle);
  const docs = [{id: 1n, revision: 1n, text: 'bronze zeppelin'}];
  const ingested: TextIngestReport = await text.ingestAsync(docs, {maintenanceWallTimeNs: 100000000n, maintenanceBytes: 67108864n});
  text.ingest(docs, {maintenanceWallTimeNs: 0n, maintenanceBytes: 0n});
  // @ts-expect-error Maintenance wall time requires an unsigned bigint.
  text.ingest(docs, {maintenanceWallTimeNs: 100});
  // @ts-expect-error Maintenance bytes require an unsigned bigint.
  await text.ingestAsync(docs, {maintenanceBytes: 1024});
  const hits: TextQueryResult = await text.queryAsync({text: 'bronze', legs: 'hybrid', tier: 'exact', signal: new AbortController().signal});
  const embeddingCalls: bigint = hits.embeddingCalls;
  const queryTokens: bigint = hits.queryTokens;
  const generation: bigint | undefined = hits.diagnostics?.generation;
  const endToEndNs: bigint | undefined = hits.timings?.endToEndNs;
  const runtimeName: string | undefined = hits.backend?.runtimeName;
  const health: TextHealthReport = text.health();
  text.query({text: 'bronze', legs: 'lexical', deadlineNs: 1000000000n});
  await text.maintainAsync({wallTimeNs: 100000000n, bytes: 67108864n});
  await text.deleteAsync([1n]);
  text.close();
  return {ingested, hits, health, embeddingCalls, queryTokens, generation, endToEndNs, runtimeName};
}
void textExample;
void TextStore.openModel({path: '/tmp/text-model', repo: 'owner/model', revision: '0'.repeat(40),
  manifestSha256: '0'.repeat(64), cacheDir: '/tmp/cache', offline: true,
  onProgress: ({done, total}) => { const progress: bigint = done + total; void progress; }});

void TextStore.openModel({path: '/tmp/default-text-model', mlxCacheLimitBytes: 0n});
void TextStore.open('/tmp/text', '/tmp/model.zem', {mlxCacheLimitBytes: 256n << 20n});
void TextStore.openAsync('/tmp/text', '/tmp/model.zem', {mlxCacheLimitBytes: 0n});
// @ts-expect-error Byte limits are unsigned bigints.
void TextStore.openModel({path: '/tmp/text', mlxCacheLimitBytes: 256});
const defaultModelRepo: string = TextStore.defaultModel.repo;
void defaultModelRepo;
// @ts-expect-error Pins must be supplied together.
void TextStore.openModel({path: '/tmp/partial-text-model', repo: 'owner/model'});

const embeddedSpec = {embeddedModel: {repo: 'fixture/tiny', revision: '0'.repeat(40), manifestSha256: '0'.repeat(64)}};
void openNamespace('/tmp/root', 'notes', embeddedSpec, {mlxCacheLimitBytes: 0n});
void openNamespaceAsync('/tmp/root', 'notes', embeddedSpec, {mlxCacheLimitBytes: 536870912n});
// @ts-expect-error The byte count is an unsigned bigint.
void openNamespace('/tmp/root', 'notes', embeddedSpec, {mlxCacheLimitBytes: 0});
// @ts-expect-error The byte count is an unsigned bigint.
void openNamespaceAsync('/tmp/root', 'notes', embeddedSpec, {mlxCacheLimitBytes: 0});
