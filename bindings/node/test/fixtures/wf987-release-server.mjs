// WF-987: loopback stand-in for Hugging Face that serves a local release folder.
// Opt-in: WF987_RELEASE names a folder made by `ze-model release`. Nothing leaves 127.0.0.1.
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync, statSync} from 'node:fs';
import {createServer} from 'node:http';
import {join} from 'node:path';

export const release = process.env.WF987_RELEASE;
// Proposed repository and a stand-in commit; the real commit exists only after upload.
export const REPO = 'volantis-labs/volantis-db-v98-e5-small';
export const REVISION = '0987'.repeat(10);

/** Serves only the manifest and the files it lists, at /{repo}/resolve/{revision}/{path}. */
export async function serveRelease(folder = release) {
  const manifestBytes = readFileSync(join(folder, 'volantis-model.json'));
  const manifest = JSON.parse(manifestBytes);
  const served = new Map([['volantis-model.json', manifestBytes.length],
    ...manifest.files.map(file => [file.path, statSync(join(folder, file.path)).size])]);
  const prefix = `/${REPO}/resolve/${REVISION}/`;
  const requests = [];
  const server = createServer((request, response) => {
    requests.push(request.url);
    const path = request.url.startsWith(prefix) ? decodeURIComponent(request.url.slice(prefix.length)) : '';
    if (!served.has(path)) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, {'content-length': served.get(path)});
    createReadStream(join(folder, path)).pipe(response);
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    requests,
    manifest,
    pin: {repo: REPO, revision: REVISION, manifestSha256: createHash('sha256').update(manifestBytes).digest('hex')},
    close: () => new Promise(resolve => server.close(resolve)),
  };
}

/** Sets the loopback endpoint for this process and returns a function that restores the environment. */
export function useEndpoint(url, extra = {}) {
  const values = {ZE_MODEL_ENDPOINT: url, ZE_MODEL_ALLOW_HTTP: '1', ZE_MODEL_OFFLINE: undefined,
    ZE_MODEL_CACHE: undefined, ZE_QUERY_COREML: undefined, ...extra};
  const previous = Object.fromEntries(Object.keys(values).map(key => [key, process.env[key]]));
  const apply = settings => {
    for (const [key, value] of Object.entries(settings)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  };
  apply(values);
  return () => apply(previous);
}

// Two documents and one query that the e5 pair must rank in this order.
export const DOCUMENTS = [{id: 98701n, revision: 1n, text: 'Paris is the capital of France.'},
  {id: 98702n, revision: 1n, text: 'Bananas are yellow.'}];
export const QUERY = 'What is the capital of France?';
