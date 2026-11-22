"""ZE-553: CPU embedding speed and vectors on the exact shipped token ids.

Usage: bench_cpu.py <onnx dir> <out dir> <variant: fp32|int8> <threads> [doc rows]
With [doc rows], only the first rows are timed and no vectors are saved.
Reads <out>/docs.ids and <out>/queries.ids (written by ze553_dump).
Docs: batches of 32 in input order, padded to the batch max (the
shipped ingest batch). Queries: one row, unpadded, warm-up then timed.
Writes <out>/{docs,queries}.ort-<variant>.npy (L2-normalized).
"""
import struct, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def read_ids(path):
    data, rows, at = path.read_bytes(), [], 0
    while at < len(data):
        (n,) = struct.unpack_from("<I", data, at)
        rows.append(np.frombuffer(data, "<i4", n, at + 4).astype(np.int64))
        at += 4 + 4 * n
    return rows


def pack(rows):
    width = max(len(r) for r in rows)
    ids = np.zeros((len(rows), width), np.int64)
    mask = np.zeros((len(rows), width), np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)], mask[i, : len(r)] = r, 1
    return {"input_ids": ids, "attention_mask": mask}


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def norm(x):
    return (x / np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)).astype(np.float32)


onnx, out, variant, threads = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
suffix = "" if variant == "fp32" else ".int8"
print(f"ort={ort.__version__} doc_rows={limit or 'all'} variant={variant} threads={threads}")

q = session(onnx / f"query{suffix}.onnx", threads)
qrows = read_ids(out / "queries.ids")
vecs, lat = [], []
for r in qrows:
    feed = pack([r])
    q.run(None, feed)
    t = time.perf_counter()
    vecs.append(q.run(None, feed)[0])
    lat.append((time.perf_counter() - t) * 1000)
if limit is None:
    np.save(out / f"queries.ort-{variant}.npy", norm(np.concatenate(vecs)))
lat = np.sort(lat)
print(f"QUERY rows={len(qrows)} p50_ms={np.percentile(lat,50):.3f} "
      f"p95_ms={np.percentile(lat,95):.3f} max_ms={lat[-1]:.3f} "
      f"mean_tokens={np.mean([len(r) for r in qrows]):.1f}")

d = session(onnx / f"doc{suffix}.onnx", threads)
drows = read_ids(out / "docs.ids")[:limit]
d.run(None, pack(drows[:32]))
vecs, t = [], time.perf_counter()
for i in range(0, len(drows), 32):
    vecs.append(d.run(None, pack(drows[i : i + 32]))[0])
secs = time.perf_counter() - t
tokens = sum(len(r) for r in drows)
if limit is None:
    np.save(out / f"docs.ort-{variant}.npy", norm(np.concatenate(vecs)))
print(f"DOC rows={len(drows)} tokens={tokens} secs={secs:.2f} "
      f"passages_per_s={len(drows)/secs:.2f} tokens_per_s={tokens/secs:.0f} "
      f"padded_tokens={sum(len(drows[i:i+32]) * max(map(len, drows[i:i+32])) for i in range(0, len(drows), 32))}")
