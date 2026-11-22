"""ZE-553: CPU vs shipped vector agreement and top-10 retrieval effect.

Usage: agree.py <mlx out dir> <cpu out dir> <scifact dir>
Shipped reference: MLX FP16 document vectors + CoreML INT8 queries.
"""
import sys
from pathlib import Path
import numpy as np

mlx, cpu, data = (Path(a) for a in sys.argv[1:4])
f32 = lambda p: np.fromfile(p, "<f4").reshape(-1, 768)
D = {"mlx": f32(mlx / "docs.mlx.f32")}
Q = {"coreml": f32(mlx / "queries.coreml.f32"), "mlxq": f32(mlx / "queries.mlx.f32")}
for v in ("fp32", "int8"):
    if (cpu / f"queries.ort-{v}.npy").exists():
        Q[v] = np.load(cpu / f"queries.ort-{v}.npy")
    if (cpu / f"docs.ort-{v}.npy").exists():
        D[v] = np.load(cpu / f"docs.ort-{v}.npy")
doc_ids = (data / "doc_ids.txt").read_text().split()
qids = (data / "query_ids.txt").read_text().split()
rel = {}
for line in (data / "qrels.tsv").read_text().splitlines():
    q, c, s = line.split("\t")
    if int(s) > 0:
        rel.setdefault(q, set()).add(c)


def cos(a, b):
    c = (a * b).sum(1)
    return f"mean={c.mean():.6f} min={c.min():.6f} p01={np.percentile(c, 1):.6f}"


def top(q, d, k=10):
    return np.argsort(-(q @ d.T), axis=1, kind="stable")[:, :k]


def ndcg(t):
    out = []
    for qi, row in zip(qids, t):
        gains = [1.0 if doc_ids[j] in rel.get(qi, ()) else 0.0 for j in row]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(rel.get(qi, ())))))
        out.append(dcg / ideal if ideal else 0.0)
    return np.mean(out)


if "fp32" in D:
    print("COSINE docs  fp32 vs mlx  ", cos(D["fp32"], D["mlx"]))
print("COSINE query fp32 vs coreml", cos(Q["fp32"], Q["coreml"]))
print("COSINE query fp32 vs mlxq  ", cos(Q["fp32"], Q["mlxq"]))
print("COSINE query coreml vs mlxq", cos(Q["coreml"], Q["mlxq"]))
if "int8" in D:
    print("COSINE docs  int8 vs fp32 ", cos(D["int8"], D["fp32"]))
    print("COSINE docs  int8 vs mlx  ", cos(D["int8"], D["mlx"]))
    print("COSINE query int8 vs fp32 ", cos(Q["int8"], Q["fp32"]))
    print("COSINE query int8 vs coreml", cos(Q["int8"], Q["coreml"]))

ref = top(Q["coreml"], D["mlx"])
mixed_half = lambda cpu_docs: np.where((np.arange(len(D["mlx"])) % 2)[:, None] == 1, cpu_docs, D["mlx"])
cells = [("shipped: coreml q / mlx d", Q["coreml"], D["mlx"])]
for v in [v for v in ("fp32", "int8") if v in Q and v not in D]:
    cells += [(f"cpu {v} q / mlx d (Intel client, arm64-built store)", Q[v], D["mlx"])]
cells += [("mlx fp16 q / mlx d (shipped fallback)", Q["mlxq"], D["mlx"])]
for v in [v for v in ("fp32", "int8") if v in D]:
    cells += [
        (f"all-cpu {v}: cpu q / cpu d", Q[v], D[v]),
        (f"cpu {v} q / mlx d (Intel client, arm64-built store)", Q[v], D["mlx"]),
        (f"coreml q / cpu {v} d (arm64 client, Intel-built store)", Q["coreml"], D[v]),
        (f"coreml q / half mlx + half cpu {v} d (mixed store)", Q["coreml"], mixed_half(D[v])),
        (f"cpu {v} q / half mlx + half cpu {v} d (mixed store)", Q[v], mixed_half(D[v])),
    ]
for name, q, d in cells:
    t = top(q, d)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(t, ref)])
    same1 = np.mean(t[:, 0] == ref[:, 0])
    print(f"TOP10 {name}: overlap@10={overlap:.4f} top1_same={same1:.4f} ndcg@10={ndcg(t):.4f}")
