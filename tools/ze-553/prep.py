"""ZE-553: build the SciFact sample. Usage: prep.py <scifact dir>.

Writes docs.txt / queries.txt (one text per line), doc_ids.txt,
query_ids.txt and qrels.tsv restricted to the sample.
1000 passages = every test-judged passage plus seeded random fill.
"""
import random, sys
from pathlib import Path
import pyarrow.parquet as pq

d = Path(sys.argv[1])
corpus = pq.read_table(d / "corpus.parquet").to_pylist()
queries = {str(r["_id"]): r["text"] for r in pq.read_table(d / "queries.parquet").to_pylist()}
qrels = [l.split("\t") for l in (d / "test.tsv").read_text().splitlines()[1:]]
judged = sorted({c for _, c, _ in qrels})
others = sorted(str(r["_id"]) for r in corpus if str(r["_id"]) not in set(judged))
random.Random(20261122).shuffle(others)
sample = set(judged) | set(others[: 1000 - len(judged)])
flat = lambda s: " ".join(s.split())
docs = [r for r in corpus if str(r["_id"]) in sample]
(d / "docs.txt").write_text("".join(flat(f"{r['title']} {r['text']}") + "\n" for r in docs))
(d / "doc_ids.txt").write_text("".join(str(r["_id"]) + "\n" for r in docs))
qids = sorted({q for q, _, _ in qrels}, key=int)
(d / "queries.txt").write_text("".join(flat(queries[q]) + "\n" for q in qids))
(d / "query_ids.txt").write_text("".join(q + "\n" for q in qids))
(d / "qrels.tsv").write_text("".join(f"{q}\t{c}\t{s}\n" for q, c, s in qrels))
print(len(docs), "docs", len(judged), "judged", len(qids), "queries")
