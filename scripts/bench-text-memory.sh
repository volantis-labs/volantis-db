#!/bin/sh
# ZE-554. Run from the repository root on an 8 GB Apple silicon Mac after
# TextStore.openModel has cached the pinned release model (MLX only, WF-987).
# Each process ingests the same one-chunk documents into a new store, then
# measures idle memory with the store open and memory after one hybrid query.
set -eu
run=$(mktemp -d "${TMPDIR:-/tmp}/ze554.XXXXXX")
echo "Results in $run"
sysctl hw.model hw.memsize machdep.cpu.brand_string > "$run/hardware.txt"
sw_vers >> "$run/hardware.txt"
git rev-parse HEAD >> "$run/hardware.txt"
test "$(sysctl -n hw.memsize)" = 8589934592
model_dir="${ZE_MODEL_CACHE:-$HOME/Library/Caches/volantis-db/models}/volantis-labs%2Fvolantis-db-v98-e5-small/5909e5a5985f1c727766f7b50a713cfbe043256c"
bundle="$model_dir/volantis-db-v98-8-r1-e5-small.zem"
test -f "$bundle"
test ! -e "${bundle%.zem}.mlmodelc"
unset ZE_QUERY_COREML
mkdir -p "$run/chunks/qrels"
python3 - "$run/chunks" <<'PY'
import json, pathlib, random, sys
root = pathlib.Path(sys.argv[1])
rng = random.Random(554)
words = "the team will meet next week to review our plan for new work and share the results with each group".split()
with (root / "corpus.jsonl").open("w") as f:
    for i in range(5000):
        text = " ".join(rng.choices(words, k=198))
        print(json.dumps({"_id": str(i), "text": text}), file=f)
(root / "queries.jsonl").write_text('{"_id":"q","text":"team plan"}\n')
(root / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\t0\t1\n")
PY
cargo build --locked --release -p volantis-db-bench --features text --bin text-user-bench
build="${CARGO_TARGET_DIR:-target}/release"
python3 scripts/release/stage-metallib.py "$build" "$build/mlx.metallib"
for mib in 256 512 1024; do
    "$build/text-user-bench" ingest --beir-root "$run" --corpus chunks \
        --bundle "$bundle" --store "$run/store-$mib" \
        --mlx-cache-limit-bytes "$((mib * 1048576))" --embed-batch-size 32 \
        --memory-sample-ms 100 --rest-ms 10000 --memory-query "team plan" \
        --memory-out "$run/$mib.samples.jsonl" > "$run/$mib.json" 2> "$run/$mib.log"
done
python3 - "$run" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print("limit_MiB tokens/chunk peak_footprint_B peak_RSS_B chunks/s idle_RSS_B post_query_RSS_B post_query_footprint_B")
for mib in (256, 512, 1024):
    r = json.loads((root / f"{mib}.json").read_text())
    m = r["memory"]
    assert r["documents"] == r["chunks"] == 5000 and r["all_threads_joined"]
    assert m["mlx_cache_limit_bytes"] == mib * 1048576
    tokens = r["tokens"] / r["chunks"]
    assert 180 <= tokens <= 220, tokens
    print(mib, tokens, m["after_ingest"]["lifetime_max_phys_footprint"],
          m["ingest_lifetime_max_resident"], m["documents_per_second"],
          m["at_rest"]["resident"], m["after_query"]["resident"],
          m["after_query"]["phys_footprint"])
PY
