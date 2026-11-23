# Release model files (ZE-498, WF-987)

Use Python 3.11.15 with `requirements-release.txt` (exact versions from the
provided model venv freeze, plus its installed `xxhash==4.0.1`). The build needs
macOS and the offline source pack. A Core ML query model also needs Xcode's
`xcrun coremlcompiler`. `release` downloads nothing.

```sh
PYTHON=/path/to/release-venv/bin/python
PACK=/path/to/zeppelin-embed-v98-8-r1-e5-small
STEM=volantis-db-v98-8-r1-e5-small
"$PYTHON" tools/ze-model/ze-model release --pack "$PACK" \
  --out "/path/to/model-release/$STEM" --name "$STEM" \
  --document-precision fp16 --query-mlx-precision fp16 \
  --query-coreml none --card-license mit
```

`release` bakes the pair with `bake --pair "$PACK/PAIR.json"`. `PAIR.json` is the
only source of tower names, prefixes and token limits (for the e5 pair: query
`query: `/96, document `passage: `/512). Alpha is 0.5. The pair alignment
identity is SHA-256 of the exact `PAIR.json` bytes.

`--query-coreml none|fp16|int8` is required. `none` ships MLX only: no Core ML
export, `"query_coreml": null` and only the `.zem` in the manifest. `fp16` and
`int8` export the query tower at its `PAIR.json` token limit. The exporter needs
a `2_Dense` head, so it does not accept the e5 pair query tower. The removed flag
`--query-coreml-precision` fails with a message that names `--query-coreml`.
`--card-license` is the Hugging Face licence identifier in the model card. It is
an owner decision.

Each output directory must be absent or empty. It contains `<name>.zem`,
`<name>.mlmodelc/` (Core ML only), `volantis-model.json`, `README.md` (the
model card), `licenses/` (a copy of the pack `licenses/`) and `SHA256SUMS`.
The manifest lists the bundle and every compiled model file, sorted by relative
path. The card and licences are not in the manifest; the loader does not need
them. SHA256SUMS covers every shipped file except itself. Both lists hash the
actual output bytes; they never reuse pack checksums. The command prints the
bundle bytes, the Core ML bytes and the manifest SHA-256.

The model card is made from `PAIR.json` (towers, parameters, prefixes, limits,
pooling, dimensions, target languages, grouped retrieval, release status),
`PROVENANCE.md`, `licenses/`, and the per-task nDCG@10 in
`benchmarks/retrieval-<query name>.json` and `benchmarks/retrieval-e5-ceiling.json`.
It lists every task that loses more than 0.05 nDCG@10 against e5 as the query
tower. A missing source is an error before anything is baked.

The earlier default (`volantis-labs/volantis-db-clean71-arctic-m-v2` at revision
`5833eb696dbc41def2421a7fb97d58f92db94fab`) was made by the ZE-498 form of this
command, with fixed 96/8192 limits and a Core ML query model.

`bake` and `release` accept `--query-mlx-precision fp32|fp16`, defaulting to
`--document-precision`. `coreml-export --precision fp16|int8` defaults to fp16.
INT8 applies symmetric, per-channel weight-only linear quantization; activations
and output remain FP16. The release query uses CPU_AND_NE for both choices.

The bundle retains MLX query weights at the selected precision. CoreML is
preferred, but `ZE_QUERY_COREML=off` and an absent companion select the separate
query MLX tower. Retaining its tensors preserves this supported path and bundle digest
validation. A discovered CoreML model that fails to load is a hard error.

For identical pack bytes and precision, `.zem` is deterministic. CoreML compiled
bytes may vary through metadata map serialization order, conversion dates, and
analytics data. Absolute build paths are excluded from model metadata. The
manifest and SHA256SUMS are always regenerated from the actual output bytes;
they change when compiled bytes change.

## Offline bake (WF-986)

`bake` has two offline forms. Neither form downloads anything.

1. Pair manifest. `--pair` reads `PAIR.json` and resolves both tower folders
   relative to it. Each tower's name, prompt prefix and token limit come from its
   own record. The alignment digest is SHA-256 of the exact `PAIR.json` bytes.

   ```sh
   PACK=/path/to/zeppelin-embed-v98-8-r1-e5-small
   "$PYTHON" tools/ze-model/ze-model bake --pair "$PACK/PAIR.json" \
     --document-precision fp32 --query-mlx-precision fp32 --output pair-fp32.zem
   ```

   Do not use `--document`, `--document-id`, `--query`, `--query-id`,
   `--query-prefix` or `--alignment-digest` with `--pair`. A limit flag is
   accepted only when it is equal to the `PAIR.json` value.

2. Explicit towers. This is the earlier form; it is unchanged.

   ```sh
   "$PYTHON" tools/ze-model/ze-model bake --document "$PACK/document" \
     --document-id arctic-m-v2 --query "$PACK/query" --query-id clean71-v1 \
     --query-prefix 'query: ' --alignment-digest "$ALIGNMENT_DIGEST" \
     --query-max-tokens 96 --document-max-tokens 8192 --output release.zem
   ```

Before it writes, `--pair` checks the manifest: `normalization` is `L2`,
`similarity` is `dot_product`, and each tower has `pooling` "mean then L2".
It checks the declared SHA-256 values, and each tower is BERT with a hidden width
equal to `dimension`. Each tower must have mean pooling that includes the prompt,
a final Normalize module and no Dense head. The limit must not be more than the
position embeddings. The checkpoint layers must agree with `num_hidden_layers`.
The tokenizer pieces must fit the embedding rows. A manifest that names a tower
of an approved pair (`RELEASE_PAIRS`) must match that pair exactly: names,
revision, prefixes, limits, layers, pieces, rows and file hashes.

A BERT tower can use an XLM-R Unigram tokenizer only in the exact source form:
normalizer Sequence[Precompiled, Replace(" {2,}", " ")], Metaspace("▁") that
prepends and splits, and a single `<s>`/sequence/`</s>` template. The five
added tokens must be non-normalized and must not strip. The bundle records this
with tokenizer flag bit 2. Readers before WF-986 refuse that bit.

The pair test is `python3 -m unittest discover -s tools/ze-model -p test_e5_pair.py`.
It bakes the invented pack from
`crates/volantis-db-text/tests/fixtures/wf986/generate.py` and compares the
result with the committed `pair.zem`. `e5-pair-tokenizer-goldens.py` writes the
real-pack token oracle `crates/volantis-db-text/tests/goldens/e5_pair.rs`.
