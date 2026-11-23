# Volantis DB Text

Local text embedding and dense (vector), lexical (BM25), or hybrid retrieval
without supplying your own vectors. Models run on macOS on Apple silicon;
source builds there need Xcode with the Metal toolchain.

| Target | Builds | Model runtime |
|---|---|---|
| `aarch64-apple-darwin` | MLX, Core ML shim, model resolver | MLX (GPU) and Core ML |
| `x86_64-apple-darwin` | model resolver, no MLX or Core ML | none yet: model opens return `ZE_ERR_UNSUPPORTED` |
| `x86_64-pc-windows-msvc` | no MLX, Core ML or resolver | none yet: model opens return `ZE_ERR_UNSUPPORTED` |

On Intel macOS and Windows every text-store open and embedded namespace open
fails with `RuntimeError::UnsupportedBackend` (C: `ZE_ERR_UNSUPPORTED`) before
any download, cache or store change. Ordinary namespaces and
`ze_text_default_model` work. The portable CPU runtime is planned (WF-991).

```sh
xcodebuild -downloadComponent MetalToolchain
cargo add volantis-db-text
```

Open the published default model, ingest, query every leg, and delete:

```rust
use volantis_db_text::{
    IngestOptions, Legs, ModelOpenOptions, ModelRef, QueryOptions, TextDocument, TextStore,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = TextStore::open_model(
        "text-store", &ModelRef::default_model(), ModelOpenOptions::default(),
    )?;
    store.ingest_text(
        &[TextDocument::new(1, 1, "The bronze airship flies above the city.")],
        IngestOptions::default(),
    )?;
    for legs in [Legs::Dense, Legs::Lexical, Legs::Hybrid] {
        let hits = store.query_text("bronze airship", QueryOptions::new(5).with_legs(legs))?;
        println!("{hits:?}");
    }
    store.delete_text(&[1])?; // deletes every chunk of this caller document
    store.close()?;
    Ok(())
}
```

`ModelRef::default_model()` uses `volantis-labs/volantis-db-v98-e5-small`
([model card](https://huggingface.co/volantis-labs/volantis-db-v98-e5-small/blob/5909e5a5985f1c727766f7b50a713cfbe043256c/README.md))
at revision `5909e5a5985f1c727766f7b50a713cfbe043256c`: query tower v98-8-r1
(`query: `, 96 tokens) and document tower `intfloat/multilingual-e5-small`
(`passage: `, 512 tokens), 384 dimensions, FP16 on MLX only, a
347,613,712-byte download. Dutch queries lose quality (Belebele nld nDCG@10
0.764 against 0.929); see the root README for the languages. Stores embedded
with the earlier clean71/Arctic default do not open under this default.
The first open downloads and verifies the pinned model; subsequent opens reuse
the verified cache. Set `ModelOpenOptions.cache_dir` to override `ZE_MODEL_CACHE`
and the default `~/Library/Caches/volantis-db/models`. Set `offline: true`
or `ZE_MODEL_OFFLINE=1` to prohibit networking; a cache miss returns
`TextError::ModelResolution(ModelError::OfflineMiss)`. `ZE_MODEL_ENDPOINT` or
`ModelOpenOptions.endpoint` overrides the download endpoint.

MLX keeps a process-global buffer cache, capped at 4 GiB by default during text
ingest. On an Apple M3 Max with 128 GiB, a full FiQA ingest (57,638 documents)
with the earlier FP16 clean71/Arctic default peaked at about 7.3 GB process footprint
(about 112 GB before the cap), with max RSS about 4.8 GB (ZE-528). Override the
cap with `TextOpenOptions::with_mlx_cache_limit_bytes` or
`ZE_MLX_CACHE_LIMIT_BYTES` (bytes), set before the first text open. Invalid
values are errors. Later opens can omit the limit or repeat it. A conflicting
explicit limit returns `RuntimeError::CacheLimitConflict`. Embedded namespaces
share this ceiling for reusable MLX buffers, including after all handles close.

```rust
let mut options = ModelOpenOptions::default();
options.text = options.text.with_mlx_cache_limit_bytes(536_870_912);
options.cache_dir = Some(app_data.join("models"));
options.offline = true;
let model = NamespaceModel::open_model(&pins, &options)?;
```

`NamespaceModel::open_model` verifies the requested cache entry before it
shares a live session for the same pins. Two verified copies of one model share
the session; a missing or damaged copy fails. The cache folder and the offline
switch are not part of the namespace epoch. C callers select the same policy
with `ZeEmbeddedNamespaceSpecV3` (tag 4).

`ModelOpenOptions.progress` receives synchronous byte counts on the calling
thread. Intermediate reports require both at least 64 KiB and at least 100 ms
since the last report; each verified file also reports completion. Initial
`(0, 0)` means fetching the manifest. Do not reenter model resolution from the
callback.

For a preinstalled bundle, `TextStore::open(path, bundle_path, options)` remains
available. Caller document IDs must fit in 96 bits; a higher revision replaces
the previous document and its chunks. See the [root text API guide](../../README.md#text-api-macos-apple-silicon)
for tower details, download size, and measured performance.

Licensed under [GPL-3.0-only](https://www.gnu.org/licenses/gpl-3.0.html).

## CI and the local text release gate

`.github/workflows/text-ci.yml` builds and lints text and FFI with `text`
enabled on hosted `macos-14`, runs every non-ignored test in both crates,
and builds the full graph+text XCFramework to run all Swift `TextStoreTests`.
Rust releases require these checks before publishing either crate.
The `text-x64-macos` and `text-windows` jobs build both crates without MLX
and run the platform contract tests (`backend_platform`, `ffi_text_platform`,
`ffi_namespace`). These two jobs have not run yet.

The dedicated real-model job runs
`default_model::real_default_model_downloads_verifies_and_reopens_offline` and
`release_tokenizer::release_tokenizer_ids_match_hf_0_19_1_for_both_towers`.
It downloads through `resolve_model` from HF, caches `ZE_MODEL_CACHE` keyed on
`DEFAULT_MODEL_REVISION`, and verifies cache hits. Locally the resolver uses
`ZE_MODEL_CACHE` or `~/Library/Caches/volantis-db/models`. The tokenizer test
compares both towers with `tests/goldens/e5_pair.rs` (SentenceTransformer 3.0.1
and HF tokenizers 0.19.1 on each `tokenizer.json`, prefixes included).

Only the following ignored tests require the local release gate. Run from the
repository root outside the sandbox with the named fixtures and native
MLX/CoreML available. Historical LEAF/Arctic/FP32 goldens require their original
models; substituting the published FP16 default would invalidate the assertions.

| Ignored tests | Required inputs | Local command |
| --- | --- | --- |
| Default-model inference | Verified HF cache or network | `cargo test --locked -p volantis-db-text --test default_model real_default_model_downloads_and_answers_queries -- --ignored --exact` |
| LEAF/Arctic parity and placement | `ZE_TEXT_TEST_BUNDLE_DIR`: original LEAF v1.5 pair and Arctic v2 symmetric bundles; LEAF `.mlmodelc` beside them or `ZE_COREML_QUERY_MODEL` | `cargo test --locked -p volantis-db-text --test arch --test placement -- --ignored` |
| clean71 FP32 parity | `ZE_CLEAN71_PACK`, ze-model Python dependencies | `cargo test --locked -p volantis-db-text --test arctic_query -- --ignored` |
| clean71 CoreML plan and parity | `ZE_CLEAN71_COREML_MODEL`, `ZE497_PACK`, `ZE497_COREML`, `ZE497_PYTHON` | `cargo test --locked -p volantis-db-text --test clean71_coreml -- --ignored` |
| Release smoke (MLX, all legs) and FP16 e5 pair layout | HF cache or `WF987_RELEASE` (a `ze-model release` folder) | `cargo test --locked -p volantis-db-text --test release -- --ignored` |
| e5 pair tokens and FP16 parity against the pack fp32 goldens (cosine >= 0.999 per row) | `E5_PACK`, `E5_BUNDLE` (the release `.zem`), `E5_PRECISION=fp16` | `cargo test --locked -p volantis-db-text --test e5_pair pack_ -- --ignored` |
| 30,001-document maintenance | Unsandboxed MLX runtime | `cargo test --locked -p volantis-db-text --test text_api -- --ignored` |

`model_cache::measure_large_cache_open` is an ignored manual benchmark, not a
correctness gate: `cargo test --locked -p volantis-db-text --test model_cache measure_large_cache_open -- --ignored --exact`.
