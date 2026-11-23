//! WF-991: the fp32 CPU runtime against MLX on the real FP16 bundle.
//!
//! Inputs: `WF991_BUNDLE` (the released `.zem`, offline) and optionally
//! `WF991_PACK` (the extracted pack, for a report-only golden comparison).
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use volantis_db::epoch::EmbeddingEpoch;
use volantis_db::ingest::{DocId, DocumentVersion, IngestBatch, IngestDocument, Revision};
use volantis_db::lifecycle::embedded::{EmbeddedChunk, EmbeddedLeg, NamespaceEmbedder};
use volantis_db::lifecycle::{CancelToken, OpenOptions, QueryControl, Store};
use volantis_db_text::NamespaceModel;
use volantis_db_text::bundle::Bundle;
use volantis_db_text::runtime::ModelRuntime;
use volantis_db_text::runtime::cpu::CpuRuntime;
use volantis_db_text::runtime::mlx::MlxRuntime;
use volantis_db_text::tower::TowerRole;

#[path = "goldens/e5_pair.rs"]
#[allow(dead_code)]
mod golden;

/// ZE-601 portable-store corpus and its fixed queries.
const CORPUS: [(u128, &str); 6] = [
    (
        60101,
        "A toy telescope measures the orbit of a pretend moon",
    ),
    (60102, "A paper violin plays an imaginary string concerto"),
    (60103, "A clockwork bakery kneads dough for a pretend loaf"),
    (60104, "A toy orchard grows imaginary pear trees"),
    (60105, "A paper sailboat races around a pretend harbor"),
    (60106, "A tiny loom weaves an imaginary wool scarf"),
];
const QUERIES: [(&str, u128); 3] = [
    ("moon orbit", 60101),
    ("violin concerto", 60102),
    ("baking bread", 60103),
];

fn bundle_path() -> PathBuf {
    PathBuf::from(
        std::env::var_os("WF991_BUNDLE")
            .unwrap_or_else(|| panic!("set WF991_BUNDLE to the released FP16 .zem")),
    )
}

/// A repetition of "pretend moon 雪 " longer than the document window.
fn long_text() -> String {
    "pretend moon 雪 ".repeat(200)
}

fn inputs() -> Vec<String> {
    let mut texts = golden::TEXTS
        .iter()
        .map(|text| (*text).to_owned())
        .collect::<Vec<_>>();
    texts.extend(CORPUS.iter().map(|(_, text)| (*text).to_owned()));
    texts.extend([
        "Fictif café 🛰️ 雪".to_owned(),
        "moon".to_owned(),
        long_text(),
    ]);
    texts
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
}

fn assert_unit(vector: &[f32], context: &str) {
    assert_eq!(vector.len(), 384, "{context}");
    assert!(vector.iter().all(|v| v.is_finite()), "{context}");
    let norm = dot(vector, vector);
    assert!((norm - 1.0).abs() < 1.0e-5, "{context}: norm {norm}");
}

/// Per-role cosine summary: (min, sum, count, worst input).
struct Stats {
    name: &'static str,
    min: f64,
    sum: f64,
    count: usize,
    worst: String,
}

impl Stats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            min: f64::INFINITY,
            sum: 0.0,
            count: 0,
            worst: String::new(),
        }
    }

    fn add(&mut self, cosine: f64, input: &str) {
        assert!(cosine.is_finite(), "{}: {input}", self.name);
        self.sum += cosine;
        self.count += 1;
        if cosine < self.min {
            self.min = cosine;
            self.worst = input.chars().take(48).collect();
        }
    }

    fn report(&self) {
        println!(
            "{}: rows={} min={:.9} mean={:.9} worst={:?}",
            self.name,
            self.count,
            self.min,
            self.sum / self.count as f64,
            self.worst
        );
        assert!(self.min >= 0.9999, "{} cosine {}", self.name, self.min);
    }
}

fn compare_chunks(
    stats: &mut Stats,
    cpu: &[EmbeddedChunk],
    mlx: &[EmbeddedChunk],
    text: &str,
) -> Vec<std::ops::Range<usize>> {
    assert_eq!(cpu.len(), mlx.len(), "chunk count: {text}");
    for (a, b) in cpu.iter().zip(mlx) {
        assert_eq!(a.bytes, b.bytes, "chunk span: {text}");
        assert_unit(&a.vector, text);
        assert_unit(&b.vector, text);
        stats.add(cosine(&a.vector, &b.vector), text);
    }
    cpu.iter().map(|chunk| chunk.bytes.clone()).collect()
}

/// Reads a little-endian f32 C-order `.npy` array with the given shape.
fn npy(path: &Path, rows: usize, columns: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden .npy");
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let text = std::str::from_utf8(&bytes[10..10 + header]).expect("npy header");
    assert!(
        text.contains(&format!("'shape': ({rows}, {columns})")),
        "{text}"
    );
    bytes[10 + header..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("f32")))
        .collect()
}

#[test]
#[ignore = "requires WF991_BUNDLE (released FP16 bundle) and native Metal; run in release"]
fn wf991_cpu_matches_mlx_both_towers() {
    let path = bundle_path();
    let cpu = NamespaceModel::open_bundle_cpu(&path).expect("CPU namespace model");
    let mlx = NamespaceModel::open_bundle(&path).expect("MLX namespace model");
    let epoch = |model: &NamespaceModel| -> EmbeddingEpoch { model.epoch() };
    assert!(epoch(&cpu) == epoch(&mlx), "the CPU keeps the bundle epoch");
    let texts = inputs();
    let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();

    // Document tower, batched and single, through the production chunker.
    let started = std::time::Instant::now();
    let cpu_batch = cpu.documents(&refs).expect("CPU documents");
    let cpu_seconds = started.elapsed().as_secs_f64();
    let mlx_batch = mlx.documents(&refs).expect("MLX documents");
    let mut batched = Stats::new("document batched");
    let mut single = Stats::new("document single");
    for ((text, cpu_chunks), mlx_chunks) in refs.iter().zip(&cpu_batch).zip(&mlx_batch) {
        let spans = compare_chunks(&mut batched, cpu_chunks, mlx_chunks, text);
        let cpu_one = cpu.document(text).expect("CPU document");
        let mlx_one = mlx.document(text).expect("MLX document");
        assert_eq!(compare_chunks(&mut single, &cpu_one, &mlx_one, text), spans);
    }
    let long = cpu_batch.last().expect("long text");
    assert!(long.len() >= 2, "the long text spans several windows");
    assert_eq!(long.first().map(|c| c.bytes.start), Some(0));
    assert_eq!(long.last().map(|c| c.bytes.end), Some(long_text().len()));
    assert!(
        long.windows(2)
            .all(|pair| pair[0].bytes.end == pair[1].bytes.start)
    );

    // Query tower, single rows through the production path.
    let query_texts = &refs[..refs.len() - 1];
    let mut query_single = Stats::new("query single");
    for text in query_texts {
        let a = cpu.query(text).expect("CPU query");
        let b = mlx.query(text).expect("MLX query");
        assert_unit(&a, text);
        assert_unit(&b, text);
        query_single.add(cosine(&a, &b), text);
    }

    // Raw batched rows: one tokenization feeds both runtimes, so both see
    // the same token IDs and masks.
    let bundle = Arc::new(Bundle::open(&path).expect("bundle"));
    let mut query_batched = Stats::new("query batched (raw)");
    let mut document_raw = Stats::new("document batched (raw)");
    for (role, stats) in [
        (TowerRole::Query, &mut query_batched),
        (TowerRole::Document, &mut document_raw),
    ] {
        let tokens = match role {
            TowerRole::Query => bundle.tokenize_queries(query_texts),
            TowerRole::Document => bundle.tokenize_document_chunks(
                &golden::TEXTS
                    .iter()
                    .map(|t| (*t).to_owned())
                    .collect::<Vec<_>>(),
            ),
        }
        .expect("tokens");
        let mut cpu_runtime = CpuRuntime::load(Arc::clone(&bundle), role).expect("CPU tower");
        let mut mlx_runtime = MlxRuntime::load(Arc::clone(&bundle), role).expect("MLX tower");
        let a = cpu_runtime.embed_batch(&tokens).expect("CPU batch");
        let b = mlx_runtime.embed_batch(&tokens).expect("MLX batch");
        for ((row_a, row_b), text) in a
            .values()
            .chunks_exact(a.dims())
            .zip(b.values().chunks_exact(b.dims()))
            .zip(query_texts)
        {
            stats.add(cosine(row_a, row_b), text);
        }
        println!(
            "{role:?}: kernels={} workers={}",
            cpu_runtime.kernels(),
            cpu_runtime.workers()
        );
    }

    // Report only: CPU (fp32 over FP16 weights) against the pack fp32 goldens.
    if let Some(pack) = std::env::var_os("WF991_PACK").map(PathBuf::from) {
        for (role, file) in [
            ("query", "query_golden_fp32.npy"),
            ("document", "document_golden_fp32.npy"),
        ] {
            let rows = npy(&pack.join("fixtures").join(file), golden::TEXTS.len(), 384);
            let (mut worst_cosine, mut worst_abs) = (f64::INFINITY, 0.0_f32);
            for (row, text) in golden::TEXTS.iter().enumerate() {
                let vector = if role == "query" {
                    cpu.query(text).expect("query")
                } else {
                    cpu.document(text).expect("document")[0].vector.clone()
                };
                let expected = &rows[row * 384..(row + 1) * 384];
                worst_cosine = worst_cosine.min(cosine(&vector, expected));
                for (a, e) in vector.iter().zip(expected) {
                    worst_abs = worst_abs.max((a - e).abs());
                }
            }
            println!(
                "report only, CPU vs pack fp32 goldens, {role}: min cosine {worst_cosine:.9}, max |diff| {worst_abs:e}"
            );
        }
    }
    for stats in [
        &batched,
        &single,
        &query_single,
        &query_batched,
        &document_raw,
    ] {
        stats.report();
    }
    println!(
        "wall clock (sanity only): CPU document batch of {} inputs in {cpu_seconds:.2} s",
        refs.len()
    );
}

/// One phase of the reopen test in a fresh process (see the test below).
#[test]
#[ignore = "child phase of wf991_cross_backend_reopen"]
fn wf991_reopen_phase() {
    let phase = std::env::var("WF991_PHASE").expect("WF991_PHASE");
    let store_path = PathBuf::from(std::env::var_os("WF991_STORE").expect("WF991_STORE"));
    let path = bundle_path();
    let (action, backend) = phase.split_once(':').expect("action:backend");
    let model = Arc::new(
        match backend {
            "cpu" => NamespaceModel::open_bundle_cpu(&path),
            "mlx" => NamespaceModel::open_bundle(&path),
            other => panic!("unknown backend {other}"),
        }
        .expect("namespace model"),
    );
    let store =
        Store::open_embedded(&store_path, OpenOptions::new(), model.clone()).expect("open store");
    match action {
        "odd" | "even" | "all" => {
            let rows = CORPUS
                .iter()
                .filter(|(id, _)| match action {
                    "odd" => id % 2 == 1,
                    "even" => id % 2 == 0,
                    _ => true,
                })
                .map(|(id, text)| {
                    IngestDocument::new(
                        DocumentVersion::new(DocId::new(*id), Revision::new(1)),
                        vec![],
                    )
                    .with_text(*text)
                })
                .collect();
            store.ingest(IngestBatch::new(rows)).expect("write");
        }
        "read" => {
            for (query, _) in QUERIES {
                let report = store
                    .query_embedded_report(
                        query,
                        EmbeddedLeg::Dense,
                        3,
                        None,
                        QueryControl::Cancel(CancelToken::new()),
                    )
                    .expect("dense report");
                let hits = report
                    .hits
                    .iter()
                    .map(|hit| {
                        format!(
                            "{}@{}..{}",
                            hit.version.doc_id().get(),
                            hit.bytes.start,
                            hit.bytes.end
                        )
                    })
                    .collect::<Vec<_>>();
                println!("WF991 HITS {query}={}", hits.join(","));
            }
            // Reopening and reading old rows never runs the document tower.
            println!(
                "WF991 LOADS {}",
                model.document_load_count().expect("load count")
            );
        }
        other => panic!("unknown action {other}"),
    }
    store.close().expect("close store");
}

/// Runs one phase in a fresh process and returns its `WF991` lines.
fn phase(store: &Path, phase: &str) -> Vec<String> {
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args(["wf991_reopen_phase", "--exact", "--ignored", "--nocapture"])
        .env("WF991_PHASE", phase)
        .env("WF991_STORE", store)
        .env("WF991_BUNDLE", bundle_path())
        .output()
        .expect("run phase");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "phase {phase} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("WF991 ").map(str::to_owned))
        .collect()
}

/// Top-1 id and top-3 id set per query, plus the chunk span of each hit.
fn ranking(lines: &[String]) -> Vec<(String, u128, Vec<String>)> {
    assert!(lines.iter().any(|line| line == "LOADS 0"), "{lines:?}");
    lines
        .iter()
        .filter_map(|line| line.strip_prefix("HITS "))
        .map(|line| {
            let (query, hits) = line.split_once('=').expect("query=hits");
            let mut hits = hits.split(',').map(str::to_owned).collect::<Vec<_>>();
            let top = hits[0]
                .split('@')
                .next()
                .and_then(|id| id.parse().ok())
                .expect("top id");
            hits.sort();
            (query.to_owned(), top, hits)
        })
        .collect()
}

#[test]
#[ignore = "requires WF991_BUNDLE (released FP16 bundle) and native Metal"]
fn wf991_cross_backend_reopen() {
    let root = tempfile::tempdir().expect("root");
    let (a, b, control) = (
        root.path().join("cpu-first"),
        root.path().join("mlx-first"),
        root.path().join("mlx-only"),
    );
    // Odd rows: A on CPU, B on MLX. Each reopens on the other backend.
    phase(&a, "odd:cpu");
    phase(&b, "odd:mlx");
    let a_on_mlx = ranking(&phase(&a, "read:mlx"));
    let b_on_cpu = ranking(&phase(&b, "read:cpu"));
    let b_on_mlx = ranking(&phase(&b, "read:mlx"));
    assert_eq!(
        a_on_mlx, b_on_mlx,
        "CPU-written rows answer like MLX-written rows on MLX"
    );
    let a_on_cpu = ranking(&phase(&a, "read:cpu"));
    assert_eq!(
        b_on_cpu, a_on_cpu,
        "MLX-written rows answer like CPU-written rows on CPU"
    );
    // Even rows: appended on the other backend. Then the mixed stores reopen.
    phase(&a, "even:mlx");
    phase(&b, "even:cpu");
    phase(&control, "all:mlx");
    let expected = ranking(&phase(&control, "read:mlx"));
    for (store, backend) in [(&a, "mlx"), (&a, "cpu"), (&b, "mlx"), (&b, "cpu")] {
        let observed = ranking(&phase(store, &format!("read:{backend}")));
        println!("{} on {backend}: {observed:?}", store.display());
        assert_eq!(observed, expected, "{} read on {backend}", store.display());
    }
    for ((query, top, _), (_, winner)) in expected.iter().zip(QUERIES) {
        println!("control: {query} -> {top}");
        assert_eq!(*top, winner, "{query}");
    }
    // A different model pin still refuses the store.
    let other = root.path().join("tiny.zem");
    std::fs::write(&other, include_bytes!("fixtures/wf986/pair.zem")).expect("tiny pair");
    let tiny = Arc::new(NamespaceModel::open_bundle_cpu(&other).expect("tiny model"));
    assert!(
        Store::open_embedded(&a, OpenOptions::new(), tiny).is_err(),
        "a different pin opens the store"
    );
}
