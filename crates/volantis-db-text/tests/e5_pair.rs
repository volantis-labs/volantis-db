#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
//! WF-986: two BERT towers with per-role XLM-R Unigram tokenizers and prompts.

use std::path::{Path, PathBuf};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::Arc;

use tempfile::tempdir;
use volantis_db::lifecycle::embedded::NamespaceEmbedder;
use volantis_db_text::NamespaceModel;
use volantis_db_text::bundle::{Bundle, BundleError};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use volantis_db_text::runtime::ModelRuntime;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use volantis_db_text::runtime::mlx::MlxRuntime;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use volantis_db_text::tower::TowerRole;
use volantis_db_text::tower::{Pooling, TokenBatch};

#[path = "fixtures/wf986/reference.rs"]
#[allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]
mod reference;

#[path = "goldens/e5_pair.rs"]
#[allow(dead_code)]
mod golden;

const FIXTURE: &[u8] = include_bytes!("fixtures/wf986/pair.zem");
const PAD: i32 = 1;

fn write(directory: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = directory.join(name);
    std::fs::write(&path, bytes).expect("write bundle fixture");
    path
}

fn redigest(bytes: &mut [u8]) {
    let end = bytes.len() - 16;
    let digest = xxhash_rust::xxh3::xxh3_128(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest.to_le_bytes());
}

/// Byte ranges of the two tokenizer records, read with the layout-3 field order.
fn tokenizer_records(bytes: &[u8]) -> [std::ops::Range<usize>; 2] {
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("u32"));
    let mut at = 24;
    for _ in 0..2 {
        at += 1;
        for _ in 0..3 {
            at += 4 + u32_at(at) as usize;
        }
        at += 6;
        at += 4 + u32_at(at) as usize;
        at += 44;
    }
    let mut ranges = Vec::new();
    for _ in 0..2 {
        let start = at;
        at += 20;
        let count = u32_at(at) as usize;
        at += 4;
        for _ in 0..count {
            at += 4 + u32_at(at) as usize + 8;
        }
        at += 4 + u32_at(at) as usize;
        ranges.push(start..at);
    }
    [ranges[0].clone(), ranges[1].clone()]
}

fn rows(batch: &TokenBatch) -> Vec<(Vec<i32>, Vec<f32>)> {
    (0..batch.rows)
        .map(|row| {
            let span = row * batch.tokens_per_row..(row + 1) * batch.tokens_per_row;
            (
                batch.token_ids[span.clone()].to_vec(),
                batch.attention_mask[span].to_vec(),
            )
        })
        .collect()
}

fn padded(expected: &[i32], width: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = expected.to_vec();
    ids.resize(width, PAD);
    let mut mask = vec![1.0; expected.len()];
    mask.resize(width, 0.0);
    (ids, mask)
}

fn query_rows(bundle: &Bundle, texts: &[&str]) -> Vec<Vec<i32>> {
    texts
        .iter()
        .map(|text| bundle.tokenize_query(text).expect("query").token_ids)
        .collect()
}

fn document_rows(bundle: &Bundle, texts: &[&str]) -> Vec<Vec<i32>> {
    texts
        .iter()
        .map(|text| {
            bundle
                .tokenize_document_chunks(&[(*text).to_owned()])
                .expect("document")
                .token_ids
        })
        .collect()
}

#[test]
fn role_tokens() {
    let directory = tempdir().expect("tempdir");
    let bundle = Bundle::open(write(directory.path(), "pair.zem", FIXTURE)).expect("pair");
    assert!(!bundle.is_symmetric());
    for (tower, name, prefix, limit) in [
        (
            bundle.query_tower(),
            reference::QUERY_NAME,
            reference::QUERY_PREFIX,
            reference::QUERY_MAX_TOKENS,
        ),
        (
            bundle.document_tower(),
            reference::DOCUMENT_NAME,
            reference::DOCUMENT_PREFIX,
            reference::DOCUMENT_MAX_TOKENS,
        ),
    ] {
        assert_eq!(tower.embedding.model_id, name);
        assert_eq!(tower.embedding.prompt_prefix, prefix);
        assert_eq!(tower.embedding.max_tokens as usize, limit);
        assert_eq!(tower.pooling, Pooling::Mean);
    }

    // Short single rows: no padding, every position attends.
    for (role, cases) in [
        ("query", reference::QUERY_CASES),
        ("document", reference::DOCUMENT_CASES),
    ] {
        for (name, text, expected) in cases {
            let batch = if role == "query" {
                bundle.tokenize_query(text)
            } else {
                bundle.tokenize_document_chunks(&[(*text).to_owned()])
            }
            .expect("tokenize case");
            assert_eq!(&batch.token_ids, expected, "{role}/{name}");
            assert_eq!(
                batch.attention_mask,
                vec![1.0; expected.len()],
                "{role}/{name}"
            );
        }
    }

    // One ragged batch per role through the batch entry points: right padding with <pad>.
    let texts = reference::QUERY_CASES
        .iter()
        .map(|case| case.1)
        .collect::<Vec<_>>();
    let queries = bundle.tokenize_queries(&texts).expect("query batch");
    let documents = bundle
        .tokenize_document_chunks(&texts.iter().map(|t| (*t).to_owned()).collect::<Vec<_>>())
        .expect("document batch");
    for (batch, cases) in [
        (&queries, reference::QUERY_CASES),
        (&documents, reference::DOCUMENT_CASES),
    ] {
        let width = cases.iter().map(|case| case.2.len()).max().expect("cases");
        assert_eq!((batch.rows, batch.tokens_per_row), (cases.len(), width));
        for (row, (name, _, expected)) in rows(batch).into_iter().zip(cases.iter()) {
            assert_eq!(row, padded(expected, width), "batched {name}");
        }
    }
    // Fixed-width padding (CoreML shape) keeps the real IDs and masks the rest.
    // TokenBatch::padded_to writes ID 0 under a zero mask; masked positions
    // do not reach attention or mean pooling, so only the mask is asserted.
    let fixed = bundle
        .tokenize_query_padded(texts[0], 12)
        .expect("padded query");
    let expected = reference::QUERY_CASES[0].2;
    assert_eq!((fixed.rows, fixed.tokens_per_row), (1, 12));
    assert_eq!(&fixed.token_ids[..expected.len()], expected);
    assert_eq!(fixed.attention_mask, padded(expected, 12).1);

    // Each record serves only its own role. Swapping them must be loud or wrong.
    let [first, second] = tokenizer_records(FIXTURE);
    let mut swapped = FIXTURE[..first.start].to_vec();
    swapped.extend_from_slice(&FIXTURE[second.clone()]);
    swapped.extend_from_slice(&FIXTURE[first.clone()]);
    swapped.extend_from_slice(&FIXTURE[second.end..]);
    let mut checked = swapped.clone();
    redigest(&mut checked);
    assert!(matches!(
        Bundle::open(write(directory.path(), "swapped.zem", &checked)),
        Err(BundleError::Format(detail)) if detail.contains("rows for tokenizer vocabulary")
    ));
    // Hide both embedding tables so only token IDs are compared.
    for name in [
        &b"query/embeddings.word_embeddings.weight"[..],
        &b"document/embeddings.word_embeddings.weight"[..],
    ] {
        let at = swapped
            .windows(name.len())
            .position(|s| s == name)
            .expect("descriptor");
        swapped[at + name.len() - 1] = b'X';
    }
    redigest(&mut swapped);
    let wrong = Bundle::open(write(directory.path(), "hidden.zem", &swapped)).expect("swapped");
    let expected_query = reference::QUERY_CASES
        .iter()
        .map(|case| case.2.to_vec())
        .collect::<Vec<_>>();
    let expected_document = reference::DOCUMENT_CASES
        .iter()
        .map(|case| case.2.to_vec())
        .collect::<Vec<_>>();
    assert_ne!(query_rows(&wrong, &texts), expected_query);
    assert_ne!(document_rows(&wrong, &texts), expected_document);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn assert_vector(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} dimensions");
    assert!(actual.iter().all(|v| v.is_finite()), "{context} finite");
    let norm = actual.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
    assert!((norm - 1.0).abs() < 1.0e-5, "{context} unit norm {norm}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < 2.0e-5,
            "{context}: {actual:?} != {expected:?}"
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn tiny_bert_unigram_vectors() {
    let directory = tempdir().expect("tempdir");
    let path = write(directory.path(), "pair.zem", FIXTURE);
    let model = NamespaceModel::open_bundle(&path).expect("namespace model");
    // Production query path: prefix, tokenizer, MLX BERT, mean pooling, L2.
    for (row, (name, text, _)) in reference::QUERY_CASES.iter().enumerate() {
        let vector = model.query(text).expect("query vector");
        assert_vector(&vector, &reference::QUERY_VECTORS[row], name);
    }
    // Production document path: one ragged batch of single-chunk documents.
    let fits = reference::DOCUMENT_CASES
        .iter()
        .enumerate()
        .filter(|(_, case)| case.2.len() < reference::DOCUMENT_MAX_TOKENS)
        .collect::<Vec<_>>();
    let texts = fits.iter().map(|(_, case)| case.1).collect::<Vec<_>>();
    let documents = model.documents(&texts).expect("document vectors");
    for ((row, (name, text, _)), chunks) in fits.iter().zip(&documents) {
        assert_eq!(chunks.len(), 1, "{name} is one chunk");
        assert_eq!(chunks[0].bytes, 0..text.len(), "{name} raw range");
        assert_vector(&chunks[0].vector, &reference::DOCUMENT_VECTORS[*row], name);
    }
    // Ragged MLX query rows equal the independent single-row reference.
    let bundle = Arc::new(Bundle::open(&path).expect("bundle"));
    let tokens = bundle
        .tokenize_queries(
            &reference::QUERY_CASES
                .iter()
                .map(|c| c.1)
                .collect::<Vec<_>>(),
        )
        .expect("ragged queries");
    let mut runtime = MlxRuntime::load(Arc::clone(&bundle), TowerRole::Query).expect("query");
    let raw = runtime.embed_batch(&tokens).expect("ragged query batch");
    for (row, values) in raw.values().chunks_exact(raw.dims()).enumerate() {
        let norm = values
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(norm > 0.0 && norm.is_finite());
        let unit = values
            .iter()
            .map(|v| (f64::from(*v) / norm) as f32)
            .collect::<Vec<_>>();
        assert_vector(&unit, &reference::QUERY_VECTORS[row], "ragged query");
    }
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "set {name}: E5_PACK=<extracted pack>, E5_BUNDLE=<baked .zem>, E5_PRECISION=fp32|fp16"
        )
    })
}

#[test]
fn pack_tests_require_explicit_inputs() {
    for test in ["pack_tokens", "pack_vectors"] {
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([test, "--exact", "--ignored", "--nocapture"])
            .env_remove("E5_PACK")
            .env_remove("E5_BUNDLE")
            .env_remove("E5_PRECISION")
            .output()
            .expect("run pack test without inputs");
        assert!(!output.status.success(), "{test} must fail without inputs");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("set E5_"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires E5_BUNDLE baked from the local e5 pair pack; no MLX"]
fn pack_tokens() {
    let bundle = Bundle::open(required("E5_BUNDLE")).expect("open e5 pair bundle");
    for (role, digest, prefix, limit, cases) in [
        (
            "query",
            golden::QUERY_SHA256,
            golden::QUERY_PREFIX,
            golden::QUERY_MAX_TOKENS,
            golden::QUERY_CASES,
        ),
        (
            "document",
            golden::DOCUMENT_SHA256,
            golden::DOCUMENT_PREFIX,
            golden::DOCUMENT_MAX_TOKENS,
            golden::DOCUMENT_CASES,
        ),
    ] {
        let tower = if role == "query" {
            bundle.query_tower()
        } else {
            bundle.document_tower()
        };
        assert_eq!(tower.embedding.prompt_prefix, prefix);
        assert_eq!(tower.embedding.max_tokens as usize, limit);
        let mut longest = 0;
        for (name, text, repeat, expected) in cases {
            let input = text.repeat(*repeat);
            let batch = if role == "query" {
                bundle.tokenize_query(&input)
            } else {
                bundle.tokenize_document_chunks(&[input])
            }
            .expect("tokenize pack case");
            assert_eq!(&batch.token_ids, expected, "{role}/{name}, source {digest}");
            assert_eq!(batch.attention_mask, vec![1.0; expected.len()]);
            longest = longest.max(expected.len());
        }
        assert_eq!(longest, limit, "{role} cases reach the true limit");
        println!(
            "{role}: {} cases match tokenizer.json {digest}",
            cases.len()
        );
    }
}

/// Reads a little-endian f32 C-order `.npy` array with the given shape.
fn npy(path: &Path, rows: usize, columns: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden .npy");
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let text = std::str::from_utf8(&bytes[10..10 + header]).expect("npy header");
    assert!(text.contains("'descr': '<f4'") && text.contains("'fortran_order': False"));
    assert!(
        text.contains(&format!("'shape': ({rows}, {columns})")),
        "{text}"
    );
    let data = &bytes[10 + header..];
    assert_eq!(data.len(), rows * columns * 4);
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("f32")))
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

#[test]
#[ignore = "requires E5_PACK, E5_BUNDLE and E5_PRECISION; inference needs native MLX/Metal"]
fn pack_vectors() {
    let pack = PathBuf::from(required("E5_PACK"));
    let bundle = PathBuf::from(required("E5_BUNDLE"));
    let precision = required("E5_PRECISION");
    let floor = match precision.as_str() {
        "fp32" => 0.9999,
        // WF-987 owner gate for the FP16 release bundle, per row against fp32 goldens.
        "fp16" => 0.999,
        other => panic!("set E5_PRECISION to fp32 or fp16, got {other}"),
    };
    let texts = std::fs::read(pack.join("fixtures/texts.json")).expect("texts.json");
    assert_eq!(
        xxhash_rust::xxh3::xxh3_64(&texts),
        golden::TEXTS_XXH3,
        "texts.json differs from the pinned fixture"
    );
    let model = NamespaceModel::open_bundle(&bundle).expect("namespace model");
    let mut query_vectors = Vec::new();
    for (role, file) in [
        ("query", "query_golden_fp32.npy"),
        ("document", "document_golden_fp32.npy"),
    ] {
        let golden_rows = npy(&pack.join("fixtures").join(file), golden::TEXTS.len(), 384);
        let mut worst = (f64::INFINITY, 0);
        for (row, text) in golden::TEXTS.iter().enumerate() {
            let vector = if role == "query" {
                model.query(text).expect("query vector")
            } else {
                let chunks = model.document(text).expect("document vector");
                assert_eq!(chunks.len(), 1, "document row {row} is one chunk");
                assert_eq!(chunks[0].bytes, 0..text.len());
                chunks[0].vector.clone()
            };
            assert_eq!(vector.len(), 384);
            assert!(vector.iter().all(|v| v.is_finite()));
            let norm = dot(&vector, &vector);
            assert!((norm - 1.0).abs() < 1.0e-5, "{role} row {row} norm {norm}");
            let expected = &golden_rows[row * 384..(row + 1) * 384];
            let cosine = dot(&vector, expected) / dot(expected, expected).sqrt();
            if cosine < worst.0 {
                worst = (cosine, row);
            }
            if role == "query" {
                query_vectors.push(vector);
            }
        }
        println!(
            "{precision} {role}: worst cosine {:.9} at row {} of {}",
            worst.0,
            worst.1,
            golden::TEXTS.len()
        );
        assert!(worst.0 >= floor, "{precision} {role} cosine {:?}", worst);
    }
    let france = model
        .query("What is the capital of France?")
        .expect("France query");
    let paris = model
        .document("Paris is the capital of France.")
        .expect("Paris");
    let banana = model.document("Bananas are yellow.").expect("banana");
    let scores = [
        dot(&france, &paris[0].vector),
        dot(&france, &banana[0].vector),
    ];
    println!("{precision} sanity scores (Paris, banana): {scores:?}");
    assert!(scores[0] > scores[1]);
    assert_eq!(query_vectors.len(), golden::TEXTS.len());
}

/// Flag bit 2 needs a Unigram record with a charsmap and no case or accent
/// flags; unknown bits stay unsupported.
#[test]
fn metaspace_flag_requires_unigram_and_charsmap() {
    let directory = tempdir().expect("tempdir");
    let [first, _] = tokenizer_records(FIXTURE);
    let open = |bytes: &mut Vec<u8>, name: &str| {
        redigest(bytes);
        Bundle::open(write(directory.path(), name, bytes))
    };
    for (kind, flags, name) in [(2_u8, 4 | 1, "case"), (2, 4 | 2, "accents"), (1, 4, "kind")] {
        let mut bytes = FIXTURE.to_vec();
        bytes[first.start] = kind;
        bytes[first.start + 1] = flags;
        assert!(
            matches!(open(&mut bytes, name), Err(BundleError::Format(_))),
            "{name}"
        );
    }
    let mut unknown = FIXTURE.to_vec();
    unknown[first.start + 1] = 4 | 8;
    assert!(matches!(
        open(&mut unknown, "unknown.zem"),
        Err(BundleError::Unsupported {
            field: "tokenizer flags",
            id: 12
        })
    ));
    // Replace the charsmap with an empty field; keep data offsets unchanged.
    let header = u32::from_le_bytes(FIXTURE[12..16].try_into().expect("header")) as usize;
    let mut at = first.start + 20;
    let count = u32::from_le_bytes(FIXTURE[at..at + 4].try_into().expect("count")) as usize;
    at += 4;
    for _ in 0..count {
        at += 4 + u32::from_le_bytes(FIXTURE[at..at + 4].try_into().expect("len")) as usize + 8;
    }
    let removed = first.end - at - 4;
    let mut empty = FIXTURE[..at].to_vec();
    empty.extend_from_slice(&0_u32.to_le_bytes());
    empty.extend_from_slice(&FIXTURE[first.end..header]);
    empty.extend(std::iter::repeat_n(0, removed));
    empty.extend_from_slice(&FIXTURE[header..]);
    assert_eq!(empty.len(), FIXTURE.len());
    assert!(matches!(
        open(&mut empty, "empty-charsmap.zem"),
        Err(BundleError::Format(detail)) if detail.contains("charsmap")
    ));
}
