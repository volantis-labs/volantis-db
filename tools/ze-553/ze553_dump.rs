#![allow(clippy::expect_used, clippy::indexing_slicing)]
//! ZE-553 measurement helper. Not shipped. Copy to
//! crates/zeppelin-embed-text/examples/ to build.
//!
//! Embeds one text per input line with the shipped runtimes and writes
//! the exact token ids plus L2-normalized f32 vectors, so a CPU runtime
//! can embed the identical ids.
//!
//! ids   <bundle> <docs> <queries> <out>  token ids only, no GPU
//! doc   <bundle> <texts> <out>          MLX GPU document tower
//! query <bundle> <texts> <out> <mlmodelc> CoreML ANE + MLX GPU query
//!
//! Files: <out>.ids = per row u32 count then i32 ids;
//!        <out>.<runtime>.f32 = rows x dims little-endian f32.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zeppelin_embed_text::bundle::Bundle;
use zeppelin_embed_text::runtime::ModelRuntime;
use zeppelin_embed_text::runtime::coreml::{ComputeUnits, CoreMlRuntime};
use zeppelin_embed_text::runtime::mlx::MlxRuntime;
use zeppelin_embed_text::tower::{TokenBatch, TowerRole};

const DOC_MAX: usize = 510;
const BATCH: usize = 32;

fn rows_of(batch: &TokenBatch) -> Vec<Vec<i32>> {
    (0..batch.rows)
        .map(|row| {
            let start = row * batch.tokens_per_row;
            let end = start + batch.tokens_per_row;
            batch.token_ids[start..end]
                .iter()
                .zip(&batch.attention_mask[start..end])
                .filter(|(_, mask)| **mask > 0.0)
                .map(|(id, _)| *id)
                .collect()
        })
        .collect()
}

fn pack(rows: &[Vec<i32>]) -> TokenBatch {
    let width = rows.iter().map(Vec::len).max().expect("rows");
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for row in rows {
        ids.extend_from_slice(row);
        mask.extend(std::iter::repeat_n(1.0_f32, row.len()));
        ids.resize(ids.len() + width - row.len(), 0);
        mask.resize(mask.len() + width - row.len(), 0.0);
    }
    TokenBatch::new(ids, mask, rows.len(), width).expect("batch")
}

fn normalize(values: &mut [f32], dims: usize) {
    for row in values.chunks_mut(dims) {
        let norm = row
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        for v in row {
            *v = (f64::from(*v) / norm) as f32;
        }
    }
}

fn write_ids(path: &str, rows: &[Vec<i32>]) {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("ids"));
    for row in rows {
        out.write_all(&(row.len() as u32).to_le_bytes()).expect("w");
        for id in row {
            out.write_all(&id.to_le_bytes()).expect("w");
        }
    }
}

fn write_f32(path: &str, values: &[f32]) {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("f32"));
    for v in values {
        out.write_all(&v.to_le_bytes()).expect("w");
    }
}

fn doc_rows(bundle: &Bundle, lines: &[String]) -> Vec<Vec<i32>> {
    let mut rows = Vec::new();
    for line in lines {
        let mut row = rows_of(&bundle.tokenize_document_chunks(&[line.clone()]).expect("tok"))
            .remove(0);
        if row.len() > DOC_MAX {
            let last = *row.last().expect("eos");
            row.truncate(DOC_MAX - 1);
            row.push(last);
        }
        rows.push(row);
    }
    rows
}

fn pct(mut s: Vec<f64>, p: usize) -> f64 {
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    s[((s.len() - 1) * p) / 100]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, bundle_path, texts, out) = (&args[0], &args[1], &args[2], &args[3]);
    let bundle = Arc::new(Bundle::open(PathBuf::from(bundle_path)).expect("bundle"));
    let lines: Vec<String> = std::io::BufReader::new(std::fs::File::open(texts).expect("texts"))
        .lines()
        .map(|l| l.expect("line"))
        .collect();
    if mode == "ids" {
        let queries: Vec<String> = std::fs::read_to_string(&args[3])
            .expect("queries")
            .lines()
            .map(str::to_owned)
            .collect();
        write_ids(&format!("{}.docs.ids", args[4]), &doc_rows(&bundle, &lines));
        let rows: Vec<Vec<i32>> = queries
            .iter()
            .map(|l| rows_of(&bundle.tokenize_query(l).expect("tok")).remove(0))
            .collect();
        write_ids(&format!("{}.queries.ids", args[4]), &rows);
        return;
    }
    if mode == "doc" {
        let dims = bundle.document_tower().embedding.dims as usize;
        let rows = doc_rows(&bundle, &lines);
        write_ids(&format!("{out}.ids"), &rows);
        let mut runtime = MlxRuntime::load(Arc::clone(&bundle), TowerRole::Document).expect("mlx");
        runtime.warm().expect("warm");
        let _ = runtime.embed_batch(&pack(&rows[..BATCH])).expect("warm batch");
        let started = Instant::now();
        let mut values = Vec::with_capacity(rows.len() * dims);
        for chunk in rows.chunks(BATCH) {
            values.extend(runtime.embed_batch(&pack(chunk)).expect("embed").into_values());
        }
        let secs = started.elapsed().as_secs_f64();
        let tokens: usize = rows.iter().map(Vec::len).sum();
        println!(
            "DOC_MLX_GPU rows={} tokens={tokens} secs={secs:.3} rows_per_s={:.1} tokens_per_s={:.0}",
            rows.len(),
            rows.len() as f64 / secs,
            tokens as f64 / secs
        );
        normalize(&mut values, dims);
        write_f32(&format!("{out}.mlx.f32"), &values);
    } else {
        let model = PathBuf::from(&args[4]);
        let dims = bundle.query_tower().embedding.dims as usize;
        let batches: Vec<TokenBatch> = lines
            .iter()
            .map(|l| bundle.tokenize_query(l).expect("tok"))
            .collect();
        let rows: Vec<Vec<i32>> = batches.iter().map(|b| rows_of(b).remove(0)).collect();
        write_ids(&format!("{out}.ids"), &rows);
        let mut coreml =
            CoreMlRuntime::load(&model, dims, ComputeUnits::CpuAndNeuralEngine).expect("coreml");
        coreml.warm().expect("warm");
        let mut mlx = MlxRuntime::load(Arc::clone(&bundle), TowerRole::Query).expect("mlx");
        mlx.warm().expect("warm");
        for (name, runtime) in [
            ("coreml", &mut coreml as &mut dyn ModelRuntime),
            ("mlx", &mut mlx as &mut dyn ModelRuntime),
        ] {
            let mut values = Vec::with_capacity(rows.len() * dims);
            let mut samples = Vec::new();
            for batch in &batches {
                let _ = runtime.embed_batch(batch).expect("warm row");
                let started = Instant::now();
                values.extend(runtime.embed_batch(batch).expect("embed").into_values());
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "QUERY_{name} rows={} p50_ms={:.3} p95_ms={:.3}",
                rows.len(),
                pct(samples.clone(), 50),
                pct(samples, 95)
            );
            normalize(&mut values, dims);
            write_f32(&format!("{out}.{name}.f32"), &values);
        }
    }
}
