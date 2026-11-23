//! Bundle-backed towers for an ordinary Store.
use super::*;
use volantis_db::epoch::{EmbedderError, EmbedderFailure, EmbeddingEpoch};
use volantis_db::lifecycle::embedded::{EmbeddedChunk, NamespaceEmbedder};

/// One verified model runtime that can be shared by Store snapshot handles.
pub struct NamespaceModel {
    bundle: Arc<Bundle>,
    runtime: RuntimeClient,
    epoch: EmbeddingEpoch,
}

impl NamespaceModel {
    /// Opens an explicit local bundle.
    pub fn open_bundle(path: &Path) -> Result<Self, TextError> {
        let bundle = Arc::new(Bundle::open(path)?);
        crate::runtime::mlx_cache::configure(None)?;
        let runtime = RuntimeClient::start(bundle.clone(), discover_query_coreml(path))?;
        Ok(Self {
            epoch: embedding_epoch(&bundle),
            bundle,
            runtime,
        })
    }

    /// Resolves an immutable model and binds all three pins into the epoch.
    pub fn open_model(
        model: &crate::ModelRef,
        options: &crate::ModelOpenOptions<'_>,
    ) -> Result<Self, TextError> {
        let path = crate::resolve_model(model, options).map_err(TextError::ModelResolution)?;
        crate::runtime::mlx_cache::configure(options.text.mlx_cache_limit_bytes)?;
        let mut this = Self::open_bundle(&path)?;
        // Length framing preserves the bundle's alignment digest and prevents
        // a different repository, revision or manifest from reopening the rows.
        let mut bytes = b"ZEMREF01".to_vec();
        for field in [
            model.repo.as_bytes(),
            model.revision.as_bytes(),
            model.manifest_sha256.as_bytes(),
            &this.epoch.alignment_digest,
        ] {
            bytes.extend_from_slice(
                &u32::try_from(field.len())
                    .map_err(|_| TextError::InvalidInput("model identity too long"))?
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(field);
        }
        this.epoch.alignment_digest = bytes;
        Ok(this)
    }

    fn evaluate(&self, role: TowerRole, tokens: TokenBatch) -> Result<Vec<f32>, TextError> {
        let mut batch = self.runtime.embed(role, tokens, false)?;
        let tower = match role {
            TowerRole::Document => self.bundle.document_tower(),
            TowerRole::Query => self.bundle.query_tower(),
        };
        normalize_batch(&mut batch, &tower.embedding)?;
        Ok(batch.into_values())
    }

    fn visit_chunks(
        &self,
        text: &str,
        mut accept: impl FnMut(std::ops::Range<usize>, Vec<u32>) -> Result<(), TextError>,
    ) -> Result<(), TextError> {
        let tower = &self.bundle.document_tower().embedding;
        let limit = tower.max_tokens as usize;
        let tokenizer = self.bundle.tokenizer(TowerRole::Document);
        let mut start = 0;
        loop {
            let rest = text
                .get(start..)
                .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
            let mut end = rest
                .char_indices()
                .nth(limit.saturating_mul(8))
                .map_or(rest.len(), |(i, _)| i);
            let tokens = loop {
                let body = rest
                    .get(..end)
                    .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
                let input = format!("{}{body}", tower.prompt_prefix);
                // One extra slot detects truncation, including prompt and specials.
                let tokens = tokenizer.encode_row(
                    &input,
                    limit
                        .checked_add(1)
                        .ok_or(TextError::InvalidInput("token limit overflow"))?,
                );
                if tokens.len() <= limit {
                    break tokens;
                }
                let midpoint = body.floor_char_boundary(end / 2);
                let next = body
                    .get(..midpoint)
                    .and_then(|s| s.rfind(char::is_whitespace).filter(|i| *i > 0))
                    .unwrap_or(midpoint);
                if next == 0 {
                    return Err(TextError::InvalidInput(
                        "source character and prefix exceed tower window",
                    ));
                }
                end = next;
            };
            let next = start
                .checked_add(end)
                .ok_or(TextError::InvalidInput("chunk offset overflow"))?;
            accept(start..next, tokens)?;
            start = next;
            if start == text.len() {
                break;
            }
        }
        Ok(())
    }
    fn batch_chunks(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, TextError> {
        let mut window = Vec::with_capacity(MAX_EVAL_ROWS);
        let mut output = (0..texts.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (source, text) in texts.iter().enumerate() {
            self.visit_chunks(text, |bytes, tokens| {
                let chunks = output
                    .get_mut(source)
                    .ok_or(TextError::InvalidInput("missing source"))?;
                window.push(PendingChunk {
                    source,
                    ordinal: chunks.len(),
                    tokens,
                });
                chunks.push(EmbeddedChunk {
                    bytes,
                    vector: Vec::new(),
                });
                if window.len() == MAX_EVAL_ROWS {
                    self.flush(&mut window, &mut output)?;
                }
                Ok(())
            })?;
        }
        self.flush(&mut window, &mut output)?;
        Ok(output)
    }

    fn flush(
        &self,
        window: &mut Vec<PendingChunk>,
        output: &mut [Vec<EmbeddedChunk>],
    ) -> Result<(), TextError> {
        let tokenizer = self.bundle.tokenizer(TowerRole::Document);
        submit_groups(window, |group| {
            let tokens = tokenizer.pack_rows(group.iter().map(|chunk| chunk.tokens.as_slice()))?;
            let mut batch = self.runtime.embed(TowerRole::Document, tokens, false)?;
            normalize_batch(&mut batch, &self.bundle.document_tower().embedding)?;
            for (chunk, row) in group.iter().zip(embedding_rows(&batch, group.len())?) {
                let slot = output
                    .get_mut(chunk.source)
                    .and_then(|chunks| chunks.get_mut(chunk.ordinal))
                    .filter(|slot| slot.vector.is_empty())
                    .ok_or(TextError::InvalidInput("invalid chunk position"))?;
                slot.vector = row.to_vec();
            }
            Ok(())
        })?;
        window.clear();
        Ok(())
    }
}

use crate::runtime::mlx::{MAX_EVAL_ROWS, fits_eval};

struct PendingChunk {
    source: usize,
    ordinal: usize,
    tokens: Vec<u32>,
}

fn submit_groups(
    window: &mut [PendingChunk],
    mut submit: impl FnMut(&[PendingChunk]) -> Result<(), TextError>,
) -> Result<(), TextError> {
    window.sort_unstable_by_key(|chunk| (chunk.tokens.len(), chunk.source, chunk.ordinal));
    let mut pending = &*window;
    while !pending.is_empty() {
        let count = pending
            .iter()
            .enumerate()
            .take_while(|(index, chunk)| fits_eval(index + 1, chunk.tokens.len()))
            .count();
        let (group, tail) = pending
            .split_at_checked(count)
            .ok_or(TextError::InvalidInput("invalid group size"))?;
        submit(group)?;
        pending = tail;
    }
    Ok(())
}

impl NamespaceEmbedder for NamespaceModel {
    fn epoch(&self) -> EmbeddingEpoch {
        self.epoch.clone()
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        self.documents(&[text])?
            .pop()
            .ok_or_else(|| EmbedderFailure::new("missing document").into())
    }
    fn documents(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, EmbedderError> {
        self.batch_chunks(texts)
            .map_err(|e| EmbedderFailure::new(e.to_string()).into())
    }
    fn query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.bundle
            .tokenize_query(text)
            .map_err(TextError::from)
            .and_then(|tokens| self.evaluate(TowerRole::Query, tokens))
            .map_err(|e| EmbedderFailure::new(e.to_string()).into())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    fn singleton_reference(
        model: &NamespaceModel,
        text: &str,
    ) -> Result<Vec<EmbeddedChunk>, TextError> {
        let tower = &model.bundle.document_tower().embedding;
        let limit = tower.max_tokens as usize;
        let tokenizer = model.bundle.tokenizer(TowerRole::Document);
        let mut start = 0;
        let mut chunks = Vec::new();
        loop {
            let rest = text
                .get(start..)
                .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
            let mut end = rest
                .char_indices()
                .nth(limit.saturating_mul(8))
                .map_or(rest.len(), |(i, _)| i);
            let tokens = loop {
                let body = rest
                    .get(..end)
                    .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
                let input = format!("{}{body}", tower.prompt_prefix);
                // One extra slot detects truncation, including prompt and specials.
                let tokens = tokenizer.encode_batch(&[input], limit + 1)?;
                if tokens.token_ids.len() <= limit {
                    break tokens;
                }
                let midpoint = body.floor_char_boundary(end / 2);
                let next = body
                    .get(..midpoint)
                    .and_then(|s| s.rfind(char::is_whitespace).filter(|i| *i > 0))
                    .unwrap_or(midpoint);
                if next == 0 {
                    return Err(TextError::InvalidInput(
                        "source character and prefix exceed tower window",
                    ));
                }
                end = next;
            };
            chunks.push(EmbeddedChunk {
                bytes: start..start + end,
                vector: model.evaluate(TowerRole::Document, tokens)?,
            });
            start += end;
            if start == text.len() {
                break;
            }
        }
        Ok(chunks)
    }
    fn tiny() -> NamespaceModel {
        NamespaceModel::open_bundle(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ze115/gte.zem"
        )))
        .expect("tiny bundle")
    }
    #[test]
    fn ze597_short_upsert_batches_reach_runtime() {
        let root = tempfile::tempdir().expect("root");
        let model = Arc::new(tiny());
        let store =
            Store::open_embedded(root.path(), OpenOptions::new(), model.clone()).expect("store");
        for count in [32, 33] {
            let rows = (0..count)
                .map(|i| {
                    IngestDocument::new(
                        DocumentVersion::new(DocId::new(count * 100 + i), Revision::new(1)),
                        vec![],
                    )
                    .with_text(if i % 2 == 0 { "bronze" } else { "zeppelin" })
                })
                .collect();
            store.ingest(IngestBatch::new(rows)).expect("upsert");
            let shapes = std::mem::take(&mut *model.runtime.observer.lock().expect("observer"));
            let mut expected = vec![(32, vec![32])];
            if count == 33 {
                expected.push((1, vec![1]));
            }
            assert_eq!(shapes, expected);
            let hit = IngestDocument::new(
                DocumentVersion::new(DocId::new(count * 100), Revision::new(2)),
                vec![],
            )
            .with_text("bronze");
            store.ingest(IngestBatch::new(vec![hit])).expect("reuse");
            assert!(matches!(
                store.ingest(IngestBatch::new(vec![])),
                Err(volantis_db::ingest::IngestError::EmptyBatch)
            ));
            assert!(model.runtime.observer.lock().expect("observer").is_empty());
        }
    }

    #[test]
    fn ze597_ragged_chunks_match_singleton_reference() {
        let model = tiny();
        let long = "the bronze zeppelin ".repeat(200);
        let texts = ["bronze", &long, "🧪 bronze 雪 zeppelin", "", "zeppelin"];
        let results = model.documents(&texts).expect("batch");
        assert_eq!(results.len(), texts.len());
        assert!(results[1].len() > 32);
        let references = texts.map(|text| singleton_reference(&model, text).expect("reference"));
        assert_ne!(references[0][0].vector, references[4][0].vector);
        for (chunks, reference) in results.iter().zip(references) {
            assert_eq!(chunks.len(), reference.len());
            for (chunk, single) in chunks.iter().zip(reference) {
                assert_eq!(chunk.bytes, single.bytes);
                assert_eq!(chunk.vector.len(), single.vector.len());
                assert!(
                    chunk
                        .vector
                        .iter()
                        .zip(single.vector)
                        .all(|(a, b)| (a - b).abs() <= 0.00002)
                );
            }
        }
    }

    #[test]
    fn ze597_length_groups_obey_eval_budget() {
        let model = tiny();
        let tokenizer = model.bundle.tokenizer(TowerRole::Document);
        let mut rows = (0..32)
            .map(|source| PendingChunk {
                source,
                ordinal: 0,
                tokens: vec![
                    source as u32 + 1;
                    if source == 31 {
                        4097
                    } else if source % 2 == 0 {
                        4
                    } else {
                        256
                    }
                ],
            })
            .collect::<Vec<_>>();
        let mut shapes = Vec::new();
        let mut seen = Vec::new();
        submit_groups(&mut rows, |group| {
            let batch = tokenizer.pack_rows(group.iter().map(|chunk| chunk.tokens.as_slice()))?;
            shapes.push((batch.rows, batch.tokens_per_row));
            for (i, chunk) in group.iter().enumerate() {
                seen.push(chunk.source);
                for col in 0..batch.tokens_per_row {
                    let pos = i * batch.tokens_per_row + col;
                    let active = col < chunk.tokens.len();
                    assert_eq!(
                        batch.token_ids[pos],
                        if active { chunk.source as i32 + 1 } else { 0 }
                    );
                    assert_eq!(batch.attention_mask[pos], if active { 1. } else { 0. });
                }
            }
            Ok(())
        })
        .expect("groups");
        assert_eq!(shapes, [(16, 4), (15, 256), (1, 4097)]);
        seen.sort_unstable();
        assert_eq!(seen, (0..32).collect::<Vec<_>>());
        // A mixed-width group exercises the real bundle's padding and attention mask.
        let packed = tokenizer
            .pack_rows([&[3, 2][..], &[2][..]].into_iter())
            .expect("pad");
        assert_eq!(packed.token_ids, [3, 2, 2, 0]);
        assert_eq!(packed.attention_mask, [1., 1., 1., 0.]);
        let mut calls = 0;
        let error = submit_groups(&mut rows, |_| {
            calls += 1;
            if calls == 2 {
                return Err(TextError::Pipeline {
                    stage: "embed submit",
                    detail: "runtime is closed".into(),
                });
            }
            Ok(())
        })
        .expect_err("closed runtime");
        assert!(matches!(
            error,
            TextError::Pipeline {
                stage: "embed submit",
                ..
            }
        ));
        assert_eq!(calls, 2);
    }

    #[test]
    #[ignore = "requires cached release model"]
    fn ze597_cached_model_batch_parity() {
        let path = std::env::var_os("ZE597_BUNDLE").expect("ZE597_BUNDLE");
        let model = NamespaceModel::open_bundle(Path::new(&path)).expect("cached bundle");
        assert_eq!(model.epoch.document.normalization, Normalization::L2);
        let long = "the bronze zeppelin ".repeat(190);
        let texts = ["FAKE bronze", "🧪 bronze 雪 zeppelin", &long, "zeppelin"];
        let chunks = model.documents(&texts).expect("batch");
        assert_eq!(chunks.len(), texts.len());
        let mut minimum = 1_f64;
        for (parent, text) in chunks.iter().zip(texts) {
            let reference = singleton_reference(&model, text).expect("reference");
            assert_eq!(parent.len(), reference.len());
            for (chunk, single) in parent.iter().zip(reference) {
                assert_eq!(chunk.bytes, single.bytes);
                assert_eq!(chunk.vector.len(), single.vector.len());
                let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                let a = norm(&chunk.vector);
                let b = norm(&single.vector);
                assert!((a - 1.).abs() < 1e-5 && (b - 1.).abs() < 1e-5);
                let cosine = chunk
                    .vector
                    .iter()
                    .zip(single.vector)
                    .map(|(x, y)| f64::from(*x) * f64::from(y))
                    .sum::<f64>()
                    / (a * b);
                assert!(cosine.is_finite());
                minimum = minimum.min(cosine);
            }
        }
        println!("minimum cosine {minimum:.10}");
        assert!(minimum >= 0.99999);
    }
}
