//! Bundle-backed towers for an ordinary Store.
use super::*;
use zeppelin_embed::epoch::{EmbedderError, EmbedderFailure, EmbeddingEpoch};
use zeppelin_embed::lifecycle::embedded::{EmbeddedChunk, NamespaceEmbedder};

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

    fn chunks(&self, text: &str) -> Result<Vec<EmbeddedChunk>, TextError> {
        let tower = &self.bundle.document_tower().embedding;
        let limit = tower.max_tokens as usize;
        let tokenizer = self.bundle.tokenizer(TowerRole::Document);
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
                vector: self.evaluate(TowerRole::Document, tokens)?,
            });
            start += end;
            if start == text.len() {
                break;
            }
        }
        Ok(chunks)
    }
}

impl NamespaceEmbedder for NamespaceModel {
    fn epoch(&self) -> EmbeddingEpoch {
        self.epoch.clone()
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        self.chunks(text)
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
