//! Text embedding in a normal Store namespace.
use super::*;
use crate::epoch::{EmbedderError, EmbeddingEpoch, StoreEpoch};
use crate::ingest::{DocumentVersion, IngestDocument};
use crate::meta::{ColumnId, PredicateValue};
use std::ops::Range;

/// The two model towers registered for this namespace. Document byte offsets
/// index the original UTF-8 input, before tokenization or normalization.
pub trait NamespaceEmbedder: Send + Sync {
    /// Complete interpretation of both towers, including model/artifact pins.
    fn epoch(&self) -> EmbeddingEpoch;
    /// Chunk and embed the complete source revision.
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError>;
    /// Embed a query using the query tower.
    fn query(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;
}

/// One source range and its document-tower output.
pub struct EmbeddedChunk {
    /// Half-open UTF-8 byte offsets.
    pub bytes: Range<usize>,
    /// Full precision model output.
    pub vector: Vec<f32>,
}

/// Retrieval leg for chunk-level text search.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedLeg {
    /// Nearest chunks by squared L2.
    Dense,
    /// BM25 over chunk text.
    Lexical,
    /// Reciprocal-rank fusion of dense and lexical results.
    Hybrid,
}

/// One matching chunk of a live source revision.
#[derive(Debug)]
pub struct EmbeddedHit {
    /// Parent identity.
    pub version: DocumentVersion,
    /// Zero-based chunk ordinal.
    pub chunk: usize,
    /// Half-open UTF-8 source offsets.
    pub bytes: Range<usize>,
    /// Exact source substring.
    pub text: String,
    /// Parent timestamp.
    pub timestamp: i64,
    /// Larger scores rank first.
    pub score: f64,
    /// Typed attributes of this revision.
    pub attributes: Vec<(ColumnId, PredicateValue)>,
}

pub(crate) const EPOCH_TAG: &[u8] = b"ZEEMBED1";
const ROW_TAG: &[u8] = b"ZECHUNK1";

fn invalid(detail: impl Into<String>) -> StoreError {
    StoreError::Embedding(crate::epoch::EmbedderFailure::new(detail).into())
}

fn word(input: &mut &[u8]) -> Result<usize, StoreError> {
    let (word, rest) = input
        .split_at_checked(4)
        .ok_or_else(|| invalid("short chunk header"))?;
    *input = rest;
    Ok(u32::from_le_bytes(
        word.try_into()
            .map_err(|_| invalid("invalid chunk header"))?,
    ) as usize)
}

fn put_word(output: &mut Vec<u8>, n: usize) -> Result<(), StoreError> {
    output.extend_from_slice(
        &u32::try_from(n)
            .map_err(|_| invalid("embedded row exceeds u32"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn unpack(input: &[u8]) -> Result<(&[u8], &[u8]), StoreError> {
    let mut data = input
        .strip_prefix(ROW_TAG)
        .ok_or_else(|| invalid("missing embedded row header"))?;
    let n = word(&mut data)?;
    data.split_at_checked(n)
        .ok_or_else(|| invalid("short embedded metadata"))
}

fn read_chunk<'a>(
    data: &mut &'a [u8],
    source: &str,
    dims: usize,
) -> Result<(Range<usize>, &'a [u8]), StoreError> {
    let range = word(data)?..word(data)?;
    if source.get(range.clone()).is_none() {
        return Err(invalid("invalid embedded source span"));
    }
    let (vector, rest) = data
        .split_at_checked(
            dims.checked_mul(4)
                .ok_or_else(|| invalid("embedding dimensions overflow"))?,
        )
        .ok_or_else(|| invalid("truncated chunk vector"))?;
    *data = rest;
    Ok((range, vector))
}

pub(crate) fn verify_row(
    source: Option<&str>,
    metadata: Option<&[u8]>,
    dims: usize,
) -> Result<(), StoreError> {
    let source = source.ok_or_else(|| invalid("missing embedded source"))?;
    let (_, mut data) = unpack(metadata.unwrap_or_default())?;
    if data.is_empty() || dims == 0 {
        return Err(invalid("missing embedded chunks or dimensions"));
    }
    while !data.is_empty() {
        let (_, vector) = read_chunk(&mut data, source, dims)?;
        for bytes in vector.chunks_exact(4) {
            let value =
                f32::from_le_bytes(bytes.try_into().map_err(|_| invalid("invalid vector"))?);
            if !value.is_finite() {
                return Err(invalid("non-finite chunk vector"));
            }
        }
    }
    Ok(())
}

fn check_vector(vector: &[f32], dims: usize) -> Result<(), StoreError> {
    if dims == 0 || vector.len() != dims || vector.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid embedding dimensions or coordinates"));
    }
    Ok(())
}

#[derive(Clone)]
pub(super) struct RegisteredModel(pub(super) Arc<dyn NamespaceEmbedder>);
impl std::fmt::Debug for RegisteredModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RegisteredModel")
    }
}
impl PartialEq for RegisteredModel {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for RegisteredModel {}

impl OpenOptions {
    /// Declares the namespace model for open or an atomic namespace mutation.
    pub fn with_embedder(mut self, model: Arc<dyn NamespaceEmbedder>) -> Result<Self, StoreError> {
        let mut embedding = model.epoch();
        let mut identity = EPOCH_TAG.to_vec();
        identity.extend_from_slice(&embedding.alignment_digest);
        embedding.alignment_digest = identity;
        if embedding.document.dims == 0 || embedding.query.dims != embedding.document.dims {
            return Err(invalid("incompatible embedding towers"));
        }
        let tokenizer = self
            .tokenizer
            .clone()
            .unwrap_or_else(crate::fts::tokenizer::TokenizerConfig::text_default);
        let epoch = StoreEpoch {
            embedding,
            tokenizer: tokenizer.epoch(),
        };
        if self.epoch.as_ref().is_some_and(|e| e != &epoch) {
            return Err(invalid("conflicting embedded namespace epoch"));
        }
        self.epoch = Some(epoch);
        self.embedder = Some(RegisteredModel(model));
        Ok(self)
    }
    /// Declared epoch, including the embedded namespace model pins.
    pub fn declared_epoch(&self) -> Option<&StoreEpoch> {
        self.epoch.as_ref()
    }
}
impl Store {
    /// Opens a namespace with its two-tower model registered.
    pub fn open_embedded(
        path: impl AsRef<Path>,
        options: OpenOptions,
        model: Arc<dyn NamespaceEmbedder>,
    ) -> Result<Self, StoreError> {
        Self::open(path, options.with_embedder(model)?)
    }
    /// Whether this namespace stores engine-generated chunk vectors.
    pub fn is_embedded(&self) -> bool {
        self.epoch
            .as_ref()
            .is_some_and(|e| e.embedding.alignment_digest.starts_with(EPOCH_TAG))
    }

    /// Evaluates text using the registered query tower.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, StoreError> {
        let model = self.embedder.as_ref().ok_or(StoreError::MissingEmbedder)?;
        let vector = model.query(text).map_err(StoreError::Embedding)?;
        check_vector(&vector, self.embedded_dims()?)?;
        Ok(vector)
    }

    fn embedded_dims(&self) -> Result<usize, StoreError> {
        self.epoch
            .as_ref()
            .map(|e| e.embedding.document.dims as usize)
            .ok_or(StoreError::MissingEmbedder)
    }

    pub(crate) fn prepare_embedded(
        &self,
        documents: &mut [IngestDocument],
    ) -> Result<(), StoreError> {
        if !self.is_embedded() || documents.is_empty() {
            return Ok(());
        }
        let model = self.embedder.as_ref().ok_or(StoreError::MissingEmbedder)?;
        for doc in documents {
            if !doc.vector().is_empty() {
                return Err(invalid(
                    "embedded upsert requires text without a caller vector",
                ));
            }
            let text = doc
                .text()
                .ok_or_else(|| invalid("embedded upsert requires text"))?;
            let chunks = model.document(text).map_err(StoreError::Embedding)?;
            let first = chunks
                .first()
                .ok_or_else(|| invalid("document encoder returned no chunks"))?;
            let mut metadata = ROW_TAG.to_vec();
            put_word(&mut metadata, doc.metadata().len())?;
            metadata.extend_from_slice(doc.metadata());
            for chunk in &chunks {
                if text.get(chunk.bytes.clone()).is_none() {
                    return Err(invalid("invalid chunk source range"));
                }
                check_vector(&chunk.vector, self.embedded_dims()?)?;
                put_word(&mut metadata, chunk.bytes.start)?;
                put_word(&mut metadata, chunk.bytes.end)?;
                for v in &chunk.vector {
                    metadata.extend_from_slice(&v.to_le_bytes());
                }
            }
            doc.set_embedded(first.vector.clone(), metadata);
        }
        Ok(())
    }

    /// Ranks the chunks of eligible documents. A concurrent mutation between
    /// scan pages fails the request with ScanStale, never mixing revisions.
    pub fn query_embedded(
        &self,
        text: &str,
        leg: EmbeddedLeg,
        k: usize,
        filter: Option<&QueryFilter>,
        control: QueryControl,
    ) -> Result<Vec<EmbeddedHit>, QueryError> {
        self.query_embedded_at(text, leg, k, filter, control)
            .map(|(_, hits)| hits)
    }
    /// Returns the exact parent generation with the ranked chunks.
    pub fn query_embedded_at(
        &self,
        text: &str,
        leg: EmbeddedLeg,
        k: usize,
        filter: Option<&QueryFilter>,
        control: QueryControl,
    ) -> Result<(u64, Vec<EmbeddedHit>), QueryError> {
        use crate::fts::index::{DEFAULT_FIELD, Document, LexicalIndex, SegmentIndex};
        use crate::fts::search::{TermQuery, search};
        let error = |detail: &str| QueryError::Store(invalid(detail));
        if !(1..=4096).contains(&k) {
            return Err(error("embedded query k must be in 1..=4096"));
        }
        control.checkpoint()?;
        let dims = self.embedded_dims().map_err(QueryError::Store)?;
        if !self.is_embedded() {
            return Err(QueryError::Store(StoreError::MissingEmbedder));
        }
        let query = (leg != EmbeddedLeg::Lexical)
            .then(|| self.embed_query(text))
            .transpose()
            .map_err(QueryError::Store)?;
        let mut index = SegmentIndex::new();
        let mut rows = Vec::new();
        let mut dense = Vec::new();
        let mut cursor = None;
        let generation = loop {
            let fields = DocumentFields::TEXT
                | DocumentFields::ATTRIBUTES
                | DocumentFields::METADATA
                | DocumentFields::CHUNKS;
            let mut request = DocumentScanRequest::new(256, fields, control.clone());
            if let Some(p) = filter.and_then(|f| f.predicate.as_ref()) {
                request = request.with_predicate(p);
            }
            if let Some(c) = cursor {
                request = request.with_cursor(c);
            }
            let page = self.scan_documents(request)?;
            for doc in page.documents {
                if filter
                    .and_then(|f| f.eligible_ids.as_ref())
                    .is_some_and(|ids| ids.numeric.binary_search(&doc.doc_id).is_err())
                {
                    continue;
                }
                let source = doc
                    .text
                    .as_deref()
                    .ok_or_else(|| error("missing embedded source"))?;
                let (_, mut data) = unpack(doc.metadata.as_deref().unwrap_or_default())
                    .map_err(QueryError::Store)?;
                if data.is_empty() {
                    return Err(error("missing embedded chunks"));
                }
                let mut chunk = 0;
                while !data.is_empty() {
                    control.checkpoint()?;
                    let (bytes, vector) =
                        read_chunk(&mut data, source, dims).map_err(QueryError::Store)?;
                    let chunk_text = source
                        .get(bytes.clone())
                        .ok_or_else(|| error("invalid embedded source span"))?;
                    if let Some(query) = &query {
                        let mut distance = 0.0;
                        for (bytes, q) in vector.chunks_exact(4).zip(query) {
                            let v = f32::from_le_bytes(
                                bytes.try_into().map_err(|_| error("invalid vector"))?,
                            );
                            distance += (f64::from(v) - f64::from(*q)).powi(2);
                        }
                        dense.push((rows.len(), -distance));
                    }
                    if leg != EmbeddedLeg::Dense {
                        index
                            .push_document(&self.tokenizer, &Document::with_text(chunk_text))
                            .map_err(|e| error(&e.to_string()))?;
                    }
                    rows.push(EmbeddedHit {
                        version: DocumentVersion::new(doc.doc_id, doc.revision),
                        chunk,
                        timestamp: doc.timestamp,
                        bytes,
                        text: chunk_text.to_owned(),
                        attributes: doc.attributes.clone().unwrap_or_default(),
                        score: 0.,
                    });
                    chunk += 1;
                }
            }
            cursor = page.continuation;
            if cursor.is_none() {
                break page.generation;
            }
        };
        dense.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut lexical = Vec::new();
        if leg != EmbeddedLeg::Dense && index.total_tokens() > 0 {
            let terms = self
                .tokenizer
                .analyze(text)
                .into_iter()
                .map(|t| t.term.into_bytes())
                .collect();
            let mut corpus = LexicalIndex::new();
            corpus
                .push_segment(index)
                .map_err(|e| error(&e.to_string()))?;
            lexical = search(
                &corpus,
                &TermQuery::flat(terms, &[DEFAULT_FIELD]),
                rows.len(),
                Default::default(),
            )
            .map_err(|e| error(&e.to_string()))?
            .hits
            .into_iter()
            .map(|h| (h.doc.row as usize, h.score))
            .collect();
        }
        let ranked = match leg {
            EmbeddedLeg::Dense => dense,
            EmbeddedLeg::Lexical => lexical,
            EmbeddedLeg::Hybrid => {
                let mut scores = vec![0.; rows.len()];
                for hits in [&dense, &lexical] {
                    for (rank, (row, _)) in hits.iter().enumerate() {
                        *scores
                            .get_mut(*row)
                            .ok_or_else(|| error("invalid chunk rank"))? +=
                            1. / (61. + rank as f64);
                    }
                }
                let mut ranked: Vec<_> = scores.into_iter().enumerate().collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                ranked
            }
        };
        control.checkpoint()?;
        let mut rows: Vec<_> = rows.into_iter().map(Some).collect();
        let hits = ranked
            .into_iter()
            .take(k)
            .map(|(i, score)| {
                let mut hit = rows
                    .get_mut(i)
                    .and_then(Option::take)
                    .ok_or_else(|| error("invalid chunk rank"))?;
                hit.score = score;
                Ok(hit)
            })
            .collect::<Result<_, _>>()?;
        Ok((generation, hits))
    }
}

impl PublishedSnapshot {
    pub(super) fn expose_metadata(
        &self,
        doc: &mut StoredDocument,
        fields: DocumentFields,
    ) -> Result<(), StoreError> {
        if self.embedded_epoch.is_some()
            && fields.contains(DocumentFields::METADATA)
            && !fields.contains(DocumentFields::CHUNKS)
        {
            let (user, _) = unpack(doc.metadata.as_deref().unwrap_or_default())?;
            doc.metadata = (!user.is_empty()).then(|| user.to_vec());
        }
        Ok(())
    }
}
