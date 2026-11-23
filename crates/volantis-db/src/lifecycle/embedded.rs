//! Text embedding in a normal Store namespace.
mod neighbours;
mod worker;
pub use neighbours::{ChunkNeighbour, NeighbourOptions};
pub(crate) use worker::EmbeddingWorker;
pub use worker::{EmbeddingErrorCode, EmbeddingStatus};

use super::*;
use crate::epoch::{EmbedderError, EmbeddingEpoch, StoreEpoch};
use crate::ingest::{DocumentVersion, IngestDocument};
use crate::meta::{ColumnId, PredicateValue};
use std::ops::Range;

/// The two model towers registered for this namespace. Document byte offsets
/// index the original UTF-8 input, before tokenization or normalization.
pub trait NamespaceEmbedder: Send + Sync {
    /// Releases an owner and reports final runtime cleanup errors.
    fn release(self: Arc<Self>) -> Result<(), EmbedderError> {
        Ok(())
    }
    /// Complete interpretation of both towers, including model/artifact pins.
    fn epoch(&self) -> EmbeddingEpoch;
    /// Chunk and embed the complete source revision.
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError>;
    /// Embed pending sources in order, returning one ordered chunk list per source.
    fn documents(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, EmbedderError> {
        texts.iter().map(|text| self.document(text)).collect()
    }
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

/// When an embedded namespace runs the document tower for a new source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DocumentSchedule {
    /// Embed inside the upsert, before it commits (the default).
    #[default]
    Immediate,
    /// Commit the source as pending; a background worker embeds it later.
    Background,
    /// As `Background`, but the worker starts paused.
    Paused,
}

/// Embedding state of one stored source revision. The values are persisted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum EmbeddedRowState {
    /// Chunks and the representative vector are stored.
    Ready = 0,
    /// The source is committed; it has no chunks and no vector yet.
    Pending = 1,
    /// The source is missing or zero bytes: complete with zero chunks and no
    /// vector. The worker never embeds it and it never counts as pending.
    Empty = 2,
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
    /// `Pending` marks a lexical match on the whole source of a row that is
    /// not embedded yet; its `chunk` is zero.
    pub state: EmbeddedRowState,
}

/// Ranked chunks with the generation they were read at and the number of
/// eligible rows that wait for embedding.
#[derive(Debug)]
pub struct EmbeddedReport {
    /// Store generation of every row read.
    pub generation: u64,
    /// Eligible pending rows after predicates and eligible ids, before top-k.
    pub pending_rows: u64,
    /// Ranked hits.
    pub hits: Vec<EmbeddedHit>,
}

pub(crate) const EPOCH_TAG: &[u8] = b"ZEEMBED1";
/// Ready row: `ROW_TAG, caller_len:u32, caller, (start:u32, end:u32, f32[dims])+`.
const ROW_TAG: &[u8] = b"ZECHUNK1";
/// Row that is not ready (pending or empty): `STATE_TAG, state:u32,
/// token:[u8; 16], caller_len:u32, caller`. The token identifies one
/// admission, so a purge and reinsert of the same id, revision and text gets a
/// different token.
const STATE_TAG: &[u8] = b"ZESTATE1";
const TOKEN_LEN: usize = 16;

/// One decoded embedded row envelope.
pub(crate) struct EmbeddedRow<'a> {
    pub(crate) state: EmbeddedRowState,
    pub(crate) caller: &'a [u8],
    /// Packed chunks of a ready row; empty otherwise.
    pub(crate) chunks: &'a [u8],
}

pub(crate) fn decode_row(input: &[u8]) -> Result<EmbeddedRow<'_>, StoreError> {
    if input.starts_with(ROW_TAG) {
        let (caller, chunks) = unpack(input)?;
        return Ok(EmbeddedRow {
            state: EmbeddedRowState::Ready,
            caller,
            chunks,
        });
    }
    let mut data = input
        .strip_prefix(STATE_TAG)
        .ok_or_else(|| invalid("missing embedded row header"))?;
    let state = match word(&mut data)? {
        1 => EmbeddedRowState::Pending,
        2 => EmbeddedRowState::Empty,
        // Ready rows use ROW_TAG.
        other => return Err(invalid(format!("unsupported embedded row state {other}"))),
    };
    let (_, rest) = data
        .split_at_checked(TOKEN_LEN)
        .ok_or_else(|| invalid("short embedded admission token"))?;
    data = rest;
    let n = word(&mut data)?;
    if data.len() != n {
        return Err(invalid("embedded row metadata length mismatch"));
    }
    Ok(EmbeddedRow {
        state,
        caller: data,
        chunks: &[],
    })
}

/// Packs a pending or empty row.
fn state_row(state: EmbeddedRowState, caller: &[u8]) -> Result<Vec<u8>, StoreError> {
    let mut packed = Vec::new();
    let size = caller
        .len()
        .checked_add(STATE_TAG.len() + 4 + TOKEN_LEN + 4)
        .ok_or_else(|| invalid("embedded row overflow"))?;
    packed
        .try_reserve_exact(size)
        .map_err(|_| invalid("embedded row allocation failed"))?;
    packed.extend_from_slice(STATE_TAG);
    put_word(&mut packed, state as usize)?;
    packed.extend_from_slice(&admission_token());
    put_word(&mut packed, caller.len())?;
    packed.extend_from_slice(caller);
    Ok(packed)
}

/// A token that differs for every admission in every process.
fn admission_token() -> [u8; TOKEN_LEN] {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static KEYS: std::sync::OnceLock<std::collections::hash_map::RandomState> =
        std::sync::OnceLock::new();
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let keys = KEYS.get_or_init(std::collections::hash_map::RandomState::new);
    let mut hasher = keys.build_hasher();
    hasher.write_u64(count);
    hasher.write_u128(nanos);
    let mut token = [0; TOKEN_LEN];
    let (time, mixed) = token.split_at_mut(8);
    time.copy_from_slice(&(nanos as u64).to_le_bytes());
    mixed.copy_from_slice(&(hasher.finish() ^ count).to_le_bytes());
    token
}

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

/// Whether an embedded upsert has no source to embed: text is omitted or
/// zero bytes. Any other text, including one space, is real input.
fn is_empty_source(text: Option<&str>) -> bool {
    text.is_none_or(str::is_empty)
}

/// Checks one stored embedded row. A ready row holds chunks and a vector; a
/// pending row holds neither; an empty row holds neither and has no text.
pub(crate) fn verify_row(
    source: Option<&str>,
    metadata: Option<&[u8]>,
    has_vector: bool,
    dims: usize,
) -> Result<(), StoreError> {
    let row = decode_row(metadata.unwrap_or_default())?;
    if row.state == EmbeddedRowState::Empty {
        if !is_empty_source(source) {
            return Err(invalid("empty embedded row holds text"));
        }
        if has_vector {
            return Err(invalid("empty embedded row holds a vector"));
        }
        return Ok(());
    }
    let source = source.ok_or_else(|| invalid("missing embedded source"))?;
    if row.state != EmbeddedRowState::Ready {
        if has_vector {
            return Err(invalid("pending embedded row holds a vector"));
        }
        return Ok(());
    }
    if !has_vector {
        return Err(invalid("ready embedded row has no vector"));
    }
    let mut data = row.chunks;
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

fn repack(
    text: &str,
    stored: &[u8],
    caller: &[u8],
    dims: usize,
) -> Result<(Vec<f32>, Vec<u8>), StoreError> {
    verify_row(Some(text), Some(stored), true, dims)?;
    let (_, suffix) = unpack(stored)?;
    let mut remaining = suffix;
    let (_, bytes) = read_chunk(&mut remaining, text, dims)?;
    let mut first = Vec::new();
    first
        .try_reserve_exact(dims)
        .map_err(|_| invalid("embedded vector allocation failed"))?;
    for word in bytes.chunks_exact(4) {
        first.push(f32::from_le_bytes(
            word.try_into().map_err(|_| invalid("short coordinate"))?,
        ));
    }
    let mut packed = Vec::new();
    let size = caller
        .len()
        .checked_add(suffix.len())
        .and_then(|n| n.checked_add(ROW_TAG.len() + 4))
        .ok_or_else(|| invalid("embedded row overflow"))?;
    packed
        .try_reserve_exact(size)
        .map_err(|_| invalid("embedded row allocation failed"))?;
    packed.extend_from_slice(ROW_TAG);
    put_word(&mut packed, caller.len())?;
    packed.extend_from_slice(caller);
    packed.extend_from_slice(suffix);
    Ok((first, packed))
}

// Private to one prepared write. The flag identifies only this admission's
// pre-publication refusal even through native writer error conversions.
#[derive(Clone, Debug)]
pub(crate) struct ReuseProof {
    owner: std::sync::Weak<stats::Accounting>,
    generation: u64,
    epoch: crate::epoch::EpochIdentity,
    invalidated: Arc<std::sync::atomic::AtomicBool>,
}
impl PartialEq for ReuseProof {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.invalidated, &other.invalidated)
    }
}
impl ReuseProof {
    fn new(store: &Store, generation: u64, epoch: crate::epoch::EpochIdentity) -> Self {
        Self {
            generation,
            epoch,
            invalidated: Arc::default(),
            owner: Arc::downgrade(&store.accounting),
        }
    }
    pub(crate) fn invalidated(&self) -> bool {
        self.invalidated.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(crate) fn validate(
        &self,
        store: &Store,
        generation: u64,
        alias: Option<crate::epoch::EpochIdentity>,
    ) -> Result<(), StoreError> {
        super::validate_epoch_identity(alias, Some(self.epoch), true, AccessMode::ReadOnly)?;
        if generation != self.generation || !self.owner.ptr_eq(&Arc::downgrade(&store.accounting)) {
            self.invalidated
                .store(true, std::sync::atomic::Ordering::Relaxed);
            return Err(StoreError::ScanStale {
                cursor_generation: self.generation,
                current_generation: generation,
            });
        }
        Ok(())
    }
}

/// Validates document-tower output for one source and packs the ready row:
/// the first chunk's vector and the ZECHUNK1 envelope.
pub(crate) fn ready_row(
    text: &str,
    caller: &[u8],
    chunks: &[EmbeddedChunk],
    dims: usize,
) -> Result<(Vec<f32>, Vec<u8>), StoreError> {
    let first = chunks
        .first()
        .ok_or_else(|| invalid("document encoder returned no chunks"))?;
    let mut metadata = ROW_TAG.to_vec();
    put_word(&mut metadata, caller.len())?;
    metadata.extend_from_slice(caller);
    for chunk in chunks {
        if text.get(chunk.bytes.clone()).is_none() {
            return Err(invalid("invalid chunk source range"));
        }
        check_vector(&chunk.vector, dims)?;
        put_word(&mut metadata, chunk.bytes.start)?;
        put_word(&mut metadata, chunk.bytes.end)?;
        for v in &chunk.vector {
            metadata.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok((first.vector.clone(), metadata))
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
    /// Selects when new sources run the document tower. It is a property of
    /// this handle, not of the persisted epoch.
    #[must_use]
    pub const fn with_document_schedule(mut self, schedule: DocumentSchedule) -> Self {
        self.document_schedule = schedule;
        self
    }
    /// Declared document schedule.
    #[must_use]
    pub const fn document_schedule(&self) -> DocumentSchedule {
        self.document_schedule
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

    fn admit_embedder(&self) -> Result<Arc<dyn NamespaceEmbedder>, StoreError> {
        let state = self
            .state
            .lock()
            .map_err(|_| StoreError::Synchronization { component: "state" })?;
        match *state {
            StoreState::Closed => Err(StoreError::Closed),
            StoreState::Closing => Err(StoreError::Closing),
            StoreState::Open => self
                .embedder
                .lock()
                .map_err(|_| StoreError::Synchronization {
                    component: "namespace embedder",
                })?
                .clone()
                .ok_or(StoreError::MissingEmbedder),
        }
    }

    /// Evaluates text using the registered query tower.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, StoreError> {
        let model = self.admit_embedder()?;
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

    pub(crate) fn validate_embedded_epoch(
        &self,
        alias: Option<crate::epoch::EpochIdentity>,
    ) -> Result<(), StoreError> {
        if self.is_embedded() {
            super::validate_epoch_identity(
                alias,
                self.epoch.as_ref().map(StoreEpoch::identity),
                true,
                AccessMode::ReadOnly,
            )?;
        }
        Ok(())
    }

    pub(crate) fn prepare_embedded(
        &self,
        documents: &mut [IngestDocument],
        declaration: Option<crate::epoch::EpochIdentity>,
        allow_reuse: bool,
    ) -> Result<Option<ReuseProof>, StoreError> {
        #[cfg_attr(not(feature = "graph-cypher"), allow(unused_mut))]
        let mut defer = self.embedding.defers();
        // A document linked to a graph node embeds at write time: completion
        // never rewrites graph-linked rows.
        #[cfg(feature = "graph-cypher")]
        if defer && self.native_graph.is_installed()? {
            let ids = documents
                .iter()
                .map(|document| document.version().doc_id())
                .collect::<Vec<_>>();
            defer = !self
                .documents_have_native_nodes(&ids)
                .map_err(|error| StoreError::DocumentMutation(Box::new(error)))?;
        }
        self.prepare_embedded_with(documents, declaration, allow_reuse, defer)
    }

    /// As `prepare_embedded`, but always runs the document tower now.
    #[cfg(feature = "graph-cypher")]
    pub(crate) fn prepare_embedded_now(
        &self,
        documents: &mut [IngestDocument],
        declaration: Option<crate::epoch::EpochIdentity>,
        allow_reuse: bool,
    ) -> Result<Option<ReuseProof>, StoreError> {
        self.prepare_embedded_with(documents, declaration, allow_reuse, false)
    }

    fn prepare_embedded_with(
        &self,
        documents: &mut [IngestDocument],
        declaration: Option<crate::epoch::EpochIdentity>,
        allow_reuse: bool,
        defer: bool,
    ) -> Result<Option<ReuseProof>, StoreError> {
        if !self.is_embedded() || documents.is_empty() {
            return Ok(None);
        }
        let epoch = self
            .epoch
            .as_ref()
            .ok_or(StoreError::MissingEmbedder)?
            .identity();
        for doc in documents.iter() {
            if !doc.vector().is_empty() || doc.absent_dims().is_some() {
                return Err(invalid(
                    "embedded upsert requires text without a caller vector",
                ));
            }
        }
        enum Source {
            Encode,
            Stored,
            Earlier(usize),
            Empty,
        }
        let mut sources = Vec::new();
        let mut generation = None;
        if allow_reuse && declaration.is_none_or(|value| value == epoch) {
            let admitted = self.admit_document_read()?;
            if admitted.snapshot.epoch_alias() == Some(epoch) {
                sources
                    .try_reserve_exact(documents.len())
                    .map_err(|_| invalid("embedded batch allocation failed"))?;
                let mut latest = std::collections::HashMap::new();
                latest
                    .try_reserve(documents.len())
                    .map_err(|_| invalid("embedded revision allocation failed"))?;
                let fields =
                    DocumentFields::TEXT | DocumentFields::METADATA | DocumentFields::CHUNKS;
                for slice in documents.chunks_mut(64) {
                    let mut ids = Vec::new();
                    ids.try_reserve_exact(slice.len())
                        .map_err(|_| invalid("embedded ID allocation failed"))?;
                    ids.extend(slice.iter().map(|doc| doc.version().doc_id()));
                    for (doc, row) in slice
                        .iter_mut()
                        .zip(Self::read_document_ids(&admitted, &ids, fields)?)
                    {
                        let version = doc.version();
                        let prior = latest.get(&version.doc_id()).copied();
                        let revision = prior
                            .map(|(revision, _)| revision)
                            .or_else(|| row.as_ref().map(|r| r.revision));
                        // Only a ready predecessor has chunks to reuse; a
                        // pending one gives a new pending admission. An empty
                        // source needs no chunks, but still advances the
                        // latest revision of its id in this batch.
                        let source = if is_empty_source(doc.text()) {
                            Source::Empty
                        } else if let Some((_, index)) = prior {
                            Source::Earlier(index)
                        } else if let Some(row) = row
                            && row.text.as_deref() == doc.text()
                            && decode_row(row.metadata.as_deref().unwrap_or_default())?.state
                                == EmbeddedRowState::Ready
                        {
                            let (vector, metadata) = repack(
                                doc.text()
                                    .ok_or_else(|| invalid("missing embedded source"))?,
                                row.metadata.as_deref().unwrap_or_default(),
                                doc.metadata(),
                                self.embedded_dims()?,
                            )?;
                            doc.set_embedded(vector, metadata);
                            Source::Stored
                        } else {
                            Source::Encode
                        };
                        if revision.is_none_or(|r| version.revision() > r) {
                            latest.insert(version.doc_id(), (version.revision(), sources.len()));
                        }
                        sources.push(source);
                    }
                }
                generation = Some(admitted.generation);
            }
        }
        let mut pending = Vec::new();
        let mut texts = Vec::new();
        for (index, doc) in documents.iter().enumerate() {
            // Classified by text, so it also holds when reuse is off.
            if is_empty_source(doc.text()) {
                continue;
            }
            match sources.get(index) {
                Some(Source::Stored) => continue,
                Some(Source::Earlier(previous)) => {
                    let previous = documents
                        .get(*previous)
                        .ok_or(StoreError::ActiveRowOverflow)?;
                    // Reuse a ready earlier row of this batch. A deferred or
                    // empty earlier row has no chunks to copy.
                    if previous.text() == doc.text() && !defer {
                        continue;
                    }
                    *sources
                        .get_mut(index)
                        .ok_or(StoreError::ActiveRowOverflow)? = Source::Encode;
                }
                _ => {}
            }
            pending.push(index);
            texts.push(
                doc.text()
                    .ok_or_else(|| invalid("missing embedded source"))?,
            );
        }
        let dims = self.embedded_dims()?;
        let absent_dims =
            u32::try_from(dims).map_err(|_| invalid("embedding dimensions exceed u32"))?;
        // An empty source is complete now, with no chunks, no vector and no
        // model call, on every schedule.
        let complete_empty = |doc: &mut IngestDocument| -> Result<bool, StoreError> {
            if !is_empty_source(doc.text()) {
                return Ok(false);
            }
            let packed = state_row(EmbeddedRowState::Empty, doc.metadata())?;
            doc.set_vector_absent(absent_dims, packed);
            Ok(true)
        };
        if defer {
            for doc in documents.iter_mut() {
                complete_empty(doc)?;
            }
            // Commit the source now; the background worker embeds it later.
            for index in pending {
                let doc = documents
                    .get_mut(index)
                    .ok_or(StoreError::ActiveRowOverflow)?;
                let packed = state_row(EmbeddedRowState::Pending, doc.metadata())?;
                doc.set_vector_absent(absent_dims, packed);
            }
            return Ok(generation
                .filter(|_| {
                    sources
                        .iter()
                        .any(|source| matches!(source, Source::Stored))
                })
                .map(|g| ReuseProof::new(self, g, epoch)));
        }
        let encoded = if texts.is_empty() {
            Vec::new()
        } else {
            self.admit_embedder()?
                .documents(&texts)
                .map_err(StoreError::Embedding)?
        };
        if encoded.len() != pending.len() {
            return Err(invalid("document encoder returned wrong document count"));
        }
        let mut encoded = pending.into_iter().zip(encoded);
        let mut reused = false;
        for index in 0..documents.len() {
            let (prior, remaining) = documents.split_at_mut(index);
            let doc = remaining.first_mut().ok_or(StoreError::ActiveRowOverflow)?;
            if complete_empty(doc)? {
                continue;
            }
            let text = doc
                .text()
                .ok_or_else(|| invalid("missing embedded source"))?;
            match sources.get(index) {
                Some(Source::Stored) => {
                    reused = true;
                    continue;
                }
                Some(Source::Earlier(index)) => {
                    let previous = prior.get(*index).ok_or(StoreError::ActiveRowOverflow)?;
                    let (vector, packed) = repack(text, previous.metadata(), doc.metadata(), dims)?;
                    doc.set_embedded(vector, packed);
                    reused = true;
                    continue;
                }
                _ => {}
            }
            let (_, chunks) = encoded
                .next()
                .filter(|(position, _)| *position == index)
                .ok_or_else(|| invalid("missing encoded document"))?;
            let (vector, metadata) = ready_row(text, doc.metadata(), &chunks, dims)?;
            doc.set_embedded(vector, metadata);
        }
        Ok(generation
            .filter(|_| reused)
            .map(|g| ReuseProof::new(self, g, epoch)))
    }
    pub(crate) fn restore_embedded_inputs(
        &self,
        docs: &mut [IngestDocument],
    ) -> Result<(), StoreError> {
        for doc in docs {
            let caller = decode_row(doc.metadata())?.caller;
            let mut metadata = Vec::new();
            metadata
                .try_reserve_exact(caller.len())
                .map_err(|_| invalid("caller metadata allocation failed"))?;
            metadata.extend_from_slice(caller);
            doc.set_embedded(Vec::new(), metadata);
        }
        Ok(())
    }

    fn embedded_page(
        &self,
        fields: DocumentFields,
        cursor: Option<DocumentScanCursor>,
        filter: Option<&QueryFilter>,
        control: &QueryControl,
    ) -> Result<DocumentScanPage, QueryError> {
        let mut request = DocumentScanRequest::new(256, fields, control.clone());
        if let Some(predicate) = filter.and_then(|f| f.predicate.as_ref()) {
            request = request.with_predicate(predicate);
        }
        if let Some(cursor) = cursor {
            request = request.with_cursor(cursor);
        }
        let mut page = self.scan_documents(request)?;
        if let Some(ids) = filter.and_then(|f| f.eligible_ids.as_ref()) {
            page.documents
                .retain(|doc| ids.numeric.binary_search(&doc.doc_id).is_ok());
        }
        Ok(page)
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
        self.query_embedded_report(text, leg, k, filter, control)
            .map(|report| (report.generation, report.hits))
    }

    /// Ranks chunks and counts eligible pending rows at one generation. A
    /// pending row is one lexical candidate over its whole source and never a
    /// dense candidate; hybrid fusion keeps only candidates with a score.
    pub fn query_embedded_report(
        &self,
        text: &str,
        leg: EmbeddedLeg,
        k: usize,
        filter: Option<&QueryFilter>,
        control: QueryControl,
    ) -> Result<EmbeddedReport, QueryError> {
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
        let mut pending_rows = 0_u64;
        let generation = loop {
            let fields = DocumentFields::TEXT
                | DocumentFields::ATTRIBUTES
                | DocumentFields::METADATA
                | DocumentFields::CHUNKS;
            let page = self.embedded_page(fields, cursor, filter, &control)?;
            for doc in page.documents {
                let row = decode_row(doc.metadata.as_deref().unwrap_or_default())
                    .map_err(QueryError::Store)?;
                // An empty row has no chunk: it is never a candidate.
                if row.state == EmbeddedRowState::Empty {
                    continue;
                }
                let source = doc
                    .text
                    .as_deref()
                    .ok_or_else(|| error("missing embedded source"))?;
                if row.state == EmbeddedRowState::Pending {
                    pending_rows = pending_rows.saturating_add(1);
                    if leg != EmbeddedLeg::Dense {
                        index
                            .push_document(&self.tokenizer, &Document::with_text(source))
                            .map_err(|e| error(&e.to_string()))?;
                        rows.push(EmbeddedHit {
                            version: DocumentVersion::new(doc.doc_id, doc.revision),
                            chunk: 0,
                            timestamp: doc.timestamp,
                            bytes: 0..source.len(),
                            text: source.to_owned(),
                            attributes: doc.attributes.clone().unwrap_or_default(),
                            score: 0.,
                            state: EmbeddedRowState::Pending,
                        });
                    }
                    continue;
                }
                let mut data = row.chunks;
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
                        state: EmbeddedRowState::Ready,
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
                // A pending row with no lexical match contributes nothing.
                let mut ranked: Vec<_> = scores
                    .into_iter()
                    .enumerate()
                    .filter(|(_, score)| *score > 0.)
                    .collect();
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
        Ok(EmbeddedReport {
            generation,
            pending_rows,
            hits,
        })
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
            let user = decode_row(doc.metadata.as_deref().unwrap_or_default())?.caller;
            doc.metadata = (!user.is_empty()).then(|| user.to_vec());
        }
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[allow(clippy::expect_used)]
fn same_text_reuse_rejects_corrupt_chunks() {
    let header = b"ZECHUNK1\x02\0\0\0\0\xff";
    let suffix = [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0];
    let pack = |suffix: &[u8]| repack("é", &[header.as_slice(), suffix].concat(), &[7, 0, 9], 2);
    let (first, packed) = pack(&suffix).expect("old format");
    assert_eq!(first, vec![1., 0.]);
    assert_eq!(
        packed,
        [b"ZECHUNK1\x03\0\0\0\x07\0\x09".as_slice(), &suffix].concat()
    );
    assert!(pack(&[]).is_err());
    assert!(pack(&suffix[..15]).is_err());
    let mut malformed = suffix;
    malformed[4] = 1;
    assert!(pack(&malformed).is_err());
    malformed = suffix;
    malformed[8..12].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(pack(&malformed).is_err());
}

#[cfg(test)]
#[test]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
fn wf988_validates_empty_envelope() {
    // ZESTATE1, state 2 (empty), admission token, caller_len 3, caller.
    let empty = [
        b"ZESTATE1\x02\0\0\0".as_slice(),
        &[0x5a; 16],
        b"\x03\0\0\0FAK",
    ]
    .concat();
    let row = decode_row(&empty).expect("complete empty row");
    assert_eq!(row.state, EmbeddedRowState::Empty);
    assert_eq!((row.caller, row.chunks), (b"FAK".as_slice(), [].as_slice()));
    for source in [None, Some("")] {
        assert!(verify_row(source, Some(&empty), false, 2).is_ok());
        // An empty row never holds a parent vector.
        assert!(verify_row(source, Some(&empty), true, 2).is_err());
    }
    // An empty row with real text, a short caller length or chunk bytes.
    assert!(verify_row(Some("x"), Some(&empty), false, 2).is_err());
    assert!(verify_row(None, Some(&empty[..empty.len() - 1]), false, 2).is_err());
    let chunked = [empty.as_slice(), &[0, 0, 0, 0, 0, 0, 0, 0]].concat();
    assert!(verify_row(None, Some(&chunked), false, 2).is_err());
    // A ready row keeps the strict checks: nonempty text without chunks.
    let ready = b"ZECHUNK1\x03\0\0\0FAK";
    assert!(verify_row(Some("x"), Some(ready), true, 2).is_err());
    assert!(verify_row(None, Some(ready), true, 2).is_err());
    // A pending row still needs its source text.
    let pending = [b"ZESTATE1\x01\0\0\0".as_slice(), &[0x5a; 16], b"\0\0\0\0"].concat();
    assert!(verify_row(Some("x"), Some(&pending), false, 2).is_ok());
    assert!(verify_row(None, Some(&pending), false, 2).is_err());
    // The engine packs the same layout with no model call.
    let packed = state_row(EmbeddedRowState::Empty, b"FAK").expect("pack");
    assert_eq!(packed.len(), empty.len());
    assert_eq!(
        decode_row(&packed).expect("decode").state,
        EmbeddedRowState::Empty
    );
}
