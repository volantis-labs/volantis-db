//! Exact chunk neighbours from the document tower's persisted vectors.
use super::*;
use crate::ingest::DocId;

/// One matching stored chunk. Ties rank by ascending document id, then chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkNeighbour {
    /// Live parent document.
    pub id: DocId,
    /// Zero-based chunk ordinal.
    pub chunk: usize,
    /// Negative squared L2 distance to the closest selected source chunk.
    pub score: f64,
}

/// Candidate selection and result limits shared by every source in the request.
#[derive(Default)]
pub struct NeighbourOptions<'a> {
    /// Maximum hits per source (1..=4096). Defaults to 10 without a threshold.
    pub k: Option<usize>,
    /// Strict lower score bound. Without k, return all hits above it.
    pub threshold: Option<f64>,
    /// Include chunks belonging to the source document.
    pub include_source: bool,
    /// Store predicate, timestamp range and eligible ids applied before ranking.
    pub filter: Option<&'a QueryFilter>,
}

fn visit_vectors(
    doc: &StoredDocument,
    dims: usize,
    control: &QueryControl,
    mut visit: impl FnMut(usize, &[u8]) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
    let source = doc
        .text
        .as_deref()
        .ok_or_else(|| QueryError::Store(invalid("missing embedded source")))?;
    let row = decode_row(doc.metadata.as_deref().unwrap_or_default()).map_err(QueryError::Store)?;
    // A pending row has no stored vector: it is never a neighbour, and as a
    // source it has no chunk to compare.
    if row.state != EmbeddedRowState::Ready {
        return Ok(());
    }
    let mut data = row.chunks;
    if data.is_empty() {
        return Err(QueryError::Store(invalid("missing embedded chunks")));
    }
    let mut ordinal = 0;
    while !data.is_empty() {
        control.checkpoint()?;
        let (_, bytes) = read_chunk(&mut data, source, dims).map_err(QueryError::Store)?;
        visit(ordinal, bytes)?;
        ordinal += 1;
    }
    Ok(())
}

fn decode(bytes: &[u8], vector: &mut Vec<f32>) {
    vector.clear();
    vector.extend(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b)),
    );
}

fn rank(hits: &mut Vec<ChunkNeighbour>, limit: usize) {
    hits.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.id.cmp(&b.id))
            .then(a.chunk.cmp(&b.chunk))
    });
    hits.truncate(limit);
}

impl Store {
    /// Finds neighbours of one chunk or, with no ordinal, of the whole document.
    pub fn nearest_chunks(
        &self,
        id: DocId,
        chunk: Option<usize>,
        options: &NeighbourOptions<'_>,
        control: QueryControl,
    ) -> Result<Vec<ChunkNeighbour>, QueryError> {
        let mut groups = self.nearest_chunks_batch(&[(id, chunk)], options, control)?;
        groups
            .pop()
            .ok_or_else(|| QueryError::Store(invalid("missing neighbour result group")))
    }

    /// Scans eligible stored chunks once for the batch. Returns one ranked list
    /// in input order, including for duplicate sources. All reads must agree on
    /// the generation. Cancellation or timeout discards the entire result.
    pub fn nearest_chunks_batch(
        &self,
        sources: &[(DocId, Option<usize>)],
        options: &NeighbourOptions<'_>,
        control: QueryControl,
    ) -> Result<Vec<Vec<ChunkNeighbour>>, QueryError> {
        control.checkpoint()?;
        if !self.is_embedded() {
            return Err(QueryError::Store(StoreError::MissingEmbedder));
        }
        if options.k.is_some_and(|k| !(1..=4096).contains(&k))
            || options.threshold.is_some_and(|v| !v.is_finite())
        {
            return Err(QueryError::Store(StoreError::InvalidScan {
                detail: "expected k in 1..=4096 and a finite score threshold".into(),
            }));
        }
        let limit = options.k.unwrap_or(if options.threshold.is_some() {
            usize::MAX
        } else {
            10
        });
        let fields = DocumentFields::TEXT | DocumentFields::METADATA | DocumentFields::CHUNKS;
        let dims = self.embedded_dims().map_err(QueryError::Store)?;
        let mut generation = None;
        let mut check_generation = |current| {
            let first = *generation.get_or_insert(current);
            if first == current {
                Ok(())
            } else {
                Err(QueryError::Store(StoreError::ScanStale {
                    cursor_generation: first,
                    current_generation: current,
                }))
            }
        };
        let mut vectors = Vec::with_capacity(sources.len());
        for batch in sources.chunks(64) {
            control.checkpoint()?;
            let ids: Vec<_> = batch.iter().map(|(id, _)| *id).collect();
            let (current, docs) = self
                .get_documents_with_generation(&ids, fields)
                .map_err(QueryError::Store)?;
            check_generation(current)?;
            for (&(id, chunk), doc) in batch.iter().zip(docs) {
                let doc = doc.ok_or(QueryError::Store(StoreError::NeighbourSourceNotFound {
                    id,
                    chunk: None,
                }))?;
                let mut selected = Vec::new();
                visit_vectors(&doc, dims, &control, |ordinal, bytes| {
                    if chunk.is_none_or(|n| n == ordinal) {
                        let mut vector = Vec::with_capacity(dims);
                        decode(bytes, &mut vector);
                        selected.push(vector);
                    }
                    Ok(())
                })?;
                if selected.is_empty() {
                    return Err(QueryError::Store(StoreError::NeighbourSourceNotFound {
                        id,
                        chunk,
                    }));
                }
                vectors.push((id, selected));
            }
        }
        let mut groups = vec![Vec::new(); sources.len()];
        if sources.is_empty() {
            return Ok(groups);
        }
        let mut candidate = Vec::with_capacity(dims);
        let mut cursor = None;
        loop {
            let page = self.embedded_page(fields, cursor, options.filter, &control)?;
            check_generation(page.generation)?;
            for doc in page.documents {
                visit_vectors(&doc, dims, &control, |chunk, bytes| {
                    decode(bytes, &mut candidate);
                    for ((source, selected), hits) in vectors.iter().zip(&mut groups) {
                        if !options.include_source && *source == doc.doc_id {
                            continue;
                        }
                        let mut best = f64::NEG_INFINITY;
                        for vector in selected {
                            control.checkpoint()?;
                            let distance = vector
                                .iter()
                                .zip(&candidate)
                                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                                .sum::<f64>();
                            best = best.max(-distance);
                        }
                        if options.threshold.is_some_and(|t| best <= t) {
                            continue;
                        }
                        hits.push(ChunkNeighbour {
                            id: doc.doc_id,
                            chunk,
                            score: best,
                        });
                        // Bound scratch space for top-k while scanning the corpus only once.
                        if hits.len() == limit.saturating_mul(2) {
                            rank(hits, limit);
                        }
                    }
                    Ok(())
                })?;
            }
            cursor = page.continuation;
            if cursor.is_none() {
                break;
            }
        }
        for hits in &mut groups {
            control.checkpoint()?;
            rank(hits, limit);
        }
        control.checkpoint()?;
        Ok(groups)
    }
}
