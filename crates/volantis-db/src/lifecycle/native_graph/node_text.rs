//! Derived lexical postings from canonical graph node text.
//! Canonical node records remain the durable source. Each publication replaces
//! only the ID blocks touched by its normalized deltas.
use super::super::LexicalIndexCache;
use super::NativeGraphError;
use crate::fts::index::{Document, SegmentIndex};
use crate::fts::sealed::SealedSegment;
use crate::ingest::{DocId, DocumentVersion, IngestError, Revision};
use crate::lifecycle::{QueryControl, Store};
use crate::meta::AliveSet;
use crate::property_graph::resources::{GraphReservation, GraphResources};
use crate::property_graph::storage::records::RecordView;
use crate::property_graph::storage::stream::PayloadSlice;
use crate::property_graph::storage::tree::directory::{BlockSource, TreeError, TreeResources};
use crate::property_graph::{GraphOperation, NodeId};
use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "test-seams")]
thread_local! { static CANCEL_BUILD: std::cell::Cell<Option<crate::lifecycle::CancelToken>> = const { std::cell::Cell::new(None) }; }

#[cfg(feature = "test-seams")]
pub(crate) fn cancel_next_build(token: crate::lifecycle::CancelToken) {
    CANCEL_BUILD.with(|slot| slot.set(Some(token)));
}

pub(crate) struct NodeTextRow {
    pub(crate) version: DocumentVersion,
    pub(crate) text: String,
    _charge: GraphReservation,
}
type Row = Arc<NodeTextRow>;

// Bound the number of existing rows a single put must retokenize.
const ROWS_PER_BLOCK: u128 = 128;

pub(crate) struct NodeTextBlock {
    pub(crate) rows: Vec<Row>,
    pub(crate) postings: Arc<SealedSegment>,
    pub(crate) alive: Arc<AliveSet>,
    _charge: GraphReservation,
}

#[derive(Default)]
pub(super) struct TextChanges {
    rows: BTreeMap<u128, BTreeMap<u128, Option<Row>>>,
    masks: BTreeMap<DocId, Option<Revision>>,
}

impl TextChanges {
    pub(super) fn remove(&mut self, node: crate::property_graph::NodeId) {
        self.rows
            .entry(node.get() / ROWS_PER_BLOCK)
            .or_default()
            .insert(node.get(), None);
    }

    pub(super) fn delete(&mut self, node: NodeId, backing: Option<DocumentVersion>) {
        self.remove(node);
        if let Some(document) = backing {
            self.masks
                .insert(document.doc_id(), Some(document.revision()));
        }
    }

    pub(super) fn observe<S: BlockSource>(
        &mut self,
        node: NodeId,
        record: &RecordView<'_, S>,
        shared: &GraphResources,
        resources: &mut TreeResources<'_>,
    ) -> Result<(), TreeError> {
        // A structured put's explicit text wins even if the two independent
        // revision counters coincide. A mixed, document-only image supplies
        // the text through its replacement document at the same revision.
        if let Some(document) = record.document_version() {
            let override_text = record.provenance().operation() == GraphOperation::StructuredPut
                && (record.canonical().stored_text().is_some()
                    || document.revision().get() != record.revision().get());
            self.masks.insert(
                document.doc_id(),
                override_text.then_some(document.revision()),
            );
            if !override_text {
                return Ok(());
            }
        }
        if let Some(text) = record.canonical().stored_text() {
            self.record(
                DocumentVersion::new(node.into(), Revision::new(record.revision().get())),
                text,
                shared,
                resources,
            )?;
        }
        Ok(())
    }

    fn record<S: BlockSource>(
        &mut self,
        version: DocumentVersion,
        payload: PayloadSlice<'_, S>,
        shared: &GraphResources,
        resources: &mut TreeResources<'_>,
    ) -> Result<(), TreeError> {
        let length = usize::try_from(payload.len()).map_err(|_| TreeError::Memory)?;
        let bytes_needed = length
            .checked_add(std::mem::size_of::<NodeTextRow>() + 2 * std::mem::size_of::<usize>())
            .ok_or(TreeError::Memory)?;
        let charge = shared
            .reserve(bytes_needed)
            .map_err(|error| TreeError::Control(crate::lifecycle::QueryError::Store(error)))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| TreeError::Memory)?;
        bytes.resize(length, 0);
        if payload.read_at(0, &mut bytes, resources)? != length {
            return Err(TreeError::Invalid("short graph node text"));
        }
        let text =
            String::from_utf8(bytes).map_err(|_| TreeError::Invalid("invalid graph node text"))?;
        let id = version.doc_id().get();
        self.rows.entry(id / ROWS_PER_BLOCK).or_default().insert(
            id,
            Some(Arc::new(NodeTextRow {
                version,
                text,
                _charge: charge,
            })),
        );
        Ok(())
    }

    pub(super) fn apply(
        self,
        base: Option<&Arc<LexicalIndexCache>>,
        store: &Store,
        shared: &GraphResources,
        control: &QueryControl,
    ) -> Result<Option<Arc<LexicalIndexCache>>, NativeGraphError> {
        let mask_changes = self.masks.into_iter().filter(|(id, revision)| {
            base.and_then(|cache| cache.masked_documents.get(id)) != revision.as_ref()
        });
        let mut mask_changes = mask_changes.peekable();
        if self.rows.is_empty() && mask_changes.peek().is_none() {
            return Ok(base.cloned());
        }
        let mut masks = base
            .map(|cache| cache.masked_documents.clone())
            .unwrap_or_default();
        for (id, revision) in mask_changes {
            if let Some(revision) = revision {
                masks.insert(id, revision);
            } else {
                masks.remove(&id);
            }
        }
        let mut blocks = base
            .map(|cache| cache.node_text.clone())
            .unwrap_or_default();
        for (key, changes) in self.rows {
            #[cfg(feature = "test-seams")]
            if let Some(token) = CANCEL_BUILD.with(|slot| slot.take()) {
                token.cancel();
            }
            control.checkpoint().map_err(TreeError::Control)?;
            let mut rows: BTreeMap<_, _> = blocks
                .get(&key)
                .into_iter()
                .flat_map(|block| &block.rows)
                .map(|row| (row.version.doc_id().get(), row.clone()))
                .collect();
            for (id, row) in changes {
                match row {
                    Some(row) => {
                        rows.insert(id, row);
                    }
                    None => {
                        rows.remove(&id);
                    }
                }
            }
            if rows.is_empty() {
                blocks.remove(&key);
                continue;
            }
            let rows: Vec<_> = rows.into_values().collect();
            let bytes =
                rows.capacity() * std::mem::size_of::<Row>() + std::mem::size_of::<NodeTextBlock>();
            let mut charge = shared.reserve(bytes)?;
            let mut index = SegmentIndex::new();
            for row in &rows {
                control.checkpoint().map_err(TreeError::Control)?;
                index
                    .push_document(&store.tokenizer, &Document::with_text(&row.text))
                    .map_err(IngestError::Lexical)?;
                charge.resize(
                    bytes
                        .checked_add(index.resident_bytes())
                        .ok_or(TreeError::Memory)?,
                )?;
            }
            let postings = Arc::new(
                SealedSegment::seal(&index)
                    .map_err(crate::fts::index::IndexError::from)
                    .map_err(IngestError::Lexical)?,
            );
            let alive = Arc::new(AliveSet::new(postings.row_count()));
            let resident = postings.resident_bytes().map_err(|_| TreeError::Memory)?;
            charge.resize(
                bytes
                    .checked_add(resident)
                    .and_then(|n| n.checked_add(alive.resident_bytes()?))
                    .ok_or(TreeError::Memory)?,
            )?;
            blocks.insert(
                key,
                Arc::new(NodeTextBlock {
                    rows,
                    postings,
                    alive,
                    _charge: charge,
                }),
            );
        }
        Ok((!blocks.is_empty() || !masks.is_empty()).then(|| {
            Arc::new(LexicalIndexCache {
                node_text: blocks,
                masked_documents: masks,
                ..LexicalIndexCache::new()
            })
        }))
    }
}

// Graph-only nodes have no document timestamp or attribute columns.
// The predicate has already been type-checked against the store schema.
pub(crate) fn matches_absent_columns(predicate: &crate::meta::Predicate) -> bool {
    use crate::meta::Predicate;
    match predicate {
        Predicate::IsNull(_) => true,
        Predicate::Eq { .. }
        | Predicate::In { .. }
        | Predicate::Range(_)
        | Predicate::Exists(_) => false,
        Predicate::And(children) => children.iter().all(matches_absent_columns),
        Predicate::Or(children) => children.iter().any(matches_absent_columns),
        Predicate::Not(inner) => !matches_absent_columns(inner),
    }
}
