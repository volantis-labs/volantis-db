//! Local Volantis DB search-extension fixture; not an original TCK fixture.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::result_large_err
)]
use volantis_db::epoch::{ComputeUnits, EmbeddingRuntime, EmbeddingTower, Normalization};
use volantis_db::lifecycle::Store;
use volantis_db::lifecycle::{CancelToken, QueryControl};
use volantis_db::property_graph::query::completed::{CompletedGraphResult, GraphQueryOptions};
use volantis_db::property_graph::query::plan::ParameterBinding;
use volantis_db::property_graph::staging::{StructuredOperation, StructuredWrite, WriteImage};
use volantis_db::property_graph::{
    ApplicationKey, CanonicalContents, CanonicalEmbedding, EntityKind, GraphName, GraphProperty,
    GraphRevision, PropertyData, PropertyValue,
};
use volantis_db_cypher::{CompileLimits, StatementError, execute};
pub struct SearchFixture {
    pub root: std::path::PathBuf,
    pub store: Option<Store>,
    pub tower: EmbeddingTower,
}
fn store_epoch(tower: &EmbeddingTower) -> volantis_db::epoch::StoreEpoch {
    use volantis_db::epoch::{EmbeddingEpoch, StoreEpoch};
    StoreEpoch {
        embedding: EmbeddingEpoch {
            query: tower.clone(),
            document: tower.clone(),
            alignment_digest: vec![],
        },
        tokenizer: volantis_db::fts::tokenizer::TokenizerConfig::text_default().epoch(),
    }
}

pub fn control() -> QueryControl {
    QueryControl::Cancel(CancelToken::new())
}
impl SearchFixture {
    pub fn create() -> Self {
        let root = crate::support::unique_temp_dir("ze58-search");
        let tower = EmbeddingTower {
            model_id: "ze58".into(),
            model_version: "1".into(),
            weights_digest: vec![58],
            dims: 2,
            normalization: Normalization::None,
            prompt_prefix: String::new(),
            max_tokens: 32,
            runtime: EmbeddingRuntime::CpuReference,
            compute_units: ComputeUnits::Cpu,
            os_build: None,
        };
        let store = Store::create_graph(
            &root,
            volantis_db::lifecycle::OpenOptions::new()
                .with_epoch(store_epoch(&tower))
                .with_max_resident_bytes(256 * 1024 * 1024),
            Some(tower.clone()),
        )
        .unwrap();
        for (key, text, vector) in [
            ("a", Some("amber birch"), Some([1.0, 1.0])),
            ("b", Some("amber"), Some([5.0, 5.0])),
            ("c", Some("birch"), None),
            ("d", None, Some([0.0, 0.0])),
        ] {
            let mut labels = [GraphName::new("Chunk").unwrap()];
            let mut properties = [GraphProperty::new(
                GraphName::new("key").unwrap(),
                PropertyValue::new(PropertyData::String(key)).unwrap(),
            )];
            let contents = CanonicalContents::node(
                &mut labels,
                &mut properties,
                text,
                vector
                    .as_ref()
                    .map(|v| CanonicalEmbedding::new(&tower, v).unwrap()),
            )
            .unwrap();
            use volantis_db::ingest::{
                DocId, DocumentVersion, IngestBatch, IngestDocument, Revision,
            };
            let receipt = store
                .graph_apply(
                    &[StructuredWrite {
                        key: ApplicationKey::new(EntityKind::Node, "ze58", key).unwrap(),
                        revision: GraphRevision::new(1).unwrap(),
                        operation: StructuredOperation::Create,
                        image: Some(WriteImage::Node(&contents)),
                    }],
                    &control(),
                )
                .unwrap();
            let volantis_db::property_graph::EntityId::Node(id) = receipt.receipts()[0].entity
            else {
                panic!("node receipt")
            };
            let document = IngestDocument::new(
                DocumentVersion::new(DocId::new(id.get()), Revision::new(1)),
                vector.unwrap_or([1000.0; 2]).to_vec(),
            );
            let document = text.map_or(document.clone(), |text| document.with_text(text));
            store
                .ingest(
                    IngestBatch::new(vec![document]).with_epoch(store.epoch_identity().unwrap()),
                )
                .unwrap();
        }
        let fixture = Self {
            root,
            store: Some(store),
            tower,
        };
        fixture.run("CREATE (alice:Person {key: 'Alice'}), (project:Project {key: 'X'}), (m:Meeting {key: 'M'}), (alice)-[:ATTENDED]->(m), (alice)-[:ATTENDED]->(m), (m)-[:ABOUT]->(project)");
        fixture.run("MATCH (a:Chunk {key: 'a'}), (b:Chunk {key: 'b'}), (m:Meeting) CREATE (a)-[:FROM_MEETING]->(m), (a)-[:FROM_MEETING]->(m), (b)-[:FROM_MEETING]->(m)");
        fixture
    }
    pub fn apply(&self, writes: &[StructuredWrite<'_, '_>]) {
        self.store().graph_apply(writes, &control()).unwrap();
    }
    pub fn store(&self) -> &Store {
        self.store.as_ref().unwrap()
    }
    pub fn run(&self, text: &str) -> CompletedGraphResult {
        self.try_run(text, &[])
            .unwrap_or_else(|e| panic!("{text}: {e}"))
    }
    pub fn try_run(
        &self,
        text: &str,
        parameters: &[ParameterBinding<'_>],
    ) -> Result<CompletedGraphResult, StatementError> {
        execute(
            self.store(),
            &control(),
            &GraphQueryOptions::default(),
            text,
            parameters,
            CompileLimits::default(),
        )
    }
    pub fn reopen(&mut self) {
        self.store.take().unwrap().close_graph().unwrap();
        self.store = Some(
            Store::open_graph(
                &self.root,
                volantis_db::lifecycle::OpenOptions::new()
                    .with_epoch(store_epoch(&self.tower))
                    .with_max_resident_bytes(256 * 1024 * 1024),
                Some(self.tower.clone()),
            )
            .unwrap(),
        );
    }
}
impl Drop for SearchFixture {
    fn drop(&mut self) {
        if let Some(store) = self.store.take() {
            let _ = store.close_graph();
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
