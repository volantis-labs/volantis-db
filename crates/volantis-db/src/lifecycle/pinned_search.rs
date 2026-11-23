//! Graph retrieval uses the document half of its existing statement lease.
use super::native_graph::documents::NativeDocuments;
use super::*;

impl Store {
    pub(crate) fn graph_text_in(
        &self,
        pin: &NativeDocuments,
        node_text: Option<&LexicalIndexCache>,
        query: &crate::fts::query::LexicalQuery,
        k: usize,
        filter: Option<&QueryFilter>,
        control: QueryControl,
    ) -> Result<(crate::ingest::StoreLexicalSearchOutcome, u64), crate::fusion::FusionError> {
        let started = self.clock.now();
        let lease = SnapshotLease::new_at(Arc::clone(&pin.snapshot), pin.generation);
        let cancellation = QueryCancellation::new(&control, &lease);
        let inputs = LexicalInputs {
            filter,
            generation: pin.generation,
            cache: node_text.unwrap_or(&self.lexical_index_cache),
            snapshot: &pin.snapshot,
            active: &pin.active,
            accounting: &self.accounting,
        };
        let lexical = match query {
            crate::fts::query::LexicalQuery::Term(term) => PinnedLexicalQuery::Term(term),
            query => PinnedLexicalQuery::Structured(query),
        };
        let mut preparation = prepared_lexical::PreparedLexicalQuery::new(lexical);
        // The lexical legs break equal scores by physical (segment, row)
        // order. Graph text breaks them by ascending full node ID, so a
        // document tied with the k-th score can lie beyond the leg bound with
        // a smaller ID. Score one extra hit, and widen the bound until the
        // hits hold every document tied with the k-th score.
        let mut bound = k.saturating_add(1);
        let mut counters = crate::fts::search::SearchCounters::default();
        let hits = loop {
            let (hits, _, leg_counters, _) = match lexical {
                PinnedLexicalQuery::Term(_) => {
                    exact_lexical_leg(inputs, bound, &cancellation, &mut preparation)?
                }
                PinnedLexicalQuery::Structured(query) => exact_structured_lexical_leg(
                    inputs,
                    &self.tokenizer,
                    query,
                    bound,
                    &cancellation,
                    &mut preparation,
                )?,
            };
            accumulate_search_counters(&mut counters, &leg_counters);
            match widened_bound(&hits, k, bound) {
                Some(wider) => bound = wider,
                None => break hits,
            }
        };
        // Count matching eligible posting membership before top-k. Scorer
        // work counters count evaluated rows and cannot stand in for this domain.
        let (assembly, scoring) = match lexical {
            PinnedLexicalQuery::Term(_) => {
                let (prepared, _) = preparation.prepare_term(inputs, &cancellation)?;
                (
                    &prepared.assembly,
                    if prepared.assembly.index.total_tokens() == 0 {
                        None
                    } else {
                        Some(prepared.scoring()?.query())
                    },
                )
            }
            PinnedLexicalQuery::Structured(_) => {
                let (prepared, _) = preparation.prepare_structured(inputs, &cancellation)?;
                (
                    &prepared.assembly,
                    prepared.query.as_ref().map(|query| query.scoring().query()),
                )
            }
        };
        let mut matching = 0;
        if let Some(scoring) = scoring {
            for (segment, alive) in assembly.index.segments().iter().zip(&assembly.alive_sets) {
                let mut rows = crate::meta::DocBitmap::new();
                for term in &scoring.terms {
                    let mut work =
                        crate::fts::control::WorkCheck::new(|| cancellation.check_graph());
                    let Some(mut stream) = crate::fts::sealed::TermStream::open_controlled(
                        segment,
                        term,
                        &scoring.fields,
                        &mut work,
                    )
                    .map_err(QueryError::Scan)?
                    else {
                        continue;
                    };
                    while let Some(row) = stream.current_row() {
                        work.check_now().map_err(QueryError::Scan)?;
                        if alive.alive_bitmap().contains(row) {
                            rows.insert(row);
                        }
                        stream
                            .advance_controlled(&mut work)
                            .map_err(QueryError::Scan)?;
                    }
                }
                matching += rows.cardinality();
            }
        }
        let candidates = hits
            .into_iter()
            .map(|hit| {
                let document = hit
                    .document
                    .ok_or_else(|| crate::fusion::FusionError::Leg {
                        leg: crate::fusion::FusionLeg::Lexical,
                        kind: crate::fusion::LegFailureKind::Lexical,
                        detail: "lexical hit has no document identity".into(),
                    })?;
                Ok(crate::ingest::LexicalCandidate {
                    document,
                    score: hit.bm25,
                })
            })
            .collect::<Result<Vec<_>, crate::fusion::FusionError>>()?;
        let candidates = rank_graph_text(candidates, k);
        let mut diagnostics =
            crate::diag::QueryDiagnostics::lexical(crate::diag::LexicalDiagnostics {
                snapshot_generation: pin.generation,
                indexed_through_seq: pin
                    .active
                    .indexed_through_seq()
                    .max(crate::wal::LogSeq::new(pin.snapshot.absorbed_through())),
                requested_k: k,
                returned: candidates.len(),
                counters,
                elapsed: self.clock.now().saturating_duration_since(started),
            });
        diagnostics.tokenizer_epoch = Some(self.tokenizer.epoch());
        Ok((
            crate::ingest::StoreLexicalSearchOutcome {
                candidates,
                generation: pin.generation,
                diagnostics,
            },
            matching,
        ))
    }
}

/// Returns a wider leg bound while the leg filled `bound` and its last hit
/// still ties the k-th score, so an unscored document can share that score.
fn widened_bound(hits: &[hybrid::LexicalHit], k: usize, bound: usize) -> Option<usize> {
    let kth = hits.get(k.checked_sub(1)?)?.bm25;
    let last = hits.last()?.bm25;
    (hits.len() >= bound && last >= kth).then_some(bound.saturating_mul(2))
}

/// Graph text order: descending BM25, then ascending full node ID. This is
/// the retrieval tie rule. It is not the physical (segment, row) key that
/// the lexical scorers use for their own ties. Node IDs are unique among
/// live hits, so an unstable sort gives one order.
fn rank_graph_text(
    mut candidates: Vec<crate::ingest::LexicalCandidate>,
    k: usize,
) -> Vec<crate::ingest::LexicalCandidate> {
    candidates.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.document.doc_id().cmp(&right.document.doc_id()))
    });
    candidates.truncate(k);
    candidates
}

impl Store {
    fn graph_search_pin(
        &self,
        pin: &NativeDocuments,
        options: SearchOptions,
    ) -> Result<AdmittedVectorSearch<'_>, QueryError> {
        Ok(AdmittedVectorSearch {
            execution_options: options,
            pool: Some(self.ensure_query_pool()?),
            snapshot: snapshot::ReadSnapshot::new(Arc::clone(&pin.snapshot)),
            active_segment: Arc::clone(&pin.active),
            generation: pin.generation,
            _active_query: None,
        })
    }

    pub(crate) fn graph_vector_in(
        &self,
        pin: &NativeDocuments,
        request: crate::ingest::SearchRequest<'_>,
        k: usize,
        options: SearchOptions,
        control: QueryControl,
    ) -> Result<crate::ingest::SearchOutcome, QueryError> {
        let eligible_filter = request.eligible.map(|ids| {
            request.filter.cloned().map_or_else(
                || QueryFilter::eligible(&self.schema, ids),
                |filter| filter.with_eligible(ids),
            )
        });
        let request = request.with_filter(eligible_filter.as_ref().or(request.filter));
        let admitted = self.graph_search_pin(pin, options)?;
        let epoch = pin.snapshot.epoch_alias();
        let mut preparation =
            prepared::PreparedVectorQuery::new(request.vector(), epoch, &self.accounting);
        search_pinned(
            admitted.pool.as_deref(),
            &pin.snapshot,
            &pin.active,
            &self.accounting,
            pin.generation,
            epoch,
            request,
            k,
            options,
            control,
            GraphBoundMode::Shared,
            options.explicit_tier(),
            None,
            self.clock.now(),
            self.clock.as_ref(),
            &mut preparation,
            #[cfg(any(test, feature = "test-seams"))]
            self.vector_fault_controller.as_ref(),
        )
    }

    pub(crate) fn graph_hybrid_in(
        &self,
        pin: &NativeDocuments,
        request: crate::ingest::SearchRequest<'_>,
        lexical: &crate::fts::query::LexicalQuery,
        hybrid: &crate::fusion::HybridQuery,
        options: SearchOptions,
        control: QueryControl,
    ) -> Result<crate::ingest::StoreHybridSearchOutcome, crate::fusion::FusionError> {
        let lexical = match lexical {
            crate::fts::query::LexicalQuery::Term(term) => PinnedLexicalQuery::Term(term),
            lexical => PinnedLexicalQuery::Structured(lexical),
        };
        self.search_hybrid_prepared_then(
            || Ok::<_, std::convert::Infallible>(request),
            request.filter,
            request.eligible,
            lexical,
            hybrid,
            options,
            control,
            false,
            Some(self.graph_search_pin(pin, options)?),
            |outcome, _, _, _, _| Ok(outcome),
        )
        .map_err(|error| match error {
            HybridPreparationError::Preparation(never) => match never {},
            HybridPreparationError::Search(error) => error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::rank_graph_text;
    use crate::ingest::{DocId, DocumentVersion, LexicalCandidate, Revision};

    fn candidate(id: u128, score: f64) -> LexicalCandidate {
        LexicalCandidate {
            document: DocumentVersion::new(DocId::new(id), Revision::new(1)),
            score,
        }
    }

    fn ids(candidates: &[LexicalCandidate]) -> Vec<u128> {
        candidates
            .iter()
            .map(|candidate| candidate.document.doc_id().get())
            .collect()
    }

    /// WF-984: equal graph text scores rank by ascending full node ID, not
    /// by the physical order in which the lexical leg offers them.
    #[test]
    fn wf984_four_equal_nodes_use_durable_id() {
        let physical = [7, 11, 13, 4].map(|id| candidate(id, 1.0));
        assert_eq!(ids(&rank_graph_text(physical.to_vec(), 4)), [4, 7, 11, 13]);
        assert_eq!(ids(&rank_graph_text(physical.to_vec(), 2)), [4, 7]);
        let mut scored = physical.to_vec();
        scored.push(candidate(99, 2.0));
        assert_eq!(ids(&rank_graph_text(scored, 2)), [99, 4]);
    }
}
