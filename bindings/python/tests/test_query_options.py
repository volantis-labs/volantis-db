from pathlib import Path

import numpy as np
import pytest

import volantis_db as ze


def test_query_filters_before_ranking(tmp_path: Path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    spec = ze.NamespaceSpec(
        attributes=(ze.AttributeDefinition(1, "rank", ze.AttributeType.U64),),
        vector_space=ze.VectorSpace(2),
    )
    with ze.open_namespace(tmp_path, "filtered", spec) as store:
        store.upsert(
            [
                ze.StoredDocument(
                    doc_id,
                    vector=vector,
                    text=text,
                    attributes=(ze.AttributeValue(1, ze.AttributeType.U64, rank),),
                )
                for doc_id, vector, text, rank in zip(
                    [1, 2, 3],
                    vectors,
                    ["cedar", "cedar among many trees", "amber"],
                    [8, 7, 7],
                    strict=True,
                )
            ]
        )
        store.seal()
        selected = ze.Filter.eq("rank", 7)
        for legs in (
            {"text": "cedar"},
            {"vector": vectors[0]},
            {"text": "cedar", "vector": vectors[0]},
        ):
            assert [hit.doc_id for hit in store.query(**legs, k=1).hits] == [1]
            assert [hit.doc_id for hit in store.query(**legs, k=1, filter=selected).hits] == [2]
            assert store.query(**legs, filter=selected, eligible_ids=[1]).hits == ()
            eligible = store.query(**legs, filter=selected, eligible_ids=[1, 2], k=1)
            assert [hit.doc_id for hit in eligible.hits] == [2]

        with pytest.raises(ze.InvalidArgument) as error:
            store.query(text="cedar", filter=ze.Filter.eq(99, 7))
        assert error.value.code_name == "ZE_ERR_INVALID_ARGUMENT"


def test_query_snippets_preserve_utf8_ranges(tmp_path: Path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with ze.open(tmp_path) as store:
        store.ingest(
            [1, 2],
            vectors,
            texts=["écho café amber 🌿 ensuite", "another tree"],
            timestamps=[10, 20],
        )
        store.seal()
        assert store.query(text="café amber").hits[0].snippet is None
        expected = ze.Snippet(
            text="café amber 🌿",
            highlights=(
                ze.SnippetHighlight(start=0, end=5, source_byte_start=6, source_byte_end=11),
                ze.SnippetHighlight(start=6, end=11, source_byte_start=12, source_byte_end=17),
            ),
            truncated_start=True,
            truncated_end=True,
            source_byte_start=6,
            source_byte_end=22,
        )
        selected = ze.Filter.eq("ts", 10)
        for options in ({}, {"filter": selected}, {"filter": selected, "eligible_ids": [1, 2]}):
            result = store.query(text="café amber", snippet_bytes=13, **options)
            assert [(hit.doc_id, hit.snippet) for hit in result.hits] == [(1, expected)]
        assert store.query(text="café", snippet_bytes=13, filter=ze.Filter.eq("ts", 20)).hits == ()
        assert store.query(text="café", snippet_bytes=13, filter=selected, eligible_ids=[2]).hits == ()

        hybrid = store.query(text="café amber", vector=vectors[0], snippet_bytes=13, k=2)
        assert {hit.doc_id: hit.snippet for hit in hybrid.hits} == {1: expected, 2: None}

        with pytest.raises(ze.InvalidArgument, match="snippets require a lexical leg") as error:
            store.query(vector=vectors[0], snippet_bytes=13)
        assert error.value.code_name == "ZE_ERR_INVALID_ARGUMENT"
