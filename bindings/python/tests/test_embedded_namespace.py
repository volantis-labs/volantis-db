"""Public Store contracts for embedded namespaces and chunk neighbours."""

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import zeppelin_embed as ze
from zeppelin_embed import _ffi
from zeppelin_embed._library import LIBRARY

REVISION = "0123456789abcdef0123456789abcdef01234567"
REPO = "fixture/tiny"
BUNDLE = (
    Path(__file__).resolve().parents[3]
    / "crates/zeppelin-embed-text/tests/fixtures/ze115/gte.zem"
)


def test_embedded_namespace_validation(tmp_path: Path) -> None:
    model = ze.EmbeddedModelRef(REPO, REVISION, "0" * 64)
    spec = ze.NamespaceSpec(vector_space=ze.VectorSpace(2), embedded_model=model)
    with pytest.raises(ze.InvalidArgument, match="determines.*vector space"):
        ze.open_namespace(tmp_path / "db", "notes", spec)
    assert not (tmp_path / "db").exists()


def test_embedded_namespace_without_text_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_ffi, "TEXT_AVAILABLE", False)
    model = ze.EmbeddedModelRef(REPO, REVISION, "0" * 64)
    with pytest.raises(ze.TextUnavailable):
        ze.open_namespace(
            tmp_path / "db", "notes", ze.NamespaceSpec(embedded_model=model)
        )
    assert not (tmp_path / "db").exists()
    with ze.open(tmp_path / "core") as store:
        with pytest.raises(ze.TextUnavailable):
            store.query_embedded("bronze")
        with pytest.raises(ze.TextUnavailable):
            store.nearest_chunks(1)
        with pytest.raises(ze.TextUnavailable):
            store.nearest_chunks_batch([1])


@pytest.mark.skipif(not ze.TEXT_AVAILABLE, reason="requires a text-enabled build")
def test_embedded_namespace_offline_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ZE_MODEL_OFFLINE", "1")
    model = ze.EmbeddedModelRef(REPO, REVISION, "0" * 64)
    with pytest.raises(ze.NotFound, match="offline"):
        ze.open_namespace(
            tmp_path / "db", "notes", ze.NamespaceSpec(embedded_model=model)
        )
    assert not (tmp_path / "db").exists()


@pytest.fixture
def vector_store(
    tmp_path: Path,
) -> Iterator[ze.Store]:
    if not ze.TEXT_AVAILABLE:
        pytest.skip("model-independent ABI checks use the text-enabled build")
    spec = ze.NamespaceSpec(vector_space=ze.VectorSpace(2))
    with ze.open_namespace(tmp_path, "vectors", spec) as store:
        yield store


def test_embedded_query_requires_embedder(vector_store: ze.Store) -> None:
    with pytest.raises(ze.MissingEmbedder) as error:
        vector_store.query_embedded("bronze")
    assert error.value.code == 62


def test_embedded_query_empty_result_has_no_arena_to_free(
    vector_store: ze.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A successful native query with no hits leaves the result arena null.
    def empty_query(*_args: object) -> int:
        return 0

    monkeypatch.setattr(LIBRARY, "ze_query_embedded", empty_query)
    assert vector_store.query_embedded("bronze", eligible_ids=[]) == ()


def test_nearest_chunks_requires_embedder(vector_store: ze.Store) -> None:
    with pytest.raises(ze.MissingEmbedder) as error:
        vector_store.nearest_chunks(1)
    assert error.value.code == 62


def test_nearest_chunks_batch_cancellation(vector_store: ze.Store) -> None:
    with ze.CancelToken() as token:
        token.cancel()
        with pytest.raises(ze.Cancelled):
            vector_store.nearest_chunks_batch(
                [1, ze.NeighbourSource(2, chunk=0)], cancel_token=token
            )


@pytest.mark.skipif(not ze.TEXT_AVAILABLE, reason="requires a text-enabled build")
def test_embedded_text_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = BUNDLE.read_bytes()
    manifest = json.dumps(
        {
            "format": 1,
            "name": "tiny",
            "bundle": "tiny.zem",
            "query_coreml": None,
            "files": [
                {
                    "path": "tiny.zem",
                    "size": len(bundle),
                    "sha256": hashlib.sha256(bundle).hexdigest(),
                }
            ],
        }
    ).encode()
    model = ze.EmbeddedModelRef(REPO, REVISION, hashlib.sha256(manifest).hexdigest())
    cache = tmp_path / "cache"
    entry = cache / "fixture%2Ftiny" / REVISION
    entry.mkdir(parents=True)
    (entry / "zeppelin-model.json").write_bytes(manifest)
    (entry / "tiny.zem").write_bytes(bundle)
    monkeypatch.setenv("ZE_MODEL_CACHE", str(cache))
    monkeypatch.setenv("ZE_MODEL_OFFLINE", "1")

    spec = ze.NamespaceSpec(
        attributes=(ze.AttributeDefinition(1, "speaker", ze.AttributeType.U64),),
        embedded_model=model,
    )
    doc_id = (1 << 120) + 42
    other = doc_id + 1
    attributes = (ze.AttributeValue(1, ze.AttributeType.U64, 7),)
    source = "café bronze zeppelin " * 600
    with ze.open_namespace(
        tmp_path / "db", "notes", spec, max_resident_bytes=1 << 28
    ) as store:
        with pytest.raises(ze.InvalidArgument, match="without a vector"):
            store.upsert(
                [
                    ze.StoredDocument(
                        doc_id, text=source, vector=np.ones(1, dtype=np.float32)
                    )
                ]
            )
        report = store.upsert(
            [
                ze.StoredDocument(
                    doc_id,
                    revision=3,
                    timestamp=12,
                    text=source,
                    metadata=b"source",
                    attributes=attributes,
                ),
                ze.StoredDocument(
                    other, timestamp=13, text="bronze zeppelin", attributes=attributes
                ),
            ]
        )
        stored = store.get([doc_id], vector=False)
        assert stored.generation == report.generation
        assert (
            stored.documents[0] is not None
            and stored.documents[0].metadata == b"source"
        )

        constraints = dict(
            filter=ze.Filter.eq("speaker", 7),
            timestamp_range=(12, 13),
            eligible_ids=[doc_id, other],
        )
        for leg in (ze.TextLegs.DENSE, ze.TextLegs.LEXICAL, ze.TextLegs.HYBRID):
            hits = store.query_embedded("bronze", leg=leg, k=2, **constraints)
            assert len(hits) == 2
            assert len({hit.chunk for hit in hits}) == 2
            assert all(isinstance(hit, ze.EmbeddedHit) for hit in hits)
            for hit in hits:
                assert (hit.doc_id, hit.revision, hit.timestamp, hit.attributes) == (
                    doc_id,
                    3,
                    12,
                    attributes,
                )
                assert (
                    hit.text == source.encode()[hit.byte_start : hit.byte_end].decode()
                )
        assert store.query_embedded("bronze", eligible_ids=[]) == ()

        probe = ze.NeighbourSource(other, chunk=0)
        hits = store.nearest_chunks(probe, k=2, **constraints)
        assert len(hits) == 2
        assert all(isinstance(hit, ze.ChunkNeighbour) for hit in hits)
        assert all(
            hit.doc_id == doc_id and math.isfinite(hit.score) and hit.score <= 0
            for hit in hits
        )
        assert hits[0].score >= hits[1].score
        assert store.nearest_chunks_batch(
            [doc_id, probe, probe], k=2, **constraints
        ) == (
            (),
            hits,
            hits,
        )
        assert store.nearest_chunks_batch([doc_id, other], threshold=0) == ((), ())
        assert store.nearest_chunks(
            other, k=1, include_source=True, eligible_ids=[other]
        ) == (ze.ChunkNeighbour(other, 0, 0.0),)
        with pytest.raises(ze.NotFound):
            store.nearest_chunks(other + 1)
