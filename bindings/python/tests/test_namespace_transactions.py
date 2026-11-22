"""Public Python coverage of namespace tokenizer and transaction operations."""

from pathlib import Path

import numpy as np
import pytest

import zeppelin_embed as ze


def test_namespace_tokenizer_profile_persists(tmp_path: Path) -> None:
    spec = ze.NamespaceSpec(tokenizer_profile=ze.TokenizerProfile.VOICE)
    with ze.open_namespace(tmp_path, "transcript", spec) as store:
        store.upsert([ze.StoredDocument(1, text="twenty five")])
        store.seal()

    with ze.open_namespace(tmp_path, "transcript", spec) as store:
        assert [hit.doc_id for hit in store.query(text="25").hits] == [1]

    with pytest.raises(ze.EpochMismatch):
        ze.open_namespace(tmp_path, "transcript", ze.NamespaceSpec(tokenizer_profile="code"))


def test_namespace_batch_is_atomic_and_returns_ordered_generations(tmp_path: Path) -> None:
    notes = ze.NamespaceSpec(
        attributes=(ze.AttributeDefinition(1, "rank", ze.AttributeType.U64, nullable=True),),
        tokenizer_profile=ze.TokenizerProfile.CODE,
    )
    vectors = ze.NamespaceSpec(vector_space=ze.VectorSpace(2))
    vector = np.array([0.5, 1.0], dtype=np.float32)
    with ze.open_namespace(tmp_path, "notes", notes) as store:
        store.upsert([ze.StoredDocument(1), ze.StoredDocument(2)])
        store.seal()
    with ze.open_namespace(tmp_path, "vectors", vectors):
        pass

    note_mutation = ze.NamespaceMutation(
        "notes",
        notes,
        upserts=(
            ze.UpsertDocument(1, revision=2, expected_revision=1),
            ze.StoredDocument(3, attributes=(ze.AttributeValue(1, ze.AttributeType.U64, 42),)),
        ),
        deletes=(2,),
        delete_where=ze.Filter.eq("rank", 42),
    )
    with pytest.raises(ze.Io, match="revision condition failed"):
        ze.namespace_batch(tmp_path, [
            note_mutation,
            ze.NamespaceMutation("vectors", vectors, upserts=(
                ze.UpsertDocument(4, vector=vector, expected_revision=7),
            )),
        ])
    with ze.open_namespace(tmp_path, "notes", notes) as store:
        result = store.get([1, 2, 3])
        assert [doc.revision if doc else None for doc in result.documents] == [1, 1, None]
    with ze.open_namespace(tmp_path, "vectors", vectors) as store:
        assert store.get([4]).documents == (None,)

    generations = ze.namespace_batch(tmp_path, [
        ze.NamespaceMutation("vectors", vectors, upserts=(ze.StoredDocument(4, vector=vector),)),
        note_mutation,
    ])
    with ze.open_namespace(tmp_path, "vectors", vectors, access_mode="read_only") as store:
        result = store.get([4])
        assert result.generation == generations[0]
        assert result.documents[0] is not None
        np.testing.assert_array_equal(result.documents[0].vector, vector)
    with ze.open_namespace(tmp_path, "notes", notes, access_mode="read_only") as store:
        result = store.get([1, 2, 3])
        # Reopening a deleting participant can include a later purge generation.
        assert result.generation >= generations[1]
        assert [doc.revision if doc else None for doc in result.documents] == [2, None, None]
    assert len(generations) == 2
    assert generations[0] < generations[1]


def test_namespace_batch_live_keeps_handles_usable(tmp_path: Path) -> None:
    spec = ze.NamespaceSpec()
    with (
        ze.open_namespace(tmp_path, "a", spec) as a,
        ze.open_namespace(tmp_path, "b", spec) as b,
    ):
        a.upsert([ze.StoredDocument(1)])
        a.upsert([ze.StoredDocument(2)])
        mutations = [
            ze.LiveNamespaceMutation("b", spec, store=b, upserts=(ze.StoredDocument(3),)),
            ze.LiveNamespaceMutation("a", spec, store=a, upserts=(ze.StoredDocument(4),)),
        ]
        b_generation, a_generation = ze.namespace_batch_live(tmp_path, mutations)
        a_result = a.get([2, 4])
        b_result = b.get([3])
        assert [doc.doc_id for doc in a_result.documents if doc] == [2, 4]
        assert [doc.doc_id for doc in b_result.documents if doc] == [3]
        assert a_result.generation == a_generation > b_generation == b_result.generation
        assert a.upsert([ze.StoredDocument(5)]).generation > a_generation

        b.close()
        before = a.get([6])
        with pytest.raises(ze.Closed):
            ze.namespace_batch_live(tmp_path, [
                ze.LiveNamespaceMutation("a", spec, store=a, upserts=(ze.StoredDocument(6),)),
                mutations[0],
            ])
        assert a.get([6]) == before


def test_declare_cascade_rejects_cycles(tmp_path: Path) -> None:
    spec = ze.NamespaceSpec(
        attributes=(ze.AttributeDefinition(1, "owner", ze.AttributeType.ID128, nullable=True),),
    )
    participants = [ze.CascadeParticipant(name, spec) for name in ("parent", "child")]
    for participant in participants:
        with ze.open_namespace(tmp_path, participant.name, participant.spec):
            pass
    rule = ze.CascadeDeclaration(parent_index=0, child_index=1, attribute_id=1)
    assert ze.declare_cascade(tmp_path, participants, rule) is None
    with ze.open_namespace(tmp_path, "child", spec):
        pass
    assert ze.declare_cascade(tmp_path, participants, rule) is None
    with pytest.raises(ze.CascadeCycle) as caught:
        ze.declare_cascade(
            tmp_path, participants, ze.CascadeDeclaration(parent_index=1, child_index=0, attribute_id=1)
        )
    assert "parent" in caught.value.message
    assert "child" in caught.value.message


def test_delete_cascade_is_atomic_and_transitive(tmp_path: Path) -> None:
    spec = ze.NamespaceSpec(
        attributes=(ze.AttributeDefinition(1, "owner", ze.AttributeType.ID128, nullable=True),),
    )
    parent_id = (1 << 127) | 7
    for name, doc_id, owner in (
        ("notes", parent_id, None),
        ("segments", 12, parent_id),
        ("words", 24, 12),
    ):
        with ze.open_namespace(tmp_path, name, spec) as store:
            attribute = ze.AttributeValue(1, ze.AttributeType.ID128, owner)
            store.upsert([ze.StoredDocument(doc_id, attributes=(attribute,)), ze.StoredDocument(42)])
            store.seal()
            document = store.get([doc_id]).documents[0]
            assert document is not None
            assert document.attributes == (() if owner is None else (attribute,))

    participants = [ze.CascadeParticipant(name, spec) for name in ("notes", "segments", "words")]
    ze.declare_cascade(tmp_path, participants, ze.CascadeDeclaration(0, 1, 1))
    ze.declare_cascade(tmp_path, participants, ze.CascadeDeclaration(1, 2, 1))

    parent = ze.CascadeDeleteParticipant("notes", spec, deletes=((1 << 63, 7),))
    segments = ze.CascadeDeleteParticipant("segments", spec)
    words = ze.CascadeDeleteParticipant("words", spec)
    with pytest.raises(ze.Io, match="missing cascade participant words"):
        ze.delete_cascade(tmp_path, [parent, segments])
    for name, doc_id in (("notes", parent_id), ("segments", 12), ("words", 24)):
        with ze.open_namespace(tmp_path, name, spec, access_mode="read_only") as store:
            assert store.get([doc_id, 42]).missing_count == 0

    generations = ze.delete_cascade(tmp_path, [words, parent, segments])
    assert len(generations) == 3
    for index, (name, doc_id) in enumerate((("words", 24), ("notes", parent_id), ("segments", 12))):
        with ze.open_namespace(tmp_path, name, spec, access_mode="read_only") as store:
            result = store.get([doc_id, 42])
            assert result.generation == generations[index]
            assert result.documents[0] is None
            assert result.documents[1] is not None
