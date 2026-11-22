"""End-to-end tests for offline store verification."""

from pathlib import Path

import numpy as np
import pytest

import zeppelin_embed as ze

WAL_HEADER_LEN = 40


def _sealed_store(path: Path) -> None:
    vector = np.asarray([[1.0, 0.5]], dtype=np.float32)
    with ze.open(path) as store:
        store.ingest([1], vector, texts=["sealed note"])
        store.seal()
        store.ingest([2], vector, texts=["WAL note"])


def test_verify_clean_store(tmp_path: Path) -> None:
    path = tmp_path / "store-é"
    _sealed_store(path)

    report = ze.verify(path)

    assert isinstance(report, ze.VerifyReport)
    assert report.ok is True
    assert report.generation > 0
    assert report.segments_checked == 1
    assert report.wal_records_checked == 1
    assert report.findings == ()


def test_verify_reports_damage_without_modifying_store(tmp_path: Path) -> None:
    path = tmp_path / "store"
    _sealed_store(path)
    (segment,) = path.glob("*.zseg")
    segment.unlink()
    wal = path / "wal.ze"
    data = bytearray(wal.read_bytes())
    data[WAL_HEADER_LEN + 20] ^= 0x5A
    wal.write_bytes(data)
    before = {file.name: file.read_bytes() for file in path.iterdir()}

    report = ze.verify(str(path))

    assert report.ok is False
    assert report.segments_checked == 1
    assert report.wal_records_checked == 0
    segment_finding, wal_finding = report.findings
    assert isinstance(segment_finding, ze.VerifyFinding)
    assert segment_finding.kind is ze.VerifyFindingKind.SEGMENT_MISSING
    assert segment_finding.file == segment.name
    assert segment_finding.offset is None
    assert segment_finding.detail
    assert wal_finding.kind is ze.VerifyFindingKind.WAL_RECORD_CORRUPT
    assert wal_finding.file == "wal.ze"
    assert wal_finding.offset == WAL_HEADER_LEN
    assert wal_finding.detail
    assert {file.name: file.read_bytes() for file in path.iterdir()} == before


def test_verify_missing_store_raises_not_found(tmp_path: Path) -> None:
    path = tmp_path / "missing"

    with pytest.raises(ze.NotFound) as caught:
        ze.verify(path)

    assert caught.value.code == ze.ErrorCode.ERR_NOT_FOUND
    assert caught.value.code_name == "ZE_ERR_NOT_FOUND"
    assert "missing" in caught.value.message
    assert not path.exists()
