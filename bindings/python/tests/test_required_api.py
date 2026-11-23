"""Require the E17 surface through the installed package's public exports."""

import inspect
import json
from pathlib import Path

import zeppelin_embed as api


def _has_public_api(path: str) -> bool:
    name, _, parameter = path.partition(":")
    parts = name.split(".")
    if parts[0] not in api.__all__ or any(part.startswith("_") for part in parts):
        return False
    value = api
    for part in parts:
        try:
            value = getattr(value, part)
        except AttributeError:
            return False
    return not parameter or parameter in inspect.signature(value).parameters


def test_required_api():
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "required_api_v1.json"
    table = json.loads(fixture.read_text(encoding="utf-8"))
    expected_rows = {
        "core": "S1 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S19 S20 S21 N1 N2 N3 N4",
        "text": "E1 E2 T1 T2 T3 T4 T5 T6 T7",
        "graph": "G1 G2 G3 G4 G5 G6 G7",
        "survey": "S2 S3 S4 S18 A1",
    }
    assert {row_id: row["build"] for row_id, row in table["rows"].items()} == {
        row_id: build for build, ids in expected_rows.items() for row_id in ids.split()
    }
    # The E17 implementation tickets are on this base. The gap ceiling is zero.
    assert table["allowed_gaps"] == {"node": {}, "python": {}, "swift": {}}
    assert set(table["build_exclusions"]) == {"python"}
    excluded = table["build_exclusions"]["python"]
    assert set(excluded) == {"graph"} and excluded["graph"]

    missing = []
    for row_id, row in table["rows"].items():
        if row["build"] == "survey" or row["build"] in excluded:
            continue
        names = row["python"]
        if not names:
            missing.append(f"{row_id}: no Python API specified")
        missing.extend(f"{row_id}: {name}" for name in names if not _has_public_api(name))
    assert missing == []
