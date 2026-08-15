from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.v2.champions.atoms.depth2_aggregate import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_LCC_REPO,
    FEATURE_KEYS,
    SOURCE_SCHEMA,
    Depth2AggregateError,
    build_depth2_payload,
    load_depth2_artifact,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_checked_in_depth2_index_is_hash_bound_and_complete() -> None:
    artifact = _artifact()
    index = load_depth2_artifact()

    assert artifact["development_only"] is True
    assert artifact["public_patch"] == "26.16"
    assert artifact["client_patch"] == "16.16"
    assert len(index) == 166
    assert len(FEATURE_KEYS) == 23
    assert set(index["galio"]) == set(FEATURE_KEYS)
    assert len(artifact["provenance"]["file_sha256"]) == 167
    assert SOURCE_SCHEMA in artifact["provenance"]["file_sha256"]


def test_depth2_index_rejects_changed_value(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["champions"]["galio"]["d2_burst"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(Depth2AggregateError, match="sha256 changed"):
        load_depth2_artifact(path)


@pytest.mark.skipif(
    not (DEFAULT_LCC_REPO / SOURCE_SCHEMA).is_file(),
    reason="private LCC depth-2 corpus is not mounted",
)
def test_depth2_index_rebuilds_from_pinned_lcc_sources() -> None:
    existing = _artifact()
    rebuilt = build_depth2_payload(
        DEFAULT_LCC_REPO,
        generated_at=str(existing["generated_at"]),
    )
    existing_unsigned = dict(existing)
    existing_unsigned.pop("artifact_sha256")
    rebuilt_provenance = dict(rebuilt["provenance"])
    existing_provenance = dict(existing_unsigned["provenance"])
    for field in ("lcc_commit",):
        rebuilt_provenance.pop(field)
        existing_provenance.pop(field)
    rebuilt["provenance"] = rebuilt_provenance
    existing_unsigned["provenance"] = existing_provenance

    assert rebuilt == existing_unsigned
