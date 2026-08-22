from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.future_value_training import (
    DUPLICATE_RESOLUTION_SCHEMA_VERSION,
    FutureValueTrainingError,
    KNOWN_DUPLICATE_BRIDGE_GAME_IDS,
    duplicate_resolution_mapping_sha256,
    validate_duplicate_resolution_block,
)


BRIDGE_ID = "oe:game:bridge-1"
SURVIVOR_ID = "annual-1"
SEMANTIC_FIELDS = [
    "date",
    "league",
    "tournament",
    "patch",
    "blue_team",
    "red_team",
    "blue_team_key",
    "red_team_key",
    "y_blue_win",
]


def _maps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": BRIDGE_ID,
                "date": pd.Timestamp("2026-08-20T10:00:00Z"),
                "league": "LEC",
                "tournament": "LEC 2026",
                "patch": "16.1",
                "blue_team": "Blue Team",
                "red_team": "Red Team",
                "blue_team_key": "blue-team",
                "red_team_key": "red-team",
                "y_blue_win": 1,
            },
            {
                "game_uid": SURVIVOR_ID,
                "date": pd.Timestamp("2026-08-20T10:00:00Z"),
                "league": "LEC",
                "tournament": "LEC 2026",
                "patch": "16.1",
                "blue_team": "Blue Team",
                "red_team": "Red Team",
                "blue_team_key": "blue-team",
                "red_team_key": "red-team",
                "y_blue_win": 1,
            },
        ]
    )


def _row_payload(game_id: str) -> dict[str, object]:
    return {
        "game_uid": game_id,
        "date": "2026-08-20T10:00:00Z",
        "league": "LEC",
        "tournament": "LEC 2026",
        "patch": "16.1",
        "blue_team": "Blue Team",
        "red_team": "Red Team",
        "blue_team_key": "blue-team",
        "red_team_key": "red-team",
        "y_blue_win": 1,
    }


def _freeze(tmp_path: Path) -> dict[str, object]:
    mapping = {
        "bridge_game_id": BRIDGE_ID,
        "annual_survivor_game_id": SURVIVOR_ID,
        "bridge_source_row": _row_payload(BRIDGE_ID),
        "annual_survivor_source_row": _row_payload(SURVIVOR_ID),
        "evidence": {
            "semantic_fields": SEMANTIC_FIELDS,
            "external_identity": {
                "scoreboard_game_id": "scoreboard-match-1_1",
                "scoreboard_riot_platform_game_id": "LOLTMNT01_1",
            },
            "field_values": {
                "date": "2026-08-20T10:00:00Z",
                "league": "LEC",
                "tournament": "LEC 2026",
                "patch": "16.1",
                "blue_team": "Blue Team",
                "red_team": "Red Team",
                "blue_team_key": "blue-team",
                "red_team_key": "red-team",
                "y_blue_win": 1,
            },
        },
        "survivor_rule": "annual_row_is_survivor_exact_semantic_match",
    }
    artifact = {
        "schema_version": "scryglass:duplicate-audit:v1",
        "assignments": [
            {
                "bridge_game_id": BRIDGE_ID,
                "scoreboard_game_id": "scoreboard-match-1_1",
                "scoreboard_riot_platform_game_id": "LOLTMNT01_1",
            },
            {
                "annual_survivor_game_id": SURVIVOR_ID,
                "scoreboard_game_id": "scoreboard-match-1_1",
                "scoreboard_riot_platform_game_id": "LOLTMNT01_1",
            },
        ],
    }
    artifact["crosswalk_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "crosswalk_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact_path = tmp_path / "duplicate-audit.json"
    artifact_bytes = json.dumps(
        artifact, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    artifact_path.write_bytes(artifact_bytes)
    artifact_record = {
        "path": str(artifact_path),
        "bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    receipt = {
        "schema_version": "scryglass:duplicate-audit-receipt:v1",
        "artifact": artifact_record,
        "source_identity_sha256": "a" * 64,
        "authority": {"research_only": True, "promotion": False},
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path = tmp_path / "duplicate-audit.receipt.json"
    receipt_bytes = json.dumps(
        receipt, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    block = {
        "schema_version": DUPLICATE_RESOLUTION_SCHEMA_VERSION,
        "survivor_rule": "annual_row_is_survivor_exact_semantic_match",
        "mappings": [mapping],
        "source_binding": {
            "kind": "duplicate_audit",
            "artifact": artifact_record,
            "receipt": {
                "path": str(receipt_path),
                "bytes": len(receipt_bytes),
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            },
            "expected_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "source_identity_sha256": "a" * 64,
        },
    }
    block["mapping_sha256"] = duplicate_resolution_mapping_sha256(
        block["mappings"]
    )
    return {
        "schema_version": "scryglass:future-value-source-freeze:v2",
        "accepted_census": {
            "excluded_game_ids": [BRIDGE_ID],
            "source_game_count": 1,
            "source_identity_sha256": "a" * 64,
        },
        "duplicate_resolution_required_bridge_game_ids": [BRIDGE_ID],
        "duplicate_resolution": block,
    }


def test_duplicate_resolution_binds_rows_and_digest(tmp_path: Path) -> None:
    result = validate_duplicate_resolution_block(_maps(), _freeze(tmp_path))
    assert result is not None
    assert result["mapping_count"] == 1
    assert result["bridge_game_ids"] == [BRIDGE_ID]
    assert result["annual_survivor_game_ids"] == [SURVIVOR_ID]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda freeze: freeze["duplicate_resolution"]["mapping_sha256"].__class__(
                "0" * 64
            ),
            "mapping digest",
        ),
    ],
)
def test_duplicate_resolution_rejects_digest_mutation(tmp_path: Path, mutator, match: str) -> None:
    freeze = _freeze(tmp_path)
    freeze["duplicate_resolution"]["mapping_sha256"] = mutator(freeze)
    with pytest.raises(FutureValueTrainingError, match=match):
        validate_duplicate_resolution_block(_maps(), freeze)


def test_duplicate_resolution_rejects_semantic_row_mutation(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    freeze["duplicate_resolution"]["mappings"][0]["evidence"]["field_values"][
        "red_team"
    ] = "Changed Team"
    freeze["duplicate_resolution"]["mapping_sha256"] = duplicate_resolution_mapping_sha256(
        freeze["duplicate_resolution"]["mappings"]
    )
    with pytest.raises(FutureValueTrainingError, match="semantic field|evidence field"):
        validate_duplicate_resolution_block(_maps(), freeze)


def test_duplicate_resolution_requires_excluded_bridge_and_accepted_survivor(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    freeze["accepted_census"]["excluded_game_ids"] = []
    with pytest.raises(FutureValueTrainingError, match="not excluded"):
        validate_duplicate_resolution_block(_maps(), freeze)

    freeze = _freeze(tmp_path)
    freeze["accepted_census"]["excluded_game_ids"] = [BRIDGE_ID, SURVIVOR_ID]
    with pytest.raises(FutureValueTrainingError, match="survivor is excluded"):
        validate_duplicate_resolution_block(_maps(), freeze)


def test_duplicate_resolution_requires_both_raw_map_ids(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    with pytest.raises(FutureValueTrainingError, match="missing from raw maps"):
        validate_duplicate_resolution_block(_maps().iloc[[0]], freeze)


def test_duplicate_resolution_requires_source_bound_identity_artifacts(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    freeze["duplicate_resolution"].pop("source_binding")
    freeze["duplicate_resolution"]["mapping_sha256"] = duplicate_resolution_mapping_sha256(
        freeze["duplicate_resolution"]["mappings"]
    )
    with pytest.raises(FutureValueTrainingError, match="source binding"):
        validate_duplicate_resolution_block(_maps(), freeze)


def test_duplicate_resolution_rejects_changed_identity_artifact(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    artifact_path = Path(
        freeze["duplicate_resolution"]["source_binding"]["artifact"]["path"]
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"changed")
    with pytest.raises(FutureValueTrainingError, match="bytes changed"):
        validate_duplicate_resolution_block(_maps(), freeze)


def test_duplicate_mapping_digest_normalizes_parquet_scalars(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    mapping = freeze["duplicate_resolution"]["mappings"][0]
    mapping["bridge_source_row"]["nullable_source_value"] = pd.NA
    mapping["annual_survivor_source_row"]["nullable_source_value"] = pd.NA
    mapping["bridge_source_row"]["source_timestamp"] = pd.Timestamp(
        "2026-08-20T10:00:00Z"
    )
    mapping["annual_survivor_source_row"]["source_timestamp"] = pd.Timestamp(
        "2026-08-20T10:00:00Z"
    )
    freeze["duplicate_resolution"]["mapping_sha256"] = duplicate_resolution_mapping_sha256(
        freeze["duplicate_resolution"]["mappings"]
    )
    result = validate_duplicate_resolution_block(_maps(), freeze)
    assert result is not None


def test_old_freeze_without_block_stays_valid_until_it_excludes_known_bridge() -> None:
    maps = _maps().rename(index={0: 0, 1: 1})
    old_freeze = {
        "schema_version": "scryglass:future-value-source-freeze:v1",
        "accepted_census": {
            "excluded_game_ids": [sorted(KNOWN_DUPLICATE_BRIDGE_GAME_IDS)[0]],
            "source_game_count": 1,
            "source_identity_sha256": "a" * 64,
        },
    }
    with pytest.raises(FutureValueTrainingError, match="block is required"):
        validate_duplicate_resolution_block(maps, old_freeze)

    old_freeze["accepted_census"]["excluded_game_ids"] = ["other-bridge"]
    assert validate_duplicate_resolution_block(maps, old_freeze) is None
