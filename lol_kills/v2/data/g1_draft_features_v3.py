"""Outcome-free completed-draft snapshot re-bound to the v3 Player identity.

The v1 feature snapshot is historical evidence and remains immutable.  Its
transform consumes only participant/champion fields, not Player posterior
values, so this module replays the same transform into a new versioned pair
whose membership-origin metadata points at the corrected v3 Player artifact.
It grants no new target, prediction, roster, publication, or SOTA authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.v2.data import g1_draft_features as base


ROOT = base.ROOT
OUTPUT_ROWS = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-rows.jsonl"
OUTPUT_MANIFEST = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-manifest.json"
SCHEMA = "scryglass:g1-lpl-completed-draft-features:v2"
G2_ARTIFACT = ROOT / "data/lol/v2/models/player/real-v1/private-development-artifact-v3.json"
G2_ARTIFACT_RAW_SHA256 = "11fb9a43c6c2bb50d9c6046eb8e0fbbed3755607518bc483b9ad82bb556568e7"
G2_ARTIFACT_CANONICAL_SHA256 = "510d2cde52a92f92f6aa373bbe5c497d2b9dc652d1f7edf15f9cae006ee0f7a0"


class G1DraftFeatureV3Error(ValueError):
    """The v3 outcome-free feature replay failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G1DraftFeatureV3Error("noncanonical v3 feature payload") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _g2_origin_v3() -> dict[str, Any]:
    path = base._expect_raw(G2_ARTIFACT, G2_ARTIFACT_RAW_SHA256, label="v3 G2 origin")
    payload = json.loads(path.read_bytes())
    unsigned = dict(payload)
    if unsigned.pop("artifact_sha256", None) != G2_ARTIFACT_CANONICAL_SHA256 or _sha256(unsigned) != G2_ARTIFACT_CANONICAL_SHA256:
        raise G1DraftFeatureV3Error("v3 G2 origin canonical identity mismatch")
    if payload.get("schema_version") != "scryglass:player-real-v1-private-development:v3":
        raise G1DraftFeatureV3Error("v3 G2 origin schema mismatch")
    pins = payload.get("adapter_input_pins")
    if not isinstance(pins, Mapping) or pins.get("fold_map_digests") != base.G2_FOLD_MAP_DIGESTS or pins.get("fold_origin_digests") != base.G2_FOLD_ORIGIN_DIGESTS:
        raise G1DraftFeatureV3Error("v3 G2 origin fold identity mismatch")
    return {
        "g2_artifact_locator": str(path.relative_to(ROOT)),
        "g2_artifact_raw_sha256": G2_ARTIFACT_RAW_SHA256,
        "g2_artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
        "g2_schema_version": payload["schema_version"],
        "fold_map_digests": base.G2_FOLD_MAP_DIGESTS,
        "fold_origin_digests": base.G2_FOLD_ORIGIN_DIGESTS,
    }


def _payload() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = base._base_rows()
    player_games_path = base._expect_raw(base.PLAYER_GAMES, base.PLAYER_GAMES_SHA256, label="OE player-games")
    frame = pd.read_parquet(player_games_path, columns=list(base.SOURCE_COLUMNS))
    selected = frame.loc[
        frame["gameid"].astype(str).isin({row["source_game_id"] for row in base_rows}),
        list(base.SOURCE_COLUMNS),
    ]
    source_rows = selected.to_dict("records")
    if len(source_rows) != 12260:
        raise G1DraftFeatureV3Error("accepted source participant row count drift")
    table, crosswalk_identity = base._crosswalk_table(str(row["champion"]) for row in source_rows)
    rows, projection = base.materialize_from_projection(base_rows=base_rows, source_rows=source_rows, champion_table=table)
    if len(rows) != 1226 or sum(len(row["picks"]) for row in rows) != 12260:
        raise G1DraftFeatureV3Error("completed-draft row/pick count drift")
    row_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "rows_locator": str(OUTPUT_ROWS.relative_to(ROOT)),
        "rows_raw_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "rows_canonical_sha256": _sha256(rows),
        "coverage": {
            "accepted_map_count": 1226,
            "feature_row_count": len(rows),
            "pick_count": 12260,
            "identity_unavailable_map_count": sum(row["availability"] != "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START" for row in rows),
        },
        "source": {
            "locator": str(player_games_path.relative_to(ROOT)),
            "raw_sha256": base.PLAYER_GAMES_SHA256,
            "rights_status": "PRIVATE_REVIEWED",
            "selected_columns": list(base.SOURCE_COLUMNS),
            "selected_projection_row_count": len(projection),
            "selected_projection_canonical_sha256": _sha256(projection),
            "selected_projection_sort": list(base.SOURCE_COLUMNS),
        },
        "base_g1": {
            "rows_locator": str(base.BASE_ROWS.relative_to(ROOT)),
            "rows_raw_sha256": base.BASE_ROWS_SHA256,
            "manifest_locator": str(base.BASE_MANIFEST.relative_to(ROOT)),
            "manifest_raw_sha256": base.BASE_MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": base.BASE_MANIFEST_SHA256,
            "selected_target_sha256": base.SELECTED_TARGET_SHA256,
            "split_payload_sha256": base.SPLIT_PAYLOAD_SHA256,
            "target_authority_raw_sha256": base.TARGET_AUTHORITY_RAW_SHA256,
            "target_evidence_canonical_sha256": base.TARGET_EVIDENCE_SHA256,
        },
        "accepted_membership_origin": _g2_origin_v3(),
        "upstream_rebind": {
            "status": "OUTCOME_FREE_FEATURE_REPLAY_WITH_V3_IDENTITY",
            "old_v1_g2_artifact_canonical_sha256": base.G2_ARTIFACT_CANONICAL_SHA256,
            "new_v3_g2_artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
            "player_posterior_values_consumed_by_transform": False,
            "does_not_authorize": ["prediction", "production", "publication", "promotion", "sota", "final_holdout"],
        },
        "champion_crosswalk": crosswalk_identity,
        "transform": {
            "locator": "lol_kills/v2/data/g1_draft_features.py",
            "raw_sha256": base.raw_sha256(Path(base.__file__)),
            "config": base.TRANSFORM_CONFIG,
            "config_sha256": base.sha256(base.TRANSFORM_CONFIG),
        },
        "availability": {
            "kind": "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START",
            "event_time": "source_local_event_start_from_accepted_G1_only",
            "limitation": "mechanically known by game start; the retrospective snapshot supplies no historical-ingest current live or forecast authority",
            "not_authorized": ["historical_ingest", "current", "live", "forecast", "prediction"],
        },
        "final_holdout": {"status": "SEALED_UNREAD", "accessed": False, "included": False},
        "claim_ceiling": {
            "private_model_fit": True,
            "private_rank_selection": True,
            "prediction": False,
            "publication": False,
            "promotion": False,
            "sota": False,
            "final_holdout": False,
            "public_pack": False,
        },
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return rows, manifest


def build(*, rows_path: Path = OUTPUT_ROWS, manifest_path: Path = OUTPUT_MANIFEST) -> dict[str, Any]:
    rows, manifest = _payload()
    row_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    base._safe_write_many(((rows_path, row_bytes), (manifest_path, _canonical_bytes(manifest) + b"\n")))
    return manifest


def verify(*, rows_path: Path = OUTPUT_ROWS, manifest_path: Path = OUTPUT_MANIFEST, expected_manifest_sha256: str) -> dict[str, Any]:
    try:
        rows_path = base._safe_file(rows_path, label="v3 draft feature rows")
        manifest_path = base._safe_file(manifest_path, label="v3 draft feature manifest")
    except base.G1DraftFeatureError as error:
        raise G1DraftFeatureV3Error("v3 feature path failed closed") from error
    raw_manifest = manifest_path.read_bytes()
    if raw_manifest != _canonical_bytes(json.loads(raw_manifest)) + b"\n":
        raise G1DraftFeatureV3Error("v3 manifest is not canonical")
    manifest = json.loads(raw_manifest)
    unsigned = dict(manifest)
    if unsigned.pop("manifest_sha256", None) != expected_manifest_sha256 or _sha256(unsigned) != expected_manifest_sha256:
        raise G1DraftFeatureV3Error("v3 manifest identity mismatch")
    rows, replayed_manifest = _payload()
    persisted_rows = [json.loads(line) for line in rows_path.read_bytes().splitlines()]
    if persisted_rows != rows or manifest != replayed_manifest:
        raise G1DraftFeatureV3Error("v3 draft feature replay mismatch")
    if manifest.get("final_holdout") != {"status": "SEALED_UNREAD", "accessed": False, "included": False}:
        raise G1DraftFeatureV3Error("v3 final holdout boundary mismatch")
    if any("target" in row for row in persisted_rows):
        raise G1DraftFeatureV3Error("v3 feature row contains outcome data")
    return manifest
