from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from lol_kills.v2.data.common import canonical_json_bytes, sha256_bytes
from lol_kills.v2.ratings.player.real_v1_adapter import (
    ACCEPTED_G1_PINS,
    AcceptedG1Pins,
    PrivatePlayerRatingAdapterError,
    build_private_player_rating_input,
    load_accepted_lpl_private_player_rating_input,
)


ROOT = Path(__file__).parents[4]
MANIFEST_PATH = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
ROWS_PATH = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"


def _payloads() -> tuple[dict, list[dict]]:
    return (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()],
    )


def _rebind(manifest: dict, rows: list[dict]) -> tuple[dict, AcceptedG1Pins, str]:
    """Create an in-memory pin set so hostile semantic tests reach the adapter."""

    row_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest = deepcopy(manifest)
    manifest["rows_sha256"] = sha256_bytes(row_bytes)
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    pins = replace(
        ACCEPTED_G1_PINS,
        manifest_sha256=manifest["manifest_sha256"],
        rows_sha256=manifest["rows_sha256"],
    )
    return manifest, pins, manifest["rows_sha256"]


def test_loads_exact_accepted_g1_snapshot_as_private_typed_input() -> None:
    handoff = load_accepted_lpl_private_player_rating_input()

    assert handoff.manifest_sha256 == ACCEPTED_G1_PINS.manifest_sha256
    assert handoff.rows_sha256 == ACCEPTED_G1_PINS.rows_sha256
    assert handoff.map_count == 1226
    assert handoff.player_observation_count == 12260
    assert [(fold.fold_id, len(fold.map_observations)) for fold in handoff.folds] == [
        ("TRAIN", 805), ("DEVELOPMENT", 214), ("VALIDATION", 207),
    ]
    assert handoff.claim_ceiling["private_development_model_fit"] is True
    assert handoff.claim_ceiling["prediction"] is False
    assert handoff.claim_ceiling["publication"] is False


def test_manifest_self_hash_is_bound_before_player_handoff() -> None:
    manifest, rows = _payloads()
    manifest["coverage"]["map_count"] += 1

    with pytest.raises(PrivatePlayerRatingAdapterError, match="self hash"):
        build_private_player_rating_input(
            manifest, rows, pins=ACCEPTED_G1_PINS, rows_sha256=ACCEPTED_G1_PINS.rows_sha256,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "same_series"))
def test_ordered_frozen_origin_identity_is_fail_closed(mutation: str) -> None:
    manifest, rows = _payloads()
    target = next(row for row in rows if len(row["eligible_prior_origin_map_ids"]) >= 2)
    origins = target["eligible_prior_origin_map_ids"]
    if mutation == "missing":
        target["eligible_prior_origin_map_ids"] = origins[1:]
    elif mutation == "extra":
        target["eligible_prior_origin_map_ids"] = [*origins, "invented-origin"]
    elif mutation == "reordered":
        target["eligible_prior_origin_map_ids"] = list(reversed(origins))
    else:
        same_series = next(
            row["source_game_id"] for row in rows
            if row["source_series_id"] == target["source_series_id"]
            and row["source_game_id"] != target["source_game_id"]
        )
        target["eligible_prior_origin_map_ids"] = [*origins, same_series]
    target["eligible_prior_origin_count"] = len(target["eligible_prior_origin_map_ids"])
    manifest, pins, rows_sha256 = _rebind(manifest, rows)

    with pytest.raises(PrivatePlayerRatingAdapterError, match="ordered origins"):
        build_private_player_rating_input(manifest, rows, pins=pins, rows_sha256=rows_sha256)


@pytest.mark.parametrize("mutation", ("final", "non_binary", "target_source", "duplicate_player", "duplicate_team"))
def test_invalid_target_or_ambiguous_observed_identity_is_fail_closed(mutation: str) -> None:
    manifest, rows = _payloads()
    target = rows[0]
    if mutation == "final":
        target["partition"] = "FINAL_TEMPORAL_HOLDOUT"
    elif mutation == "non_binary":
        target["target"]["y_blue_win"] = 2
    elif mutation == "target_source":
        target["target"]["source_red_result_id"] = "oe-team-row:wrong:200"
    elif mutation == "duplicate_player":
        target["observed_lineups"][1]["player_ids_by_role"]["top"] = target["observed_lineups"][0]["player_ids_by_role"]["top"]
    else:
        target["observed_lineups"][1]["team_id"] = target["observed_lineups"][0]["team_id"]
    manifest, pins, rows_sha256 = _rebind(manifest, rows)

    expected = "final or unknown split|binary|target/source|ambiguous"
    with pytest.raises(PrivatePlayerRatingAdapterError, match=expected):
        build_private_player_rating_input(manifest, rows, pins=pins, rows_sha256=rows_sha256)
