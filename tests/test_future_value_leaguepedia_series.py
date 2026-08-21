from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY,
    LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION,
    FutureValueSourceError,
    _canonical_json_bytes,
    _leaguepedia_assignment_sha256,
    _map_model_frame,
    _roster_change_labels,
    bind_verified_leaguepedia_series_crosswalk,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _source_receipt(game_ids: list[str]) -> dict[str, object]:
    ids = sorted(game_ids)
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-01-05T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
        "model_eligible_game_count": len(ids),
        "model_eligible_identity_sha256": identity_sha256(ids),
        "model_eligible_game_ids": ids,
        "source_rows": {},
        "source_extra_game_ids": {},
        "identity_coverage": {},
        "checkpoint_coverage": {},
        "model_exclusions": {},
        "source_files": {
            label: {"bytes": 1, "sha256": "0" * 64, "locator": f"fixture/{label}"}
            for label in ("maps", "players", "teams", "accepted_census")
        },
        "model_contract": {},
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _write_crosswalk(
    tmp_path,
    source: dict[str, object],
    assignments: list[dict[str, object]],
) -> tuple[str, str, str]:
    accepted = list(source["accepted_game_ids"])
    series_rows: dict[str, dict[str, object]] = {}
    for row in assignments:
        series_id = str(row["series_id"])
        series_rows.setdefault(
            series_id,
            {
                "series_id": series_id,
                "oe_game_ids": [],
                "normalized_team_set": list(row["normalized_team_set"]),
            },
        )["oe_game_ids"].append(str(row["oe_game_id"]))
    for row in series_rows.values():
        row["oe_game_ids"] = sorted(row["oe_game_ids"])
    artifact: dict[str, object] = {
        "schema_version": "scryglass:oe-leaguepedia-series-crosswalk:v1",
        "captured_at": "2026-01-03T00:00:00Z",
        "status": "partial_authoritative_coverage",
        "authority": {
            "research_only": True,
            "public": False,
            "probability": False,
            "draft": False,
            "promotion": False,
            "deployment": False,
        },
        "source_binding": {
            "accepted_game_count": len(accepted),
            "accepted_game_identity_sha256": identity_sha256(accepted),
            "accepted_game_ids": accepted,
            "model_eligible_game_count": len(accepted),
            "model_eligible_game_identity_sha256": identity_sha256(accepted),
            "model_eligible_game_ids": accepted,
            "receipt_sha256": source["receipt_sha256"],
            "selected_game_count": len(accepted),
            "selected_game_identity_sha256": identity_sha256(accepted),
            "selected_game_ids": accepted,
            "selected_is_full_accepted_census": True,
        },
        "source_records": {},
        "competition_mapping": {},
        "raw_sources": {},
        "join_contract": {"outcome_used": False},
        "coverage": {
            "complete": False,
            "accepted_game_count": len(accepted),
            "mapped_game_count": len(assignments),
            "selected_game_count": len(accepted),
        },
        "assignments": assignments,
        "series": list(series_rows.values()),
        "issues": [],
    }
    artifact["crosswalk_sha256"] = hashlib.sha256(
        _canonical_json_bytes(artifact)
    ).hexdigest()
    artifact_path = tmp_path / "crosswalk.json"
    artifact_path.write_bytes(_canonical_json_bytes(artifact))
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    receipt: dict[str, object] = {
        "schema_version": LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION,
        "status": "verified_research_only",
        "authority": dict(LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY),
        "artifact": {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": artifact_sha256,
        },
        "crosswalk_sha256": artifact["crosswalk_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "accepted_game_count": len(accepted),
        "accepted_game_identity_sha256": identity_sha256(accepted),
        "assignment_count": len(assignments),
        "assignment_sha256": _leaguepedia_assignment_sha256(assignments),
        "mapped_game_count": len(assignments),
        "mapped_game_identity_sha256": identity_sha256(
            sorted(str(row["oe_game_id"]) for row in assignments)
        ),
        "mapped_game_ids": sorted(str(row["oe_game_id"]) for row in assignments),
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path = tmp_path / "crosswalk-receipt.json"
    receipt_path.write_bytes(_canonical_json_bytes(receipt))
    receipt_file_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return str(artifact_path), str(receipt_path), receipt_file_sha256


def _assignment(game_id: str, series_id: str = "series-a") -> dict[str, object]:
    return {
        "oe_game_id": game_id,
        "series_id": series_id,
        "normalized_team_set": ["blue", "red"],
        "outcome_used": False,
    }


def _maps(*game_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": f"2026-01-0{index + 1}T00:00:00Z",
                "y_blue_win": index % 2,
                "league": "LEC",
                "tournament": "Spring",
                "blue_team_key": "blue",
                "red_team_key": "red",
            }
            for index, game_id in enumerate(game_ids)
        ]
    )


def test_verified_crosswalk_promotes_only_complete_proxy_series(tmp_path) -> None:
    source = _source_receipt(["g1", "g2", "g3"])
    assignments = [_assignment("g1"), _assignment("g2")]
    artifact, receipt, receipt_sha256 = _write_crosswalk(tmp_path, source, assignments)
    maps = _maps("g1", "g2", "g3")
    bound = bind_verified_leaguepedia_series_crosswalk(
        maps,
        crosswalk_path=artifact,
        receipt_path=receipt,
        source_receipt=source,
        expected_receipt_file_sha256=receipt_sha256,
    )
    frame = _map_model_frame(
        bound,
        verified_source_receipt=source,
        verified_source_receipt_sha256=str(source["receipt_sha256"]),
        verified_crosswalk_receipt_file_sha256=receipt_sha256,
    )
    assert frame["series_id"].tolist() == [
        "proxy:lec|spring|blue|red",
        "proxy:lec|spring|blue|red",
        "proxy:lec|spring|blue|red",
    ]
    assert frame.attrs["series_cluster_source"] == (
        "mixed:leaguepedia_crosswalk+conservative_series_superset"
    )
    audit = frame.attrs["series_cluster_audit"]
    assert audit["authoritative"] is False
    assert audit["promoted_game_count"] == 0
    assert audit["partial_series_blocker"] is True


def test_verified_crosswalk_promotes_when_proxy_has_exact_series_membership(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    assignments = [_assignment("g1"), _assignment("g2")]
    artifact, receipt, receipt_sha256 = _write_crosswalk(tmp_path, source, assignments)
    bound = bind_verified_leaguepedia_series_crosswalk(
        _maps("g1", "g2"),
        crosswalk_path=artifact,
        receipt_path=receipt,
        source_receipt=source,
        expected_receipt_file_sha256=receipt_sha256,
    )
    frame = _map_model_frame(
        bound,
        verified_source_receipt=source,
        verified_source_receipt_sha256=str(source["receipt_sha256"]),
        verified_crosswalk_receipt_file_sha256=receipt_sha256,
    )
    assert frame["series_id"].tolist() == ["leaguepedia:series-a"] * 2
    assert frame.attrs["series_cluster_audit"]["promoted_game_count"] == 2


def test_verified_crosswalk_splits_fully_mapped_proxy_into_two_series(tmp_path) -> None:
    source = _source_receipt(["g1", "g2", "g3", "g4"])
    assignments = [
        _assignment("g1", "series-a"),
        _assignment("g2", "series-a"),
        _assignment("g3", "series-b"),
        _assignment("g4", "series-b"),
    ]
    artifact, receipt, receipt_sha256 = _write_crosswalk(tmp_path, source, assignments)
    bound = bind_verified_leaguepedia_series_crosswalk(
        _maps("g1", "g2", "g3", "g4"),
        crosswalk_path=artifact,
        receipt_path=receipt,
        source_receipt=source,
        expected_receipt_file_sha256=receipt_sha256,
    )
    frame = _map_model_frame(
        bound,
        verified_source_receipt=source,
        verified_source_receipt_sha256=str(source["receipt_sha256"]),
        verified_crosswalk_receipt_file_sha256=receipt_sha256,
    )
    assert frame["series_id"].tolist() == [
        "leaguepedia:series-a",
        "leaguepedia:series-a",
        "leaguepedia:series-b",
        "leaguepedia:series-b",
    ]
    assert frame.attrs["series_cluster_audit"]["promoted_series_count"] == 2


def test_verified_crosswalk_rejects_mutated_receipt_even_when_attrs_change(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    assignments = [_assignment("g1"), _assignment("g2")]
    artifact, receipt, receipt_sha256 = _write_crosswalk(tmp_path, source, assignments)
    bound = bind_verified_leaguepedia_series_crosswalk(
        _maps("g1", "g2"),
        crosswalk_path=artifact,
        receipt_path=receipt,
        source_receipt=source,
        expected_receipt_file_sha256=receipt_sha256,
    )
    receipt_payload = json.loads(open(receipt, encoding="utf-8").read())
    receipt_payload["mapped_game_count"] = 1
    receipt_payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes({key: value for key, value in receipt_payload.items() if key != "receipt_sha256"})
    ).hexdigest()
    with open(receipt, "wb") as handle:
        handle.write(_canonical_json_bytes(receipt_payload))
    attrs = dict(bound.attrs["verified_leaguepedia_series_crosswalk"])
    attrs["receipt_sha256"] = receipt_payload["receipt_sha256"]
    bound.attrs["verified_leaguepedia_series_crosswalk"] = attrs
    with pytest.raises(FutureValueSourceError, match="receipt file changed"):
        _map_model_frame(
            bound,
            verified_source_receipt=source,
            verified_source_receipt_sha256=str(source["receipt_sha256"]),
            verified_crosswalk_receipt_file_sha256=receipt_sha256,
        )


def test_verified_crosswalk_rejects_team_pair_drift(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    assignments = [_assignment("g1"), _assignment("g2")]
    artifact, receipt, receipt_sha256 = _write_crosswalk(tmp_path, source, assignments)
    bound = bind_verified_leaguepedia_series_crosswalk(
        _maps("g1", "g2"),
        crosswalk_path=artifact,
        receipt_path=receipt,
        source_receipt=source,
        expected_receipt_file_sha256=receipt_sha256,
    )
    mutated = bound.copy()
    mutated.attrs = dict(bound.attrs)
    mutated.loc[0, "blue_team_key"] = "different-team"
    with pytest.raises(FutureValueSourceError, match="team pair"):
        _map_model_frame(
            mutated,
            verified_source_receipt=source,
            verified_source_receipt_sha256=str(source["receipt_sha256"]),
            verified_crosswalk_receipt_file_sha256=receipt_sha256,
        )


def test_roster_labels_distinguish_missing_prior_roster_from_roster_change() -> None:
    frame = pd.DataFrame(
        {
            "blue_roster_continuity": [None, 1.0, 0.0, None],
            "red_roster_continuity": [None, 1.0, 0.5, 0.5],
        }
    )
    labels = _roster_change_labels(frame)
    assert labels is not None
    assert labels.tolist() == [
        "prior_roster_unavailable",
        "stable_roster",
        "roster_change",
        "prior_roster_unavailable",
    ]
    assert _roster_change_labels(
        pd.DataFrame(
            {
                "blue_roster_continuity": [None],
                "red_roster_continuity": [None],
            }
        )
    ) is None
