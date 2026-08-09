from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lol_kills.live_totals_model as live_totals
import pytest


def rows(series_count: int = 40) -> list[dict]:
    output = []
    for index in range(series_count):
        output.append(
            {
                "game_id": f"game-{index}",
                "series_id": f"series-{index}",
                "date": f"2026-01-{1 + index // 2:02d}T{index % 2:02d}:00:00",
                "league": "LCK",
                "patch": "16.01",
                "blue_team": "Blue",
                "red_team": "Red",
                "champions": list("ABCDEFGHIJ"),
                "total_kills": 25.0 + index % 5,
                "checkpoints": {
                    "10": {"current_kills": 5.0, "gold_difference": 500.0},
                    "15": {"current_kills": 9.0, "gold_difference": 800.0},
                },
            }
        )
    return output


def supported_payload() -> dict:
    residuals = [-6.0, -2.0, 0.0, 2.0, 6.0]
    return {
        "meta": {
            "data_cutoff_by_league": {"LCK": "2026-07-25T00:00:00+00:00"},
        },
        "model": {
            "families": ["gold_difference"],
            "champions": [],
            "feature_names": [
                "absolute_gold_difference",
                "checkpoint:10",
            ],
            "centers": {
                "absolute_gold_difference": 1000.0,
                "checkpoint:10": 0.0,
            },
            "scales": {
                "absolute_gold_difference": 500.0,
                "checkpoint:10": 1.0,
            },
            "coefficients": [12.0, -1.0, 0.0],
        },
        "windows": {"10": {"LCK": {"status": "supported"}}},
        "test_patch_counts": {"10": {"LCK": {"16.14": 30}}},
        "calibration_residuals": {
            "10": {"LCK": residuals},
        },
        "calibration_residual_clusters": {
            "10": {
                "LCK": {
                    "series_n": len(residuals),
                    "games_n": len(residuals),
                    "clusters": [
                        {
                            "series_id_sha256": f"{index:x}" * 64,
                            "residuals": [residual],
                        }
                        for index, residual in enumerate(residuals, start=1)
                    ],
                }
            }
        },
        "runtime_priors": {
            "league_median": {"LCK": 28.0},
            "teams": {
                "Blue": {"n": 20, "median": 28.0},
                "Red": {"n": 20, "median": 30.0},
            },
            "pairs": {},
        },
    }


def test_split_is_chronological_and_series_disjoint() -> None:
    split = live_totals.split_series(rows())
    identities = {
        name: {row["series_id"] for row in part} for name, part in split.items()
    }
    names = list(identities)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            assert identities[left].isdisjoint(identities[right])
    assert max(row["date"] for row in split["train"]) <= min(
        row["date"] for row in split["selection"]
    )
    assert max(row["date"] for row in split["selection"]) <= min(
        row["date"] for row in split["calibration"]
    )
    assert max(row["date"] for row in split["calibration"]) <= min(
        row["date"] for row in split["test"]
    )


def test_pregame_priors_do_not_read_other_games_in_same_series() -> None:
    source = rows(2)
    source[0]["series_id"] = "shared"
    source[1]["series_id"] = "shared"
    source[0]["total_kills"] = 5.0
    source[1]["total_kills"] = 80.0
    enriched = live_totals.attach_pregame_priors(source)
    assert enriched[0]["team_pace_median"] == enriched[1]["team_pace_median"]
    assert enriched[0]["h2h_n"] == enriched[1]["h2h_n"] == 0


def test_22_minutes_is_not_inferred_from_a_nearby_checkpoint() -> None:
    result = live_totals.price_live_totals(
        supported_payload(),
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=22,
        current_kills=14,
        gold_difference=1200.0,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        lines=[{"line": 32.5, "under_odds": 1.8}],
    )
    assert result["eligibility"]["status"] == "unavailable"
    assert "minute_not_validated:22" in result["eligibility"]["blockers"]
    assert result["projected_mean"] is None
    assert result["lines"][0]["under_probability"] is None
    assert result["lines"][0]["under_edge_pp"] is None


def test_missing_selected_gold_feature_fails_closed() -> None:
    eligibility = live_totals.runtime_eligibility(
        supported_payload(),
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=None,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    assert eligibility["status"] == "unavailable"
    assert "gold_difference_missing" in eligibility["blockers"]


def test_stale_or_unseen_patch_fails_closed() -> None:
    stale = live_totals.runtime_eligibility(
        supported_payload(),
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.14",
        as_of=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    assert "data_stale" in stale["blockers"]
    unseen = live_totals.runtime_eligibility(
        supported_payload(),
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.15",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    assert "exact_patch_holdout_unavailable:16.15" in unseen["blockers"]


def test_collapsed_decimal_patch_is_normalized_to_trailing_zero() -> None:
    assert live_totals._normalize_patch("16.1") == "16.10"
    assert live_totals._normalize_patch("16.14") == "16.14"


def test_supported_checkpoint_prices_from_heldout_residuals(monkeypatch) -> None:
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_GAMES", 5)
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_SERIES", 5)
    result = live_totals.price_live_totals(
        supported_payload(),
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        lines=[{"line": 32.5, "under_odds": 1.8}],
    )
    assert result["eligibility"]["status"] == "supported"
    assert result["projected_mean"] is not None
    assert result["lines"][0]["under_probability"] is not None
    assert result["lines"][0]["under_probability_interval"] is not None
    assert result["lines"][0]["classification"] == "WITHHELD"
    assert result["lines"][0]["under_edge_pp"] is None
    assert result["lines"][0]["under_expected_return"] is None
    assert result["uncertainty"]["status"] == "available"


def test_old_artifact_keeps_point_diagnostic_but_has_no_dependence_interval(
    monkeypatch,
) -> None:
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_GAMES", 5)
    payload = copy.deepcopy(supported_payload())
    payload.pop("calibration_residual_clusters")
    result = live_totals.price_live_totals(
        payload,
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        lines=[{"line": 32.5, "under_odds": 8.0}],
    )
    assert result["lines"][0]["under_probability"] is not None
    assert result["lines"][0]["under_probability_interval"] is None
    assert result["lines"][0]["classification"] == "WITHHELD"
    assert result["lines"][0]["under_expected_return"] is None
    assert result["uncertainty"] == {
        "status": "unavailable",
        "method": "series_cluster_weighted_hoeffding",
        "confidence": 0.95,
        "blockers": ["series_cluster_calibration_missing"],
    }


def test_repeating_maps_within_series_does_not_inflate_effective_support() -> None:
    independent = live_totals._series_cluster_cdf(
        [[-1.0], [1.0]] * 10,
        0.0,
        minimum_series=20,
    )
    repeated_within_series = live_totals._series_cluster_cdf(
        [[-1.0] * 5, [1.0] * 5] * 10,
        0.0,
        minimum_series=20,
    )
    assert independent["status"] == "available"
    assert repeated_within_series["status"] == "available"
    assert independent["effective_series_n"] == 20.0
    assert repeated_within_series["effective_series_n"] == 20.0
    assert independent["interval"] == repeated_within_series["interval"]


def test_malformed_cluster_artifact_fails_interval_closed(monkeypatch) -> None:
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_GAMES", 5)
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_SERIES", 5)
    payload = copy.deepcopy(supported_payload())
    payload["calibration_residual_clusters"]["10"]["LCK"]["clusters"][0][
        "residuals"
    ] = ["not-a-number"]
    result = live_totals.price_live_totals(
        payload,
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        lines=[{"line": 32.5}],
    )
    assert result["lines"][0]["under_probability"] is not None
    assert result["lines"][0]["under_probability_interval"] is None
    assert "series_cluster_record_invalid" in result["uncertainty"]["blockers"]


def test_missing_flat_residuals_withholds_probability(monkeypatch) -> None:
    monkeypatch.setattr(live_totals, "MIN_CALIBRATION_GAMES", 5)
    payload = copy.deepcopy(supported_payload())
    payload["calibration_residuals"]["10"].pop("LCK")
    result = live_totals.price_live_totals(
        payload,
        league="LCK",
        blue_team="Blue",
        red_team="Red",
        champions=list("ABCDEFGHIJ"),
        minute=10,
        current_kills=6,
        gold_difference=500.0,
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        lines=[{"line": 32.5}],
    )
    assert result["eligibility"]["status"] == "unavailable"
    assert "calibration_residuals_unavailable" in result["eligibility"]["blockers"]
    assert result["lines"][0]["under_probability"] is None


def test_built_artifact_preserves_series_clusters(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(live_totals, "ROOT", tmp_path)
    source = tmp_path / "maps.parquet"
    source.write_bytes(b"source fixture")
    artifact = live_totals.build_artifact(
        rows(),
        source_path=source,
        built_at="2026-02-01T00:00:00Z",
    )
    calibration_series = artifact["protocol"]["splits"]["calibration"]["series"]
    clustered = artifact["calibration_residual_clusters"]["10"]["LCK"]
    assert artifact["schema_version"] == "scryglass.live-total-kills.v2"
    assert clustered["series_n"] == calibration_series
    assert clustered["games_n"] == calibration_series
    assert len(clustered["clusters"]) == calibration_series
    assert artifact["authority"]["betting_decision_authorized"] is False


def test_artifact_writer_is_content_addressed_and_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path / "live_totals_model_v2.json"
    payload = {"schema_version": "fixture", "probability_authorized": False}
    raw_sha256 = live_totals.write_artifact_no_clobber(output, payload)
    expected = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")

    assert output.read_bytes() == expected
    assert raw_sha256 == hashlib.sha256(expected).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        live_totals.write_artifact_no_clobber(output, {"changed": True})


def test_source_snapshot_freezes_maps_and_refresh_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(live_totals, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert live_totals._root_locator(Path("warehouse/maps.parquet")) == (
        "warehouse/maps.parquet"
    )
    source = tmp_path / "warehouse" / "maps.parquet"
    source.parent.mkdir(parents=True)
    original = b"immutable map source"
    source.write_bytes(original)
    source_sha256 = hashlib.sha256(original).hexdigest()
    refresh = {
        "schema_version": "scryglass:warehouse-refresh-manifest:v2",
        "refreshed_at": "2026-08-01T22:58:25+00:00",
        "outputs": {"maps": {"raw_sha256": source_sha256}},
        "authority": {
            "descriptive_warehouse_provenance": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
    }
    refresh["manifest_canonical_sha256"] = live_totals._canonical_sha256(refresh)
    refresh_path = tmp_path / "warehouse" / "refresh_meta.json"
    refresh_path.write_text(json.dumps(refresh))

    package = live_totals.snapshot_source_package(
        source,
        refresh_path,
        tmp_path / "snapshots",
    )
    snapshot = package["maps_path"]
    assert snapshot.read_bytes() == original
    manifest = live_totals.validate_source_snapshot_manifest(
        snapshot, package["manifest_path"]
    )
    assert manifest["source"]["raw_sha256"] == source_sha256
    assert manifest["authority"]["betting_authority"] is False

    source.write_bytes(b"later mutable warehouse")
    assert snapshot.read_bytes() == original
