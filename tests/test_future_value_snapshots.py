from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.build_future_value_snapshots import (
    _canonical_json_bytes as _snapshot_canonical_json_bytes,
    _snapshot_schema_digest,
    _snapshot_value_digest,
    _verify_current_rating_inputs,
    _verify_source_inputs,
)
from lol_kills.research.future_value_rating import (
    FORM_METRICS,
    RATING_VARIANT_SCHEMA_VERSION,
    _canonical_json_bytes,
    build_strict_prior_player_form,
)
from lol_kills.research.future_value_snapshots import (
    CURRENT_MU_EFFECTIVE_SCOPE,
    FORM_COMPONENT_SCOPE,
    SCALING_CONTEXT_BLOCKER,
    SNAPSHOT_AUTHORITY,
    SNAPSHOT_CAPABILITY_MATRIX,
    SNAPSHOT_RECEIPT_SCHEMA_VERSION,
    TEAM_CONTEXT_BINDING_SCHEMA_VERSION,
    FutureValueSnapshotError,
    _current_rating_feature_binding,
    _validated_team_context_binding,
    _latest_player_form,
    _player_contributions,
    authorize_final_fit,
    build_snapshot_capability_manifest,
    build_future_value_snapshots,
    snapshot_capability_matrix,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _source_receipt(ids: list[str]) -> dict[str, object]:
    ids = sorted(ids)
    authority = dict(SNAPSHOT_AUTHORITY)
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": f"2025-01-{len(ids):02d}T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
        "model_eligible_game_count": len(ids),
        "model_eligible_identity_sha256": identity_sha256(ids),
        "model_eligible_game_ids": ids,
        "source_rows": {"maps": len(ids), "players": len(ids) * 10, "teams": len(ids) * 2},
        "source_extra_game_ids": {"maps": [], "players": [], "teams": []},
        "identity_coverage": {},
        "checkpoint_coverage": {},
        "model_exclusions": {},
        "source_files": {
            "maps": {"locator": "fixture/maps.parquet", "bytes": 1, "sha256": "a" * 64},
            "players": {"locator": "fixture/players.parquet", "bytes": 1, "sha256": "b" * 64},
            "teams": {"locator": "fixture/teams.parquet", "bytes": 1, "sha256": "c" * 64},
            "accepted_census": {"locator": "fixture/census.json", "bytes": 1, "sha256": "d" * 64},
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
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _rows(game_count: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roles = ["top", "jungle", "mid", "bot", "support"]
    maps: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    for game_number in range(game_count):
        game_id = f"g{game_number + 1}"
        date = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=game_number)
        maps.append({"game_uid": game_id, "date": date, "y_blue_win": game_number % 2})
        for side in ("Blue", "Red"):
            team_id = f"oe:team:{side.casefold()}"
            teams.append({"game_uid": game_id, "date": date, "side": side, "teamid": team_id})
            for role_index, role in enumerate(roles):
                player_index = role_index + (0 if side == "Blue" else 5)
                row: dict[str, object] = {
                    "game_uid": game_id,
                    "date": date,
                    "side": side,
                    "position": role,
                    "playerid": f"oe:player:p{player_index}",
                    "playername": f"P{player_index}",
                    "teamid": team_id,
                    "champion": f"Champ{player_index}",
                    "competition_tier": "tier1",
                    "gamelength": 1800.0,
                    "totalgold": 1000.0 + 20 * game_number + player_index,
                    "cspm": 5.0 + player_index / 10.0,
                    "dpm": 300.0 + player_index,
                    "damageshare": 20.0 + player_index / 10.0,
                    "kills": 2.0,
                    "deaths": 1.0,
                    "assists": 4.0,
                    "wpm": 0.5,
                    "wcpm": 0.1,
                }
                players.append(row)
    return pd.DataFrame(maps), pd.DataFrame(players), pd.DataFrame(teams)


def test_strict_prior_form_excludes_same_timestamp_and_future_rows() -> None:
    maps, players, _teams = _rows(12)
    form = build_strict_prior_player_form(maps, players)
    first = form[form["game_id"].eq("g1")]
    latest = form[form["game_id"].eq("g12")]
    metric = "prior_form_gold_per_min"
    assert first[metric].isna().all()
    assert latest[metric].notna().all()
    before = form.loc[form["game_id"].eq("g11"), metric].copy()
    changed = players.copy()
    changed.loc[changed["game_uid"].eq("g12"), "totalgold"] = 999999.0
    after = build_strict_prior_player_form(maps, changed)
    pd.testing.assert_series_equal(
        before.reset_index(drop=True),
        after.loc[after["game_id"].eq("g11"), metric].reset_index(drop=True),
        check_names=False,
    )


def test_missing_final_fit_returns_exact_research_blocker() -> None:
    maps, players, teams = _rows(6)
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source,
    )
    assert result.status == "blocked"
    assert "final_fit_receipt_missing" in result.blockers
    assert result.player_rows == ()
    assert result.receipt["authority"] == {
        **source["authority"],
        "odds": False,
        "expected_value": False,
        "recommendation": False,
        "betting": False,
    }
    assert result.receipt["receipt_sha256"]


def test_snapshot_capability_matrix_is_closed_and_keeps_form_as_component() -> None:
    matrix = snapshot_capability_matrix()
    assert set(matrix) == {
        "current_only",
        "future_player_form",
        "scaling_curve",
        "both",
    }
    assert SNAPSHOT_CAPABILITY_MATRIX == matrix
    assert matrix["current_only"]["player"]["scope"] == CURRENT_MU_EFFECTIVE_SCOPE
    assert matrix["future_player_form"]["player"]["scope"] == FORM_COMPONENT_SCOPE
    assert matrix["future_player_form"]["player"]["full_composite_rating"] is False
    assert matrix["both"]["player"]["scope"] == FORM_COMPONENT_SCOPE
    assert matrix["both"]["scaling_context"]["status"] == "omitted"
    assert matrix["both"]["scaling_context"]["blocker"] == SCALING_CONTEXT_BLOCKER


def test_current_only_reuses_mu_effective_and_self_diff_is_exact_zero() -> None:
    maps, players, teams = _rows(2)
    source = _source_receipt(["g1", "g2"])
    current_players = pd.DataFrame(
        {
            "player": ["P1", "P0"],
            "player_id": ["oe:player:p1", "oe:player:p0"],
            "team_id": ["oe:team:red", "oe:team:blue"],
            "mu_effective": [1.0, 2.0],
        }
    )
    current_teams = pd.DataFrame(
        {
            "team": ["Red", "Blue"],
            "team_id": ["oe:team:red", "oe:team:blue"],
            "mu_effective": [1.0, 2.0],
        }
    )
    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source,
        current_player_ratings=current_players,
        current_team_ratings=current_teams,
        variant="current_only",
    )
    assert result.status == "research_only"
    assert [row["player_id"] for row in result.player_rows] == [
        "oe:player:p0",
        "oe:player:p1",
    ]
    assert all(row["rating_scope"] == CURRENT_MU_EFFECTIVE_SCOPE for row in result.player_rows)
    assert all(row["rank_delta"] == 0 for row in result.player_rank_diffs)
    assert all(row["self_diff"] == "exact_zero" for row in result.team_rank_diffs)
    assert result.receipt["source"]["source_receipt_sha256"] == source["receipt_sha256"]
    assert result.receipt["authority"]["public_player_rating"] is False


def test_scaling_curve_snapshot_is_typed_not_applicable_without_context() -> None:
    maps, players, teams = _rows(2)
    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=_source_receipt(["g1", "g2"]),
        variant="scaling_curve",
    )
    assert result.status == "research_only"
    assert result.player_rows == ()
    assert result.team_rows == ()
    assert result.player_rank_diffs == ()
    assert result.team_rank_diffs == ()
    assert result.receipt["rank_coverage"]["player"]["status"] == "not_applicable"
    assert result.receipt["rank_coverage"]["player"]["row_policy"] == "no_rows"
    assert result.receipt["capability"]["player"]["status"] == "not_applicable"
    assert result.receipt["authority"]["public_team_rating"] is False


def test_capability_manifest_records_all_variants_and_source_binding() -> None:
    source = _source_receipt(["g1", "g2"])
    manifest = build_snapshot_capability_manifest(source)
    assert set(manifest["variants"]) == {
        "current_only",
        "future_player_form",
        "scaling_curve",
        "both",
    }
    assert manifest["source"]["source_receipt_sha256"] == source["receipt_sha256"]
    assert all(value is False for key, value in manifest["authority"].items() if key != "research_only")


def test_final_fit_gate_rejects_fold_receipt_without_current_ledger() -> None:
    source = _source_receipt(["g1", "g2"])
    model = {
        "schema_version": "scryglass:future-value-model-fit:v1",
        "status": "research_only_blocked",
        "variant": "future_player_form",
        "fit_game_ids": ["g1"],
        "fit_window_end": "2025-01-01T00:00:00Z",
        "source_binding": {
            "source_receipt_sha256": source["receipt_sha256"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_as_of": source["source_as_of"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        },
        "regularization_selection": {"blockers": ["nested_inner_feature_ledger_missing_fixed_c_used"]},
        "optimizer_evidence": {"success": True, "finite_coefficients": True},
        "rank_3": {"parameter_sha256": "a" * 64},
    }
    payload = dict(model)
    model["parameter_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    auth = authorize_final_fit(model, source)
    assert not auth.authorized
    assert "final_fit_not_bound_to_complete_model_eligible_census" in auth.blockers
    assert "current_rating_feature_ledger_binding_missing" in auth.blockers
    assert "nested_inner_feature_ledger_missing_fixed_c_used" in auth.blockers


def test_current_rating_binding_prefers_exact_producer_over_aggregate_design() -> None:
    exact = {
        "artifact": {"path": "/tmp/current-rating-ledger.parquet"},
        "producer_receipt_file": {"path": "/tmp/current-rating-ledger-receipt.json"},
        "feature_names": ["base_team_logit"],
    }
    aggregate = {
        "feature_names": ["base_team_logit", "player_form_gold_per_min"],
        "producer_names": ["current_sequential_rating", "strict_prior_player_form"],
    }
    assert _current_rating_feature_binding(
        {
            "current_rating_feature_binding": exact,
            "feature_ledger_binding": aggregate,
        }
    ) is exact


def test_malformed_explicit_current_rating_binding_does_not_use_legacy_fallback() -> None:
    legacy = {
        "artifact": {"path": "/tmp/current-rating-ledger.parquet"},
        "producer_receipt_file": {"path": "/tmp/current-rating-ledger-receipt.json"},
        "feature_names": ["base_team_logit"],
    }
    assert _current_rating_feature_binding(
        {
            "current_rating_feature_binding": "forged",
            "feature_ledger_binding": legacy,
        }
    ) is None


class _Atoms:
    def transform(self, form: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=form.index)
        for index in range(1, 4):
            output[f"rank_3_player_atom_{index}"] = 0.1 * index
            output[f"rank_3_champion_role_atom_{index}"] = 0.2 * index
        output["rank_3_player_atom_available"] = True
        output["rank_3_champion_role_atom_available"] = True
        output["rank_3_champion_role_support"] = 10
        return output


def test_atom_missing_flags_follow_row_position_after_latest_selection() -> None:
    maps, players, _teams = _rows(6)
    form = build_strict_prior_player_form(maps, players)
    latest = form[form["game_id"].eq("g6")].iloc[[0]].copy()
    latest["team_id"] = "oe:team:blue"
    latest["playername"] = "P0"
    latest["champion"] = "Champ0"
    latest.index = [99117]
    model = SimpleNamespace(
        feature_names=("rank_3_atom_missing_rate",),
        scales=np.ones(1),
        coefficients=np.ones(1),
        imputation_values=np.zeros(1),
        atom_model=_Atoms(),
    )

    output = _player_contributions(model, latest)

    assert output["model_feature_missing"].tolist() == [False]


def test_team_value_requires_exact_five_players_and_preserves_champion_split() -> None:
    maps, players, teams = _rows(6)
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    feature_names = (
        "player_form_gold_per_min",
        "rank_3_player_atom_1",
        "rank_3_champion_role_atom_1",
    )
    model = SimpleNamespace(
        feature_names=feature_names,
        scales=np.ones(3),
        coefficients=np.ones(3),
        imputation_values=np.zeros(3),
        atom_model=_Atoms(),
        receipt=lambda: {
            "schema_version": "scryglass:future-value-model-fit:v1",
            "status": "final_fit_authorized",
            "variant": "future_player_form",
            "fit_game_ids": [f"g{i}" for i in range(1, 7)],
            "fit_window_end": "2025-01-06T00:00:00Z",
            "source_binding": {
                "source_receipt_sha256": source["receipt_sha256"],
                "source_identity_sha256": source["source_identity_sha256"],
                "source_as_of": source["source_as_of"],
                "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
            },
            "feature_ledger_binding": {
                "source_receipt_sha256": source["receipt_sha256"],
                "source_identity_sha256": source["source_identity_sha256"],
                "producer_receipt_sha256": "a" * 64,
                "producer_names": ["current_sequential_rating"],
            },
            "regularization_selection": {"selected_c": 0.1, "blockers": []},
            "optimizer_evidence": {"success": True, "finite_coefficients": True},
            "rank_3": {"parameter_sha256": "b" * 64},
        },
    )
    receipt = model.receipt()
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes_without_hash(receipt)).hexdigest()
    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source,
        model=model,
        model_receipt=receipt,
        current_player_ratings=pd.DataFrame(
            {"player_id": [f"oe:player:p{i}" for i in range(10)], "mu_total": list(range(10))}
        ),
        current_team_ratings=pd.DataFrame(
            {"team_id": ["oe:team:blue", "oe:team:red"], "mu_total": [1.0, 0.0]}
        ),
    )
    assert result.status == "research_only_partial"
    assert len(result.player_rows) == 10
    assert len(result.team_rows) == 2
    assert all(row["roster_player_count"] == 5 for row in result.team_rows)
    assert all(
        row["champion_role_atom_logit"] is not None
        for row in result.player_rows
        if not row["model_feature_missing"]
    )
    assert all(
        row["rating_scope"] == FORM_COMPONENT_SCOPE
        and row["full_composite_rating"] is False
        for row in result.player_rows
    )
    assert all(row["rating_scope"] == FORM_COMPONENT_SCOPE for row in result.team_rows)
    assert any(row["status"] == "research_only_missing_features" for row in result.player_rows)
    assert all(row["team_context_logit"] is None for row in result.team_rows)
    assert "team_context_not_in_final_model" in result.blockers

    both_receipt = dict(receipt)
    both_receipt["variant"] = "both"
    both_receipt.pop("receipt_sha256", None)
    both_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(both_receipt)
    ).hexdigest()
    both_model = SimpleNamespace(
        feature_names=model.feature_names,
        scales=model.scales,
        coefficients=model.coefficients,
        imputation_values=model.imputation_values,
        atom_model=model.atom_model,
        receipt=lambda: both_receipt,
    )
    both = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source,
        model=both_model,
        model_receipt=both_receipt,
        current_player_ratings=pd.DataFrame(
            {"player_id": [f"oe:player:p{i}" for i in range(10)], "mu_total": list(range(10))}
        ),
        current_team_ratings=pd.DataFrame(
            {"team_id": ["oe:team:blue", "oe:team:red"], "mu_total": [1.0, 0.0]}
        ),
        variant="both",
    )
    assert both.receipt["scaling_context"]["status"] == "omitted"
    assert SCALING_CONTEXT_BLOCKER in both.blockers
    assert all(row["rating_scope"] == FORM_COMPONENT_SCOPE for row in both.player_rows)


def test_team_context_binding_requires_source_and_parameter_proof() -> None:
    source = _source_receipt(["g1", "g2"])
    binding: dict[str, object] = {
        "schema_version": TEAM_CONTEXT_BINDING_SCHEMA_VERSION,
        "status": "available",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "fit_game_ids": ["g1", "g2"],
        "fit_window_end": source["source_as_of"],
        "strict_prior_timing": "fit_rows_strictly_before_cutoff",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "series_safety": "whole_series_disjoint",
        "feature_names": ["team_prior_win_diff", "roster_continuity_diff"],
        "authority": {
            "research_only": True,
            "public_team_rating": False,
        },
    }
    binding["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(binding)
    ).hexdigest()
    receipt = {
        "fit_game_ids": ["g1", "g2"],
        "fit_window_end": source["source_as_of"],
        "team_context_binding": binding,
    }
    model = SimpleNamespace(
        feature_names=("team_prior_win_diff", "roster_continuity_diff")
    )
    verified = _validated_team_context_binding(model, receipt, {
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
    })
    assert verified is not None
    assert verified["feature_names"] == [
        "team_prior_win_diff",
        "roster_continuity_diff",
    ]

    binding["source_receipt_sha256"] = "f" * 64
    binding["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes({key: value for key, value in binding.items() if key != "receipt_sha256"})
    ).hexdigest()
    with pytest.raises(FutureValueSnapshotError, match="source receipt"):
        _validated_team_context_binding(model, receipt, {
            "source_receipt_sha256": source["receipt_sha256"],
            "source_identity_sha256": source["source_identity_sha256"],
        })


def test_team_context_receipt_clears_only_its_specific_blocker() -> None:
    source = _source_receipt(["g1", "g2"])
    binding: dict[str, object] = {
        "schema_version": TEAM_CONTEXT_BINDING_SCHEMA_VERSION,
        "status": "available",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "fit_game_ids": ["g1", "g2"],
        "fit_window_end": source["source_as_of"],
        "strict_prior_timing": "fit_rows_strictly_before_cutoff",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "series_safety": "whole_series_disjoint",
        "feature_names": ["team_prior_win_diff"],
    }
    binding["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(binding)
    ).hexdigest()
    receipt = {
        "fit_game_ids": ["g1", "g2"],
        "fit_window_end": source["source_as_of"],
        "team_context_binding": binding,
    }
    model = SimpleNamespace(feature_names=("team_prior_win_diff",))
    assert _validated_team_context_binding(
        model,
        receipt,
        {
            "source_receipt_sha256": source["receipt_sha256"],
            "source_identity_sha256": source["source_identity_sha256"],
        },
    )["status"] == "available"


def _canonical_bytes_without_hash(payload: dict[str, object]) -> bytes:
    value = dict(payload)
    value.pop("parameter_sha256", None)
    value.pop("receipt_sha256", None)
    return _canonical_json_bytes(value)


def test_ambiguous_latest_roster_fails_closed() -> None:
    maps, players, teams = _rows(6)
    duplicate = players[players["game_uid"].eq("g6")].iloc[[0]].copy()
    duplicate["game_uid"] = "g5"
    duplicate["date"] = pd.Timestamp("2025-01-06", tz="UTC")
    players = pd.concat([players, duplicate], ignore_index=True)
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    with pytest.raises(FutureValueSnapshotError, match="ten rows|duplicate"):
        build_future_value_snapshots(maps, players, teams, source_receipt=source)


def test_player_date_mutation_fails_against_map_date() -> None:
    maps, players, teams = _rows(6)
    players.loc[players.index[0], "date"] = pd.Timestamp("2025-01-07", tz="UTC")
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    with pytest.raises(FutureValueSnapshotError, match="dates"):
        build_future_value_snapshots(maps, players, teams, source_receipt=source)


def test_explicit_model_receipt_must_bind_model_parameters() -> None:
    source = _source_receipt(["g1", "g2"])
    model = SimpleNamespace(
        parameter_receipt=lambda: {"parameter_sha256": "a" * 64},
        receipt=lambda: {"parameter_sha256": "a" * 64},
    )
    receipt = {"parameter_sha256": "b" * 64}
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    with pytest.raises(FutureValueSnapshotError, match="parameters"):
        # This checks the object binding before any source rows are scored.
        from lol_kills.research.future_value_snapshots import _validate_model_object_binding

        _validate_model_object_binding(model, receipt)


def test_explicit_model_receipt_must_bind_model_metadata() -> None:
    object_receipt = {
        "schema_version": "scryglass:future-value-model-fit:v1",
        "fit_game_ids": ["g1", "g2"],
        "fit_window_end": "2025-01-02T00:00:00Z",
        "variant": "future_player_form",
        "parameter_sha256": "a" * 64,
        "source_binding": {"source_receipt_sha256": "b" * 64},
    }
    model = SimpleNamespace(
        parameter_receipt=lambda: {"parameter_sha256": "a" * 64},
        receipt=lambda: dict(object_receipt),
    )
    receipt = dict(object_receipt)
    receipt["fit_window_end"] = "2025-01-03T00:00:00Z"
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()

    with pytest.raises(FutureValueSnapshotError, match="metadata"):
        from lol_kills.research.future_value_snapshots import _validate_model_object_binding

        _validate_model_object_binding(model, receipt)


def test_snapshot_cli_rejects_mutated_player_bytes(tmp_path: Path) -> None:
    maps, players, teams = _rows(6)
    source_root = tmp_path / "source"
    source_root.mkdir()
    paths = {
        "maps": source_root / "maps.parquet",
        "players": source_root / "oe_player_games.parquet",
        "teams": source_root / "oe_team_games.parquet",
    }
    maps.to_parquet(paths["maps"], index=False)
    players.to_parquet(paths["players"], index=False)
    teams.to_parquet(paths["teams"], index=False)
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    source_files = dict(source["source_files"])
    for label, path in paths.items():
        source_files[label] = {
            "locator": str(path.relative_to(tmp_path)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    source["source_files"] = source_files
    source.pop("receipt_sha256", None)
    source["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(source)
    ).hexdigest()
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    expected_receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    players.loc[0, "playername"] = "Mutated Player Name"
    players.to_parquet(paths["players"], index=False)

    with pytest.raises(FutureValueSnapshotError, match="players file.*changed"):
        _verify_source_inputs(
            source_root,
            receipt_path,
            source,
            expected_source_receipt_sha256=expected_receipt_hash,
        )


def _write_current_snapshot_receipt(
    tmp_path: Path,
    source: dict[str, object],
) -> tuple[Path, Path, Path, dict[str, object], str]:
    current_root = tmp_path / "current"
    (current_root / "player").mkdir(parents=True)
    (current_root / "team").mkdir()
    player = pd.DataFrame(
        {
            "player": ["Alpha", "Beta"],
            "player_id": ["oe:player:a", "oe:player:b"],
            "team_id": ["oe:team:t", "oe:team:u"],
            "mu_effective": [100.0, 90.0],
        }
    )
    team = pd.DataFrame(
        {
            "team": ["Team A", "Team B"],
            "team_id": ["oe:team:t", "oe:team:u"],
            "mu_effective": [100.0, 90.0],
        }
    )
    player_path = current_root / "player/player_ratings_snapshot.parquet"
    team_path = current_root / "team/ratings_snapshot.parquet"
    player.to_parquet(player_path, index=False)
    team.to_parquet(team_path, index=False)

    def record(frame: pd.DataFrame, path: Path, identity_column: str) -> dict[str, object]:
        raw = path.read_bytes()
        return {
            "locator": path.relative_to(current_root).as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
            "schema_sha256": _snapshot_schema_digest(frame),
            "identity_column": identity_column,
            "value_column": "mu_effective",
            "value_digest_sha256": _snapshot_value_digest(
                frame, identity_column=identity_column, value_column="mu_effective"
            ),
        }

    receipt = {
        "schema_version": "scryglass:current-rating-snapshot-receipt:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "snapshots": {
            "player": record(player, player_path, "player_id"),
            "team": record(team, team_path, "team_id"),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _snapshot_canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path = current_root / "current-rating-snapshot-receipt.json"
    receipt_path.write_bytes(_snapshot_canonical_json_bytes(receipt))
    return current_root, receipt_path, player_path, receipt, hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()


def test_snapshot_cli_requires_independent_current_receipt_for_mutated_values(
    tmp_path: Path,
) -> None:
    source = _source_receipt(["g1", "g2"])
    current_root, receipt_path, player_path, receipt, original_receipt_hash = (
        _write_current_snapshot_receipt(tmp_path, source)
    )
    _verify_current_rating_inputs(
        current_root,
        receipt_path,
        receipt,
        source_receipt=source,
        expected_current_receipt_sha256=original_receipt_hash,
    )

    mutated = pd.read_parquet(player_path)
    mutated.loc[1, "mu_effective"] = 110.0
    mutated.to_parquet(player_path, index=False)
    receipt["snapshots"]["player"]["sha256"] = hashlib.sha256(
        player_path.read_bytes()
    ).hexdigest()
    receipt["snapshots"]["player"]["bytes"] = player_path.stat().st_size
    receipt["snapshots"]["player"]["value_digest_sha256"] = _snapshot_value_digest(
        mutated, identity_column="player_id", value_column="mu_effective"
    )
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        _snapshot_canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path.write_bytes(_snapshot_canonical_json_bytes(receipt))

    with pytest.raises(FutureValueSnapshotError, match="receipt file hash changed"):
        _verify_current_rating_inputs(
            current_root,
            receipt_path,
            receipt,
            source_receipt=source,
            expected_current_receipt_sha256=original_receipt_hash,
        )
    updated_receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    _verify_current_rating_inputs(
        current_root,
        receipt_path,
        receipt,
        source_receipt=source,
        expected_current_receipt_sha256=updated_receipt_hash,
    )
