from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    FORM_METRICS,
    RATING_VARIANT_SCHEMA_VERSION,
    _canonical_json_bytes,
    build_strict_prior_player_form,
)
from lol_kills.research.future_value_snapshots import (
    SNAPSHOT_AUTHORITY,
    SNAPSHOT_RECEIPT_SCHEMA_VERSION,
    FutureValueSnapshotError,
    authorize_final_fit,
    build_future_value_snapshots,
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
    assert any(row["status"] == "research_only_missing_features" for row in result.player_rows)
    assert all(row["team_context_logit"] is None for row in result.team_rows)
    assert "team_context_not_in_final_model" in result.blockers


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
        receipt=lambda: {"receipt_sha256": "b" * 64},
    )
    receipt = {
        "receipt_sha256": "c" * 64,
        "parameter_sha256": "b" * 64,
    }
    with pytest.raises(FutureValueSnapshotError, match="parameters"):
        # This checks the object binding before any source rows are scored.
        from lol_kills.research.future_value_snapshots import _validate_model_object_binding

        _validate_model_object_binding(model, receipt)
