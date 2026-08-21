from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.build_strict_prior_composition_atoms import (
    StrictPriorAtomError,
    _edge_from_signal,
    build_player_form,
    score_static_atoms,
)
from benchmarks.future_value_draft_score_fourway import (
    _sha_bytes,
    _verify_strict_prior_atom_artifact,
    _verify_strict_prior_form_artifact,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _canonical(value: object) -> bytes:
    import json

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _players(days: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = ("top", "jng", "mid", "bot", "sup")
    for day in range(days):
        for side in ("Blue", "Red"):
            for role in roles:
                rows.append(
                    {
                        "game_uid": f"g{day}",
                        "date": f"2026-01-{day + 1:02d}T12:00:00Z",
                        "side": side,
                        "position": role,
                        "playername": f"{side}-{role}",
                        "playerid": f"{side}-{role}",
                        "teamname": side,
                        "teamid": side,
                        "dpm": 100.0 + day * 10.0 + (1.0 if side == "Blue" else 0.0),
                        "damageshare": 0.1,
                        "earnedgoldshare": 0.1,
                        "cspm": 5.0,
                        "kills": 1.0,
                        "deaths": 1.0,
                        "assists": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_player_form_uses_only_prior_calendar_dates() -> None:
    players = _players()
    maps = pd.DataFrame(
        {
            "game_id": ["g0", "g1", "g2"],
            "date": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        }
    )
    baseline = {row["game_id"]: row for row in build_player_form(players, maps)}
    changed = players.copy()
    changed.loc[changed["game_uid"].eq("g1"), "dpm"] = 999999.0
    changed_rows = {row["game_id"]: row for row in build_player_form(changed, maps)}
    assert baseline["g0"]["future_player_form_logit"] is None
    assert baseline["g1"]["fit_through"] == "2026-01-01T12:00:00Z"
    assert changed_rows["g1"]["future_player_form_logit"] == baseline["g1"]["future_player_form_logit"]
    assert changed_rows["g2"]["future_player_form_logit"] != baseline["g2"]["future_player_form_logit"]


def test_same_date_games_do_not_enter_each_other_form() -> None:
    players = _players(2)
    players.loc[players["game_uid"].eq("g1"), "date"] = "2026-01-01T18:00:00Z"
    players.loc[players["game_uid"].eq("g1") & players["side"].eq("Blue"), "dpm"] = 100000.0
    maps = pd.DataFrame(
        {
            "game_id": ["g0", "g1"],
            "date": pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-01T18:00:00Z"], utc=True),
        }
    )
    rows = build_player_form(players, maps)
    assert rows[0]["fit_through"] is None
    assert rows[1]["fit_through"] is None
    assert rows[0]["future_player_form_logit"] is None
    assert rows[1]["future_player_form_logit"] is None


def _reference_player_form(players: pd.DataFrame) -> list[dict[str, object]]:
    """The pre-optimization list implementation for parity checks."""

    frame = players.copy()
    frame["game_id"] = frame["game_uid"].map(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["role"] = frame["position"].astype(str).str.casefold()
    frame["player_key"] = frame.apply(
        lambda row: str(row.get("playerid") or "").strip()
        or "name:" + str(row.get("playername") or "").strip().casefold()
        + "|team:" + str(row.get("teamid") or row.get("teamname") or "").strip().casefold(),
        axis=1,
    )
    frame = frame.sort_values(["date", "game_id", "side", "position"], kind="stable")
    role_metrics = defaultdict(lambda: defaultdict(list))
    player_metrics = defaultdict(lambda: defaultdict(list))
    global_metrics = defaultdict(lambda: defaultdict(list))
    metrics = ("dpm", "damageshare", "earnedgoldshare", "cspm", "kills", "deaths", "assists")
    weights = {"dpm": 1.0, "damageshare": 1.0, "earnedgoldshare": 1.0, "cspm": 0.5, "kills": 0.5, "deaths": -0.5, "assists": 0.5}
    scales = {"dpm": 1000.0, "damageshare": 0.20, "earnedgoldshare": 0.20, "cspm": 10.0, "kills": 5.0, "deaths": 5.0, "assists": 8.0}
    rows = []
    for day, day_frame in frame.groupby(frame["date"].dt.normalize(), sort=True):
        for game_id, group in day_frame.groupby("game_id", sort=True):
            side_scores = {"Blue": [], "Red": []}
            side_metrics = {side: {metric: [] for metric in metrics} for side in ("Blue", "Red")}
            support = {"Blue": 0, "Red": 0}
            for _, raw in group.iterrows():
                key = str(raw["player_key"])
                role = str(raw["role"])
                role_history = role_metrics.get((key, role), {})
                player_history = player_metrics.get(key, {})
                z_values = []
                for metric in metrics:
                    values = role_history.get(metric, [])
                    value = float(np.mean(values)) if values else None
                    if value is None:
                        values = player_history.get(metric, [])
                        value = float(np.mean(values)) if values else None
                    all_values = global_metrics.get(role, {}).get(metric, [])
                    if value is None or not all_values:
                        continue
                    mean = float(np.mean(all_values))
                    std = float(np.std(all_values))
                    value_scale = max(std, scales[metric] * 0.05)
                    z_values.append(weights[metric] * ((value - mean) / value_scale))
                    side_metrics[str(raw["side"])][metric].append(value)
                if z_values:
                    side = str(raw["side"])
                    side_scores[side].append(float(np.mean(z_values)))
                    support[side] += 1
            blue = float(np.mean(side_scores["Blue"])) if side_scores["Blue"] else None
            red = float(np.mean(side_scores["Red"])) if side_scores["Red"] else None
            feature = blue - red if blue is not None and red is not None else None
            rows.append(
                {
                    "game_id": str(game_id),
                    "date": group["date"].max().isoformat().replace("+00:00", "Z"),
                    "fit_through": frame.loc[frame["date"] < day, "date"].max().isoformat().replace("+00:00", "Z") if frame["date"].lt(day).any() else None,
                    "status": "available" if feature is not None else "unavailable",
                    "future_player_form_logit": feature,
                    "support_blue": support["Blue"],
                    "support_red": support["Red"],
                }
            )
        for _, raw in day_frame.iterrows():
            key = str(raw["player_key"])
            role = str(raw["role"])
            for metric in metrics:
                try:
                    value = float(raw[metric])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                role_metrics[(key, role)][metric].append(value)
                player_metrics[key][metric].append(value)
                global_metrics[role][metric].append(value)
    return sorted(rows, key=lambda row: str(row["game_id"]))


def test_player_form_running_aggregates_match_reference(tmp_path) -> None:
    players = _players(4)
    maps = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(4)],
            "date": pd.date_range("2026-01-01", periods=4, tz="UTC"),
        }
    )
    optimized = build_player_form(players, maps)
    reference = _reference_player_form(players)
    assert [row["game_id"] for row in optimized] == [row["game_id"] for row in reference]
    for actual, expected in zip(optimized, reference):
        assert actual["status"] == expected["status"]
        assert actual["support_blue"] == expected["support_blue"]
        assert actual["support_red"] == expected["support_red"]
        if expected["future_player_form_logit"] is None:
            assert actual["future_player_form_logit"] is None
        else:
            assert actual["future_player_form_logit"] == pytest.approx(
                expected["future_player_form_logit"], rel=1e-12, abs=1e-12
            )


def test_static_atom_mapping_is_additive() -> None:
    signal = {
        "status": "available",
        "blue": {"components": {"base": 1.0, "ally_synergy": 2.0, "enemy_counter": 3.0, "same_role": 4.0, "atomized": 5.0}},
        "red": {"components": {"base": 0.5, "ally_synergy": 1.0, "enemy_counter": 1.5, "same_role": 2.0, "atomized": 2.5}},
    }
    edge = _edge_from_signal(signal)
    assert edge == {
        "base": 0.5,
        "ally_synergy": 1.0,
        "enemy_counter": 1.5,
        "same_role": 2.0,
        "archetype_interactions": 2.5,
        "total": 7.5,
    }


def test_static_atom_rejects_target_date_fit(monkeypatch, tmp_path) -> None:
    games = [{"game_uid": "g1", "date": pd.Timestamp("2026-01-02T12:00:00Z")}]
    maps = pd.DataFrame({"game_id": ["g1"], "date": pd.to_datetime(["2026-01-02T12:00:00Z"], utc=True)})
    source = {"source_identity_sha256": "a" * 64, "source_game_count": 1, "accepted_game_ids": ["g1"]}
    fake_signal = {
        "status": "available",
        "fit_through": "2026-01-02T00:00:00Z",
        "blue": {"components": {"base": 1, "ally_synergy": 0, "enemy_counter": 0, "same_role": 0, "atomized": 0}},
        "red": {"components": {"base": 0, "ally_synergy": 0, "enemy_counter": 0, "same_role": 0, "atomized": 0}},
    }
    monkeypatch.setattr(
        "benchmarks.build_strict_prior_composition_atoms.composition_signal.score_games_temporally",
        lambda *args, **kwargs: SimpleNamespace(signals={"g1": fake_signal}, audit={}),
    )
    with pytest.raises(StrictPriorAtomError):
        score_static_atoms(games, source, maps, cache_dir=tmp_path)


def test_benchmark_verifies_strict_prior_atom_and_form_receipts(tmp_path) -> None:
    game_ids = ["g1", "g2"]
    maps = pd.DataFrame(
        {
            "game_id": game_ids,
            "date": pd.to_datetime(["2026-01-02T12:00:00Z", "2026-01-03T12:00:00Z"], utc=True),
        }
    )
    maps_path = tmp_path / "maps.parquet"
    maps.to_parquet(maps_path)
    maps_raw = maps_path.read_bytes()
    players_raw = b"players"
    source = {
        "source_as_of": "2026-01-04T00:00:00Z",
        "source_game_count": 2,
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": game_ids,
        "receipt_sha256": "r" * 64,
        "source_files": {
            "maps": {"bytes": len(maps_raw), "sha256": _sha_bytes(maps_raw)},
            "players": {"bytes": len(players_raw), "sha256": _sha_bytes(players_raw)},
        },
    }
    common = {
        "source_as_of": source["source_as_of"],
        "source_game_count": 2,
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "input_files": {
            "maps": {"bytes": len(maps_raw), "sha256": _sha_bytes(maps_raw)},
            "players": {"bytes": len(players_raw), "sha256": _sha_bytes(players_raw)},
        },
    }
    atom_payload = {
        "schema_version": "scryglass:strict-prior-composition-atoms:v1",
        "status": "research_only",
        "authority": {"research_only": True, "public": False, "probability": False, "promotion": False, "deployment": False},
        "source": common,
        "producer": {
            "training_order": "earlier accepted calendar-date clusters only",
            "producer_code_sha256": "a" * 64,
            "composition_signal_code_sha256": "b" * 64,
        },
        "coverage": {"fit_through_max": "2026-01-02T00:00:00Z"},
        "rows": [
            {"game_id": "g1", "date": "2026-01-02T12:00:00Z", "fit_through": None, "status": "unavailable", "edge_components": None},
            {"game_id": "g2", "date": "2026-01-03T12:00:00Z", "fit_through": "2026-01-02T12:00:00Z", "status": "available", "edge_components": {"base": 1.0, "ally_synergy": 2.0, "enemy_counter": 3.0, "same_role": 4.0, "archetype_interactions": 5.0, "total": 15.0}},
        ],
    }
    atom_payload["artifact_sha256"] = _sha_bytes(_canonical(atom_payload))
    atom_path = tmp_path / "atoms.json"
    atom_path.write_bytes(_canonical(atom_payload) + b"\n")
    atom, atom_receipt = _verify_strict_prior_atom_artifact(
        atom_path,
        source,
        maps,
        expected_sha256=_sha_bytes(atom_path.read_bytes()),
    )
    assert atom_receipt["available_game_count"] == 1
    assert atom.loc[atom["game_id"].eq("g2"), "composition_base_logit"].iloc[0] == 1.0

    form_payload = {
        "schema_version": "scryglass:strict-prior-player-form:v1",
        "status": "research_only",
        "authority": {"research_only": True, "public": False, "probability": False, "promotion": False, "deployment": False},
        "source": common,
        "producer": {"training_order": "earlier accepted calendar-date clusters only", "producer_code_sha256": "c" * 64},
        "coverage": {"fit_through_max": "2026-01-02T00:00:00Z"},
        "rows": [
            {"game_id": "g1", "date": "2026-01-02T12:00:00Z", "fit_through": None, "status": "unavailable", "future_player_form_logit": None},
            {"game_id": "g2", "date": "2026-01-03T12:00:00Z", "fit_through": "2026-01-02T12:00:00Z", "status": "available", "future_player_form_logit": 0.25},
        ],
    }
    form_payload["artifact_sha256"] = _sha_bytes(_canonical(form_payload))
    form_path = tmp_path / "form.json"
    form_path.write_bytes(_canonical(form_payload) + b"\n")
    form, form_receipt = _verify_strict_prior_form_artifact(
        form_path,
        source,
        maps,
        expected_sha256=_sha_bytes(form_path.read_bytes()),
    )
    assert form_receipt["artifact_verified"] is True
    assert form.loc[form["game_id"].eq("g2"), "future_player_form_logit"].iloc[0] == 0.25
