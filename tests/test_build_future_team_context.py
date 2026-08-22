from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.build_future_team_context import (
    TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION,
    TEAM_CONTEXT_RECEIPT_SCHEMA_VERSION,
    FutureTeamContextError,
    TeamContextBuild,
    _parameter_component,
    _roster_rows,
    write_team_context_artifact,
)
from lol_kills.research.future_value_rating import _canonical_json_bytes
from lol_kills.research.future_value_snapshots import _sha256_file

from tests.test_future_value_snapshots import _rows, _source_receipt


def test_parameter_component_records_exact_final_coefficients() -> None:
    model = SimpleNamespace(
        feature_names=("team_prior_win_diff", "roster_continuity_diff"),
        coefficients=np.asarray([0.25, -0.5]),
        scales=np.asarray([2.0, 4.0]),
        imputation_values=np.asarray([0.0, 0.1]),
        parameter_receipt=lambda: {"parameter_sha256": "a" * 64},
    )
    component = _parameter_component(
        model,
        {
            "feature_names": [
                "team_prior_win_diff",
                "roster_continuity_diff",
            ]
        },
    )

    assert component["status"] == "available"
    assert component["model_parameter_sha256"] == "a" * 64
    assert component["parameters"] == [
        {
            "feature": "team_prior_win_diff",
            "coefficient": 0.25,
            "scale": 2.0,
            "imputation": 0.0,
        },
        {
            "feature": "roster_continuity_diff",
            "coefficient": -0.5,
            "scale": 4.0,
            "imputation": 0.1,
        },
    ]
    payload = dict(component)
    claimed = payload.pop("parameter_component_sha256")
    assert claimed == hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def test_neutral_component_is_explicit_and_blocked() -> None:
    component = _parameter_component(
        SimpleNamespace(parameter_receipt=lambda: {"parameter_sha256": "a" * 64}),
        None,
    )

    assert component["status"] == "neutral_zero"
    assert component["feature_names"] == []
    assert component["parameters"] == []
    assert component["model_parameter_sha256"] == "a" * 64
    assert len(component["parameter_component_sha256"]) == 64


def test_roster_rows_require_exact_five_stable_roles() -> None:
    maps, players, teams = _rows(6)
    source = _source_receipt([f"g{i}" for i in range(1, 7)])
    rows = _roster_rows(
        maps,
        players,
        teams,
        source,
        model_receipt={},
    )

    assert set(rows["team_id"]) == {"oe:team:blue", "oe:team:red"}
    assert all(rows["roster_player_count"].eq(5))
    assert all(
        set(value) == {"top", "jungle", "mid", "bot", "support"}
        for value in rows["roster_roles"]
    )
    assert all(
        len(value) == 5 and len(set(value)) == 5
        for value in rows["roster_player_ids"]
    )


def test_roster_rows_fail_closed_for_incomplete_latest_roster() -> None:
    maps, players, teams = _rows(6)
    players = players.drop(
        players.index[
            (players["game_uid"].eq("g6"))
            & (players["side"].eq("Blue"))
            & (players["position"].eq("support"))
        ]
    )
    source = _source_receipt([f"g{i}" for i in range(1, 7)])

    with pytest.raises(FutureTeamContextError, match="ten rows|five"):
        _roster_rows(
            maps,
            players,
            teams,
            source,
            model_receipt={},
        )


def test_writer_binds_artifact_receipt_and_research_authority(tmp_path: Path) -> None:
    model_receipt_path = tmp_path / "model-receipt.json"
    model_artifact_path = tmp_path / "model.json"
    model_receipt_path.write_text("receipt", encoding="utf-8")
    model_artifact_path.write_text("artifact", encoding="utf-8")
    source = {
        "source_as_of": "2025-01-06T00:00:00Z",
        "source_game_count": 6,
        "source_identity_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
    }
    build = TeamContextBuild(
        status="research_only_partial",
        blockers=("team_context_not_in_final_model",),
        rows=(
            {
                "team_id": "oe:team:blue",
                "expected_starters": True,
                "team_context_logit": 0.0,
                "team_context_missing": True,
            },
        ),
        source=source,
        model={
            "receipt_sha256": "c" * 64,
            "declared_receipt_sha256": "c" * 64,
            "fit_game_identity_sha256": "d" * 64,
            "receipt_file": {
                "path": str(model_receipt_path),
                "bytes": model_receipt_path.stat().st_size,
                "sha256": _sha256_file(model_receipt_path),
            },
            "artifact_file": {
                "path": str(model_artifact_path),
                "bytes": model_artifact_path.stat().st_size,
                "sha256": _sha256_file(model_artifact_path),
            },
        },
        component={
            "status": "neutral_zero",
            "parameter_component_sha256": "1" * 64,
        },
        snapshot_receipt_sha256=None,
    )

    result = write_team_context_artifact(tmp_path / "team-context", build)
    artifact_path = Path(result["artifact_path"])
    receipt_path = Path(result["receipt_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION
    assert receipt["schema_version"] == TEAM_CONTEXT_RECEIPT_SCHEMA_VERSION
    assert artifact["authority"]["research_only"] is True
    assert artifact["authority"]["public_team_rating"] is False
    assert artifact["model"]["artifact_file"]["sha256"] == _sha256_file(
        model_artifact_path
    )
    assert receipt["artifact"]["sha256"] == _sha256_file(artifact_path)
    assert receipt["receipt_sha256"] == result["receipt"]["receipt_sha256"]
