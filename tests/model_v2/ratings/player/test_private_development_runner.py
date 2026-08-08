from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

import lol_kills.v2.ratings.player.private_development_runner as runner
from lol_kills.v2.ratings.player.model import DISPLAY_LOGIT_SCALE
from lol_kills.v2.ratings.player.private_development_runner import Candidate, PrivateDevelopmentError, build_private_development_artifact, verify_private_development_artifact, write_private_development_artifact
from lol_kills.v2.ratings.player.real_v1_adapter import ACCEPTED_G1_PINS, CLAIM_CEILING, MapObservation, PlayerLineupObservation, PrivatePlayerRatingFold, PrivatePlayerRatingInput


def _sha(value) -> str:
    return hashlib.sha256(runner._canonical_bytes(value)).hexdigest()


def _map(number: int, fold: str, at: str, y: int, origins: tuple[str, ...]) -> MapObservation:
    game_id = f"{100 + number}-{100 + number}_game_1"
    origin_sha = _sha(list(origins))
    observations = []
    for side, team in (("blue", "team-blue"), ("red", "team-red")):
        for role in ("top", "jungle", "mid", "bot", "support"):
            observations.append(PlayerLineupObservation(
                observation_id=f"{game_id}:{side}:{role}", source_game_id=game_id, fold_id=fold,
                game_side=side, role=role, source_player_id=f"{side}-{role}", source_team_id=team,
                blue_win=y, ordered_origin_map_ids=origins, ordered_origin_sha256=origin_sha,
            ))
    return MapObservation(
        source_game_id=game_id, source_series_id=f"oe:lpl:bmid:{100 + number}", fold_id=fold,
        source_local_event_start=at, source_blue_result_id=f"oe-team-row:{game_id}:100",
        source_red_result_id=f"oe-team-row:{game_id}:200", blue_win=y,
        player_observations=tuple(observations), ordered_origin_map_ids=origins,
        ordered_origin_sha256=origin_sha,
    )


def _input(*, validation_y: int = 0, reordered_development_origins: bool = False) -> PrivatePlayerRatingInput:
    train_one = _map(1, "TRAIN", "2025-01-01T10:00:00", 1, ())
    train_two = _map(2, "TRAIN", "2025-01-04T10:00:00", 0, (train_one.source_game_id,))
    development_origins = (train_one.source_game_id, train_two.source_game_id)
    if reordered_development_origins:
        development_origins = tuple(reversed(development_origins))
    development = _map(3, "DEVELOPMENT", "2025-01-07T10:00:00", 1, development_origins)
    if reordered_development_origins:
        development = replace(development, ordered_origin_sha256=_sha([train_one.source_game_id, train_two.source_game_id]))
    validation = _map(4, "VALIDATION", "2025-01-10T10:00:00", validation_y, (train_one.source_game_id, train_two.source_game_id, development.source_game_id))
    grouped = (("TRAIN", (train_one, train_two)), ("DEVELOPMENT", (development,)), ("VALIDATION", (validation,)))
    folds = tuple(PrivatePlayerRatingFold(
        fold_id=fold_id,
        map_observations=maps,
        ordered_map_ids_sha256=_sha([item.source_game_id for item in maps]),
        ordered_origin_identities_sha256=_sha([{"source_game_id": item.source_game_id, "ordered_origin_map_ids": list(item.ordered_origin_map_ids)} for item in maps]),
    ) for fold_id, maps in grouped)
    return PrivatePlayerRatingInput(
        schema_version="scryglass:player-rating-private-g2-observed-lineups:v1",
        manifest_sha256=ACCEPTED_G1_PINS.manifest_sha256, rows_sha256=ACCEPTED_G1_PINS.rows_sha256,
        selected_target_sha256=ACCEPTED_G1_PINS.selected_target_sha256,
        split_payload_sha256=ACCEPTED_G1_PINS.split_payload_sha256, folds=folds, map_count=4, player_observation_count=40,
        claim_ceiling=dict(CLAIM_CEILING),
    )


def test_private_runner_consumes_only_adapter_pins_and_uses_adf_predictive_metrics() -> None:
    artifact = build_private_development_artifact(input_loader=_input)
    assert artifact["adapter_input_pins"] == {
        "manifest_sha256": ACCEPTED_G1_PINS.manifest_sha256, "rows_sha256": ACCEPTED_G1_PINS.rows_sha256,
        "selected_target_sha256": ACCEPTED_G1_PINS.selected_target_sha256,
        "split_payload_sha256": ACCEPTED_G1_PINS.split_payload_sha256, "map_count": 4, "player_observation_count": 40,
        "fold_map_digests": {fold.fold_id: fold.ordered_map_ids_sha256 for fold in _input().folds},
        "fold_origin_digests": {fold.fold_id: fold.ordered_origin_identities_sha256 for fold in _input().folds},
    }
    assert artifact["config"]["evaluation"]["posterior_predictive"] == "existing_validated_logistic_normal_integral"
    assert artifact["config"]["display"]["scale"] == pytest.approx(DISPLAY_LOGIT_SCALE)
    assert artifact["config"]["candidates"][1]["process_variance_per_day"] == 0.0005
    assert artifact["config"]["candidates"][2]["half_life_days"] == 120.0
    assert artifact["private_scope"]["authorizes"] == ["private_model_fit", "private_rank_selection"]
    assert artifact["posterior_ratings"] == [] if artifact["result_state"] == "NO_WINNER" else artifact["posterior_ratings"]
    for candidate in artifact["candidate_results"]:
        assert candidate["development"]["n"] == 1
        assert candidate["validation"]["n"] == 1
        assert candidate["diagnostics"]["covariance"] == "DIAGONAL_ASSUMED_DENSITY_PSD"
        assert candidate["diagnostics"]["convergence_kind"] == "ONE_STEP_ADF_LAPLACE_NO_ITERATIVE_OPTIMIZER"
    assert verify_private_development_artifact(artifact, expected_artifact_sha256=artifact["artifact_sha256"])["artifact_sha256"] == artifact["artifact_sha256"]


def test_reordered_missing_and_future_origins_fail_closed_or_do_not_affect_dev_selection() -> None:
    first = build_private_development_artifact(input_loader=_input)
    second = build_private_development_artifact(input_loader=lambda: _input(validation_y=1))
    assert first["decision"]["development_winner_candidate_id"] == second["decision"]["development_winner_candidate_id"]
    assert [item["development"] for item in first["candidate_results"]] == [item["development"] for item in second["candidate_results"]]
    with pytest.raises(PrivateDevelopmentError, match="origin digest"):
        build_private_development_artifact(input_loader=lambda: _input(reordered_development_origins=True))
    input_data = _input()
    validation_fold = input_data.folds[-1]
    validation = validation_fold.map_observations[0]
    illegal_origins = (*validation.ordered_origin_map_ids, validation.source_game_id)
    illegal_map = replace(validation, ordered_origin_map_ids=illegal_origins, ordered_origin_sha256=_sha(list(illegal_origins)))
    illegal_fold = replace(
        validation_fold,
        map_observations=(illegal_map,),
        ordered_origin_identities_sha256=_sha([{"source_game_id": illegal_map.source_game_id, "ordered_origin_map_ids": list(illegal_origins)}]),
    )
    illegal_input = replace(input_data, folds=(*input_data.folds[:2], illegal_fold))
    with pytest.raises(PrivateDevelopmentError, match="48-hour/source-series"):
        build_private_development_artifact(input_loader=lambda: illegal_input)


def test_nonfinite_psd_and_artifact_tampering_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = build_private_development_artifact(input_loader=_input)
    tampered = deepcopy(artifact)
    tampered["output_checks"]["all_finite"] = False
    tampered["artifact_sha256"] = _sha({key: value for key, value in tampered.items() if key != "artifact_sha256"})
    with pytest.raises(PrivateDevelopmentError, match="finite/PSD"):
        verify_private_development_artifact(tampered, expected_artifact_sha256=tampered["artifact_sha256"])
    with pytest.raises(PrivateDevelopmentError, match="independently pinned"):
        verify_private_development_artifact(artifact, expected_artifact_sha256="0" * 64)
    self_rehashed = deepcopy(artifact)
    self_rehashed["decision"]["selected_candidate_id"] = None
    self_rehashed["artifact_sha256"] = _sha({key: value for key, value in self_rehashed.items() if key != "artifact_sha256"})
    with pytest.raises(PrivateDevelopmentError, match="independently pinned"):
        verify_private_development_artifact(self_rehashed, expected_artifact_sha256=artifact["artifact_sha256"])
    monkeypatch.setattr(runner, "CANDIDATES", (Candidate("bad", "RANDOM_WALK", None, float("nan")),))
    with pytest.raises(PrivateDevelopmentError, match="non-finite|canonical"):
        build_private_development_artifact(input_loader=_input)


def test_private_runner_is_byte_identical_across_two_fresh_processes() -> None:
    root = Path(__file__).parents[4]
    test_path = Path(__file__).resolve()
    code = (
        "import importlib.util,sys; "
        "from lol_kills.v2.ratings.player.private_development_runner import build_private_development_artifact,_canonical_bytes; "
        "s=importlib.util.spec_from_file_location('fixture',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "sys.stdout.buffer.write(_canonical_bytes(build_private_development_artifact(input_loader=m._input)))"
    )
    command = [sys.executable, "-c", code, str(test_path)]
    first = subprocess.run(command, cwd=root, check=True, capture_output=True).stdout
    second = subprocess.run(command, cwd=root, check=True, capture_output=True).stdout
    assert first == second


def test_persisted_real_artifact_has_pinned_digest_and_default_replay_is_identical() -> None:
    root = Path(__file__).parents[4]
    artifact_path = root / "data/lol/v2/models/player/real-v1/private-development-artifact-v3.json"
    expected = "510d2cde52a92f92f6aa373bbe5c497d2b9dc652d1f7edf15f9cae006ee0f7a0"
    persisted = verify_private_development_artifact(json.loads(artifact_path.read_text(encoding="utf-8")), expected_artifact_sha256=expected)
    assert persisted["decision"] == {
        "development_winner_candidate_id": "static_baseline",
        "external_validation_gate_passed": True,
        "selected_candidate_id": "static_baseline",
    }
    code = (
        "from lol_kills.v2.ratings.player.private_development_runner import build_private_development_artifact; "
        "print(build_private_development_artifact()['artifact_sha256'])"
    )
    command = [sys.executable, "-c", code]
    first = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    assert first == second == expected + "\n"


def test_private_artifact_writer_rejects_output_aliases(tmp_path: Path) -> None:
    artifact = build_private_development_artifact(input_loader=_input)
    backing = tmp_path / "backing.json"
    backing.write_bytes(b"keep")
    alias = tmp_path / "alias.json"
    alias.symlink_to(backing)
    with pytest.raises(PrivateDevelopmentError, match="symlink"):
        write_private_development_artifact(artifact, alias)
    assert backing.read_bytes() == b"keep"
