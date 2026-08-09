from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

import lol_kills.v2.ratings.team.real_v1_private_runner as team
from lol_kills.v2.ratings.player.real_v1_adapter import ACCEPTED_G1_PINS


TEAM_ARTIFACT_SHA256 = "d4874f12a479dabdabf9c11c537058ddf4c3fe7c0ffa3225a0b46f7229f86b79"
TEAM_ARTIFACT_RAW_SHA256 = "c71b5d50564a0cc55acc8c46698eb52ed23504c67257069b06d449f262e0acf5"


def _accepted_player() -> dict:
    return team.load_accepted_static_player_artifact()


def _roster(player: dict) -> dict:
    development = player["development_winner_posterior_ratings"]
    return {
        "organization_id": "private-lpl-team",
        "league_id": "LPL",
        "as_of_source_game_id": development["as_of_source_game_id"],
        "identity_receipt_sha256": "a" * 64,
        "official": True,
        "active": True,
        "fresh": True,
        "ambiguous": False,
        "substitute": False,
        "players": [
            {"role": role, "player_id": rating["player_id"]}
            for role, rating in zip(team.ROLES, development["ratings"][:5])
        ],
    }


def test_default_private_team_artifact_is_pinned_unavailable_and_carries_exact_g2_pins() -> None:
    artifact = team.build_private_team_real_v1_artifact()
    verified = team.verify_private_team_real_v1_artifact(artifact, expected_artifact_sha256=TEAM_ARTIFACT_SHA256)
    assert verified["result_state"] == "UNAVAILABLE"
    assert verified["aggregation"]["status"] == "unavailable"
    assert verified["predictive_comparison"]["status"] == "unavailable"
    carried_baseline = verified["predictive_comparison"]["accepted_player_only_static_baseline"]
    assert {"development", "validation", "fold_prediction_sha256"} <= set(carried_baseline)
    assert carried_baseline["development"]["status"] == "development_only"
    assert carried_baseline["validation"]["status"] == "development_only"
    assert carried_baseline["development"]["n"] > 0
    assert carried_baseline["validation"]["n"] > 0
    assert set(carried_baseline["fold_prediction_sha256"]) == {"DEVELOPMENT", "VALIDATION"}
    pins = verified["accepted_player_artifact"]["adapter_input_pins"]
    assert pins["manifest_sha256"] == ACCEPTED_G1_PINS.manifest_sha256
    assert pins["rows_sha256"] == ACCEPTED_G1_PINS.rows_sha256
    assert pins["selected_target_sha256"] == ACCEPTED_G1_PINS.selected_target_sha256
    assert pins["split_payload_sha256"] == ACCEPTED_G1_PINS.split_payload_sha256
    assert {"forecast", "publication", "team_rank", "final_holdout"} <= set(verified["private_scope"]["blocked"])


def test_checked_private_team_artifact_has_frozen_raw_and_canonical_identities() -> None:
    raw = team.TEAM_ARTIFACT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TEAM_ARTIFACT_RAW_SHA256
    parsed = json.loads(raw)
    verified = team.verify_private_team_real_v1_artifact(parsed, expected_artifact_sha256=TEAM_ARTIFACT_SHA256)
    assert verified["artifact_sha256"] == TEAM_ARTIFACT_SHA256


def test_exact_ordered_current_five_propagates_full_covariance_and_never_zeroes_unidentified_components() -> None:
    player = _accepted_player()
    roster = _roster(player)
    aggregate = team.aggregate_exact_current_lpl_five(
        roster,
        expected_identity_receipt_sha256="a" * 64,
        player_artifact=player,
    )
    assert aggregate["roles"] == list(team.ROLES)
    assert len(aggregate["player_ids"]) == 5 == len(set(aggregate["player_ids"]))
    covariance = aggregate["player_display_covariance"]
    assert len(covariance) == 5 and all(len(row) == 5 for row in covariance)
    assert aggregate["team_posterior_display_variance"] == pytest.approx(sum(covariance[i][i] for i in range(5)) / 25.0)
    assert aggregate["team_posterior_display_uncertainty"] > 0.0
    assert aggregate["scale"] == {"display_anchor": 1500.0, "display_scale": pytest.approx(400.0 / math.log(10.0)), "team_latent_definition": "mean_of_five_player_latents"}
    for name in ("lineup_synergy", "policy", "league_rating"):
        assert aggregate["components"][name]["status"] == "unavailable"
        assert aggregate["components"][name]["value"] is None


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda roster: roster["players"].__setitem__(1, dict(roster["players"][0])), "EXACT_ROSTER_ROLE_OR_PLAYER_IDENTITY_MISMATCH"),
        (lambda roster: roster.__setitem__("substitute", True), "EXACT_ROSTER_SUBSTITUTE"),
        (lambda roster: roster.__setitem__("fresh", False), "EXACT_ROSTER_INACTIVE_OR_STALE"),
        (lambda roster: roster.__setitem__("as_of_source_game_id", "future-map"), "EXACT_ROSTER_STALE_OR_AS_OF_MISMATCH"),
        (lambda roster: roster.__setitem__("identity_receipt_sha256", "bad"), "EXACT_ROSTER_IDENTITY_RECEIPT_MISSING"),
    ],
)
def test_roster_identity_substitute_and_staleness_fail_with_typed_unavailable(mutate, code: str) -> None:
    player = _accepted_player()
    roster = _roster(player)
    mutate(roster)
    with pytest.raises(team.TeamRealV1Unavailable, match=code):
        team.aggregate_exact_current_lpl_five(
            roster,
            expected_identity_receipt_sha256="a" * 64,
            player_artifact=player,
        )


def test_exact_roster_requires_an_independent_external_identity_receipt_pin() -> None:
    player = _accepted_player()
    with pytest.raises(team.TeamRealV1Unavailable, match="EXACT_ROSTER_EXTERNAL_RECEIPT_PIN_REQUIRED"):
        team.aggregate_exact_current_lpl_five(_roster(player), player_artifact=player)


def test_future_player_mutation_and_self_rehash_cannot_bypass_exact_player_or_team_pins() -> None:
    player = _accepted_player()
    mutated_player = deepcopy(player)
    mutated_player["development_winner_posterior_ratings"]["ratings"][0]["posterior_mean"] += 400.0
    # Even without repairing the inner digest, the injected future mutation is
    # rejected against the accepted external Player artifact pin.
    with pytest.raises(Exception, match="digest|pin|artifact"):
        team.aggregate_exact_current_lpl_five(
            _roster(player),
            expected_identity_receipt_sha256="a" * 64,
            player_artifact=mutated_player,
        )
    artifact = team.build_private_team_real_v1_artifact()
    self_rehashed = deepcopy(artifact)
    self_rehashed["private_scope"]["blocked"].remove("publication")
    self_rehashed["artifact_sha256"] = team._sha256({key: value for key, value in self_rehashed.items() if key != "artifact_sha256"})
    with pytest.raises(team.TeamRealV1Error, match="EXTERNAL_PIN"):
        team.verify_private_team_real_v1_artifact(self_rehashed, expected_artifact_sha256=TEAM_ARTIFACT_SHA256)


def test_default_team_artifact_is_byte_identical_across_two_fresh_processes() -> None:
    root = Path(__file__).parents[4]
    code = (
        "from lol_kills.v2.ratings.team.real_v1_private_runner import build_private_team_real_v1_artifact; "
        "print(build_private_team_real_v1_artifact()['artifact_sha256'])"
    )
    command = [sys.executable, "-c", code]
    first = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    assert first == second == TEAM_ARTIFACT_SHA256 + "\n"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_writer_rejects_existing_output_aliases_without_touching_their_backing_file(tmp_path: Path, alias_kind: str) -> None:
    backing = tmp_path / "backing.json"
    backing.write_text("do-not-replace", encoding="utf-8")
    output = tmp_path / "artifact.json"
    if alias_kind == "symlink":
        output.symlink_to(backing)
    else:
        os.link(backing, output)

    with pytest.raises(team.TeamRealV1Error, match="TEAM_ARTIFACT_OUTPUT_UNSAFE"):
        team.write_private_team_real_v1_artifact(output)
    assert backing.read_text(encoding="utf-8") == "do-not-replace"


def test_writer_rejects_a_symlinked_parent_before_writing_to_its_target(tmp_path: Path) -> None:
    backing_parent = tmp_path / "backing-parent"
    backing_parent.mkdir()
    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.symlink_to(backing_parent, target_is_directory=True)

    with pytest.raises(team.TeamRealV1Error, match="TEAM_ARTIFACT_PARENT_UNSAFE"):
        team.write_private_team_real_v1_artifact(unsafe_parent / "artifact.json")
    assert not (backing_parent / "artifact.json").exists()


def test_writer_rejects_a_symlinked_ancestor_before_writing_to_its_target(tmp_path: Path) -> None:
    trusted_parent = tmp_path / "trusted-parent"
    trusted_parent.mkdir()
    backing_parent = tmp_path / "backing-parent"
    backing_parent.mkdir()
    unsafe_ancestor = trusted_parent / "unsafe-ancestor"
    unsafe_ancestor.symlink_to(backing_parent, target_is_directory=True)

    with pytest.raises(team.TeamRealV1Error, match="TEAM_ARTIFACT_PARENT_UNSAFE"):
        team.write_private_team_real_v1_artifact(unsafe_ancestor / "nested" / "artifact.json")
    assert not (backing_parent / "nested" / "artifact.json").exists()


def test_writer_rejects_a_non_directory_parent_with_a_typed_error(tmp_path: Path) -> None:
    unsafe_parent = tmp_path / "not-a-directory"
    unsafe_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(team.TeamRealV1Error, match="TEAM_ARTIFACT_PARENT_UNSAFE"):
        team.write_private_team_real_v1_artifact(unsafe_parent / "artifact.json")
