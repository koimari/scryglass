from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import lol_kills.v2.draft.interactions.oe_target_authority as target_authority
import lol_kills.v2.draft.interactions.oe_target_evidence as target


DATES = (
    "2025-09-01T10:00:00",
    "2025-11-01T10:00:00",
    "2026-05-01T10:00:00",
    "2026-06-15T10:00:00",
)
ROLES = ("top", "jng", "mid", "bot", "sup")


def _maps(*, durations: tuple[object, ...] = (1800, 1900, 2000, 2100)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oe_gameid": f"game-{index}",
                "game_uid": f"game-{index}",
                "league": "lec",
                "date": pd.Timestamp(date),
                "patch": 16.1 + index / 100,
                "blue_result": index % 2,
                "red_result": 1 - index % 2,
                "y_blue_win": index % 2,
                "gamelength": durations[index],
            }
            for index, date in enumerate(DATES)
        ],
        columns=target.MAP_COLUMNS,
    )


def _proxy() -> dict:
    return {
        "artifact_sha256": target.PINNED_PROXY_PAYLOAD_SHA256,
        "assignments": [
            {"game_id": f"game-{index}", "dependence_cluster_id": f"cluster-{index}"}
            for index in range(4)
        ],
        "eligibility": {
            "assigned_maps": 4,
            "excluded_maps": 2,
            "exclusion_ledger": [
                {"game_id": "collision-a", "reason": "exact_context_time_game_collision"},
                {"game_id": "collision-b", "reason": "exact_context_time_game_collision"},
            ],
        },
    }


def _crosswalk() -> dict:
    return {
        "artifact_sha256": target.PINNED_CROSSWALK_PAYLOAD_SHA256,
        "entries": [
            {
                "normalized_oe_name": f"champion {number}",
                "stable_champion_id": f"riot:champion:{number}",
            }
            for number in range(1, 11)
        ],
    }


def _players() -> pd.DataFrame:
    rows = []
    for game in range(4):
        for side_index, side in enumerate(("Blue", "Red")):
            for role_index, role in enumerate(ROLES):
                number = side_index * 5 + role_index + 1
                rows.append(
                    {
                        "gameid": f"game-{game}",
                        "participantid": side_index * 5 + role_index + 1,
                        "side": side,
                        "position": role,
                        "champion": f"Champion {number}",
                    }
                )
    return pd.DataFrame(rows, columns=target.PLAYER_COLUMNS)


def _teams() -> pd.DataFrame:
    rows = []
    for game in range(4):
        blue = game % 2
        rows.extend(
            [
                {"gameid": f"game-{game}", "participantid": 100, "side": "Blue", "result": blue},
                {"gameid": f"game-{game}", "participantid": 200, "side": "Red", "result": 1 - blue},
            ]
        )
    return pd.DataFrame(rows, columns=target.TEAM_COLUMNS)


def _annual() -> dict[int, pd.DataFrame]:
    by_year: dict[int, list[dict]] = {2025: [], 2026: []}
    for game, date in enumerate(DATES):
        year = 2025 if game < 2 else 2026
        blue = game % 2
        for participant in range(1, 11):
            side = "Blue" if participant <= 5 else "Red"
            role = ROLES[(participant - 1) % 5]
            by_year[year].append(
                {
                    "gameid": f"game-{game}",
                    "participantid": participant,
                    "side": side,
                    "position": role,
                    "champion": f"Champion {participant}",
                    "result": blue if side == "Blue" else 1 - blue,
                    "date": date,
                    "patch": 16.1 + game / 100,
                    "league": "LEC",
                    "gamelength": 1800 + game * 100,
                }
            )
        for participant, side, result in ((100, "Blue", blue), (200, "Red", 1 - blue)):
            by_year[year].append(
                {
                    "gameid": f"game-{game}", "participantid": participant,
                    "side": side, "position": "team", "champion": None,
                    "result": result, "date": date, "patch": 16.1 + game / 100,
                    "league": "LEC", "gamelength": 1800 + game * 100,
                }
            )
    return {
        year: pd.DataFrame(rows, columns=target.RAW_COLUMNS)
        for year, rows in by_year.items()
    }


def _manifest() -> dict:
    return {
        "raw_annual_oe_csv": {
            "2025": {"locator": "2025.csv", "raw_sha256": "1" * 64, "selected_input_sha256": "2" * 64},
            "2026": {"locator": "2026.csv", "raw_sha256": "3" * 64, "selected_input_sha256": "4" * 64},
        },
        "warehouse_parquet": {
            name: {"locator": f"{name}.parquet", "raw_sha256": digit * 64, "selected_input_sha256": digit * 64}
            for name, digit in (("maps", "5"), ("teams", "6"), ("players", "7"))
        },
        "pinned_inputs": {
            name: {"locator": f"{name}.json", "raw_sha256": digit * 64, "payload_sha256": digit * 64}
            for name, digit in (("preflight", "8"), ("proxy", "9"), ("crosswalk", "a"))
        },
    }


def _build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, maps: pd.DataFrame | None = None):
    monkeypatch.setattr(target, "EXPECTED_ASSIGNED_MAPS", 4)
    frame = _maps() if maps is None else maps
    proxy = _proxy()
    split = target.build_outcome_free_split(
        frame[["oe_gameid", "date"]],
        proxy,
        maps_source={"locator": "maps", "raw_sha256": "b" * 64},
        proxy_source={"locator": "proxy", "payload_sha256": target.PINNED_PROXY_PAYLOAD_SHA256},
    )
    evidence = target.analyze_frames(
        frame,
        _teams(),
        _players(),
        _annual(),
        proxy,
        _crosswalk(),
        split,
        source_manifest=_manifest(),
        private_rows_path=tmp_path / "private.parquet",
    )
    return split, evidence


def test_outcome_free_split_is_cluster_atomic_strictly_chronological_and_final_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    split, _ = _build(monkeypatch, tmp_path)
    assert split["counts"]["by_split"] == {
        "development": 1,
        "final_temporal_holdout": 1,
        "train": 1,
        "validation": 1,
    }
    assert split["counts"]["maps"] == 4
    assert split["once_reserved_final_temporal_holdout"] is True
    assert len({row["dependence_cluster_id"] for row in split["assignments"]}) == 4
    ordered = [split["chronology"][name] for name in ("train", "development", "validation", target.FINAL_SPLIT)]
    assert all(left["maximum_date_naive"] < right["minimum_date_naive"] for left, right in zip(ordered, ordered[1:]))


def test_label_mutation_changes_target_domain_and_ledger_not_membership_features_or_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    split, first = _build(monkeypatch, tmp_path / "first")
    changed_maps = _maps()
    changed_maps.loc[0, ["y_blue_win", "blue_result", "red_result"]] = [1, 1, 0]
    changed_teams = _teams()
    changed_teams.loc[changed_teams["gameid"] == "game-0", "result"] = [1, 0]
    changed_annual = _annual()
    mask = (changed_annual[2025]["gameid"] == "game-0") & changed_annual[2025]["participantid"].isin((100, 200))
    changed_annual[2025].loc[mask, "result"] = [1, 0]
    second = target.analyze_frames(
        changed_maps, changed_teams, _players(), changed_annual, _proxy(), _crosswalk(), split,
        source_manifest=_manifest(), private_rows_path=tmp_path / "second.parquet",
    )
    assert first["target_domain_sha256"] != second["target_domain_sha256"]
    assert first["consistency_ledger_sha256"] != second["consistency_ledger_sha256"]
    for field in ("membership_sha256", "dependence_assignment_sha256", "feature_domain_sha256"):
        assert first[field] == second[field]
    assert split["artifact_sha256"] == first["split_assignment"]["payload_sha256"]


def test_target_disagreement_rejects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(target, "EXPECTED_ASSIGNED_MAPS", 4)
    maps = _maps()
    maps.loc[0, "red_result"] = maps.loc[0, "blue_result"]
    split = target.build_outcome_free_split(
        maps[["oe_gameid", "date"]], _proxy(), maps_source={}, proxy_source={}
    )
    with pytest.raises(target.OETargetEvidenceError, match="semantics disagree"):
        target.analyze_frames(
            maps, _teams(), _players(), _annual(), _proxy(), _crosswalk(), split,
            source_manifest=_manifest(), private_rows_path=tmp_path / "bad.parquet",
        )


def test_raw_annual_oe_target_disagreement_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(target, "EXPECTED_ASSIGNED_MAPS", 4)
    maps = _maps()
    split = target.build_outcome_free_split(
        maps[["oe_gameid", "date"]], _proxy(), maps_source={}, proxy_source={}
    )
    annual = _annual()
    annual[2025].loc[
        (annual[2025]["gameid"] == "game-0")
        & (annual[2025]["participantid"] == 100),
        "result",
    ] = 1
    with pytest.raises(target.OETargetEvidenceError, match="raw annual OE outcome"):
        target.analyze_frames(
            maps, _teams(), _players(), annual, _proxy(), _crosswalk(), split,
            source_manifest=_manifest(), private_rows_path=tmp_path / "bad.parquet",
        )


def test_duration_mutation_or_missingness_only_changes_resolution_materialization_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    split, first = _build(monkeypatch, tmp_path / "first")
    changed = _maps(durations=(None, 9999, 2000, 2100))
    second = target.analyze_frames(
        changed, _teams(), _players(), _annual(), _proxy(), _crosswalk(), split,
        source_manifest=_manifest(), private_rows_path=tmp_path / "second.parquet",
    )
    for field in ("membership_sha256", "dependence_assignment_sha256", "feature_domain_sha256", "target_domain_sha256", "target_transform_sha256"):
        assert first[field] == second[field]
    assert first["derived_resolution_annotation_sha256"] != second["derived_resolution_annotation_sha256"]
    assert first["private_materialization"]["raw_sha256"] != second["private_materialization"]["raw_sha256"]


def test_stable_champion_ids_exact_oe_patch_tokens_and_no_invented_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, evidence = _build(monkeypatch, tmp_path)
    rows = pd.read_parquet(evidence["private_materialization"]["locator"])
    assert set(rows["oe_patch_token"]) == {"16.10", "16.11", "16.12", "16.13"}
    assert rows.filter(like="_stable_champion_id").map(lambda x: x.startswith("riot:champion:")).all().all()
    assert rows["draft_completed_at"].isna().all()
    assert rows["forecast_at"].isna().all()
    assert evidence["historical_live_forecast_claim"] is False


def test_collision_exclusions_and_authority_flags_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, evidence = _build(monkeypatch, tmp_path)
    assert evidence["population"]["collision_exclusions"] == _proxy()["eligibility"]["exclusion_ledger"]
    for field in (
        "predictive_target_authority", "authorizes_model_fit", "authorizes_rank_selection",
        "authorizes_prediction", "authorizes_publication", "authorizes_production",
        "authorizes_sota_claim", "content_addressing_confers_authority",
    ):
        assert evidence[field] is False


def test_no_self_authorization_and_caller_rehash_cannot_create_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    split, evidence = _build(monkeypatch, tmp_path)
    with pytest.raises(target.OETargetEvidenceError, match="missing"):
        target.require_exact_human_authority(None, evidence, split, action="rank_selection")
    generator_authored = {
        "schema_id": "scryglass.oe-private-target-human-authority.v1",
        "decision_id": "human-decision-1",
        "reviewer_identity": "independent-reviewer",
        "reviewed_at_rfc3339": "2026-07-28T18:00:00Z",
        "approval_scope": "private_retrospective_oe_target_v1",
        "decision": "approve",
        "source_rights_reviewed": True,
        "target_semantics_reviewed": True,
        "temporal_leakage_reviewed": True,
        "fixed_boundaries_reviewed": True,
        "generator_authored": True,
        "independent_from_generator": False,
        "evidence_payload_sha256": evidence["artifact_sha256"],
        "split_payload_sha256": split["artifact_sha256"],
        "approved_actions": ["rank_selection"],
    }
    with pytest.raises(target.OETargetEvidenceError, match="generator-authored"):
        target.require_exact_human_authority(
            json.dumps(generator_authored).encode(), evidence, split, action="rank_selection"
        )
    caller_rehashed = {
        **generator_authored,
        "generator_authored": False,
        "independent_from_generator": True,
    }
    with pytest.raises(target.OETargetEvidenceError, match="caller-rehashed"):
        target.require_exact_human_authority(
            json.dumps(caller_rehashed).encode(), evidence, split, action="rank_selection"
        )


def test_exact_human_authority_is_pinned_to_reviewed_payloads_and_actions() -> None:
    evidence = json.loads(target.DEFAULT_EVIDENCE_PATH.read_bytes())
    split = json.loads(target.DEFAULT_SPLIT_PATH.read_bytes())
    for action in ("model_fit", "rank_selection"):
        approval = target_authority.load_and_require_exact_human_authority(
            evidence,
            split,
            action=action,
        )
        assert approval["reviewer_identity"] == "KOI_MARI"
        assert approval["final_temporal_holdout_sealed"] is True
    with pytest.raises(
        target.OETargetEvidenceError, match="requested action lacks human approval"
    ):
        target_authority.load_and_require_exact_human_authority(
            evidence,
            split,
            action="prediction",
        )

    changed = json.loads(
        target_authority.DEFAULT_HUMAN_AUTHORITY_PATH.read_bytes()
    )
    changed["decision_id"] += ":caller-change"
    with pytest.raises(target.OETargetEvidenceError, match="caller-rehashed"):
        target_authority.require_exact_human_authority(
            target.canonical_bytes(changed),
            evidence,
            split,
            action="rank_selection",
        )


def test_caller_rehashed_evidence_mutation_is_not_the_persisted_source_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, evidence = _build(monkeypatch, tmp_path)
    changed = copy.deepcopy(evidence)
    changed["feature_domain_sha256"] = "0" * 64
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = target.canonical_sha256(changed)
    target.validate_evidence(changed)  # integrity is not authority
    assert changed["artifact_sha256"] != evidence["artifact_sha256"]
    assert changed["content_addressing_confers_authority"] is False


def test_private_materialization_path_is_expected_ignored() -> None:
    path = target.DEFAULT_PRIVATE_ROWS_PATH.as_posix()
    result = subprocess.run(
        ["git", "check-ignore", path],
        cwd=Path(__file__).resolve().parents[4],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == path


def test_real_source_replay_rejects_caller_rehashed_evidence_mutation(
    tmp_path: Path,
) -> None:
    split = json.loads(target.DEFAULT_SPLIT_PATH.read_text())
    evidence = json.loads(target.DEFAULT_EVIDENCE_PATH.read_text())
    changed = copy.deepcopy(evidence)
    changed["feature_domain_sha256"] = "0" * 64
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = target.canonical_sha256(changed)
    split_path = tmp_path / "split.json"
    evidence_path = tmp_path / "evidence.json"
    split_path.write_bytes(target.canonical_bytes(split))
    evidence_path.write_bytes(target.canonical_bytes(changed))
    with pytest.raises(target.OETargetEvidenceError, match="source-backed replay"):
        target.load_and_replay_artifacts(
            split_path, evidence_path, source_root=Path(__file__).resolve().parents[4]
        )


def test_real_source_replay_rejects_rehashed_private_target_row_tamper(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    split = json.loads(target.DEFAULT_SPLIT_PATH.read_text())
    evidence = json.loads(target.DEFAULT_EVIDENCE_PATH.read_text())
    private_path = tmp_path / "tampered-private.parquet"
    shutil.copy2(root / target.DEFAULT_PRIVATE_ROWS_PATH, private_path)
    rows = pd.read_parquet(private_path)
    rows.loc[0, "y_blue_win"] = 1 - int(rows.loc[0, "y_blue_win"])
    rows.to_parquet(private_path, index=False)
    changed = copy.deepcopy(evidence)
    changed["private_materialization"].update(
        {
            "locator": private_path.as_posix(),
            "raw_sha256": target.raw_sha256(private_path),
            "logical_rows_sha256": target.selected_input_sha256(rows),
            "ordered_logical_rows_sha256": target.ordered_input_sha256(rows),
        }
    )
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = target.canonical_sha256(changed)
    split_path = tmp_path / "split.json"
    evidence_path = tmp_path / "evidence.json"
    split_path.write_bytes(target.canonical_bytes(split))
    evidence_path.write_bytes(target.canonical_bytes(changed))
    with pytest.raises(
        target.OETargetEvidenceError,
        match="source-backed private logical rows",
    ):
        target.load_and_replay_artifacts(
            split_path, evidence_path, source_root=root
        )
