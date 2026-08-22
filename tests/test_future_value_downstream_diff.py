from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.future_value_downstream_diff import (
    BASELINE_VARIANT,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    VARIANT_SPECS,
    DownstreamDiffError,
    VariantSpec,
    compare_downstream_variants,
    draft_score_component_diff_metrics,
    numeric_delta_stats,
    rank_movement_metrics,
    tier_transition_metrics,
    write_downstream_diff_report,
)
from lol_kills.research.future_value_downstream import (
    SOURCE_RECEIPT_SCHEMA_VERSION,
    SOURCE_RECEIPT_STATUS,
)
from lol_kills.research.future_value_rating import RatingVariant
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE = {
    "source_as_of": "2026-08-20T11:31:37Z",
    "source_game_count": 2,
    "source_identity_sha256": identity_sha256(("game-1", "game-2")),
    "accepted_game_ids": ["game-1", "game-2"],
}
_RECEIPT_PAYLOAD = {
    "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
    "status": SOURCE_RECEIPT_STATUS,
    **SOURCE,
    "source_files": {
        "fixture": {
            "bytes": 0,
            "sha256": "0" * 64,
            "locator": "fixture-source.csv",
        }
    },
    "authority": {"research_only": True},
}
SOURCE_RECEIPT = {
    **_RECEIPT_PAYLOAD,
    "receipt_sha256": hashlib.sha256(
        json.dumps(
            _RECEIPT_PAYLOAD,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest(),
}


def _write_json(root: Path, relative: str, value: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8")


def _bound(value: object) -> object:
    common = {
        "source_as_of": SOURCE["source_as_of"],
        "source_game_count": SOURCE["source_game_count"],
        "source_identity_sha256": SOURCE["source_identity_sha256"],
        "accepted_game_ids": list(SOURCE["accepted_game_ids"]),
        "source_receipt_sha256": SOURCE_RECEIPT["receipt_sha256"],
    }
    if isinstance(value, list):
        return {"rows": value, **common}
    if isinstance(value, dict):
        return {**value, **common}
    raise TypeError(value)


def _draft_row(variant: str) -> dict[str, object]:
    future = 1.0 if variant in {"future_player_form", "both"} else 0.0
    scaling = 0.25 if variant in {"scaling_curve", "both"} else 0.0
    independent = {
        "base": 1.0,
        "synergy": 2.0,
        "counter": 3.0,
        "same_role": 4.0,
        "ally_synergy": 0.5,
        "enemy_counter": 0.5,
        "archetype_interactions": 0.5,
        "current_rating_logit": 0.5,
        "future_player_form_logit": future,
        "scaling_raw_logit": scaling,
        "scaling_shape_logit": scaling,
        "curve_atom_interaction_logit": scaling,
        "crossfit_composition_total": 1.0,
    }
    crossfit_parts = {
        "crossfit_champion_main": 0.1,
        "crossfit_role_champion": 0.1,
        "crossfit_ally_synergy": 0.1,
        "crossfit_archetype_synergy": 0.1,
        "crossfit_enemy_counter": 0.1,
        "crossfit_archetype_counter": 0.1,
        "crossfit_same_role": 0.1,
    }
    total = sum(independent.values())
    return {
        "game_uid": "game-1",
        **independent,
        **crossfit_parts,
        "composite_logit": total,
        "score": total,
        "draft_edge": total,
        "target": 1,
    }


def _write_variant(root: Path, *, variant: str, row_loss: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root, "future-value-source-receipt.json", SOURCE_RECEIPT)
    future_value = 1.0 if variant in {"future_player_form", "both"} else 0.0
    scaling_value = 0.25 if variant in {"scaling_curve", "both"} else 0.0
    players = [
        {"player_id": "A", "mu_total": 100.0, "future_value": future_value, "future_player_form_logit": future_value, "forecast_gold_diff_10": scaling_value},
        {"player_id": "B", "mu_total": 90.0, "future_value": future_value, "future_player_form_logit": future_value, "forecast_gold_diff_10": scaling_value},
    ]
    if row_loss:
        players.pop()
    _write_json(root, "features/player_ratings_snapshot.json", _bound(players))
    _write_json(
        root,
        "features/ratings_snapshot.json",
        _bound([
            {"team_key": "alpha", "team": "Alpha", "mu_total": 110.0},
            {"team_key": "beta", "team": "Beta", "mu_total": 95.0},
        ]),
    )
    _write_json(
        root,
        "rankings/tierlists.json",
        _bound({
            "rows": [
                {"champion": "Aatrox", "role": "top", "scope": "LCK", "patch": "16.16", "rank": 1, "tier": "S", "score": 0.5},
                {"champion": "Ahri", "role": "mid", "scope": "LCK", "patch": "16.16", "rank": 2, "tier": "A", "score": 0.4},
            ]
        }),
    )
    _write_json(root, "features/draft_records.json", _bound({"games": [_draft_row(variant)]}))
    _write_json(
        root,
        "features/profile_records.json",
        _bound({"games": [{"game_uid": "game-1", "future_value": 0.5, "forecast_gold_diff_10": scaling_value}]}),
    )
    _write_json(
        root,
        "features/match_records_2026_q3.json",
        _bound({"games": [{"game_uid": "game-1", "future_team_value": 0.2, "blue_result": 1}]}),
    )
    _write_json(root, "manifest.json", _bound({"pack_id": "pack-1", "total_files": 7, "total_bytes": 100}))


def _materialise_roots(tmp_path: Path) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in VARIANT_NAMES}
    for name, root in roots.items():
        _write_variant(root, variant=name)
    return roots


def test_variant_registry_is_exact_and_immutable() -> None:
    assert tuple(VARIANT_SPECS) == VARIANT_NAMES
    assert tuple(spec.ordinal for spec in VARIANT_SPECS.values()) == (1, 2, 3, 4)
    with pytest.raises(TypeError):
        VARIANT_SPECS["extra"] = VARIANT_SPECS[BASELINE_VARIANT]  # type: ignore[index]
    with pytest.raises((AttributeError, DownstreamDiffError)):
        VARIANT_SPECS[BASELINE_VARIANT].changed_families += ("scaling_curve",)  # type: ignore[misc]
    assert VariantSpec(BASELINE_VARIANT, RatingVariant.CURRENT_ONLY, 1, ()).label == "V1"


def test_numeric_and_rank_metrics_are_deterministic() -> None:
    assert numeric_delta_stats([1.0, 2.0], [2.0, 2.0])["mean"] == pytest.approx(0.5)
    result = rank_movement_metrics(
        [{"id": "a", "score": 3.0}, {"id": "b", "score": 2.0}, {"id": "c", "score": 1.0}],
        [{"id": "b", "score": 3.0}, {"id": "a", "score": 2.0}, {"id": "c", "score": 1.0}],
        "score",
    )
    assert result["spearman"] == pytest.approx(0.5)
    assert result["pairwise_inversions"] == 1
    assert result["changed_fraction"] == pytest.approx(2 / 3)
    assert result["top_5_overlap"] == 1.0


def test_tier_and_draft_component_metrics() -> None:
    old = [{"id": "a", "tier": "S", **_draft_row(BASELINE_VARIANT)}]
    candidate = [{"id": "a", "tier": "A", **_draft_row(BASELINE_VARIANT)}]
    tier = tier_transition_metrics(old, candidate, identity_fields=("id",))
    assert tier["changed_count"] == 1
    assert tier["transitions"] == {"S->A": 1}
    draft = draft_score_component_diff_metrics(old, candidate, identity_fields=("id",))
    assert draft["static_components_identical"] is True
    assert draft["reconstruction"]["candidate"]["status"] == "passed"


def test_four_variant_report_keeps_authority_closed(tmp_path: Path) -> None:
    report = compare_downstream_variants(_materialise_roots(tmp_path), source_receipt=SOURCE_RECEIPT)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ready_research_only"
    assert set(report["variants"]) == set(VARIANT_NAMES)
    assert not any(report["authority"].values())
    assert report["draft_score"]["future_player_form"]["component_deltas"]["base"]["changed_count"] == 0
    assert report["artifacts"]["player_ratings"]["comparisons"]["future_player_form"]["deltas"]["future_value"]["changed_count"] == 2


def test_row_loss_and_source_mismatch_block(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    _write_variant(roots["scaling_curve"], variant="scaling_curve", row_loss=True)
    report = compare_downstream_variants(roots, source_receipt=SOURCE_RECEIPT)
    assert report["status"] == "blocked"
    assert any("row_loss" in blocker for blocker in report["blockers"])

    _write_variant(roots["future_player_form"], variant="future_player_form")
    changed = dict(_bound({"pack_id": "pack-1"}))
    changed["source_game_count"] = 3
    _write_json(roots["future_player_form"], "manifest.json", changed)
    report = compare_downstream_variants(roots, source_receipt=SOURCE_RECEIPT)
    assert report["status"] == "blocked"
    assert any("source_binding_mismatch" in blocker for blocker in report["blockers"])


def test_unknown_field_authority_and_missing_component_block(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    player_path = roots["future_player_form"] / "features" / "player_ratings_snapshot.json"
    payload = json.loads(player_path.read_text(encoding="utf-8"))
    payload["rows"][0]["unregistered_signal"] = 1.0
    player_path.write_text(json.dumps(payload), encoding="utf-8")
    draft_path = roots["scaling_curve"] / "features" / "draft_records.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    del draft["games"][0]["crossfit_same_role"]
    draft["public_probability"] = True
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    report = compare_downstream_variants(roots, source_receipt=SOURCE_RECEIPT)
    assert report["status"] == "blocked"
    assert any("unknown_fields" in blocker for blocker in report["blockers"])
    assert any("forbidden_authority" in blocker for blocker in report["blockers"])
    assert any("component_reconstruction_failed" in blocker for blocker in report["blockers"])


def test_tier_identity_requires_scope_patch_role_champion(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    tier_path = roots["future_player_form"] / "rankings" / "tierlists.json"
    payload = json.loads(tier_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["league"] = row.pop("scope")
    tier_path.write_text(json.dumps(payload), encoding="utf-8")
    report = compare_downstream_variants(roots, source_receipt=SOURCE_RECEIPT)
    assert report["status"] == "blocked"
    assert any("tierlists_row_identity_missing" in blocker for blocker in report["blockers"])


def test_artifact_census_ids_must_match_verified_receipt(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    manifest_path = roots["both"] / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["accepted_game_ids"] = ["game-1", "forged"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    report = compare_downstream_variants(roots, source_receipt=SOURCE_RECEIPT)
    assert report["status"] == "blocked"
    assert any("accepted_game_ids_mismatch" in blocker for blocker in report["blockers"])


def test_nonfinite_and_shared_runtime_root_block(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    path = roots["future_player_form"] / "features" / "player_ratings_snapshot.json"
    path.write_text('[{"player_id":"A","mu_total":NaN}]', encoding="utf-8")
    shared = tmp_path / "runtime"
    shared.mkdir()
    report = compare_downstream_variants(
        roots,
        source_receipt=SOURCE_RECEIPT,
        runtime_roots={name: shared for name in VARIANT_NAMES},
    )
    assert report["status"] == "blocked"
    assert any("nonfinite" in blocker or "missing_or_invalid" in blocker for blocker in report["blockers"])
    assert any("shared_runtime_root" in blocker for blocker in report["blockers"])


def test_source_binding_only_is_rejected(tmp_path: Path) -> None:
    roots = _materialise_roots(tmp_path)
    with pytest.raises(DownstreamDiffError, match="verified source receipt"):
        compare_downstream_variants(roots, source_binding=SOURCE)


def test_report_destination_parent_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(DownstreamDiffError, match="parent"):
        write_downstream_diff_report(link / "report.json", {"schema_version": SCHEMA_VERSION, "public_authority": False})
