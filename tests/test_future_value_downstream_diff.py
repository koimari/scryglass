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
)
from lol_kills.research.future_value_rating import RatingVariant


SOURCE = {
    "source_as_of": "2026-08-20T11:31:37Z",
    "source_game_count": 2,
    "source_identity_sha256": hashlib.sha256(b"game-1\ngame-2\n").hexdigest(),
}


def _write_json(root: Path, relative: str, value: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8")


def _write_variant(root: Path, *, variant: str, row_loss: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root, "future-value-source-receipt.json", SOURCE)
    future_value = 1.0 if variant in {"future_player_form", "both"} else 0.0
    scaling_value = 0.25 if variant in {"scaling_curve", "both"} else 0.0
    future_delta = {"future_value": future_value}
    scaling_delta = {"forecast_gold_diff_10": scaling_value}
    players = [
        {"player": "A", "mu_total": 100.0, **future_delta, **scaling_delta},
        {"player": "B", "mu_total": 90.0, **future_delta, **scaling_delta},
    ]
    if row_loss:
        players.pop()
    _write_json(root, "features/player_ratings_snapshot.json", players)
    _write_json(
        root,
        "features/ratings_snapshot.json",
        [
            {"team_key": "alpha", "team": "Alpha", "mu_total": 110.0},
            {"team_key": "beta", "team": "Beta", "mu_total": 95.0},
        ],
    )
    _write_json(
        root,
        "rankings/tierlists.json",
        {
            "rows": [
                {"champion": "Aatrox", "role": "top", "league": "LCK", "patch": "16.16", "rank": 1, "tier": "S", "score": 0.5},
                {"champion": "Ahri", "role": "mid", "league": "LCK", "patch": "16.16", "rank": 2, "tier": "A", "score": 0.4},
            ]
        },
    )
    _write_json(
        root,
        "features/draft_records.json",
        {
            "games": [
                {
                    "game_uid": "game-1",
                    "base": 1.0,
                    "synergy": 2.0,
                    "counter": 3.0,
                    "same_role": 4.0,
                    "score": 10.0,
                    "draft_edge": 10.0,
                    "target": 1,
                }
            ]
        },
    )
    _write_json(root, "features/profile_records.json", {"games": [{"game_uid": "game-1", "future_value": 0.5, **scaling_delta}]})
    _write_json(root, "features/match_records_2026_q3.json", {"games": [{"game_uid": "game-1", "future_team_value": 0.2, "blue_result": 1}]})
    _write_json(root, "manifest.json", {"pack_id": "pack-1", **SOURCE, "total_files": 7, "total_bytes": 100})


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name in VARIANT_NAMES:
        root = tmp_path / name
        _write_variant(root, variant=name)
        roots[name] = root
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
    old = [{"id": "a", "tier": "S", "base": 1.0, "synergy": 2.0, "counter": 3.0, "same_role": 4.0, "score": 10.0}]
    candidate = [{"id": "a", "tier": "A", "base": 1.0, "synergy": 2.0, "counter": 3.0, "same_role": 4.0, "score": 10.0}]
    tier = tier_transition_metrics(old, candidate, identity_fields=("id",))
    assert tier["changed_count"] == 1
    assert tier["transitions"] == {"S->A": 1}
    draft = draft_score_component_diff_metrics(old, candidate, identity_fields=("id",))
    assert draft["static_components_identical"] is True
    assert draft["reconstruction"]["candidate"]["status"] == "passed"


def test_four_variant_report_keeps_authority_closed(tmp_path: Path) -> None:
    report = compare_downstream_variants(_roots(tmp_path), source_binding=SOURCE)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ready_research_only"
    assert set(report["variants"]) == set(VARIANT_NAMES)
    assert not any(report["authority"].values())
    assert report["draft_score"]["future_player_form"]["component_deltas"]["base"]["changed_count"] == 0
    assert report["artifacts"]["player_ratings"]["comparisons"]["future_player_form"]["deltas"]["future_value"]["changed_count"] == 2


def test_row_loss_and_source_mismatch_block(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write_variant(roots["scaling_curve"], variant="scaling_curve", row_loss=True)
    report = compare_downstream_variants(roots, source_binding=SOURCE)
    assert report["status"] == "blocked"
    assert any("row_loss" in blocker for blocker in report["blockers"])

    changed = dict(SOURCE)
    changed["source_game_count"] = 3
    _write_variant(roots["future_player_form"], variant="future_player_form")
    _write_json(roots["future_player_form"], "manifest.json", {"pack_id": "pack-1", **changed})
    report = compare_downstream_variants(roots, source_binding=SOURCE)
    assert report["status"] == "blocked"
    assert any("source_binding_mismatch" in blocker for blocker in report["blockers"])


def test_nonfinite_and_shared_runtime_root_block(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    path = roots["future_player_form"] / "features" / "player_ratings_snapshot.json"
    path.write_text('[{"player":"A","mu_total":NaN}]', encoding="utf-8")
    shared = tmp_path / "runtime"
    shared.mkdir()
    report = compare_downstream_variants(
        roots,
        source_binding=SOURCE,
        runtime_roots={name: shared for name in VARIANT_NAMES},
    )
    assert report["status"] == "blocked"
    assert any("nonfinite" in blocker or "missing_or_invalid" in blocker for blocker in report["blockers"])
    assert any("shared_runtime_root" in blocker for blocker in report["blockers"])
