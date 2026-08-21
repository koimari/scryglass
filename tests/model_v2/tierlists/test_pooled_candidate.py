"""Tests for the public coach-facing pooled tier fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pytest

from lol_kills.v2.tierlists.pooled_candidate import (
    PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
    PooledCandidateError,
    _blind_point_estimate,
    _build_regional_views,
    _counter_count_point_estimate,
    _matchup_metrics_available,
    _pre_map_offset_values_sha256,
    _regional_contexts,
    _response_basis,
    _resolve_pre_map_offsets,
    _scope_atom_patch,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _offset_provenance(values: Mapping[str, float], game_ids: list[str]) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
        "status": "research_only",
        "authority": False,
        "producer": "test_future_value_rating",
        "timing": "strict_prior_pre_map",
        "source_receipt_sha256": "a" * 64,
        "source_identity_sha256": identity_sha256(game_ids),
        "source_game_count": len(game_ids),
        "offsets_sha256": _pre_map_offset_values_sha256(values),
    }
    unsigned["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return unsigned


def test_regional_contexts_use_exact_league_identity() -> None:
    assert _regional_contexts({"league": "LCK"}) == ("LCK",)
    assert _regional_contexts({"league": "MSI", "event_kind": "msi"}) == ("INTERNATIONAL",)
    assert _regional_contexts({"league": "LCKC"}) == ()


def test_regional_view_keeps_patch_wide_strength_order() -> None:
    rows = [
        {
            "champion": "Lower global pick",
            "champion_id": "riot:champion:2",
            "rank": 2,
            "tier_value_pp": 4.0,
        },
        {
            "champion": "Higher global pick",
            "champion_id": "riot:champion:1",
            "rank": 1,
            "tier_value_pp": 10.0,
        },
    ]
    views = _build_regional_views(
        rows=rows,
        scope_id="patch:16.14",
        role="mid",
        regional_counts={
            ("patch:16.14", "LCK", "mid"): {
                "riot:champion:2": 4,
                "riot:champion:1": 1,
            }
        },
        regional_game_ids={("patch:16.14", "LCK"): {"game-1", "game-2"}},
    )

    assert len(views) == 1
    assert views[0]["id"] == "LCK"
    assert views[0]["maps"] == 2
    assert [row["champion_id"] for row in views[0]["rows"]] == [
        "riot:champion:1",
        "riot:champion:2",
    ]


def test_complete_oe_matchup_support_does_not_require_an_atom_snapshot() -> None:
    assert _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=5,
        contrast_sd=0.4,
    )


def test_oe_matchup_support_remains_fail_closed_for_thin_or_uncertain_rows() -> None:
    assert not _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=4,
        contrast_sd=0.4,
    )
    assert not _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=5,
        contrast_sd=1.2,
    )


def test_blind_point_estimate_uses_expected_weakest_matchup() -> None:
    probabilities = np.asarray(
        [
            [0.56, 0.54, 0.55],
            [0.61, 0.59, 0.60],
            [0.67, 0.65, 0.66],
            [0.58, 0.56, 0.57],
            [0.63, 0.61, 0.62],
        ]
    )
    score = _blind_point_estimate(probabilities, np.full(5, 0.2))
    assert score == 0.55


def test_counter_count_uses_positive_model_contrasts() -> None:
    theta = np.asarray(
        [
            [0.10, 0.08, 0.09],
            [0.06, 0.05, 0.07],
            [0.01, 0.02, 0.03],
            [-0.02, -0.01, 0.00],
            [0.20, 0.18, 0.19],
        ]
    )
    assert _counter_count_point_estimate(theta) == 3


def test_response_basis_names_observed_atom_and_strength_only_estimates() -> None:
    assert _response_basis(effective_maps=4.5, atom_supported=True) == "observed_pair_plus_model"
    assert _response_basis(effective_maps=0.0, atom_supported=True) == "atom_and_strength_inferred"
    assert _response_basis(effective_maps=0.0, atom_supported=False) == "strength_only_inferred"


def test_scope_atom_patch_uses_the_audited_snapshot_instead_of_the_oe_token() -> None:
    games = [
        {"oe_patch_id": "16.15", "atom_snapshot_patch": "26.15"},
        {"oe_patch_id": "16.15", "atom_snapshot_patch": "26.15"},
    ]
    assert _scope_atom_patch(games) == "26.15"
    assert _scope_atom_patch([*games, {"atom_snapshot_patch": "26.16"}]) is None


def test_pre_map_offset_default_path_uses_the_existing_team_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [0.12, -0.34]
    monkeypatch.setattr(
        "lol_kills.v2.tierlists.pooled_candidate._team_offsets",
        lambda _maps: expected,
    )
    values, provenance = _resolve_pre_map_offsets(
        [{"game_id": "g1"}, {"game_id": "g2"}],
        override=None,
        provenance=None,
    )
    assert values is expected
    assert provenance is None


def test_pre_map_offset_override_requires_exact_finite_game_coverage() -> None:
    maps = [{"game_id": "g1"}, {"game_id": "g2"}]
    values = {"g1": 0.12, "g2": -0.34}
    provenance = _offset_provenance(values, ["g1", "g2"])
    resolved, bound = _resolve_pre_map_offsets(
        maps,
        override=values,
        provenance=provenance,
        expected_source_receipt_sha256="a" * 64,
    )
    assert resolved == [0.12, -0.34]
    assert bound == provenance
    with pytest.raises(PooledCandidateError, match="missing"):
        _resolve_pre_map_offsets(
            maps,
            override={"g1": 0.12},
            provenance=provenance,
            expected_source_receipt_sha256="a" * 64,
        )
    with pytest.raises(PooledCandidateError, match="extra"):
        _resolve_pre_map_offsets(
            maps,
            override={**values, "g3": 0.0},
            provenance=provenance,
            expected_source_receipt_sha256="a" * 64,
        )
    with pytest.raises(PooledCandidateError, match="finite"):
        _resolve_pre_map_offsets(
            maps,
            override={"g1": float("nan"), "g2": -0.34},
            provenance=provenance,
            expected_source_receipt_sha256="a" * 64,
        )


def test_pre_map_offset_override_rejects_duplicate_id_entries() -> None:
    class DuplicateMapping(Mapping[str, float]):
        def __getitem__(self, key: str) -> float:
            return {"g1": 0.12, "g2": -0.34}[key]

        def __iter__(self):
            return iter(("g1", "g1", "g2"))

        def __len__(self) -> int:
            return 3

        def items(self):
            return (("g1", 0.12), ("g1", 0.13), ("g2", -0.34))

    values = {"g1": 0.12, "g2": -0.34}
    provenance = _offset_provenance(values, ["g1", "g2"])
    with pytest.raises(PooledCandidateError, match="duplicate"):
        _resolve_pre_map_offsets(
            [{"game_id": "g1"}, {"game_id": "g2"}],
            override=DuplicateMapping(),
            provenance=provenance,
            expected_source_receipt_sha256="a" * 64,
        )


def test_pre_map_offset_override_requires_a_closed_self_bound_provenance() -> None:
    maps = [{"game_id": "g1"}, {"game_id": "g2"}]
    values = {"g1": 0.12, "g2": -0.34}
    provenance = _offset_provenance(values, ["g1", "g2"])
    changed = dict(provenance)
    changed["producer"] = "forged"
    with pytest.raises(PooledCandidateError, match="receipt hash"):
        _resolve_pre_map_offsets(
            maps,
            override=values,
            provenance=changed,
            expected_source_receipt_sha256="a" * 64,
        )
    changed = dict(provenance)
    changed["extra"] = "closed"
    with pytest.raises(PooledCandidateError, match="closed"):
        _resolve_pre_map_offsets(
            maps,
            override=values,
            provenance=changed,
            expected_source_receipt_sha256="a" * 64,
        )
    with pytest.raises(PooledCandidateError, match="together"):
        _resolve_pre_map_offsets(
            maps,
            override=values,
            provenance=None,
            expected_source_receipt_sha256="a" * 64,
        )


def test_pre_map_offset_override_requires_an_independent_source_receipt() -> None:
    maps = [{"game_id": "g1"}, {"game_id": "g2"}]
    values = {"g1": 0.12, "g2": -0.34}
    provenance = _offset_provenance(values, ["g1", "g2"])
    with pytest.raises(PooledCandidateError, match="required"):
        _resolve_pre_map_offsets(
            maps,
            override=values,
            provenance=provenance,
            expected_source_receipt_sha256=None,
        )
    forged = dict(provenance)
    forged["source_receipt_sha256"] = "b" * 64
    unsigned = dict(forged)
    unsigned.pop("receipt_sha256")
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    with pytest.raises(PooledCandidateError, match="not trusted"):
        _resolve_pre_map_offsets(
            maps,
            override=values,
            provenance=forged,
            expected_source_receipt_sha256="a" * 64,
        )
