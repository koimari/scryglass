from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import lol_kills.v2.draft.interactions.representation_rank_assay as assay

from lol_kills.v2.draft.interactions.generate_representation_rank_assay_artifacts import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    verify_pending_shell,
)
from lol_kills.v2.draft.interactions.representation_rank_assay import (
    PENALTY_GRID,
    RepresentationRankAssayError,
    bootstrap_multiplicity,
    canonical_sha256,
    center_by_role,
    combine_prepared_folds,
    deterministic_starts,
    evaluate_candidate_gates,
    fit_latent_candidate,
    latent_objective_and_gradient,
    load_nonholdout_rows,
    interaction_logits,
    nearest_rank_endpoint,
    outcome_free_coverage,
    prepare_outer_fold,
    require_m0_offsets,
    select_development_width,
    validate_locked_width,
    select_separate_penalty,
    symplectic_J,
    validate_config,
    validate_report,
)


NODE_DOMAIN = assay._build_node_domain(
    [{"stable_champion_id": f"champion-{index}"} for index in range(15)],
    source_raw_sha256="a" * 64,
)
NODE_ROLES = NODE_DOMAIN.node_roles
NODE_CHAMPIONS = NODE_DOMAIN.node_champion_ids
BLUE_DRAFT = np.array([0, 6, 12, 18, 24])
RED_DRAFT = np.array([25, 31, 37, 43, 49])
LEGAL_DRAFT = np.concatenate((BLUE_DRAFT, RED_DRAFT))
INELIGIBLE_TOP_NODE = 50


def _targets(mapping):
    return assay._build_target_domain(mapping, source_raw_sha256="d" * 64)


def _rehash(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def test_pending_shell_is_source_bound_and_has_not_loaded_candidates() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_bytes())
    report = json.loads(DEFAULT_REPORT_PATH.read_bytes())
    validate_config(config)
    validate_report(report)
    checkpoint = verify_pending_shell()
    assert checkpoint["real_candidate_outcomes_loaded"] is False
    assert checkpoint["final_temporal_holdout_loaded"] is False


def test_authoritative_node_and_cluster_domains_parse_pinned_bytes() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_bytes())
    sources = config["source_identity"]
    node = assay.load_node_domain(
        Path(sources["champion_crosswalk"]["locator"]),
        expected_raw_sha256=sources["champion_crosswalk"]["raw_sha256"],
    )
    assert node.source_raw_sha256 == sources["champion_crosswalk"]["raw_sha256"]
    cluster = assay.load_cluster_domain(
        cluster_proxy_path=Path(sources["dependence_cluster_proxy"]["locator"]),
        split_path=Path(sources["outcome_free_split"]["locator"]),
        expected_cluster_proxy_raw_sha256=sources[
            "dependence_cluster_proxy"
        ]["raw_sha256"],
        expected_split_raw_sha256=sources["outcome_free_split"]["raw_sha256"],
    )
    assert len(cluster.ordered_game_clusters) == 6310
    with pytest.raises(RepresentationRankAssayError, match="bytes changed"):
        assay.load_node_domain(
            Path(sources["champion_crosswalk"]["locator"]),
            expected_raw_sha256="0" * 64,
        )


def test_fit_availability_domain_is_identity_only_and_tamper_evident() -> None:
    domain = assay._build_fit_availability_domain(
        ["g2", "g1"], source_raw_sha256="a" * 64
    )
    assert domain.ordered_game_ids == ("g1", "g2")
    assay.validate_fit_availability_domain(domain)
    with pytest.raises(RepresentationRankAssayError, match="fit-availability"):
        assay.validate_fit_availability_domain(
            replace(domain, ordered_game_ids=("g2", "g1"))
        )


def test_estimand_rejects_sparse_full_rank_or_whole_composition_claims() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_bytes())
    changed = copy.deepcopy(config)
    changed["estimand"]["intrinsic_pair_table_rank"] = True
    with pytest.raises(RepresentationRankAssayError, match="estimand"):
        validate_config(_rehash(changed))
    changed = copy.deepcopy(config)
    changed["bootstrap"]["development_family"] = "prose only"
    with pytest.raises(RepresentationRankAssayError, match="bootstrap"):
        validate_config(_rehash(changed))
    changed = copy.deepcopy(config)
    changed["decision"]["M8"] = "full"
    with pytest.raises(RepresentationRankAssayError, match="decision"):
        validate_config(_rehash(changed))


def test_m0_offsets_are_exact_by_game_id_and_not_refit() -> None:
    verified = {"g1": 0.4, "g2": 0.7}
    offsets = require_m0_offsets(["g2", "g1"], [0.7, 0.4], verified)
    assert np.allclose(offsets, np.log([0.7 / 0.3, 0.4 / 0.6]))
    with pytest.raises(RepresentationRankAssayError, match="missing"):
        require_m0_offsets(["g3"], [0.5], verified)
    with pytest.raises(RepresentationRankAssayError, match="reproduction"):
        require_m0_offsets(["g1"], [0.41], verified)


def test_role_centering_is_exact_and_unseen_nodes_are_zero_internal_only() -> None:
    values = np.arange(24, dtype=float).reshape(8, 3)
    roles = ["top"] * 4 + ["jungle"] * 4
    eligible = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    centered = center_by_role(values, roles, eligible)
    assert np.allclose(centered[:3].mean(axis=0), 0)
    assert np.allclose(centered[4:7].mean(axis=0), 0)
    assert np.array_equal(centered[[3, 7]], np.zeros((2, 3)))


def test_ally_and_enemy_scores_have_exact_normalization_and_side_swap() -> None:
    rng = np.random.default_rng(4)
    blue = np.vstack((BLUE_DRAFT, RED_DRAFT))
    red = np.vstack((RED_DRAFT, BLUE_DRAFT))
    ally = rng.normal(size=(len(NODE_ROLES), 2))
    enemy = rng.normal(size=(len(NODE_ROLES), 4))
    a, e, total = interaction_logits(
        blue, red, ally, enemy, width=2,
        node_domain=NODE_DOMAIN,
    )
    swapped = interaction_logits(
        red, blue, ally, enemy, width=2,
        node_domain=NODE_DOMAIN,
    )
    assert np.allclose(swapped[0], -a)
    assert np.allclose(swapped[1], -e)
    assert np.allclose(swapped[2], -total)
    J = symplectic_J(2)
    assert np.allclose(J, -J.T)
    active_enemy = enemy[LEGAL_DRAFT]
    induced = active_enemy @ J @ active_enemy.T
    assert np.linalg.matrix_rank(induced, tol=1e-10) <= 4


def test_vectorized_scores_equal_literal_pair_enumeration() -> None:
    rng = np.random.default_rng(404)
    blue = np.vstack((BLUE_DRAFT, RED_DRAFT))
    red = np.vstack((RED_DRAFT, BLUE_DRAFT))
    ally = rng.normal(size=(len(NODE_ROLES), 4))
    enemy = rng.normal(size=(len(NODE_ROLES), 8))
    observed = interaction_logits(
        blue, red, ally, enemy, width=4,
        node_domain=NODE_DOMAIN,
    )
    J = symplectic_J(4)
    expected_ally = np.zeros(2)
    expected_enemy = np.zeros(2)
    for row in range(2):
        for first in range(5):
            for second in range(first + 1, 5):
                expected_ally[row] += ally[blue[row, first]] @ ally[blue[row, second]]
                expected_ally[row] -= ally[red[row, first]] @ ally[red[row, second]]
        for first in range(5):
            for second in range(5):
                expected_enemy[row] += (
                    enemy[blue[row, first]] @ J @ enemy[red[row, second]]
                )
    expected_ally /= 10
    expected_enemy /= 25
    assert np.allclose(observed[0], expected_ally, atol=1e-14)
    assert np.allclose(observed[1], expected_enemy, atol=1e-14)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("negative", "canonical role domain"),
        ("float", "integers"),
        ("wrong_role", "canonical role domain"),
        ("duplicate_identity", "node-domain"),
    ],
)
def test_scoring_rejects_impossible_drafts_before_math(mutation, message) -> None:
    blue = BLUE_DRAFT[None, :].copy()
    red = RED_DRAFT[None, :].copy()
    domain = NODE_DOMAIN
    if mutation == "negative":
        blue[0, 0] = -1
    elif mutation == "float":
        blue = blue.astype(float)
    elif mutation == "wrong_role":
        blue[0, 0] = BLUE_DRAFT[1]
    else:
        domain = replace(
            NODE_DOMAIN,
            node_champion_ids=NODE_CHAMPIONS[:-1] + (NODE_CHAMPIONS[0],),
        )
    with pytest.raises(RepresentationRankAssayError, match=message):
        interaction_logits(
            blue, red, np.zeros((len(NODE_ROLES), 1)),
            np.zeros((len(NODE_ROLES), 2)), width=1,
            node_domain=domain,
        )


def _coverage_fixture(
    low_coverage: bool = False, fit_node_rows: np.ndarray | None = None
):
    maps = 45
    row_nodes = np.tile(LEGAL_DRAFT, (maps, 1))
    if low_coverage:
        row_nodes[:20, 0] = INELIGIBLE_TOP_NODE
    score_ids = [f"score-{index}" for index in range(maps)]
    row_clusters = [f"c-{index // 2}" for index in range(maps)]
    fit_nodes = (
        np.tile(LEGAL_DRAFT, (30, 1))
        if fit_node_rows is None
        else np.asarray(fit_node_rows, dtype=int)
    )
    fit_ids = [f"fit-game-{index}" for index in range(len(fit_nodes))]
    fit_clusters = [f"fit-{index}" for index in range(len(fit_nodes))]
    assignments = [
        {
            "game_id": game_id,
            "dependence_cluster_id": cluster,
            "oe_date_naive": "2026-04-01T00:00:00",
        }
        for game_id, cluster in zip(score_ids, row_clusters)
    ] + [
        {
            "game_id": game_id,
            "dependence_cluster_id": cluster,
            "oe_date_naive": "2026-03-01T00:00:00",
        }
        for game_id, cluster in zip(fit_ids, fit_clusters)
    ]
    cluster_domain = assay._build_cluster_domain(
        assignments, source_raw_sha256="b" * 64
    )
    feature_rows = [
        {"game_id": game_id, "split": "validation", "league": "LEC", "nodes": nodes}
        for game_id, nodes in zip(score_ids, row_nodes)
    ] + [
        {"game_id": game_id, "split": "train", "league": "LEC", "nodes": nodes}
        for game_id, nodes in zip(fit_ids, fit_nodes)
    ]
    feature_domain = assay._build_feature_domain(
        feature_rows,
        node_domain=NODE_DOMAIN,
        cluster_domain=cluster_domain,
        source_raw_sha256="c" * 64,
    )
    return feature_domain, score_ids, fit_ids


def _fixture_fit_availability(feature_domain, *, source_raw_sha256="d" * 64):
    return assay._build_fit_availability_domain(
        [row[0] for row in feature_domain.records],
        source_raw_sha256=source_raw_sha256,
    )


def test_outcome_free_coverage_uses_one_mask_and_frozen_gates() -> None:
    values = _coverage_fixture()
    result = outcome_free_coverage(
        feature_domain=values[0],
        score_game_ids=values[1],
        fit_game_ids=values[2],
        split="validation",
        fit_availability_domain=_fixture_fit_availability(values[0]),
    )
    assert result.report["passed"] is True
    assert result.eligible_rows.all()
    failed_values = _coverage_fixture(low_coverage=True)
    failed = outcome_free_coverage(
        feature_domain=failed_values[0],
        score_game_ids=failed_values[1],
        fit_game_ids=failed_values[2],
        split="validation",
        fit_availability_domain=_fixture_fit_availability(failed_values[0]),
    )
    assert failed.report["passed"] is False
    assert len(failed.report["excluded_maps"]) == 20
    tampered = replace(
        values[0],
        records=values[0].records[:-1]
        + (
            values[0].records[-1][:-1]
            + (
                (int(BLUE_DRAFT[1]),)
                + tuple(int(value) for value in LEGAL_DRAFT[1:]),
            ),
        ),
    )
    with pytest.raises(RepresentationRankAssayError, match="feature-domain"):
        outcome_free_coverage(
            feature_domain=tampered,
            score_game_ids=values[1],
            fit_game_ids=values[2],
            split="validation",
            fit_availability_domain=_fixture_fit_availability(values[0]),
        )


def _coverage_counts(
    *,
    maps: int,
    eligible_maps: int,
    clusters: int,
    eligible_clusters: int,
) -> dict[str, int]:
    return {
        "maps": maps,
        "eligible_maps": eligible_maps,
        "clusters": clusters,
        "eligible_clusters": eligible_clusters,
    }


def test_coverage_gate_decision_enforces_exact_aggregate_boundaries() -> None:
    four_fifths = _coverage_counts(
        maps=100,
        eligible_maps=80,
        clusters=20,
        eligible_clusters=16,
    )
    assert assay.coverage_gate_decision(
        overall=four_fifths,
        month_rows=[four_fifths],
        league_rows=[four_fifths],
    ) is True

    below_overall_maps = {
        **four_fifths,
        "eligible_maps": 79,
    }
    assert assay.coverage_gate_decision(
        overall=below_overall_maps,
        month_rows=[below_overall_maps],
        league_rows=[below_overall_maps],
    ) is False

    below_overall_clusters = {
        **four_fifths,
        "eligible_clusters": 15,
    }
    assert assay.coverage_gate_decision(
        overall=below_overall_clusters,
        month_rows=[below_overall_clusters],
        league_rows=[below_overall_clusters],
    ) is False

    month_cluster_minimum = _coverage_counts(
        maps=100,
        eligible_maps=80,
        clusters=18,
        eligible_clusters=15,
    )
    assert assay.coverage_gate_decision(
        overall=month_cluster_minimum,
        month_rows=[month_cluster_minimum],
        league_rows=[month_cluster_minimum],
    ) is True
    below_month_cluster_minimum = {
        **month_cluster_minimum,
        "clusters": 17,
        "eligible_clusters": 14,
    }
    assert assay.coverage_gate_decision(
        overall=below_month_cluster_minimum,
        month_rows=[below_month_cluster_minimum],
        league_rows=[below_month_cluster_minimum],
    ) is False

    # A one-month aggregate at the inclusive 2/3 month boundary still fails
    # the stronger frozen 4/5 overall rule.
    exact_month_fraction = _coverage_counts(
        maps=30,
        eligible_maps=20,
        clusters=20,
        eligible_clusters=16,
    )
    assert assay.coverage_gate_decision(
        overall=exact_month_fraction,
        month_rows=[exact_month_fraction],
        league_rows=[exact_month_fraction],
    ) is False


def test_coverage_gate_decision_applies_conditional_league_boundary() -> None:
    overall = _coverage_counts(
        maps=100,
        eligible_maps=80,
        clusters=20,
        eligible_clusters=16,
    )
    exact = [
        _coverage_counts(
            maps=40,
            eligible_maps=30,
            clusters=10,
            eligible_clusters=8,
        ),
        _coverage_counts(
            maps=60,
            eligible_maps=50,
            clusters=10,
            eligible_clusters=8,
        ),
    ]
    assert assay.coverage_gate_decision(
        overall=overall,
        month_rows=[overall],
        league_rows=exact,
    ) is True

    below = copy.deepcopy(exact)
    below[0]["eligible_maps"] = 29
    below[1]["eligible_maps"] = 51
    assert assay.coverage_gate_decision(
        overall=overall,
        month_rows=[overall],
        league_rows=below,
    ) is False

    below_minimum_population = [
        _coverage_counts(
            maps=29,
            eligible_maps=20,
            clusters=10,
            eligible_clusters=8,
        ),
        _coverage_counts(
            maps=71,
            eligible_maps=60,
            clusters=10,
            eligible_clusters=8,
        ),
    ]
    assert assay.coverage_gate_decision(
        overall=overall,
        month_rows=[overall],
        league_rows=below_minimum_population,
    ) is True

    exact_population_activation = [
        _coverage_counts(
            maps=30,
            eligible_maps=22,
            clusters=10,
            eligible_clusters=8,
        ),
        _coverage_counts(
            maps=70,
            eligible_maps=58,
            clusters=10,
            eligible_clusters=8,
        ),
    ]
    assert assay.coverage_gate_decision(
        overall=overall,
        month_rows=[overall],
        league_rows=exact_population_activation,
    ) is False

    below_cluster_activation = [
        _coverage_counts(
            maps=40,
            eligible_maps=29,
            clusters=9,
            eligible_clusters=7,
        ),
        _coverage_counts(
            maps=60,
            eligible_maps=51,
            clusters=11,
            eligible_clusters=9,
        ),
    ]
    assert assay.coverage_gate_decision(
        overall=overall,
        month_rows=[overall],
        league_rows=below_cluster_activation,
    ) is True


@pytest.mark.parametrize("league_count", (0, 33))
def test_coverage_gate_decision_bounds_league_cardinality(
    league_count: int,
) -> None:
    overall = _coverage_counts(
        maps=330,
        eligible_maps=264,
        clusters=33,
        eligible_clusters=33,
    )
    leagues = [
        _coverage_counts(
            maps=10,
            eligible_maps=8,
            clusters=1,
            eligible_clusters=1,
        )
        for _ in range(league_count)
    ]
    with pytest.raises(
        RepresentationRankAssayError,
        match="coverage aggregate",
    ):
        assay.coverage_gate_decision(
            overall=overall,
            month_rows=[overall],
            league_rows=leagues,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda overall, month, leagues: month.__setitem__(
            "eligible_maps", month["eligible_maps"] - 1
        ),
        lambda overall, month, leagues: leagues[0].__setitem__(
            "maps", leagues[0]["maps"] - 1
        ),
        lambda overall, month, leagues: leagues[0].__setitem__(
            "clusters", leagues[0]["clusters"] - 1
        ),
        lambda overall, month, leagues: overall.__setitem__(
            "eligible_maps", overall["maps"] + 1
        ),
        lambda overall, month, leagues: overall.__setitem__("maps", 100.0),
        lambda overall, month, leagues: overall.__setitem__(
            "eligible_clusters", True
        ),
    ),
)
def test_coverage_gate_decision_rejects_impossible_count_identities(
    mutate,
) -> None:
    overall = _coverage_counts(
        maps=100,
        eligible_maps=80,
        clusters=20,
        eligible_clusters=16,
    )
    month = dict(overall)
    leagues = [
        _coverage_counts(
            maps=40,
            eligible_maps=30,
            clusters=10,
            eligible_clusters=8,
        ),
        _coverage_counts(
            maps=60,
            eligible_maps=50,
            clusters=10,
            eligible_clusters=8,
        ),
    ]
    mutate(overall, month, leagues)
    with pytest.raises(
        RepresentationRankAssayError,
        match="coverage",
    ):
        assay.coverage_gate_decision(
            overall=overall,
            month_rows=[month],
            league_rows=leagues,
        )


def test_fit_availability_omission_and_provenance_substitution_fail() -> None:
    feature, score_ids, fit_ids = _coverage_fixture()
    availability_a = _fixture_fit_availability(
        feature, source_raw_sha256="a" * 64
    )
    availability_b = _fixture_fit_availability(
        feature, source_raw_sha256="b" * 64
    )
    with pytest.raises(
        RepresentationRankAssayError, match="fit-availability domain is required"
    ):
        outcome_free_coverage(
            feature_domain=feature,
            score_game_ids=score_ids,
            fit_game_ids=fit_ids,
            split="validation",
            fit_availability_domain=None,
        )
    bound_a = outcome_free_coverage(
        feature_domain=feature,
        score_game_ids=score_ids,
        fit_game_ids=fit_ids,
        split="validation",
        fit_availability_domain=availability_a,
    ).eligibility_binding
    bound_b = outcome_free_coverage(
        feature_domain=feature,
        score_game_ids=score_ids,
        fit_game_ids=fit_ids,
        split="validation",
        fit_availability_domain=availability_b,
    ).eligibility_binding
    assert bound_a.artifact_sha256 != bound_b.artifact_sha256
    assert bound_a.fit_availability_domain_sha256 != (
        bound_b.fit_availability_domain_sha256
    )
    substituted = replace(
        bound_a,
        fit_availability_domain=availability_b,
        artifact_sha256=bound_a.artifact_sha256,
    )
    with pytest.raises(
        RepresentationRankAssayError, match="eligibility source domain changed"
    ):
        assay._validate_eligibility_binding(substituted)


def test_coverage_rejects_score_cluster_relabel_and_fit_overlap() -> None:
    feature, score_ids, fit_ids = _coverage_fixture()
    first = feature.records[0]
    relabeled_record = first[:2] + ("fabricated-cluster",) + first[3:]
    relabeled_records = (relabeled_record,) + feature.records[1:]
    unsigned = {
        "source_raw_sha256": feature.source_raw_sha256,
        "node_domain_sha256": feature.node_domain.artifact_sha256,
        "cluster_domain_sha256": feature.cluster_domain.artifact_sha256,
        "records": relabeled_records,
        "authoritative_source_verified": False,
        "authoritative_loader_status": "future_private_runner_required",
    }
    relabeled = replace(
        feature,
        records=relabeled_records,
        artifact_sha256=canonical_sha256(unsigned),
    )
    with pytest.raises(RepresentationRankAssayError, match="provenance"):
        outcome_free_coverage(
            feature_domain=relabeled,
            score_game_ids=score_ids,
            fit_game_ids=fit_ids,
            split="validation",
            fit_availability_domain=_fixture_fit_availability(relabeled),
        )

    assignments = [
        {
            "game_id": row[0],
            "dependence_cluster_id": (
                feature.records[0][2] if row[0] == fit_ids[0] else row[2]
            ),
            "oe_date_naive": f"{row[3]}-01T00:00:00",
        }
        for row in feature.records
    ]
    overlapping_clusters = assay._build_cluster_domain(
        assignments, source_raw_sha256="b" * 64
    )
    overlapping = assay._build_feature_domain(
        [
            {"game_id": row[0], "split": row[1], "league": row[4], "nodes": row[5]}
            for row in feature.records
        ],
        node_domain=NODE_DOMAIN,
        cluster_domain=overlapping_clusters,
        source_raw_sha256="c" * 64,
    )
    with pytest.raises(RepresentationRankAssayError, match="overlap|strictly earlier"):
        outcome_free_coverage(
            feature_domain=overlapping,
            score_game_ids=score_ids,
            fit_game_ids=fit_ids,
            split="validation",
            fit_availability_domain=_fixture_fit_availability(overlapping),
        )


def _penalty_rows(family: str, best: float = 0.1):
    rows = []
    for penalty in PENALTY_GRID:
        for month in range(4, 10):
            maps = 260
            score = 0.69 + abs(np.log10(penalty / best)) * 0.001
            rows.append(
                {
                    "family": family,
                    "lambda": penalty,
                    "width": 8,
                    "calendar_month": f"2025-{month:02d}",
                    "split": "train",
                    "maps": maps,
                    "clusters": 50,
                    "membership_sha256": f"{month:064x}",
                    "log_loss_total": score * maps,
                    "brier_total": (0.24 + abs(penalty - best) * 1e-4) * maps,
                    "strictly_earlier_fit": True,
                    "cluster_atomic": True,
                }
            )
    return rows


def test_penalties_are_tuned_separately_without_cartesian_search() -> None:
    assert select_separate_penalty(_penalty_rows("ally"), family="ally") == 0.1
    assert select_separate_penalty(
        _penalty_rows("enemy", best=1.0), family="enemy"
    ) == 1.0
    missing = _penalty_rows("ally")[:-1]
    with pytest.raises(RepresentationRankAssayError, match="fold"):
        select_separate_penalty(missing, family="ally")
    mismatched = _penalty_rows("ally")
    mismatched[-1]["membership_sha256"] = "f" * 64
    with pytest.raises(RepresentationRankAssayError, match="memberships"):
        select_separate_penalty(mismatched, family="ally")


def test_all_three_starts_are_deterministic_nonzero_and_not_zero_padded() -> None:
    nodes = np.tile(LEGAL_DRAFT, (20, 1))
    residual = np.linspace(-0.2, 0.2, 20)
    first = deterministic_starts(sorted(LEGAL_DRAFT), 4, nodes, residual)
    second = deterministic_starts(sorted(LEGAL_DRAFT), 4, nodes, residual)
    assert len(first) == 3
    for (a, e), (a2, e2) in zip(first, second):
        assert np.any(a) and np.any(e)
        assert np.array_equal(a, a2)
        assert np.array_equal(e, e2)


def test_random_starts_ignore_appended_or_interspersed_ineligible_ids() -> None:
    residual = np.linspace(-0.2, 0.2, 20)
    base_ids = np.arange(10)
    interspersed_ids = np.arange(10) * 3
    base_rows = np.tile(base_ids, (20, 1))
    interspersed_rows = np.tile(interspersed_ids, (20, 1))
    for width in (1, 2, 4, 8):
        base = deterministic_starts(base_ids, width, base_rows, residual)
        interspersed = deterministic_starts(
            interspersed_ids, width, interspersed_rows, residual
        )
        for (base_a, base_e), (other_a, other_e) in zip(base, interspersed):
            assert np.array_equal(base_a, other_a)
            assert np.array_equal(base_e, other_e)


def _fixture_eligibility_binding(fit_node_rows=None):
    values = _coverage_fixture(fit_node_rows=fit_node_rows)
    return outcome_free_coverage(
        feature_domain=values[0],
        score_game_ids=values[1],
        fit_game_ids=values[2],
        split="validation",
        fit_availability_domain=_fixture_fit_availability(values[0]),
    ).eligibility_binding


def test_nonconvex_fixture_requires_stable_converged_solutions_and_kkt() -> None:
    rng = np.random.default_rng(5)
    blue, red = [], []
    for _ in range(100):
        choose_red = rng.integers(0, 2, size=5).astype(bool)
        blue.append(np.where(choose_red, RED_DRAFT, BLUE_DRAFT))
        red.append(np.where(choose_red, BLUE_DRAFT, RED_DRAFT))
    targets = (rng.random(100) < 0.5).astype(int)
    fit_rows = np.column_stack((np.asarray(blue), np.asarray(red)))
    binding = _fixture_eligibility_binding(fit_rows)
    ids = list(binding.ordered_fit_game_ids)
    fit = fit_latent_candidate(
        blue_nodes=np.asarray(blue),
        red_nodes=np.asarray(red),
        target_domain=_targets(dict(zip(ids, targets))),
        split_identity="train",
        game_ids=ids,
        verified_nuisance_oof={game_id: 0.5 for game_id in ids},
        eligibility_binding=binding,
        width=1,
        lambda_ally=1.0,
        lambda_enemy=1.0,
    )
    assert fit.converged_starts >= 2
    assert fit.maximum_absolute_gradient <= 1e-5
    assert fit.best_two_interaction_logit_rms <= 0.01
    assert fit.eligibility_binding_sha256 == binding.artifact_sha256


def test_fitter_rejects_final_split_and_active_unseen_node() -> None:
    fit_rows = np.tile(LEGAL_DRAFT, (5, 1))
    blue = fit_rows[:, :5]
    red = fit_rows[:, 5:]
    binding = _fixture_eligibility_binding(fit_rows)
    ids = list(binding.ordered_fit_game_ids)
    kwargs = dict(
        blue_nodes=blue,
        red_nodes=red,
        target_domain=_targets(
            {game_id: index % 2 for index, game_id in enumerate(ids)}
        ),
        game_ids=ids,
        verified_nuisance_oof={game_id: 0.5 for game_id in ids},
        eligibility_binding=binding,
        width=1,
        lambda_ally=1.0,
        lambda_enemy=1.0,
    )
    with pytest.raises(RepresentationRankAssayError, match="sealed"):
        fit_latent_candidate(split_identity="final_temporal_holdout", **kwargs)
    omitted = dict(kwargs)
    omitted["game_ids"] = ids[:-1]
    omitted["blue_nodes"] = blue[:-1]
    omitted["red_nodes"] = red[:-1]
    with pytest.raises(RepresentationRankAssayError, match="required fit"):
        fit_latent_candidate(split_identity="train", **omitted)
    substituted = dict(kwargs)
    substituted_blue = blue.copy()
    substituted_red = red.copy()
    substituted_blue[0, 0], substituted_red[0, 0] = (
        substituted_red[0, 0],
        substituted_blue[0, 0],
    )
    substituted["blue_nodes"] = substituted_blue
    substituted["red_nodes"] = substituted_red
    with pytest.raises(RepresentationRankAssayError, match="FeatureDomain"):
        fit_latent_candidate(split_identity="train", **substituted)
    later_id = binding.ordered_source_game_ids[0]
    later_record = {
        row[0]: row for row in binding.feature_domain.records
    }[later_id]
    later = dict(kwargs)
    later["game_ids"] = [*ids, later_id]
    later["blue_nodes"] = np.vstack((blue, later_record[5][:5]))
    later["red_nodes"] = np.vstack((red, later_record[5][5:]))
    later["target_domain"] = _targets(
        {
            **{game_id: index % 2 for index, game_id in enumerate(ids)},
            later_id: 0,
        }
    )
    later["verified_nuisance_oof"] = {
        **kwargs["verified_nuisance_oof"],
        later_id: 0.5,
    }
    with pytest.raises(RepresentationRankAssayError, match="required fit"):
        fit_latent_candidate(split_identity="train", **later)
    kwargs["target_domain"] = _targets(
        {game_id: index % 2 for index, game_id in enumerate(ids[:-1])}
    )
    with pytest.raises(RepresentationRankAssayError, match="required fit"):
        fit_latent_candidate(split_identity="train", **kwargs)
    kwargs["target_domain"] = _targets(
        {game_id: index % 2 for index, game_id in enumerate(ids)}
    )
    changed_mask = list(binding.eligible_nodes)
    changed_mask[BLUE_DRAFT[0]] = False
    provisional = replace(
        binding, eligible_nodes=tuple(changed_mask), artifact_sha256=""
    )
    kwargs["eligibility_binding"] = replace(
        provisional,
        artifact_sha256=assay._eligibility_binding_sha256(provisional),
    )
    with pytest.raises(RepresentationRankAssayError, match="bound fit evidence"):
        fit_latent_candidate(split_identity="train", **kwargs)


def test_central_finite_difference_gradient_matches_analytic() -> None:
    rng = np.random.default_rng(22)
    blue = np.vstack((BLUE_DRAFT, RED_DRAFT))
    red = np.vstack((RED_DRAFT, BLUE_DRAFT))
    vector = rng.normal(0, 0.02, size=len(NODE_ROLES) * 3)
    eligible = np.zeros(len(NODE_ROLES), dtype=bool)
    eligible[LEGAL_DRAFT] = True
    kwargs = dict(
        blue=blue,
        red=red,
        target=np.array([1.0, 0.0]),
        p0=np.array([0.45, 0.55]),
        node_domain=NODE_DOMAIN,
        eligible_nodes=eligible,
        width=1,
        lambda_ally=0.1,
        lambda_enemy=0.1,
        mode="joint",
    )
    _, analytic = latent_objective_and_gradient(vector, **kwargs)
    step = 1e-6
    checked = list(LEGAL_DRAFT)
    checked += [len(NODE_ROLES) + node * 2 + axis for node in LEGAL_DRAFT for axis in (0, 1)]
    numeric = np.empty(len(checked))
    for output_index, index in enumerate(checked):
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        numeric[output_index] = (
            latent_objective_and_gradient(plus, **kwargs)[0]
            - latent_objective_and_gradient(minus, **kwargs)[0]
        ) / (2 * step)
    assert np.max(np.abs(numeric - analytic[checked])) < 1e-8


def test_inactive_penalty_family_is_mathematically_invariant() -> None:
    rng = np.random.default_rng(20260729)
    eligible = np.zeros(len(NODE_ROLES), dtype=bool)
    eligible[LEGAL_DRAFT] = True
    common = {
        "vector": rng.normal(0, 0.02, size=len(NODE_ROLES) * 3),
        "blue": np.vstack((BLUE_DRAFT, RED_DRAFT)),
        "red": np.vstack((RED_DRAFT, BLUE_DRAFT)),
        "target": np.asarray([1.0, 0.0]),
        "p0": np.asarray([0.45, 0.55]),
        "node_domain": NODE_DOMAIN,
        "eligible_nodes": eligible,
        "width": 1,
    }
    ally_low = latent_objective_and_gradient(
        **common,
        lambda_ally=0.1,
        lambda_enemy=0.01,
        mode="ally_only",
    )
    ally_high = latent_objective_and_gradient(
        **common,
        lambda_ally=0.1,
        lambda_enemy=100.0,
        mode="ally_only",
    )
    enemy_low = latent_objective_and_gradient(
        **common,
        lambda_ally=0.01,
        lambda_enemy=0.1,
        mode="enemy_only",
    )
    enemy_high = latent_objective_and_gradient(
        **common,
        lambda_ally=100.0,
        lambda_enemy=0.1,
        mode="enemy_only",
    )
    assert ally_low[0] == ally_high[0]
    assert np.array_equal(ally_low[1], ally_high[1])
    assert enemy_low[0] == enemy_high[0]
    assert np.array_equal(enemy_low[1], enemy_high[1])


def test_penalty_and_gradient_ignore_unused_ineligible_vocabulary() -> None:
    rng = np.random.default_rng(303)
    reduced_domain = assay._build_node_domain(
        [{"stable_champion_id": f"champion-{index}"} for index in range(10)],
        source_raw_sha256="a" * 64,
    )
    base_a = rng.normal(0, 0.02, size=(50, 1))
    base_e = rng.normal(0, 0.02, size=(50, 2))
    base_vector = np.concatenate((base_a.ravel(), base_e.ravel()))
    common = dict(
        blue=np.vstack((BLUE_DRAFT, RED_DRAFT)),
        red=np.vstack((RED_DRAFT, BLUE_DRAFT)),
        target=np.array([1.0, 0.0]),
        p0=np.array([0.45, 0.55]),
        width=1,
        lambda_ally=0.1,
        lambda_enemy=0.1,
        mode="joint",
    )
    base_value, base_gradient = latent_objective_and_gradient(
        base_vector,
        node_domain=reduced_domain,
        eligible_nodes=np.ones(50, dtype=bool),
        **common,
    )
    extended_a = np.vstack((base_a, np.zeros((25, 1))))
    extended_e = np.vstack((base_e, np.zeros((25, 2))))
    extended_value, extended_gradient = latent_objective_and_gradient(
        np.concatenate((extended_a.ravel(), extended_e.ravel())),
        node_domain=NODE_DOMAIN,
        eligible_nodes=np.array([True] * 50 + [False] * 25),
        **common,
    )
    extended_a_gradient = extended_gradient[:75].reshape(75, 1)
    extended_e_gradient = extended_gradient[75:].reshape(75, 2)
    reconstructed = np.concatenate(
        (extended_a_gradient[:50].ravel(), extended_e_gradient[:50].ravel())
    )
    assert extended_value == pytest.approx(base_value, abs=1e-15)
    assert np.allclose(reconstructed, base_gradient, atol=1e-15)
    assert np.array_equal(extended_a_gradient[50:], np.zeros((25, 1)))
    assert np.array_equal(extended_e_gradient[50:], np.zeros((25, 2)))


def _gate_fixture():
    y = np.array([0, 1] * 20, dtype=float)
    clusters = np.array([f"c-{index // 2}" for index in range(40)])
    blocks = np.array(["a"] * 20 + ["b"] * 20)
    m0 = np.full(40, 0.5)
    m8 = np.where(y == 1, 0.7, 0.3)
    unique, multiplicity = bootstrap_multiplicity(
        clusters, replicates=100, seed=77
    )
    return y, clusters, blocks, m0, m8, unique, multiplicity


def test_pure_metric_and_block_decisions_use_exact_comparators() -> None:
    strict_boundary = assay.metric_gate_decision(
        comparator="M0",
        metric="log_loss",
        upper=0.0,
    )
    assert strict_boundary == {
        "limit": 0.0,
        "strict": True,
        "passed": False,
    }
    assert assay.metric_gate_decision(
        comparator="M0",
        metric="log_loss",
        upper=np.nextafter(0.0, -np.inf),
    )["passed"] is True

    for comparator, metric, limit in (
        ("M0", "brier", 0.001),
        ("M0", "calibration", 0.01),
        ("M8", "log_loss", 0.002),
        ("M8", "brier", 0.001),
        ("M8", "calibration", 0.01),
    ):
        decision = assay.metric_gate_decision(
            comparator=comparator,
            metric=metric,
            upper=limit,
        )
        assert decision == {
            "limit": limit,
            "strict": False,
            "passed": True,
        }
        assert assay.metric_gate_decision(
            comparator=comparator,
            metric=metric,
            upper=np.nextafter(limit, np.inf),
        )["passed"] is False

    assert assay.block_gate_decision(0.01) is True
    assert assay.block_gate_decision(
        np.nextafter(0.01, np.inf)
    ) is False


@pytest.mark.parametrize(
    ("starts", "gradient", "rms", "expected"),
    (
        (2, assay.GRADIENT_TOLERANCE, assay.STABILITY_RMS_TOLERANCE, True),
        (3, assay.GRADIENT_TOLERANCE, assay.STABILITY_RMS_TOLERANCE, True),
        (1, 0.0, 0.0, False),
        (4, 0.0, 0.0, False),
        (
            2,
            np.nextafter(assay.GRADIENT_TOLERANCE, np.inf),
            0.0,
            False,
        ),
        (
            2,
            0.0,
            np.nextafter(assay.STABILITY_RMS_TOLERANCE, np.inf),
            False,
        ),
        (2, -np.finfo(float).tiny, 0.0, False),
        (2, 0.0, -np.finfo(float).tiny, False),
    ),
)
def test_pure_optimizer_decision_enforces_closed_stability_bounds(
    starts,
    gradient,
    rms,
    expected,
) -> None:
    assert assay.optimization_gate_decision(
        converged_starts=starts,
        max_gradient=gradient,
        stability_rms=rms,
    ) is expected


@pytest.mark.parametrize(
    ("starts", "gradient", "rms"),
    (
        (True, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2, float("nan"), 0.0),
        (2, 0.0, float("inf")),
    ),
)
def test_pure_optimizer_decision_rejects_noncanonical_inputs(
    starts,
    gradient,
    rms,
) -> None:
    with pytest.raises(RepresentationRankAssayError, match="optimization"):
        assay.optimization_gate_decision(
            converged_starts=starts,
            max_gradient=gradient,
            stability_rms=rms,
        )


def test_gate_engine_enforces_all_dual_comparator_gates() -> None:
    y, clusters, blocks, m0, m8, unique, multiplicity = _gate_fixture()

    def evaluate(candidate):
        return evaluate_candidate_gates(
            y=y,
            candidate_probability=candidate,
            m0_probability=m0,
            m8_probability=m8,
            cluster_ids=clusters,
            chronological_blocks=blocks,
            multiplicity=multiplicity,
            unique_clusters=unique,
            endpoint_1_indexed=95,
        )

    assert evaluate(m8)["passed"] is True
    superiority = evaluate(m0)
    assert superiority["comparators"]["M0"]["log_loss"]["passed"] is False
    noninferior = evaluate(np.where(y == 1, 0.58, 0.42))
    assert noninferior["comparators"]["M8"]["log_loss"]["passed"] is False
    brier = evaluate(np.where(y == 1, 0.66, 0.34))
    assert brier["comparators"]["M8"]["brier"]["passed"] is False
    calibration = evaluate(np.clip(m8 + 0.03, 1e-6, 1 - 1e-6))
    assert calibration["comparators"]["M8"]["calibration"]["passed"] is False
    block_candidate = m8.copy()
    block_candidate[:20] = np.where(y[:20] == 1, 0.55, 0.45)
    block = evaluate(block_candidate)
    assert block["comparators"]["M8"]["blocks"]["a"]["passed"] is False


def _prepared_components(split: str):
    source_contract = dict(
        (month, (maps, clusters))
        for month, maps, clusters in assay.CHRONOLOGICAL_BLOCKS[split]
    )
    eligible_contract = dict(
        (month, (maps, clusters))
        for month, maps, clusters in assay.ELIGIBLE_GATE_BLOCKS[split]
    )
    components = []
    for month, (maps, full_clusters) in source_contract.items():
        eligible_maps, eligible_clusters = eligible_contract[month]
        rows = np.tile(LEGAL_DRAFT, (maps, 1))
        rows[eligible_maps:, 0] = INELIGIBLE_TOP_NODE
        clusters = [
            f"{month}-eligible-{index % eligible_clusters}"
            for index in range(eligible_maps)
        ]
        new_clusters = full_clusters - eligible_clusters
        clusters.extend(
            f"{month}-excluded-{index}"
            if index < new_clusters
            else f"{month}-eligible-0"
            for index in range(maps - eligible_maps)
        )
        ids = [f"{month}-game-{index}" for index in range(maps)]
        fit_ids = [f"{month}-fit-game-{index}" for index in range(5)]
        fit_clusters = [f"{month}-fit-{index}" for index in range(5)]
        assignments = [
            {
                "game_id": game_id,
                "dependence_cluster_id": cluster,
                "oe_date_naive": f"{month}-15T00:00:00",
            }
            for game_id, cluster in zip(ids, clusters)
        ] + [
            {
                "game_id": game_id,
                "dependence_cluster_id": cluster,
                "oe_date_naive": "2025-01-01T00:00:00",
            }
            for game_id, cluster in zip(fit_ids, fit_clusters)
        ]
        cluster_domain = assay._build_cluster_domain(
            assignments, source_raw_sha256="b" * 64
        )
        feature_domain = assay._build_feature_domain(
            [
                {
                    "game_id": game_id,
                    "split": split,
                    "league": "LEC",
                    "nodes": nodes,
                }
                for game_id, nodes in zip(ids, rows)
            ]
            + [
                {
                    "game_id": game_id,
                    "split": "train",
                    "league": "LEC",
                    "nodes": LEGAL_DRAFT,
                }
                for game_id in fit_ids
            ],
            node_domain=NODE_DOMAIN,
            cluster_domain=cluster_domain,
            source_raw_sha256="c" * 64,
        )
        verified = {game_id: 0.5 for game_id in ids}
        components.append(
            prepare_outer_fold(
                feature_domain=feature_domain,
                score_game_ids=ids,
                fit_game_ids=fit_ids,
                nuisance_probability=np.full(maps, 0.5),
                verified_nuisance_oof=verified,
                split=split,
                fit_availability_domain=_fixture_fit_availability(feature_domain),
            )
        )
    return components


def _prepared_binding(split: str):
    return combine_prepared_folds(_prepared_components(split), split=split)


def test_development_requires_M8_and_locks_smallest_width(monkeypatch) -> None:
    prepared = _prepared_binding("development")
    y = np.arange(len(prepared.ordered_eligible_game_ids)) % 2
    m0 = np.full(len(y), 0.5)
    good = np.where(y == 1, 0.7, 0.3)
    predictions = {width: good for width in (1, 2, 4, 8)}
    predictions.update({"M0": m0, "M8": good})
    monkeypatch.setattr(assay, "PRIMARY_REPLICATES", 100)
    monkeypatch.setattr(assay, "DEVELOPMENT_ENDPOINT", 95)
    selected, _ = select_development_width(
        target_domain=_targets(dict(
            zip(prepared.ordered_eligible_game_ids, y)
        )),
        predictions=predictions,
        prepared_fold=prepared,
        game_ids=prepared.ordered_eligible_game_ids,
        m8_optimization_stable=True,
    )
    assert selected == 1
    predictions["M8"] = m0
    predictions[8] = m0
    with pytest.raises(RepresentationRankAssayError, match="M8 prerequisite"):
        select_development_width(
            target_domain=_targets(dict(
                zip(prepared.ordered_eligible_game_ids, y)
            )),
            predictions=predictions,
            prepared_fold=prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            m8_optimization_stable=True,
        )


def test_gate_inventory_rejects_full_source_and_accepts_eligible(
    monkeypatch,
) -> None:
    monkeypatch.setattr(assay, "PRIMARY_REPLICATES", 100)
    monkeypatch.setattr(assay, "DEVELOPMENT_ENDPOINT", 95)

    prepared = _prepared_binding("development")
    eligible_y = np.arange(len(prepared.ordered_eligible_game_ids)) % 2
    eligible_m0 = np.full(len(eligible_y), 0.5)
    eligible_good = np.where(eligible_y == 1, 0.7, 0.3)
    eligible_predictions = {width: eligible_good for width in (1, 2, 4, 8)}
    eligible_predictions.update({"M0": eligible_m0, "M8": eligible_good})
    selected, _ = select_development_width(
        target_domain=_targets(dict(
            zip(prepared.ordered_eligible_game_ids, eligible_y)
        )),
        predictions=eligible_predictions,
        prepared_fold=prepared,
        game_ids=prepared.ordered_eligible_game_ids,
        m8_optimization_stable=True,
    )
    assert selected == 1


def test_combiner_rejects_reused_excluded_cluster_across_months() -> None:
    components = _prepared_components("development")
    first, second = components[1], components[2]
    reused = next(
        cluster
        for cluster, eligible in zip(
            first.ordered_full_cluster_ids, first.eligible_rows
        )
        if not eligible
    )
    counts = {
        cluster: second.ordered_full_cluster_ids.count(cluster)
        for cluster in set(second.ordered_full_cluster_ids)
    }
    replace_index = next(
        index
        for index, (cluster, eligible) in enumerate(
            zip(second.ordered_full_cluster_ids, second.eligible_rows)
        )
        if not eligible and counts[cluster] == 1
    )
    changed_clusters = list(second.ordered_full_cluster_ids)
    changed_clusters[replace_index] = reused
    binding = second.eligibility_bindings_by_block[0]
    component_digest = assay._membership_sha256(
        split="development",
        full_game_ids=second.ordered_full_game_ids,
        full_cluster_ids=changed_clusters,
        full_blocks=second.ordered_full_blocks,
        eligible_game_ids=second.ordered_eligible_game_ids,
        eligible_cluster_ids=second.ordered_eligible_cluster_ids,
        eligible_blocks=second.ordered_eligible_blocks,
        m0_probability=second.m0_probability,
        eligibility_binding_sha256=[binding.artifact_sha256],
    )
    changed = replace(
        second,
        ordered_full_cluster_ids=tuple(changed_clusters),
        membership_sha256=component_digest,
        component_membership_sha256=(component_digest,),
    )
    with pytest.raises(RepresentationRankAssayError, match="spans"):
        combine_prepared_folds(
            [components[0], first, changed, components[3]], split="development"
        )


def test_gate_binding_rejects_reordering_substitution_and_cross_block_cluster(
    monkeypatch,
) -> None:
    prepared = _prepared_binding("development")
    rows = len(prepared.ordered_eligible_game_ids)
    y = np.arange(rows) % 2
    good = np.where(y == 1, 0.7, 0.3)
    predictions = {width: good.copy() for width in (1, 2, 4, 8)}
    predictions.update({"M0": prepared.m0_probability.copy(), "M8": good.copy()})
    monkeypatch.setattr(assay, "PRIMARY_REPLICATES", 100)
    monkeypatch.setattr(assay, "DEVELOPMENT_ENDPOINT", 95)
    reordered = list(prepared.ordered_eligible_game_ids)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RepresentationRankAssayError, match="identity/order"):
        select_development_width(
            prepared_fold=prepared,
            game_ids=reordered,
            target_domain=_targets(dict(
                zip(prepared.ordered_eligible_game_ids, y)
            )),
            predictions=predictions,
            m8_optimization_stable=True,
        )
    substituted = copy.deepcopy(predictions)
    substituted["M0"][0] = np.nextafter(substituted["M0"][0], 1.0)
    with pytest.raises(RepresentationRankAssayError, match="bitwise"):
        select_development_width(
            prepared_fold=prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            target_domain=_targets(dict(
                zip(prepared.ordered_eligible_game_ids, y)
            )),
            predictions=substituted,
            m8_optimization_stable=True,
        )
    clusters = list(prepared.ordered_eligible_cluster_ids)
    clusters[0] = clusters[-1]
    full_clusters = list(prepared.ordered_full_cluster_ids)
    first_eligible_full_index = int(np.flatnonzero(prepared.eligible_rows)[0])
    full_clusters[first_eligible_full_index] = clusters[-1]
    corrupt = replace(
        prepared,
        ordered_full_cluster_ids=tuple(full_clusters),
        ordered_eligible_cluster_ids=tuple(clusters),
    )
    with pytest.raises(RepresentationRankAssayError, match="spans"):
        select_development_width(
            prepared_fold=corrupt,
            game_ids=prepared.ordered_eligible_game_ids,
            target_domain=_targets(dict(
                zip(prepared.ordered_eligible_game_ids, y)
            )),
            predictions=predictions,
            m8_optimization_stable=True,
        )
    first_binding = prepared.eligibility_bindings_by_block[0]
    flipped_mask = list(first_binding.eligible_nodes)
    flipped_mask[INELIGIBLE_TOP_NODE] = not flipped_mask[INELIGIBLE_TOP_NODE]
    corrupt_binding = replace(
        first_binding, eligible_nodes=tuple(flipped_mask)
    )
    corrupt_prepared = replace(
        prepared,
        eligibility_bindings_by_block=(
            corrupt_binding,
            *prepared.eligibility_bindings_by_block[1:],
        ),
    )
    with pytest.raises(RepresentationRankAssayError, match="eligibility binding"):
        select_development_width(
            prepared_fold=corrupt_prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            target_domain=_targets(
                dict(zip(prepared.ordered_eligible_game_ids, y))
            ),
            predictions=predictions,
            m8_optimization_stable=True,
        )
    swapped_rows = prepared.eligible_rows.copy()
    true_index = int(np.flatnonzero(swapped_rows)[0])
    false_index = int(np.flatnonzero(~swapped_rows)[0])
    swapped_rows[true_index], swapped_rows[false_index] = (
        swapped_rows[false_index],
        swapped_rows[true_index],
    )
    swapped_prepared = replace(prepared, eligible_rows=swapped_rows)
    with pytest.raises(RepresentationRankAssayError, match="filter full"):
        select_development_width(
            prepared_fold=swapped_prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            target_domain=_targets(
                dict(zip(prepared.ordered_eligible_game_ids, y))
            ),
            predictions=predictions,
            m8_optimization_stable=True,
        )
    valid_target = _targets(dict(zip(prepared.ordered_eligible_game_ids, y)))
    tampered_rows = list(valid_target.ordered_targets)
    tampered_rows[0] = (tampered_rows[0][0], 1 - tampered_rows[0][1])
    corrupt_target = replace(
        valid_target, ordered_targets=tuple(tampered_rows)
    )
    with pytest.raises(RepresentationRankAssayError, match="target-domain"):
        select_development_width(
            prepared_fold=prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            target_domain=corrupt_target,
            predictions=predictions,
            m8_optimization_stable=True,
        )


def test_rehashed_eligibility_mask_and_deleted_full_row_still_fail() -> None:
    prepared = _prepared_binding("development")
    first_binding = prepared.eligibility_bindings_by_block[0]
    substituted_availability = assay._build_fit_availability_domain(
        first_binding.fit_availability_domain.ordered_game_ids,
        source_raw_sha256="e" * 64,
    )
    substituted = replace(
        first_binding,
        fit_availability_domain=substituted_availability,
        fit_availability_domain_sha256=substituted_availability.artifact_sha256,
        fit_availability_source_raw_sha256=(
            substituted_availability.source_raw_sha256
        ),
        artifact_sha256="",
    )
    substituted = replace(
        substituted,
        artifact_sha256=assay._eligibility_binding_sha256(substituted),
    )
    substituted_prepared = replace(
        prepared,
        eligibility_bindings_by_block=(
            substituted,
            *prepared.eligibility_bindings_by_block[1:],
        ),
    )
    with pytest.raises(
        RepresentationRankAssayError, match="membership digest mismatch"
    ):
        assay._validate_prepared_fold(substituted_prepared, split="development")

    changed_mask = list(first_binding.eligible_nodes)
    changed_mask[0], changed_mask[1] = changed_mask[1], changed_mask[0]
    provisional = replace(
        first_binding, eligible_nodes=tuple(changed_mask), artifact_sha256=""
    )
    rehashed_binding = replace(
        provisional,
        artifact_sha256=assay._eligibility_binding_sha256(provisional),
    )
    rehashed_prepared = replace(
        prepared,
        eligibility_bindings_by_block=(
            rehashed_binding,
            *prepared.eligibility_bindings_by_block[1:],
        ),
    )
    with pytest.raises(RepresentationRankAssayError, match="bound fit evidence"):
        assay._validate_prepared_fold(rehashed_prepared, split="development")

    shortened_source = replace(
        first_binding,
        ordered_source_game_ids=first_binding.ordered_source_game_ids[1:],
        ordered_source_cluster_ids=first_binding.ordered_source_cluster_ids[1:],
        artifact_sha256="",
    )
    shortened_source = replace(
        shortened_source,
        artifact_sha256=assay._eligibility_binding_sha256(shortened_source),
    )
    changed_bindings = (
        shortened_source,
        *prepared.eligibility_bindings_by_block[1:],
    )
    first_block = prepared.ordered_full_blocks[0]
    full_selected = np.asarray(prepared.ordered_full_blocks) == first_block
    eligible_selected = (
        np.asarray(prepared.ordered_eligible_blocks) == first_block
    )
    changed_components = list(prepared.component_membership_sha256)
    changed_components[0] = assay._membership_sha256(
        split="development",
        full_game_ids=tuple(
            np.asarray(prepared.ordered_full_game_ids)[full_selected]
        ),
        full_cluster_ids=tuple(
            np.asarray(prepared.ordered_full_cluster_ids)[full_selected]
        ),
        full_blocks=tuple(
            np.asarray(prepared.ordered_full_blocks)[full_selected]
        ),
        eligible_game_ids=tuple(
            np.asarray(prepared.ordered_eligible_game_ids)[eligible_selected]
        ),
        eligible_cluster_ids=tuple(
            np.asarray(prepared.ordered_eligible_cluster_ids)[eligible_selected]
        ),
        eligible_blocks=tuple(
            np.asarray(prepared.ordered_eligible_blocks)[eligible_selected]
        ),
        m0_probability=prepared.m0_probability[eligible_selected],
        eligibility_binding_sha256=[shortened_source.artifact_sha256],
    )
    ordered_digest = assay._membership_sha256(
        split="development",
        full_game_ids=prepared.ordered_full_game_ids,
        full_cluster_ids=prepared.ordered_full_cluster_ids,
        full_blocks=prepared.ordered_full_blocks,
        eligible_game_ids=prepared.ordered_eligible_game_ids,
        eligible_cluster_ids=prepared.ordered_eligible_cluster_ids,
        eligible_blocks=prepared.ordered_eligible_blocks,
        m0_probability=prepared.m0_probability,
        eligibility_binding_sha256=[
            binding.artifact_sha256 for binding in changed_bindings
        ],
    )
    source_attacker = replace(
        prepared,
        eligibility_bindings_by_block=changed_bindings,
        component_membership_sha256=tuple(changed_components),
        membership_sha256=canonical_sha256(
            {
                "split": "development",
                "ordered_component_membership_sha256": changed_components,
                "ordered_eligible_membership_sha256": ordered_digest,
            }
        ),
    )
    with pytest.raises(
        RepresentationRankAssayError, match="differs from parent"
    ):
        assay._validate_prepared_fold(source_attacker, split="development")

    delete_index = next(
        index
        for index, (block, eligible) in enumerate(
            zip(prepared.ordered_full_blocks, prepared.eligible_rows)
        )
        if block == "2025-10" and not eligible
    )
    keep = np.arange(len(prepared.eligible_rows)) != delete_index
    shortened = replace(
        prepared,
        eligible_rows=prepared.eligible_rows[keep],
        ordered_full_game_ids=tuple(
            np.asarray(prepared.ordered_full_game_ids, dtype=object)[keep]
        ),
        ordered_full_cluster_ids=tuple(
            np.asarray(prepared.ordered_full_cluster_ids, dtype=object)[keep]
        ),
        ordered_full_blocks=tuple(
            np.asarray(prepared.ordered_full_blocks, dtype=object)[keep]
        ),
    )
    ordered_digest = assay._membership_sha256(
        split="development",
        full_game_ids=shortened.ordered_full_game_ids,
        full_cluster_ids=shortened.ordered_full_cluster_ids,
        full_blocks=shortened.ordered_full_blocks,
        eligible_game_ids=shortened.ordered_eligible_game_ids,
        eligible_cluster_ids=shortened.ordered_eligible_cluster_ids,
        eligible_blocks=shortened.ordered_eligible_blocks,
        m0_probability=shortened.m0_probability,
        eligibility_binding_sha256=[
            binding.artifact_sha256
            for binding in shortened.eligibility_bindings_by_block
        ],
    )
    shortened = replace(
        shortened,
        membership_sha256=canonical_sha256(
            {
                "split": "development",
                "ordered_component_membership_sha256": list(
                    shortened.component_membership_sha256
                ),
                "ordered_eligible_membership_sha256": ordered_digest,
            }
        ),
    )
    with pytest.raises(
        RepresentationRankAssayError, match="inventory|component"
    ):
        assay._validate_prepared_fold(shortened, split="development")


def test_width8_and_M8_must_be_bitwise_identical(monkeypatch) -> None:
    prepared = _prepared_binding("development")
    y = np.arange(len(prepared.ordered_eligible_game_ids)) % 2
    m0 = np.full(len(y), 0.5)
    good = np.where(y == 1, 0.7, 0.3)
    predictions = {width: good.copy() for width in (1, 2, 4, 8)}
    predictions.update({"M0": m0, "M8": good.copy()})
    predictions[8][0] += 1e-12
    monkeypatch.setattr(assay, "PRIMARY_REPLICATES", 100)
    monkeypatch.setattr(assay, "DEVELOPMENT_ENDPOINT", 95)
    with pytest.raises(RepresentationRankAssayError, match="diverged"):
        select_development_width(
            target_domain=_targets(dict(
                zip(prepared.ordered_eligible_game_ids, y)
            )),
            predictions=predictions,
            prepared_fold=prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            m8_optimization_stable=True,
        )


def test_validation_candidate_set_cannot_reselect() -> None:
    prepared = _prepared_binding("validation")
    with pytest.raises(RepresentationRankAssayError, match="reselection"):
        validate_locked_width(
            prepared_fold=prepared,
            game_ids=prepared.ordered_eligible_game_ids,
            locked_width=2,
            target_domain=_targets({}),
            predictions={
                1: [0.4, 0.6],
                2: [0.4, 0.6],
                "M0": [0.5, 0.5],
                "M8": [0.4, 0.6],
            },
            m8_optimization_stable=True,
        )


def test_nonholdout_loader_applies_predicate_before_return(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "private.parquet"
    path.write_bytes(b"fixture")
    observed = {}

    def fake_read(path_arg, *, columns, filters):
        observed["filters"] = filters
        return pd.DataFrame(
            {"split": ["train"], "y_blue_win": [1]}
        ).loc[:, columns]

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    frame = load_nonholdout_rows(
        path, columns=["split", "y_blue_win"]
    )
    assert observed["filters"] == [
        ("split", "in", ["train", "development", "validation"])
    ]
    assert frame["split"].tolist() == ["train"]


def test_cluster_bootstrap_is_pcg64dxsm_sorted_and_uses_frozen_endpoints() -> None:
    names, first = bootstrap_multiplicity(
        ["z", "a", "z", "m"], replicates=20_000, seed=2026072901
    )
    _, second = bootstrap_multiplicity(
        ["m", "z", "a", "z"], replicates=20_000, seed=2026072901
    )
    assert names.tolist() == ["a", "m", "z"]
    assert np.array_equal(first, second)
    assert nearest_rank_endpoint(np.arange(20_000), one_indexed=19_875) == 19_874
    assert nearest_rank_endpoint(np.arange(20_000), one_indexed=19_667) == 19_666


def test_frozen_gate_blocks_directly_reject_cross_block_cluster() -> None:
    prepared = _prepared_binding("validation")
    blocks = np.asarray(prepared.ordered_eligible_blocks, dtype=object)
    clusters = np.asarray(
        prepared.ordered_eligible_cluster_ids, dtype=object
    ).copy()
    first_second_block = int(np.flatnonzero(blocks != blocks[0])[0])
    clusters[first_second_block] = clusters[0]
    with pytest.raises(RepresentationRankAssayError, match="spans"):
        assay._validate_frozen_blocks(blocks, clusters, split="validation")


def test_config_freezes_bonferroni_families_and_cluster_max_blocks() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_bytes())
    assert config["bootstrap"]["development_family"] == "4 widths x 2 hypotheses"
    assert (
        config["bootstrap"]["validation_family"]
        == "M8 superiority to M0; locked superiority to M0; locked noninferiority to M8"
    )
    assert config["chronological_blocks"]["development"][1] == {
        "calendar_month": "2026-01",
        "maps": 342,
        "clusters": 190,
    }
