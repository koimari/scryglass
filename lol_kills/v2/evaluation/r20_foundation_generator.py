"""Deterministic generator→observation→inference R-20 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

from .r20_foundation_inference import (
    INFERENCE_ADAPTER_ID,
    INFERENCE_SEED,
    POSTERIOR_DRAWS,
    PRIOR_ALPHA,
    PRIOR_BETA,
    REFERENCE_MODES,
    infer_beta_binomial,
)
from .types import canonical_json, canonical_sha256


OUTPUT_STRATA = (
    ("player_rating", "stratum-player"),
    ("team_rating", "stratum-team"),
    ("draft_score", "stratum-draft"),
    ("partial_draft_state", "stratum-prefix"),
    ("tier_list", "stratum-tier"),
)
REGIMES = (
    "symmetric",
    "low_skew",
    "high_skew",
    "boundary_heavy",
    "volume_quadratic_null",
)
SOURCE_CONTEXT_PATTERNS = (
    "all_good",
    "bridge_missing",
    "context_missing",
    "lineage_incomplete",
    "fallback_used",
    "combined_missing",
)

FOUNDATION_DRAWS = POSTERIOR_DRAWS
FOUNDATION_MAPS_PER_SERIES = 2
INITIAL_TRAIN_SERIES_PER_CELL = 40
TEST_SERIES_PER_CELL_PER_FOLD = 40
FOUNDATION_FOLDS = 3
FOUNDATION_SERIES_PER_CELL = (
    INITIAL_TRAIN_SERIES_PER_CELL + FOUNDATION_FOLDS * TEST_SERIES_PER_CELL_PER_FOLD
)
FOUNDATION_SERIES_TOTAL = len(OUTPUT_STRATA) * FOUNDATION_SERIES_PER_CELL
FAMILY_DEFAULT_SEED = 20260728
FIXTURE_LABEL_DGP_SEED = 20260731
FIXTURE_LABEL_DGP_ID = "r20-preregistered-balanced-fixture-classification-v3"
VOLUME_BASIS_ID = "r20-volume-basis-v2"
VOLUME_CONDITION_BOUND = 1.0e8
VOLUME_RANK_TOLERANCE = 1.0e-10
REGISTERED_VOLUME_FIELDS = (
    "volume_signal",
    "sample_size",
    "game_count",
    "pick_rate",
    "play_rate",
    "popularity",
)


@dataclass(frozen=True)
class GeneratorContract:
    artifact_id: str
    module: str
    entrypoint: str
    simulation_seed: int
    chronological_folds: int


FAMILY_DEFAULT = GeneratorContract(
    artifact_id="scryglass:b2:r20-foundation-generator:v2",
    module="lol_kills.v2.evaluation.r20_foundation_generator",
    entrypoint="build_r20_benchmark",
    simulation_seed=FAMILY_DEFAULT_SEED,
    chronological_folds=FOUNDATION_FOLDS,
)


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-value)))


def _series_theta(
    rng: np.random.Generator,
    *,
    regime: str,
    volume_signal: float,
    support_stratum: int,
) -> float:
    if type(support_stratum) is not int or not 0 <= support_stratum < 8:
        raise ValueError("support stratum must be in 0..7")
    quantile = (support_stratum + float(rng.random())) / 8.0
    if regime == "symmetric":
        theta = beta_distribution.ppf(quantile, 4.0, 4.0)
    elif regime == "low_skew":
        theta = beta_distribution.ppf(quantile, 2.0, 7.0)
    elif regime == "high_skew":
        theta = beta_distribution.ppf(quantile, 7.0, 2.0)
    elif regime == "boundary_heavy":
        theta = beta_distribution.ppf(quantile, 0.65, 0.65)
    elif regime == "volume_quadratic_null":
        # Keep the null signal distinct from candidate IDs and from inference truth.
        theta = _logistic(-0.35 + 4.0 * (volume_signal - 0.5) ** 2)
    else:
        raise ValueError("unknown generator regime")
    return float(min(1.0 - 1e-9, max(1e-9, theta)))


def _source_context(pattern: str, *, output_type: str, series_index: int) -> dict[str, Any]:
    lineage_complete = pattern not in {"lineage_incomplete", "combined_missing"}
    context_registered = pattern not in {"context_missing", "combined_missing"}
    bridge_registered = pattern not in {"bridge_missing", "combined_missing"}
    fallback_used = pattern in {"fallback_used", "combined_missing"}
    return {
        "source_lineage": {
            "complete": lineage_complete,
            "registered": True,
        },
        "context_registry": {
            "registered": context_registered,
            "registry_version": "r20-context-v1",
            "path": f"{output_type}:series-{series_index:03d}",
        },
        "fallback_registry": {
            "used": fallback_used,
            "profile": "fallback" if fallback_used else "none",
        },
        "bridge_registry": {
            "registered": bridge_registered,
            "bridge_id": "r20-synthetic-bridge-v1",
        },
        "pattern_id": pattern,
    }


def _ordered_series_plan() -> list[tuple[str, str, str, str, int, int]]:
    """Yield ordered (output, stratum, cohort_id, regime, output_index, regime_offset)."""

    series_design: list[tuple[str, str, str, str, int, int]] = []
    for cohort_id in ("initial_train", "test_fold_0", "test_fold_1", "test_fold_2"):
        for output_index, (output_type, stratum_id) in enumerate(OUTPUT_STRATA):
            for regime_index, regime in enumerate(REGIMES):
                for series_within_regime in range(8):
                    series_design.append(
                        (
                            output_type,
                            stratum_id,
                            cohort_id,
                            regime,
                            output_index,
                            series_within_regime,
                        )
                    )
    return series_design


def _volume_profile(
    global_series_index: int,
    *,
    regime: str | None = None,
    support_stratum: int | None = None,
) -> tuple[float, int, float, float, float, float]:
    # Deterministic synthetic volume and derived fields.
    if support_stratum is None:
        raise ValueError("volume profile requires preregistered fixture stratum")
    group_index = global_series_index // 8
    paired_stratum = support_stratum % 4
    # Each volume profile is shared by one label-0 and one label-1 series
    # inside every output×cohort×regime group.
    volume_index = group_index * 7 + paired_stratum * 5 + 11
    volume_signal = float(0.08 + 0.84 * ((volume_index % 31) / 30.0))
    sample_size = 12 + (volume_index % 8)
    game_count = 1 + (volume_index % 9)
    pick_rate = float(0.05 + 0.9 * volume_signal)
    play_rate = float(0.08 + 0.8 * volume_signal)
    popularity = float(0.1 + 0.7 * volume_signal)
    return volume_signal, sample_size, game_count, pick_rate, play_rate, popularity


def _build_series_rows(
    *,
    seed: int,
    base: datetime,
    start_index: int,
    output_type: str,
    stratum_id: str,
    cohort_id: str,
    regime: str,
    regime_index: int,
    series_in_regime_offset: int,
    fixture_label: int,
    fixture_label_probability: float,
    fixture_label_uniform: float,
) -> list[dict[str, Any]]:
    series_id = f"{output_type}-{stratum_id}-{start_index:04d}"
    volume_signal, sample_size, game_count, pick_rate, play_rate, popularity = _volume_profile(
        start_index,
        regime=regime,
        support_stratum=series_in_regime_offset,
    )
    theta = _series_theta(
        np.random.default_rng(seed + start_index),
        regime=regime,
        volume_signal=volume_signal,
        support_stratum=series_in_regime_offset,
    )
    context_pattern = SOURCE_CONTEXT_PATTERNS[
        (start_index + regime_index) % len(SOURCE_CONTEXT_PATTERNS)
    ]
    context = _source_context(context_pattern, output_type=output_type, series_index=start_index)
    rows: list[dict[str, Any]] = []

    # Keep generator truths fixed across maps while rows vary through stable row-level evidence.
    for map_index in range(FOUNDATION_MAPS_PER_SERIES):
        issued = base + timedelta(minutes=start_index * 30 + map_index * 8 + 2 * map_index)
        event = issued + timedelta(minutes=2)
        resolved = issued + timedelta(minutes=6)

        trials = [12, 24, 36, 48][(start_index + map_index) % 4]
        successes = int(
            np.random.default_rng(seed + start_index * 17 + map_index + (start_index % 11)).binomial(
                trials,
                theta,
            )
        )
        inference_seed = INFERENCE_SEED + start_index * 11 + map_index
        inference = infer_beta_binomial(
            observation={"successes": successes, "trials": trials},
            inference_seed=inference_seed,
            draw_count=FOUNDATION_DRAWS,
            prior_alpha=PRIOR_ALPHA,
            prior_beta=PRIOR_BETA,
            reference_mode=REFERENCE_MODES[(start_index + map_index) % len(REFERENCE_MODES)],
        )

        row_id = f"{output_type}-{start_index:03d}-{map_index}"

        rows.append(
            {
                "row_id": row_id,
                "case_id": f"case-{start_index * FOUNDATION_MAPS_PER_SERIES + map_index:05d}",
                "series_id": series_id,
                "output_type": output_type,
                "stratum_id": stratum_id,
                "cohort_id": cohort_id,
                "issued": issued.isoformat(),
                "event": event.isoformat(),
                "resolved": resolved.isoformat(),
                "latent_truth": theta,
                "fixture_label": fixture_label,
                "fixture_label_dgp": {
                    "dgp_id": FIXTURE_LABEL_DGP_ID,
                    "seed": FIXTURE_LABEL_DGP_SEED,
                    "stratum_rule": (
                        "balanced_fixture_strata_0_3_vs_4_7"
                    ),
                    "probability_rule": "p0_strata_0_3_p1_strata_4_7",
                    "target_kind": "balanced_fixture_classification",
                    "probability_semantics": "fixture_class_probability",
                    "proper_score_eligible": False,
                    "probability": fixture_label_probability,
                    "uniform_draw": fixture_label_uniform,
                    "support_stratum": series_in_regime_offset,
                    "sampling_weight": 1.0,
                },
                "volume_inputs": {
                    "volume_signal": volume_signal,
                    "sample_size": sample_size,
                    "game_count": game_count,
                    "pick_rate": pick_rate,
                    "play_rate": play_rate,
                    "popularity": popularity,
                },
                "candidate_inputs": {
                    "generator_regime": regime,
                    "observation": {"successes": successes, "trials": trials},
                    "inference": {
                        "adapter_id": INFERENCE_ADAPTER_ID,
                        "inference_seed": inference_seed,
                        "draw_count": FOUNDATION_DRAWS,
                        "prior": {
                            "alpha": PRIOR_ALPHA,
                            "beta": PRIOR_BETA,
                        },
                        "reference_mode": inference["reference_mode"],
                        "posterior_parameters": inference["posterior_parameters"],
                        "inference_output_sha256": inference["inference_output_sha256"],
                    },
                    "posterior_draws": inference["posterior_draws"],
                    "prior_draws": inference["prior_draws"],
                    "registered_reference_draws": inference["registered_reference_draws"],
                    "source_lineage": context["source_lineage"],
                    "context_registry": context["context_registry"],
                    "fallback_registry": context["fallback_registry"],
                    "bridge_registry": context["bridge_registry"],
                    "source_context_pattern": context["pattern_id"],
                },
                "lineage": {
                    "generator": FAMILY_DEFAULT.entrypoint,
                    "generator_regime": regime,
                    "cohort_id": cohort_id,
                    "source_id": f"synthetic-source-{start_index:04d}",
                    "inference_output_sha256": inference["inference_output_sha256"],
                },
            }
        )

    return rows


def _build_series_layout() -> list[list[dict[str, Any]]]:
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    design = _ordered_series_plan()
    for series_index, (output_type, stratum_id, cohort_id, regime, output_index, support_stratum) in enumerate(
        design
    ):
        probability = float(support_stratum >= 4)
        uniform_draw = float(
            np.random.default_rng(FIXTURE_LABEL_DGP_SEED + series_index).random(),
        )
        fixture_label = int(uniform_draw < probability)
        rows.extend(
            _build_series_rows(
                seed=FAMILY_DEFAULT.simulation_seed,
                base=base,
                start_index=series_index,
                output_type=output_type,
                stratum_id=stratum_id,
                cohort_id=cohort_id,
                regime=regime,
                regime_index=REGIMES.index(regime),
                series_in_regime_offset=support_stratum,
                fixture_label=fixture_label,
                fixture_label_probability=probability,
                fixture_label_uniform=uniform_draw,
            )
        )
    return sorted(rows, key=lambda row: (row["issued"], row["row_id"]))


def _volume_readiness_for_fold(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    fold_id: str,
) -> dict[str, Any]:
    def series_representatives(
        rows: Sequence[Mapping[str, Any]],
        output_type: str,
    ) -> list[Mapping[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if row["output_type"] == output_type:
                grouped.setdefault(str(row["series_id"]), []).append(row)
        representatives: list[Mapping[str, Any]] = []
        for series_id, members in sorted(grouped.items()):
            if len(members) != FOUNDATION_MAPS_PER_SERIES:
                raise ValueError("volume readiness requires exact two-map series")
            first = members[0]["volume_inputs"]
            if any(member["volume_inputs"] != first for member in members[1:]):
                raise ValueError("map-level volume inputs differ within series")
            representatives.append(min(members, key=lambda row: row["row_id"]))
        return representatives

    def transform_columns(
        members: Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Mapping[str, float]],
    ) -> tuple[list[str], np.ndarray]:
        names = ["intercept"]
        columns = [np.ones(len(members), dtype=float)]
        for field_id in REGISTERED_VOLUME_FIELDS:
            values = np.asarray(
                [float(row["volume_inputs"][field_id]) for row in members],
                dtype=float,
            )
            missing = ~np.isfinite(values)
            parameter = parameters[field_id]
            imputed = values.copy()
            imputed[missing] = parameter["imputation"]
            normalized = (imputed - parameter["center"]) / parameter["scale"]
            names.extend(
                [
                    f"{field_id}:linear",
                    f"{field_id}:quadratic",
                    f"{field_id}:missing",
                ],
            )
            columns.extend([normalized, normalized**2, missing.astype(float)])
        return names, np.column_stack(columns)

    outputs: list[dict[str, Any]] = []
    for output_type, _ in OUTPUT_STRATA:
        training = series_representatives(train_rows, output_type)
        testing = series_representatives(test_rows, output_type)
        if not training or not testing:
            raise ValueError("every output must appear in fold train and test")
        parameters: dict[str, dict[str, float]] = {}
        for field_id in REGISTERED_VOLUME_FIELDS:
            values = np.asarray(
                [float(row["volume_inputs"][field_id]) for row in training],
                dtype=float,
            )
            valid = np.isfinite(values)
            if not valid.any():
                raise ValueError("fold readiness requires finite volume values")
            imputation = float(np.median(values[valid]))
            imputed = values.copy()
            imputed[~valid] = imputation
            center = float(np.mean(imputed))
            scale = float(np.std(imputed, ddof=1))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("fold volume readiness requires nonzero spread")
            parameters[field_id] = {
                "imputation": imputation,
                "center": center,
                "scale": scale,
                "missing_count": int(np.sum(~valid)),
            }

        names, candidate_matrix = transform_columns(training, parameters)
        test_names, test_candidate_matrix = transform_columns(testing, parameters)
        if names != test_names:
            raise ValueError("volume term order changed between train and test")
        retained_indices: list[int] = []
        dropped_terms: list[str] = []
        current = np.empty((candidate_matrix.shape[0], 0), dtype=float)
        current_rank = 0
        for index, name in enumerate(names):
            proposed = np.column_stack([current, candidate_matrix[:, index]])
            proposed_rank = int(
                np.linalg.matrix_rank(proposed, tol=VOLUME_RANK_TOLERANCE),
            )
            if proposed_rank > current_rank:
                retained_indices.append(index)
                current = proposed
                current_rank = proposed_rank
            else:
                dropped_terms.append(name)
        retained_terms = [names[index] for index in retained_indices]
        for required in (
            "volume_signal:linear",
            "volume_signal:quadratic",
        ):
            if required not in retained_terms:
                raise ValueError("generator-null volume span was dropped")
        condition_number = float(np.linalg.cond(current))
        if (
            not math.isfinite(condition_number)
            or condition_number > VOLUME_CONDITION_BOUND
        ):
            raise ValueError("combined fold volume design exceeds condition bound")
        retained_test = test_candidate_matrix[:, retained_indices]
        volume_signal = np.asarray(
            [float(row["volume_inputs"]["volume_signal"]) for row in training],
            dtype=float,
        )
        generator_null = (volume_signal - 0.5) ** 2
        appended_rank = int(
            np.linalg.matrix_rank(
                np.column_stack([current, generator_null]),
                tol=VOLUME_RANK_TOLERANCE,
            ),
        )
        if appended_rank != current_rank:
            raise ValueError("generator quadratic null is outside trained basis")
        training_series_ids = [str(row["series_id"]) for row in training]
        test_series_ids = [str(row["series_id"]) for row in testing]
        design_payload = {
            "training_series_ids": training_series_ids,
            "parameters": parameters,
            "candidate_term_order": names,
            "retained_terms": retained_terms,
            "dropped_terms": dropped_terms,
            "rank_tolerance": VOLUME_RANK_TOLERANCE,
        }
        outputs.append(
            {
                "output_type": output_type,
                "training_series_ids": training_series_ids,
                "training_series_sha256": canonical_sha256(training_series_ids),
                "test_series_ids": test_series_ids,
                "test_series_sha256": canonical_sha256(test_series_ids),
                "field_parameters": parameters,
                "candidate_term_order": names,
                "retained_terms": retained_terms,
                "dropped_terms": dropped_terms,
                "retained_rank": current_rank,
                "condition_bound": VOLUME_CONDITION_BOUND,
                "achieved_condition": condition_number,
                "design_sha256": canonical_sha256(
                    {
                        **design_payload,
                        "matrix": current.tolist(),
                    },
                ),
                "test_transform_sha256": canonical_sha256(
                    {
                        "training_design": design_payload,
                        "test_series_ids": test_series_ids,
                        "matrix": retained_test.tolist(),
                    },
                ),
                "generator_null": {
                    "expression": "(volume_signal-0.5)^2",
                    "base_rank": current_rank,
                    "appended_rank": appended_rank,
                    "additional_rank": appended_rank - current_rank,
                    "arbitrary_center_identity": (
                        "(v-0.5)^2=(c-0.5)^2+2(c-0.5)(v-c)+(v-c)^2"
                    ),
                },
            },
        )

    return {
        "fold_id": fold_id,
        "basis_id": VOLUME_BASIS_ID,
        "weighting": "one_equal_weight_observation_per_series",
        "preprocessing": "training_only",
        "rank_policy": "deterministic_incremental_rank_revealing",
        "ready": all(
            output["retained_rank"] == len(output["retained_terms"])
            and output["achieved_condition"] <= output["condition_bound"]
            and output["generator_null"]["additional_rank"] == 0
            for output in outputs
        ),
        "outputs": outputs,
    }


def _series_fold_id(output_type: str, stratum_id: str) -> str:
    return f"{output_type}:{stratum_id}"


def _volume_target_nonseparability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    representatives: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        representatives.setdefault(str(row["series_id"]), row)
    scopes: list[tuple[str, list[Mapping[str, Any]]]] = [
        ("aggregate", list(representatives.values())),
    ]
    for output_type, _ in OUTPUT_STRATA:
        for cohort_id in (
            "initial_train",
            "test_fold_0",
            "test_fold_1",
            "test_fold_2",
        ):
            for regime in REGIMES:
                scopes.append(
                    (
                        f"{output_type}:{cohort_id}:{regime}",
                        [
                            row
                            for row in representatives.values()
                            if row["output_type"] == output_type
                            and row["cohort_id"] == cohort_id
                            and row["candidate_inputs"]["generator_regime"] == regime
                        ],
                    ),
                )
    scope_reports: list[dict[str, Any]] = []
    for scope_id, members in scopes:
        if not members:
            raise ValueError("volume target audit scope is empty")
        fields: list[dict[str, Any]] = []
        for field_id in REGISTERED_VOLUME_FIELDS:
            by_value: dict[float, list[int]] = {}
            for row in members:
                value = float(row["volume_inputs"][field_id])
                by_value.setdefault(value, []).append(int(row["fixture_label"]))
            correct = sum(
                max(labels.count(0), labels.count(1))
                for labels in by_value.values()
            )
            accuracy = correct / len(members)
            mixed_value_count = sum(
                1 for labels in by_value.values() if set(labels) == {0, 1}
            )
            if accuracy >= 1.0 or mixed_value_count == 0:
                raise ValueError(
                    "registered volume field perfectly predicts fixture target "
                    f"in {scope_id}: {field_id}",
                )
            fields.append(
                {
                    "field_id": field_id,
                    "exact_lookup_accuracy": accuracy,
                    "mixed_value_count": mixed_value_count,
                    "unique_value_count": len(by_value),
                    "perfect_prediction": False,
                },
            )
        scope_reports.append(
            {
                "scope_id": scope_id,
                "series_count": len(members),
                "fields": fields,
                "passes": True,
            },
        )
    return {
        "target_kind": "balanced_fixture_classification",
        "series_count": len(representatives),
        "audit": "exact_single_field_lookup_by_conditioning_scope",
        "scope_count": len(scope_reports),
        "scopes": scope_reports,
        "passes": True,
    }


def build_prequential_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    chronological_folds: int = FOUNDATION_FOLDS,
    initial_series_per_cell: int = INITIAL_TRAIN_SERIES_PER_CELL,
    test_series_per_cell: int = TEST_SERIES_PER_CELL_PER_FOLD,
) -> dict[str, Any]:
    if chronological_folds != FOUNDATION_FOLDS:
        raise ValueError("exactly three folds are required")
    if initial_series_per_cell != INITIAL_TRAIN_SERIES_PER_CELL:
        raise ValueError("initial per-cell series must be exactly 40")
    if test_series_per_cell != TEST_SERIES_PER_CELL_PER_FOLD:
        raise ValueError("test per-cell series per fold must be exactly 40")
    if not rows:
        raise ValueError("rows are required")

    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate row IDs")

    required_series = len(OUTPUT_STRATA) * (
        initial_series_per_cell + chronological_folds * test_series_per_cell
    )
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        series_id = str(row["series_id"])
        by_series.setdefault(series_id, []).append(dict(row))
    if len(by_series) != required_series:
        raise ValueError("series support is incomplete")

    expected_cells = {
        _series_fold_id(output_type, stratum_id)
        for output_type, stratum_id in OUTPUT_STRATA
    }

    for series_id, members in by_series.items():
        members.sort(key=lambda row: (row["issued"], row["row_id"]))
        if len(members) != FOUNDATION_MAPS_PER_SERIES:
            raise ValueError("every series must contain exactly two maps")
        if members[0]["issued"] >= members[1]["issued"]:
            raise ValueError("series maps must be strictly ordered")
        if members[0]["resolved"] >= members[1]["issued"]:
            raise ValueError("series maps must be non-overlapping")
        if any(not (row["issued"] < row["event"] < row["resolved"]) for row in members):
            raise ValueError("row chronology is invalid")
        if len({(row["output_type"], row["stratum_id"], row["cohort_id"]) for row in members}) != 1:
            raise ValueError("series mixes registered cells")
        if len({row["fixture_label"] for row in members}) != 1:
            raise ValueError("series maps must share one fixture label")
        if len({row["latent_truth"] for row in members}) != 1:
            raise ValueError("series maps must share one latent truth")
        if len({
            row["candidate_inputs"]["generator_regime"] for row in members
        }) != 1:
            raise ValueError("series maps must share one generator regime")
        if len({canonical_sha256(row["fixture_label_dgp"]) for row in members}) != 1:
            raise ValueError("series maps must share one fixture-label DGP record")
        if len({canonical_sha256(row["volume_inputs"]) for row in members}) != 1:
            raise ValueError("series maps must share volume inputs")
        for row in members:
            if row["cohort_id"] not in {"initial_train", "test_fold_0", "test_fold_1", "test_fold_2"}:
                raise ValueError("series cohort is unregistered")

    ordered_series = sorted(
        by_series,
        key=lambda sid: (min(row["issued"] for row in by_series[sid]), sid),
    )
    for index in range(1, len(ordered_series)):
        previous = by_series[ordered_series[index - 1]]
        current = by_series[ordered_series[index]]
        if max(row["resolved"] for row in previous) >= min(
            row["issued"] for row in current
        ):
            raise ValueError("series intervals overlap")

    by_cell = {item: [] for item in expected_cells}
    for sid in ordered_series:
        members = by_series[sid]
        row = members[0]
        key = _series_fold_id(row["output_type"], row["stratum_id"])
        if key not in by_cell:
            raise ValueError("unregistered cell detected in plan")
        by_cell[key].append(sid)

    for key, members in by_cell.items():
        if len(members) != initial_series_per_cell + chronological_folds * test_series_per_cell:
            raise ValueError("cell support is incomplete")
        for cohort_id in ("initial_train", "test_fold_0", "test_fold_1", "test_fold_2"):
            cohort_members = [
                sid for sid in members if by_series[sid][0]["cohort_id"] == cohort_id
            ]
            if len(cohort_members) != test_series_per_cell:
                raise ValueError("cell cohort support mismatch")
            for regime in REGIMES:
                regime_series = [
                    sid
                    for sid in cohort_members
                    if by_series[sid][0]["candidate_inputs"]["generator_regime"] == regime
                ]
                if len(regime_series) != 8:
                    raise ValueError("cell regime support mismatch")
                series_labels = [
                    by_series[sid][0]["fixture_label"] for sid in regime_series
                ]
                if series_labels.count(0) != 4 or series_labels.count(1) != 4:
                    raise ValueError("cell cohort regime requires exact 4/4 fixture labels")
                stratified = sorted(
                    (
                        by_series[sid][0]["fixture_label_dgp"]["support_stratum"],
                        by_series[sid][0]["latent_truth"],
                        by_series[sid][0]["fixture_label_dgp"]["probability"],
                    )
                    for sid in regime_series
                )
                if [item[0] for item in stratified] != list(range(8)):
                    raise ValueError("fixture-label support strata must be exact 0..7")
                if [item[2] for item in stratified] != [0.0] * 4 + [1.0] * 4:
                    raise ValueError("fixture-label probability is not registered stratum step")

    all_rows = {row["row_id"]: row for row in rows}
    all_test_series: set[str] = set()

    initial_count = len(OUTPUT_STRATA) * initial_series_per_cell
    test_count = len(OUTPUT_STRATA) * test_series_per_cell
    folds: list[dict[str, Any]] = []
    for fold_index in range(chronological_folds):
        train_end = initial_count + fold_index * test_count
        test_end = train_end + test_count
        train_series = ordered_series[:train_end]
        test_series = ordered_series[train_end:test_end]
        if len(train_series) + len(test_series) != len(OUTPUT_STRATA) * (
            initial_series_per_cell + fold_index * test_series_per_cell + test_series_per_cell
        ):
            raise ValueError("fold window is malformed")
        if all_test_series.intersection(test_series):
            raise ValueError("test series reused")
        all_test_series.update(test_series)

        train_rows = [row_id for sid in train_series for row_id in [r["row_id"] for r in by_series[sid]]]
        test_rows = [row_id for sid in test_series for row_id in [r["row_id"] for r in by_series[sid]]]
        train_end_resolved = max(all_rows[row_id]["resolved"] for row_id in train_rows)
        test_start_issued = min(all_rows[row_id]["issued"] for row_id in test_rows)
        if not train_end_resolved < test_start_issued:
            raise ValueError("train resolution crosses test issuance")

        support: dict[str, dict[str, Any]] = {}
        for output_type, stratum_id in OUTPUT_STRATA:
            fold_cell_series = [
                sid
                for sid in test_series
                if (
                    by_series[sid][0]["output_type"] == output_type
                    and by_series[sid][0]["stratum_id"] == stratum_id
                )
            ]
            if len(fold_cell_series) != test_series_per_cell:
                raise ValueError("per-cell fold test series support mismatch")
            labels = [
                all_rows[row_id]["fixture_label"]
                for row_id in test_rows
                if all_rows[row_id]["output_type"] == output_type
                and all_rows[row_id]["stratum_id"] == stratum_id
            ]
            if len(labels) != 2 * test_series_per_cell:
                raise ValueError("per-cell fold fixture-label support mismatch")
            if set(labels) != {0, 1}:
                raise ValueError("per-cell fold fixture labels must contain both classes")

            for regime in REGIMES:
                regime_rows = [
                    all_rows[row_id]
                    for row_id in test_rows
                    if all_rows[row_id]["output_type"] == output_type
                        and all_rows[row_id]["stratum_id"] == stratum_id
                        and all_rows[row_id]["candidate_inputs"]["generator_regime"] == regime
                ]
                if not regime_rows:
                    raise ValueError("regime series are missing")
                regime_series_labels = [
                    by_series[sid][0]["fixture_label"]
                    for sid in fold_cell_series
                    if by_series[sid][0]["candidate_inputs"]["generator_regime"] == regime
                ]
                if len(regime_rows) < 2 * 8:
                    raise ValueError("regime support is insufficient")
                if (
                    regime_series_labels.count(0) != 4
                    or regime_series_labels.count(1) != 4
                ):
                    raise ValueError("regime series fixture labels must have exact 4/4 support")

            support[output_type] = {
                "raw_series": len(fold_cell_series),
                "effective_series": len(fold_cell_series),
                "rows": len(labels),
            }

        folds.append(
            {
                "fold_id": f"r20_fold_{fold_index}",
                "train_series_ids": train_series,
                "test_series_ids": test_series,
                "train_row_ids": train_rows,
                "test_row_ids": test_rows,
                "train_end_resolved": train_end_resolved,
                "test_start_issued": test_start_issued,
                "test_support_by_output": support,
            }
        )

    # No silent rows: every row is placed exactly once.
    if len(all_test_series) != test_count * chronological_folds:
        raise ValueError("test series pool does not match configured support")
    if folds[-1]["test_series_ids"][-1] != ordered_series[-1]:
        raise ValueError("prequential plan has a silent tail")

    return {
        "chronological_folds": chronological_folds,
        "initial_series_per_cell": initial_series_per_cell,
        "test_series_per_cell": test_series_per_cell,
        "ordered_series_ids": ordered_series,
        "ordered_series_count": len(ordered_series),
        "folds": folds,
        "volume_target_nonseparability": _volume_target_nonseparability(rows),
    }


def build_r20_benchmark(
    *,
    seed: int = FAMILY_DEFAULT.simulation_seed,
    chronological_folds: int = FOUNDATION_FOLDS,
) -> dict[str, Any]:
    if seed != FAMILY_DEFAULT.simulation_seed:
        raise ValueError("benchmark seed is fixed by contract")
    rows = _build_series_layout()
    plan = build_prequential_plan(
        rows,
        chronological_folds=chronological_folds,
    )
    readiness = [
        _volume_readiness_for_fold(
            train_rows=[
                row
                for row in rows
                if row["row_id"] in plan_fold["train_row_ids"]
            ],
            test_rows=[
                row
                for row in rows
                if row["row_id"] in plan_fold["test_row_ids"]
            ],
            fold_id=plan_fold["fold_id"],
        )
        for plan_fold in plan["folds"]
    ]
    return {
        "seed": seed,
        "chronological_folds": chronological_folds,
        "rows": rows,
        "prequential_plan": plan,
        "volume_readiness": readiness,
        "rows_row_id_sha256": canonical_sha256([row["row_id"] for row in rows]),
        "prequential_plan_sha256": canonical_sha256(plan),
    }


def verify_prequential_plan(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_prequential_plan(rows)
    if dict(plan) != expected:
        raise ValueError("submitted prequential plan is not the exact rebuilt plan")
    return expected


__all__ = [
    "FAMILY_DEFAULT",
    "FOUNDATION_DRAWS",
    "FOUNDATION_FOLDS",
    "FOUNDATION_MAPS_PER_SERIES",
    "FOUNDATION_SERIES_PER_CELL",
    "FOUNDATION_SERIES_TOTAL",
    "INITIAL_TRAIN_SERIES_PER_CELL",
    "OUTPUT_STRATA",
    "REGIMES",
    "SOURCE_CONTEXT_PATTERNS",
    "TEST_SERIES_PER_CELL_PER_FOLD",
    "build_prequential_plan",
    "build_r20_benchmark",
    "verify_prequential_plan",
]
