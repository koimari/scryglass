"""Frozen observation-only inference adapter for the synthetic R-20 foundation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution

from .checks import ValidationFailure
from .r20_foundation_algorithms import (
    replay_foundation_method,
    replay_registered_precision_batch,
)
from .types import canonical_sha256


INFERENCE_ADAPTER_ID = "scryglass:b2:r20-beta-binomial-inference:v1"
INFERENCE_SEED = 20260729
POSTERIOR_DRAWS = 256
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0
REFERENCE_MODES = ("prior_reference", "posterior_equal", "compressed_reference")


def _strict_int(value: Any, path: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise ValueError(f"{path} must be int")
    return value


def _strict_float(value: Any, path: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def infer_beta_binomial(
    *,
    observation: dict[str, Any],
    inference_seed: int,
    draw_count: int,
    prior_alpha: float = PRIOR_ALPHA,
    prior_beta: float = PRIOR_BETA,
    reference_mode: str = "prior_reference",
) -> dict[str, Any]:
    """Infer from sufficient statistics only; hidden truth is never accepted."""

    if set(observation) != {"successes", "trials"}:
        raise ValueError("observation must contain exact successes/trials")

    successes = _strict_int(observation["successes"], "observation.successes")
    trials = _strict_int(observation["trials"], "observation.trials")
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("invalid binomial sufficient statistics")

    inference_seed = _strict_int(inference_seed, "inference_seed")
    if inference_seed < 0:
        raise ValueError("inference_seed must be nonnegative")

    draw_count = _strict_int(draw_count, "draw_count")
    if draw_count < 128:
        raise ValueError("draw_count must be >=128")

    prior_alpha = _strict_float(prior_alpha, "prior_alpha")
    prior_beta = _strict_float(prior_beta, "prior_beta")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior parameters must be strictly positive")

    if reference_mode not in REFERENCE_MODES:
        raise ValueError("reference_mode is not registered")

    rng = np.random.default_rng(inference_seed)
    prior_draws = rng.beta(prior_alpha, prior_beta, size=draw_count)

    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + trials - successes
    posterior_draws = rng.beta(posterior_alpha, posterior_beta, size=draw_count)

    if reference_mode == "prior_reference":
        reference_draws = rng.beta(prior_alpha, prior_beta, size=draw_count)
    elif reference_mode == "posterior_equal":
        reference_draws = posterior_draws.copy()
    else:
        posterior_mean = float(np.mean(posterior_draws))
        reference_draws = np.clip(
            posterior_mean + 0.5 * (posterior_draws - posterior_mean),
            1e-15,
            1 - 1e-15,
        )

    payload = {
        "adapter_id": INFERENCE_ADAPTER_ID,
        "inference_seed": inference_seed,
        "draw_count": draw_count,
        "observation": {"successes": successes, "trials": trials},
        "prior": {"alpha": float(prior_alpha), "beta": float(prior_beta)},
        "reference_mode": reference_mode,
        "posterior_parameters": {
            "alpha": float(posterior_alpha),
            "beta": float(posterior_beta),
        },
        "prior_draws": prior_draws.tolist(),
        "posterior_draws": posterior_draws.tolist(),
        "registered_reference_draws": reference_draws.tolist(),
    }
    payload["inference_output_sha256"] = canonical_sha256(payload)
    return payload


def _central_width(values: np.ndarray, mass: float = 0.95) -> float:
    tail = (1.0 - mass) / 2.0
    return float(np.quantile(values, 1.0 - tail) - np.quantile(values, tail))


def _beta_central_width(alpha: float, beta: float, mass: float = 0.95) -> float:
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("beta parameters must be finite")
    if alpha <= 0 or beta <= 0:
        raise ValueError("beta parameters must be strictly positive")
    if not 0 < mass < 1:
        raise ValueError("mass must lie in (0,1)")
    tail = (1.0 - mass) / 2.0
    return float(
        beta_distribution.ppf(1.0 - tail, alpha, beta)
        - beta_distribution.ppf(tail, alpha, beta)
    )


def _beta_mad(alpha: float, beta: float) -> float:
    median = float(beta_distribution.ppf(0.5, alpha, beta))
    maximum = max(median, 1.0 - median)

    def enclosed(radius: float) -> float:
        lower = max(0.0, median - radius)
        upper = min(1.0, median + radius)
        return float(
            beta_distribution.cdf(upper, alpha, beta)
            - beta_distribution.cdf(lower, alpha, beta)
            - 0.5
        )

    return float(brentq(enclosed, 0.0, maximum))


def _row_quantile_width(draws: np.ndarray, mass: float = 0.95) -> np.ndarray:
    tail = (1.0 - mass) / 2.0
    return np.quantile(draws, 1.0 - tail, axis=1) - np.quantile(
        draws,
        tail,
        axis=1,
    )


def _row_mad(draws: np.ndarray) -> np.ndarray:
    medians = np.median(draws, axis=1)
    return np.median(np.abs(draws - medians[:, None]), axis=1)


def _precision_boundary_parity_fixture() -> dict[str, Any]:
    """Exercise the exact registered -1e-12 precision eligibility boundary."""

    boundaries = {"minimum_draws": POSTERIOR_DRAWS, "central_mass": 0.95}
    requested = (
        ("just_below_tolerance", -2e-12, False),
        ("inside_tolerance", -5e-13, True),
        ("exact_zero", 0.0, True),
        ("positive", 1e-6, True),
    )
    base = np.linspace(0.25, 0.75, POSTERIOR_DRAWS, dtype=float)
    posterior = np.tile(base, (len(requested), 1))
    references = np.vstack(
        [
            0.5 + (base - 0.5) / (1.0 - target)
            for _, target, _ in requested
        ],
    )
    methods: list[dict[str, Any]] = []
    for method_id in (
        "central_interval_contraction_v2",
        "robust_mad_contraction_v1",
    ):
        replay = replay_registered_precision_batch(
            method_id=method_id,
            posterior_draws=posterior,
            reference_draws=references,
            boundaries=boundaries,
        )
        cases: list[dict[str, Any]] = []
        for index, (case_id, requested_target, expected_accept) in enumerate(requested):
            actual_raw = float(replay["raw_contraction"][index])
            actual_accept = bool(replay["accepted"][index])
            try:
                scalar = replay_foundation_method(
                    method_id=method_id,
                    dependencies={
                        "posterior_draws": posterior[index].tolist(),
                        "registered_reference_draws": references[index].tolist(),
                    },
                    boundaries=boundaries,
                )
            except ValidationFailure:
                scalar_status = "reject"
                scalar_value = None
            else:
                scalar_status = "accept"
                scalar_value = float(scalar["value"])
            batch_status = "accept" if actual_accept else "reject"
            batch_value = (
                float(replay["value"][index]) if actual_accept else None
            )
            position_ok = (
                actual_raw < -1e-12
                if case_id == "just_below_tolerance"
                else -1e-12 <= actual_raw < 0.0
                if case_id == "inside_tolerance"
                else actual_raw == 0.0
                if case_id == "exact_zero"
                else actual_raw > 0.0
            )
            cases.append(
                {
                    "case_id": case_id,
                    "requested_raw_contraction": requested_target,
                    "actual_raw_contraction": actual_raw,
                    "expected_status": "accept" if expected_accept else "reject",
                    "batch_status": batch_status,
                    "batch_value": batch_value,
                    "scalar_status": scalar_status,
                    "scalar_value": scalar_value,
                    "position_ok": bool(position_ok),
                    "expected_parity_ok": actual_accept is expected_accept,
                    "scalar_batch_parity_ok": (
                        scalar_status == batch_status
                        and (
                            scalar_value == batch_value
                            if scalar_value is not None
                            else batch_value is None
                        )
                    ),
                },
            )
        methods.append(
            {
                "method_id": method_id,
                "boundary_sha256": replay["boundary_sha256"],
                "cases": cases,
                "passes": all(
                    case["position_ok"]
                    and case["expected_parity_ok"]
                    and case["scalar_batch_parity_ok"]
                    for case in cases
                ),
            },
        )
    return {
        "fixture_id": "r20-precision-exact-boundary-parity-v1",
        "eligibility_threshold": -1e-12,
        "draw_construction": (
            "bounded linear base draws with centered scale-only references"
        ),
        "base_draw_min": float(base.min()),
        "base_draw_max": float(base.max()),
        "boundaries": boundaries,
        "boundary_sha256": canonical_sha256(boundaries),
        "methods": methods,
        "passes": all(method["passes"] for method in methods),
    }


def monte_carlo_width_design(
    *,
    seed: int = 20260730,
    draw_count: int = POSTERIOR_DRAWS,
    replications: int = 2_000,
    target_absolute_width_error: float = 0.05,
    registered_observation_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replayable MC-error and neutral-rank checks for the registered grid."""

    _strict_int(seed, "seed")
    draw_count = _strict_int(draw_count, "draw_count")
    replications = _strict_int(replications, "replications")
    if draw_count != POSTERIOR_DRAWS:
        raise ValueError("draw_count must equal registered candidate boundary")
    if replications < 2_000:
        raise ValueError("Monte Carlo replications support mismatch")

    target_absolute_width_error = _strict_float(
        target_absolute_width_error, "target_absolute_width_error"
    )
    if not 0.0 < target_absolute_width_error < 1.0:
        raise ValueError("target tolerance must be in (0,1)")

    regimes = (
        "symmetric",
        "low_skew",
        "high_skew",
        "boundary_heavy",
        "volume_quadratic_null",
    )
    trial_grid = (12, 24, 36, 48)
    if registered_observation_cells is None:
        registered_observation_cells = []
    if not isinstance(registered_observation_cells, list):
        raise ValueError("registered observation cells must be a list")
    normalized_observed: list[dict[str, Any]] = []
    for cell in registered_observation_cells:
        if not isinstance(cell, dict) or set(cell) != {
            "regime", "successes", "trials",
        }:
            raise ValueError("registered observation cell schema mismatch")
        if cell["regime"] not in regimes:
            raise ValueError("registered observation regime mismatch")
        successes = _strict_int(cell["successes"], "registered successes")
        trials = _strict_int(cell["trials"], "registered trials")
        if trials not in trial_grid or not 0 <= successes <= trials:
            raise ValueError("registered observation cell values invalid")
        normalized_observed.append(
            {"regime": cell["regime"], "successes": successes, "trials": trials},
        )
    normalized_observed.sort(
        key=lambda cell: (cell["regime"], cell["trials"], cell["successes"]),
    )
    if len({
        (cell["regime"], cell["successes"], cell["trials"])
        for cell in normalized_observed
    }) != len(normalized_observed):
        raise ValueError("registered observation cells are duplicated")
    regimes_by_conditional_cell: dict[tuple[int, int], set[str]] = {}
    for cell in normalized_observed:
        regimes_by_conditional_cell.setdefault(
            (cell["successes"], cell["trials"]),
            set(),
        ).add(cell["regime"])
    if not regimes_by_conditional_cell:
        for trials in trial_grid:
            for successes in (0, trials // 2, trials):
                regimes_by_conditional_cell[(successes, trials)] = set(regimes)
    cases = [
        {
            "case_id": f"conditional:s{successes}:n{trials}",
            "trials": trials,
            "successes": successes,
            "mapped_regimes": sorted(mapped_regimes),
        }
        for (successes, trials), mapped_regimes in sorted(
            regimes_by_conditional_cell.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]
    observed_to_conditional = [
        {
            **cell,
            "conditional_case_id": (
                f"conditional:s{cell['successes']}:n{cell['trials']}"
            ),
        }
        for cell in normalized_observed
    ]

    all_posterior_width_errors: list[float] = []
    all_reference_width_errors: list[float] = []
    all_central_contraction_errors: list[float] = []
    all_mad_contraction_errors: list[float] = []
    case_results: list[dict[str, Any]] = []
    rank_samples: dict[str, dict[str, list[np.ndarray]]] = {
        mode: {
            "posterior_mean": [],
            "posterior_median": [],
            "central_width": [],
            "mad": [],
        }
        for mode in REFERENCE_MODES
    }
    rank_oracles: dict[str, dict[str, list[float]]] = {
        mode: {method: [] for method in rank_samples[mode]}
        for mode in REFERENCE_MODES
    }

    for case_index, case in enumerate(cases):
        posterior_alpha = PRIOR_ALPHA + case["successes"]
        posterior_beta = PRIOR_BETA + case["trials"] - case["successes"]
        posterior_mean_oracle = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_median_oracle = float(
            beta_distribution.ppf(0.5, posterior_alpha, posterior_beta),
        )
        prior_mean_oracle = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)
        prior_median_oracle = float(
            beta_distribution.ppf(0.5, PRIOR_ALPHA, PRIOR_BETA),
        )
        prior_sd_oracle = float(
            math.sqrt(
                PRIOR_ALPHA
                * PRIOR_BETA
                / (
                    (PRIOR_ALPHA + PRIOR_BETA) ** 2
                    * (PRIOR_ALPHA + PRIOR_BETA + 1.0)
                )
            )
        )
        posterior_width_oracle = _beta_central_width(
            posterior_alpha,
            posterior_beta,
        )
        posterior_mad_oracle = _beta_mad(posterior_alpha, posterior_beta)
        case_report: dict[str, Any] = {
            "case_id": case["case_id"],
            "mapped_regimes": case["mapped_regimes"],
            "parameters": {
                "trials": case["trials"],
                "successes": case["successes"],
                "posterior_alpha": posterior_alpha,
                "posterior_beta": posterior_beta,
            },
            "reference_mode_reports": {},
        }
        rng = np.random.default_rng(seed + case_index * 101)
        posterior = rng.beta(
            posterior_alpha,
            posterior_beta,
            size=(replications, draw_count),
        )
        prior = rng.beta(
            PRIOR_ALPHA,
            PRIOR_BETA,
            size=(replications, draw_count),
        )
        prior_reference = rng.beta(
            PRIOR_ALPHA,
            PRIOR_BETA,
            size=(replications, draw_count),
        )
        for mode in REFERENCE_MODES:
            if mode == "prior_reference":
                reference_oracle_width = _beta_central_width(PRIOR_ALPHA, PRIOR_BETA)
                reference_oracle_mad = _beta_mad(PRIOR_ALPHA, PRIOR_BETA)
            elif mode == "posterior_equal":
                reference_oracle_width = posterior_width_oracle
                reference_oracle_mad = posterior_mad_oracle
            else:
                reference_oracle_width = 0.5 * posterior_width_oracle
                reference_oracle_mad = 0.5 * posterior_mad_oracle
            true_central_contraction = (
                1.0 - posterior_width_oracle / reference_oracle_width
            )
            true_mad_contraction = (
                1.0 - posterior_mad_oracle / reference_oracle_mad
            )
            if mode == "prior_reference":
                reference = prior_reference
            elif mode == "posterior_equal":
                reference = posterior.copy()
            else:
                posterior_means = np.mean(posterior, axis=1)
                reference = np.clip(
                    posterior_means[:, None]
                    + 0.5 * (posterior - posterior_means[:, None]),
                    1e-15,
                    1 - 1e-15,
                )
            posterior_width = _row_quantile_width(posterior)
            reference_width = _row_quantile_width(reference)
            posterior_mad = _row_mad(posterior)
            reference_mad = _row_mad(reference)
            sampled_central_contraction = 1.0 - posterior_width / reference_width
            sampled_mad_contraction = 1.0 - posterior_mad / reference_mad
            posterior_width_errors = np.abs(
                posterior_width - posterior_width_oracle,
            )
            reference_width_errors = np.abs(reference_width - reference_oracle_width)
            central_contraction_errors = np.abs(
                sampled_central_contraction - true_central_contraction,
            )
            mad_contraction_errors = np.abs(
                sampled_mad_contraction - true_mad_contraction,
            )
            all_posterior_width_errors.extend(posterior_width_errors.tolist())
            all_reference_width_errors.extend(reference_width_errors.tolist())
            all_central_contraction_errors.extend(
                central_contraction_errors.tolist(),
            )
            all_mad_contraction_errors.extend(mad_contraction_errors.tolist())
            precision_boundaries = {
                "minimum_draws": POSTERIOR_DRAWS,
                "central_mass": 0.95,
            }
            central_replay = replay_registered_precision_batch(
                method_id="central_interval_contraction_v2",
                posterior_draws=posterior,
                reference_draws=reference,
                boundaries=precision_boundaries,
            )
            mad_replay = replay_registered_precision_batch(
                method_id="robust_mad_contraction_v1",
                posterior_draws=posterior,
                reference_draws=reference,
                boundaries=precision_boundaries,
            )
            if not np.allclose(
                central_replay["raw_contraction"],
                sampled_central_contraction,
                rtol=0.0,
                atol=1e-15,
            ) or not np.allclose(
                mad_replay["raw_contraction"],
                sampled_mad_contraction,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError("registered precision replay diverged from MC summary")

            def candidate_agreement(
                observed_accepted: np.ndarray,
                oracle_contraction: float,
                method_id: str,
            ) -> dict[str, Any]:
                expected_accepted = oracle_contraction >= -1e-12
                correct = int(np.sum(observed_accepted == expected_accepted))
                accuracy = correct / replications
                z = 1.959963984540054
                denominator = 1.0 + z * z / replications
                center = (
                    accuracy + z * z / (2.0 * replications)
                ) / denominator
                radius = (
                    z
                    * math.sqrt(
                        accuracy * (1.0 - accuracy) / replications
                        + z * z / (4.0 * replications * replications)
                    )
                    / denominator
                )
                return {
                    "method_id": method_id,
                    "boundary_sha256": canonical_sha256(precision_boundaries),
                    "expected_status": (
                        "accept" if expected_accepted else "reject"
                    ),
                    "observed_accept_count": int(np.sum(observed_accepted)),
                    "observed_reject_count": int(np.sum(~observed_accepted)),
                    "correct_count": correct,
                    "incorrect_count": replications - correct,
                    "accuracy": accuracy,
                    "accuracy_wilson_lower_95": max(0.0, center - radius),
                    "mcse": math.sqrt(
                        accuracy * (1.0 - accuracy) / replications,
                    ),
                }

            central_candidate = candidate_agreement(
                central_replay["accepted"],
                true_central_contraction,
                "central_interval_contraction_v2",
            )
            mad_candidate = candidate_agreement(
                mad_replay["accepted"],
                true_mad_contraction,
                "robust_mad_contraction_v1",
            )
            mean_values = np.abs(
                np.mean(posterior, axis=1) - np.mean(prior, axis=1),
            ) / np.std(prior, axis=1, ddof=1)
            median_values = np.abs(
                np.median(posterior, axis=1) - np.median(prior, axis=1),
            ) / np.std(prior, axis=1, ddof=1)
            rank_samples[mode]["posterior_mean"].append(mean_values)
            rank_samples[mode]["posterior_median"].append(median_values)
            rank_samples[mode]["central_width"].append(sampled_central_contraction)
            rank_samples[mode]["mad"].append(sampled_mad_contraction)
            rank_oracles[mode]["posterior_mean"].append(
                abs(posterior_mean_oracle - prior_mean_oracle) / prior_sd_oracle,
            )
            rank_oracles[mode]["posterior_median"].append(
                abs(posterior_median_oracle - prior_median_oracle) / prior_sd_oracle,
            )
            rank_oracles[mode]["central_width"].append(true_central_contraction)
            rank_oracles[mode]["mad"].append(true_mad_contraction)
            case_report["reference_mode_reports"][mode] = {
                "oracle": {
                    "posterior_width": posterior_width_oracle,
                    "reference_width": reference_oracle_width,
                    "central_contraction": true_central_contraction,
                    "posterior_mad": posterior_mad_oracle,
                    "reference_mad": reference_oracle_mad,
                    "mad_contraction": true_mad_contraction,
                },
                "posterior_width_error": {
                    "p90": float(np.quantile(posterior_width_errors, 0.90)),
                    "max": float(np.max(posterior_width_errors)),
                },
                "reference_width_error": {
                    "p90": float(np.quantile(reference_width_errors, 0.90)),
                    "max": float(np.max(reference_width_errors)),
                },
                "central_contraction_error": {
                    "p90": float(np.quantile(central_contraction_errors, 0.90)),
                    "max": float(np.max(central_contraction_errors)),
                },
                "mad_contraction_error": {
                    "p90": float(np.quantile(mad_contraction_errors, 0.90)),
                    "max": float(np.max(mad_contraction_errors)),
                },
                "actual_candidate_replay": {
                    "central_interval_contraction_v2": central_candidate,
                    "robust_mad_contraction_v1": mad_candidate,
                },
            }
        case_results.append(case_report)

    rank_evidence: list[dict[str, Any]] = []
    rank_tolerance = 0.02
    for mode in REFERENCE_MODES:
        for method, case_samples in rank_samples[mode].items():
            matrix = np.vstack(case_samples)
            oracle = np.asarray(rank_oracles[mode][method], dtype=float)
            for regime in regimes:
                indices = [
                    index for index, case in enumerate(cases)
                    if regime in case["mapped_regimes"]
                ]
                pairs = [
                    (left, right)
                    for position, left in enumerate(indices)
                    for right in indices[position + 1 :]
                    if abs(oracle[left] - oracle[right]) > rank_tolerance
                ]
                if not pairs:
                    agreement = 1.0
                    comparisons = 0
                    agreed = 0
                    clustered_mcse = 0.0
                    clustered_lower_95 = 1.0
                else:
                    pair_agreements: list[np.ndarray] = []
                    comparisons = len(pairs) * replications
                    for left, right in pairs:
                        oracle_sign = math.copysign(1.0, oracle[left] - oracle[right])
                        sample_difference = matrix[left] - matrix[right]
                        pair_agreements.append(
                            sample_difference * oracle_sign > 0,
                        )
                    agreement_matrix = np.vstack(pair_agreements)
                    replication_rates = np.mean(agreement_matrix, axis=0)
                    agreement = float(np.mean(replication_rates))
                    agreed = int(np.sum(agreement_matrix))
                    clustered_mcse = float(
                        np.std(replication_rates, ddof=1)
                        / math.sqrt(replications),
                    )
                    clustered_lower_95 = max(
                        0.0,
                        agreement - 1.959963984540054 * clustered_mcse,
                    )
                rank_evidence.append(
                    {
                        "method_summary": method,
                        "reference_mode": mode,
                        "regime": regime,
                        "case_count": len(indices),
                        "oracle_tie_tolerance": rank_tolerance,
                        "informative_pair_count": len(pairs),
                        "comparison_count": comparisons,
                        "agreement_count": agreed,
                        "pairwise_rank_agreement": agreement,
                        "replication_clustered_mcse": clustered_mcse,
                        "replication_clustered_lower_95": clustered_lower_95,
                    },
                )

    neutrality_seeds = [seed, seed + 1_000_003, seed + 2_000_003]
    neutrality: list[dict[str, Any]] = []
    for neutrality_seed in neutrality_seeds:
        seed_rng = np.random.default_rng(neutrality_seed)
        mirrored: dict[str, dict[str, float]] = {}
        for regime, alpha, beta in (
            ("low_skew", 2.0, 7.0),
            ("high_skew", 7.0, 2.0),
        ):
            draws = seed_rng.beta(alpha, beta, size=(replications, draw_count))
            mirrored[regime] = {
                "mean": float(np.mean(np.mean(draws, axis=1))),
                "median": float(np.mean(np.median(draws, axis=1))),
                "mad": float(np.mean(_row_mad(draws))),
            }
        neutrality.append(
            {
                "seed": neutrality_seed,
                "low_high_mean_complement_error": abs(
                    mirrored["low_skew"]["mean"]
                    + mirrored["high_skew"]["mean"]
                    - 1.0
                ),
                "low_high_median_complement_error": abs(
                    mirrored["low_skew"]["median"]
                    + mirrored["high_skew"]["median"]
                    - 1.0
                ),
                "low_high_mad_symmetry_error": abs(
                    mirrored["low_skew"]["mad"]
                    - mirrored["high_skew"]["mad"]
                ),
            },
        )

    achieved_p90_posterior_width_error = float(
        np.quantile(all_posterior_width_errors, 0.90),
    )
    achieved_max_posterior_width_error = float(max(all_posterior_width_errors))
    achieved_p90_reference_width_error = float(
        np.quantile(all_reference_width_errors, 0.90),
    )
    achieved_max_reference_width_error = float(max(all_reference_width_errors))
    achieved_p90_central_contraction_error = float(
        np.quantile(all_central_contraction_errors, 0.90),
    )
    achieved_p90_mad_contraction_error = float(
        np.quantile(all_mad_contraction_errors, 0.90),
    )
    candidate_reports = [
        method_report
        for case in case_results
        for report in case["reference_mode_reports"].values()
        for method_report in report["actual_candidate_replay"].values()
    ]
    max_candidate_agreement_mcse = max(
        report["mcse"] for report in candidate_reports
    )
    minimum_candidate_agreement_accuracy = min(
        report["accuracy"] for report in candidate_reports
    )
    minimum_candidate_agreement_wilson = min(
        report["accuracy_wilson_lower_95"] for report in candidate_reports
    )
    informative_ranks = [
        item for item in rank_evidence if item["informative_pair_count"] > 0
    ]
    rank_na_cell_count = len(rank_evidence) - len(informative_ranks)
    minimum_rank_agreement = min(
        (item["pairwise_rank_agreement"] for item in informative_ranks),
        default=0.0,
    )
    minimum_rank_clustered_lower_95 = min(
        (
            item["replication_clustered_lower_95"]
            for item in informative_ranks
        ),
        default=0.0,
    )
    neutrality_pass = all(
        item["low_high_mean_complement_error"] <= 0.02
        and item["low_high_median_complement_error"] <= 0.02
        and item["low_high_mad_symmetry_error"] <= 0.02
        for item in neutrality
    )
    boundary_parity = _precision_boundary_parity_fixture()
    candidate_regime_matrix: list[dict[str, Any]] = []
    for item in rank_evidence:
        actual_candidate_status: dict[str, Any]
        if item["method_summary"] in {"central_width", "mad"}:
            candidate_id = (
                "central_interval_contraction_v2"
                if item["method_summary"] == "central_width"
                else "robust_mad_contraction_v1"
            )
            matching = [
                case["reference_mode_reports"][item["reference_mode"]][
                    "actual_candidate_replay"
                ][candidate_id]
                for case in case_results
                if item["regime"] in case["mapped_regimes"]
            ]
            cell_wilson = min(
                report["accuracy_wilson_lower_95"] for report in matching
            )
            actual_candidate_status = {
                "applicable": True,
                "candidate_id": candidate_id,
                "expected_statuses": sorted({
                    report["expected_status"] for report in matching
                }),
                "observed_accept_count": sum(
                    report["observed_accept_count"] for report in matching
                ),
                "observed_reject_count": sum(
                    report["observed_reject_count"] for report in matching
                ),
                "minimum_agreement_wilson_lower_95": cell_wilson,
                "status": (
                    "pass" if cell_wilson >= 0.99 else "reject"
                ),
            }
        else:
            actual_candidate_status = {
                "applicable": False,
                "reason": "no registered precision eligibility rule",
                "status": "not_applicable_non_precision",
            }
        adequacy_status = (
            "pass"
            if item["informative_pair_count"] > 0
            and item["replication_clustered_lower_95"] >= 0.60
            else "not_applicable_exact_oracle_ties"
            if item["informative_pair_count"] == 0
            else "reject"
        )
        candidate_regime_matrix.append(
            {
                "method_summary": item["method_summary"],
                "reference_mode": item["reference_mode"],
                "regime": item["regime"],
                "actual_candidate_status": actual_candidate_status,
                "pairwise_rank_agreement": item["pairwise_rank_agreement"],
                "replication_clustered_rank_lower_95": item[
                    "replication_clustered_lower_95"
                ],
                "informative_pair_count": item["informative_pair_count"],
                "adequacy_status": adequacy_status,
                "cell_status": (
                    "reject"
                    if actual_candidate_status["status"] == "reject"
                    or adequacy_status == "reject"
                    else "pass"
                ),
            },
        )

    return {
        "design_id": "r20-monte-carlo-adequacy-design-v4",
        "seed": seed,
        "draw_count": draw_count,
        "replications": replications,
        "target_absolute_width_error": target_absolute_width_error,
        "target_p90_contraction_error": 0.15,
        "target_max_width_error": 0.20,
        "target_min_candidate_agreement_wilson_lower_95": 0.99,
        "target_min_replication_clustered_rank_lower_95": 0.60,
        "replication_precision_target_max_mcse": 0.012,
        "cases": [item["case_id"] for item in cases],
        "regime_grid": list(regimes),
        "conditional_oracle_semantics": (
            "numerical posterior oracles depend only on registered successes/trials "
            "with the frozen Beta(2,2) prior; regime labels are mapping metadata"
        ),
        "trial_grid": list(trial_grid),
        "reference_modes": list(REFERENCE_MODES),
        "case_results": case_results,
        "registered_benchmark_observation_coverage": {
            "registered_count": len(normalized_observed),
            "unique_conditional_cell_count": len(cases),
            "mapped_count": len(observed_to_conditional),
            "omitted_count": 0,
            "mapping": observed_to_conditional,
            "claim": "complete_unique_conditional_success_trial_coverage",
        },
        "candidate_rank_evidence": rank_evidence,
        "candidate_reference_mode_regime_matrix": candidate_regime_matrix,
        "precision_boundary_parity": boundary_parity,
        "neutrality_multi_seed": neutrality,
        "achieved_p90_absolute_width_error": max(
            achieved_p90_posterior_width_error,
            achieved_p90_reference_width_error,
        ),
        "achieved_width_errors": {
            "posterior_p90": achieved_p90_posterior_width_error,
            "posterior_max": achieved_max_posterior_width_error,
            "reference_p90": achieved_p90_reference_width_error,
            "reference_max": achieved_max_reference_width_error,
        },
        "achieved_p90_true_contraction_error": max(
            achieved_p90_central_contraction_error,
            achieved_p90_mad_contraction_error,
        ),
        "achieved_contraction_errors": {
            "central_p90": achieved_p90_central_contraction_error,
            "mad_p90": achieved_p90_mad_contraction_error,
        },
        "minimum_candidate_agreement_accuracy": (
            minimum_candidate_agreement_accuracy
        ),
        "minimum_candidate_agreement_wilson_lower_95": (
            minimum_candidate_agreement_wilson
        ),
        "minimum_pairwise_rank_agreement": minimum_rank_agreement,
        "minimum_replication_clustered_rank_lower_95": (
            minimum_rank_clustered_lower_95
        ),
        "rank_na_exact_tie_cell_count": rank_na_cell_count,
        "rank_cells_accounted": len(rank_evidence),
        "max_candidate_agreement_mcse": max_candidate_agreement_mcse,
        "neutrality_pass": neutrality_pass,
        "passes": (
            achieved_p90_posterior_width_error <= target_absolute_width_error
            and achieved_p90_reference_width_error <= target_absolute_width_error
            and achieved_max_posterior_width_error <= 0.20
            and achieved_max_reference_width_error <= 0.20
            and achieved_p90_central_contraction_error <= 0.15
            and achieved_p90_mad_contraction_error <= 0.15
            and minimum_candidate_agreement_wilson >= 0.99
            and bool(informative_ranks)
            and minimum_rank_clustered_lower_95 >= 0.60
            and neutrality_pass
            and boundary_parity["passes"]
        ),
    }


__all__ = [
    "INFERENCE_ADAPTER_ID",
    "INFERENCE_SEED",
    "POSTERIOR_DRAWS",
    "PRIOR_ALPHA",
    "PRIOR_BETA",
    "REFERENCE_MODES",
    "infer_beta_binomial",
    "monte_carlo_width_design",
]
