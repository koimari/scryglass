"""Frozen synthetic B3 SBC and dependence-aware coverage authority.

This module is deliberately synthetic-only.  It validates mechanics and never
authorizes real-competition coverage, calibration, reliability, or promotion.
"""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import stat
import statistics
from types import CodeType, MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import gammaincc
from scipy.stats import beta as beta_distribution
from scipy.stats import binom, binomtest, chi2

from .types import CONTRACT_TREE_SHA256


ARTIFACT_ROOT = Path("data/lol/v2/evaluation/b3")
AUTHORITY_LOCATOR = ARTIFACT_ROOT / "coverage-authority.json"
B2_PROCEDURE_LOCATOR = Path("data/lol/v2/evaluation/b2/coverage-procedure.json")
PUBLIC_INTERVAL_WORDING = "95% model range"
RNG_ALGORITHM = "PCG64"
RANK_TIE_POLICY = "count_below_plus_discrete_uniform_zero_through_ties"
OUTPUT_TYPES = (
    "player_rating",
    "team_rating",
    "draft_score",
    "partial_draft_state",
    "tier_list",
)
REGIME_KINDS = (
    "known_latent_rating",
    "interaction_composition",
    "sparse_new_entity",
    "dependence",
)
CONTROL_NAMES = (
    "known_good",
    "biased",
    "underdispersed",
    "overdispersed",
    "centre_ranked_degenerate",
)
ARTIFACT_ROLES = (
    "config",
    "regimes",
    "replications",
    "heldout_rows",
    "heldout_cells",
    "dependence",
    "report",
)
CLAIM_CEILING = (
    "synthetic_sbc_coverage_dependence_mechanics_only",
)
FORBIDDEN_CLAIMS = (
    "model_validity",
    "real_95_percent_coverage",
    "Calibration",
    "Reliability",
    "PASS-B2",
    "C1",
    "promotion",
    "probability_wording",
    "production_coverage",
    "SOTA",
)
AUTHORITY_THREAT_MODEL = MappingProxyType(
    {
        "scope": "process_local_misuse_and_ordinary_forgery_guard",
        "honest_interpreter_required": True,
        "hostile_same_process_security": False,
        "closure_cells_module_globals_and_class_code_are_mutable": True,
        "content_revalidated_on_every_public_use": True,
        "production_authority_requires": (
            "independently_pinned_signature_native_process_or_os_trust_boundary"
        ),
    }
)
_EXPECTED_B2_OBJECT_SHA256 = (
    "e8b57884bd2970dc3d52bc0f5a73cfb1be64b9bd332e3705c4787911757df926"
)


class B3CoverageError(ValueError):
    """Raised when frozen B3 evidence fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _object_hash(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _raw_hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise B3CoverageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                B3CoverageError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B3CoverageError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise B3CoverageError(f"non-canonical JSON: {path}")
    return value, raw


def _safe_file(root: Path, locator: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts:
        raise B3CoverageError(f"unsafe artifact locator: {locator}")
    if stat.S_ISLNK(root.lstat().st_mode):
        raise B3CoverageError("symlink authority root rejected")
    root_resolved = root.resolve(strict=True)
    candidate = root / relative
    component = root
    for part in relative.parts:
        component = component / part
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError as exc:
            raise B3CoverageError(f"missing artifact path component: {locator}") from exc
        if stat.S_ISLNK(mode):
            raise B3CoverageError(f"symlink artifact path rejected: {locator}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise B3CoverageError(f"path escape rejected: {locator}")
    file_stat = resolved.stat()
    if not resolved.is_file() or file_stat.st_nlink != 1:
        raise B3CoverageError(f"non-regular or hard-linked artifact: {locator}")
    return resolved


def _code_payload(code: CodeType) -> dict[str, Any]:
    constants: list[Any] = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            constants.append({"code": _code_payload(value)})
        elif value is None or isinstance(value, (bool, int, float, str, bytes)):
            constants.append(
                {"type": type(value).__name__, "value": value.hex() if isinstance(value, bytes) else value}
            )
        else:
            constants.append({"type": type(value).__name__, "repr": repr(value)})
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_fingerprint(function: Callable[..., Any]) -> str:
    payload = {
        "name": function.__name__,
        "source_sha256": sha256(inspect.getsource(function).encode("utf-8")).hexdigest(),
        "defaults": repr(function.__defaults__),
        "kwdefaults": repr(function.__kwdefaults__),
    }
    return _object_hash(payload)


def _regime_universe() -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    kind_parameters = {
        "known_latent_rating": (2.0, 2.0, 18, 0.0),
        "interaction_composition": (1.7, 2.3, 24, 0.0),
        "sparse_new_entity": (1.2, 1.2, 5, 0.0),
        "dependence": (2.5, 2.5, 16, 0.0),
    }
    for output_index, output in enumerate(OUTPUT_TYPES):
        for kind_index, kind in enumerate(REGIME_KINDS):
            alpha, beta, observations, shared_scale = kind_parameters[kind]
            regimes.append(
                {
                    "regime_id": f"{output}:{kind}",
                    "output_type": output,
                    "frozen_stratum": f"{output}:registered",
                    "regime_kind": kind,
                    "prior_alpha": alpha + output_index * 0.03,
                    "prior_beta": beta + kind_index * 0.02,
                    "observation_count": observations,
                    "shared_effect_scale": shared_scale,
                }
            )
    return regimes


def _config() -> dict[str, Any]:
    return {
        "artifact_id": "scryglass:b3:coverage-config:v1",
        "schema_version": 1,
        "synthetic_only": True,
        "production_eligible": False,
        "rng_algorithm": RNG_ALGORITHM,
        "seed_schedule": "sha256_domain_separated_master_regime_replication_stage",
        "master_seed": 730241,
        "replications_per_regime": 120,
        "posterior_draw_count": 128,
        "nominal_interval": 0.95,
        "posterior_interval_rule": (
            "symmetric_finite_draw_order_statistics_floor_tail"
        ),
        "finite_draw_interval_coverage": 123.0 / 129.0,
        "rank_tie_policy": RANK_TIE_POLICY,
        "rank_bins": 10,
        "simultaneous_family_alpha": 0.01,
        "simultaneous_family_size": 84,
        "simultaneous_per_test_alpha": 0.01 / 84.0,
        "simultaneous_rule": (
            "pooled_and_all_20_regimes_pass_bonferroni_family_alpha_0.01"
        ),
        "minimum_clusters_per_dimension": 4,
        "minimum_ess_per_dimension": 3.5,
        "multiway_bootstrap_replicates": 512,
        "small_cluster_correction": "G_over_G_minus_1",
        "resampling_design": (
            "hierarchical_identity_component_then_series_within_component_"
            "crossed_with_tournament_time_and_patch"
        ),
        "identity_dependence_construction": (
            "connected_components_of_series_sharing_any_participant_or_team_identity"
        ),
        "methodological_basis": [
            "Owen_2007_pigeonhole_bootstrap_one_label_per_dimension",
            "Davezies_DHaultfoeuille_Guyonvarch_2018_multiway_clustering",
        ],
        "public_interval_wording": PUBLIC_INTERVAL_WORDING,
        "claim_ceiling": list(CLAIM_CEILING),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "authority_threat_model": dict(AUTHORITY_THREAT_MODEL),
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "immutable_input": B2_PROCEDURE_LOCATOR.as_posix(),
        "immutable_input_object_sha256": _EXPECTED_B2_OBJECT_SHA256,
    }


def _seed(master: int, regime_index: int, replication: int, stage: int) -> int:
    material = f"{master}:{regime_index}:{replication}:{stage}".encode("ascii")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _prior_draw(regime: Mapping[str, Any], seed: int) -> float:
    rng = np.random.Generator(np.random.PCG64(seed))
    return float(rng.beta(regime["prior_alpha"], regime["prior_beta"]))


def _simulate_observations(
    regime: Mapping[str, Any], latent_truth: float, seed: int
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    n = int(regime["observation_count"])
    outcomes = rng.binomial(1, latent_truth, size=n).astype(int).tolist()
    return {
        "outcomes": outcomes,
        "observation_count": n,
        "dependence_block_id": (
            f"shared-latent-{seed % 7}"
            if regime["regime_kind"] == "dependence"
            else None
        ),
    }


def _inference_adapter(
    regime_public: Mapping[str, Any],
    observations: Mapping[str, Any],
    seed: int,
    draw_count: int,
) -> list[float]:
    allowed = {
        "regime_id",
        "output_type",
        "frozen_stratum",
        "regime_kind",
        "prior_alpha",
        "prior_beta",
        "observation_count",
        "shared_effect_scale",
    }
    if set(regime_public) != allowed or "latent_truth" in observations:
        raise B3CoverageError("inference input contains forbidden or substituted fields")
    outcomes = observations["outcomes"]
    if len(outcomes) != regime_public["observation_count"]:
        raise B3CoverageError("observation count mismatch")
    successes = int(sum(outcomes))
    alpha = float(regime_public["prior_alpha"]) + successes
    beta = float(regime_public["prior_beta"]) + len(outcomes) - successes
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.beta(alpha, beta, size=draw_count).astype(float).tolist()


def _control_inference_adapter(
    mode: str,
    regime_public: Mapping[str, Any],
    observations: Mapping[str, Any],
    seed: int,
    draw_count: int,
) -> list[float]:
    base = _inference_adapter(
        regime_public, observations, seed, draw_count
    )
    centre = statistics.fmean(base)
    if mode == "known_good":
        return base
    if mode == "biased":
        return [min(1.0, draw + 0.20) for draw in base]
    if mode == "underdispersed":
        return [centre + 0.18 * (draw - centre) for draw in base]
    if mode == "overdispersed":
        return [min(1.0, max(0.0, centre + 2.8 * (draw - centre))) for draw in base]
    if mode == "centre_ranked_degenerate":
        return [centre] * draw_count
    raise B3CoverageError(f"unknown frozen control adapter: {mode}")


def _randomized_rank(
    truth: float, draws: Sequence[float], tie_seed: int
) -> tuple[int, int, int]:
    below = sum(draw < truth for draw in draws)
    ties = sum(draw == truth for draw in draws)
    rng = np.random.Generator(np.random.PCG64(tie_seed))
    tie_offset = int(rng.integers(0, ties + 1)) if ties else 0
    return below + tie_offset, ties, tie_offset


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _finite_draw_interval(
    draws: Sequence[float], nominal: float
) -> tuple[list[float], dict[str, Any]]:
    draw_count = len(draws)
    tail_ranks = math.floor((draw_count + 1) * (1.0 - nominal) / 2.0)
    if tail_ranks < 1:
        raise B3CoverageError("insufficient posterior draws for finite-rank interval")
    ordered = sorted(float(draw) for draw in draws)
    lower_index = tail_ranks - 1
    upper_index = draw_count - tail_ranks
    covered_rank_count = draw_count - 2 * tail_ranks + 1
    exact_coverage = covered_rank_count / (draw_count + 1)
    return (
        [ordered[lower_index], ordered[upper_index]],
        {
            "rule": "symmetric_finite_draw_order_statistics_floor_tail",
            "draw_count": draw_count,
            "rank_support": [0, draw_count],
            "covered_rank_bounds": [tail_ranks, draw_count - tail_ranks],
            "covered_rank_count": covered_rank_count,
            "lower_order_index_zero_based": lower_index,
            "upper_order_index_zero_based": upper_index,
            "exact_finite_draw_coverage": exact_coverage,
            "registered_nominal": nominal,
        },
    )


def _replications(
    config: Mapping[str, Any], regimes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    count = int(config["replications_per_regime"])
    draws_n = int(config["posterior_draw_count"])
    alpha_tail = (1.0 - float(config["nominal_interval"])) / 2.0
    adapter_hash = _callable_fingerprint(_inference_adapter)
    simulator_hash = _callable_fingerprint(_simulate_observations)
    for regime_index, regime in enumerate(regimes):
        for replication in range(count):
            prior_seed = _seed(config["master_seed"], regime_index, replication, 1)
            observation_seed = _seed(
                config["master_seed"], regime_index, replication, 2
            )
            inference_seed = _seed(
                config["master_seed"], regime_index, replication, 3
            )
            tie_seed = _seed(config["master_seed"], regime_index, replication, 4)
            truth = _prior_draw(regime, prior_seed)
            observations = _simulate_observations(regime, truth, observation_seed)
            inference_input = {
                "regime": dict(regime),
                "observations": observations,
                "seed": inference_seed,
                "draw_count": draws_n,
            }
            draws = _inference_adapter(
                inference_input["regime"],
                inference_input["observations"],
                inference_seed,
                draws_n,
            )
            rank, ties, tie_offset = _randomized_rank(truth, draws, tie_seed)
            interval, interval_rule = _finite_draw_interval(
                draws, float(config["nominal_interval"])
            )
            lower, upper = interval
            successes = sum(observations["outcomes"])
            analytical_lower, analytical_upper = beta_distribution.ppf(
                [alpha_tail, 1.0 - alpha_tail],
                float(regime["prior_alpha"]) + successes,
                float(regime["prior_beta"])
                + len(observations["outcomes"])
                - successes,
            )
            records.append(
                {
                    "replication_id": f"{regime['regime_id']}:{replication:03d}",
                    "regime_id": regime["regime_id"],
                    "output_type": regime["output_type"],
                    "frozen_stratum": regime["frozen_stratum"],
                    "rng_algorithm": RNG_ALGORITHM,
                    "draw_indices": {
                        "prior": replication,
                        "observation": replication,
                        "inference": replication,
                        "tie": replication,
                    },
                    "seeds": {
                        "prior": prior_seed,
                        "observation": observation_seed,
                        "inference": inference_seed,
                        "tie": tie_seed,
                    },
                    "latent_truth": truth,
                    "observation": observations,
                    "observation_sha256": _object_hash(observations),
                    "inference_input_sha256": _object_hash(inference_input),
                    "inference_adapter_sha256": adapter_hash,
                    "simulator_sha256": simulator_hash,
                    "posterior_draws": draws,
                    "posterior_draws_sha256": _object_hash(draws),
                    "posterior_support": {
                        "draw_count": draws_n,
                        "finite_draw_count": draws_n,
                        "unique_draw_count": len(set(draws)),
                        "exact_ess": float(draws_n),
                        "support_min": min(draws),
                        "support_max": max(draws),
                    },
                    "randomized_rank": rank,
                    "rank_support": [0, draws_n],
                    "tie_count": ties,
                    "tie_offset": tie_offset,
                    "tie_policy": RANK_TIE_POLICY,
                    "interval": [lower, upper],
                    "interval_rule": interval_rule,
                    "covered": lower <= truth <= upper,
                    "interval_width": upper - lower,
                    "analytical_beta_interval": [
                        float(analytical_lower),
                        float(analytical_upper),
                    ],
                    "analytical_beta_covered": bool(
                        analytical_lower <= truth <= analytical_upper
                    ),
                    "analytical_beta_interval_width": float(
                        analytical_upper - analytical_lower
                    ),
                    "secondary_test_quantity": {
                        "sample_mean": statistics.fmean(observations["outcomes"]),
                        "posterior_mean": statistics.fmean(draws),
                        "absolute_mean_residual": abs(
                            statistics.fmean(draws)
                            - statistics.fmean(observations["outcomes"])
                        ),
                    },
                }
            )
    return records


def _control_replications(
    base_records: Sequence[Mapping[str, Any]],
    regimes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    regime_by_id = {regime["regime_id"]: regime for regime in regimes}
    draw_count = int(config["posterior_draw_count"])
    alpha_tail = (1.0 - float(config["nominal_interval"])) / 2.0
    adapter_hash = _callable_fingerprint(_control_inference_adapter)
    controls: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(CONTROL_NAMES):
        for base in base_records:
            regime = regime_by_id[base["regime_id"]]
            seed = int(base["seeds"]["inference"]) + mode_index * 10_000_000
            draws = _control_inference_adapter(
                mode, regime, base["observation"], seed, draw_count
            )
            rank, ties, tie_offset = _randomized_rank(
                base["latent_truth"], draws, int(base["seeds"]["tie"])
            )
            interval, interval_rule = _finite_draw_interval(
                draws, float(config["nominal_interval"])
            )
            lower, upper = interval
            inference_input = {
                "adapter_id": f"frozen:{mode}:v1",
                "mode": mode,
                "regime": regime,
                "observations": base["observation"],
                "seed": seed,
                "draw_count": draw_count,
            }
            controls.append(
                {
                    "control_id": f"{mode}:{base['replication_id']}",
                    "control": mode,
                    "replication_id": base["replication_id"],
                    "regime_id": base["regime_id"],
                    "output_type": base["output_type"],
                    "frozen_stratum": base["frozen_stratum"],
                    "latent_truth": base["latent_truth"],
                    "observation_sha256": base["observation_sha256"],
                    "adapter_id": f"frozen:{mode}:v1",
                    "adapter_code_sha256": adapter_hash,
                    "inference_seed": seed,
                    "inference_input_sha256": _object_hash(inference_input),
                    "posterior_draws": draws,
                    "posterior_draws_sha256": _object_hash(draws),
                    "posterior_support": {
                        "draw_count": draw_count,
                        "finite_draw_count": draw_count,
                        "unique_draw_count": len(set(draws)),
                        "exact_ess": float(draw_count),
                        "support_min": min(draws),
                        "support_max": max(draws),
                    },
                    "randomized_rank": rank,
                    "rank_support": [0, draw_count],
                    "tie_count": ties,
                    "tie_offset": tie_offset,
                    "tie_policy": RANK_TIE_POLICY,
                    "interval": [lower, upper],
                    "interval_rule": interval_rule,
                    "covered": lower <= base["latent_truth"] <= upper,
                    "interval_width": upper - lower,
                }
            )
    return controls


def _diagnostics(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    control: str,
) -> dict[str, Any]:
    draw_count = int(config["posterior_draw_count"])
    ranks = [int(record["randomized_rank"]) for record in records]
    truths = [float(record["latent_truth"]) for record in records]
    intervals = [list(record["interval"]) for record in records]
    if control not in CONTROL_NAMES:
        raise B3CoverageError(f"unknown control: {control}")

    bin_count = int(config["rank_bins"])
    histogram = [0] * bin_count
    for rank in ranks:
        index = min(bin_count - 1, (rank * bin_count) // (draw_count + 1))
        histogram[index] += 1
    expected = len(ranks) / bin_count
    chi_square = sum((count - expected) ** 2 / expected for count in histogram)
    normalized = sorted((rank + 0.5) / (draw_count + 1) for rank in ranks)
    ecdf_max = max(
        max(abs((index + 1) / len(normalized) - value), abs(index / len(normalized) - value))
        for index, value in enumerate(normalized)
    )
    covered = [
        low <= truth <= high
        for truth, (low, high) in zip(truths, intervals)
    ]
    widths = [high - low for low, high in intervals]
    empirical_coverage = statistics.fmean(covered)
    expected_interval_coverage = float(config["finite_draw_interval_coverage"])
    coverage_error = abs(empirical_coverage - expected_interval_coverage)
    rank_mean_z = abs(
        statistics.fmean(ranks) - draw_count / 2.0
    ) / math.sqrt(draw_count * (draw_count + 2) / (12.0 * len(ranks)))
    failures: list[str] = []
    adjusted_alpha = float(config["simultaneous_per_test_alpha"])
    chi_square_critical = float(chi2.ppf(1.0 - adjusted_alpha, bin_count - 1))
    ecdf_critical = math.sqrt(
        math.log(2.0 / adjusted_alpha) / (2.0 * len(ranks))
    )
    covered_count = sum(covered)
    coverage_count_lower = int(
        binom.ppf(
            adjusted_alpha / 2.0,
            len(ranks),
            expected_interval_coverage,
        )
    )
    coverage_count_upper = int(
        binom.ppf(
            1.0 - adjusted_alpha / 2.0,
            len(ranks),
            expected_interval_coverage,
        )
    )
    rank_fraction_deviation = abs(
        statistics.fmean(ranks) / draw_count - 0.5
    )
    rank_fraction_critical = math.sqrt(
        math.log(2.0 / adjusted_alpha) / (2.0 * len(ranks))
    )
    if chi_square > chi_square_critical:
        failures.append("rank_nonuniform_chi_square")
    if ecdf_max > ecdf_critical:
        failures.append("rank_nonuniform_ecdf")
    if not coverage_count_lower <= covered_count <= coverage_count_upper:
        failures.append("interval_coverage")
    if rank_fraction_deviation > rank_fraction_critical:
        failures.append("rank_location")
    if min(widths) <= 0.0:
        failures.append("point_or_negative_interval")
    if max(widths) >= 1.0:
        failures.append("trivial_interval")
    expected_pass = control == "known_good"
    passed = not failures
    return {
        "control": control,
        "expected_pass": expected_pass,
        "passed": passed,
        "intended_control_rejected": (not expected_pass and not passed),
        "failure_reasons": failures,
        "rank_histogram": histogram,
        "rank_histogram_support": [0, draw_count],
        "rank_chi_square": chi_square,
        "rank_chi_square_p_value": float(gammaincc(9.0 / 2.0, chi_square / 2.0)),
        "rank_chi_square_critical": chi_square_critical,
        "rank_ecdf_max": ecdf_max,
        "rank_ecdf_critical": ecdf_critical,
        "rank_ecdf_p_value_bound": min(
            1.0, 2.0 * math.exp(-2.0 * len(ranks) * ecdf_max**2)
        ),
        "rank_mean_z": rank_mean_z,
        "rank_fraction_deviation": rank_fraction_deviation,
        "rank_fraction_critical": rank_fraction_critical,
        "rank_location_p_value_bound": min(
            1.0,
            2.0
            * math.exp(-2.0 * len(ranks) * rank_fraction_deviation**2),
        ),
        "rank_bin_mcse": math.sqrt(0.1 * 0.9 / len(ranks)),
        "empirical_interval_coverage": empirical_coverage,
        "covered_count": covered_count,
        "coverage_count_acceptance": [
            coverage_count_lower,
            coverage_count_upper,
        ],
        "coverage_exact_p_value": float(
            binomtest(
                covered_count,
                len(ranks),
                expected_interval_coverage,
                alternative="two-sided",
            ).pvalue
        ),
        "simultaneous_adjusted_alpha": adjusted_alpha,
        "coverage_mcse": math.sqrt(
            empirical_coverage * (1.0 - empirical_coverage) / len(ranks)
        ),
        "coverage_null": expected_interval_coverage,
        "interval_width": {
            "mean": statistics.fmean(widths),
            "median": statistics.median(widths),
            "upper_tail_p90": _quantile(widths, 0.90),
        },
        "secondary_test_quantity": {
            "mean_rank_fraction": statistics.fmean(ranks) / draw_count
        },
    }


def _analytical_coverage_reference(
    records: Sequence[Mapping[str, Any]],
    regimes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    adjusted_alpha = float(config["simultaneous_per_test_alpha"])

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        covered = int(
            sum(bool(record["analytical_beta_covered"]) for record in selected)
        )
        lower = int(
            binom.ppf(
                adjusted_alpha / 2.0,
                count,
                float(config["nominal_interval"]),
            )
        )
        upper = int(
            binom.ppf(
                1.0 - adjusted_alpha / 2.0,
                count,
                float(config["nominal_interval"]),
            )
        )
        return {
            "replication_count": count,
            "covered_count": covered,
            "acceptance": [lower, upper],
            "exact_p_value": float(
                binomtest(
                    covered,
                    count,
                    float(config["nominal_interval"]),
                    alternative="two-sided",
                ).pvalue
            ),
            "mean_interval_width": statistics.fmean(
                record["analytical_beta_interval_width"] for record in selected
            ),
            "passed": lower <= covered <= upper,
        }

    per_regime = [
        summarize(
            [
                record
                for record in records
                if record["regime_id"] == regime["regime_id"]
            ]
        )
        | {"regime_id": regime["regime_id"]}
        for regime in regimes
    ]
    pooled = summarize(records)
    return {
        "purpose": "posterior_draw_interval_mc_support_reference",
        "pooled": pooled,
        "per_regime": per_regime,
        "all_pass": pooled["passed"]
        and all(item["passed"] for item in per_regime),
    }


def _heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0
    for cell_index, output in enumerate(OUTPUT_TYPES):
        for stratum_index, stratum in enumerate(("established", "sparse")):
            cell_id = f"{output}:{stratum}"
            for series_local in range(6):
                series_id = f"S{cell_index:02d}{stratum_index}{series_local}"
                tournament = f"T{series_local % 3}"
                time_block = (
                    "2026-W10"
                    if series_local == 0
                    else f"2026-W{10 + cell_index + series_local:02d}"
                )
                patch = (
                    "26.14"
                    if series_local < 2
                    else f"26.{13 + series_local}"
                )
                teams = [
                    f"TEAM_ANCHOR_{series_local}",
                    f"TEAM_OPP_{series_local}_{cell_index}_{stratum_index}",
                ]
                participants = [
                    f"P_ANCHOR_{series_local}",
                    f"P_OPP_{series_local}_{cell_index}_{stratum_index}",
                ]
                for map_index in range(2):
                    as_of = f"2026-04-{1 + cell_index:02d}T12:00:00Z"
                    resolved_at = f"2026-04-{2 + cell_index:02d}T20:00:00Z"
                    if cell_index == 0 and stratum_index == 1:
                        observed = False
                    elif cell_index == len(OUTPUT_TYPES) - 1 and stratum_index == 0:
                        observed = True
                    else:
                        observed = (
                            (cell_index + stratum_index + series_local + map_index) % 3
                        ) != 0
                    rows.append(
                        {
                            "row_id": f"R{row_index:03d}",
                            "row_type": "atomic_series_map",
                            "series_id": series_id,
                            "map_index": map_index,
                            "cell_id": cell_id,
                            "output_type": output,
                            "frozen_stratum": stratum,
                            "participant_ids": participants,
                            "team_ids": teams,
                            "participant_team_id": f"{participants[0]}:{teams[0]}|{participants[1]}:{teams[1]}",
                            "tournament_id": tournament,
                            "event_id": f"{tournament}:E{cell_index}",
                            "adjacent_time_block": time_block,
                            "tournament_time_id": f"{tournament}:{time_block}",
                            "patch_shock_id": patch,
                            "as_of_time": as_of,
                            "resolved": True,
                            "resolved_at": resolved_at,
                            "outcome_available_at": resolved_at,
                            "observed_outcome": observed,
                        }
                    )
                    row_index += 1
    for unresolved_series in range(2):
        for map_index in range(2):
            rows.append(
                {
                    "row_id": f"R{row_index:03d}",
                    "row_type": "atomic_series_map",
                    "series_id": f"U{unresolved_series}",
                    "map_index": map_index,
                    "cell_id": "draft_score:sparse",
                    "output_type": "draft_score",
                    "frozen_stratum": "sparse",
                    "participant_ids": [f"UP{unresolved_series}", f"UP{unresolved_series + 2}"],
                    "team_ids": [f"UTEAM{unresolved_series}", f"UTEAM{unresolved_series + 2}"],
                    "participant_team_id": f"UPAIR{unresolved_series}",
                    "tournament_id": "UT",
                    "event_id": "UT:E0",
                    "adjacent_time_block": "2026-W30",
                    "tournament_time_id": "UT:2026-W30",
                    "patch_shock_id": "26.16",
                    "as_of_time": "2026-05-01T12:00:00Z",
                    "resolved": False,
                    "resolved_at": None,
                    "outcome_available_at": None,
                    "observed_outcome": None,
                }
            )
            row_index += 1
    return rows


def _heldout_cells(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    resolved = [row for row in rows if row["resolved"]]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in resolved:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    cells: list[dict[str, Any]] = []
    for cell_index, cell_id in enumerate(sorted(grouped)):
        members = sorted(grouped[cell_id], key=lambda row: row["row_id"])
        column_ids = [str(row["row_id"]) for row in members]
        draw_matrix: list[list[float]] = []
        rng = np.random.Generator(np.random.PCG64(880_000 + cell_index))
        base = 0.60 if members[0]["frozen_stratum"] == "established" else 0.52
        for _ in range(320):
            shared = float(rng.normal(0.0, 0.10))
            probabilities = [
                min(0.94, max(0.06, base + shared + (index % 2) * 0.03))
                for index in range(len(members))
            ]
            draw_matrix.append(
                rng.binomial(1, probabilities).astype(float).tolist()
            )
        aggregate_draws = [
            statistics.fmean(draw) for draw in draw_matrix
        ]
        lower = _quantile(aggregate_draws, 0.025)
        upper = _quantile(aggregate_draws, 0.975)
        observed_aggregate = statistics.fmean(
            float(row["observed_outcome"]) for row in members
        )
        predictive_payload = {
            "column_row_ids": column_ids,
            "joint_draws": draw_matrix,
            "aggregate_draws": aggregate_draws,
        }
        cells.append(
            {
                "cell_id": cell_id,
                "output_type": members[0]["output_type"],
                "frozen_stratum": members[0]["frozen_stratum"],
                "row_ids": column_ids,
                "series_ids": sorted({row["series_id"] for row in members}),
                "joint_posterior_predictive": predictive_payload,
                "predictive_bytes_sha256": _object_hash(predictive_payload),
                "aggregate_interval": [lower, upper],
                "observed_aggregate": observed_aggregate,
                "covered": lower <= observed_aggregate <= upper,
                "interval_width": upper - lower,
                "baseline_independent_width": 2.0
                * 1.96
                * math.sqrt(base * (1.0 - base) / len(members)),
                "nominal": config["nominal_interval"],
            }
        )
    return cells


def _level_metrics(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    sizes: dict[str, int] = {}
    for row in rows:
        for label in _dimension_labels(row, field):
            sizes[label] = sizes.get(label, 0) + 1
    values = sorted(sizes.values())
    kish = (sum(values) ** 2) / sum(value * value for value in values)
    count = len(values)
    return {
        "field": field,
        "raw_cluster_count": count,
        "size_distribution": {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "counts": values,
        },
        "recurrence": sum(value > 1 for value in values),
        "membership_count": sum(values),
        "multi_membership": field in ("participant_ids", "team_ids"),
        "consumed_identity_sha256": _object_hash(sorted(sizes)),
        "kish_ess": kish,
        "small_cluster_correction": count / (count - 1) if count > 1 else None,
    }


def _identity_components(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    series_ids = sorted({str(row["series_id"]) for row in rows})
    parent = {series_id: series_id for series_id in series_ids}

    def find(series_id: str) -> str:
        while parent[series_id] != series_id:
            parent[series_id] = parent[parent[series_id]]
            series_id = parent[series_id]
        return series_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    identity_series: dict[str, set[str]] = {}
    for row in rows:
        series_id = str(row["series_id"])
        for identity_type in ("participant_ids", "team_ids"):
            for identity in row[identity_type]:
                key = f"{identity_type}:{identity}"
                identity_series.setdefault(key, set()).add(series_id)
    for members in identity_series.values():
        ordered = sorted(members)
        for member in ordered[1:]:
            union(ordered[0], member)
    component_members: dict[str, list[str]] = {}
    for series_id in series_ids:
        component_members.setdefault(find(series_id), []).append(series_id)
    component_id_by_series = {
        series_id: f"identity-component:{root}"
        for root, members in sorted(component_members.items())
        for series_id in members
    }
    row_component = {
        row["row_id"]: component_id_by_series[str(row["series_id"])]
        for row in rows
    }
    participant_metrics = _level_metrics(rows, "participant_ids")
    team_metrics = _level_metrics(rows, "team_ids")
    return {
        "construction": (
            "connected_components_of_series_sharing_any_participant_or_team_identity"
        ),
        "one_label_per_row_dimension": True,
        "row_component": row_component,
        "component_count": len(component_members),
        "component_series": {
            f"identity-component:{root}": members
            for root, members in sorted(component_members.items())
        },
        "participant_identity_support": participant_metrics,
        "team_identity_support": team_metrics,
        "assignment_sha256": _object_hash(row_component),
    }


def _weighted_coverage_statistic(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    row_weights: Mapping[str, float],
    record_evidence: bool = True,
) -> dict[str, Any]:
    row_by_id = {row["row_id"]: row for row in rows}
    contributions: list[dict[str, Any]] = []
    for cell in cells:
        predictive = cell["joint_posterior_predictive"]
        column_ids = predictive["column_row_ids"]
        if len(column_ids) != len(set(column_ids)):
            raise B3CoverageError("duplicate predictive columns during recomputation")
        column_index = {row_id: index for index, row_id in enumerate(column_ids)}
        if any(row_id not in column_index for row_id in cell["row_ids"]):
            raise B3CoverageError("predictive column substitution during recomputation")
        members = [
            row_by_id[row_id]
            for row_id in cell["row_ids"]
            if row_id in row_by_id
        ]
        weights = [float(row_weights.get(row["row_id"], 0.0)) for row in members]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            continue
        weight_array = np.asarray(weights, dtype=float)
        observed_array = np.asarray(
            [float(row["observed_outcome"]) for row in members], dtype=float
        )
        observed = float(np.sum(observed_array * weight_array) / total_weight)
        indices = [column_index[row["row_id"]] for row in members]
        predictive_matrix = np.asarray(
            predictive["joint_draws"], dtype=float
        )[:, indices]
        predictive_aggregates = (
            np.sum(predictive_matrix * weight_array, axis=1) / total_weight
        ).astype(float).tolist()
        lower = _quantile(predictive_aggregates, 0.025)
        upper = _quantile(predictive_aggregates, 0.975)
        covered = lower <= observed <= upper
        contributions.append(
            {
                "cell_id": cell["cell_id"],
                "weighted_observed_aggregate": observed,
                "weighted_predictive_interval": [lower, upper],
                "weighted_predictive_draws_sha256": (
                    _object_hash(predictive_aggregates)
                    if record_evidence
                    else None
                ),
                "covered": covered,
                "cell_weight": total_weight,
                "row_ids": [row["row_id"] for row in members],
            }
        )
    if not contributions:
        raise B3CoverageError("multiway replicate has no supported cells")
    numerator = sum(
        contribution["cell_weight"] * float(contribution["covered"])
        for contribution in contributions
    )
    denominator = sum(contribution["cell_weight"] for contribution in contributions)
    return {
        "estimate": numerator / denominator,
        "cell_contributions": contributions,
        "cell_count": len(contributions),
        "total_weight": denominator,
    }


def _dimension_labels(
    row: Mapping[str, Any], dimension: str
) -> tuple[str, ...]:
    if dimension in ("participant_ids", "team_ids"):
        labels = tuple(sorted(str(value) for value in row[dimension]))
        if len(labels) != 2 or len(set(labels)) != 2:
            raise B3CoverageError(f"invalid multi-membership dimension: {dimension}")
        return labels
    return (str(row[dimension]),)


def _pigeonhole_distribution(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
    seed: int,
    replicates: int,
    nested_parent_dimension: str | None = None,
    nested_series_dimension: str | None = None,
) -> list[float]:
    if (nested_parent_dimension is None) != (nested_series_dimension is None):
        raise B3CoverageError("nested bootstrap requires parent and series dimensions")
    rng = np.random.Generator(np.random.PCG64(seed))
    clusters = {
        dimension: sorted(
            {
                label
                for row in rows
                for label in _dimension_labels(row, dimension)
            }
        )
        for dimension in dimensions
    }
    series_by_parent: dict[str, list[str]] = {}
    parent_by_series: dict[str, str] = {}
    if nested_parent_dimension is not None and nested_series_dimension is not None:
        for row in rows:
            parent = str(row[nested_parent_dimension])
            series = str(row[nested_series_dimension])
            previous = parent_by_series.setdefault(series, parent)
            if previous != parent:
                raise B3CoverageError("series split across identity components")
        for series, parent in parent_by_series.items():
            series_by_parent.setdefault(parent, []).append(series)
        for parent in series_by_parent:
            series_by_parent[parent].sort()
    distribution: list[float] = []
    attempts = 0
    while len(distribution) < replicates and attempts < replicates * 4:
        attempts += 1
        weights_by_dimension: dict[str, dict[str, int]] = {}
        for dimension in dimensions:
            labels = clusters[dimension]
            counts = rng.multinomial(
                len(labels), np.full(len(labels), 1.0 / len(labels))
            )
            weights_by_dimension[dimension] = dict(zip(labels, counts.tolist()))
        nested_series_weights: dict[str, int] | None = None
        if nested_parent_dimension is not None:
            parents = sorted(series_by_parent)
            parent_counts = rng.multinomial(
                len(parents), np.full(len(parents), 1.0 / len(parents))
            )
            nested_series_weights = {}
            for parent, parent_count in zip(parents, parent_counts.tolist()):
                members = series_by_parent[parent]
                draws = parent_count * len(members)
                counts = (
                    rng.multinomial(
                        draws, np.full(len(members), 1.0 / len(members))
                    )
                    if draws
                    else np.zeros(len(members), dtype=int)
                )
                nested_series_weights.update(dict(zip(members, counts.tolist())))
        row_weights = {
            row["row_id"]: float(
                (
                    nested_series_weights[str(row[nested_series_dimension])]
                    if nested_series_weights is not None
                    else 1.0
                )
                * math.prod(
                    statistics.fmean(
                        weights_by_dimension[dimension][label]
                        for label in _dimension_labels(row, dimension)
                    )
                    for dimension in dimensions
                )
            )
            for row in rows
        }
        if not any(row_weights.values()):
            continue
        distribution.append(
            _weighted_coverage_statistic(
                rows, cells, row_weights, record_evidence=False
            )["estimate"]
        )
    if len(distribution) != replicates:
        raise B3CoverageError("insufficient valid multiway bootstrap replicates")
    return distribution


def _support_status(
    levels: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    minimum_count = int(config["minimum_clusters_per_dimension"])
    minimum_ess = float(config["minimum_ess_per_dimension"])
    failures = [
        level["field"]
        for level in levels
        if level["raw_cluster_count"] < minimum_count
        or level["kish_ess"] < minimum_ess
    ]
    effective = min(level["kish_ess"] for level in levels)
    coarsest = min(
        levels,
        key=lambda level: (
            level["kish_ess"],
            level["raw_cluster_count"],
            level["field"],
        ),
    )
    return {
        "status": "available" if not failures else "unavailable_dependence_support",
        "required_dimensions": [level["field"] for level in levels],
        "failed_dimensions": failures,
        "effective_support": effective,
        "coarsest_dimension": coarsest["field"],
        "minimum_cluster_count": minimum_count,
        "minimum_kish_ess": minimum_ess,
    }


def _bootstrap_summary(
    point: float,
    distribution: Sequence[float],
    correction: float,
) -> dict[str, Any]:
    adjusted = [
        min(1.0, max(0.0, point + math.sqrt(correction) * (value - point)))
        for value in distribution
    ]
    lower = _quantile(adjusted, 0.025)
    upper = _quantile(adjusted, 0.975)
    return {
        "replicate_count": len(distribution),
        "replicate_distribution": list(distribution),
        "corrected_interval": [lower, upper],
        "replicate_mean": statistics.fmean(distribution),
        "replicate_sd": statistics.stdev(distribution),
        "mcse_mean": statistics.stdev(distribution) / math.sqrt(len(distribution)),
        "small_cluster_correction": correction,
        "coverage_performance_decision": "unavailable_mechanics_fixture_only",
    }


def _dependence(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_raw = [row for row in rows if row["resolved"]]
    identity_network = _identity_components(primary_raw)
    primary = [
        dict(
            row,
            identity_component_id=identity_network["row_component"][row["row_id"]],
        )
        for row in primary_raw
    ]
    dimensions = (
        "series_id",
        "identity_component_id",
        "tournament_time_id",
        "patch_shock_id",
    )
    levels = [_level_metrics(primary, field) for field in dimensions]
    support = _support_status(levels, config)
    unresolved = [row for row in rows if not row["resolved"]]
    sensitivity_groups: dict[str, list[str]] = {}
    for row in unresolved:
        key = f"{row['tournament_time_id']}:{row['participant_team_id']}"
        sensitivity_groups.setdefault(key, []).append(row["row_id"])
    if support["status"] != "available":
        return {
            "artifact_id": "scryglass:b3:dependence-procedure:v1",
            "estimand": (
                "coverage of registered cell-level joint posterior-predictive "
                "aggregate intervals over resolved heldout series"
            ),
            "design": config["resampling_design"],
            "methodological_basis": config["methodological_basis"],
            "identity_dependence_assumption": config[
                "identity_dependence_construction"
            ],
            "nested_resampling": {
                "higher_level": "identity_component_id",
                "inner_atomic_block": "series_id",
                "crossed_dimensions": [
                    "tournament_time_id",
                    "patch_shock_id",
                ],
                "global_series_times_component_product": False,
            },
            "series_are_indivisible": True,
            "map_resampling_allowed": False,
            "naive_series_iid_allowed": False,
            "levels": levels,
            "identity_network": identity_network,
            "top_level_support": support,
            "point_estimate": None,
            "multiway_inference": {
                "status": "unavailable_dependence_support",
                "replicate_distribution": [],
            },
            "leave_largest_cluster": {
                "scope": "not_run_inadequate_required_dimension",
                "cases": [],
                "mechanics_availability_stable": False,
                "coverage_decision": "unavailable_mechanics_fixture_only",
            },
            "unresolved_primary_row_count": 0,
            "unresolved_sensitivity": {
                "included": False,
                "coarser_key": "tournament_time_id:participant_team_id",
                "groups": sensitivity_groups,
                "singleton_independent_clusters": False,
            },
            "naive_iid_diagnostic": {
                "authoritative": False,
                "label": "not_run_because_authoritative_support_unavailable",
            },
        }
    unit_weights = {row["row_id"]: 1.0 for row in primary}
    point_detail = _weighted_coverage_statistic(primary, cells, unit_weights)
    correction = max(
        float(level["small_cluster_correction"]) for level in levels
    )
    multiway_distribution = _pigeonhole_distribution(
        primary,
        cells,
        ("tournament_time_id", "patch_shock_id"),
        991_771,
        int(config["multiway_bootstrap_replicates"]),
        nested_parent_dimension="identity_component_id",
        nested_series_dimension="series_id",
    )
    inference = _bootstrap_summary(
        point_detail["estimate"],
        multiway_distribution,
        correction,
    )
    series_distribution = _pigeonhole_distribution(
        primary,
        cells,
        ("series_id",),
        991_771,
        int(config["multiway_bootstrap_replicates"]),
    )
    naive = _bootstrap_summary(
        point_detail["estimate"],
        series_distribution,
        float(levels[0]["small_cluster_correction"]),
    )
    sensitivity_cases: list[dict[str, Any]] = []
    for dimension_index, dimension in enumerate(dimensions[1:], start=1):
        counts = {
            cluster: sum(
                cluster in _dimension_labels(row, dimension) for row in primary
            )
            for cluster in sorted(
                {
                    label
                    for row in primary
                    for label in _dimension_labels(row, dimension)
                }
            )
        }
        largest_size = max(counts.values())
        for candidate_index, cluster in enumerate(
            cluster for cluster, size in counts.items() if size == largest_size
        ):
            kept = [
                row
                for row in primary
                if cluster not in _dimension_labels(row, dimension)
            ]
            kept_ids = {row["row_id"] for row in kept}
            kept_cells: list[dict[str, Any]] = []
            for cell in cells:
                kept_row_ids = [row_id for row_id in cell["row_ids"] if row_id in kept_ids]
                if kept_row_ids:
                    cloned = dict(cell)
                    cloned["row_ids"] = kept_row_ids
                    kept_cells.append(cloned)
            kept_levels = [_level_metrics(kept, field) for field in dimensions]
            kept_support = _support_status(kept_levels, config)
            kept_detail = _weighted_coverage_statistic(
                kept, kept_cells, {row["row_id"]: 1.0 for row in kept}
            )
            kept_point = kept_detail["estimate"]
            kept_distribution = _pigeonhole_distribution(
                kept,
                kept_cells,
                ("tournament_time_id", "patch_shock_id"),
                992_000 + dimension_index * 1_000 + candidate_index,
                96,
                nested_parent_dimension="identity_component_id",
                nested_series_dimension="series_id",
            )
            kept_inference = _bootstrap_summary(
                kept_point,
                kept_distribution,
                max(float(level["small_cluster_correction"]) for level in kept_levels),
            )
            sensitivity_cases.append(
                {
                    "dimension": dimension,
                    "removed_cluster_id": cluster,
                    "removed_row_count": largest_size,
                    "estimate": kept_point,
                    "interval": kept_inference["corrected_interval"],
                    "support_status": kept_support["status"],
                    "recomputed_cell_count": len(kept_cells),
                    "changed_cell_aggregate_count": sum(
                        full["weighted_observed_aggregate"]
                        != reduced["weighted_observed_aggregate"]
                        for full, reduced in zip(
                            point_detail["cell_contributions"],
                            kept_detail["cell_contributions"],
                        )
                        if full["cell_id"] == reduced["cell_id"]
                    ),
                    "remaining_row_ids_sha256": _object_hash(
                        sorted(row["row_id"] for row in kept)
                    ),
                }
            )
    stable = all(
        case["support_status"] == "available"
        and case["changed_cell_aggregate_count"] > 0
        for case in sensitivity_cases
    )
    return {
        "artifact_id": "scryglass:b3:dependence-procedure:v1",
        "estimand": (
            "coverage of registered cell-level joint posterior-predictive "
            "aggregate intervals over resolved heldout series"
        ),
        "design": config["resampling_design"],
        "methodological_basis": config["methodological_basis"],
        "identity_dependence_assumption": config[
            "identity_dependence_construction"
        ],
        "nested_resampling": {
            "higher_level": "identity_component_id",
            "inner_atomic_block": "series_id",
            "crossed_dimensions": [
                "tournament_time_id",
                "patch_shock_id",
            ],
            "global_series_times_component_product": False,
        },
        "series_are_indivisible": True,
        "map_resampling_allowed": False,
        "naive_series_iid_allowed": False,
        "levels": levels,
        "identity_network": identity_network,
        "top_level_support": support,
        "point_estimate": point_detail,
        "multiway_inference": inference,
        "leave_largest_cluster": {
            "scope": "every_tied_largest_higher_level_cluster",
            "cases": sensitivity_cases,
            "estimate_range": [
                min(case["estimate"] for case in sensitivity_cases),
                max(case["estimate"] for case in sensitivity_cases),
            ],
            "worst_absolute_change": max(
                abs(case["estimate"] - point_detail["estimate"])
                for case in sensitivity_cases
            ),
            "mechanics_availability_stable": stable,
            "coverage_decision": "unavailable_mechanics_fixture_only",
        },
        "unresolved_primary_row_count": 0,
        "unresolved_sensitivity": {
            "included": False,
            "coarser_key": "tournament_time_id:participant_team_id",
            "groups": sensitivity_groups,
            "singleton_independent_clusters": False,
        },
        "naive_iid_diagnostic": {
            "authoritative": False,
            "label": "series_only_non_authoritative_contrast",
            "series_only_inference": naive,
            "common_random_number_seed": 991_771,
            "sd_difference": naive["replicate_sd"] - inference["replicate_sd"],
            "interval_width_difference": (
                naive["corrected_interval"][1] - naive["corrected_interval"][0]
            )
            - (
                inference["corrected_interval"][1]
                - inference["corrected_interval"][0]
            ),
            "mcse_sd_difference": math.sqrt(
                naive["replicate_sd"] ** 2
                / (2.0 * (len(series_distribution) - 1))
                + inference["replicate_sd"] ** 2
                / (2.0 * (len(multiway_distribution) - 1))
            ),
            "materiality_rule": "descriptive_only_no_authority_gate",
        },
    }


def _literal_gate(
    gate_id: str, predicate: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "predicate": bool(predicate),
        "evidence": dict(evidence),
        "evidence_sha256": _object_hash(evidence),
    }


def _report(
    config: Mapping[str, Any],
    regimes: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    dependence: Mapping[str, Any],
) -> dict[str, Any]:
    analytical_reference = _analytical_coverage_reference(
        records, regimes, config
    )
    controls: list[dict[str, Any]] = []
    for control in CONTROL_NAMES:
        selected = [record for record in control_records if record["control"] == control]
        pooled = _diagnostics(selected, config, control)
        per_regime = [
            _diagnostics(
                [record for record in selected if record["regime_id"] == regime["regime_id"]],
                config,
                control,
            )
            | {"regime_id": regime["regime_id"]}
            for regime in regimes
        ]
        pooled["per_regime"] = per_regime
        pooled["simultaneous_rule"] = config["simultaneous_rule"]
        pooled["simultaneous_family"] = {
            "family_alpha": config["simultaneous_family_alpha"],
            "family_size": config["simultaneous_family_size"],
            "per_test_alpha": config["simultaneous_per_test_alpha"],
            "datasets": 21,
            "diagnostics_per_dataset": 4,
            "diagnostics": [
                "rank_chi_square",
                "rank_ecdf",
                "interval_coverage",
                "rank_location",
            ],
        }
        pooled["all_regimes_pass"] = all(item["passed"] for item in per_regime)
        pooled["pooled_diagnostic"] = {
            "passed": pooled["passed"],
            "failure_reasons": list(pooled["failure_reasons"]),
            "rank_chi_square": pooled["rank_chi_square"],
            "rank_chi_square_p_value": pooled["rank_chi_square_p_value"],
            "rank_ecdf_max": pooled["rank_ecdf_max"],
        }
        controls.append(pooled)
    widths = [cell["interval_width"] for cell in cells]
    all_strata = sorted(
        {(cell["output_type"], cell["frozen_stratum"]) for cell in cells}
    )
    expected_strata = sorted(
        {(output, stratum) for output in OUTPUT_TYPES for stratum in ("established", "sparse")}
    )
    aggregate = {
        "nominal": config["nominal_interval"],
        "empirical_coverage": dependence["point_estimate"]["estimate"],
        "dependence_aware_interval": dependence["multiway_inference"][
            "corrected_interval"
        ],
        "dependence_aware_mcse": dependence["multiway_inference"]["mcse_mean"],
        "interval_width": {
            "mean": statistics.fmean(widths),
            "median": statistics.median(widths),
            "upper_tail_p90": _quantile(widths, 0.90),
        },
        "baseline_width_mean": statistics.fmean(
            cell["baseline_independent_width"] for cell in cells
        ),
        "cell_count": len(cells),
        "resolved_row_count": sum(row["resolved"] for row in rows),
        "resolved_series_count": len(
            {row["series_id"] for row in rows if row["resolved"]}
        ),
        "strata": [
            {"output_type": output, "frozen_stratum": stratum}
            for output, stratum in all_strata
        ],
    }
    lineage_complete = all(
        record["posterior_draws_sha256"] == _object_hash(record["posterior_draws"])
        and record["observation_sha256"]
        and record["interval_rule"]["exact_finite_draw_coverage"]
        == config["finite_draw_interval_coverage"]
        for record in control_records
    )
    unresolved_excluded = not any(
        row_id
        for cell in cells
        for row_id in cell["row_ids"]
        if row_id not in {row["row_id"] for row in rows if row["resolved"]}
    )
    reconciled = (
        sorted(row_id for cell in cells for row_id in cell["row_ids"])
        == sorted(row["row_id"] for row in rows if row["resolved"])
        and all_strata == expected_strata
        and all(0.0 < width < 1.0 for width in widths)
    )
    gates = [
        _literal_gate(
            "sbc_lineage_complete",
            lineage_complete,
            {
                "base_replications": len(records),
                "control_replications": len(control_records),
                "adapter_ids": sorted({record["adapter_id"] for record in control_records}),
                "posterior_interval_rule": config["posterior_interval_rule"],
                "exact_finite_draw_coverage": config[
                    "finite_draw_interval_coverage"
                ],
            },
        ),
        _literal_gate(
            "sbc_every_regime_uniform",
            controls[0]["passed"]
            and controls[0]["all_regimes_pass"]
            and analytical_reference["all_pass"],
            {
                "simultaneous_rule": config["simultaneous_rule"],
                "simultaneous_family": controls[0]["simultaneous_family"],
                "per_regime_pass": {
                    item["regime_id"]: item["passed"]
                    for item in controls[0]["per_regime"]
                },
                "analytical_coverage_reference": analytical_reference,
            },
        ),
        _literal_gate(
            "faulty_inference_controls_rejected",
            all(control["intended_control_rejected"] for control in controls[1:]),
            {
                "controls": {
                    control["control"]: control["failure_reasons"]
                    for control in controls[1:]
                }
            },
        ),
        _literal_gate(
            "multiway_dependence_available",
            dependence["top_level_support"]["status"] == "available"
            and dependence["leave_largest_cluster"][
                "mechanics_availability_stable"
            ]
            and dependence["multiway_inference"]["replicate_count"]
            == config["multiway_bootstrap_replicates"]
            and dependence["nested_resampling"][
                "global_series_times_component_product"
            ]
            is False,
            # The series weight must be conditional on the selected identity
            # component, never an independently crossed global series weight.
            {
                "support": dependence["top_level_support"],
                "multiway_interval": dependence["multiway_inference"][
                    "corrected_interval"
                ],
                "leave_largest": dependence["leave_largest_cluster"],
                "naive_authoritative": dependence["naive_iid_diagnostic"][
                    "authoritative"
                ],
                "coverage_performance_status": (
                    "unavailable_mechanics_fixture_only"
                ),
                "nested_resampling": dependence["nested_resampling"],
            },
        ),
        _literal_gate(
            "unresolved_rows_excluded",
            unresolved_excluded
            and dependence["unresolved_primary_row_count"] == 0,
            {
                "unresolved_rows": sum(not row["resolved"] for row in rows),
                "primary_unresolved": dependence["unresolved_primary_row_count"],
                "coarser_key": dependence["unresolved_sensitivity"]["coarser_key"],
            },
        ),
        _literal_gate(
            "aggregate_reconciliation",
            reconciled,
            {
                "resolved_rows": sum(row["resolved"] for row in rows),
                "consumed_rows": sum(len(cell["row_ids"]) for cell in cells),
                "cell_count": len(cells),
                "strata": aggregate["strata"],
            },
        ),
        _literal_gate(
            "wording_and_nonpromotion",
            PUBLIC_INTERVAL_WORDING == "95% model range"
            and "production_coverage" in FORBIDDEN_CLAIMS
            and CLAIM_CEILING == (
                "synthetic_sbc_coverage_dependence_mechanics_only",
            )
            and AUTHORITY_THREAT_MODEL["hostile_same_process_security"] is False,
            {
                "wording": PUBLIC_INTERVAL_WORDING,
                "claim_ceiling": list(CLAIM_CEILING),
                "forbidden_claims": list(FORBIDDEN_CLAIMS),
                "authority_threat_model": dict(AUTHORITY_THREAT_MODEL),
            },
        ),
    ]
    mechanics_pass = all(gate["predicate"] for gate in gates)
    return {
        "artifact_id": "scryglass:b3:coverage-report:v1",
        "synthetic_only": True,
        "mechanics_status": "PASS" if mechanics_pass else "REMAND",
        "real_coverage_status": "unavailable_real_competition_data_and_posterior_authority",
        "synthetic_coverage_performance_status": (
            "unavailable_ten_hand_patterned_cells_mechanics_fixture_only"
        ),
        "controls": controls,
        "analytical_coverage_reference": analytical_reference,
        "hard_gates": gates,
        "aggregate_coverage": aggregate,
        "public_interval_wording": PUBLIC_INTERVAL_WORDING,
        "claim_ceiling": list(CLAIM_CEILING),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "authority_threat_model": dict(AUTHORITY_THREAT_MODEL),
        "regime_count": len(regimes),
        "replication_count": len(records),
        "heldout_row_count": len(rows),
        "heldout_cell_count": len(cells),
    }


def build_frozen_payloads(project_root: Path) -> dict[str, dict[str, Any]]:
    """Build the exact synthetic payloads; thresholds are intentionally fixed."""
    root = project_root.resolve(strict=True)
    b2, b2_raw = _read_canonical_json(root / B2_PROCEDURE_LOCATOR)
    if _object_hash(b2) != _EXPECTED_B2_OBJECT_SHA256:
        raise B3CoverageError("immutable B2 coverage procedure changed")
    config = _config()
    regimes = _regime_universe()
    records = _replications(config, regimes)
    control_records = _control_replications(records, regimes, config)
    rows = _heldout_rows()
    cells = _heldout_cells(rows, config)
    dependence = _dependence(rows, cells, config)
    report = _report(
        config,
        regimes,
        records,
        control_records,
        rows,
        cells,
        dependence,
    )
    payloads = {
        "config": config,
        "regimes": {
            "artifact_id": "scryglass:b3:simulation-regimes:v1",
            "regimes": regimes,
        },
        "replications": {
            "artifact_id": "scryglass:b3:simulation-replications:v1",
            "replications": records,
            "control_replications": control_records,
        },
        "heldout_rows": {
            "artifact_id": "scryglass:b3:heldout-rows:v1",
            "rows": rows,
        },
        "heldout_cells": {
            "artifact_id": "scryglass:b3:heldout-cells:v1",
            "cells": cells,
        },
        "dependence": dependence,
        "report": report,
    }
    source_hash = _raw_hash(Path(__file__).read_bytes())
    refs = {
        role: {
            "locator": f"{ARTIFACT_ROOT.as_posix()}/{role.replace('_', '-')}.json",
            "object_sha256": _object_hash(payloads[role]),
            "raw_sha256": _raw_hash(_canonical_bytes(payloads[role])),
        }
        for role in ARTIFACT_ROLES
    }
    source_gate = _literal_gate(
        "source_and_authority_closure",
        len(source_hash) == 64
        and all(
            len(reference["object_sha256"]) == 64
            and len(reference["raw_sha256"]) == 64
            for reference in refs.values()
        )
        and _object_hash(b2) == _EXPECTED_B2_OBJECT_SHA256,
        {
            "source_sha256": source_hash,
            "generator_source_sha256": _raw_hash(
                (
                    root
                    / "lol_kills/v2/evaluation/generate_b3_coverage_artifacts.py"
                ).read_bytes()
            ),
            "immutable_b2_object_sha256": _object_hash(b2),
            "artifact_object_sha256": {
                role: reference["object_sha256"] for role, reference in refs.items()
            },
        },
    )
    payloads["authority"] = {
        "artifact_id": "scryglass:b3:coverage-authority:v1",
        "schema_version": 1,
        "synthetic_only": True,
        "production_eligible": False,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "source_sha256": source_hash,
        "generator_source_sha256": _raw_hash(
            (root / "lol_kills/v2/evaluation/generate_b3_coverage_artifacts.py").read_bytes()
        ),
        "immutable_b2_raw_sha256": _raw_hash(b2_raw),
        "immutable_b2_object_sha256": _object_hash(b2),
        "artifacts": refs,
        "replay_sha256": _object_hash(
            {role: _object_hash(payloads[role]) for role in ARTIFACT_ROLES}
        ),
        "mechanics_status": report["mechanics_status"],
        "real_coverage_status": report["real_coverage_status"],
        "public_interval_wording": PUBLIC_INTERVAL_WORDING,
        "claim_ceiling": list(CLAIM_CEILING),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "hard_gates": [source_gate],
        "authority_threat_model": dict(AUTHORITY_THREAT_MODEL),
    }
    return payloads


def _validate_regimes(regimes: Sequence[Mapping[str, Any]]) -> None:
    expected = [regime["regime_id"] for regime in _regime_universe()]
    actual = [str(regime.get("regime_id")) for regime in regimes]
    if actual != expected or len(actual) != len(set(actual)):
        raise B3CoverageError("missing, duplicated, reordered, or substituted regime")


def _validate_hard_gates(
    gates: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> None:
    actual_ids = [gate.get("gate_id") for gate in gates]
    if actual_ids != list(expected_ids) or len(actual_ids) != len(set(actual_ids)):
        raise B3CoverageError("missing, extra, reordered, or duplicate hard gate")
    for gate in gates:
        if gate.get("predicate") is not True:
            raise B3CoverageError(f"hard gate failed: {gate.get('gate_id')}")
        if gate.get("evidence_sha256") != _object_hash(gate.get("evidence")):
            raise B3CoverageError(f"hard gate evidence mutation: {gate.get('gate_id')}")


def _validate_reconciliation(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    config = payloads["config"]
    regimes = payloads["regimes"]["regimes"]
    records = payloads["replications"]["replications"]
    control_records = payloads["replications"]["control_replications"]
    rows = payloads["heldout_rows"]["rows"]
    cells = payloads["heldout_cells"]["cells"]
    dependence = payloads["dependence"]
    report = payloads["report"]
    _validate_regimes(regimes)
    expected_record_count = len(regimes) * config["replications_per_regime"]
    if len(records) != expected_record_count:
        raise B3CoverageError("replication universe incomplete")
    regime_by_id = {regime["regime_id"]: regime for regime in regimes}
    base_by_id = {record["replication_id"]: record for record in records}
    for record in records:
        regime = regime_by_id[record["regime_id"]]
        observations = record["observation"]
        if record["observation_sha256"] != _object_hash(observations):
            raise B3CoverageError("observation mutation")
        inference_input = {
            "regime": regime,
            "observations": observations,
            "seed": record["seeds"]["inference"],
            "draw_count": config["posterior_draw_count"],
        }
        if record["inference_input_sha256"] != _object_hash(inference_input):
            raise B3CoverageError("inference input mutation or truth leakage")
        replay_draws = _inference_adapter(
            regime,
            observations,
            record["seeds"]["inference"],
            config["posterior_draw_count"],
        )
        if (
            record["posterior_draws"] != replay_draws
            or record["posterior_draws_sha256"] != _object_hash(replay_draws)
        ):
            raise B3CoverageError("posterior replay mismatch")
        rank = _randomized_rank(
            record["latent_truth"], replay_draws, record["seeds"]["tie"]
        )
        if (
            record["randomized_rank"],
            record["tie_count"],
            record["tie_offset"],
        ) != rank or record["tie_policy"] != RANK_TIE_POLICY:
            raise B3CoverageError("rank or tie manipulation")
        interval, interval_rule = _finite_draw_interval(
            replay_draws, float(config["nominal_interval"])
        )
        if record["interval"] != interval or record["interval_rule"] != interval_rule:
            raise B3CoverageError("finite-draw interval rule mutation")
        support = record["posterior_support"]
        if (
            support["draw_count"] != len(replay_draws)
            or support["finite_draw_count"] != len(replay_draws)
            or support["unique_draw_count"] != len(set(replay_draws))
            or support["exact_ess"] != float(len(replay_draws))
        ):
            raise B3CoverageError("posterior support or ESS fabrication")
    expected_control_count = expected_record_count * len(CONTROL_NAMES)
    if len(control_records) != expected_control_count:
        raise B3CoverageError("control replication universe incomplete")
    for record in control_records:
        base = base_by_id.get(record["replication_id"])
        if base is None or record["control"] not in CONTROL_NAMES:
            raise B3CoverageError("control replication substitution")
        regime = regime_by_id[record["regime_id"]]
        if record["observation_sha256"] != base["observation_sha256"]:
            raise B3CoverageError("control observation lineage mutation")
        inference_input = {
            "adapter_id": record["adapter_id"],
            "mode": record["control"],
            "regime": regime,
            "observations": base["observation"],
            "seed": record["inference_seed"],
            "draw_count": config["posterior_draw_count"],
        }
        if (
            record["adapter_id"] != f"frozen:{record['control']}:v1"
            or record["adapter_code_sha256"]
            != _callable_fingerprint(_control_inference_adapter)
            or record["inference_input_sha256"] != _object_hash(inference_input)
        ):
            raise B3CoverageError("control adapter/config lineage mutation")
        replay_draws = _control_inference_adapter(
            record["control"],
            regime,
            base["observation"],
            record["inference_seed"],
            config["posterior_draw_count"],
        )
        if (
            record["posterior_draws"] != replay_draws
            or record["posterior_draws_sha256"] != _object_hash(replay_draws)
        ):
            raise B3CoverageError("control posterior replay mismatch")
        rank = _randomized_rank(
            base["latent_truth"], replay_draws, base["seeds"]["tie"]
        )
        if (
            record["randomized_rank"],
            record["tie_count"],
            record["tie_offset"],
        ) != rank:
            raise B3CoverageError("control rank/tie manipulation")
        interval, interval_rule = _finite_draw_interval(
            replay_draws, float(config["nominal_interval"])
        )
        if record["interval"] != interval or record["interval_rule"] != interval_rule:
            raise B3CoverageError("control finite-draw interval rule mutation")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise B3CoverageError("duplicate heldout row")
    resolved_ids = {row["row_id"] for row in rows if row["resolved"]}
    consumed: list[str] = []
    for cell in cells:
        ids = cell["row_ids"]
        predictive = cell["joint_posterior_predictive"]
        if ids != predictive["column_row_ids"]:
            raise B3CoverageError("predictive columns reordered or substituted")
        if len(ids) != len(set(ids)) or any(row_id not in resolved_ids for row_id in ids):
            raise B3CoverageError("missing, duplicate, or unresolved predictive row")
        if any(len(draw) != len(ids) for draw in predictive["joint_draws"]):
            raise B3CoverageError("predictive draw column count mismatch")
        if cell["predictive_bytes_sha256"] != _object_hash(predictive):
            raise B3CoverageError("predictive bytes mutation")
        low, high = cell["aggregate_interval"]
        if not (0.0 <= low < high <= 1.0) or (low == 0.0 and high == 1.0):
            raise B3CoverageError("point, invalid, or trivial aggregate interval")
        consumed.extend(ids)
        series_to_rows: dict[str, set[str]] = {}
        for row in rows:
            if row["row_id"] in ids:
                series_to_rows.setdefault(row["series_id"], set()).add(row["row_id"])
        for series_id, member_ids in series_to_rows.items():
            all_series_ids = {
                row["row_id"]
                for row in rows
                if row["resolved"] and row["series_id"] == series_id
            }
            if member_ids != all_series_ids:
                raise B3CoverageError("split atomic series")
    if sorted(consumed) != sorted(resolved_ids):
        raise B3CoverageError("heldout row-to-cell reconciliation failed")
    if dependence["map_resampling_allowed"] or dependence["naive_series_iid_allowed"]:
        raise B3CoverageError("map or naive IID authority substitution")
    required_fields = {
        "series_id",
        "identity_component_id",
        "tournament_time_id",
        "patch_shock_id",
    }
    level_fields = {level["field"] for level in dependence["levels"]}
    if level_fields != required_fields:
        raise B3CoverageError("dependence dimension collapse")
    replay_dependence = _dependence(rows, cells, config)
    if dependence != replay_dependence:
        raise B3CoverageError("dependence support, ESS, or sensitivity mutation")
    if dependence["top_level_support"]["status"] != "available":
        raise B3CoverageError("unavailable_dependence_support")
    if any(
        row["observed_outcome"] is not None
        or row["resolved_at"] is not None
        or row["outcome_available_at"] is not None
        for row in rows
        if not row["resolved"]
    ):
        raise B3CoverageError("outcome present before resolution")
    groups = dependence["unresolved_sensitivity"]["groups"]
    if any(len(group) < 2 for group in groups.values()):
        raise B3CoverageError("unresolved singleton treatment")
    replay_report = _report(
        config,
        regimes,
        records,
        control_records,
        rows,
        cells,
        dependence,
    )
    if report != replay_report:
        raise B3CoverageError("report self-rehash or result mutation")
    if report["mechanics_status"] != "PASS":
        raise B3CoverageError("synthetic mechanics did not pass")
    if report["public_interval_wording"] != PUBLIC_INTERVAL_WORDING:
        raise B3CoverageError("public interval wording mutation")
    if not report["synthetic_coverage_performance_status"].startswith("unavailable_"):
        raise B3CoverageError("synthetic coverage performance claim exceeds authority")
    _validate_hard_gates(
        report["hard_gates"],
        (
            "sbc_lineage_complete",
            "sbc_every_regime_uniform",
            "faulty_inference_controls_rejected",
            "multiway_dependence_available",
            "unresolved_rows_excluded",
            "aggregate_reconciliation",
            "wording_and_nonpromotion",
        ),
    )


def _authenticate_bundle(
    root: Path,
) -> tuple[Mapping[str, Mapping[str, Any]], str]:
    authority_path = _safe_file(root, AUTHORITY_LOCATOR.as_posix())
    authority, authority_raw = _read_canonical_json(authority_path)
    _validate_hard_gates(
        authority["hard_gates"], ("source_and_authority_closure",)
    )
    payloads: dict[str, Mapping[str, Any]] = {"authority": authority}
    seen_inodes: set[tuple[int, int]] = set()
    for role in ARTIFACT_ROLES:
        reference = authority["artifacts"].get(role)
        if not isinstance(reference, dict):
            raise B3CoverageError(f"missing authority artifact role: {role}")
        path = _safe_file(root, reference["locator"])
        inode = (path.stat().st_dev, path.stat().st_ino)
        if inode in seen_inodes:
            raise B3CoverageError("artifact alias or hardlink")
        seen_inodes.add(inode)
        payload, raw = _read_canonical_json(path)
        if (
            _raw_hash(raw) != reference["raw_sha256"]
            or _object_hash(payload) != reference["object_sha256"]
        ):
            raise B3CoverageError(f"artifact authentication failed: {role}")
        payloads[role] = payload
    expected = build_frozen_payloads(root)
    if authority != expected["authority"]:
        raise B3CoverageError("authority replay mismatch")
    for role in ARTIFACT_ROLES:
        if payloads[role] != expected[role]:
            raise B3CoverageError(f"artifact replay mismatch: {role}")
    _validate_reconciliation(payloads)
    return MappingProxyType(payloads), _raw_hash(authority_raw)


def _authority_api_factory() -> tuple[type, Any, Any, Any]:
    namespace = globals()
    helper_names = (
        "_canonical_bytes",
        "_object_hash",
        "_raw_hash",
        "_deep_freeze",
        "_reject_duplicate_keys",
        "_read_canonical_json",
        "_safe_file",
        "_callable_fingerprint",
        "_regime_universe",
        "_config",
        "_seed",
        "_prior_draw",
        "_simulate_observations",
        "_inference_adapter",
        "_control_inference_adapter",
        "_randomized_rank",
        "_quantile",
        "_finite_draw_interval",
        "_replications",
        "_control_replications",
        "_diagnostics",
        "_analytical_coverage_reference",
        "_heldout_rows",
        "_heldout_cells",
        "_level_metrics",
        "_identity_components",
        "_weighted_coverage_statistic",
        "_dimension_labels",
        "_pigeonhole_distribution",
        "_support_status",
        "_bootstrap_summary",
        "_dependence",
        "_literal_gate",
        "_report",
        "build_frozen_payloads",
        "_validate_regimes",
        "_validate_hard_gates",
        "_validate_reconciliation",
        "_authenticate_bundle",
    )
    helpers = MappingProxyType(
        {name: namespace[name] for name in helper_names}
    )
    dependencies = MappingProxyType(
        {
            name: namespace[name]
            for name in (
            "np",
            "gammaincc",
            "beta_distribution",
            "binom",
            "binomtest",
            "chi2",
            "json",
            "statistics",
            "stat",
            "math",
            "sha256",
        )
        }
    )
    code_objects = MappingProxyType(
        {name: function.__code__ for name, function in helpers.items()}
    )
    defaults = MappingProxyType(
        {
            name: (repr(function.__defaults__), repr(function.__kwdefaults__))
            for name, function in helpers.items()
        }
    )
    registry = (
        tuple(OUTPUT_TYPES),
        tuple(REGIME_KINDS),
        tuple(CONTROL_NAMES),
        tuple(ARTIFACT_ROLES),
        tuple(CLAIM_CEILING),
        tuple(FORBIDDEN_CLAIMS),
        tuple(sorted(AUTHORITY_THREAT_MODEL.items())),
    )
    canonical_root = Path(__file__).resolve().parents[3]

    def assert_integrity() -> None:
        if (
            tuple(namespace.get("OUTPUT_TYPES", ())),
            tuple(namespace.get("REGIME_KINDS", ())),
            tuple(namespace.get("CONTROL_NAMES", ())),
            tuple(namespace.get("ARTIFACT_ROLES", ())),
            tuple(namespace.get("CLAIM_CEILING", ())),
            tuple(namespace.get("FORBIDDEN_CLAIMS", ())),
            tuple(
                sorted(
                    namespace.get("AUTHORITY_THREAT_MODEL", {}).items()
                )
            ),
        ) != registry:
            raise B3CoverageError("frozen registry mutation")
        for name, original in helpers.items():
            current = namespace.get(name)
            if current is not original:
                raise B3CoverageError(f"callable rebound: {name}")
            if current.__code__ is not code_objects[name]:
                raise B3CoverageError(f"callable code/default mutation: {name}")
            if (
                repr(current.__defaults__),
                repr(current.__kwdefaults__),
            ) != defaults[name]:
                raise B3CoverageError(f"callable code/default mutation: {name}")
        for name, original in dependencies.items():
            if namespace.get(name) is not original:
                raise B3CoverageError(f"dependency substitution: {name}")
        if namespace.get("load_b3_coverage_authority") is not loader:
            raise B3CoverageError("public loader rebound")
        if namespace.get("validate_b3_coverage_authority") is not validator:
            raise B3CoverageError("public validator rebound")
        if namespace.get("snapshot_b3_coverage_authority") is not snapshot:
            raise B3CoverageError("public snapshot reader rebound")
        if namespace.get("LoadedB3CoverageAuthority") is not LoadedAuthority:
            raise B3CoverageError("public authority type rebound")
        for name, original in authority_class_objects.items():
            current = LoadedAuthority.__dict__.get(name)
            if current is not original:
                raise B3CoverageError(f"authority class member rebound: {name}")
            function = (
                current.fget
                if isinstance(current, property)
                else getattr(current, "__func__", current)
            )
            if function.__code__ is not authority_class_code_objects[name]:
                raise B3CoverageError(f"authority class code mutation: {name}")
            if (
                repr(function.__defaults__),
                repr(function.__kwdefaults__),
            ) != authority_class_defaults[name]:
                raise B3CoverageError(f"authority class default mutation: {name}")

    def require_exact(authority: Any) -> None:
        if type(authority) is not LoadedAuthority or authority is not capability:
            raise B3CoverageError("authority was not issued by the frozen loader")

    class LoadedAuthority:
        """Opaque stateless singleton; authorization is complete replay, not a token."""

        __slots__ = ()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise TypeError("LoadedB3CoverageAuthority is loader-issued only")

        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise TypeError("LoadedB3CoverageAuthority cannot be subclassed")

        @property
        def project_root(self) -> Path:
            return canonical_root

        @property
        def payloads(self) -> Mapping[str, Mapping[str, Any]]:
            return snapshot(self)["payloads"]

        @property
        def authority_raw_sha256(self) -> str:
            return snapshot(self)["authority_raw_sha256"]

    LoadedAuthority.__name__ = "LoadedB3CoverageAuthority"
    capability = object.__new__(LoadedAuthority)
    authority_class_objects = MappingProxyType(
        {
            name: LoadedAuthority.__dict__[name]
            for name in (
                "__init__",
                "__init_subclass__",
                "project_root",
                "payloads",
                "authority_raw_sha256",
            )
        }
    )
    authority_class_functions = {
        name: (
            value.fget
            if isinstance(value, property)
            else getattr(value, "__func__", value)
        )
        for name, value in authority_class_objects.items()
    }
    authority_class_code_objects = MappingProxyType(
        {
            name: function.__code__
            for name, function in authority_class_functions.items()
        }
    )
    authority_class_defaults = MappingProxyType(
        {
            name: (repr(function.__defaults__), repr(function.__kwdefaults__))
            for name, function in authority_class_functions.items()
        }
    )

    def loader(project_root: Path) -> Any:
        assert_integrity()
        root = project_root.resolve(strict=True)
        if root != canonical_root:
            raise B3CoverageError("detached source root cannot issue B3 authority")
        _authenticate_bundle(root)
        return capability

    def validator(authority: Any) -> None:
        assert_integrity()
        require_exact(authority)
        _authenticate_bundle(canonical_root)

    def snapshot(authority: Any) -> Mapping[str, Any]:
        assert_integrity()
        require_exact(authority)
        payloads, authority_hash = _authenticate_bundle(canonical_root)
        return MappingProxyType(
            {
                "project_root": canonical_root,
                "payloads": _deep_freeze(payloads),
                "authority_raw_sha256": authority_hash,
            }
        )

    return LoadedAuthority, loader, validator, snapshot


(
    LoadedB3CoverageAuthority,
    load_b3_coverage_authority,
    validate_b3_coverage_authority,
    snapshot_b3_coverage_authority,
) = _authority_api_factory()
del _authority_api_factory


def predictive_bytes_hash(cell: Mapping[str, Any]) -> str:
    """Hash only pre-outcome predictive bytes for label-invariance tests."""
    predictive = cell.get("joint_posterior_predictive")
    if not isinstance(predictive, dict):
        raise B3CoverageError("missing joint posterior predictive payload")
    return _object_hash(predictive)


__all__ = [
    "ARTIFACT_ROOT",
    "AUTHORITY_LOCATOR",
    "B3CoverageError",
    "CLAIM_CEILING",
    "FORBIDDEN_CLAIMS",
    "LoadedB3CoverageAuthority",
    "OUTPUT_TYPES",
    "PUBLIC_INTERVAL_WORDING",
    "REGIME_KINDS",
    "build_frozen_payloads",
    "load_b3_coverage_authority",
    "predictive_bytes_hash",
    "snapshot_b3_coverage_authority",
    "validate_b3_coverage_authority",
]
