"""Research-only robust dynamic organization-strength challenger.

This module deliberately does not replace :mod:`lol_kills.ratings.dynamic_bt`.
It reuses that module's validated map ledger, immutable organization keys,
bridge-support graph, chronological ordering, and diagonal-Gaussian update
contract, then changes one assumption: an observed organization can have
either a regular state innovation or a rare, high-variance innovation.

The approximation is a two-component Bayesian innovation mixture:

``regular``
    Continue from the propagated Gaussian state.

``shock``
    Continue from the same mean with additional transition variance.

For each map, the regular/shock combinations for the two organizations are
weighted by their prior hazards.  Their one-step Bernoulli evidence determines
posterior branch weights, each branch receives the ordinary approximate
Bradley--Terry update, and the branches are moment-matched back to one Gaussian
per state.  A surprising result therefore gives more posterior mass to the
high-variance branch and permits a larger update.  This is a bounded
assumed-density/change-point approximation, not exact Bayesian online
changepoint detection.

Scientific guardrails:

* hyperparameters are selected on validation rows only;
* only the selected challenger is evaluated on the frozen final period;
* every exact-timestamp batch is predicted before any result in that batch is
  assimilated;
* supplied immutable organization IDs are never reconstructed from names;
* disconnected competition contexts retain the base model's fail-closed
  uncertainty rather than receiving an inferred bridge offset;
* no series score, series result, draft, gold, kills, duration, roster label,
  patch label, or other post-map feature enters the likelihood;
* side-swapped predictions are exact complements by construction;
* baseline comparison requires an explicit, aligned, pre-outcome ledger and
  discloses each baseline's historical update contract;
* this file never authorizes production promotion or a state-of-the-art claim.

The change-point motivation follows the online hazard/mixture framing of Adams
and MacKay (2007, arXiv:0710.3742).  The implementation is intentionally much
simpler: it retains only a regular and a shock branch and moment-matches after
each observation so it remains auditable at map grain.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lol_kills.model_tournament import (
    corp_calibration_diagnostics,
    proper_score_vector,
)
from lol_kills.ratings.dynamic_bt import (
    BinaryEvaluation,
    DynamicBradleyTerry,
    DynamicBTColumns,
    DynamicBTConfig,
    DynamicBTPrediction,
    GaussianState,
    PreparedDynamicBTData,
    _bounded_antisymmetric_probability,
    _clean_key,
    _plain_logistic,
    _utc_timestamp,
    evaluate_binary_predictions,
    prepare_dynamic_bt_data,
)

RESEARCH_STATUS = "research_challenger_not_for_production_or_sota_claim"
ROBUST_MODEL_ID = "robust_dynamic_bt_raw"
REQUIRED_BASELINE_MODELS = ("dynamic_bt_raw", "dual_elo")


@dataclass(frozen=True)
class RobustDynamicBTConfig:
    """Regular/shock innovation-mixture hyperparameters.

    ``shock_probability`` is a per-observed-map prior hazard for each
    participating organization after ``minimum_team_observations``.  The shock
    branch adds ``shock_variance`` in log-odds-squared units, subject to the
    base filter's variance cap.  Setting either value to zero exactly recovers
    the existing raw DynamicBT path.
    """

    base_config: DynamicBTConfig = field(default_factory=DynamicBTConfig)
    shock_probability: float = 0.025
    shock_variance: float = 0.75
    minimum_team_observations: int = 4

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.shock_probability)
            or not 0.0 <= self.shock_probability < 1.0
        ):
            raise ValueError("shock_probability must be finite in [0, 1)")
        if not math.isfinite(self.shock_variance) or self.shock_variance < 0.0:
            raise ValueError("shock_variance must be finite and nonnegative")
        if self.minimum_team_observations < 0:
            raise ValueError("minimum_team_observations cannot be negative")

    @property
    def shock_enabled(self) -> bool:
        return bool(self.shock_probability > 0.0 and self.shock_variance > 0.0)


@dataclass(frozen=True)
class RobustHyperparameterCandidate:
    """One validation-only challenger configuration.

    Lower ``complexity_rank`` is simpler and wins only when the proper-score
    criteria are tied within the tournament's explicit numerical tolerance.
    """

    name: str
    config: RobustDynamicBTConfig
    complexity_rank: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name must be non-empty")
        if self.complexity_rank < 0:
            raise ValueError("complexity_rank cannot be negative")


@dataclass(frozen=True)
class _InnovationBranch:
    blue_shock: bool
    red_shock: bool
    prior_weight: float
    blue_variance_increment: float
    red_variance_increment: float

    @property
    def any_shock(self) -> bool:
        return self.blue_shock or self.red_shock


@dataclass(frozen=True)
class RobustDynamicBTPrediction:
    probability: float
    latent_logit: float
    predictive_variance: float
    base_predictive_variance: float
    predictive_sigma: float
    blue_team_mean: float
    red_team_mean: float
    blue_team_variance: float
    red_team_variance: float
    blue_side_logit: float
    blue_side_variance: float
    blue_context_mean: float
    red_context_mean: float
    bridge_status: str
    bridge_component: str | None
    bridge_direct_maps: int
    blue_shock_prior: float
    red_shock_prior: float
    any_shock_prior: float
    innovation_components: int


@dataclass(frozen=True)
class RobustUpdateDiagnostic:
    prior_any_shock: float
    posterior_any_shock: float
    assimilation_evidence: float
    innovation_components: int
    used_for_current_prediction: bool = False


@dataclass(frozen=True)
class RobustPrequentialRun:
    predictions: pd.DataFrame
    update_diagnostics: pd.DataFrame
    exclusions: pd.DataFrame
    audit: Mapping[str, Any]
    model: RobustDynamicBradleyTerry


@dataclass(frozen=True)
class RobustTournamentResult:
    """Frozen validation selection and final-period benchmark evidence."""

    selected_candidate: str
    selected_config: RobustDynamicBTConfig
    validation_scores: pd.DataFrame
    validation_evaluation: BinaryEvaluation
    test_evaluation: BinaryEvaluation
    test_scores: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    test_comparison_ledger: pd.DataFrame
    update_diagnostics: pd.DataFrame
    exclusions: pd.DataFrame
    audit: Mapping[str, Any]

    def paired_block_comparison(
        self,
        baseline_model_id: str,
        *,
        cluster_ids: Mapping[str, str] | pd.Series | Sequence[str],
        score: str = "log_loss",
        bootstrap_replicates: int = 1000,
        moving_block_size: int = 8,
        alpha: float = 0.05,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Return paired temporal-cluster uncertainty for one raw baseline.

        Cluster IDs are required explicitly.  They may be canonical series IDs
        or another predeclared dependence cluster, but this method never
        infers series boundaries from map order.
        """

        return paired_cluster_block_comparison(
            self.test_comparison_ledger,
            baseline_model_id=baseline_model_id,
            cluster_ids=cluster_ids,
            score=score,
            bootstrap_replicates=bootstrap_replicates,
            moving_block_size=moving_block_size,
            alpha=alpha,
            random_seed=random_seed,
        )


def _candidate_fingerprint(
    candidate: RobustHyperparameterCandidate,
    *,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    data_cutoff: str | None,
) -> str:
    payload = {
        "candidate": candidate.name,
        "config": asdict(candidate.config),
        "complexity_rank": candidate.complexity_rank,
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "data_cutoff": data_cutoff,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _innovation_branches(
    *,
    blue_probability: float,
    red_probability: float,
    blue_variance_increment: float,
    red_variance_increment: float,
) -> tuple[_InnovationBranch, ...]:
    blue_options = (
        ((False, 1.0),)
        if blue_probability <= 0.0 or blue_variance_increment <= 0.0
        else (
            (False, 1.0 - blue_probability),
            (True, blue_probability),
        )
    )
    red_options = (
        ((False, 1.0),)
        if red_probability <= 0.0 or red_variance_increment <= 0.0
        else (
            (False, 1.0 - red_probability),
            (True, red_probability),
        )
    )
    branches = tuple(
        _InnovationBranch(
            blue_shock=blue_shock,
            red_shock=red_shock,
            prior_weight=float(blue_weight * red_weight),
            blue_variance_increment=(blue_variance_increment if blue_shock else 0.0),
            red_variance_increment=(red_variance_increment if red_shock else 0.0),
        )
        for blue_shock, blue_weight in blue_options
        for red_shock, red_weight in red_options
    )
    total = math.fsum(branch.prior_weight for branch in branches)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-15):
        branches = tuple(
            _InnovationBranch(
                blue_shock=branch.blue_shock,
                red_shock=branch.red_shock,
                prior_weight=branch.prior_weight / total,
                blue_variance_increment=branch.blue_variance_increment,
                red_variance_increment=branch.red_variance_increment,
            )
            for branch in branches
        )
    return branches


def _mixture_probability(
    latent_logit: float,
    *,
    base_predictive_variance: float,
    branches: Sequence[_InnovationBranch],
    probability_floor: float,
) -> tuple[float, float]:
    """Integrate innovation branches with exact sign complementarity."""

    # Sorting by a side-invariant key makes the floating-point accumulation
    # identical after blue/red are swapped.
    components = sorted(
        (
            (
                float(
                    base_predictive_variance
                    + branch.blue_variance_increment
                    + branch.red_variance_increment
                ),
                float(branch.prior_weight),
            )
            for branch in branches
        ),
        key=lambda item: (item[0], item[1]),
    )
    magnitude = abs(float(latent_logit))
    positive_probability = math.fsum(
        weight
        * _bounded_antisymmetric_probability(
            magnitude,
            variance,
            probability_floor,
        )
        for variance, weight in components
    )
    mean_variance = math.fsum(variance * weight for variance, weight in components)
    if latent_logit > 0.0:
        probability = positive_probability
    elif latent_logit < 0.0:
        probability = 1.0 - positive_probability
    else:
        probability = 0.5
    return probability, mean_variance


class RobustDynamicBradleyTerry(DynamicBradleyTerry):
    """DynamicBT with a moment-matched regular/shock innovation mixture."""

    def __init__(self, config: RobustDynamicBTConfig | None = None) -> None:
        self.robust_config = config or RobustDynamicBTConfig()
        super().__init__(self.robust_config.base_config)
        self.posterior_shock_sum = 0.0
        self.shock_updates = 0

    def _shock_prior(self, state: GaussianState) -> float:
        if (
            not self.robust_config.shock_enabled
            or state.observations < self.robust_config.minimum_team_observations
        ):
            return 0.0
        return self.robust_config.shock_probability

    def _shock_increment(self, state: GaussianState) -> float:
        return max(
            min(
                self.config.max_team_variance,
                state.variance + self.robust_config.shock_variance,
            )
            - state.variance,
            0.0,
        )

    def _branches(
        self,
        blue: GaussianState,
        red: GaussianState,
    ) -> tuple[tuple[_InnovationBranch, ...], float, float]:
        blue_prior = self._shock_prior(blue)
        red_prior = self._shock_prior(red)
        blue_increment = self._shock_increment(blue) if blue_prior > 0.0 else 0.0
        red_increment = self._shock_increment(red) if red_prior > 0.0 else 0.0
        if blue_increment <= 0.0:
            blue_prior = 0.0
        if red_increment <= 0.0:
            red_prior = 0.0
        return (
            _innovation_branches(
                blue_probability=blue_prior,
                red_probability=red_prior,
                blue_variance_increment=blue_increment,
                red_variance_increment=red_increment,
            ),
            blue_prior,
            red_prior,
        )

    def predict(
        self,
        blue_team_key: str,
        red_team_key: str,
        *,
        timestamp: Any,
        blue_context: str = "",
        red_context: str = "",
        side_indicator: float = 1.0,
    ) -> RobustDynamicBTPrediction:
        base: DynamicBTPrediction = super().predict(
            blue_team_key,
            red_team_key,
            timestamp=timestamp,
            blue_context=blue_context,
            red_context=red_context,
            side_indicator=side_indicator,
        )
        blue_key = _clean_key(blue_team_key)
        red_key = _clean_key(red_team_key)
        blue = self.teams[blue_key]
        red = self.teams[red_key]
        branches, blue_prior, red_prior = self._branches(blue, red)
        probability, mean_variance = _mixture_probability(
            base.latent_logit,
            base_predictive_variance=base.predictive_variance,
            branches=branches,
            probability_floor=self.config.probability_floor,
        )
        any_prior = math.fsum(
            branch.prior_weight for branch in branches if branch.any_shock
        )
        return RobustDynamicBTPrediction(
            probability=probability,
            latent_logit=base.latent_logit,
            predictive_variance=mean_variance,
            base_predictive_variance=base.predictive_variance,
            predictive_sigma=math.sqrt(mean_variance),
            blue_team_mean=base.blue_team_mean,
            red_team_mean=base.red_team_mean,
            blue_team_variance=base.blue_team_variance,
            red_team_variance=base.red_team_variance,
            blue_side_logit=base.blue_side_logit,
            blue_side_variance=base.blue_side_variance,
            blue_context_mean=base.blue_context_mean,
            red_context_mean=base.red_context_mean,
            bridge_status=base.bridge_status,
            bridge_component=base.bridge_component,
            bridge_direct_maps=base.bridge_direct_maps,
            blue_shock_prior=blue_prior,
            red_shock_prior=red_prior,
            any_shock_prior=any_prior,
            innovation_components=len(branches),
        )

    def observe(
        self,
        blue_team_key: str,
        red_team_key: str,
        outcome: float,
        *,
        timestamp: Any,
        blue_context: str = "",
        red_context: str = "",
        side_indicator: float = 1.0,
    ) -> RobustUpdateDiagnostic:
        if float(outcome) not in (0.0, 1.0):
            raise ValueError("outcome must be binary")
        moment = _utc_timestamp(timestamp)
        if moment is None:
            raise ValueError("observation timestamp is invalid")
        blue_team_key = _clean_key(blue_team_key)
        red_team_key = _clean_key(red_team_key)
        if not blue_team_key or not red_team_key:
            raise ValueError("observation requires immutable team keys")
        if blue_team_key == red_team_key:
            raise ValueError("observation requires two different team keys")
        if float(side_indicator) not in (-1.0, 1.0):
            raise ValueError("side_indicator must be +1 or -1")
        blue_context = _clean_key(blue_context)
        red_context = _clean_key(red_context)

        blue = self._team(blue_team_key, moment)
        red = self._team(red_team_key, moment)
        branches, _, _ = self._branches(blue, red)
        prior_any = math.fsum(
            branch.prior_weight for branch in branches if branch.any_shock
        )
        if len(branches) == 1:
            super().observe(
                blue_team_key,
                red_team_key,
                outcome,
                timestamp=moment,
                blue_context=blue_context,
                red_context=red_context,
                side_indicator=side_indicator,
            )
            return RobustUpdateDiagnostic(
                prior_any_shock=0.0,
                posterior_any_shock=0.0,
                assimilation_evidence=math.nan,
                innovation_components=1,
            )

        self._propagate_side(moment)
        bridge = self.bridges.support(blue_context, red_context)
        features: list[tuple[str, GaussianState, float, float]] = [
            (
                "blue_team",
                blue,
                1.0,
                self.config.max_team_variance,
            ),
            (
                "red_team",
                red,
                -1.0,
                self.config.max_team_variance,
            ),
            (
                "blue_side",
                self.blue_side,
                float(side_indicator),
                self.config.max_side_variance,
            ),
        ]
        unsupported_variance = 0.0
        if bridge.status == "supported":
            features.extend(
                (
                    (
                        "blue_context",
                        self._context(blue_context, moment),
                        1.0,
                        self.config.max_context_variance,
                    ),
                    (
                        "red_context",
                        self._context(red_context, moment),
                        -1.0,
                        self.config.max_context_variance,
                    ),
                )
            )
        elif bridge.status == "unsupported":
            unsupported_variance = self.config.unsupported_bridge_variance

        latent_logit = math.fsum(
            coefficient * state.mean for _, state, coefficient, _ in features
        )
        latent_probability = _plain_logistic(latent_logit)
        curvature = max(
            latent_probability * (1.0 - latent_probability),
            1e-9,
        )

        branch_rows: list[
            tuple[
                _InnovationBranch,
                float,
                dict[str, tuple[float, float]],
            ]
        ] = []
        evidence_terms: list[float] = []
        for branch in branches:
            branch_variances = {
                name: (
                    state.variance
                    + (
                        branch.blue_variance_increment
                        if name == "blue_team"
                        else branch.red_variance_increment
                        if name == "red_team"
                        else 0.0
                    )
                )
                for name, state, _, _ in features
            }
            linear_variance = math.fsum(
                coefficient * coefficient * branch_variances[name]
                for name, _, coefficient, _ in features
            )
            branch_probability = _bounded_antisymmetric_probability(
                latent_logit,
                linear_variance + unsupported_variance,
                self.config.probability_floor,
            )
            likelihood = (
                branch_probability
                if float(outcome) == 1.0
                else 1.0 - branch_probability
            )
            evidence_term = branch.prior_weight * likelihood
            evidence_terms.append(evidence_term)
            denominator = 1.0 + curvature * linear_variance
            residual = float(outcome) - latent_probability
            updates: dict[str, tuple[float, float]] = {}
            for name, state, coefficient, max_variance in features:
                old_variance = branch_variances[name]
                gain = old_variance * coefficient / denominator
                new_mean = state.mean + gain * residual
                new_variance = old_variance - (
                    curvature * (old_variance * coefficient) ** 2 / denominator
                )
                updates[name] = (
                    min(
                        self.config.max_abs_mean,
                        max(-self.config.max_abs_mean, new_mean),
                    ),
                    min(
                        max_variance,
                        max(self.config.min_variance, new_variance),
                    ),
                )
            branch_rows.append((branch, evidence_term, updates))

        evidence = math.fsum(evidence_terms)
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise RuntimeError("shock-mixture update has invalid evidence")
        posterior_weights = [
            evidence_term / evidence for _, evidence_term, _ in branch_rows
        ]

        for name, state, _, max_variance in features:
            component_means = [updates[name][0] for _, _, updates in branch_rows]
            component_variances = [updates[name][1] for _, _, updates in branch_rows]
            matched_mean = math.fsum(
                weight * mean
                for weight, mean in zip(posterior_weights, component_means)
            )
            matched_variance = math.fsum(
                weight * (variance + (mean - matched_mean) * (mean - matched_mean))
                for weight, mean, variance in zip(
                    posterior_weights,
                    component_means,
                    component_variances,
                )
            )
            state.mean = min(
                self.config.max_abs_mean,
                max(-self.config.max_abs_mean, matched_mean),
            )
            state.variance = min(
                max_variance,
                max(self.config.min_variance, matched_variance),
            )
            state.last_observed = moment
            state.observations += 1

        posterior_any = math.fsum(
            weight
            for weight, (branch, _, _) in zip(posterior_weights, branch_rows)
            if branch.any_shock
        )
        self.observed_maps += 1
        self.posterior_shock_sum += posterior_any
        self.shock_updates += 1
        return RobustUpdateDiagnostic(
            prior_any_shock=prior_any,
            posterior_any_shock=posterior_any,
            assimilation_evidence=evidence,
            innovation_components=len(branches),
        )

    def audit_snapshot(self) -> dict[str, Any]:
        base = super().audit_snapshot()
        return {
            **base,
            "status": RESEARCH_STATUS,
            "robust_config": asdict(self.robust_config),
            "innovation_model": (
                "per-organization regular/shock Gaussian transition mixture; "
                "Bernoulli-evidence weighting; posterior moment matching"
            ),
            "shock_updates": int(self.shock_updates),
            "mean_posterior_any_shock": (
                float(self.posterior_shock_sum / self.shock_updates)
                if self.shock_updates
                else None
            ),
            "limitations": [
                "two-branch assumed-density approximation, not exact BOCPD",
                "shock evidence comes only from binary map outcomes",
                "organization state cannot identify which roster member changed",
                "patch and roster labels are intentionally absent from the likelihood",
            ],
        }


def _prediction_record(
    row: pd.Series,
    prediction: RobustDynamicBTPrediction,
) -> dict[str, Any]:
    return {
        "row_id": str(row["row_id"]),
        "timestamp": row["timestamp"],
        "y_true": float(row["y_true"]),
        "blue_team_key": row["blue_team_key"],
        "red_team_key": row["red_team_key"],
        "blue_context": row["blue_context"],
        "red_context": row["red_context"],
        "competition": row["competition"],
        "side_indicator": float(row["side_indicator"]),
        "p_blue": prediction.probability,
        "latent_logit": prediction.latent_logit,
        "predictive_variance": prediction.predictive_variance,
        "base_predictive_variance": prediction.base_predictive_variance,
        "predictive_sigma": prediction.predictive_sigma,
        "blue_team_mean": prediction.blue_team_mean,
        "red_team_mean": prediction.red_team_mean,
        "blue_team_variance": prediction.blue_team_variance,
        "red_team_variance": prediction.red_team_variance,
        "blue_side_logit": prediction.blue_side_logit,
        "blue_side_variance": prediction.blue_side_variance,
        "blue_context_mean": prediction.blue_context_mean,
        "red_context_mean": prediction.red_context_mean,
        "bridge_status": prediction.bridge_status,
        "bridge_component": prediction.bridge_component,
        "bridge_direct_maps": prediction.bridge_direct_maps,
        "blue_shock_prior": prediction.blue_shock_prior,
        "red_shock_prior": prediction.red_shock_prior,
        "any_shock_prior": prediction.any_shock_prior,
        "innovation_components": prediction.innovation_components,
        "prediction_before_outcome": True,
        "uses_same_event_post_map_features": False,
        "historical_update_contract": "binary_map_outcome_only",
    }


def _run_prepared_robust(
    prepared: PreparedDynamicBTData,
    config: RobustDynamicBTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustDynamicBradleyTerry]:
    model = RobustDynamicBradleyTerry(config)
    prediction_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    if prepared.frame.empty:
        return pd.DataFrame(), pd.DataFrame(), model

    ordered = prepared.frame.sort_values(["timestamp", "row_id"], kind="mergesort")
    for timestamp, batch in ordered.groupby("timestamp", sort=True):
        batch = batch.sort_values("row_id", kind="mergesort")
        for _, row in batch.iterrows():
            prediction = model.predict(
                row["blue_team_key"],
                row["red_team_key"],
                timestamp=timestamp,
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                side_indicator=float(row["side_indicator"]),
            )
            prediction_rows.append(_prediction_record(row, prediction))

        # Match the existing DynamicBT protocol: structural bridge evidence is
        # registered only after every prediction in the exact-time batch.
        for _, row in batch.iterrows():
            model.register_bridge_evidence(
                blue_team_key=row["blue_team_key"],
                red_team_key=row["red_team_key"],
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                competition=row["competition"],
            )
        for _, row in batch.iterrows():
            diagnostic = model.observe(
                row["blue_team_key"],
                row["red_team_key"],
                float(row["y_true"]),
                timestamp=timestamp,
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                side_indicator=float(row["side_indicator"]),
            )
            update_rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "timestamp": timestamp,
                    "prior_any_shock": diagnostic.prior_any_shock,
                    "posterior_any_shock": (diagnostic.posterior_any_shock),
                    "assimilation_evidence": (diagnostic.assimilation_evidence),
                    "innovation_components": (diagnostic.innovation_components),
                    "used_for_current_prediction": (
                        diagnostic.used_for_current_prediction
                    ),
                }
            )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(update_rows),
        model,
    )


def run_prequential_robust_dynamic_bt(
    maps: pd.DataFrame,
    *,
    config: RobustDynamicBTConfig | None = None,
    columns: DynamicBTColumns | None = None,
    data_cutoff: Any = None,
) -> RobustPrequentialRun:
    """Run the robust challenger at chronological binary-map grain."""

    selected_config = config or RobustDynamicBTConfig()
    prepared = prepare_dynamic_bt_data(
        maps,
        columns=columns,
        data_cutoff=data_cutoff,
    )
    predictions, updates, model = _run_prepared_robust(prepared, selected_config)
    audit = {
        **dict(prepared.audit),
        "status": RESEARCH_STATUS,
        "prediction_protocol": (
            "chronological online/prequential; every exact-timestamp batch "
            "is predicted before any outcome in that batch is assimilated"
        ),
        "estimand": "pre-map probability that supplied blue organization wins",
        "permitted_predictors": [
            "immutable blue organization state",
            "immutable red organization state",
            "side intercept",
            "historically identified context bridge offset",
            "elapsed time through the base state transition",
        ],
        "forbidden_predictors": [
            "series score or scheduled format",
            "same-map or future result",
            "gold, kills, duration, draft, or objectives",
            "post-map roster, patch, or tournament outcome features",
        ],
        "model": model.audit_snapshot(),
        "predictive_variance_summary": (
            "prior-weighted mean component variance for reporting only; "
            "the probability integrates mixture components separately"
        ),
    }
    return RobustPrequentialRun(
        predictions=predictions,
        update_diagnostics=updates,
        exclusions=prepared.exclusions,
        audit=audit,
        model=model,
    )


def default_robust_candidates(
    base_config: DynamicBTConfig | None = None,
) -> tuple[RobustHyperparameterCandidate, ...]:
    """Small predeclared shock grid, including the simpler no-shock control."""

    base = base_config or DynamicBTConfig()
    return (
        RobustHyperparameterCandidate(
            "gaussian_no_shock",
            RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.0,
                shock_variance=0.0,
            ),
            complexity_rank=0,
        ),
        RobustHyperparameterCandidate(
            "rare_moderate_shock",
            RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.015,
                shock_variance=0.50,
            ),
            complexity_rank=1,
        ),
        RobustHyperparameterCandidate(
            "rare_large_shock",
            RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.025,
                shock_variance=1.25,
            ),
            complexity_rank=2,
        ),
        RobustHyperparameterCandidate(
            "frequent_large_shock",
            RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.075,
                shock_variance=1.25,
            ),
            complexity_rank=3,
        ),
    )


def _calibration_summary(evaluation: BinaryEvaluation) -> dict[str, float]:
    if evaluation.n == 0:
        return {
            "ece": math.nan,
            "mean_probability_minus_rate": math.nan,
        }
    inputs = evaluation.calibration_inputs
    bins = evaluation.calibration_bins
    ece = float(
        (bins["n"] * (bins["mean_probability"] - bins["observed_rate"]).abs()).sum()
        / evaluation.n
    )
    return {
        "ece": ece,
        "mean_probability_minus_rate": float(
            inputs["probability"].mean() - inputs["y_true"].mean()
        ),
    }


def _score_row(
    model_id: str,
    frame: pd.DataFrame,
    *,
    probability_column: str,
    calibration_bins: int,
) -> tuple[dict[str, Any], BinaryEvaluation]:
    evaluation = evaluate_binary_predictions(
        frame,
        probability_column=probability_column,
        calibration_bins=calibration_bins,
    )
    calibration = _calibration_summary(evaluation)
    pav = corp_calibration_diagnostics(
        frame["y_true"].to_numpy(float),
        frame[probability_column].to_numpy(float),
    )
    return (
        {
            "model_id": model_id,
            "n": evaluation.n,
            "log_loss": evaluation.log_loss,
            "brier": evaluation.brier,
            "ece": calibration["ece"],
            "mean_probability_minus_rate": (calibration["mean_probability_minus_rate"]),
            "pav_in_sample_miscalibration": float(pav["miscalibration"]),
            "calibration_is_descriptive_only": True,
        },
        evaluation,
    )


def _select_validation_candidate(
    scores: pd.DataFrame,
    *,
    tie_tolerance: float,
) -> str:
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("selection tie_tolerance must be nonnegative")
    minimum_log_loss = float(scores["validation_log_loss"].min())
    log_tied = scores.loc[
        scores["validation_log_loss"] <= minimum_log_loss + tie_tolerance
    ]
    minimum_brier = float(log_tied["validation_brier"].min())
    proper_score_tied = log_tied.loc[
        log_tied["validation_brier"] <= minimum_brier + tie_tolerance
    ]
    selected = proper_score_tied.sort_values(
        ["complexity_rank", "candidate"],
        kind="mergesort",
    ).iloc[0]
    return str(selected["candidate"])


def _validate_baseline_ledger(
    baseline_ledger: pd.DataFrame,
    *,
    candidate_test: pd.DataFrame,
    required_model_ids: Sequence[str],
) -> pd.DataFrame:
    required = {
        "model_id",
        "row_id",
        "timestamp",
        "y_true",
        "probability",
        "prediction_before_outcome",
        "uses_same_event_post_map_features",
        "historical_update_contract",
    }
    if baseline_ledger is None or baseline_ledger.empty:
        raise ValueError("raw baseline ledger is required")
    missing = sorted(required - set(baseline_ledger.columns))
    if missing:
        raise ValueError(f"baseline ledger missing columns: {missing}")
    ledger = baseline_ledger.copy()
    ledger["model_id"] = ledger["model_id"].astype(str).str.strip()
    ledger["row_id"] = ledger["row_id"].astype(str).str.strip()
    if ledger[["model_id", "row_id"]].eq("").any().any():
        raise ValueError("baseline model_id and row_id must be non-empty")
    if ledger.duplicated(["model_id", "row_id"]).any():
        raise ValueError("baseline ledger has duplicate model/row predictions")
    if not ledger["prediction_before_outcome"].eq(True).all():
        raise ValueError("all baseline rows must declare pre-outcome prediction")
    if not ledger["uses_same_event_post_map_features"].eq(False).all():
        raise ValueError(
            "baseline rows using same-event post-map features are inadmissible"
        )
    historical_contract = ledger["historical_update_contract"].astype(str).str.strip()
    if historical_contract.eq("").any():
        raise ValueError("baseline historical_update_contract must be non-empty")

    requested = tuple(str(model_id) for model_id in required_model_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("required baseline model IDs must be unique")
    missing_models = sorted(set(requested) - set(ledger["model_id"]))
    if missing_models:
        raise ValueError(f"required raw baseline models are absent: {missing_models}")

    candidate_ids = candidate_test["row_id"].astype(str).tolist()
    candidate_id_set = set(candidate_ids)
    candidate_outcomes = dict(
        zip(
            candidate_test["row_id"].astype(str),
            candidate_test["y_true"].astype(float),
        )
    )
    candidate_timestamps = dict(
        zip(
            candidate_test["row_id"].astype(str),
            pd.to_datetime(candidate_test["timestamp"], errors="raise", utc=True),
        )
    )
    rows: list[pd.DataFrame] = []
    for model_id in requested:
        model = ledger.loc[ledger["model_id"].eq(model_id)].copy()
        if set(model["row_id"]) != candidate_id_set or len(model) != len(candidate_ids):
            raise ValueError(
                f"baseline {model_id!r} must cover exactly the frozen test rows"
            )
        model = pd.DataFrame({"row_id": candidate_ids}).merge(
            model, on="row_id", how="left", validate="one_to_one"
        )
        outcome = pd.to_numeric(model["y_true"], errors="coerce")
        probability = pd.to_numeric(model["probability"], errors="coerce")
        timestamp = pd.to_datetime(model["timestamp"], errors="coerce", utc=True)
        expected_outcome = np.array(
            [candidate_outcomes[row_id] for row_id in candidate_ids],
            dtype=float,
        )
        expected_timestamp = pd.DatetimeIndex(
            [candidate_timestamps[row_id] for row_id in candidate_ids]
        )
        if outcome.isna().any() or not np.array_equal(
            outcome.to_numpy(float), expected_outcome
        ):
            raise ValueError(
                f"baseline {model_id!r} outcomes do not match frozen test labels"
            )
        if timestamp.isna().any() or not np.array_equal(
            timestamp.to_numpy(), expected_timestamp.to_numpy()
        ):
            raise ValueError(
                f"baseline {model_id!r} timestamps do not match test events"
            )
        if (
            probability.isna().any()
            or not np.isfinite(probability.to_numpy(float)).all()
            or np.any(
                (probability.to_numpy(float) <= 0.0)
                | (probability.to_numpy(float) >= 1.0)
            )
        ):
            raise ValueError(
                f"baseline {model_id!r} probabilities must be within (0, 1)"
            )
        model["probability"] = probability.to_numpy(float)
        model["y_true"] = outcome.to_numpy(float)
        model["timestamp"] = timestamp
        rows.append(
            model[
                [
                    "model_id",
                    "row_id",
                    "timestamp",
                    "y_true",
                    "probability",
                    "prediction_before_outcome",
                    "uses_same_event_post_map_features",
                    "historical_update_contract",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _comparison_ledger(
    candidate_test: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    candidate = pd.DataFrame(
        {
            "model_id": ROBUST_MODEL_ID,
            "row_id": candidate_test["row_id"].astype(str),
            "timestamp": pd.to_datetime(
                candidate_test["timestamp"], errors="raise", utc=True
            ),
            "y_true": candidate_test["y_true"].astype(float),
            "probability": candidate_test["p_blue"].astype(float),
            "prediction_before_outcome": True,
            "uses_same_event_post_map_features": False,
            "historical_update_contract": "binary_map_outcome_only",
        }
    )
    return pd.concat([candidate, baselines], ignore_index=True)


def run_robust_dynamic_bt_tournament(
    maps: pd.DataFrame,
    *,
    validation_start: Any,
    test_start: Any,
    baseline_ledger: pd.DataFrame,
    candidates: Sequence[RobustHyperparameterCandidate] | None = None,
    required_baseline_models: Sequence[str] = REQUIRED_BASELINE_MODELS,
    columns: DynamicBTColumns | None = None,
    data_cutoff: Any = None,
    calibration_bins: int = 10,
    selection_tie_tolerance: float = 1e-12,
) -> RobustTournamentResult:
    """Select the robust challenger on validation, then benchmark final test.

    The supplied baseline ledger is an explicit seam: this module validates
    event alignment, outcomes, timestamps, probability bounds, and declarations
    that predictions were pre-outcome and used no post-map features.  It does
    not recreate or silently repair baseline predictions.
    """

    validation_at = _utc_timestamp(validation_start)
    test_at = _utc_timestamp(test_start)
    if validation_at is None or test_at is None:
        raise ValueError("validation_start and test_start must be valid")
    if validation_at >= test_at:
        raise ValueError("validation_start must precede test_start")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")

    candidate_grid = tuple(candidates or default_robust_candidates())
    if not candidate_grid:
        raise ValueError("at least one robust candidate is required")
    names = [candidate.name for candidate in candidate_grid]
    if len(set(names)) != len(names):
        raise ValueError("robust candidate names must be unique")

    prepared = prepare_dynamic_bt_data(maps, columns=columns, data_cutoff=data_cutoff)
    train = prepared.frame.loc[prepared.frame["timestamp"] < validation_at]
    validation = prepared.frame.loc[
        (prepared.frame["timestamp"] >= validation_at)
        & (prepared.frame["timestamp"] < test_at)
    ]
    test = prepared.frame.loc[prepared.frame["timestamp"] >= test_at]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "tournament requires non-empty train, validation, and test rows"
        )

    selection_prepared = PreparedDynamicBTData(
        frame=pd.concat([train, validation], ignore_index=True),
        exclusions=prepared.exclusions.iloc[0:0].copy(),
        audit=prepared.audit,
    )
    validation_rows: list[dict[str, Any]] = []
    for candidate in candidate_grid:
        predictions, _, _ = _run_prepared_robust(selection_prepared, candidate.config)
        candidate_validation = predictions.loc[
            (predictions["timestamp"] >= validation_at)
            & (predictions["timestamp"] < test_at)
        ].reset_index(drop=True)
        evaluation = evaluate_binary_predictions(
            candidate_validation,
            probability_column="p_blue",
            calibration_bins=calibration_bins,
        )
        calibration = _calibration_summary(evaluation)
        validation_rows.append(
            {
                "candidate": candidate.name,
                "complexity_rank": candidate.complexity_rank,
                "validation_n": evaluation.n,
                "validation_log_loss": evaluation.log_loss,
                "validation_brier": evaluation.brier,
                "validation_ece": calibration["ece"],
            }
        )
    validation_scores = pd.DataFrame(validation_rows)
    selected_name = _select_validation_candidate(
        validation_scores,
        tie_tolerance=selection_tie_tolerance,
    )
    validation_scores["selected"] = validation_scores["candidate"].eq(selected_name)
    validation_scores = validation_scores.sort_values(
        [
            "validation_log_loss",
            "validation_brier",
            "complexity_rank",
            "candidate",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = next(
        candidate for candidate in candidate_grid if candidate.name == selected_name
    )

    predictions, updates, final_model = _run_prepared_robust(prepared, selected.config)
    selected_validation = predictions.loc[
        (predictions["timestamp"] >= validation_at)
        & (predictions["timestamp"] < test_at)
    ].reset_index(drop=True)
    selected_test = predictions.loc[predictions["timestamp"] >= test_at].reset_index(
        drop=True
    )
    validation_evaluation = evaluate_binary_predictions(
        selected_validation,
        probability_column="p_blue",
        calibration_bins=calibration_bins,
    )
    candidate_score, test_evaluation = _score_row(
        ROBUST_MODEL_ID,
        selected_test,
        probability_column="p_blue",
        calibration_bins=calibration_bins,
    )

    baselines = _validate_baseline_ledger(
        baseline_ledger,
        candidate_test=selected_test,
        required_model_ids=required_baseline_models,
    )
    comparison = _comparison_ledger(selected_test, baselines)
    score_rows = [candidate_score]
    for model_id in required_baseline_models:
        baseline = comparison.loc[comparison["model_id"].eq(model_id)].rename(
            columns={"probability": "_baseline_probability"}
        )
        score, _ = _score_row(
            str(model_id),
            baseline,
            probability_column="_baseline_probability",
            calibration_bins=calibration_bins,
        )
        score_rows.append(score)
    test_scores = pd.DataFrame(score_rows)
    model_order = {
        model_id: index
        for index, model_id in enumerate((ROBUST_MODEL_ID, *required_baseline_models))
    }
    test_scores["_order"] = test_scores["model_id"].map(model_order)
    test_scores = (
        test_scores.sort_values("_order", kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    selected_fingerprint = _candidate_fingerprint(
        selected,
        validation_start=validation_at,
        test_start=test_at,
        data_cutoff=(
            str(prepared.audit.get("data_cutoff"))
            if prepared.audit.get("data_cutoff") is not None
            else None
        ),
    )
    audit: dict[str, Any] = {
        "status": RESEARCH_STATUS,
        "input": dict(prepared.audit),
        "split": {
            "validation_start": validation_at.isoformat(),
            "test_start": test_at.isoformat(),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "selection": {
            "candidate_names": names,
            "criterion": (
                "minimum validation log loss, then validation Brier; "
                "complexity loses ties within the declared tolerance"
            ),
            "tie_tolerance": float(selection_tie_tolerance),
            "selected_candidate": selected_name,
            "selected_fingerprint": selected_fingerprint,
            "test_used_for_selection": False,
            "hyperparameters_frozen_before_test": True,
        },
        "baseline_seam": {
            "required_models": list(required_baseline_models),
            "rows": len(baselines),
            "alignment": "exact frozen-test row ids, labels, and timestamps",
            "pre_outcome_declaration_required": True,
            "same_event_post_map_features_forbidden": True,
            "historical_update_contracts": {
                str(model_id): sorted(
                    set(
                        baselines.loc[
                            baselines["model_id"].eq(model_id),
                            "historical_update_contract",
                        ].astype(str)
                    )
                )
                for model_id in required_baseline_models
            },
            "limitation": (
                "the seam validates supplied declarations and alignment; it "
                "does not independently reconstruct either baseline"
            ),
        },
        "calibration": {
            "metrics": [
                "equal-width ECE",
                "mean probability minus observed rate",
                "in-sample PAV Brier miscalibration diagnostic",
            ],
            "test_fitted_calibration_applied": False,
            "diagnostics_are_descriptive_only": True,
        },
        "uncertainty": {
            "hook": "RobustTournamentResult.paired_block_comparison",
            "cluster_ids_required": True,
            "series_boundaries_inferred": False,
        },
        "test_protocol": (
            "only the validation-selected robust configuration is run through "
            "the final period; each final-period result updates only later "
            "timestamps; no test label selects a hyperparameter"
        ),
        "promotion_authorized": False,
        "sota_claim_authorized": False,
        "final_model": final_model.audit_snapshot(),
    }
    return RobustTournamentResult(
        selected_candidate=selected_name,
        selected_config=selected.config,
        validation_scores=validation_scores,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
        test_scores=test_scores,
        validation_predictions=selected_validation,
        test_predictions=selected_test,
        test_comparison_ledger=comparison,
        update_diagnostics=updates,
        exclusions=prepared.exclusions,
        audit=audit,
    )


def _cluster_values(
    row_ids: pd.Series,
    cluster_ids: Mapping[str, str] | pd.Series | Sequence[str],
) -> np.ndarray:
    ids = row_ids.astype(str)
    if isinstance(cluster_ids, Mapping):
        missing = [row_id for row_id in ids if row_id not in cluster_ids]
        if missing:
            raise ValueError(f"cluster IDs missing test rows: {missing[:5]}")
        values = np.array([str(cluster_ids[row_id]) for row_id in ids], dtype=object)
    elif isinstance(cluster_ids, pd.Series):
        lookup = {str(index): str(value) for index, value in cluster_ids.items()}
        if set(ids).issubset(lookup):
            values = np.array([lookup[row_id] for row_id in ids], dtype=object)
        else:
            values = cluster_ids.astype(str).to_numpy()
    else:
        values = np.asarray(list(cluster_ids), dtype=object)
    if len(values) != len(ids):
        raise ValueError("cluster IDs must align one-for-one with test rows")
    cleaned = np.array([_clean_key(value) for value in values], dtype=object)
    if any(not value for value in cleaned):
        raise ValueError("cluster IDs must be non-empty")
    return cleaned


def paired_cluster_block_comparison(
    comparison_ledger: pd.DataFrame,
    *,
    baseline_model_id: str,
    cluster_ids: Mapping[str, str] | pd.Series | Sequence[str],
    score: str = "log_loss",
    bootstrap_replicates: int = 1000,
    moving_block_size: int = 8,
    alpha: float = 0.05,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Paired event-weighted circular moving-block comparison.

    The candidate-minus-baseline convention means negative values favor the
    robust challenger.  No superiority/promotion rule is embedded here.
    """

    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if moving_block_size < 1:
        raise ValueError("moving_block_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    required = {"model_id", "row_id", "timestamp", "y_true", "probability"}
    missing = sorted(required - set(comparison_ledger.columns))
    if missing:
        raise ValueError(f"comparison ledger missing columns: {missing}")
    selected = comparison_ledger.loc[
        comparison_ledger["model_id"].isin([ROBUST_MODEL_ID, baseline_model_id])
    ].copy()
    if set(selected["model_id"]) != {ROBUST_MODEL_ID, baseline_model_id}:
        raise ValueError("candidate or requested baseline is absent")
    if selected.duplicated(["model_id", "row_id"]).any():
        raise ValueError("comparison ledger has duplicate model/row keys")
    event = selected.pivot(
        index=["row_id", "timestamp", "y_true"],
        columns="model_id",
        values="probability",
    ).reset_index()
    if event[[ROBUST_MODEL_ID, baseline_model_id]].isna().any().any():
        raise ValueError("candidate and baseline must cover identical events")
    event = event.sort_values(["timestamp", "row_id"], kind="mergesort").reset_index(
        drop=True
    )
    event["cluster_id"] = _cluster_values(event["row_id"], cluster_ids)
    candidate_score = proper_score_vector(
        event["y_true"], event[ROBUST_MODEL_ID], score
    )
    baseline_score = proper_score_vector(
        event["y_true"], event[baseline_model_id], score
    )
    event["score_delta"] = candidate_score - baseline_score
    event["_timestamp"] = pd.to_datetime(event["timestamp"], errors="raise", utc=True)
    cluster_order = (
        event.groupby("cluster_id", as_index=False)
        .agg(timestamp=("_timestamp", "min"))
        .sort_values(["timestamp", "cluster_id"], kind="mergesort")["cluster_id"]
        .tolist()
    )
    cluster_deltas = [
        event.loc[event["cluster_id"].eq(cluster_id), "score_delta"].to_numpy(float)
        for cluster_id in cluster_order
    ]
    n_clusters = len(cluster_deltas)
    if n_clusters == 0:
        raise ValueError("at least one dependence cluster is required")
    block_size = min(moving_block_size, n_clusters)
    blocks_needed = (n_clusters + block_size - 1) // block_size
    offsets = np.arange(block_size)
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        starts = rng.integers(0, n_clusters, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n_clusters).ravel()
        sampled = np.concatenate(
            [cluster_deltas[index] for index in indices[:n_clusters]]
        )
        bootstrap[replicate] = float(sampled.mean())
    low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "status": RESEARCH_STATUS,
        "candidate_model_id": ROBUST_MODEL_ID,
        "baseline_model_id": baseline_model_id,
        "score": score,
        "events": len(event),
        "clusters": int(n_clusters),
        "candidate_score": float(candidate_score.mean()),
        "baseline_score": float(baseline_score.mean()),
        "candidate_minus_baseline": float(event["score_delta"].mean()),
        "confidence_level": float(1.0 - alpha),
        "confidence_interval": [float(low), float(high)],
        "bootstrap": {
            "method": (
                "paired circular moving-block bootstrap over explicitly "
                "supplied ordered dependence clusters; event-weighted score"
            ),
            "replicates": int(bootstrap_replicates),
            "block_size_clusters": int(block_size),
            "seed": int(random_seed),
        },
        "candidate_calibration": {
            key: value
            for key, value in corp_calibration_diagnostics(
                event["y_true"], event[ROBUST_MODEL_ID]
            ).items()
            if key != "fitted_probability"
        },
        "baseline_calibration": {
            key: value
            for key, value in corp_calibration_diagnostics(
                event["y_true"], event[baseline_model_id]
            ).items()
            if key != "fitted_probability"
        },
        "promotion_authorized": False,
        "sota_claim_authorized": False,
    }


__all__ = [
    "REQUIRED_BASELINE_MODELS",
    "RESEARCH_STATUS",
    "ROBUST_MODEL_ID",
    "RobustDynamicBTConfig",
    "RobustDynamicBTPrediction",
    "RobustDynamicBradleyTerry",
    "RobustHyperparameterCandidate",
    "RobustPrequentialRun",
    "RobustTournamentResult",
    "default_robust_candidates",
    "paired_cluster_block_comparison",
    "run_prequential_robust_dynamic_bt",
    "run_robust_dynamic_bt_tournament",
]
