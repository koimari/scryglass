"""Regularized hierarchical Bradley--Terry ratings for current ladders.

This is the conservative public-rating reference model.  It is deliberately
separate from the sequential Dual Elo feature generator: the latter remains a
useful pre-match benchmark, while this module fits a global organization
effect plus a partially pooled home-league effect for the current ladder.

Important design choices:

* maps are collapsed to one observation per series so Bo3/Bo5 maps do not
  receive five times the weight of a Bo1;
* organization identity is independent of the event label, so LCS/MSI/EWC
  appearances share one team effect;
* league effects are strongly regularized and only become precise through
  cross-league bridges; a disconnected domestic ladder therefore gets a wide
  interval instead of an unjustified global rank;
* recency weights are explicit and the fit can be cut off at any date for
  rolling-origin validation.

The posterior uncertainty is a local Laplace approximation to the penalized
MAP fit.  It is suitable for conservative display/ranking and diagnostics;
it is not presented as a fully sampled Bayesian posterior.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from lol_kills.etl.competition import (
    INTERNATIONAL_LEAGUES,
    REGIONAL_LEAGUES,
    TAXONOMY_VERSION,
    team_identity_key,
)
from lol_kills.etl.paths import FEATURES_DIR
from lol_kills.etl.series_ledger import build_canonical_series_ledger
from lol_kills.ratings.validation import audit_rating_inputs

LOGIT_TO_ELO = 400.0 / math.log(10.0)
RATING_LOWER_TAIL_PROBABILITY = 0.05
RATING_P05_Z = NormalDist().inv_cdf(
    1.0 - RATING_LOWER_TAIL_PROBABILITY
)
MODEL_FAMILY = "hierarchical_bt"


class HierarchicalRatingError(RuntimeError):
    """Raised when an audited hierarchical rating cannot be published."""


@dataclass(frozen=True)
class HierarchicalBTConfig:
    base_rating: float = 1500.0
    half_life_days: float = 365.0
    team_l2: float = 40.0
    league_l2: float = 100.0
    min_sigma: float = 20.0
    unbridged_league_sigma: float = 45.0
    bridge_target_series: int = 8
    max_iter: int = 500

    def __post_init__(self) -> None:
        for name, value in {
            "half_life_days": self.half_life_days,
            "team_l2": self.team_l2,
            "league_l2": self.league_l2,
            "min_sigma": self.min_sigma,
            "unbridged_league_sigma": self.unbridged_league_sigma,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.bridge_target_series < 1 or self.max_iter < 1:
            raise ValueError(
                "bridge_target_series and max_iter must be positive"
            )


def _model_identity(cfg: HierarchicalBTConfig) -> dict[str, str]:
    """Return a code-and-config identity that validation must match exactly."""

    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_payload = json.dumps(
        cfg.__dict__,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    return {
        "model_id": MODEL_FAMILY,
        "model_code_sha256": code_sha256,
        "model_config_sha256": config_sha256,
        "model_version": (
            f"{MODEL_FAMILY}:{code_sha256[:12]}:{config_sha256[:12]}"
        ),
    }


def _observations(
    maps: pd.DataFrame,
    as_of: pd.Timestamp | None,
    half_life_days: float,
) -> pd.DataFrame:
    canonical = build_canonical_series_ledger(maps)
    frame = canonical.maps
    ledger = canonical.series
    if frame.empty:
        out = pd.DataFrame()
        out.attrs["series_ledger_audit"] = canonical.audit
        out.attrs["skipped_tied_series"] = 0
        out.attrs["skipped_gapped_series"] = 0
        out.attrs["skipped_unverified_series"] = len(ledger)
        return out
    eligible_ledger = ledger.loc[ledger["rating_eligible"]].copy()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        cutoff = (
            cutoff.tz_localize("UTC")
            if cutoff.tzinfo is None
            else cutoff.tz_convert("UTC")
        )
        completion_time = pd.to_datetime(
            eligible_ledger["date_end"],
            errors="coerce",
            utc=True,
            format="mixed",
        )
        # A series enters the state only after its final verified map. Filtering
        # maps first would turn a future-completed Bo3 prefix into a false Bo1.
        eligible_ledger = eligible_ledger.loc[
            completion_time.notna() & completion_time.le(cutoff)
        ]
    eligible_ids = set(
        eligible_ledger["canonical_series_id"].astype(str)
    )
    frame = frame[
        frame["canonical_series_id"].astype(str).isin(eligible_ids)
    ].copy()
    frame["date"] = pd.to_datetime(
        frame.get("date"),
        errors="coerce",
        utc=True,
        format="mixed",
    )
    frame["y_blue_win"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame.dropna(subset=["date", "y_blue_win"]).copy()
    frame = frame.sort_values(["date", "game_uid"] if "game_uid" in frame.columns else ["date"], kind="mergesort")
    if frame.empty:
        out = pd.DataFrame()
        gapped = (
            ledger["quarantine_reasons"]
            .map(
                lambda reasons: bool(
                    {
                        "missing_source_game_index",
                        "duplicate_source_game_index",
                        "non_contiguous_source_game_index",
                    }.intersection(set(reasons or []))
                )
            )
            .sum()
            if not ledger.empty
            else 0
        )
        out.attrs["series_ledger_audit"] = canonical.audit
        out.attrs["skipped_tied_series"] = int(
            (
                ledger["score_a"].eq(ledger["score_b"])
                if not ledger.empty
                else pd.Series(dtype=bool)
            ).sum()
        )
        out.attrs["skipped_gapped_series"] = int(gapped)
        out.attrs["skipped_unverified_series"] = int(
            (~ledger["rating_eligible"]).sum()
        )
        return out

    # A domestic event establishes both teams' affiliation once at its series
    # boundary. International events preserve the most recent domestic state.
    # The frozen affiliation is then used for every map in the series.
    home_league: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    series_groups = sorted(
        frame.groupby("canonical_series_id", sort=False),
        key=lambda item: (
            item[1]["date"].min(),
            str(item[0]),
        ),
    )
    for series_key, group in series_groups:
        ordered_group = group.sort_values(
            ["date", "canonical_game_index"],
            kind="mergesort",
        )
        first = ordered_group.iloc[0]
        source_league = str(first.get("league") or "UNKNOWN")
        is_international = bool(
            ordered_group.get(
                "is_international",
                pd.Series(
                    source_league in INTERNATIONAL_LEAGUES,
                    index=ordered_group.index,
                ),
            )
            .fillna(False)
            .astype(bool)
            .any()
        )
        team_names: dict[str, str] = {}
        for _, row in ordered_group.iterrows():
            for side in ("blue", "red"):
                name = str(row.get(f"{side}_team") or "")
                team_names[team_identity_key(name)] = name
        if len(team_names) != 2:
            raise RuntimeError(
                "rating-eligible series must contain exactly two team identities"
            )
        if not is_international and source_league != "UNKNOWN":
            for team in team_names:
                home_league[team] = source_league
        frozen_home = {
            team: home_league.get(team, "UNKNOWN")
            for team in team_names
        }
        for _, row in ordered_group.iterrows():
            blue_name = str(row.get("blue_team") or "")
            red_name = str(row.get("red_team") or "")
            blue = team_identity_key(blue_name)
            red = team_identity_key(red_name)
            records.append(
                {
                    "series_key": str(series_key),
                    "date": row["date"],
                    "blue": blue,
                    "red": red,
                    "blue_name": blue_name,
                    "red_name": red_name,
                    "blue_home": frozen_home[blue],
                    "red_home": frozen_home[red],
                    "y_blue": float(row["y_blue_win"]),
                    "league": source_league,
                    "is_international": is_international,
                    "canonical_game_index": row.get(
                        "canonical_game_index"
                    ),
                    "scheduled_best_of": row.get("scheduled_best_of"),
                }
            )

    frame_rows = pd.DataFrame(records)
    collapsed: list[dict[str, Any]] = []
    skipped_tied_series = int(
        (
            ledger["score_a"].eq(ledger["score_b"])
            if not ledger.empty
            else pd.Series(dtype=bool)
        ).sum()
    )
    skipped_gapped_series = int(
        ledger["quarantine_reasons"]
        .map(
            lambda reasons: bool(
                {
                    "missing_source_game_index",
                    "duplicate_source_game_index",
                    "non_contiguous_source_game_index",
                }.intersection(set(reasons or []))
            )
        )
        .sum()
        if not ledger.empty
        else 0
    )
    for _, group in frame_rows.groupby("series_key", sort=False):
        first = group.iloc[0]
        ordered_indices = sorted(
            int(value)
            for value in pd.to_numeric(
                group["canonical_game_index"], errors="coerce"
            ).dropna()
        )
        if ordered_indices != list(range(1, len(group) + 1)):
            raise RuntimeError(
                "rating-eligible canonical series has invalid completed-map indices"
            )
        a, b = sorted((str(first["blue"]), str(first["red"])))
        a_rows = group[group["blue"].eq(a)]
        a_wins = float(a_rows["y_blue"].sum()) + float((group[group["red"].eq(a)]["y_blue"] == 0).sum())
        n_maps = len(group)
        # This is a defensive invariant; the canonical series ledger excludes
        # ties and scores incompatible with the scheduled format.
        if a_wins == n_maps / 2:
            raise RuntimeError("rating-eligible canonical series cannot be tied")
        y_a = 1.0 if a_wins > n_maps / 2 else 0.0
        a_name_rows = pd.concat(
            [
                group.loc[group["blue"].eq(a), ["date", "blue_name"]].rename(
                    columns={"blue_name": "name"}
                ),
                group.loc[group["red"].eq(a), ["date", "red_name"]].rename(
                    columns={"red_name": "name"}
                ),
            ]
        ).sort_values("date", kind="mergesort")
        b_name_rows = pd.concat(
            [
                group.loc[group["blue"].eq(b), ["date", "blue_name"]].rename(
                    columns={"blue_name": "name"}
                ),
                group.loc[group["red"].eq(b), ["date", "red_name"]].rename(
                    columns={"red_name": "name"}
                ),
            ]
        ).sort_values("date", kind="mergesort")
        home_by_team = {
            str(row["blue"]): str(row["blue_home"])
            for _, row in group.iterrows()
        } | {
            str(row["red"]): str(row["red_home"])
            for _, row in group.iterrows()
        }
        collapsed.append(
            {
                "series_key": first["series_key"],
                "prediction_time": group["date"].min(),
                "date": group["date"].max(),
                "team_a": a,
                "team_b": b,
                "team_a_name": str(a_name_rows.iloc[-1]["name"]),
                "team_b_name": str(b_name_rows.iloc[-1]["name"]),
                "home_a": home_by_team[a],
                "home_b": home_by_team[b],
                "y_a": y_a,
                "n_maps": n_maps,
                "international": bool(group["is_international"].any()),
                "league": str(first["league"]),
                "scheduled_best_of": pd.to_numeric(
                    group["scheduled_best_of"], errors="coerce"
                ).dropna().iloc[0]
                if pd.to_numeric(
                    group["scheduled_best_of"], errors="coerce"
                ).notna().any()
                else pd.NA,
            }
        )
    if not collapsed:
        out = pd.DataFrame()
    else:
        out = pd.DataFrame(collapsed).sort_values("date", kind="mergesort").reset_index(drop=True)
    out.attrs["skipped_tied_series"] = skipped_tied_series
    out.attrs["skipped_gapped_series"] = skipped_gapped_series
    out.attrs["skipped_unverified_series"] = int(
        (~ledger["rating_eligible"]).sum()
    )
    out.attrs["series_ledger_audit"] = canonical.audit
    if not out.empty:
        weight_cutoff = (
            pd.Timestamp(as_of)
            if as_of is not None
            else out["date"].max()
        )
        weight_cutoff = (
            weight_cutoff.tz_localize("UTC")
            if weight_cutoff.tzinfo is None
            else weight_cutoff.tz_convert("UTC")
        )
        out["weight"] = np.exp2(
            -(
                (weight_cutoff - out["date"]).dt.total_seconds()
                / 86400.0
            )
            / half_life_days
        )
    return out


def _design(observations: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    teams = sorted(set(observations["team_a"]) | set(observations["team_b"]))
    leagues = sorted(
        (set(observations["home_a"]) | set(observations["home_b"])) - {"UNKNOWN"}
    )
    team_idx = {value: i for i, value in enumerate(teams)}
    league_idx = {value: i for i, value in enumerate(leagues)}
    X = np.zeros((len(observations), len(teams) + len(leagues)), dtype=float)
    for i, row in observations.iterrows():
        X[i, team_idx[row["team_a"]]] += 1.0
        X[i, team_idx[row["team_b"]]] -= 1.0
        if row["home_a"] in league_idx:
            X[i, len(teams) + league_idx[row["home_a"]]] += 1.0
        if row["home_b"] in league_idx:
            X[i, len(teams) + league_idx[row["home_b"]]] -= 1.0
    return X, teams, leagues


def fit_hierarchical_bt(
    maps: pd.DataFrame,
    cfg: HierarchicalBTConfig | None = None,
    as_of: pd.Timestamp | None = None,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the current conservative ladder and optionally persist its snapshot."""

    cfg = cfg or HierarchicalBTConfig()
    model_identity = _model_identity(cfg)
    input_audit = audit_rating_inputs(maps)
    if not bool(input_audit.get("ok")):
        raise HierarchicalRatingError(
            "refusing to fit hierarchical ratings because the input audit "
            f"failed: {input_audit.get('reason') or input_audit}"
        )
    obs = _observations(maps, as_of, cfg.half_life_days)
    if obs.empty:
        empty = pd.DataFrame(
            columns=["team", "team_key", "mu_total", "sigma", "rating_p05"]
        )
        return empty, {
            "model": MODEL_FAMILY,
            **model_identity,
            "n_series": 0,
            "skipped_tied_series": int(obs.attrs.get("skipped_tied_series", 0)),
            "skipped_gapped_series": int(obs.attrs.get("skipped_gapped_series", 0)),
            "skipped_unverified_series": int(
                obs.attrs.get("skipped_unverified_series", 0)
            ),
            "series_ledger_audit": obs.attrs.get("series_ledger_audit", {}),
            "taxonomy_version": TAXONOMY_VERSION,
            "input_audit": input_audit,
            "uncertainty": {
                "field": "rating_p05",
                "lower_tail_probability": RATING_LOWER_TAIL_PROBABILITY,
                "one_sided_coverage": 1.0
                - RATING_LOWER_TAIL_PROBABILITY,
                "z": RATING_P05_Z,
                "formula": "rating_p05 = mu_total - z * sigma",
            },
        }

    X, teams, leagues = _design(obs)
    y = obs["y_a"].to_numpy(float)
    weight = obs["weight"].to_numpy(float)
    n_team = len(teams)
    n_league = len(leagues)
    penalty = np.array(
        [cfg.team_l2] * n_team + [cfg.league_l2] * n_league,
        dtype=float,
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        # L-BFGS-B may probe a non-finite point during a failed line search;
        # keep the likelihood numerically bounded and let the explicit
        # parameter bounds handle separation.
        safe_beta = np.clip(np.nan_to_num(beta, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
        eta = np.einsum("ij,j->i", X, safe_beta, optimize=True)
        p = expit(eta)
        value = float(
            np.sum(weight * (np.logaddexp(0.0, eta) - y * eta))
            + 0.5 * np.sum(penalty * safe_beta * safe_beta)
        )
        gradient = np.einsum("i,ij->j", weight * (p - y), X, optimize=True) + penalty * safe_beta
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(X.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * X.shape[1],
        options={"maxiter": cfg.max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )
    if (
        not bool(result.success)
        or result.x is None
        or not np.isfinite(np.asarray(result.x, dtype=float)).all()
        or not math.isfinite(float(result.fun))
    ):
        raise HierarchicalRatingError(
            "hierarchical rating optimizer failed; no snapshot was written: "
            f"{result.message}"
        )
    beta = np.clip(np.nan_to_num(result.x, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
    p = expit(np.einsum("ij,j->i", X, beta, optimize=True))
    hessian = np.einsum(
        "i,ij,ik->jk", weight * p * (1.0 - p), X, X, optimize=True
    ) + np.diag(penalty)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian, rcond=1e-10)
    if not np.isfinite(covariance).all():
        raise HierarchicalRatingError(
            "hierarchical rating covariance is non-finite; no snapshot was written"
        )

    team_idx = {value: i for i, value in enumerate(teams)}
    league_idx = {value: i for i, value in enumerate(leagues)}
    latest = obs.sort_values("date").groupby("team_a").tail(1)
    latest_b = obs.sort_values("date").groupby("team_b").tail(1)
    home_by_team: dict[str, str] = {}
    home_at: dict[str, pd.Timestamp] = {}
    for _, row in pd.concat([latest, latest_b]).iterrows():
        for team, home in ((row["team_a"], row["home_a"]), (row["team_b"], row["home_b"])):
            if team not in home_at or row["date"] >= home_at[team]:
                home_at[team] = row["date"]
                home_by_team[team] = home
    display_by_team: dict[str, str] = {}
    for _, row in obs.sort_values("date", kind="mergesort").iterrows():
        # A rating identity survives rebrands, while the public label follows
        # the latest display name observed no later than the fit cutoff.
        display_by_team[row["team_a"]] = row["team_a_name"]
        display_by_team[row["team_b"]] = row["team_b_name"]

    rows: list[dict[str, Any]] = []
    for team in teams:
        home = home_by_team.get(team, "UNKNOWN")
        vector = np.zeros(X.shape[1], dtype=float)
        vector[team_idx[team]] = 1.0
        if home in league_idx:
            vector[n_team + league_idx[home]] = 1.0
        mean_logit = float(np.dot(vector, beta))
        variance = max(float(np.einsum("i,ij,j->", vector, covariance, vector)), 0.0)
        laplace_sigma = LOGIT_TO_ELO * math.sqrt(variance)
        sigma = max(cfg.min_sigma, laplace_sigma)
        rating = cfg.base_rating + LOGIT_TO_ELO * mean_logit
        team_obs = obs[(obs["team_a"] == team) | (obs["team_b"] == team)]
        intl = int(team_obs["international"].sum())
        bridge_gap = max(0.0, 1.0 - min(intl, cfg.bridge_target_series) / max(cfg.bridge_target_series, 1))
        sigma = math.sqrt(sigma * sigma + (cfg.unbridged_league_sigma * bridge_gap) ** 2)
        rows.append(
            {
                "team": display_by_team.get(team, team),
                "team_key": team,
                "mu_total": rating,
                "mu_regional": cfg.base_rating + LOGIT_TO_ELO * beta[team_idx[team]],
                "mu_meta": LOGIT_TO_ELO * (beta[n_team + league_idx[home]] if home in league_idx else 0.0),
                "laplace_sigma": laplace_sigma,
                "display_sigma": sigma,
                "sigma": sigma,
                "rating_p05": rating - RATING_P05_Z * sigma,
                "n_series": len(team_obs),
                "n_maps": int(team_obs["n_maps"].sum()),
                "international_series": intl,
                "home_league": home,
                "model": MODEL_FAMILY,
                "model_version": model_identity["model_version"],
            }
        )
    snapshot = (
        pd.DataFrame(rows)
        .sort_values("rating_p05", ascending=False)
        .reset_index(drop=True)
    )
    meta: dict[str, Any] = {
        "model": MODEL_FAMILY,
        **model_identity,
        "taxonomy_version": TAXONOMY_VERSION,
        "n_series": len(obs),
        "n_maps": int(obs["n_maps"].sum()),
        "skipped_tied_series": int(obs.attrs.get("skipped_tied_series", 0)),
        "skipped_gapped_series": int(obs.attrs.get("skipped_gapped_series", 0)),
        "skipped_unverified_series": int(
            obs.attrs.get("skipped_unverified_series", 0)
        ),
        "series_ledger_audit": obs.attrs.get("series_ledger_audit", {}),
        "n_teams": len(teams),
        "n_leagues": len(leagues),
        "as_of": str(obs["date"].max()),
        "config": cfg.__dict__,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "input_audit": input_audit,
        "uncertainty": {
            "field": "rating_p05",
            "lower_tail_probability": RATING_LOWER_TAIL_PROBABILITY,
            "one_sided_coverage": 1.0
            - RATING_LOWER_TAIL_PROBABILITY,
            "z": RATING_P05_Z,
            "formula": "rating_p05 = mu_total - z * sigma",
            "interpretation": (
                "Uncertainty-adjusted display score. Sigma contains a floor "
                "and bridge inflation in addition to local Laplace variance; "
                "coverage is not claimed until validated."
            ),
        },
        "note": (
            "Series-collapsed penalized MAP Bradley-Terry with local Laplace "
            "uncertainty plus explicit uncertainty inflation for teams without "
            "international bridges. rating_p05 is retained as a compatibility "
            "field for mean minus z times display uncertainty; it is not a "
            "calibrated posterior percentile."
        ),
    }
    if write:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(FEATURES_DIR / "ratings_hierarchical_snapshot.parquet", index=False)
        # The hierarchical fit is the public ladder snapshot.  The sequential
        # benchmark remains available as ratings_dual_snapshot.parquet.
        snapshot.to_parquet(FEATURES_DIR / "ratings_snapshot.parquet", index=False)
        (FEATURES_DIR / "ratings_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (FEATURES_DIR / "ratings_hierarchical_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return snapshot, meta
