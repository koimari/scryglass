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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from lol_kills.etl.competition import (
    INTERNATIONAL_LEAGUES,
    REGIONAL_LEAGUES,
    TAXONOMY_VERSION,
    canonicalize_competition_frame,
    team_identity_key,
)
from lol_kills.etl.paths import FEATURES_DIR
from lol_kills.ratings.validation import audit_rating_inputs


LOGIT_TO_ELO = 400.0 / math.log(10.0)


@dataclass(frozen=True)
class HierarchicalBTConfig:
    base_rating: float = 1500.0
    half_life_days: float = 365.0
    team_l2: float = 40.0
    league_l2: float = 100.0
    side_l2: float = 100.0
    min_sigma: float = 20.0
    unbridged_league_sigma: float = 45.0
    bridge_target_series: int = 8
    conservative_z: float = 1.6448536269514722  # one-sided 90th percentile
    max_iter: int = 500


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() in {"", "nan", "None"}


def _series_key(row: pd.Series) -> str:
    explicit = row.get("grid_series_id")
    if not _is_missing(explicit):
        return f"grid:{explicit}"
    date = pd.Timestamp(row["date"]).floor("4h") if pd.notna(row.get("date")) else "unknown-date"
    a, b = sorted((team_identity_key(row.get("blue_team")), team_identity_key(row.get("red_team"))))
    return f"derived:{date}|{a}|{b}"


def _observations(
    maps: pd.DataFrame,
    as_of: pd.Timestamp | None,
    half_life_days: float,
) -> pd.DataFrame:
    frame = canonicalize_competition_frame(maps)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame["y_blue_win"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame.dropna(subset=["date", "y_blue_win"]).copy()
    if as_of is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of)].copy()
    frame = frame.sort_values("date")
    if frame.empty:
        return pd.DataFrame()

    # Home league is the latest observed regional affiliation before the
    # match.  A first domestic row establishes the affiliation only after its
    # pre-match state is recorded; international rows never overwrite it.
    home_league: dict[str, str] = {}
    display: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        blue_name = str(row.get("blue_team") or "")
        red_name = str(row.get("red_team") or "")
        blue = team_identity_key(blue_name)
        red = team_identity_key(red_name)
        display.setdefault(blue, blue_name)
        display.setdefault(red, red_name)
        source_league = str(row.get("league") or "UNKNOWN")
        blue_home = home_league.get(blue, source_league if source_league in REGIONAL_LEAGUES else "UNKNOWN")
        red_home = home_league.get(red, source_league if source_league in REGIONAL_LEAGUES else "UNKNOWN")
        records.append(
            {
                "series_key": _series_key(row),
                "date": row["date"],
                "blue": blue,
                "red": red,
                "blue_name": blue_name,
                "red_name": red_name,
                "blue_home": blue_home,
                "red_home": red_home,
                "y_blue": float(row["y_blue_win"]),
                "league": source_league,
                "is_international": bool(row.get("is_international", source_league in INTERNATIONAL_LEAGUES)),
                "blue_side": 1.0,
            }
        )
        if source_league in REGIONAL_LEAGUES:
            home_league[blue] = source_league
            home_league[red] = source_league

    frame_rows = pd.DataFrame(records)
    collapsed: list[dict[str, Any]] = []
    for _, group in frame_rows.groupby("series_key", sort=False):
        first = group.iloc[0]
        a, b = sorted((str(first["blue"]), str(first["red"])))
        a_rows = group[group["blue"].eq(a)]
        a_wins = float(a_rows["y_blue"].sum()) + float((group[group["red"].eq(a)]["y_blue"] == 0).sum())
        n_maps = len(group)
        # Complete series normally have an odd number of maps.  For an
        # incomplete/duplicate feed, use the first map only on a tie.
        y_a = 1.0 if a_wins > n_maps / 2 else 0.0 if a_wins < n_maps / 2 else (float(first["y_blue"]) if first["blue"] == a else 1.0 - float(first["y_blue"]))
        a_row = group[group["blue"].eq(a)]
        b_row = group[group["blue"].eq(b)]
        source_a = a_row.iloc[0] if not a_row.empty else first
        source_b = b_row.iloc[0] if not b_row.empty else first
        collapsed.append(
            {
                "series_key": first["series_key"],
                "date": first["date"],
                "team_a": a,
                "team_b": b,
                "team_a_name": first["blue_name"] if first["blue"] == a else first["red_name"],
                "team_b_name": first["red_name"] if first["blue"] == a else first["blue_name"],
                "home_a": source_a["blue_home"] if source_a["blue"] == a else source_a["red_home"],
                "home_b": source_b["blue_home"] if source_b["blue"] == b else source_b["red_home"],
                "y_a": y_a,
                "n_maps": n_maps,
                "international": bool(group["is_international"].any()),
                "a_was_blue": 1.0 if first["blue"] == a else -1.0,
            }
        )
    out = pd.DataFrame(collapsed).sort_values("date").reset_index(drop=True)
    if not out.empty:
        cutoff = out["date"].max()
        out["weight"] = np.exp(
            -((cutoff - out["date"]).dt.total_seconds() / 86400.0) / max(half_life_days, 1.0)
        )
    return out


def _design(observations: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    teams = sorted(set(observations["team_a"]) | set(observations["team_b"]))
    leagues = sorted(
        (set(observations["home_a"]) | set(observations["home_b"])) - {"UNKNOWN"}
    )
    team_idx = {value: i for i, value in enumerate(teams)}
    league_idx = {value: i for i, value in enumerate(leagues)}
    X = np.zeros((len(observations), len(teams) + len(leagues) + 1), dtype=float)
    for i, row in observations.iterrows():
        X[i, team_idx[row["team_a"]]] += 1.0
        X[i, team_idx[row["team_b"]]] -= 1.0
        if row["home_a"] in league_idx:
            X[i, len(teams) + league_idx[row["home_a"]]] += 1.0
        if row["home_b"] in league_idx:
            X[i, len(teams) + league_idx[row["home_b"]]] -= 1.0
        X[i, -1] = float(row["a_was_blue"])
    return X, teams, leagues


def fit_hierarchical_bt(
    maps: pd.DataFrame,
    cfg: HierarchicalBTConfig | None = None,
    as_of: pd.Timestamp | None = None,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the current conservative ladder and optionally persist its snapshot."""

    cfg = cfg or HierarchicalBTConfig()
    input_audit = audit_rating_inputs(maps)
    obs = _observations(maps, as_of, cfg.half_life_days)
    if obs.empty:
        empty = pd.DataFrame(columns=["team", "team_key", "mu_total", "sigma"])
        return empty, {
            "model": "hierarchical_bt",
            "n_series": 0,
            "taxonomy_version": TAXONOMY_VERSION,
            "input_audit": input_audit,
        }

    X, teams, leagues = _design(obs)
    y = obs["y_a"].to_numpy(float)
    weight = obs["weight"].to_numpy(float)
    n_team = len(teams)
    n_league = len(leagues)
    penalty = np.array(
        [cfg.team_l2] * n_team + [cfg.league_l2] * n_league + [cfg.side_l2],
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
    beta = np.clip(np.nan_to_num(result.x, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
    p = expit(np.einsum("ij,j->i", X, beta, optimize=True))
    hessian = np.einsum(
        "i,ij,ik->jk", weight * p * (1.0 - p), X, X, optimize=True
    ) + np.diag(penalty)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian, rcond=1e-10)

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
    for _, row in obs.iterrows():
        display_by_team.setdefault(row["team_a"], row["team_a_name"])
        display_by_team.setdefault(row["team_b"], row["team_b_name"])

    rows: list[dict[str, Any]] = []
    for team in teams:
        home = home_by_team.get(team, "UNKNOWN")
        vector = np.zeros(X.shape[1], dtype=float)
        vector[team_idx[team]] = 1.0
        if home in league_idx:
            vector[n_team + league_idx[home]] = 1.0
        mean_logit = float(np.dot(vector, beta))
        variance = max(float(np.einsum("i,ij,j->", vector, covariance, vector)), 0.0)
        sigma = max(cfg.min_sigma, LOGIT_TO_ELO * math.sqrt(variance))
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
                "sigma": sigma,
                "rating_p10": rating - cfg.conservative_z * sigma,
                "n_series": int(len(team_obs)),
                "n_maps": int(team_obs["n_maps"].sum()),
                "international_series": intl,
                "home_league": home,
                "model": "hierarchical_bt",
            }
        )
    snapshot = pd.DataFrame(rows).sort_values("rating_p10", ascending=False).reset_index(drop=True)
    meta: dict[str, Any] = {
        "model": "hierarchical_bt",
        "taxonomy_version": TAXONOMY_VERSION,
        "n_series": int(len(obs)),
        "n_maps": int(obs["n_maps"].sum()),
        "n_teams": int(len(teams)),
        "n_leagues": int(len(leagues)),
        "as_of": str(obs["date"].max()),
        "config": cfg.__dict__,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "input_audit": input_audit,
        "note": "Series-collapsed penalized MAP Bradley-Terry with local Laplace uncertainty plus explicit uncertainty inflation for teams without international bridges; use rating_p10 for conservative rank.",
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
