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


def _authoritative_series_id(row: pd.Series) -> str | None:
    """Return the source-provided series id when present.

    Only the GRID adapter writes ``grid_series_id`` today; it is the
    authoritative series identity and carries source evidence with it (the
    ``source``/``source_grid`` flags on the row and the adapter revision in
    ``lol_kills/etl/grid_ingest.py``).  A missing or empty id means the source
    has no safe series identity for this map.
    """

    explicit = row.get("grid_series_id")
    if not _is_missing(explicit):
        return str(explicit).strip()
    return None


def _game_key(row: pd.Series) -> str:
    """Return a stable game-level identity that never merges unrelated maps.

    ``game_uid`` is the canonical per-map identity in every warehouse frame
    (Oracle's Elixir ``gameid`` or Leaguepedia ``GameId``) and is unique per
    match.  When a frame lacks it, fall back to a date/teams/game-number key.

    The four-hour date bucket and sorted-team pairing are intentionally NOT
    used as a grouping key here: they merge unrelated matches and change
    outcome, side, recency, series count, uncertainty, and every downstream
    rating.
    """

    uid = row.get("game_uid")
    if not _is_missing(uid):
        return f"game:{str(uid).strip()}"
    date = (
        pd.Timestamp(row["date"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        if pd.notna(row.get("date"))
        else "unknown-date"
    )
    a, b = sorted((team_identity_key(row.get("blue_team")), team_identity_key(row.get("red_team"))))
    game = row.get("game")
    game_bit = f"|game-{game}" if not _is_missing(game) else ""
    return f"derived-map:{date}|{a}|{b}{game_bit}"


def _series_identity(frame_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign every map an explicit series identity; audit unsafe/tied maps.

    Rules (issue #44):

    * An authoritative source series id (``grid_series_id``) groups maps into
      one series observation only when the group is internally consistent:
      the same unordered team pair appears in every map of the group.  A
      reused id that points at different team pairs is unsafe and its maps
      fall back to stable game-level keys.
    * Without a safe series id, each map keeps its own stable game-level key.
      There is no derived time/team bucket.
    * Series maps whose results do not produce a strict majority (a tied or
      incomplete feed) are unresolved: they are preserved in the returned
      audit trail and excluded from primary series inference because the
      series outcome is not identified.

    Returns ``(frame_rows, audit)`` where ``frame_rows`` gains ``series_key``
    and ``series_source`` columns (``grid`` or ``none``).
    """

    out = frame_rows.copy()
    out["series_key"] = out.apply(_game_key, axis=1)
    out["series_source"] = "none"
    out["series_id_present"] = out["grid_series_id"].fillna("").astype(str).str.strip().ne("")

    unsafe: set[str] = set()
    authoritative = out[out["series_id_present"]].copy()
    if not authoritative.empty:
        authoritative["_pair"] = authoritative.apply(
            lambda row: "|".join(sorted((str(row["blue"]), str(row["red"])))), axis=1
        )
        pair_counts = authoritative.groupby("grid_series_id")["_pair"].nunique()
        unsafe = set(str(value) for value in pair_counts[pair_counts > 1].index)
        safe = authoritative[~authoritative["grid_series_id"].astype(str).isin(unsafe)]
        out.loc[safe.index, "series_key"] = "grid:" + safe["grid_series_id"].astype(str)
        out.loc[safe.index, "series_source"] = "grid"

    audit: dict[str, Any] = {
        "unsafe_series_ids": sorted(unsafe),
        "n_unsafe_maps": int(out["series_id_present"].sum() - (out["series_source"] == "grid").sum()),
    }
    return out, audit


def _observations(
    maps: pd.DataFrame,
    as_of: pd.Timestamp | None,
    half_life_days: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build series-collapsed observations with explicit series identity.

    Returns ``(observations, audit)``.  Each row of ``observations`` is either
    one authoritative source series (``series_source == "grid"``) or one map
    with a stable game-level key (``series_source == "none"``).  The audit
    dict records maps excluded from primary inference (unsafe series ids and
    tied/incomplete feeds) so they stay inspectable.
    """

    frame = canonicalize_competition_frame(maps)
    if frame is None or frame.empty:
        return pd.DataFrame(), {
            "n_unresolved_maps": 0,
            "n_unresolved_series": 0,
            "unresolved_series_ids": [],
            "unresolved_map_uids": [],
            "unsafe_series_ids": [],
            "n_unsafe_maps": 0,
        }
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    frame["y_blue_win"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame.dropna(subset=["date", "y_blue_win"]).copy()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["date"] <= cutoff].copy()
    frame = frame.sort_values("date")
    if frame.empty:
        return pd.DataFrame(), {
            "n_unresolved_maps": 0,
            "n_unresolved_series": 0,
            "unresolved_series_ids": [],
            "unresolved_map_uids": [],
            "unsafe_series_ids": [],
            "n_unsafe_maps": 0,
        }

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
        game_uid = row.get("game_uid")
        grid_series_id = _authoritative_series_id(row) or ""
        game = row.get("game")
        records.append(
            {
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
                "game_uid": "" if _is_missing(game_uid) else str(game_uid).strip(),
                "grid_series_id": grid_series_id,
                "game": "" if _is_missing(game) else str(game).strip(),
            }
        )
        if source_league in REGIONAL_LEAGUES:
            home_league[blue] = source_league
            home_league[red] = source_league

    frame_rows = pd.DataFrame(records)
    frame_rows, identity_audit = _series_identity(frame_rows)

    collapsed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for key, group in frame_rows.groupby("series_key", sort=False):
        pairs = set(
            group.apply(
                lambda row: "|".join(sorted((str(row["blue"]), str(row["red"])))), axis=1
            )
        )
        if len(pairs) != 1:
            # Exact-duplicate fallback keys with different team pairs cannot
            # be merged into one observation; keep them for audit only.
            unresolved.extend(group.to_dict("records"))
            continue
        a, b = sorted((str(group["blue"].iloc[0]), str(group["red"].iloc[0])))
        a_rows = group[group["blue"].eq(a)]
        a_wins = float(a_rows["y_blue"].sum()) + float((group[group["red"].eq(a)]["y_blue"] == 0).sum())
        n_maps = len(group)
        if a_wins * 2 == n_maps:
            # Tied/incomplete feed: the series outcome is not identified.
            # Preserve the maps for audit; exclude from primary inference.
            unresolved.extend(group.to_dict("records"))
            continue
        # A strict majority over ALL maps defines the series winner; the
        # first map is never selected as an outcome shortcut.
        y_a = 1.0 if a_wins > n_maps / 2 else 0.0
        a_blue_share = float(a_rows["blue"].eq(a).sum()) / n_maps
        first = group.iloc[0]
        source_a = a_rows.iloc[0] if not a_rows.empty else first
        b_rows = group[group["blue"].eq(b)]
        source_b = b_rows.iloc[0] if not b_rows.empty else first
        collapsed.append(
            {
                "series_key": key,
                "series_source": str(group["series_source"].iloc[0]),
                "game_uid": ",".join(str(value) for value in group["game_uid"] if str(value)),
                "date": first["date"],
                "team_a": a,
                "team_b": b,
                "team_a_name": first["blue_name"] if first["blue"] == a else first["red_name"],
                "team_b_name": first["red_name"] if first["blue"] == a else first["blue_name"],
                "home_a": source_a["blue_home"] if source_a["blue"] == a else source_a["red_home"],
                "home_b": source_b["blue_home"] if source_b["blue"] == b else source_b["red_home"],
                "y_a": y_a,
                "n_maps": n_maps,
                "a_blue_share": a_blue_share,
                "international": bool(group["is_international"].any()),
            }
        )
    out = pd.DataFrame(collapsed).sort_values("date").reset_index(drop=True)
    if not out.empty:
        cutoff = out["date"].max()
        out["weight"] = np.exp(
            -((cutoff - out["date"]).dt.total_seconds() / 86400.0) / max(half_life_days, 1.0)
        )
    unresolved_frame = pd.DataFrame(unresolved)
    unresolved_ids: list[str] = []
    unresolved_uids: list[str] = []
    if not unresolved_frame.empty:
        unresolved_ids = sorted(
            set(
                str(value)
                for value in unresolved_frame["grid_series_id"].fillna("")
                if str(value)
            )
        )
        unresolved_uids = sorted(
            set(str(value) for value in unresolved_frame["game_uid"] if str(value))
        )
    audit: dict[str, Any] = {
        "n_unresolved_maps": len(unresolved),
        "n_unresolved_series": int(unresolved_frame["series_key"].nunique()) if not unresolved_frame.empty else 0,
        "unresolved_series_ids": unresolved_ids,
        "unresolved_map_uids": unresolved_uids,
        "unsafe_series_ids": identity_audit["unsafe_series_ids"],
        "n_unsafe_maps": identity_audit["n_unsafe_maps"],
    }
    return out, audit


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
        # Side exposure: the observation's team A blue share minus team B's.
        # For a Bo1 this is +/-1 exactly; for a multi-map series it keeps
        # every map's side information instead of the first map only.
        X[i, -1] = 2.0 * float(row["a_blue_share"]) - 1.0
    return X, teams, leagues


def fit_hierarchical_bt(
    maps: pd.DataFrame,
    cfg: HierarchicalBTConfig | None = None,
    as_of: pd.Timestamp | None = None,
    write: bool = True,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the current conservative ladder and optionally persist its snapshot."""

    cfg = cfg or HierarchicalBTConfig()
    input_audit = audit_rating_inputs(maps)
    obs, series_audit = _observations(maps, as_of, cfg.half_life_days)
    if obs.empty:
        empty = pd.DataFrame(columns=["team", "team_key", "mu_total", "sigma"])
        return empty, {
            "model": "hierarchical_bt",
            "n_series": 0,
            "taxonomy_version": TAXONOMY_VERSION,
            "input_audit": input_audit,
            "series_identity": {
                "revision": "2026-08-09.1",
                "n_authoritative_series": 0,
                "n_game_level_maps": 0,
                **series_audit,
            },
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
        last_game_date = None
        if not team_obs.empty and pd.notna(team_obs["date"].max()):
            last_game_date = pd.Timestamp(team_obs["date"].max()).isoformat()
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
                "last_game_date": last_game_date,
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
        "series_identity": {
            "revision": "2026-08-09.1",
            "n_authoritative_series": int((obs["series_source"] == "grid").sum()),
            "n_game_level_maps": int((obs["series_source"] == "none").sum()),
            **series_audit,
        },
        "note": "Series-collapsed penalized MAP Bradley-Terry with explicit series identity (authoritative GRID series id when safe, stable game-level keys otherwise) and local Laplace uncertainty plus explicit uncertainty inflation for teams without international bridges; use rating_p10 for conservative rank.",
    }
    if write:
        destination = Path(output_dir or FEATURES_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(destination / "ratings_hierarchical_snapshot.parquet", index=False)
        # The hierarchical fit is the public ladder snapshot.  The sequential
        # benchmark remains available as ratings_dual_snapshot.parquet.
        snapshot.to_parquet(destination / "ratings_snapshot.parquet", index=False)
        (destination / "ratings_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (destination / "ratings_hierarchical_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return snapshot, meta


def _sunday_utc(as_of: pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize() - pd.Timedelta(days=(stamp.weekday() + 1) % 7)


def _recent_team_baseline_anchor(
    previous_as_of: pd.Timestamp | None,
    sunday_baseline: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.Timestamp:
    """Previous-refresh movement anchor with safe fallbacks."""
    if previous_as_of is None:
        return sunday_baseline
    anchor = pd.Timestamp(previous_as_of)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    if anchor >= cutoff or anchor < sunday_baseline - pd.Timedelta(days=400):
        return sunday_baseline
    return anchor


def build_team_weekly_ranks(
    maps: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    min_series: int = 5,
    previous_as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Return team rank movement against the previous refresh's ladder.

    Both snapshots use the same hierarchical fit and the same conservative
    ``rating_p10`` ordering as the public team ladder. The recent baseline is
    the previous refresh's cutoff when ``previous_as_of`` is provided (so
    movement reflects every published cycle), falling back to the prior
    Sunday snapshot otherwise. New games therefore change the ladder and its
    movement in one refresh.
    """

    if min_series < 1:
        raise ValueError("min_series must be positive")
    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    recent_anchor = _recent_team_baseline_anchor(previous_as_of, previous_start, cutoff)
    current, _ = fit_hierarchical_bt(maps, as_of=cutoff, write=False)
    previous, _ = fit_hierarchical_bt(maps, as_of=recent_anchor - pd.Timedelta(microseconds=1), write=False)

    def order(snapshot: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
        if snapshot.empty:
            return {}, {}
        eligible = snapshot[snapshot["n_series"].fillna(0).ge(min_series)].copy()
        eligible["rank_value"] = pd.to_numeric(eligible["rating_p10"], errors="coerce")
        eligible["mu_value"] = pd.to_numeric(eligible["mu_total"], errors="coerce")
        eligible = eligible.dropna(subset=["rank_value"])
        eligible["team_sort"] = eligible["team"].astype(str).str.casefold()
        eligible = eligible.sort_values(["rank_value", "team_sort"], ascending=[False, True])
        ranks = {str(team): rank for rank, team in enumerate(eligible["team"].astype(str), start=1)}
        mus = {
            str(team): float(mu) for team, mu in zip(eligible["team"].astype(str), eligible["mu_value"])
        }
        return ranks, mus

    current_rank, current_mu = order(current)
    previous_rank, previous_mu = order(previous)
    current_through = pd.Timestamp(cutoff)
    if current_through.tzinfo is not None:
        current_through = current_through.tz_convert("UTC").tz_localize(None)
    by_team: dict[str, dict[str, int | None]] = {}
    for team, rank in current_rank.items():
        prior = previous_rank.get(team)
        prior_mu = previous_mu.get(team)
        mu = current_mu.get(team)
        by_team[team] = {
            "rank": rank,
            "delta": (prior - rank) if prior is not None else None,
            "mu_delta": (mu - prior_mu) if (mu is not None and prior_mu is not None) else None,
        }

    return {
        "as_of": f"{week_start.isoformat()}Z",
        "previous_as_of": f"{recent_anchor.isoformat()}Z",
        "current_through": f"{current_through.isoformat()}Z",
        "min_series": int(min_series),
        "by_team": by_team,
        "note": "Rank movement compares conservative team rating against the previous refresh (or the prior Sunday when no earlier refresh exists); positive delta means a climb.",
    }
