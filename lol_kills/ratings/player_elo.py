#!/usr/bin/env python3
"""Player Dual-Elo team aggregate (DESCRIPTIVE BASELINE).

This track is the descriptive baseline for the public player ladder.  It is
NOT the v2 dynamic Player Rating: a shared team outcome updates every player
on a side with the same residual scaled only by fixed role weights, so the
baseline cannot identify individual contribution, posterior displacement,
precision, or source/context coverage.  Roster moves travel with the player:
team strength is the mean of the five pre-match player μs (regional + meta),
not a sticky org rating.

The v2 dynamic Player Rating lives in ``lol_kills/v2/ratings/player/`` and
remains development-only until its acceptance record passes; until then this
baseline carries the public label with an explicit descriptive claim ceiling.

This module measures historical results. It does not identify a player's
causal contribution and does not authorize predictions or betting decisions.

  python3 -m lol_kills.ratings.player_elo
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import canonicalize_competition_frame, is_team_affiliation_league
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import DualEloConfig, _is_intl, expected_score
from lol_kills.ratings.global_player_bt import (
    GlobalPlayerBTConfig,
    GlobalPlayerRatingError,
    fit_global_player_bt,
)

# Slight role weights for aggregation (still sums≈5)
ROLE_WEIGHT = {
    "top": 0.95,
    "jng": 1.05,
    "jungle": 1.05,
    "mid": 1.10,
    "bot": 1.05,
    "adc": 1.05,
    "sup": 0.90,
    "support": 0.90,
    "utility": 0.90,
}


@dataclass
class PlayerState:
    mu_regional: float = 1500.0
    mu_meta: float = 0.0
    sigma: float = 90.0
    last_date: pd.Timestamp | None = None
    n_maps: int = 0
    last_team: str | None = None
    home_league: str | None = None


@dataclass
class PlayerEloConfig:
    k_regional: float = 18.0
    k_meta: float = 10.0
    sigma0: float = 90.0
    sigma_min: float = 28.0
    sigma_month_inflate: float = 1.0
    team_switch_sigma_bump: float = 12.0
    mov_scale: float = 1.0
    use_role_weights: bool = True
    tier2_bridge_sigma: float = 45.0
    tier3_bridge_sigma: float = 60.0
    bridge_support_scale: float = 10.0
    # Blend toward prior when <5 known starters
    prior_mu: float = 1500.0


def total_mu(st: PlayerState) -> float:
    return st.mu_regional + st.mu_meta


def _norm_role(r: str) -> str:
    r = str(r or "").lower().strip()
    if r in ROLE_WEIGHT:
        return r if r not in ("jungle", "adc", "support", "utility") else {
            "jungle": "jng",
            "adc": "bot",
            "support": "sup",
            "utility": "sup",
        }[r]
    for a, b in (
        ("jng", "jng"),
        ("jung", "jng"),
        ("mid", "mid"),
        ("top", "top"),
        ("bot", "bot"),
        ("adc", "bot"),
        ("sup", "sup"),
        ("supp", "sup"),
        ("util", "sup"),
    ):
        if r.startswith(a):
            return b
    return r[:3] if r else "unk"


def _role_w(role: str, cfg: PlayerEloConfig) -> float:
    if not cfg.use_role_weights:
        return 1.0
    return float(ROLE_WEIGHT.get(_norm_role(role), 1.0))


def _lineups_by_game(players: pd.DataFrame) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """
    game_uid → {Blue|Red: [(playername, role), ...]}
    Only position rows with a player name (skip team aggregates).
    """
    if players is None or players.empty or "playername" not in players.columns:
        return {}
    p = players.copy()
    if "game_uid" in p.columns:
        fallback = p["gameid"] if "gameid" in p.columns else None
        p["_gid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in p["game_uid"].items()
        ]
    elif "gameid" in p.columns:
        p["_gid"] = p["gameid"].map(canonical_source_game_key)
    else:
        return {}
    p = p[p["_gid"].notna() & p["_gid"].str.strip().ne("")]
    p["side"] = p["side"].astype(str).str.title()
    p["position"] = p.get("position", pd.Series("unk", index=p.index)).astype(str)
    pos = p["position"].str.lower()
    p = p[pos != "team"].copy()
    p = p[p["playername"].notna() & (p["playername"].astype(str).str.len() > 0)]
    p["_role"] = p["position"].map(_norm_role)
    p["_name"] = p["playername"].astype(str).str.strip()
    p = p[p["_name"].str.lower() != "nan"]

    out: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: {"Blue": [], "Red": []})
    for (gid, side), g in p.groupby(["_gid", "side"], sort=False):
        if side not in ("Blue", "Red"):
            continue
        # stable role order
        order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
        rows = list(zip(g["_name"].tolist(), g["_role"].tolist()))
        rows.sort(key=lambda x: order.get(x[1], 9))
        # dedupe by role keep first
        seen = set()
        cleaned = []
        for name, role in rows:
            if role in seen:
                continue
            seen.add(role)
            cleaned.append((name, role))
        out[str(gid)][side] = cleaned[:5]
    return out


def _aggregate(
    states: dict[str, PlayerState],
    lineup: list[tuple[str, str]],
    cfg: PlayerEloConfig,
) -> tuple[float, float, int, list[dict]]:
    """Return (mu, sigma_mean, n_known, per-player detail)."""
    if not lineup:
        return cfg.prior_mu, cfg.sigma0, 0, []
    details = []
    w_sum = 0.0
    mu_acc = 0.0
    sig_acc = 0.0
    known = 0
    for name, role in lineup[:5]:
        st = states.get(name)
        w = _role_w(role, cfg)
        if st is None:
            mu = cfg.prior_mu
            sig = cfg.sigma0
        else:
            mu = total_mu(st)
            sig = st.sigma
            known += 1
        details.append({"player": name, "role": role, "mu": round(mu, 2), "sigma": round(sig, 2), "w": w})
        mu_acc += w * mu
        sig_acc += w * sig
        w_sum += w
    if w_sum <= 0:
        return cfg.prior_mu, cfg.sigma0, 0, details
    # If fewer than 5 known, shrink toward prior
    mu = mu_acc / w_sum
    if known < 5:
        shrink = known / 5.0
        mu = cfg.prior_mu + shrink * (mu - cfg.prior_mu)
    sig = sig_acc / w_sum
    return mu, sig, known, details


def _snapshot_rows(
    states: dict[str, PlayerState],
    recent_mus: dict[str, list[float]] | None = None,
) -> list[dict[str, object]]:
    recent = recent_mus or {}
    rows = []
    for name, st in states.items():
        history = recent.get(name) or []
        stability = None
        if len(history) >= 2:
            deltas = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            stability = float(sum(deltas) / len(deltas))
        rows.append(
            {
                "player": name,
                "mu_total": total_mu(st),
                "mu_regional": st.mu_regional,
                "mu_meta": st.mu_meta,
                "sigma": st.sigma,
                "n_maps": st.n_maps,
                "last_team": st.last_team,
                "home_league": st.home_league,
                "last_game_date": st.last_date.isoformat() if st.last_date is not None else None,
                "evidence_stability": stability,
            }
        )
    return rows


def _apply_global_scale(
    rows: list[dict[str, object]],
    global_snapshot: pd.DataFrame,
) -> list[dict[str, object]]:
    """Replace local-pool means with the connected global results scale."""

    by_player = {
        str(row["player"]): row
        for _, row in global_snapshot.iterrows()
    }
    output = []
    for source in rows:
        row = dict(source)
        global_row = by_player.get(str(row["player"]))
        connected = int(global_row.get("global_connected") or 0) if global_row is not None else 0
        row["global_connected"] = connected
        row["rating_model"] = "regularized_global_player_bt"
        if connected:
            rating = float(global_row["global_rating"])
            row["mu_total"] = rating
            row["mu_regional"] = rating
            row["mu_meta"] = 0.0
            row["global_component_size"] = int(global_row["global_component_size"])
            row["global_model_maps"] = int(global_row["global_model_maps"])
        output.append(row)
    return output


def _apply_bridge_uncertainty(
    rows: list[dict[str, object]],
    players: pd.DataFrame,
    player_records: Mapping[str, Mapping[str, object]],
    cfg: PlayerEloConfig,
    *,
    through: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    """Widen weak cross-tier anchors without moving the fitted mean."""

    frame = canonicalize_competition_frame(players).copy()
    date_source = frame["date"] if "date" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    frame["_date"] = pd.to_datetime(date_source, utc=True, errors="coerce").dt.tz_localize(None)
    if through is not None:
        cutoff = pd.Timestamp(through)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["_date"].le(cutoff)]
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        frame["_game_id"] = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in frame["game_uid"].items()
        ]
    elif "gameid" in frame.columns:
        frame["_game_id"] = frame["gameid"].map(canonical_source_game_key)
    else:
        frame["_game_id"] = ""
    frame["_player"] = frame.get("playername", pd.Series("", index=frame.index)).astype("string").str.strip()
    frame = frame[
        frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_game_id"].astype(str).str.strip().ne("")
        & frame["competition_tier"].isin({"tier1", "tier2", "tier3"})
    ].drop_duplicates(["_player", "_game_id"])
    tier_counts = frame.groupby(["_player", "competition_tier"], sort=False).size()

    output = []
    for source in rows:
        row = dict(source)
        player = str(row["player"])
        tier = str((player_records.get(player) or {}).get("current_tier") or "")
        if tier == "tier2":
            stronger_maps = int(tier_counts.get((player, "tier1"), 0))
            base = cfg.tier2_bridge_sigma
        elif tier == "tier3":
            stronger_maps = int(tier_counts.get((player, "tier1"), 0)) + int(
                tier_counts.get((player, "tier2"), 0)
            )
            base = cfg.tier3_bridge_sigma
        else:
            stronger_maps = 0
            base = 0.0
        bridge_sigma = base / math.sqrt(1.0 + stronger_maps / cfg.bridge_support_scale)
        row["global_bridge_maps"] = stronger_maps
        row["global_bridge_sigma"] = bridge_sigma
        row["sigma"] = min(
            160.0,
            math.sqrt(float(row.get("sigma") or cfg.sigma0) ** 2 + bridge_sigma**2),
        )
        output.append(row)
    return output


def _run_player_elo(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig,
    checkpoint_dates: list[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, dict[str, PlayerState], dict[pd.Timestamp, list[dict[str, object]]]]:
    """Run the sequential player model and optionally capture dated states."""

    # Apply the same source-preserving competition taxonomy as team ratings so
    # player regional/meta updates cannot drift from the public team contract.
    df = canonicalize_competition_frame(maps).copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    if "game_uid" in df.columns:
        fallback = df["gameid"] if "gameid" in df.columns else None
        df["game_uid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in df["game_uid"].items()
        ]
    elif "gameid" in df.columns:
        df["game_uid"] = df["gameid"].map(canonical_source_game_key)
    else:
        raise ValueError("player Elo maps have no game identity column")
    df = df[df["game_uid"].str.strip().ne("")].copy()
    lineups = _lineups_by_game(players)
    states: dict[str, PlayerState] = {}
    recent_mus: dict[str, list[float]] = {}
    targets = sorted({pd.Timestamp(value).tz_localize(None) for value in (checkpoint_dates or [])})
    checkpoints: dict[pd.Timestamp, list[dict[str, object]]] = {}
    target_idx = 0

    def capture_before(date: pd.Timestamp | None) -> None:
        nonlocal target_idx
        while target_idx < len(targets) and (date is None or date > targets[target_idx]):
            target = targets[target_idx]
            checkpoints[target] = _snapshot_rows(states)
            target_idx += 1

    # Pre-extract the columns the sequential loop reads (avoids per-row pandas access).
    _gid_arr = df["game_uid"].to_numpy(dtype=object)
    _date_arr = df["date"].to_numpy(dtype="datetime64[ns]")
    _bt_col = "blue_team" if "blue_team" in df.columns else "blue_teamname"
    _rt_col = "red_team" if "red_team" in df.columns else "red_teamname"
    _bt_arr = df[_bt_col].astype(str).to_numpy(dtype=object)
    _rt_arr = df[_rt_col].astype(str).to_numpy(dtype=object)
    _y_arr = df["y_blue_win"].to_numpy(dtype=object) if "y_blue_win" in df.columns else np.full(len(df), np.nan, dtype=object)
    _league_arr = df["league"].astype(str).to_numpy(dtype=object) if "league" in df.columns else np.full(len(df), "", dtype=object)
    _tourn_arr = df["tournament"].astype(str).to_numpy(dtype=object) if "tournament" in df.columns else np.full(len(df), "", dtype=object)
    _g15_arr = df["blue_golddiffat15"].to_numpy(dtype=object) if "blue_golddiffat15" in df.columns else np.full(len(df), np.nan, dtype=object)
    _g10_arr = df["blue_golddiffat10"].to_numpy(dtype=object) if "blue_golddiffat10" in df.columns else np.full(len(df), np.nan, dtype=object)
    _len_arr = df["length_min"].to_numpy(dtype=object) if "length_min" in df.columns else np.full(len(df), np.nan, dtype=object)
    _glen_arr = df["gamelength"].to_numpy(dtype=object) if "gamelength" in df.columns else np.full(len(df), np.nan, dtype=object)

    rows = []
    for i in range(len(df)):
        gid = str(_gid_arr[i] or "")
        _dv = _date_arr[i]
        d = pd.Timestamp(_dv) if not pd.isna(_dv) else None
        capture_before(d)
        blue_lu = lineups.get(gid, {}).get("Blue") or []
        red_lu = lineups.get(gid, {}).get("Red") or []
        bt = normalize_team(str(_bt_arr[i] or ""))
        rt = normalize_team(str(_rt_arr[i] or ""))

        # inactivity + team-switch uncertainty
        for name, role in list(blue_lu[:5]) + list(red_lu[:5]):
            if name not in states:
                states[name] = PlayerState(sigma=cfg.sigma0)
            st = states[name]
            if d is not None and st.last_date is not None:
                months = max((d - st.last_date).days / 30.0, 0.0)
                st.sigma = min(160.0, st.sigma + cfg.sigma_month_inflate * months)
            team_now = bt if any(n == name for n, _ in blue_lu[:5]) else rt
            if st.last_team and team_now and st.last_team != team_now:
                st.sigma = min(160.0, st.sigma + cfg.team_switch_sigma_bump)
            states[name] = st

        mu_b, sig_b, known_b, det_b = _aggregate(states, blue_lu, cfg)
        mu_r, sig_r, known_r, det_r = _aggregate(states, red_lu, cfg)
        sig = math.sqrt(sig_b**2 + sig_r**2)
        p = expected_score(mu_b, mu_r)
        shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
        p_shrunk = 0.5 + (p - 0.5) * shrink

        rows.append(
            {
                "game_uid": gid,
                "date": _dv,
                "blue_team": bt,
                "red_team": rt,
                "player_mu_blue": mu_b,
                "player_mu_red": mu_r,
                "player_mu_diff": mu_b - mu_r,
                "player_sigma_blue": sig_b,
                "player_sigma_red": sig_r,
                "player_sigma_pair": sig,
                "player_known_blue": known_b,
                "player_known_red": known_r,
                "p_player_elo": p_shrunk,
                "p_player_elo_raw": p,
            }
        )

        y = _y_arr[i]
        if pd.isna(y):
            continue
        y = float(y)
        intl = _is_intl(str(_league_arr[i] or ""), _tourn_arr[i])

        g10 = _g15_arr[i]
        if pd.isna(g10):
            g10 = _g10_arr[i]
        _len = _len_arr[i]
        if pd.notna(_len):
            length = float(_len)
        elif pd.notna(_glen_arr[i]):
            length = float(_glen_arr[i]) / 60.0
        else:
            length = 30.0
        mov = 1.0
        if pd.notna(g10) and length:
            mov = 1.0 + cfg.mov_scale * math.tanh(float(g10) / (200.0 * max(float(length), 1.0)))

        exp_b = expected_score(mu_b, mu_r)

        for name, role in blue_lu[:5]:
            st = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
            k_scale = st.sigma / cfg.sigma0
            if intl:
                st.mu_meta += cfg.k_meta * k_scale * mov * (y - exp_b)
            else:
                st.mu_regional += cfg.k_regional * k_scale * mov * (y - exp_b)
            st.sigma = max(cfg.sigma_min, st.sigma * 0.985)
            st.n_maps += 1
            if d is not None:
                st.last_date = d
            st.last_team = bt
            league = str(_league_arr[i] or "")
            if is_team_affiliation_league(league):
                st.home_league = league
            states[name] = st
        for name, role in red_lu[:5]:
            st = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
            k_scale = st.sigma / cfg.sigma0
            if intl:
                st.mu_meta += cfg.k_meta * k_scale * mov * ((1 - y) - (1 - exp_b))
            else:
                st.mu_regional += cfg.k_regional * k_scale * mov * ((1 - y) - (1 - exp_b))
            st.sigma = max(cfg.sigma_min, st.sigma * 0.985)
            st.n_maps += 1
            if d is not None:
                st.last_date = d
            st.last_team = rt
            league = str(_league_arr[i] or "")
            if is_team_affiliation_league(league):
                st.home_league = league
            states[name] = st

        # Stability history: keep the last 10 posterior totals per player so
        # the snapshot can expose mean displacement per game.
        for name, role in list(blue_lu[:5]) + list(red_lu[:5]):
            if name in states:
                recent_mus.setdefault(name, []).append(total_mu(states[name]))
                recent_mus[name] = recent_mus[name][-10:]

    while target_idx < len(targets):
        target = targets[target_idx]
        checkpoints[target] = _snapshot_rows(states)
        target_idx += 1
    return pd.DataFrame(rows), states, checkpoints, recent_mus


def build_player_ratings(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
    output_dir: Path | None = None,
    player_records: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Sequential player Elo; player ratings travel across org changes."""

    cfg = cfg or PlayerEloConfig()
    out, states, _checkpoints, recent_mus = _run_player_elo(maps, players, cfg)
    destination = Path(output_dir or FEATURES_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "player_ratings.parquet"
    out.to_parquet(path, index=False)

    global_snapshot, global_meta = fit_global_player_bt(maps, players)
    snap = _apply_global_scale(_snapshot_rows(states, recent_mus), global_snapshot)
    if player_records is not None:
        for row in snap:
            record = player_records.get(str(row["player"]))
            if record is None:
                continue
            row["last_team"] = record.get("current_team")
            row["home_league"] = record.get("current_league") or "UNKNOWN"
        snap = _apply_bridge_uncertainty(snap, players, player_records, cfg)
    snap_df = pd.DataFrame(snap).sort_values("mu_total", ascending=False)
    snap_df.to_parquet(destination / "player_ratings_snapshot.parquet", index=False)
    (destination / "player_ratings_meta.json").write_text(
        json.dumps(
            {
                "n_maps": len(out),
                "n_players": len(snap),
                "config": cfg.__dict__,
                "global_rating": global_meta,
                "note": (
                    "PUBLIC RESULTS RATING: one regularized Bradley-Terry fit uses every "
                    "accepted complete lineup on one connected scale. Competition tier is "
                    "not a bonus or penalty. The rating remains lineup-linked and does not "
                    "identify individual causal contribution."
                ),
            },
            indent=2,
        )
    )
    print(f"[player_elo] wrote {path} n={len(out)} players={len(snap)}")
    return out


def _sunday_utc(as_of: pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize() - pd.Timedelta(days=(stamp.weekday() + 1) % 7)


def _recent_baseline_anchor(
    previous_as_of: pd.Timestamp | None,
    sunday_baseline: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.Timestamp:
    """Previous-refresh movement anchor with safe fallbacks.

    The CLI passes ISO strings with a trailing ``Z`` (tz-aware); the cutoff
    from the refresh is naive.  Normalize every input so the comparison is
    robust regardless of caller convention.
    """
    if previous_as_of is None:
        return sunday_baseline
    anchor = pd.Timestamp(previous_as_of)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    base = pd.Timestamp(sunday_baseline)
    if base.tzinfo is not None:
        base = base.tz_convert("UTC").tz_localize(None)
    cap = pd.Timestamp(cutoff)
    if cap.tzinfo is not None:
        cap = cap.tz_convert("UTC").tz_localize(None)
    if anchor >= cap or anchor < base - pd.Timedelta(days=400):
        return base
    return anchor


def build_player_weekly_ranks(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
    *,
    as_of: pd.Timestamp | None = None,
    min_games: int = 20,
    player_records: Mapping[str, Mapping[str, object]] | None = None,
    previous_as_of: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Return current ranks with recent and calendar-month movement.

    The player ladder is still the current sequential Elo snapshot.  The
    recent movement baseline is the previous refresh's cutoff when
    ``previous_as_of`` is provided (so movement reflects every published
    cycle - after every batch of games), falling back to the prior Sunday
    00:00 UTC snapshot otherwise.  Calendar-month comparisons (1/3/12m)
    are unchanged.
    """

    cfg = cfg or PlayerEloConfig()
    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    frame = maps.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    comparison_cutoffs = {
        "1m": cutoff - pd.DateOffset(months=1),
        "3m": cutoff - pd.DateOffset(months=3),
        "12m": cutoff - pd.DateOffset(months=12),
    }
    if as_of is not None:
        frame = frame[frame["date"].le(cutoff)]

    recent_anchor = _recent_baseline_anchor(previous_as_of, previous_start, cutoff)
    _, states, checkpoints, _recent_mus = _run_player_elo(
        frame,
        players,
        cfg,
        checkpoint_dates=[recent_anchor, *comparison_cutoffs.values()],
    )
    current_global, _current_meta = fit_global_player_bt(
        frame,
        players,
        GlobalPlayerBTConfig(minimum_maps=1),
        through=cutoff,
        validate=False,
    )
    # Current affiliation is the publication filter.  Historical matches in a
    # different circuit remain evidence for the rating but cannot place a
    # developmental player in the current Tier 1 board.
    from lol_kills.export.pack_records import build_player_records

    current_records = dict(player_records) if player_records is not None else build_player_records(players)
    current_rows = _apply_bridge_uncertainty(
        _apply_global_scale(_snapshot_rows(states), current_global),
        players,
        current_records,
        cfg,
        through=cutoff,
    )
    def historical_rows(anchor: pd.Timestamp) -> list[dict[str, object]]:
        snapshot = checkpoints.get(anchor, [])
        if not snapshot:
            return []
        try:
            historical_global, _historical_meta = fit_global_player_bt(
                frame,
                players,
                GlobalPlayerBTConfig(minimum_maps=1),
                through=anchor,
                validate=False,
            )
        except GlobalPlayerRatingError:
            return []
        return _apply_bridge_uncertainty(
            _apply_global_scale(snapshot, historical_global),
            players,
            current_records,
            cfg,
            through=anchor,
        )

    previous_rows = historical_rows(recent_anchor)
    comparison_rows = {
        label: historical_rows(anchor)
        for label, anchor in comparison_cutoffs.items()
    }
    current_tiers = {
        player: record.get("current_tier")
        for player, record in current_records.items()
    }

    def order(rows: list[dict[str, object]], scope: str) -> dict[str, int]:
        eligible = []
        for row in rows:
            player = str(row["player"])
            games = int(row.get("n_maps") or 0)
            tier = current_tiers.get(player)
            if int(row.get("global_connected") or 0) != 1:
                continue
            if games < max(1, int(min_games)):
                continue
            if scope != "all" and tier != scope:
                continue
            mu = float(row.get("mu_total") or 0)
            sigma = float(row.get("sigma") or 0)
            adjusted = mu - max(0.0, sigma - 28.0)
            eligible.append((adjusted, player))
        eligible.sort(key=lambda value: (-value[0], value[1].casefold()))
        return {player: rank for rank, (_, player) in enumerate(eligible, start=1)}

    scopes = ("all", "tier1", "tier2", "tier3")
    current_rank = {scope: order(current_rows, scope) for scope in scopes}
    previous_rank = {scope: order(previous_rows, scope) for scope in scopes}
    comparison_rank = {
        label: {scope: order(rows, scope) for scope in scopes}
        for label, rows in comparison_rows.items()
    }
    by_player: dict[str, dict[str, dict[str, object]]] = {}
    for player, rank in current_rank["all"].items():
        values: dict[str, dict[str, object]] = {}
        for scope in scopes:
            current = current_rank[scope].get(player)
            if current is None:
                continue
            prior = previous_rank[scope].get(player)
            position_deltas: dict[str, dict[str, object]] = {}
            for label, anchor in comparison_cutoffs.items():
                historical = comparison_rank[label][scope].get(player)
                position_deltas[label] = {
                    "as_of": f"{anchor.isoformat()}Z",
                    "rank": historical,
                    "delta": (historical - current) if historical is not None else None,
                }
            values[scope] = {
                "rank": current,
                "delta": (prior - current) if prior is not None else None,
                "position_deltas": position_deltas,
            }
        by_player[player] = values

    return {
        "as_of": f"{week_start.isoformat()}Z",
        "previous_as_of": f"{recent_anchor.isoformat()}Z",
        "current_through": f"{cutoff.isoformat()}Z",
        "position_delta_as_of": {
            label: f"{anchor.isoformat()}Z"
            for label, anchor in comparison_cutoffs.items()
        },
        "min_games": int(min_games),
        "by_player": by_player,
        "note": "Rank movement compares the adjusted global player results rating with the previous refresh (or the prior Sunday when no earlier refresh exists) and the positions one, three, and twelve calendar months earlier. Positive delta means a climb.",
    }


def build_maps_frame_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """One map row per OE game_uid from player rows (full history, not warehouse-filtered)."""
    pl = players.copy()
    if "game_uid" in pl.columns:
        fallback = pl["gameid"] if "gameid" in pl.columns else None
        pl["_gid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in pl["game_uid"].items()
        ]
    elif "gameid" in pl.columns:
        pl["_gid"] = pl["gameid"].map(canonical_source_game_key)
    else:
        return pd.DataFrame()
    pl = pl[pl["_gid"].notna() & pl["_gid"].str.strip().ne("")]
    pl["side"] = pl["side"].astype(str).str.title()
    pl["position"] = pl.get("position", pd.Series("", index=pl.index)).astype(str).str.lower()
    pl = pl[pl["position"] != "team"]
    blue = pl[pl["side"] == "Blue"].drop_duplicates("_gid")
    red = pl[pl["side"] == "Red"].drop_duplicates("_gid")
    m = blue[["_gid", "date", "league", "result", "teamname"]].rename(
        columns={"_gid": "game_uid", "result": "y_blue_win", "teamname": "blue_team"}
    )
    m = m.merge(
        red[["_gid", "teamname"]].rename(columns={"_gid": "game_uid", "teamname": "red_team"}),
        on="game_uid",
        how="inner",
    )
    m["y_blue_win"] = pd.to_numeric(m["y_blue_win"], errors="coerce")
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return m.dropna(subset=["date", "y_blue_win"]).sort_values("date").reset_index(drop=True)


# Module caches — board hot path (avoid re-reading parquet / remapping teamnames)
_PLAYERS_CACHE: pd.DataFrame | None = None
_ROSTER_CACHE: dict[str, list[tuple[str, str]]] = {}
_SNAPSHOT_BY: dict | None = None


def load_players_cached() -> pd.DataFrame:
    global _PLAYERS_CACHE
    if _PLAYERS_CACHE is None:
        path = PARQUET_DIR / "players.parquet"
        _PLAYERS_CACHE = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _PLAYERS_CACHE


def _snapshot_by_player() -> dict:
    global _SNAPSHOT_BY
    if _SNAPSHOT_BY is not None:
        return _SNAPSHOT_BY
    path = FEATURES_DIR / "player_ratings_snapshot.parquet"
    snap = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    by: dict = {}
    if not snap.empty:
        n_maps_col = (
            snap["n_maps"].fillna(0).astype(int)
            if "n_maps" in snap.columns
            else pd.Series([0] * len(snap))
        )
        last_team_col = (
            snap["last_team"] if "last_team" in snap.columns else pd.Series([None] * len(snap))
        )
        for player, mu_r, mu_m, sig, n_maps, last_team in zip(
            snap["player"].astype(str),
            snap["mu_regional"].astype(float),
            snap["mu_meta"].astype(float),
            snap["sigma"].astype(float),
            n_maps_col,
            last_team_col,
        ):
            by[player] = PlayerState(
                mu_regional=float(mu_r),
                mu_meta=float(mu_m),
                sigma=float(sig),
                n_maps=int(n_maps),
                last_team=last_team,
            )
    _SNAPSHOT_BY = by
    return by


def score_player_lineups(
    blue_players: list[str],
    red_players: list[str],
    *,
    blue_roles: list[str] | None = None,
    red_roles: list[str] | None = None,
    snapshot: pd.DataFrame | None = None,
) -> dict:
    """Score a concrete roster from the player-rating snapshot (roster moves travel)."""
    cfg = PlayerEloConfig()
    if snapshot is None:
        by = _snapshot_by_player()
    else:
        by = {}
        if not snapshot.empty:
            by = {
                str(r["player"]): PlayerState(
                    mu_regional=float(r["mu_regional"]),
                    mu_meta=float(r["mu_meta"]),
                    sigma=float(r["sigma"]),
                    n_maps=int(r.get("n_maps") or 0),
                    last_team=r.get("last_team"),
                )
                for _, r in snapshot.iterrows()
            }
    br = blue_roles or ["top", "jng", "mid", "bot", "sup"]
    rr = red_roles or ["top", "jng", "mid", "bot", "sup"]
    blu = list(zip([str(x) for x in blue_players], br[: len(blue_players)]))
    red = list(zip([str(x) for x in red_players], rr[: len(red_players)]))
    mu_b, sig_b, known_b, det_b = _aggregate(by, blu, cfg)
    mu_r, sig_r, known_r, det_r = _aggregate(by, red, cfg)
    sig = math.sqrt(sig_b**2 + sig_r**2)
    mu_diff = mu_b - mu_r
    p = expected_score(mu_b, mu_r)
    shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
    p_shrunk = 0.5 + (p - 0.5) * shrink
    # Prefer time-safe Elo→WR calibration when available (avoids hot player scale)
    try:
        from lol_kills.ratings.calibrate_elo_wr import calibrated_player_p, load_calibration

        cal = load_calibration()
        if cal.get("player"):
            p_cal = calibrated_player_p(mu_diff, cal)
            p_shrunk = 0.5 + (p_cal - 0.5) * shrink
    except Exception:
        pass
    return {
        "player_mu_blue": round(mu_b, 2),
        "player_mu_red": round(mu_r, 2),
        "player_mu_diff": round(mu_diff, 2),
        "p_player_elo": round(p_shrunk, 4),
        "player_known_blue": known_b,
        "player_known_red": known_r,
        "blue_detail": det_b,
        "red_detail": det_r,
    }


def latest_roster_for_team(players: pd.DataFrame, team: str, n: int = 5) -> list[tuple[str, str]]:
    """Most recent 5-man roster (name, role) for a team from OE player rows."""
    team_n = normalize_team(team)
    # Normalize unique teamnames once (not every row) — board hot path.
    uniq = players["teamname"].dropna().astype(str).unique()
    mapping = {t: normalize_team(t) for t in uniq}
    want_raw = {t for t, nrm in mapping.items() if nrm == team_n}
    if not want_raw:
        return []
    p = players[players["teamname"].astype(str).isin(want_raw)]
    p = p[p["position"].astype(str).str.lower() != "team"]
    p = p[p["playername"].notna()]
    if p.empty or "date" not in p.columns:
        return []
    dates = pd.to_datetime(p["date"], errors="coerce")
    last_idx = dates.idxmax()
    if pd.isna(last_idx):
        return []
    last_gid = str(p.loc[last_idx, "game_uid"])
    g = p[p["game_uid"].astype(str) == last_gid].copy()
    g["_role"] = g["position"].map(_norm_role)
    order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    rows = [(str(r["playername"]), str(r["_role"])) for _, r in g.iterrows()]
    rows.sort(key=lambda x: order.get(x[1], 9))
    seen = set()
    out = []
    for name, role in rows:
        if role in seen:
            continue
        seen.add(role)
        out.append((name, role))
    return out[:n]


def latest_roster_cached(team: str, n: int = 5) -> list[tuple[str, str]]:
    key = f"{normalize_team(team)}|{n}"
    if key not in _ROSTER_CACHE:
        _ROSTER_CACHE[key] = latest_roster_for_team(load_players_cached(), team, n=n)
    return _ROSTER_CACHE[key]


def main() -> None:
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    maps_all = build_maps_frame_from_players(players)
    print(f"[player_elo] full OE maps={len(maps_all)}")
    build_player_ratings(maps_all, players)


if __name__ == "__main__":
    main()
