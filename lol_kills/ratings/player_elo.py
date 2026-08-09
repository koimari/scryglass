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

import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import canonicalize_competition_frame, is_team_affiliation_league
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import DualEloConfig, _is_intl, expected_score

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

    rows = []
    for _, row in df.iterrows():
        gid = str(row.get("game_uid") or "")
        d = pd.Timestamp(row["date"]) if pd.notna(row.get("date")) else None
        capture_before(d)
        blue_lu = lineups.get(gid, {}).get("Blue") or []
        red_lu = lineups.get(gid, {}).get("Red") or []
        bt = normalize_team(str(row.get("blue_team") or row.get("blue_teamname") or ""))
        rt = normalize_team(str(row.get("red_team") or row.get("red_teamname") or ""))

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
                "date": row.get("date"),
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

        y = row.get("y_blue_win")
        if pd.isna(y):
            continue
        y = float(y)
        intl = _is_intl(str(row.get("league") or ""), row.get("tournament"))

        g10 = row.get("blue_golddiffat15")
        if pd.isna(g10):
            g10 = row.get("blue_golddiffat10")
        length = row.get("length_min") or (
            float(row["gamelength"]) / 60.0 if pd.notna(row.get("gamelength")) else 30.0
        )
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
            league = str(row.get("league") or "")
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
            league = str(row.get("league") or "")
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

    snap = _snapshot_rows(states, recent_mus)
    if player_records is not None:
        for row in snap:
            record = player_records.get(str(row["player"]))
            if record is None:
                continue
            row["last_team"] = record.get("current_team")
            row["home_league"] = record.get("current_league") or "UNKNOWN"
    snap_df = pd.DataFrame(snap).sort_values("mu_total", ascending=False)
    snap_df.to_parquet(destination / "player_ratings_snapshot.parquet", index=False)
    (destination / "player_ratings_meta.json").write_text(
        json.dumps(
            {
                "n_maps": len(out),
                "n_players": len(snap),
                "config": cfg.__dict__,
                "note": (
                    "DESCRIPTIVE BASELINE: shared team-result updates use fixed role weights. "
                    "This rating does not identify individual causal contribution. "
                    "Team μ is the role-weighted mean of five player μ values."
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


def build_player_weekly_ranks(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
    *,
    as_of: pd.Timestamp | None = None,
    min_games: int = 20,
    player_records: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Return current ranks and movement from the preceding Sunday snapshot.

    The player ladder is still the current sequential Elo snapshot.  The
    movement baseline is deliberately discrete: it is captured at Sunday
    00:00 UTC and compared with the prior Sunday, which makes rank changes
    auditable and avoids a noisy day-to-day pseudo-trend.
    """

    cfg = cfg or PlayerEloConfig()
    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    frame = maps.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    if as_of is not None:
        frame = frame[frame["date"].le(cutoff)]

    _, states, checkpoints, _recent_mus = _run_player_elo(
        frame,
        players,
        cfg,
        checkpoint_dates=[previous_start],
    )
    current_rows = _snapshot_rows(states)
    previous_rows = checkpoints.get(previous_start, [])

    # Current affiliation is the publication filter.  Historical matches in a
    # different circuit remain evidence for the rating but cannot place a
    # developmental player in the current Tier 1 board.
    from lol_kills.export.pack_records import build_player_records

    current_records = dict(player_records) if player_records is not None else build_player_records(players)
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
    by_player: dict[str, dict[str, dict[str, int | None]]] = {}
    for player, rank in current_rank["all"].items():
        values: dict[str, dict[str, int | None]] = {}
        for scope in scopes:
            current = current_rank[scope].get(player)
            if current is None:
                continue
            prior = previous_rank[scope].get(player)
            values[scope] = {
                "rank": current,
                "delta": (prior - current) if prior is not None else None,
            }
        by_player[player] = values

    return {
        "as_of": f"{week_start.isoformat()}Z",
        "previous_as_of": f"{previous_start.isoformat()}Z",
        "current_through": f"{cutoff.isoformat()}Z",
        "min_games": int(min_games),
        "by_player": by_player,
        "note": "Rank movement compares adjusted player Elo at Sunday 00:00 UTC snapshots; positive delta means a climb.",
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
