#!/usr/bin/env python3
"""
Player Dual-Elo → team aggregate.

Roster moves travel with the player: team strength is the mean of the five
pre-match player μs (regional + meta), not a sticky org rating.

  python3 -m lol_kills.ratings.player_elo
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR
from lol_kills.ratings.dual_elo import (
    _is_intl,
    expected_score,
    winner_relative_mov,
)
from lol_kills.ratings.player_identifiability import (
    build_player_outcome_identifiability,
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
    player_id: str | None = None
    display_name: str | None = None
    aliases: set[str] = field(default_factory=set)
    identity_source: str = "unknown"


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


@dataclass(frozen=True)
class LineupPlayer:
    identity: str
    player_id: str | None
    display_name: str
    role: str


@dataclass
class PlayerLineupResolution:
    lineups: dict[str, dict[str, list[LineupPlayer]]]
    resolved_rows: pd.DataFrame
    audit: dict[str, Any]


class PlayerIdentityError(ValueError):
    """Raised when a public name would merge distinct player identities."""


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


def _clean_identity_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none"} else text


def _lineups_by_game(players: pd.DataFrame) -> PlayerLineupResolution:
    """Resolve complete lineups without ever merging ID-bearing rows by name.

    If the provider supplies a ``playerid`` column, every published lineup is
    required to contain ten non-missing, unique IDs in the canonical five role
    slots.  A reused display name that maps to multiple IDs is also quarantined
    because downstream name-keyed surfaces cannot represent it safely.
    """

    empty_audit: dict[str, Any] = {
        "ok": False,
        "identity_mode": "unavailable",
        "stable_provider_ids": False,
        "n_player_rows": 0,
        "n_valid_maps": 0,
        "n_quarantined_maps": 0,
        "quarantined_game_uids": [],
        "quarantine_reasons": {},
        "display_name_collisions": {},
    }
    if (
        players is None
        or players.empty
        or "playername" not in players.columns
        or "side" not in players.columns
    ):
        return PlayerLineupResolution({}, pd.DataFrame(), empty_audit)

    p = players.copy()
    gcol = "game_uid" if "game_uid" in p.columns else "gameid"
    if gcol not in p.columns:
        audit = dict(empty_audit)
        audit["quarantine_reasons"] = {"missing_game_identity_column": 1}
        return PlayerLineupResolution({}, pd.DataFrame(), audit)

    p["position"] = p.get(
        "position", pd.Series("unk", index=p.index)
    ).astype(str)
    p = p[p["position"].str.casefold().ne("team")].copy()
    p["_gid"] = p[gcol].map(_clean_identity_text)
    p["_side"] = p["side"].astype(str).str.strip().str.title()
    p["_role"] = p["position"].map(_norm_role)
    p["_display_name"] = p["playername"].map(_clean_identity_text)
    p["_name_key"] = p["_display_name"].str.casefold()

    provider_id_mode = "playerid" in p.columns
    if provider_id_mode:
        p["_player_id"] = p["playerid"].map(_clean_identity_text)
        p["_player_identity"] = p["_player_id"].map(
            lambda value: f"provider:{value}" if value else ""
        )
    else:
        p["_player_id"] = ""
        p["_player_identity"] = p["_name_key"].map(
            lambda value: f"name:{value}" if value else ""
        )

    ids_by_name: dict[str, list[str]] = {}
    if provider_id_mode:
        ids_by_name = (
            p.loc[p["_player_id"].ne("") & p["_name_key"].ne("")]
            .groupby("_name_key", sort=True)["_player_id"]
            .agg(lambda values: sorted({str(value) for value in values}))
            .to_dict()
        )
    display_collisions = {
        name: ids for name, ids in ids_by_name.items() if len(ids) > 1
    }

    canonical_roles = {"top", "jng", "mid", "bot", "sup"}
    role_order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    lineups: dict[str, dict[str, list[LineupPlayer]]] = {}
    resolved_parts: list[pd.DataFrame] = []
    quarantined: dict[str, list[str]] = {}

    for gid, group in p.groupby("_gid", sort=False, dropna=False):
        game_id = str(gid)
        reasons: set[str] = set()
        if not game_id:
            reasons.add("missing_game_identity")
        if group["_display_name"].eq("").any():
            reasons.add("missing_player_name")
        if provider_id_mode and group["_player_id"].eq("").any():
            reasons.add("missing_player_id")
        if group["_name_key"].isin(display_collisions).any():
            reasons.add("display_name_maps_to_multiple_player_ids")
        if not group["_side"].isin({"Blue", "Red"}).all():
            reasons.add("invalid_side")
        if not group["_role"].isin(canonical_roles).all():
            reasons.add("invalid_role")

        valid_identity = group["_player_identity"].ne("")
        if group.loc[valid_identity, "_player_identity"].duplicated().any():
            reasons.add("duplicate_player_identity")

        by_side: dict[str, list[LineupPlayer]] = {"Blue": [], "Red": []}
        for side in ("Blue", "Red"):
            side_rows = group[group["_side"].eq(side)]
            if len(side_rows) != 5:
                reasons.add(f"{side.casefold()}_lineup_size")
                continue
            if set(side_rows["_role"]) != canonical_roles:
                reasons.add(f"{side.casefold()}_role_slots")
                continue
            ordered = side_rows.sort_values(
                "_role", key=lambda values: values.map(role_order), kind="mergesort"
            )
            by_side[side] = [
                LineupPlayer(
                    identity=str(row["_player_identity"]),
                    player_id=(
                        str(row["_player_id"])
                        if provider_id_mode
                        else None
                    ),
                    display_name=str(row["_display_name"]),
                    role=str(row["_role"]),
                )
                for _, row in ordered.iterrows()
            ]

        if reasons:
            quarantined[game_id] = sorted(reasons)
            continue
        lineups[game_id] = by_side
        resolved_parts.append(group)

    reason_counts: dict[str, int] = defaultdict(int)
    for reasons in quarantined.values():
        for reason in reasons:
            reason_counts[reason] += 1
    resolved = (
        pd.concat(resolved_parts, ignore_index=True)
        if resolved_parts
        else pd.DataFrame(columns=p.columns)
    )
    audit = {
        "ok": not quarantined,
        "identity_mode": (
            "provider_playerid"
            if provider_id_mode
            else "name_fallback_no_playerid_column"
        ),
        "stable_provider_ids": provider_id_mode,
        "n_player_rows": len(p),
        "n_valid_maps": len(lineups),
        "n_quarantined_maps": len(quarantined),
        "quarantined_game_uids": sorted(quarantined),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "display_name_collisions": display_collisions,
    }
    return PlayerLineupResolution(lineups, resolved, audit)


def _aggregate(
    states: dict[str, PlayerState],
    lineup: list[LineupPlayer],
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
    for member in lineup[:5]:
        st = states.get(member.identity)
        w = _role_w(member.role, cfg)
        if st is None:
            mu = cfg.prior_mu
            sig = cfg.sigma0
        else:
            mu = total_mu(st)
            sig = st.sigma
            known += 1
        details.append(
            {
                "player": member.display_name,
                "player_id": member.player_id,
                "player_identity": member.identity,
                "role": member.role,
                "mu": round(mu, 2),
                "sigma": round(sig, 2),
                "w": w,
            }
        )
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


def _snapshot_rows(states: dict[str, PlayerState]) -> list[dict[str, object]]:
    return [
        {
            "player": st.display_name or identity,
            "player_id": st.player_id,
            "player_identity": identity,
            "identity_source": st.identity_source,
            "player_aliases": sorted(st.aliases),
            "mu_total": total_mu(st),
            "mu_regional": st.mu_regional,
            "mu_meta": st.mu_meta,
            "sigma": st.sigma,
            "n_maps": st.n_maps,
            "last_team": st.last_team,
        }
        for identity, st in states.items()
    ]


def _run_player_elo(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig,
    checkpoint_dates: list[pd.Timestamp] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, PlayerState],
    dict[pd.Timestamp, list[dict[str, object]]],
    PlayerLineupResolution,
]:
    """Run the sequential player model and optionally capture dated states."""

    # Apply the same source-preserving competition taxonomy as team ratings so
    # player regional/meta updates cannot drift from the public team contract.
    df = canonicalize_competition_frame(maps).sort_values("date").copy().reset_index(drop=True)
    df["game_uid"] = df["game_uid"].astype(str)
    resolution = _lineups_by_game(players)
    lineups = resolution.lineups
    states: dict[str, PlayerState] = {}
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
        identity_eligible = len(blue_lu) == 5 and len(red_lu) == 5

        # inactivity + team-switch uncertainty
        for member, team_now in [
            *((member, bt) for member in blue_lu[:5]),
            *((member, rt) for member in red_lu[:5]),
        ]:
            if member.identity not in states:
                states[member.identity] = PlayerState(
                    sigma=cfg.sigma0,
                    player_id=member.player_id,
                    display_name=member.display_name,
                    aliases={member.display_name},
                    identity_source=(
                        "provider_playerid"
                        if member.player_id is not None
                        else "name_fallback_no_playerid_column"
                    ),
                )
            st = states[member.identity]
            st.display_name = member.display_name
            st.aliases.add(member.display_name)
            if d is not None and st.last_date is not None:
                months = max((d - st.last_date).days / 30.0, 0.0)
                st.sigma = min(160.0, st.sigma + cfg.sigma_month_inflate * months)
            if st.last_team and team_now and st.last_team != team_now:
                st.sigma = min(160.0, st.sigma + cfg.team_switch_sigma_bump)
            states[member.identity] = st

        mu_b, sig_b, known_b, _ = _aggregate(states, blue_lu, cfg)
        mu_r, sig_r, known_r, _ = _aggregate(states, red_lu, cfg)
        sig = math.sqrt(sig_b**2 + sig_r**2)
        p = expected_score(mu_b, mu_r)
        shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
        p_shrunk = 0.5 + (p - 0.5) * shrink
        if not identity_eligible:
            mu_b = mu_r = sig_b = sig_r = sig = float("nan")
            known_b = known_r = 0
            p = p_shrunk = float("nan")

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
                "player_identity_eligible": identity_eligible,
            }
        )

        y = row.get("y_blue_win")
        if pd.isna(y) or not identity_eligible:
            continue
        y = float(y)
        intl = _is_intl(str(row.get("league") or ""), row.get("tournament"))

        g10 = row.get("blue_golddiffat15")
        if pd.isna(g10):
            g10 = row.get("blue_golddiffat10")
        length_value = row.get("length_min")
        if pd.isna(length_value):
            length_value = (
                float(row["gamelength"]) / 60.0
                if pd.notna(row.get("gamelength"))
                else 30.0
            )
        mov = winner_relative_mov(y, g10, length_value, cfg.mov_scale)

        exp_b = expected_score(mu_b, mu_r)

        for member in blue_lu[:5]:
            st = states[member.identity]
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
            states[member.identity] = st
        for member in red_lu[:5]:
            st = states[member.identity]
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
            states[member.identity] = st

    while target_idx < len(targets):
        target = targets[target_idx]
        checkpoints[target] = _snapshot_rows(states)
        target_idx += 1
    return pd.DataFrame(rows), states, checkpoints, resolution


def build_player_rating_artifacts(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return fresh player-outcome history, snapshot, and metadata in memory."""

    cfg = cfg or PlayerEloConfig()
    out, states, _, resolution = _run_player_elo(maps, players, cfg)
    snap = _snapshot_rows(states)
    snap_df = pd.DataFrame(snap)
    if not snap_df.empty:
        snap_df = snap_df.sort_values("mu_total", ascending=False)
    diagnostic_players = resolution.resolved_rows.copy()
    if not diagnostic_players.empty:
        diagnostic_players["playername"] = diagnostic_players[
            "_player_identity"
        ]
    identifiability = build_player_outcome_identifiability(diagnostic_players)
    if not identifiability.empty:
        identifiability = identifiability.rename(
            columns={"player": "player_identity"}
        )
        display_by_identity = {
            identity: state.display_name or identity
            for identity, state in states.items()
        }
        id_by_identity = {
            identity: state.player_id for identity, state in states.items()
        }
        identifiability["outcome_identical_player_ids"] = identifiability[
            "outcome_identical_players"
        ].map(
            lambda identities: [
                id_by_identity.get(str(identity))
                for identity in identities
                if id_by_identity.get(str(identity)) is not None
            ]
        )
        identifiability["outcome_identical_players"] = identifiability[
            "outcome_identical_players"
        ].map(
            lambda identities: [
                display_by_identity.get(str(identity), str(identity))
                for identity in identities
            ]
        )
        snap_df = snap_df.merge(
            identifiability,
            on="player_identity",
            how="left",
            validate="one_to_one",
        )
    effective_maps = int(
        out.get(
            "player_identity_eligible",
            pd.Series(False, index=out.index, dtype=bool),
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )
    identity_audit = dict(resolution.audit)
    quarantined_ids = list(
        identity_audit.pop("quarantined_game_uids", [])
    )
    display_collisions = dict(
        identity_audit.pop("display_name_collisions", {})
    )
    identity_audit["quarantined_game_uid_examples"] = quarantined_ids[:20]
    identity_audit["n_display_name_collisions"] = len(display_collisions)
    identity_audit["display_name_collision_examples"] = sorted(
        display_collisions
    )[:20]
    meta = {
        # Retained for pack compatibility; this is the map-input denominator,
        # not the number of maps that updated player states.
        "n_maps": len(out),
        "n_input_maps": len(out),
        "n_identity_eligible_maps": effective_maps,
        "identity_eligible_map_rate": (
            effective_maps / len(out) if len(out) else 0.0
        ),
        "n_players": len(snap),
        "n_unique_outcome_exposure_players": int(
            identifiability["outcome_separately_identified"].sum()
        )
        if not identifiability.empty
        else 0,
        "n_shared_outcome_history_players": int(
            (~identifiability["outcome_separately_identified"].astype(bool)).sum()
        )
        if not identifiability.empty
        else 0,
        "identity_audit": identity_audit,
        "outcome_ordering_verified": False,
        "individual_skill_estimand": False,
        "config": cfg.__dict__,
        "claim_boundary": (
            "Shared team-outcome exposure signal; not an individually "
            "identified general player-skill rating."
        ),
        "note": (
            "Team μ = role-weighted mean of five player μs. Provider "
            "player IDs are the rating key when the source supplies "
            "them; incomplete or colliding ID maps are quarantined. "
            "Players with identical signed map exposure are explicitly "
            "marked as sharing an outcome-exposure group. A unique exposure "
            "history is not sufficient to identify individual skill, so "
            "public player rank ordering is withheld."
        ),
    }
    return out, snap_df, meta


def build_player_ratings(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
) -> pd.DataFrame:
    """Sequential player Elo; player ratings travel across org changes."""

    out, snap_df, meta = build_player_rating_artifacts(maps, players, cfg)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / "player_ratings.parquet"
    out.to_parquet(path, index=False)
    snap_df.to_parquet(
        FEATURES_DIR / "player_ratings_snapshot.parquet", index=False
    )
    (FEATURES_DIR / "player_ratings_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"[player_elo] wrote {path} n={len(out)} players={len(snap_df)}")
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
    current_membership: dict[str, object] | None = None,
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
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["date"].le(cutoff)]

    _, states, checkpoints, _ = _run_player_elo(
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

    current_records = build_player_records(players, current_membership)
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
        ranks: dict[str, int] = {}
        previous_score: float | None = None
        current_rank = 0
        for position, (score, player) in enumerate(eligible, start=1):
            if previous_score is None or score != previous_score:
                current_rank = position
                previous_score = score
            ranks[player] = current_rank
        return ranks

    scopes = ("all", "tier1", "tier2", "tier3")
    current_rank = {scope: order(current_rows, scope) for scope in scopes}
    previous_rank = {scope: order(previous_rows, scope) for scope in scopes}
    by_player: dict[str, dict[str, dict[str, int | None]]] = {}
    for player in current_rank["all"]:
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
        "min_games": int(min_games),
        "by_player": by_player,
        "note": (
            "Rank movement compares adjusted player Elo at Sunday 00:00 UTC "
            "snapshots; positive delta means a climb. Exact rating ties share "
            "the same competition rank."
        ),
    }


def build_maps_frame_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """One map row per OE game_uid from player rows (full history, not warehouse-filtered)."""
    pl = players.copy()
    gcol = "game_uid" if "game_uid" in pl.columns else "gameid"
    pl["_gid"] = pl[gcol].astype(str)
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
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
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
        name_keys = snap["player"].astype(str).str.strip().str.casefold()
        collisions = (
            snap.assign(_name_key=name_keys)
            .groupby("_name_key", sort=True)
            .size()
        )
        ambiguous = sorted(collisions[collisions > 1].index)
        if ambiguous:
            raise PlayerIdentityError(
                "player snapshot contains display names for multiple identities: "
                f"{ambiguous[:5]}"
            )
        n_maps_col = (
            snap["n_maps"].fillna(0).astype(int)
            if "n_maps" in snap.columns
            else pd.Series([0] * len(snap))
        )
        last_team_col = (
            snap["last_team"] if "last_team" in snap.columns else pd.Series([None] * len(snap))
        )
        player_id_col = (
            snap["player_id"]
            if "player_id" in snap.columns
            else pd.Series([None] * len(snap))
        )
        for player, player_id, mu_r, mu_m, sig, n_maps, last_team in zip(
            snap["player"].astype(str),
            player_id_col,
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
                player_id=(
                    str(player_id)
                    if player_id is not None and pd.notna(player_id)
                    else None
                ),
                display_name=player,
                aliases={player},
                identity_source=(
                    "provider_playerid"
                    if player_id is not None and pd.notna(player_id)
                    else "name_fallback_no_playerid_column"
                ),
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
            name_keys = (
                snapshot["player"].astype(str).str.strip().str.casefold()
            )
            ambiguous = sorted(
                name_keys[name_keys.duplicated(keep=False)].unique()
            )
            if ambiguous:
                raise PlayerIdentityError(
                    "provided player snapshot contains colliding display names: "
                    f"{ambiguous[:5]}"
                )
            by = {
                str(r["player"]): PlayerState(
                    mu_regional=float(r["mu_regional"]),
                    mu_meta=float(r["mu_meta"]),
                    sigma=float(r["sigma"]),
                    n_maps=int(r.get("n_maps") or 0),
                    last_team=r.get("last_team"),
                    player_id=(
                        str(r["player_id"])
                        if r.get("player_id") is not None
                        and pd.notna(r.get("player_id"))
                        else None
                    ),
                    display_name=str(r["player"]),
                    aliases={str(r["player"])},
                    identity_source=str(r.get("identity_source") or "unknown"),
                )
                for _, r in snapshot.iterrows()
            }
    br = blue_roles or ["top", "jng", "mid", "bot", "sup"]
    rr = red_roles or ["top", "jng", "mid", "bot", "sup"]
    blu = [
        LineupPlayer(
            identity=str(player),
            player_id=by.get(str(player)).player_id
            if by.get(str(player)) is not None
            else None,
            display_name=str(player),
            role=role,
        )
        for player, role in zip(
            blue_players, br[: len(blue_players)]
        )
    ]
    red = [
        LineupPlayer(
            identity=str(player),
            player_id=by.get(str(player)).player_id
            if by.get(str(player)) is not None
            else None,
            display_name=str(player),
            role=role,
        )
        for player, role in zip(
            red_players, rr[: len(red_players)]
        )
    ]
    mu_b, sig_b, known_b, det_b = _aggregate(by, blu, cfg)
    mu_r, sig_r, known_r, det_r = _aggregate(by, red, cfg)
    sig = math.sqrt(sig_b**2 + sig_r**2)
    mu_diff = mu_b - mu_r
    p = expected_score(mu_b, mu_r)
    shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
    p_shrunk = 0.5 + (p - 0.5) * shrink
    # Prefer time-safe Elo→WR calibration when available (avoids hot player scale)
    try:
        from lol_kills.ratings.calibrate_elo_wr import (
            CalibrationArtifactError,
            calibrated_player_p,
            load_calibration,
        )

        cal = load_calibration()
        if cal.get("player"):
            p_cal = calibrated_player_p(mu_diff, cal)
            p_shrunk = 0.5 + (p_cal - 0.5) * shrink
    except CalibrationArtifactError:
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
