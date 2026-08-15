"""
Dual rating system: regional μ + international/meta μ, with σ uncertainty.

Inspired by PandaSkill / dual-Elo esports predictors.
Pre-match ratings only (no leakage).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import classify_competition
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR
from lol_kills.ratings.momentum_config import (
    DEFAULT_MOMENTUM_SCALE,
    DEFAULT_MOMENTUM_WINDOW_GAMES,
    registered_momentum_bundle,
    selected_momentum_configuration,
)


def _is_intl(league: str, tournament: str | None = None) -> bool:
    return classify_competition(league, tournament).is_international


@dataclass
class TeamState:
    mu_regional: float = 1500.0
    mu_meta: float = 0.0
    sigma: float = 80.0
    last_date: pd.Timestamp | None = None
    lineup_hash: str | None = None
    momentum_history: list[float] = field(default_factory=list)


@dataclass
class DualEloConfig:
    k_regional: float = 20.0
    k_meta: float = 12.0
    sigma0: float = 80.0
    sigma_min: float = 25.0
    sigma_month_inflate: float = 0.8
    roster_sigma_bump: float = 15.0
    mov_scale: float = 1.0  # gold-diff / length scaling cap
    momentum_window_games: int = DEFAULT_MOMENTUM_WINDOW_GAMES
    momentum_scale: float = DEFAULT_MOMENTUM_SCALE

    def __post_init__(self) -> None:
        if (
            isinstance(self.momentum_window_games, bool)
            or not isinstance(self.momentum_window_games, int)
            or self.momentum_window_games < 0
        ):
            raise ValueError("momentum_window_games must be a non-negative integer")
        try:
            scale = float(self.momentum_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("momentum_scale must be a finite non-negative value") from exc
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("momentum_scale must be a finite non-negative value")
        self.momentum_scale = scale


def expected_score(mu_a: float, mu_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((mu_b - mu_a) / 400.0))


def total_mu(st: TeamState) -> float:
    return st.mu_regional + st.mu_meta


def _momentum_residual(st: TeamState, cfg: DualEloConfig) -> float:
    if cfg.momentum_window_games <= 0 or not st.momentum_history:
        return 0.0
    return float(np.mean(st.momentum_history[-cfg.momentum_window_games :]))


def _append_momentum(st: TeamState, residual: float, cfg: DualEloConfig) -> None:
    if cfg.momentum_window_games <= 0:
        return
    st.momentum_history.append(float(residual))
    if len(st.momentum_history) > cfg.momentum_window_games:
        del st.momentum_history[:-cfg.momentum_window_games]


def apply_team_momentum_snapshot(
    snapshot: pd.DataFrame,
    sequential_snapshot: pd.DataFrame,
    cfg: DualEloConfig,
) -> pd.DataFrame:
    """Attach sequential momentum to the public team rating scale.

    The public team ladder keeps its hierarchical base rating.  The sequential
    rating run supplies only the pre-map residual state.  This keeps the two
    estimands separate while allowing an explicitly enabled research run to
    serialize an effective ``mu_total``.
    """

    output = snapshot.copy()
    if output.empty:
        return output
    if "mu_total" not in output.columns:
        raise ValueError("team snapshot has no mu_total column")
    base = pd.to_numeric(output["mu_total"], errors="coerce")
    if base.isna().any():
        raise ValueError("team snapshot has non-numeric mu_total values")

    residual_by_team: dict[str, float] = {}
    if sequential_snapshot is not None and not sequential_snapshot.empty:
        for _, row in sequential_snapshot.iterrows():
            team = normalize_team(str(row.get("team") or "")).casefold()
            if not team:
                continue
            value = row.get("momentum_residual")
            if pd.notna(value):
                residual_by_team[team] = float(value)

    key_column = "team_key" if "team_key" in output.columns else "team"
    keys = output[key_column].map(
        lambda value: normalize_team(str(value or "")).casefold()
    )
    residual = keys.map(residual_by_team).fillna(0.0).astype(float)
    points = residual * float(cfg.momentum_scale)
    output["mu_base_total"] = base.astype(float)
    output["momentum_residual"] = residual
    output["mu_effective"] = base + points
    output["mu_total"] = output["mu_effective"]
    if "rating_p10" in output.columns:
        output["rating_p10"] = pd.to_numeric(output["rating_p10"], errors="coerce") + points
    return output


def build_dual_ratings(
    maps: pd.DataFrame,
    cfg: DualEloConfig | None = None,
    lineup_by_game: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Sequential dual Elo. Returns frame aligned to maps index with pre-match features.
    """
    cfg = cfg or DualEloConfig()
    sort_columns = ["date"] + (["game_uid"] if "game_uid" in maps.columns else [])
    df = maps.sort_values(sort_columns, kind="mergesort").copy().reset_index(drop=True)
    states: dict[str, TeamState] = defaultdict(lambda: TeamState(sigma=cfg.sigma0))

    rows = []
    for _, row in df.iterrows():
        bt = normalize_team(str(row.get("blue_team") or ""))
        rt = normalize_team(str(row.get("red_team") or ""))
        d = pd.Timestamp(row["date"]) if pd.notna(row.get("date")) else None
        sb, sr = states[bt], states[rt]

        # inactivity inflate
        for st in (sb, sr):
            if d is not None and st.last_date is not None:
                months = max((d - st.last_date).days / 30.0, 0.0)
                st.sigma = min(150.0, st.sigma + cfg.sigma_month_inflate * months)

        # roster change
        gid = str(row.get("game_uid") or "")
        if lineup_by_game:
            hb = lineup_by_game.get(f"{gid}|{bt}")
            hr = lineup_by_game.get(f"{gid}|{rt}")
            if hb and sb.lineup_hash and hb != sb.lineup_hash:
                sb.sigma = min(150.0, sb.sigma + cfg.roster_sigma_bump)
            if hr and sr.lineup_hash and hr != sr.lineup_hash:
                sr.sigma = min(150.0, sr.sigma + cfg.roster_sigma_bump)
            if hb:
                sb.lineup_hash = hb
            if hr:
                sr.lineup_hash = hr

        base_mu_b, base_mu_r = total_mu(sb), total_mu(sr)
        momentum_b = cfg.momentum_scale * _momentum_residual(sb, cfg)
        momentum_r = cfg.momentum_scale * _momentum_residual(sr, cfg)
        mu_b, mu_r = base_mu_b + momentum_b, base_mu_r + momentum_r
        sig = math.sqrt(sb.sigma**2 + sr.sigma**2)
        p_base = expected_score(base_mu_b, base_mu_r)
        p = expected_score(mu_b, mu_r)
        # uncertainty shrink toward 0.5
        shrink = 1.0 / (1.0 + (sig / 120.0) ** 2)
        p_shrunk = 0.5 + (p - 0.5) * shrink

        rows.append(
            {
                "game_uid": gid,
                "date": row.get("date"),
                "blue_team": bt,
                "red_team": rt,
                "mu_blue": mu_b,
                "mu_red": mu_r,
                "mu_diff": mu_b - mu_r,
                "mu_base_blue": base_mu_b,
                "mu_base_red": base_mu_r,
                "momentum_blue": momentum_b,
                "momentum_red": momentum_r,
                "momentum_diff": momentum_b - momentum_r,
                "mu_regional_blue": sb.mu_regional,
                "mu_regional_red": sr.mu_regional,
                "mu_meta_blue": sb.mu_meta,
                "mu_meta_red": sr.mu_meta,
                "sigma_blue": sb.sigma,
                "sigma_red": sr.sigma,
                "sigma_pair": sig,
                "p_dual_elo": p_shrunk,
                "p_dual_elo_raw": p,
                "p_dual_elo_base": 0.5 + (p_base - 0.5) * shrink,
                "p_dual_elo_base_raw": p_base,
                "momentum_window_games": cfg.momentum_window_games,
                "momentum_scale": cfg.momentum_scale,
            }
        )

        y = row.get("y_blue_win")
        if pd.isna(y):
            continue
        y = float(y)
        intl = _is_intl(str(row.get("league") or ""), row.get("tournament"))

        # MoV from gold diff if present (same-game OK for *update* after features recorded)
        g10 = row.get("blue_golddiffat15")
        if pd.isna(g10):
            g10 = row.get("blue_golddiffat10")
        length = row.get("length_min") or (float(row["gamelength"]) / 60.0 if pd.notna(row.get("gamelength")) else 30.0)
        mov = 1.0
        if pd.notna(g10) and length:
            mov = 1.0 + cfg.mov_scale * math.tanh(float(g10) / (200.0 * max(float(length), 1.0)))

        # The observed result updates the skill rating from the effective
        # pre-map probability. The residual history stays anchored to base
        # skill so the transient state cannot feed back into itself.
        exp_b = p
        if intl:
            sb.mu_meta += cfg.k_meta * mov * (y - exp_b)
            sr.mu_meta += cfg.k_meta * mov * ((1 - y) - (1 - exp_b))
        else:
            sb.mu_regional += cfg.k_regional * mov * (y - exp_b)
            sr.mu_regional += cfg.k_regional * mov * ((1 - y) - (1 - exp_b))

        _append_momentum(sb, y - p_base, cfg)
        _append_momentum(sr, (1 - y) - (1 - p_base), cfg)

        # sigma shrink after observed game
        sb.sigma = max(cfg.sigma_min, sb.sigma * 0.98)
        sr.sigma = max(cfg.sigma_min, sr.sigma * 0.98)
        if d is not None:
            sb.last_date = d
            sr.last_date = d
        states[bt], states[rt] = sb, sr

    out = pd.DataFrame(rows)
    destination = Path(output_dir or FEATURES_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "ratings.parquet"
    out.to_parquet(path, index=False)

    snap = [
        {
            "team": t,
            "mu_base_total": total_mu(s),
            "mu_total": total_mu(s) + cfg.momentum_scale * _momentum_residual(s, cfg),
            "mu_effective": total_mu(s) + cfg.momentum_scale * _momentum_residual(s, cfg),
            "momentum_residual": _momentum_residual(s, cfg),
            "mu_regional": s.mu_regional,
            "mu_meta": s.mu_meta,
            "sigma": s.sigma,
        }
        for t, s in states.items()
    ]
    snap_df = pd.DataFrame(snap).sort_values("mu_effective", ascending=False)
    snap_df.to_parquet(destination / "ratings_dual_snapshot.parquet", index=False)
    snap_df.to_parquet(destination / "ratings_snapshot.parquet", index=False)
    (destination / "ratings_meta.json").write_text(
        json.dumps(
            {
                "n_maps": len(out),
                "n_teams": len(snap),
                "config": cfg.__dict__,
                "momentum": selected_momentum_configuration(
                    window_games=cfg.momentum_window_games,
                    scale=cfg.momentum_scale,
                ),
                "registered_momentum": registered_momentum_bundle(),
            },
            indent=2,
        )
    )
    print(f"[ratings] wrote {path} n={len(out)} teams={len(snap)}")
    return out


def lineup_hashes_from_players(players: pd.DataFrame) -> dict[str, str]:
    """Map game_uid|team → sorted champ hash (or player names if present)."""
    if players is None or players.empty:
        return {}
    out: dict[str, str] = {}
    key_col = "champion" if "champion" in players.columns else None
    name_col = "playername" if "playername" in players.columns else None
    gcol = "game_uid" if "game_uid" in players.columns else "gameid"
    tcol = "teamname" if "teamname" in players.columns else "team"
    groups = players.groupby([gcol, tcol])
    for (gid, team), g in groups:
        team_n = normalize_team(str(team))
        if name_col and g[name_col].notna().any():
            parts = sorted(str(x) for x in g[name_col].dropna().unique())
        elif key_col:
            parts = sorted(str(x) for x in g[key_col].dropna().unique())
        else:
            continue
        out[f"{gid}|{team_n}"] = "|".join(parts)
    return out


def main() -> None:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    players_path = PARQUET_DIR / "players.parquet"
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()
    hashes = lineup_hashes_from_players(players)
    build_dual_ratings(maps, lineup_by_game=hashes)


if __name__ == "__main__":
    main()
