#!/usr/bin/env python3
"""
Fit champion presence effects on total kills.

Model (per game):
  total_kills ≈ μ_league + Σ_c β_c * 1[champion c is in the draft]

Ridge-regularized least squares so rare champs shrink toward 0.
Also stores role-conditioned effects and mean totals when champ present vs absent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES_IN = ROOT / "data" / "lol" / "draft_games.json"
PLAYERS_IN = ROOT / "data" / "lol" / "draft_players.json"
OUT = ROOT / "data" / "lol" / "draft_model.json"


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Solve (X'X + λI) β = X'y. X includes intercept column."""
    n = len(X)
    p = len(X[0])
    # XtX, Xty
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for a in range(p):
            Xty[a] += xi[a] * yi
            xa = xi[a]
            row = XtX[a]
            for b in range(a, p):
                row[b] += xa * xi[b]
    for a in range(p):
        for b in range(a):
            XtX[a][b] = XtX[b][a]
        if a > 0:  # don't regularize intercept
            XtX[a][a] += lam
    return _solve(XtX, Xty)


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivot."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        if abs(div) < 1e-12:
            continue
        for j in range(col, n + 1):
            M[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def build_dataset(games: list[dict], players: list[dict], min_champ_games: int = 15):
    by_game: dict[str, list[dict]] = defaultdict(list)
    for p in players:
        by_game[p["game_id"]].append(p)

    rows = []
    champ_counts: dict[str, int] = defaultdict(int)
    for g in games:
        plist = by_game.get(g["game_id"], [])
        champs = sorted({p["champion"] for p in plist if p.get("champion")})
        if len(champs) < 8:  # incomplete draft row
            continue
        for c in champs:
            champ_counts[c] += 1
        role_map = {}
        for p in plist:
            if p.get("champion") and p.get("role"):
                role_map[f"{p['champion']}|{p['role']}"] = True
        rows.append(
            {
                "game_id": g["game_id"],
                "league": g["league"],
                "total_kills": g["total_kills"],
                "champs": champs,
                "roles": list(role_map.keys()),
                "length_min": g.get("length_min"),
            }
        )

    keep = sorted([c for c, n in champ_counts.items() if n >= min_champ_games])
    return rows, keep, dict(champ_counts)


def fit_model(rows: list[dict], champions: list[str], lam: float = 25.0) -> dict:
    leagues = sorted({r["league"] for r in rows})
    # features: intercept + league dummies (drop first) + champion presence
    league_idx = {lg: i for i, lg in enumerate(leagues)}
    champ_idx = {c: i for i, c in enumerate(champions)}
    n_league_d = max(0, len(leagues) - 1)
    p = 1 + n_league_d + len(champions)

    X: list[list[float]] = []
    y: list[float] = []
    for r in rows:
        x = [0.0] * p
        x[0] = 1.0
        lg = r["league"]
        li = league_idx[lg]
        if li > 0:
            x[li] = 1.0  # leagues[1..] → columns 1..n_league_d
        present = set(r["champs"])
        for c, ci in champ_idx.items():
            if c in present:
                x[1 + n_league_d + ci] = 1.0
        X.append(x)
        y.append(float(r["total_kills"]))

    beta = ridge_fit(X, y, lam=lam)

    # Predictions / RMSE
    sse = 0.0
    for i, r in enumerate(rows):
        pred = sum(X[i][j] * beta[j] for j in range(p))
        sse += (pred - y[i]) ** 2
    rmse = math.sqrt(sse / len(rows)) if rows else 0.0
    baseline = sum(y) / len(y)
    sst = sum((yi - baseline) ** 2 for yi in y)
    r2 = 1 - sse / sst if sst > 0 else 0.0

    intercept = beta[0]
    league_effects = {leagues[0]: 0.0}
    for i, lg in enumerate(leagues[1:], start=1):
        league_effects[lg] = round(beta[i], 4)

    champ_effects = {}
    for c, ci in champ_idx.items():
        champ_effects[c] = round(beta[1 + n_league_d + ci], 4)

    # Univariate presence deltas (descriptive)
    uni = {}
    for c in champions:
        with_c = [r["total_kills"] for r in rows if c in r["champs"]]
        without = [r["total_kills"] for r in rows if c not in r["champs"]]
        if len(with_c) < 10 or not without:
            continue
        uni[c] = {
            "n": len(with_c),
            "mean_when_present": round(sum(with_c) / len(with_c), 3),
            "mean_when_absent": round(sum(without) / len(without), 3),
            "delta": round(sum(with_c) / len(with_c) - sum(without) / len(without), 3),
        }

    return {
        "intercept": round(intercept, 4),
        "league_effects": league_effects,
        "champion_effects": champ_effects,
        "univariate": uni,
        "champions": champions,
        "leagues": leagues,
        "lam": lam,
        "n_games": len(rows),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "baseline_mean": round(baseline, 3),
    }


def predict_total(
    model: dict,
    champions: list[str],
    league: str | None = None,
) -> dict:
    """Expected total kills given a full or partial draft."""
    mu = model["intercept"]
    if league and league in model["league_effects"]:
        mu += model["league_effects"][league]
    effects = model["champion_effects"]
    used = []
    missing = []
    for c in champions:
        # fuzzy: exact match first
        if c in effects:
            mu += effects[c]
            used.append({"champion": c, "effect": effects[c]})
        else:
            # case-insensitive
            hit = next((k for k in effects if k.lower() == c.lower()), None)
            if hit:
                mu += effects[hit]
                used.append({"champion": hit, "effect": effects[hit]})
            else:
                missing.append(c)
    # Uncertainty: use model RMSE as predictive sd (conservative)
    sd = model["rmse"]
    return {
        "expected_total": round(mu, 2),
        "sd": sd,
        "league": league,
        "n_champs_applied": len(used),
        "effects": sorted(used, key=lambda x: -abs(x["effect"])),
        "unknown_champions": missing,
        "baseline_mean": model["baseline_mean"],
        "delta_vs_baseline": round(mu - model["baseline_mean"], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-champ-games", type=int, default=20)
    ap.add_argument("--lam", type=float, default=30.0)
    args = ap.parse_args()

    games = json.loads(GAMES_IN.read_text())["games"]
    players = json.loads(PLAYERS_IN.read_text())["players"]
    rows, champs, counts = build_dataset(games, players, min_champ_games=args.min_champ_games)
    print(f"usable games={len(rows)} champions_kept={len(champs)}")
    by_lg: dict[str, int] = defaultdict(int)
    for r in rows:
        by_lg[r["league"]] += 1
    print("by league", dict(by_lg))

    model = fit_model(rows, champs, lam=args.lam)
    # attach top/bottom effects
    ranked = sorted(model["champion_effects"].items(), key=lambda x: x[1])
    payload = {
        "meta": {
            "source": "Leaguepedia ScoreboardGames+ScoreboardPlayers",
            "n_games": model["n_games"],
            "n_champions": len(champs),
            "by_league": dict(by_lg),
            "min_champ_games": args.min_champ_games,
            "lam": args.lam,
            "rmse": model["rmse"],
            "r2": model["r2"],
        },
        "model": model,
        "bloodiest": [{"champion": c, "effect": e} for c, e in ranked[::-1][:25]],
        "safest": [{"champion": c, "effect": e} for c, e in ranked[:25]],
        "champ_game_counts": {c: counts[c] for c in champs},
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} rmse={model['rmse']} r2={model['r2']}")
    print("bloodiest", payload["bloodiest"][:8])
    print("safest", payload["safest"][:8])


if __name__ == "__main__":
    main()
