#!/usr/bin/env python3
"""
Multi-market draft model:
  - winner (logistic, side-aware picks + team strength)
  - total kills (ridge)
  - race-to-K via team kill expectations
  - first inhibitor (logistic; Leaguepedia proxy label)
  - first blood (proxy: early-aggression champs + team strength; marked low-confidence
    until FB labels exist in Cargo)

Also exports per-pick ΔWR% (normalized average marginal effects).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data" / "lol" / "draft_games.json"
PLAYERS = ROOT / "data" / "lol" / "draft_players.json"
OUT = ROOT / "data" / "lol" / "markets_model.json"


def sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    n, p = len(X), len(X[0])
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        xi, yi = X[i], y[i]
        for a in range(p):
            Xty[a] += xi[a] * yi
            for b in range(a, p):
                XtX[a][b] += xi[a] * xi[b]
    for a in range(p):
        for b in range(a):
            XtX[a][b] = XtX[b][a]
        if a > 0:
            XtX[a][a] += lam
    return _solve(XtX, Xty)


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        div = M[col][col]
        if abs(div) < 1e-12:
            continue
        for j in range(col, n + 1):
            M[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def logistic_ridge(
    X: list[list[float]], y: list[int], lam: float = 5.0, iters: int = 80
) -> list[float]:
    """IRLS / Newton-ish with L2 on non-intercept."""
    n, p = len(X), len(X[0])
    beta = [0.0] * p
    for _ in range(iters):
        Wz = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for i in range(n):
            eta = sum(beta[j] * X[i][j] for j in range(p))
            p_i = sigmoid(eta)
            w = max(p_i * (1 - p_i), 1e-4)
            z = eta + (y[i] - p_i) / w
            for a in range(p):
                Wz[a] += X[i][a] * w * z
                for b in range(a, p):
                    H[a][b] += X[i][a] * w * X[i][b]
        for a in range(p):
            for b in range(a):
                H[a][b] = H[b][a]
            if a > 0:
                H[a][a] += lam
        beta = _solve(H, Wz)
    return beta


def team_strength(games: list[dict]) -> dict[str, float]:
    """Empirical WR logit strength centered at 0."""
    wins: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    for g in games:
        if g.get("winner") not in (1, 2):
            continue
        t1, t2 = g["team1"], g["team2"]
        n[t1] += 1
        n[t2] += 1
        if g["winner"] == 1:
            wins[t1] += 1
        else:
            wins[t2] += 1
    strength = {}
    for t, nn in n.items():
        wr = (wins[t] + 1) / (nn + 2)  # Laplace
        strength[t] = math.log(wr / (1 - wr))
    # center
    mu = sum(strength.values()) / len(strength) if strength else 0.0
    return {t: v - mu for t, v in strength.items()}


# Early-game aggression prior for FB proxy (domain knowledge + shrinkable)
EARLY_AGGRO = {
    "Draven": 0.35,
    "Pyke": 0.40,
    "Pantheon": 0.45,
    "Renekton": 0.25,
    "Lee Sin": 0.30,
    "Nidalee": 0.25,
    "Xin Zhao": 0.30,
    "Nocturne": 0.20,
    "Ryze": -0.05,
    "Ornn": -0.25,
    "Sion": -0.20,
    "Milio": -0.20,
    "Lulu": -0.15,
    "Janna": -0.20,
    "Twisted Fate": 0.05,
    "Kalista": 0.30,
    "Thresh": 0.15,
    "Nautilus": 0.10,
    "Blitzcrank": 0.35,
    "Alistar": 0.20,
    "Rell": 0.15,
}


def build_rows(games: list[dict], players: list[dict], min_champ: int = 15):
    by_game: dict[str, list[dict]] = defaultdict(list)
    for p in players:
        by_game[p["game_id"]].append(p)

    champ_n: dict[str, int] = defaultdict(int)
    rows = []
    for g in games:
        plist = by_game.get(g["game_id"], [])
        if len(plist) < 8 or g.get("winner") not in (1, 2):
            continue
        side1 = {p["champion"] for p in plist if p.get("team") == g["team1"] and p.get("champion")}
        side2 = {p["champion"] for p in plist if p.get("team") == g["team2"] and p.get("champion")}
        if len(side1) < 4 or len(side2) < 4:
            continue
        for c in side1 | side2:
            champ_n[c] += 1
        rows.append(
            {
                "game": g,
                "side1": sorted(side1),
                "side2": sorted(side2),
                "all": sorted(side1 | side2),
            }
        )
    keep = sorted([c for c, n in champ_n.items() if n >= min_champ])
    return rows, keep, dict(champ_n)


def ame_pp(beta: float, base_p: float = 0.5) -> float:
    """Average marginal effect in percentage points at base_p."""
    return 100.0 * beta * base_p * (1 - base_p)


def fit_all(rows: list[dict], champs: list[str], strengths: dict[str, float], lam: float = 8.0):
    c_idx = {c: i for i, c in enumerate(champs)}
    # features: intercept, strength_diff, champ diffs (side1 - side2) for each champ
    p = 2 + len(champs)

    def feat(r, perspective_team1: bool = True) -> list[float]:
        g = r["game"]
        x = [0.0] * p
        x[0] = 1.0
        s1 = strengths.get(g["team1"], 0.0)
        s2 = strengths.get(g["team2"], 0.0)
        x[1] = (s1 - s2) if perspective_team1 else (s2 - s1)
        a = set(r["side1"] if perspective_team1 else r["side2"])
        b = set(r["side2"] if perspective_team1 else r["side1"])
        for c, i in c_idx.items():
            val = 0.0
            if c in a:
                val += 1.0
            if c in b:
                val -= 1.0
            x[2 + i] = val
        return x

    # Winner: y=1 if team1 wins
    Xw, yw = [], []
    for r in rows:
        Xw.append(feat(r, True))
        yw.append(1 if r["game"]["winner"] == 1 else 0)
    beta_win = logistic_ridge(Xw, yw, lam=lam)

    # First inhibitor requires an event-order label. Never substitute final
    # inhibitor totals or the map-winner model.
    Xi, yi = [], []
    for r in rows:
        fi = r["game"].get("first_inhib_side")
        if fi not in (1, 2):
            continue
        Xi.append(feat(r, True))
        yi.append(1 if fi == 1 else 0)
    beta_inhib = logistic_ridge(Xi, yi, lam=lam) if len(yi) >= 50 else None

    # Total kills ridge on presence (both sides together)
    leagues = sorted({r["game"]["league"] for r in rows})
    lg_idx = {lg: i for i, lg in enumerate(leagues)}
    n_lg = max(0, len(leagues) - 1)
    pk = 1 + n_lg + len(champs)
    Xk, yk = [], []
    for r in rows:
        x = [0.0] * pk
        x[0] = 1.0
        li = lg_idx[r["game"]["league"]]
        if li > 0:
            x[li] = 1.0
        present = set(r["all"])
        for c, i in c_idx.items():
            if c in present:
                x[1 + n_lg + i] = 1.0
        Xk.append(x)
        yk.append(float(r["game"]["total_kills"]))
    beta_kills = ridge_fit(Xk, yk, lam=40.0)

    # Team kill expectation ridge: predict team1 kills with side-aware feats
    Xt, yt = [], []
    for r in rows:
        # reuse win features style + intercept/strength for kills of side1
        x = feat(r, True)
        Xt.append(x)
        yt.append(float(r["game"]["kills1"]))
        # also flip for more data
        x2 = feat(r, False)
        Xt.append(x2)
        yt.append(float(r["game"]["kills2"]))
    beta_team_kills = ridge_fit(Xt, yt, lam=25.0)

    # Metrics
    def logloss(beta, X, y):
        s = 0.0
        for i in range(len(y)):
            p_i = sigmoid(sum(beta[j] * X[i][j] for j in range(len(beta))))
            p_i = min(max(p_i, 1e-6), 1 - 1e-6)
            s += -(y[i] * math.log(p_i) + (1 - y[i]) * math.log(1 - p_i))
        return s / len(y)

    def rmse(beta, X, y):
        s = 0.0
        for i in range(len(y)):
            pred = sum(beta[j] * X[i][j] for j in range(len(beta)))
            s += (pred - y[i]) ** 2
        return math.sqrt(s / len(y))

    win_ll = logloss(beta_win, Xw, yw)
    # accuracy
    acc = sum(
        1
        for i in range(len(yw))
        if (sigmoid(sum(beta_win[j] * Xw[i][j] for j in range(p))) >= 0.5) == bool(yw[i])
    ) / len(yw)

    champ_wr = {}
    for c, i in c_idx.items():
        b = beta_win[2 + i]
        champ_wr[c] = {
            "logit": round(b, 5),
            "delta_wr_pp_at_50": round(ame_pp(b, 0.5), 3),
            "delta_wr_pp_at_60": round(ame_pp(b, 0.6), 3),
        }

    champ_kills_presence = {
        c: round(beta_kills[1 + n_lg + i], 4) for c, i in c_idx.items()
    }
    champ_team_kills = {
        c: round(beta_team_kills[2 + i], 4) for c, i in c_idx.items()
    }
    champ_inhib = (
        {
            c: {
                "logit": round(beta_inhib[2 + i], 5),
                "delta_pp_at_50": round(ame_pp(beta_inhib[2 + i], 0.5), 3),
            }
            for c, i in c_idx.items()
        }
        if beta_inhib is not None
        else {}
    )

    return {
        "champs": champs,
        "leagues": leagues,
        "beta_win": beta_win,
        "beta_inhib": beta_inhib,
        "beta_kills": beta_kills,
        "beta_team_kills": beta_team_kills,
        "n_lg": n_lg,
        "lg_idx": lg_idx,
        "metrics": {
            "n_games": len(rows),
            "win_logloss": round(win_ll, 4),
            "win_accuracy": round(acc, 4),
            "kills_rmse": round(rmse(beta_kills, Xk, yk), 3),
            "team_kills_rmse": round(rmse(beta_team_kills, Xt, yt), 3),
            "n_inhib_labels": len(yi),
        },
        "champion_wr": champ_wr,
        "champion_kills_presence": champ_kills_presence,
        "champion_team_kills": champ_team_kills,
        "champion_inhib": champ_inhib,
        "strength_coef": round(beta_win[1], 4),
        "win_intercept": round(beta_win[0], 4),
    }


def poisson_race_prob(mu_a: float, mu_b: float, target: int, n_sims: int = 20000, seed: int = 0) -> float:
    """P(A reaches `target` kills before B), via independent Poisson totals as proxy
    for race (higher team kill mean → more likely to race first). Better: sequential
    Poisson process race using rates proportional to mu."""
    import random

    rng = random.Random(seed)
    # Race with exponential clocks: rate_a : rate_b = mu_a : mu_b
    ra = max(mu_a, 0.05)
    rb = max(mu_b, 0.05)
    a_first = 0
    for _ in range(n_sims):
        ca = cb = 0
        while ca < target and cb < target:
            # next kill by A with prob ra/(ra+rb)
            if rng.random() < ra / (ra + rb):
                ca += 1
            else:
                cb += 1
        if ca >= target:
            a_first += 1
    return a_first / n_sims


def predict_markets(
    fitted: dict,
    strengths: dict[str, float],
    *,
    team1: str,
    team2: str,
    blue: list[str],
    red: list[str],
    league: str = "LCK",
    race_targets: list[int] | None = None,
) -> dict:
    if race_targets is None:
        race_targets = [10, 15, 20]
    champs = fitted["champs"]
    c_idx = {c: i for i, c in enumerate(champs)}
    p = 2 + len(champs)

    def norm_list(xs: list[str]) -> list[str]:
        out = []
        for c in xs:
            hit = next((k for k in champs if k.lower() == c.lower()), c)
            out.append(hit)
        return out

    blue = norm_list(blue)
    red = norm_list(red)

    def side_feat(side_a: list[str], side_b: list[str], str_a: float, str_b: float) -> list[float]:
        x = [0.0] * p
        x[0] = 1.0
        x[1] = str_a - str_b
        a, b = set(side_a), set(side_b)
        for c, i in c_idx.items():
            val = 0.0
            if c in a:
                val += 1.0
            if c in b:
                val -= 1.0
            x[2 + i] = val
        return x

    s1 = strengths.get(team1, 0.0)
    s2 = strengths.get(team2, 0.0)
    # Assume team1 = blue for prediction API (document this)
    x = side_feat(blue, red, s1, s2)
    logit = sum(fitted["beta_win"][j] * x[j] for j in range(p))
    p_win1 = sigmoid(logit)

    beta_inhib = fitted.get("beta_inhib")
    logit_i = (
        sum(beta_inhib[j] * x[j] for j in range(p))
        if isinstance(beta_inhib, list)
        else None
    )
    p_inhib1 = sigmoid(logit_i) if logit_i is not None else None

    # FB proxy: team strength + early aggro draft score
    aggro1 = sum(EARLY_AGGRO.get(c, 0.0) for c in blue) + 0.35 * s1
    aggro2 = sum(EARLY_AGGRO.get(c, 0.0) for c in red) + 0.35 * s2
    # logistic on difference
    p_fb1 = sigmoid(0.15 + 0.9 * (aggro1 - aggro2))

    # kills
    n_lg = fitted["n_lg"]
    lg_idx = fitted["lg_idx"]
    pk = 1 + n_lg + len(champs)
    xk = [0.0] * pk
    xk[0] = 1.0
    if league in lg_idx and lg_idx[league] > 0:
        xk[lg_idx[league]] = 1.0
    present = set(blue) | set(red)
    for c, i in c_idx.items():
        if c in present:
            xk[1 + n_lg + i] = 1.0
    mu_total = sum(fitted["beta_kills"][j] * xk[j] for j in range(pk))
    sd_total = fitted["metrics"]["kills_rmse"]

    mu_k1 = sum(fitted["beta_team_kills"][j] * x[j] for j in range(p))
    x2 = side_feat(red, blue, s2, s1)
    mu_k2 = sum(fitted["beta_team_kills"][j] * x2[j] for j in range(p))
    # rescale so k1+k2 ≈ mu_total
    if mu_k1 + mu_k2 > 1:
        scale = mu_total / (mu_k1 + mu_k2)
        mu_k1 *= scale
        mu_k2 *= scale

    # Per-pick WR contributions for blue team picks
    pick_wr = []
    for c in blue:
        if c in fitted["champion_wr"]:
            pick_wr.append(
                {
                    "champion": c,
                    "team": team1,
                    **fitted["champion_wr"][c],
                }
            )
    for c in red:
        if c in fitted["champion_wr"]:
            # from team2 perspective the sign flips for their own pick
            row = fitted["champion_wr"][c]
            pick_wr.append(
                {
                    "champion": c,
                    "team": team2,
                    "logit": row["logit"],
                    "delta_wr_pp_at_50": row["delta_wr_pp_at_50"],
                    "delta_wr_pp_at_60": row["delta_wr_pp_at_60"],
                    "note": "effect if this champ is on YOUR team vs enemy",
                }
            )

    races = {}
    for t in race_targets:
        races[str(t)] = {
            "p_team1_first": round(poisson_race_prob(mu_k1, mu_k2, t), 4),
            "p_team2_first": round(poisson_race_prob(mu_k2, mu_k1, t, seed=t), 4),
        }

    # Normal approx for total kills
    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def p_under(line: float) -> float:
        # P(total <= floor(line)) with continuity correction
        thr = math.floor(line) + 0.5
        return norm_cdf((thr - mu_total) / max(sd_total, 1e-3))

    return {
        "team1": team1,
        "team2": team2,
        "blue": blue,
        "red": red,
        "league": league,
        "winner": {
            "p_team1": round(p_win1, 4),
            "p_team2": round(1 - p_win1, 4),
            "logit": round(logit, 4),
            "fair_odds_team1": round(1 / p_win1, 3) if p_win1 > 0 else None,
            "fair_odds_team2": round(1 / (1 - p_win1), 3) if p_win1 < 1 else None,
        },
        "first_blood": {
            "p_team1": round(p_fb1, 4),
            "p_team2": round(1 - p_fb1, 4),
            "confidence": "low",
            "method": "proxy_early_aggro_draft+team_strength (FB not in Leaguepedia Cargo)",
        },
        "first_inhibitor": {
            "p_team1": round(p_inhib1, 4) if p_inhib1 is not None else None,
            "p_team2": round(1 - p_inhib1, 4) if p_inhib1 is not None else None,
            "status": "available" if p_inhib1 is not None else "withheld",
            "method": (
                "logistic on verified event-order labels"
                if p_inhib1 is not None
                else "unavailable: final inhibitor totals do not identify event order"
            ),
        },
        "total_kills": {
            "mean": round(mu_total, 2),
            "sd": round(sd_total, 2),
            "distribution": "Normal(mean, sd) approximate",
            "p_under_25_5": round(p_under(25.5), 4),
            "p_under_34_5": round(p_under(34.5), 4),
        },
        "team_kills": {
            "team1_mean": round(mu_k1, 2),
            "team2_mean": round(mu_k2, 2),
        },
        "race_to_kills": races,
        "pick_wr_contributions": sorted(
            pick_wr, key=lambda r: -abs(r["delta_wr_pp_at_50"])
        ),
        "team_strength": {
            team1: round(s1, 4),
            team2: round(s2, 4),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-champ", type=int, default=15)
    ap.add_argument("--lam", type=float, default=8.0)
    args = ap.parse_args()

    games = json.loads(GAMES.read_text())["games"]
    players = json.loads(PLAYERS.read_text())["players"]
    if not games or games[0].get("winner") is None:
        raise SystemExit("Run: python -m lol_kills.enrich_games  first")

    strengths = team_strength(games)
    rows, champs, counts = build_rows(games, players, min_champ=args.min_champ)
    print(f"rows={len(rows)} champs={len(champs)}")
    fitted = fit_all(rows, champs, strengths, lam=args.lam)

    # serialize betas + tables
    payload = {
        "meta": {
            "source": "Leaguepedia ScoreboardGames+Players enriched",
            **fitted["metrics"],
            "n_champions": len(champs),
            "note_first_blood": "Proxy only — FirstBlood not in Cargo; replace when labels available",
        },
        "team_strengths": {k: round(v, 5) for k, v in sorted(strengths.items(), key=lambda x: -x[1])[:80]},
        "champion_wr": fitted["champion_wr"],
        "champion_kills_presence": fitted["champion_kills_presence"],
        "champion_team_kills": fitted["champion_team_kills"],
        "champion_inhib": fitted["champion_inhib"],
        "model": {
            "champs": fitted["champs"],
            "leagues": fitted["leagues"],
            "beta_win": fitted["beta_win"],
            "beta_inhib": fitted["beta_inhib"],
            "beta_kills": fitted["beta_kills"],
            "beta_team_kills": fitted["beta_team_kills"],
            "n_lg": fitted["n_lg"],
            "lg_idx": fitted["lg_idx"],
            "metrics": fitted["metrics"],
            "champion_wr": fitted["champion_wr"],
            "champion_kills_presence": fitted["champion_kills_presence"],
            "champion_team_kills": fitted["champion_team_kills"],
            "champion_inhib": fitted["champion_inhib"],
            "strength_coef": fitted["strength_coef"],
            "win_intercept": fitted["win_intercept"],
        },
        "top_wr_picks": sorted(
            (
                {"champion": c, **v}
                for c, v in fitted["champion_wr"].items()
            ),
            key=lambda r: -r["delta_wr_pp_at_50"],
        )[:20],
        "worst_wr_picks": sorted(
            (
                {"champion": c, **v}
                for c, v in fitted["champion_wr"].items()
            ),
            key=lambda r: r["delta_wr_pp_at_50"],
        )[:20],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print("metrics", fitted["metrics"])
    print("top WR", payload["top_wr_picks"][:5])
    print("worst WR", payload["worst_wr_picks"][:5])


if __name__ == "__main__":
    main()
