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
import bisect
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GAMES_IN = ROOT / "data" / "lol" / "draft_games.json"
PLAYERS_IN = ROOT / "data" / "lol" / "draft_players.json"
WAREHOUSE_MAPS_IN = ROOT / "data" / "lol" / "warehouse" / "parquet" / "maps.parquet"
OUT = ROOT / "data" / "lol" / "draft_model.json"

SCHEMA_VERSION = "scryglass.total-kills-draft.v2"
SUPPORTED_LEAGUES = ("LCK", "LPL", "LEC", "LCS", "CBLOL")
TRAIN_SERIES_FRACTION = 0.60
CALIBRATION_SERIES_FRACTION = 0.20
FRESHNESS_LIMIT_DAYS = 14
MIN_CALIBRATION_GAMES = 150
MIN_TEST_GAMES = 200
MIN_LEAGUE_TEST_GAMES = 25
MAX_CDF_CALIBRATION_ERROR = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _series_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("series_id") or "").strip()
    if explicit:
        return explicit
    game_id = str(row.get("game_id") or "").strip()
    if game_id and "_" in game_id:
        return game_id.rsplit("_", 1)[0]
    raise ValueError("row is missing a series identity")


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
                "series_id": str(g["game_id"]).rsplit("_", 1)[0],
                "date": g["date"],
                "league": g["league"],
                "patch": str(g.get("patch") or ""),
                "total_kills": g["total_kills"],
                "champs": champs,
                "roles": list(role_map.keys()),
                "length_min": g.get("length_min"),
            }
        )

    keep = sorted([c for c, n in champ_counts.items() if n >= min_champ_games])
    return rows, keep, dict(champ_counts)


def build_warehouse_dataset(
    path: Path = WAREHOUSE_MAPS_IN,
    leagues: tuple[str, ...] = SUPPORTED_LEAGUES,
) -> list[dict[str, Any]]:
    """Load complete professional drafts from the reconciled warehouse."""
    import pandas as pd

    wanted = [
        "game_uid",
        "date",
        "league",
        "patch",
        "blue_team",
        "red_team",
        "total_kills",
        *[f"blue_pick{i}" for i in range(1, 6)],
        *[f"red_pick{i}" for i in range(1, 6)],
    ]
    frame = pd.read_parquet(path, columns=wanted)
    frame = frame[frame["league"].astype(str).str.upper().isin(leagues)].copy()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        if pd.isna(record.get("date")) or pd.isna(record.get("total_kills")):
            continue
        champions = []
        for side in ("blue", "red"):
            for index in range(1, 6):
                value = record.get(f"{side}_pick{index}")
                champions.append("" if pd.isna(value) else str(value).strip())
        if len(set(champions)) != 10 or any(not champion for champion in champions):
            continue
        date = pd.Timestamp(record["date"])
        if date.tzinfo is not None:
            date = date.tz_convert("UTC").tz_localize(None)
        blue_team = str(record.get("blue_team") or "").strip()
        red_team = str(record.get("red_team") or "").strip()
        date_text = date.isoformat()
        series_key = "|".join(
            [
                date.strftime("%Y-%m-%d"),
                str(record["league"]).upper(),
                *sorted((blue_team, red_team)),
            ]
        )
        rows.append(
            {
                "game_id": str(record["game_uid"]),
                "series_id": series_key,
                "date": date_text,
                "league": str(record["league"]).upper(),
                "patch": str(record.get("patch") or ""),
                "total_kills": float(record["total_kills"]),
                "champs": sorted(champions),
                "roles": [],
                "blue_team": blue_team,
                "red_team": red_team,
            }
        )
    if not rows:
        raise ValueError(f"no complete total-kills drafts found in {path}")
    return rows


def champion_inventory(
    rows: list[dict[str, Any]], min_champ_games: int
) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for champion in set(row["champs"]):
            counts[champion] += 1
    kept = sorted(champion for champion, count in counts.items() if count >= min_champ_games)
    return kept, dict(counts)


def chronological_series_split(
    rows: list[dict[str, Any]],
    train_fraction: float = TRAIN_SERIES_FRACTION,
    calibration_fraction: float = CALIBRATION_SERIES_FRACTION,
) -> dict[str, list[dict[str, Any]]]:
    """Split whole match series in time order into train/calibration/test."""
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[_series_id(row)].append(row)
    ordered = sorted(
        by_series.items(),
        key=lambda item: (
            max(_parse_time(row["date"]) for row in item[1]),
            item[0],
        ),
    )
    n_series = len(ordered)
    if n_series < 15:
        raise ValueError("at least 15 series are required for chronological evaluation")
    train_end = max(1, int(n_series * train_fraction))
    calibration_end = max(train_end + 1, int(n_series * (train_fraction + calibration_fraction)))
    calibration_end = min(calibration_end, n_series - 1)

    def flatten(part: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        return [row for _, series_rows in part for row in series_rows]

    return {
        "train": flatten(ordered[:train_end]),
        "calibration": flatten(ordered[train_end:calibration_end]),
        "test": flatten(ordered[calibration_end:]),
    }


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


def _predict_mean(model: dict[str, Any], row: dict[str, Any]) -> float:
    mean = float(model["intercept"])
    mean += float(model["league_effects"].get(row["league"], 0.0))
    effects = model["champion_effects"]
    mean += sum(float(effects.get(champion, 0.0)) for champion in row["champs"])
    return mean


def _regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("metrics require equally sized non-empty arrays")
    errors = [prediction - outcome for outcome, prediction in zip(actual, predicted)]
    mse = sum(error * error for error in errors) / len(errors)
    mean_actual = sum(actual) / len(actual)
    sst = sum((outcome - mean_actual) ** 2 for outcome in actual)
    sse = sum(error * error for error in errors)
    return {
        "rmse": round(math.sqrt(mse), 4),
        "mae": round(sum(abs(error) for error in errors) / len(errors), 4),
        "r2": round(1.0 - sse / sst, 4) if sst > 0 else 0.0,
    }


def _league_baseline(train: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[float]:
    global_mean = sum(float(row["total_kills"]) for row in train) / len(train)
    totals: dict[str, list[float]] = defaultdict(list)
    for row in train:
        totals[row["league"]].append(float(row["total_kills"]))
    means = {league: sum(values) / len(values) for league, values in totals.items()}
    return [means.get(row["league"], global_mean) for row in rows]


def _nearest_rank(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantiles require non-empty values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _cdf_calibration(
    calibration_residuals: list[float],
    test_residuals: list[float],
) -> dict[str, Any]:
    points = []
    for nominal in (0.10, 0.25, 0.50, 0.75, 0.90):
        threshold = _nearest_rank(calibration_residuals, nominal)
        observed = sum(residual <= threshold for residual in test_residuals) / len(test_residuals)
        points.append(
            {
                "nominal": nominal,
                "observed": round(observed, 4),
                "absolute_error": round(abs(observed - nominal), 4),
                "residual_threshold": round(threshold, 4),
            }
        )
    max_error = max(point["absolute_error"] for point in points)
    return {
        "points": points,
        "max_absolute_error": round(max_error, 4),
        "threshold": MAX_CDF_CALIBRATION_ERROR,
        "passed": max_error <= MAX_CDF_CALIBRATION_ERROR,
    }


def _split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series = sorted({_series_id(row) for row in rows})
    encoded = "\n".join(series).encode("utf-8")
    dates = [_parse_time(row["date"]) for row in rows]
    return {
        "games": len(rows),
        "series": len(series),
        "start": min(dates).isoformat(),
        "end": max(dates).isoformat(),
        "series_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "by_league": dict(sorted(_counts(rows, "league").items())),
        "by_patch": dict(sorted(_counts(rows, "patch").items())),
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(field) or "")
        counts[value] += 1
    return dict(counts)


def evaluate_chronological_holdout(
    rows: list[dict[str, Any]],
    *,
    min_champ_games: int,
    lam: float,
) -> dict[str, Any]:
    split = chronological_series_split(rows)
    train = split["train"]
    calibration = split["calibration"]
    test = split["test"]
    champions, _ = champion_inventory(train, min_champ_games)
    train_model = fit_model(train, champions, lam=lam)

    calibration_predictions = [_predict_mean(train_model, row) for row in calibration]
    calibration_residuals = [
        float(row["total_kills"]) - prediction
        for row, prediction in zip(calibration, calibration_predictions)
    ]
    test_predictions = [_predict_mean(train_model, row) for row in test]
    test_actual = [float(row["total_kills"]) for row in test]
    test_residuals = [
        outcome - prediction for outcome, prediction in zip(test_actual, test_predictions)
    ]
    model_metrics = _regression_metrics(test_actual, test_predictions)
    baseline_predictions = _league_baseline(train, test)
    baseline_metrics = _regression_metrics(test_actual, baseline_predictions)
    calibration_report = _cdf_calibration(calibration_residuals, test_residuals)

    by_league: dict[str, Any] = {}
    for league in sorted({row["league"] for row in test}):
        indices = [index for index, row in enumerate(test) if row["league"] == league]
        actual = [test_actual[index] for index in indices]
        predicted = [test_predictions[index] for index in indices]
        residuals = [test_residuals[index] for index in indices]
        metrics = _regression_metrics(actual, predicted)
        baseline = _regression_metrics(
            actual, [baseline_predictions[index] for index in indices]
        )
        cdf = _cdf_calibration(calibration_residuals, residuals)
        passed = (
            len(indices) >= MIN_LEAGUE_TEST_GAMES
            and metrics["rmse"] <= baseline["rmse"]
            and cdf["passed"]
        )
        by_league[league] = {
            "n": len(indices),
            "model": metrics,
            "league_mean_baseline": baseline,
            "cdf_calibration": cdf,
            "predictive_probability_supported": passed,
        }

    pooled_passed = (
        len(calibration) >= MIN_CALIBRATION_GAMES
        and len(test) >= MIN_TEST_GAMES
        and model_metrics["rmse"] <= baseline_metrics["rmse"]
        and calibration_report["passed"]
    )
    unsupported_leagues = sorted(
        league
        for league, report in by_league.items()
        if report["predictive_probability_supported"] is not True
    )
    global_passed = pooled_passed and not unsupported_leagues
    residual_mean = sum(calibration_residuals) / len(calibration_residuals)
    residual_sd = math.sqrt(
        sum((value - residual_mean) ** 2 for value in calibration_residuals)
        / max(1, len(calibration_residuals) - 1)
    )
    blockers = []
    if len(calibration) < MIN_CALIBRATION_GAMES:
        blockers.append("insufficient_calibration_games")
    if len(test) < MIN_TEST_GAMES:
        blockers.append("insufficient_test_games")
    if model_metrics["rmse"] > baseline_metrics["rmse"]:
        blockers.append("heldout_rmse_does_not_beat_league_mean")
    if not calibration_report["passed"]:
        blockers.append("heldout_cdf_calibration_failed")
    blockers.extend(f"league_holdout_not_supported:{league}" for league in unsupported_leagues)

    return {
        "protocol": {
            "kind": "chronological_series_train_calibration_test",
            "series_grouping": "same UTC date, league, and unordered team pair",
            "train_series_fraction": TRAIN_SERIES_FRACTION,
            "calibration_series_fraction": CALIBRATION_SERIES_FRACTION,
            "preprocessing": "training_only",
            "mean_model_hyperparameters_frozen_before_test": True,
            "test_labels_used_for_refit_only_after_evaluation": True,
        },
        "splits": {name: _split_manifest(part) for name, part in split.items()},
        "test": {
            "model": model_metrics,
            "league_mean_baseline": baseline_metrics,
            "cdf_calibration": calibration_report,
            "by_league": by_league,
        },
        "calibration": {
            "residual_count": len(calibration_residuals),
            "residual_mean": round(residual_mean, 4),
            "residual_sd": round(residual_sd, 4),
            "residuals": [round(value, 6) for value in sorted(calibration_residuals)],
            "probability_method": "add_half_smoothed_empirical_residual_cdf",
        },
        "authority": {
            "predictive_probability_supported": global_passed,
            "pooled_predictive_diagnostic_passed": pooled_passed,
            "betting_classification_requires_runtime_freshness_and_exact_patch": True,
            "content_addressing_confers_authority": False,
            "blockers": blockers,
        },
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
    predictive_sd = model.get("predictive_sd")
    uncertainty_status = (
        "chronological_calibration_residuals"
        if predictive_sd is not None
        else "unavailable_no_heldout_calibration"
    )
    return {
        "expected_total": round(mu, 2),
        "sd": predictive_sd,
        "uncertainty_status": uncertainty_status,
        "league": league,
        "n_champs_applied": len(used),
        "effects": sorted(used, key=lambda x: -abs(x["effect"])),
        "unknown_champions": missing,
        "baseline_mean": model["baseline_mean"],
        "delta_vs_baseline": round(mu - model["baseline_mean"], 2),
    }


def pricing_eligibility(
    payload: dict[str, Any],
    *,
    champions: list[str],
    league: str,
    patch: str | None,
    as_of: datetime,
) -> dict[str, Any]:
    """Return the exact runtime gates for a pregame total-kills probability."""
    blockers: list[str] = []
    evaluation = payload.get("evaluation") or {}
    authority = evaluation.get("authority") or {}
    if authority.get("predictive_probability_supported") is not True:
        blockers.append("heldout_predictive_probability_not_supported")

    league_report = (
        ((evaluation.get("test") or {}).get("by_league") or {}).get(league) or {}
    )
    if league_report.get("predictive_probability_supported") is not True:
        blockers.append(f"league_holdout_not_supported:{league}")

    meta = payload.get("meta") or {}
    data_cutoff = meta.get("data_cutoff")
    if not data_cutoff:
        blockers.append("data_cutoff_missing")
        age_days = None
    else:
        age_seconds = (as_of.astimezone(timezone.utc) - _parse_time(data_cutoff)).total_seconds()
        age_days = age_seconds / 86400.0
        if age_seconds < 0:
            blockers.append("as_of_precedes_data_cutoff")
        elif age_days > FRESHNESS_LIMIT_DAYS:
            blockers.append("data_stale")

    normalized_patch = str(patch or "").strip()
    if not normalized_patch:
        blockers.append("competition_patch_unverified")
    else:
        test_patches = ((evaluation.get("splits") or {}).get("test") or {}).get(
            "by_patch", {}
        )
        if int(test_patches.get(normalized_patch, 0)) < MIN_LEAGUE_TEST_GAMES:
            blockers.append(f"exact_patch_holdout_unavailable:{normalized_patch}")

    known = set((payload.get("model") or {}).get("champion_effects") or {})
    unknown = sorted(champion for champion in champions if champion not in known)
    if unknown:
        blockers.append("unknown_champions:" + ",".join(unknown))

    return {
        "status": "supported" if not blockers else "unavailable",
        "blockers": blockers,
        "data_age_days": round(age_days, 3) if age_days is not None else None,
        "freshness_limit_days": FRESHNESS_LIMIT_DAYS,
        "league": league,
        "patch": normalized_patch or None,
        "unknown_champions": unknown,
    }


def price_under(
    payload: dict[str, Any],
    *,
    champions: list[str],
    league: str,
    patch: str | None,
    as_of: datetime,
    line: float,
    odds: float,
) -> dict[str, Any]:
    """Price a half-kill under only when every evidence gate passes."""
    if abs(line * 2 - round(line * 2)) > 1e-9 or float(line).is_integer():
        raise ValueError("price_under requires a half-kill line")
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("decimal odds must be finite and greater than 1")
    eligibility = pricing_eligibility(
        payload,
        champions=champions,
        league=league,
        patch=patch,
        as_of=as_of,
    )
    prediction = predict_total(payload["model"], champions, league=league)
    result = {
        "line": line,
        "odds": odds,
        "expected_total": prediction["expected_total"],
        "eligibility": eligibility,
        "under_probability": None,
        "implied_probability": round(1.0 / odds, 6),
        "edge_pp": None,
        "expected_return_per_unit": None,
        "classification": "WITHHELD",
    }
    if eligibility["status"] != "supported":
        return result

    residuals = list((payload["evaluation"]["calibration"] or {}).get("residuals") or [])
    if len(residuals) < MIN_CALIBRATION_GAMES:
        result["eligibility"] = {
            **eligibility,
            "status": "unavailable",
            "blockers": [*eligibility["blockers"], "calibration_residuals_unavailable"],
        }
        return result
    cutoff = math.floor(line) - float(prediction["expected_total"])
    count = bisect.bisect_right(residuals, cutoff)
    probability = (count + 0.5) / (len(residuals) + 1.0)
    implied = 1.0 / odds
    edge = probability - implied
    expected_return = probability * odds - 1.0
    result.update(
        {
            "under_probability": round(probability, 6),
            "edge_pp": round(100.0 * edge, 4),
            "expected_return_per_unit": round(expected_return, 6),
            "classification": (
                "POSITIVE_MODEL_EV" if expected_return > 0 else "NEGATIVE_MODEL_EV"
            ),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-champ-games", type=int, default=20)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument(
        "--source",
        choices=("warehouse", "legacy"),
        default="warehouse",
        help="warehouse is the authoritative reconciled source; legacy is diagnostic only",
    )
    ap.add_argument(
        "--built-at",
        default=None,
        help="optional ISO timestamp for byte-reproducible builds",
    )
    args = ap.parse_args()

    if args.source == "warehouse":
        rows = build_warehouse_dataset()
        source_name = "reconciled Oracle's Elixir/GRID warehouse maps"
        source_paths = [WAREHOUSE_MAPS_IN]
    else:
        games = json.loads(GAMES_IN.read_text())["games"]
        players = json.loads(PLAYERS_IN.read_text())["players"]
        rows, _, _ = build_dataset(
            games, players, min_champ_games=args.min_champ_games
        )
        source_name = "legacy Leaguepedia ScoreboardGames+ScoreboardPlayers"
        source_paths = [GAMES_IN, PLAYERS_IN]

    champs, counts = champion_inventory(rows, args.min_champ_games)
    print(f"usable games={len(rows)} champions_kept={len(champs)} source={args.source}")
    by_lg: dict[str, int] = defaultdict(int)
    for r in rows:
        by_lg[r["league"]] += 1
    print("by league", dict(by_lg))

    evaluation = evaluate_chronological_holdout(
        rows,
        min_champ_games=args.min_champ_games,
        lam=args.lam,
    )
    model = fit_model(rows, champs, lam=args.lam)
    model["predictive_sd"] = evaluation["calibration"]["residual_sd"]
    model["predictive_sd_source"] = "chronological_calibration_residuals"
    # attach top/bottom effects
    ranked = sorted(model["champion_effects"].items(), key=lambda x: x[1])
    dates = [_parse_time(row["date"]) for row in rows]
    built_at = (
        _parse_time(args.built_at)
        if args.built_at
        else datetime.now(timezone.utc)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "source": source_name,
            "source_kind": args.source,
            "source_files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in source_paths
            ],
            "built_at": built_at.isoformat(),
            "data_start": min(dates).isoformat(),
            "data_cutoff": max(dates).isoformat(),
            "n_games": model["n_games"],
            "n_champions": len(champs),
            "by_league": dict(by_lg),
            "by_patch": dict(sorted(_counts(rows, "patch").items())),
            "min_champ_games": args.min_champ_games,
            "lam": args.lam,
            "training_fit_rmse": model["rmse"],
            "training_fit_r2": model["r2"],
            "training_fit_metrics_are_predictive_evidence": False,
        },
        "model": model,
        "evaluation": evaluation,
        "bloodiest": [{"champion": c, "effect": e} for c, e in ranked[::-1][:25]],
        "safest": [{"champion": c, "effect": e} for c, e in ranked[:25]],
        "champ_game_counts": {c: counts[c] for c in champs},
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {OUT} training_rmse={model['rmse']} "
        f"heldout_rmse={evaluation['test']['model']['rmse']} "
        f"authority={evaluation['authority']}"
    )
    print("bloodiest", payload["bloodiest"][:8])
    print("safest", payload["safest"][:8])


if __name__ == "__main__":
    main()
