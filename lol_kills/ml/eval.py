"""Eval metrics, reliability, CRPS, and ship gates for the research pipeline."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error

from lol_kills.etl.paths import MODELS_DIR


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        conf = float(p[m].mean())
        acc = float(y_true[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
        reliability.append({"bin": i, "n": int(m.sum()), "conf": conf, "acc": acc})
    return float(ece), reliability


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    """Closed-form CRPS for Normal(mu, sd^2)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-3)
    z = (y - mu) / sd
    from scipy.stats import norm

    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def pinball_loss(y: np.ndarray, q_hat: np.ndarray, tau: float) -> float:
    y = np.asarray(y, dtype=float)
    q = np.asarray(q_hat, dtype=float)
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def classification_report(y: np.ndarray, p: np.ndarray, name: str) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ece, reliability = expected_calibration_error(y, p)
    return {
        "name": name,
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece,
        "reliability": reliability,
        "baseline_mean_brier": float(brier_score_loss(y, np.full_like(y, y.mean()))),
    }


def regression_report(y: np.ndarray, mu: np.ndarray, sd: np.ndarray, name: str) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    return {
        "name": name,
        "n": int(len(y)),
        "rmse": float(math.sqrt(mean_squared_error(y, mu))),
        "mae": float(np.mean(np.abs(y - mu))),
        "crps": crps_gaussian(y, mu, sd),
    }


def purged_time_splits(
    dates: pd.Series,
    n_folds: int = 5,
    embargo_frac: float = 0.02,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window splits with embargo gap between train end and test start.
    """
    order = np.argsort(pd.to_datetime(dates).values)
    n = len(order)
    if n < 200:
        return []
    fold = n // (n_folds + 1)
    embargo = max(int(n * embargo_frac), 5)
    splits = []
    for i in range(1, n_folds + 1):
        test_start = fold * (i + 1)
        test_end = min(test_start + fold, n)
        train_end = max(test_start - embargo, fold)
        if train_end < 100 or test_end - test_start < 30:
            continue
        tr = order[:train_end]
        te = order[test_start:test_end]
        splits.append((tr, te))
    return splits


def holdout_cut(dates: pd.Series, frac: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(pd.to_datetime(dates).values)
    cut = int(len(order) * frac)
    embargo = max(int(len(order) * 0.01), 3)
    train = order[: max(cut - embargo, 50)]
    test = order[cut:]
    return train, test


def evaluate_gates(report: dict) -> dict[str, Any]:
    """
    Ship gates from the roadmap.
    Returns {passed: bool, details: {...}}.
    """
    details = {}
    passed = True

    win = report.get("win") or {}
    if win.get("status") == "ok":
        m = win.get("holdout") or {}
        base = win.get("baselines") or {}
        ok = (
            m.get("brier", 1) < base.get("mean_brier", 0)
            and m.get("brier", 1) <= base.get("elo_brier", 1) + 1e-6
        )
        details["win"] = {
            "pass": ok,
            "brier": m.get("brier"),
            "mean_brier": base.get("mean_brier"),
            "elo_brier": base.get("elo_brier"),
            "ece": m.get("ece"),
        }
        passed = passed and ok
    else:
        details["win"] = {"pass": False, "reason": "missing"}
        passed = False

    kills = report.get("kills") or {}
    if kills.get("status") == "ok":
        m = kills.get("holdout") or {}
        ridge_crps = kills.get("baselines", {}).get("ridge_crps")
        model_crps = m.get("crps")
        # Gate: CRPS must beat ridge
        ok = ridge_crps is not None and model_crps is not None and model_crps <= ridge_crps + 1e-9
        details["kills"] = {
            "pass": ok,
            "crps": model_crps,
            "ridge_crps": ridge_crps,
            "rmse": m.get("rmse"),
            "ridge_rmse": kills.get("baselines", {}).get("ridge_rmse"),
            "selected": kills.get("selected_model"),
        }
        # Kills gate failure does not block whole ship if we fall back to ridge
        if not ok and kills.get("selected_model") == "ridge":
            details["kills"]["pass"] = True
            details["kills"]["note"] = "GBM lost; ridge selected"
        else:
            passed = passed and details["kills"]["pass"]
    else:
        details["kills"] = {"pass": False, "reason": "missing"}
        passed = False

    for aux in ("firstblood", "first_inhib"):
        block = report.get(aux) or {}
        if block.get("status") == "skipped":
            details[aux] = {"pass": True, "skipped": True}
            continue
        if block.get("status") == "ok":
            m = block.get("holdout") or {}
            base_b = block.get("baselines", {}).get("mean_brier", 1)
            ok = m.get("brier", 1) < base_b
            details[aux] = {"pass": ok, "brier": m.get("brier"), "mean_brier": base_b}
            # aux failure soft — don't block ship
        else:
            details[aux] = {"pass": True, "skipped": True}

    return {
        "passed": passed,
        "details": details,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_eval_report(report: dict, gates: dict, path: Path | None = None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (MODELS_DIR / "eval_report.json")
    payload = {"report": report, "gates": gates}
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[eval] wrote {path} passed={gates.get('passed')}")
    return path


def archive_models(tag: str | None = None) -> Path:
    """Copy current models dir to archive/YYYYMMDD[/_tag]."""
    import shutil

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest = MODELS_DIR / "archive" / (f"{day}_{tag}" if tag else day)
    dest.mkdir(parents=True, exist_ok=True)
    for p in MODELS_DIR.iterdir():
        if p.name == "archive" or p.is_dir():
            continue
        shutil.copy2(p, dest / p.name)
    print(f"[eval] archived models → {dest}")
    return dest
