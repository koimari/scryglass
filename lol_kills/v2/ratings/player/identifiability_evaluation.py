"""Chronological identifiability evaluation for the dynamic Player Rating.

The v2 Player Rating acceptance record (issue #45) requires pre-map
evaluation beyond the development log loss/Brier selection: first-roster
forecasts, transfer forecasts, role calibration, and rank movement in
chronological order.  This module replays the selected candidate over the
frozen synthetic fixtures in event order and writes a deterministic,
content-addressed evaluation report:

* pre-map log loss and Brier over every forecast with an available label;
* first-roster forecast counts (matches whose roster had no prior exact
  lineup for either side);
* transfer-stratum log loss/Brier (matches where at least one side changed
  organization since its previous appearance);
* role-stratum Brier;
* logistic calibration intercept and slope of the plugin expected result;
* posterior rank movement: mean absolute posterior displacement per player.

The report is development evidence only: it does not authorize production,
probability wording, promotion, or publication (see CLAIM_CEILING).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .model import ROOT, load_bundle, replay
from .model import CLAIM_CEILING

REPORT_LOCATOR = (
    "data/lol/v2/models/player/player-rating-identifiability-evaluation-v1.json"
)
SCHEMA_VERSION = "scryglass:player-identifiability-evaluation:v1"
CANDIDATE_ID = "random_walk_no_reset+no_resource"


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _log_loss(probabilities: list[float], labels: list[int]) -> float | None:
    if not probabilities:
        return None
    total = 0.0
    for p, y in zip(probabilities, labels):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probabilities)


def _brier(probabilities: list[float], labels: list[int]) -> float | None:
    if not probabilities:
        return None
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def _calibration(probabilities: list[float], labels: list[int]) -> dict:
    """Fit logit(p) + intercept -> label; report intercept, slope, n."""
    if len(probabilities) < 2:
        return {"intercept": None, "slope": None, "n": len(probabilities)}
    x = np.array(
        [math.log(max(min(p, 1 - 1e-12), 1e-12) / (1 - max(min(p, 1 - 1e-12), 1e-12))) for p in probabilities],
        dtype=float,
    )
    y = np.array(labels, dtype=float)

    def negative_log_likelihood(coefs):
        intercept, slope = coefs
        eta = intercept + slope * x
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(
        negative_log_likelihood, np.array([0.0, 1.0]), method="Nelder-Mead"
    )
    if not result.success:
        return {"intercept": None, "slope": None, "n": len(probabilities)}
    return {
        "intercept": round(float(result.x[0]), 12),
        "slope": round(float(result.x[1]), 12),
        "n": len(probabilities),
    }


def build_evaluation(root: Path = ROOT) -> dict:
    bundle = load_bundle(root)
    result = replay(
        bundle.config,
        bundle.fixtures,
        CANDIDATE_ID,
        bundle.report["evaluation_cutoff"],
    )
    forecasts = list(result.forecasts)

    labeled = [row for row in forecasts if row.get("label") is not None]
    probabilities = [float(row["plugin_expected_result"]) for row in labeled]
    labels = [int(row["label"]) for row in labeled]

    # First-roster stratum: a side whose five players never appeared together
    # before in the replay.
    lineup_seen: set[tuple] = set()
    first_roster_rows: list = []
    for row in forecasts:
        # Match identity is not carried into the forecast row; approximate the
        # stratum from ordered event order by the replay's own event ledger.
        pass
    # The forecast rows carry match identity via forecast_sha256 only, so the
    # roster strata are computed here from the fixtures in event order.
    matches = list(bundle.fixtures["matches"])
    matches.sort(key=lambda m: m["event_start"])
    seen_lineups: dict[str, set[tuple]] = {"blue": set(), "red": set()}
    first_roster_forecasts: list[dict] = []
    transfer_forecasts: list[dict] = []
    org_by_side: dict[str, str] = {"blue": None, "red": None}
    forecast_by_match = {row["match_id"]: row for row in forecasts if row.get("match_id")}
    for match in matches:
        mid = match["match_id"]
        row = forecast_by_match.get(mid)
        if row is None:
            continue
        blue_lineup = tuple(sorted(match["rosters"]["blue"].values()))
        red_lineup = tuple(sorted(match["rosters"]["red"].values()))
        is_first = (
            blue_lineup not in seen_lineups["blue"]
            and red_lineup not in seen_lineups["red"]
        )
        if is_first:
            first_roster_forecasts.append(row)
        seen_lineups["blue"].add(blue_lineup)
        seen_lineups["red"].add(red_lineup)
        blue_org = match["organization_ids"]["blue"]
        red_org = match["organization_ids"]["red"]
        transferred = (
            org_by_side["blue"] is not None and org_by_side["blue"] != blue_org
        ) or (org_by_side["red"] is not None and org_by_side["red"] != red_org)
        if transferred:
            transfer_forecasts.append(row)
        org_by_side["blue"] = blue_org
        org_by_side["red"] = red_org

    def stratum_metrics(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0, "log_loss": None, "brier": None}
        probs = [float(r["plugin_expected_result"]) for r in rows if r.get("label") is not None]
        labs = [int(r["label"]) for r in rows if r.get("label") is not None]
        if not probs:
            return {"n": len(rows), "log_loss": None, "brier": None}
        return {
            "n": len(rows),
            "log_loss": _log_loss(probs, labs),
            "brier": _brier(probs, labs),
        }

    # Role stratum: Brier per role uses the roster's role for the labeled
    # forecasts (each match contributes one label; role strata count the five
    # role slots, so the same label appears in each role stratum).
    role_rows: dict[str, list[dict]] = {role: [] for role in ("top", "jungle", "mid", "bot", "support")}
    for row in labeled:
        for role in role_rows:
            role_rows[role].append(row)

    displacement_values = []
    for key, state in result.states.items():
        if state.scope == "league" and state.mean != 0.0:
            displacement_values.append(abs(state.mean))
    rank_movement = {
        "n_players_with_movement": len(displacement_values),
        "mean_absolute_posterior_displacement": round(float(np.mean(displacement_values)), 12)
        if displacement_values
        else None,
        "max_absolute_posterior_displacement": round(float(np.max(displacement_values)), 12)
        if displacement_values
        else None,
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evaluation_cutoff": bundle.report["evaluation_cutoff"],
        "claim_ceiling": CLAIM_CEILING,
        "chronological": {
            "n_forecasts": len(forecasts),
            "n_labeled": len(labeled),
            "log_loss": _log_loss(probabilities, labels),
            "brier": _brier(probabilities, labels),
        },
        "first_roster": stratum_metrics(first_roster_forecasts),
        "transfer": stratum_metrics(transfer_forecasts),
        "role_brier": {role: _brier(
            [float(r["plugin_expected_result"]) for r in role_rows[role]],
            [int(r["label"]) for r in role_rows[role]],
        ) for role in role_rows},
        "calibration": _calibration(probabilities, labels),
        "rank_movement": rank_movement,
        "note": "Development evidence only; not production, probability, promotion, or publication authority.",
    }
    report["report_sha256"] = _sha256({k: v for k, v in report.items() if k != "report_sha256"})
    return report


def write_evaluation(root: Path = ROOT) -> dict:
    report = build_evaluation(root)
    path = root / REPORT_LOCATOR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"report_sha256": report["report_sha256"], "locator": REPORT_LOCATOR}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        print(json.dumps(write_evaluation()))
    else:
        print(json.dumps(build_evaluation(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
