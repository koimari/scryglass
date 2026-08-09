"""Chronological identifiability evaluation fixtures (issue #45)."""

from __future__ import annotations

import json
import math

from lol_kills.v2.ratings.player.identifiability_evaluation import (
    REPORT_LOCATOR,
    build_evaluation,
    write_evaluation,
)
from lol_kills.v2.ratings.player.model import ROOT


def test_evaluation_is_deterministic_and_content_addressed():
    first = build_evaluation(ROOT)
    second = build_evaluation(ROOT)
    assert first == second
    assert first["schema_version"] == "scryglass:player-identifiability-evaluation:v1"
    # Pre-map log loss must beat the constant 0.5 baseline (log 2) and Brier
    # must beat the constant 0.25 baseline on the labeled forecasts.
    assert first["chronological"]["log_loss"] < math.log(2)
    assert first["chronological"]["brier"] < 0.25
    assert first["chronological"]["n_labeled"] == first["chronological"]["n_forecasts"]
    # Stratified evaluation exists for first-roster and transfer rows.
    assert first["first_roster"]["n"] >= 1
    assert first["transfer"]["n"] >= 1
    # Rank movement is reported from posterior displacement, not map count.
    assert first["rank_movement"]["n_players_with_movement"] >= 1
    assert first["rank_movement"]["mean_absolute_posterior_displacement"] is not None
    # Calibration is reported (intercept/slope on the available labels).
    assert first["calibration"]["n"] >= 1
    # The claim ceiling is explicit: development only.
    assert first["claim_ceiling"]["production_eligible"] is False
    assert first["claim_ceiling"]["promotion_authorized"] is False
    assert first["report_sha256"] == first["report_sha256"]


def test_written_report_replays_byte_identical():
    result = write_evaluation(ROOT)
    path = ROOT / REPORT_LOCATOR
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_evaluation(ROOT)
    assert stored["report_sha256"] == result["report_sha256"]
