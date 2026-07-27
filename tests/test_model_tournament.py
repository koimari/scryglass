from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lol_kills.ml.eval import evaluate_gates
from lol_kills.model_tournament import (
    PredictionLedgerError,
    TournamentSpec,
    corp_calibration_diagnostics,
    paired_moving_block_comparison,
    pav_calibrated_probabilities,
    validate_prediction_ledger,
)


def _ledger(n_events: int = 240) -> pd.DataFrame:
    rows = []
    base_time = pd.Timestamp("2026-01-01T12:00:00Z")
    for event in range(n_events):
        outcome = event % 2
        event_time = base_time + pd.Timedelta(days=event)
        for model_id, probability in (
            ("candidate", 0.8 if outcome else 0.2),
            ("baseline", 0.55 if outcome else 0.45),
        ):
            rows.append(
                {
                    "prediction_id": f"{model_id}:{event}",
                    "model_id": model_id,
                    "model_version": "v1",
                    "event_id": f"event:{event}",
                    "series_id": f"series:{event // 3}",
                    "prediction_time": event_time - pd.Timedelta(hours=1),
                    "event_time": event_time,
                    "data_as_of": event_time - pd.Timedelta(hours=2),
                    "outcome": outcome,
                    "probability": probability,
                    "split": "test",
                    "league": "LCK",
                    "patch": "26.1",
                    "roster_state_id": f"roster:{event // 30}",
                }
            )
    return pd.DataFrame(rows)


class PredictionLedgerTest(unittest.TestCase):
    def test_future_data_is_rejected(self) -> None:
        frame = _ledger()
        frame.loc[0, "data_as_of"] = frame.loc[0, "event_time"] + pd.Timedelta(
            minutes=1
        )
        with self.assertRaisesRegex(PredictionLedgerError, "temporal leakage"):
            validate_prediction_ledger(frame)

    def test_event_outcome_must_agree_across_models(self) -> None:
        frame = _ledger()
        frame.loc[1, "outcome"] = 1 - frame.loc[1, "outcome"]
        with self.assertRaisesRegex(PredictionLedgerError, "event outcome"):
            validate_prediction_ledger(frame)


class CalibrationTest(unittest.TestCase):
    def test_pav_is_monotone_and_decomposition_reconciles(self) -> None:
        probability = np.array([0.1, 0.2, 0.2, 0.5, 0.8, 0.9])
        outcome = np.array([0, 1, 0, 0, 1, 1])
        fitted = pav_calibrated_probabilities(outcome, probability)
        order = np.argsort(probability, kind="mergesort")
        self.assertTrue(np.all(np.diff(fitted[order]) >= -1e-15))
        self.assertEqual(fitted[1], fitted[2])

        report = corp_calibration_diagnostics(outcome, probability)
        self.assertGreaterEqual(report["miscalibration"], 0.0)
        self.assertAlmostEqual(report["decomposition_residual"], 0.0, places=12)


class ModelComparisonTest(unittest.TestCase):
    def test_clear_candidate_superiority_is_detected(self) -> None:
        spec = TournamentSpec(
            estimand_id="map_win_pre_draft",
            primary_score="brier",
            minimum_test_events=200,
            bootstrap_replicates=500,
            moving_block_size=8,
            random_seed=17,
        )
        result = paired_moving_block_comparison(
            _ledger(),
            candidate_model_id="candidate",
            baseline_model_id="baseline",
            spec=spec,
        )
        self.assertEqual(result["decision"], "superior")
        self.assertLess(result["candidate_score"], result["baseline_score"])
        self.assertAlmostEqual(
            result["candidate_minus_baseline"],
            result["candidate_score"] - result["baseline_score"],
            places=12,
        )
        self.assertLess(result["confidence_interval"][1], 0.0)

    def test_ship_gate_cannot_ignore_a_better_elo_baseline(self) -> None:
        report = {
            "win": {
                "status": "ok",
                "holdout": {"brier": 0.22, "ece": 0.01},
                "baselines": {"mean_brier": 0.25, "elo_brier": 0.20},
            },
            "kills": {"status": "skipped"},
            "firstblood": {"status": "skipped"},
            "first_inhib": {"status": "skipped"},
        }
        gates = evaluate_gates(report)
        self.assertFalse(gates["details"]["win"]["pass"])


if __name__ == "__main__":
    unittest.main()
