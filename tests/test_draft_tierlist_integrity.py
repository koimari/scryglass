from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from lol_kills.draft_tierlist import (
    LEGACY_TIERLIST_PATHS,
    TierlistArtifactContractError,
    blend_win_with_tierlist,
    load_tierlist_artifact,
    score_draft_tierlist,
    validate_tierlist_artifact,
)


ROLES = ("top", "jng", "mid", "bot", "sup")
BLUE_CHAMPS = ("Aatrox", "Vi", "Ahri", "Jinx", "Nautilus")
RED_CHAMPS = ("Gnar", "Sejuani", "Azir", "KaiSa", "Rakan")


def valid_artifact() -> dict:
    def side_board(champions: tuple[str, ...], sign: float) -> dict:
        return {
            "board": {
                "rated": [
                    {
                        "champ": champion,
                        "role": role,
                        "delta_wr_pp": sign * (index + 1) / 10.0,
                        "n": 25 + index,
                    }
                    for index, (champion, role) in enumerate(
                        zip(champions, ROLES)
                    )
                ]
            }
        }

    return {
        "artifact_type": "calibrated_champion_delta_wr_tierlist",
        "schema_version": "2.0.0",
        "publication_status": "validated",
        "patch_contract": {
            "storage": "string",
            "source_dtype": "string",
            "numeric_coercion": "forbidden",
            "format": "major.minor",
            "patches": ["16.13"],
        },
        "controls": {
            "elo": True,
            "team": True,
            "full_composition": True,
        },
        "estimand": {
            "quantity": "conditional_delta_win_probability",
            "unit": "percentage_points",
            "calibrated": True,
        },
        "validation": {
            "chronological_holdout": {
                "train_end": "2026-05-31",
                "test_start": "2026-06-01",
                "test_end": "2026-06-30",
                "n_games": 100,
                "scored_out_of_sample": True,
            },
            "proper_score_comparison": {
                "passed": True,
                "evaluated_on": "chronological_holdout",
                "baseline": {
                    "name": "elo_team_full_composition",
                    "log_loss": 0.69,
                    "brier": 0.25,
                },
                "candidate": {
                    "log_loss": 0.68,
                    "brier": 0.24,
                },
            },
            "calibration": {
                "method": "isotonic_on_pre_holdout_calibration_split",
                "fit_end": "2026-05-31",
                "evaluated_on": "chronological_holdout",
                "ece": 0.03,
            },
            "leakage_checks": {
                "passed": True,
                "copied_map_outcomes_detected": False,
                "post_match_features_detected": False,
            },
        },
        "by_scope": {
            "lec": {
                "patch": "16.13",
                "blue": side_board(BLUE_CHAMPS, 1.0),
                "red": side_board(RED_CHAMPS, -1.0),
            }
        },
    }


class DraftTierlistIntegrityTests(unittest.TestCase):
    def test_checked_in_legacy_artifacts_are_rejected(self) -> None:
        existing = [path for path in LEGACY_TIERLIST_PATHS if path.is_file()]
        self.assertTrue(existing, "expected checked-in legacy tierlist artifacts")

        for path in existing:
            with self.subTest(path=path.name):
                with self.assertRaises(TierlistArtifactContractError):
                    load_tierlist_artifact(path)

    def test_backend_rejects_legacy_payload(self) -> None:
        path = next(path for path in LEGACY_TIERLIST_PATHS if path.is_file())
        legacy = json.loads(path.read_text(encoding="utf-8"))

        with self.assertRaises(TierlistArtifactContractError):
            score_draft_tierlist(
                list(BLUE_CHAMPS),
                list(RED_CHAMPS),
                league="LEC",
                patch="16.13",
                artifact=legacy,
            )

    def test_contract_rejects_each_known_integrity_failure(self) -> None:
        mutations = {
            "in_sample": lambda artifact: artifact["validation"][
                "chronological_holdout"
            ].__setitem__("scored_out_of_sample", False),
            "proper_score": lambda artifact: artifact["validation"][
                "proper_score_comparison"
            ]["candidate"].__setitem__("brier", 0.26),
            "patch_float": lambda artifact: artifact["patch_contract"].__setitem__(
                "patches", [16.13]
            ),
            "patch_numeric_coercion": lambda artifact: artifact[
                "patch_contract"
            ].__setitem__("numeric_coercion", "allowed"),
            "elo_control": lambda artifact: artifact["controls"].__setitem__(
                "elo", False
            ),
            "team_control": lambda artifact: artifact["controls"].__setitem__(
                "team", False
            ),
            "composition_control": lambda artifact: artifact["controls"].__setitem__(
                "full_composition", False
            ),
            "uncalibrated_units": lambda artifact: artifact["estimand"].__setitem__(
                "calibrated", False
            ),
            "calibration_fit_on_holdout": lambda artifact: artifact["validation"][
                "calibration"
            ].__setitem__("fit_end", "2026-06-15"),
            "copied_outcomes": lambda artifact: artifact["validation"][
                "leakage_checks"
            ].__setitem__("copied_map_outcomes_detected", True),
        }

        for label, mutate in mutations.items():
            artifact = copy.deepcopy(valid_artifact())
            mutate(artifact)
            with self.subTest(label=label):
                with self.assertRaises(TierlistArtifactContractError):
                    validate_tierlist_artifact(artifact)

    def test_valid_artifact_stays_descriptive_and_never_changes_probability(
        self,
    ) -> None:
        result = score_draft_tierlist(
            list(BLUE_CHAMPS),
            list(RED_CHAMPS),
            league="LEC",
            patch="16.13",
            blue_roles=list(ROLES),
            red_roles=list(ROLES),
            artifact=valid_artifact(),
        )

        self.assertEqual(result["status"], "validated_champion_estimates")
        self.assertEqual(result["unit"], "percentage_points")
        self.assertIsNone(result["match_win_probability"])
        self.assertFalse(any("edge" in key for key in result))

        probability, meta = blend_win_with_tierlist(0.63, result)
        self.assertEqual(probability, 0.63)
        self.assertFalse(meta["applied"])
        self.assertNotIn("edge_pp", meta)


if __name__ == "__main__":
    unittest.main()
