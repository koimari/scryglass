import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lol_kills import update_public_pack


class UpdatePublicPackTests(unittest.TestCase):
    def test_data_refresh_cannot_refit_or_promote_draft_models(self) -> None:
        source = inspect.getsource(update_public_pack.main)
        self.assertNotIn("fit_draft_score_scaler", source)
        self.assertNotIn("fit_from_paths", source)
        self.assertNotIn("fit_elo_wr_calibration", source)

    def test_publish_audit_failure_prevents_uploader_invocation(self) -> None:
        failed_report = {
            "counts": {
                "launch blocker": 0,
                "major": 1,
                "minor": 0,
                "informational": 0,
            },
            "release_gate": {
                "ready": False,
                "blocking_severities": ["launch blocker", "major"],
                "blocking_findings": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temp, patch.object(
            update_public_pack,
            "_run_module",
        ) as run_module, patch(
            "pandas.read_parquet",
        ), patch(
            "lol_kills.ratings.player_elo.build_maps_frame_from_players",
        ), patch(
            "lol_kills.ratings.player_elo.build_player_ratings",
        ), patch(
            "lol_kills.export.public_pack.export_public_pack",
            return_value={"pack_id": "vsafe"},
        ), patch(
            "lol_kills.audit_public_pack.audit_pack",
            return_value=failed_report,
        ) as audit:
            with self.assertRaisesRegex(RuntimeError, "failed release gate"):
                update_public_pack.main(
                    [
                        "--skip-oe",
                        "--skip-grid",
                        "--pack-id",
                        "vsafe",
                        "--out",
                        str(Path(temp)),
                        "--publish",
                    ]
                )

        audit.assert_called_once_with(Path(temp) / "vsafe")
        self.assertEqual(run_module.call_count, 1)
        self.assertEqual(
            run_module.call_args.args[0],
            "lol_kills.refresh_warehouse",
        )

    def test_unsafe_pack_id_fails_before_refresh(self) -> None:
        with patch.object(update_public_pack, "_run_module") as run_module:
            with self.assertRaisesRegex(ValueError, "Unsafe pack_id"):
                update_public_pack.main(["--pack-id", "../outside"])
        run_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
