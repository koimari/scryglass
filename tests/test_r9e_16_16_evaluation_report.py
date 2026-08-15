from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/lol/v2/evaluation/r9e-16.16-evaluation-report.json"


def test_r9e_16_16_report_records_auc_floor_failure_and_closes_authority() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()

    assert claimed == actual
    assert payload["metrics"]["auc"] == 0.68939
    assert payload["gates"]["auc_floor"] == 0.705
    assert payload["gates"]["auc_floor_passes"] is False
    assert payload["gates"]["promotion_status"] == "blocked_auc_floor"
    assert payload["authority"] == {
        "private_r9e_promotion": False,
        "public_draft": False,
        "public_probability": False,
    }
