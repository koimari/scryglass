from __future__ import annotations

import hashlib
from pathlib import Path

from lol_kills.live_totals_candidate import (
    DEVELOPMENT_CANDIDATE_RAW_SHA256,
    development_candidate_path,
    validate_development_candidate,
)


def test_registered_live_totals_candidate_and_source_replay_exactly() -> None:
    root = Path(".").resolve()
    path = development_candidate_path(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        DEVELOPMENT_CANDIDATE_RAW_SHA256
    )
    artifact = validate_development_candidate(root)
    assert artifact["schema_version"] == "scryglass.live-total-kills.v2"
    assert artifact["authority"]["betting_decision_authorized"] is False
    assert artifact["meta"]["data_cutoff_by_league"]["LCS"] == (
        "2026-07-26T23:12:26+00:00"
    )
