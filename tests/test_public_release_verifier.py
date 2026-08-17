from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / ".codex/skills/run-scryglass-public-release/scripts/verify_live_release.py"
)
SPEC = importlib.util.spec_from_file_location("verify_live_release", SCRIPT)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _census():
    identity = "a" * 64
    manifest = {
        "release_id": "v2026.08.17.120000",
        "source_as_of": "2026-08-17T12:00:00Z",
        "source_game_count": 2,
        "source_identity_sha256": identity,
    }
    draft = {
        "release_id": manifest["release_id"],
        "source_as_of": manifest["source_as_of"],
        "source_identity_sha256": identity,
        "sample_window": {"target_games": 2},
        "games": {"game-1": {}, "game-2": {}},
    }
    matches = {
        "games": [
            {
                "game_id": game_id,
                "date": "2026-08-17T12:00:00Z",
                "competition_tier": "tier1",
                "champions": [f"Champion {offset + index * 10}" for offset in range(10)],
            }
            for index, game_id in enumerate(("game-1", "game-2"))
        ]
    }
    tiers = {"as_of": manifest["source_as_of"]}
    return manifest, draft, matches, tiers


def test_release_verifier_requires_draft_score_for_every_complete_composition() -> None:
    manifest, draft, matches, tiers = _census()
    assert VERIFIER._verify_census(manifest, draft, matches, tiers)["draft_scores"] == 2

    draft["games"].pop("game-2")
    with pytest.raises(RuntimeError, match="missing=1"):
        VERIFIER._verify_census(manifest, draft, matches, tiers)


def test_release_verifier_requires_one_source_binding() -> None:
    manifest, draft, matches, tiers = _census()
    draft["source_identity_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="source identity"):
        VERIFIER._verify_census(manifest, draft, matches, tiers)


def test_empty_default_results_are_allowed_only_without_current_tier_one_games() -> None:
    _, _, matches, _ = _census()
    now = datetime(2026, 8, 17, tzinfo=timezone.utc)
    assert VERIFIER._current_results_expected(matches, now)
    matches["games"][0]["competition_tier"] = "tier2"
    matches["games"][1]["competition_tier"] = "tier2"
    assert not VERIFIER._current_results_expected(matches, now)
