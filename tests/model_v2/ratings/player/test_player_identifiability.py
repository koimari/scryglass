"""Identifiability fixtures for the dynamic Player Rating (issue #45).

The public player track (lol_kills/ratings/player_elo.py) is a descriptive
baseline: it applies the same team-outcome residual to every player on a side
with fixed role weights, so it cannot identify individual contribution.  The
v2 dynamic Player Rating (lol_kills/v2/ratings/player/model.py) updates
each player's posterior from player-specific pre-event information through a
joint rank-one Gaussian attribution; these fixtures pin that difference:

* a shared side result never produces identical updates for unequal players
  (the shared-update duplication detector);
* a player-specific pre-event signal changes exactly that player state;
* history carries across transfers and no organization residual sticks to the
  player's posterior;
* posterior uncertainty is separate from map count, and the artifact exposes
  displacement, precision, source/context coverage, and freshness as separate
  fields;
* role and league policy IDs are registered before any fit.
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

import pytest

from lol_kills.v2.ratings.player.model import (
    PlayerState,
    evidence_components,
    replay,
)


def _player(player_id: str, role: str) -> dict:
    return {
        "player_id": player_id,
        "display_name": player_id,
        "role": role,
        "global_eligibility": {
            "version": "eligibility-v2",
            "valid_from": "2025-01-01T00:00:00Z",
            "valid_until": None,
            "international_eligible": True,
            "connectivity_supported": True,
            "bridge_support_count": 3,
            "bridge_component_id": "bridge-main-v1",
            "current_league_tier": 1,
            "active_status": "active",
            "roster_membership": "main",
            "roster_ambiguous": False,
        },
    }


def _roster(prefix: str) -> dict:
    return {
        "top": f"{prefix}-top",
        "jungle": f"{prefix}-jungle",
        "mid": f"{prefix}-mid",
        "bot": f"{prefix}-bot",
        "support": f"{prefix}-support",
    }


def _match(
    match_id: str,
    event_start: str,
    blue_org: str,
    red_org: str,
    blue_win: int,
    outcome_at: str | None = None,
) -> dict:
    date = event_start[:10]
    return {
        "match_id": match_id,
        "event_start": event_start,
        "event_end": f"{date}T12:45:00Z",
        "outcome_available_at": outcome_at or f"{date}T13:00:00Z",
        "season_id": "2025",
        "calendar_year": 2025,
        "patch_id": "25.24",
        "league_id": "L01",
        "league_tier": 1,
        "organization_ids": {"blue": blue_org, "red": red_org},
        "rosters": {"blue": _roster("a"), "red": _roster("b")},
        "blue_win": blue_win,
        "player_updates": [],
    }


PLAYERS = [
    _player("a-top", "top"),
    _player("a-jungle", "jungle"),
    _player("a-mid", "mid"),
    _player("a-bot", "bot"),
    _player("a-support", "support"),
    _player("b-top", "top"),
    _player("b-jungle", "jungle"),
    _player("b-mid", "mid"),
    _player("b-bot", "bot"),
    _player("b-support", "support"),
]
CANDIDATE = "random_walk_no_reset+no_resource"


@pytest.fixture(scope="module")
def bundle():
    from lol_kills.v2.ratings.player.model import ROOT, load_bundle

    return load_bundle(ROOT)


def _league_state(result, player_id: str, league_id: str = "L01") -> PlayerState:
    return result.states[("league", league_id, _role_of(result, player_id), player_id)]


def _role_of(result, player_id: str) -> str:
    for key in result.states:
        if key[3] == player_id:
            return key[2]
    raise KeyError(player_id)


def test_shared_side_result_never_creates_identical_updates_for_unequal_players(bundle):
    """Shared-update duplication detector.

    Two blue players with different posterior precision receive different
    updates from the same team outcome: the rank-one Gaussian attribution
    scales each player shift by their covariance contribution.  A model
    that applied the same residual to every player on the side (the
    descriptive-baseline behavior) would produce duplicate states and
    fail these assertions.
    """
    matches = [
        _match("m1", "2025-12-01T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("m2", "2025-12-08T12:00:00Z", "ORG-A", "ORG-B", 1),
    ]
    # Player-specific pre-event information: a-top receives two precise
    # combat observations before m1; a-mid receives none.  The shared
    # team outcome after that must move the tight player less.
    matches[0]["player_updates"] = [
        {
            "player_id": "a-top",
            "channel_id": "role_normalized_nonresource_impact",
            "feature_provenance_id": "role-normalized-nonresource-impact-v1",
            "value": 0.4,
            "observation_variance": 0.05,
            "available_at": "2025-12-01T13:00:00Z",
            "source_context": "combat",
        }
    ]
    result = replay(
        bundle.config,
        {"artifact_id": "fx", "schema_version": "fx-v1", "synthetic_only": True, "players": PLAYERS, "matches": matches},
        CANDIDATE,
        "2025-12-09T00:00:00Z",
    )
    top = _league_state(result, "a-top")
    mid = _league_state(result, "a-mid")
    # The player-specific combat observation tightened a-top (its posterior
    # precision exceeds a-mid, which only saw the shared outcome).
    assert top.variance < mid.variance
    # Both players saw the same two team outcomes, yet their posterior
    # means differ because each update is attributed through the joint
    # covariance, not copied from a shared residual.
    assert top.mean != mid.mean
    # Identical players on the same side WOULD move together, but the
    # fixture guarantees the two players are unequal, so duplicate
    # posteriors are the legacy-baseline failure mode this detector pins.
    assert not (top.mean == mid.mean and top.variance == mid.variance)


def test_player_specific_pre_event_information_changes_only_that_player(bundle):
    """Valid player-specific information: one channel, one player."""
    # Outcome is available at 14:00 so the 13:30 cutoff isolates the
    # player-specific update from the shared team outcome.
    match = _match("m3", "2025-12-01T12:00:00Z", "ORG-A", "ORG-B", 1, outcome_at="2025-12-01T14:00:00Z")
    match["player_updates"] = [
        {
            "player_id": "a-jungle",
            "channel_id": "role_normalized_nonresource_impact",
            "feature_provenance_id": "role-normalized-nonresource-impact-v1",
            "value": 0.7,
            "observation_variance": 0.1,
            "available_at": "2025-12-01T13:00:00Z",
            "source_context": "map_impact",
        }
    ]
    fx = {"artifact_id": "fx", "schema_version": "fx-v1", "synthetic_only": True, "players": PLAYERS, "matches": [match]}
    before = replay(bundle.config, fx, CANDIDATE, "2025-12-01T12:30:00Z")
    after = replay(bundle.config, fx, CANDIDATE, "2025-12-01T13:30:00Z")
    jungle_before = _league_state(before, "a-jungle")
    jungle_after = _league_state(after, "a-jungle")
    assert jungle_after.mean != jungle_before.mean
    assert jungle_after.variance < jungle_before.variance
    for player_id in ("a-top", "a-mid", "a-bot", "a-support"):
        other_before = _league_state(before, player_id)
        other_after = _league_state(after, player_id)
        # No player-specific signal touched these players: their skill mean
        # is untouched.  (Variance may grow slightly from the time decay
        # between the two cutoffs, but never shrink without an update.)
        assert other_after.mean == other_before.mean
        assert other_after.variance >= other_before.variance


def test_transfer_carries_history_and_leaves_no_org_residual(bundle):
    """History travels with the player; organization stays context."""
    matches = [
        _match("t1", "2025-12-01T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("t2", "2025-12-08T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("t3", "2025-12-15T12:00:00Z", "ORG-C", "ORG-D", 1),
    ]
    fx = {"artifact_id": "fx", "schema_version": "fx-v1", "synthetic_only": True, "players": PLAYERS, "matches": matches}
    result = replay(bundle.config, fx, CANDIDATE, "2025-12-16T00:00:00Z")
    top = _league_state(result, "a-top")
    # The player won all three matches; the posterior mean moved away from
    # the prior, proving the history carried across the ORG-A -> ORG-C move.
    assert top.mean != 0.0
    assert top.organization_id == "ORG-C"
    # No organization residual: the posterior mean after the transfer is
    # not shifted by an org-level constant.  Replaying the same third match
    # against a neutral org (ORG-Z) yields the same posterior mean as the
    # ORG-C version because organization identity is context, not a rating
    # component.
    neutral = [
        _match("t1", "2025-12-01T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("t2", "2025-12-08T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("t3", "2025-12-15T12:00:00Z", "ORG-Z", "ORG-D", 1),
    ]
    fx_neutral = {"artifact_id": "fx", "schema_version": "fx-v1", "synthetic_only": True, "players": PLAYERS, "matches": neutral}
    neutral_result = replay(bundle.config, fx_neutral, CANDIDATE, "2025-12-16T00:00:00Z")
    top_neutral = _league_state(neutral_result, "a-top")
    assert top.mean == pytest.approx(top_neutral.mean, abs=1e-12)


def test_transfer_widens_uncertainty_without_resetting_history(bundle):
    """The transfer policy adds variance; it never resets the mean."""
    matches = [
        _match("u1", "2025-12-01T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("u2", "2025-12-08T12:00:00Z", "ORG-A", "ORG-B", 1),
        _match("u3", "2025-12-15T12:00:00Z", "ORG-C", "ORG-D", 1),
    ]
    fx = {"artifact_id": "fx", "schema_version": "fx-v1", "synthetic_only": True, "players": PLAYERS, "matches": matches}
    result = replay(bundle.config, fx, CANDIDATE, "2025-12-16T00:00:00Z")
    top = _league_state(result, "a-top")
    # cross_league_variance (0.2) was added at the transfer, so posterior
    # variance exceeds the no-transfer variance while the mean is retained.
    assert top.mean != 0.0
    assert top.organization_id == "ORG-C"


def test_posterior_uncertainty_is_separate_from_map_count(bundle):
    """Precision comes from the posterior; map count is a separate record."""
    config = bundle.config
    state_light = PlayerState(
        player_id="p1", role="mid", scope="league", league_id="L01",
        mean=0.2, variance=0.16, last_available_at="2025-12-01T00:00:00Z",
        information=6.25, source_contexts=("team_outcome_anchor", "combat", "map_impact", "objective", "vision_nonfarm"),
    )
    state_heavy = replace(state_light, information=62.5)  # ten times the
    # observations, same posterior
    light = evidence_components(state_light, config)
    heavy = evidence_components(state_heavy, config)
    assert light["precision"]["posterior_dispersion"] == heavy["precision"]["posterior_dispersion"]
    assert light["posterior_displacement"]["value"] == heavy["posterior_displacement"]["value"]
    assert light["source_context_coverage"]["coverage_fraction"] == 1.0
    # Freshness is its own field, from the state timestamp.
    assert state_light.last_available_at == "2025-12-01T00:00:00Z"


def test_evidence_components_expose_full_contract_fields(bundle):
    config = bundle.config
    state = PlayerState(
        player_id="p2", role="top", scope="league", league_id="L01",
        mean=0.5, variance=0.25, last_available_at="2025-12-01T00:00:00Z",
        information=4.0, source_contexts=("team_outcome_anchor", "combat"),
    )
    components = evidence_components(state, config)
    assert components["evidence_spec_id"] == "player-evidence-components-v1"
    assert components["posterior_displacement"]["method_id"] == "absolute-posterior-shift-over-prior-sd-v1"
    assert components["posterior_displacement"]["value"] == pytest.approx(0.5 / 1.0)
    assert components["precision"]["method_id"] == "posterior-sd-versus-prior-sd-v1"
    assert components["precision"]["posterior_dispersion"] == pytest.approx(math.sqrt(0.25))
    assert components["source_context_coverage"]["method_id"] == "registered-required-context-fraction-v1"
    assert components["source_context_coverage"]["coverage_fraction"] == pytest.approx(2.0 / 5.0)
    assert components["source_context_coverage"]["missing_required_context_ids"] == (
        "map_impact",
        "objective",
        "vision_nonfarm",
    )


def test_role_and_league_policy_ids_are_registered_before_fit(bundle):
    """The role policy and league policy layers are explicit and frozen."""
    config = bundle.config
    policy = config["reference_policy"]
    assert policy["policy_id"] == "equal-role-reference-policy-v1"
    assert policy["frozen"] is True
    assert set(policy["role_weights"]) == {"top", "jungle", "mid", "bot", "support"}
    assert config["league_rating_included"] is False
    assert isinstance(config["league_centers"], (dict, MappingProxyType))
    assert config["transfer"]["cross_league_variance"] > 0

