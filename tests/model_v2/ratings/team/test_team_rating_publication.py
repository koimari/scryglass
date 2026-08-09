"""Exact-roster publication contract fixtures (issue #47).

The public Team Rating must publish the exact ordered five with names,
roles, source time, receipt hash, model scope, and evidence state, and
must fail closed for duplicate players, missing roles, substitutes,
hypothetical rosters, and stale/inactive rosters.  A new roster gets a
wide policy/synergy prior: when separation is not identified, only the
identified total roster strength is published.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.ratings.team import (
    CLAIM_CEILING,
    ExactRoster,
    LeagueRating,
    RosterValidationError,
    TeamRatingUnavailable,
    aggregate_team_rating,
)


@pytest.fixture(scope="module")
def fixtures():
    import json

    
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "data/lol/v2/models/team/team-rating-fixtures.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _regional_roster(fixtures):
    return deepcopy(fixtures["regional"]["roster"])


def test_published_payload_carries_the_exact_publication_contract(fixtures):
    roster = ExactRoster.from_mapping(_regional_roster(fixtures))
    rating = aggregate_team_rating(
        roster, fixtures["regional"]["covariance"], scope="regional"
    )
    payload = rating.to_dict()
    assert payload["status"] == "development_only"
    # Five names, roles, and ids in exact role order.
    assert payload["players"] == [
        "regional-top", "regional-jungle", "regional-mid", "regional-bot", "regional-support"
    ]
    assert payload["player_roles"] == ["top", "jungle", "mid", "bot", "support"]
    assert len(payload["player_names"]) == 5
    assert all(name for name in payload["player_names"])
    # Source time, receipt, scope, and evidence state.
    assert payload["roster_effective_at"] == "2026-01-01T00:00:00Z"
    assert payload["roster_as_of"] == "2026-01-15T00:00:00Z"
    assert payload["model_scope"] == "regional"
    assert len(payload["roster_receipt_sha256"]) == 64
    assert payload["evidence_state"] == "development_only"
    # League Rating is its own field for the global scope.
    global_rating = aggregate_team_rating(
        ExactRoster.from_mapping(fixtures["global"]["roster"]),
        fixtures["global"]["covariance"],
        scope="global",
        league_rating=LeagueRating.from_mapping(
            fixtures["global"]["league_rating"]
        ),
    )
    assert global_rating.to_dict()["league_rating_component"] != 0.0


def test_receipt_hash_binds_every_player_and_role_change(fixtures):
    base = _regional_roster(fixtures)
    first = ExactRoster.from_mapping(base)
    changed = deepcopy(base)
    changed["players"][2]["posterior_mean"] += 0.05
    second = ExactRoster.from_mapping(changed)
    assert second.source_receipt_sha256 != first.source_receipt_sha256
    # A role swap also rebinds the receipt (exact role order is part of the
    # roster identity).
    swapped = deepcopy(base)
    swapped["players"][0], swapped["players"][1] = swapped["players"][1], swapped["players"][0]
    with pytest.raises(RosterValidationError):
        ExactRoster.from_mapping(swapped)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("hypothetical", True), "hypothetical"),
        (lambda value: value["players"].pop(), "exactly five"),
        (
            lambda value: value["players"].__setitem__(
                1, deepcopy(value["players"][0])
            ),
            "duplicate",
        ),
        (lambda value: value["players"][0].__setitem__("role", "mid"), "role order"),
        (lambda value: value.__setitem__("substitute", True), "substitute"),
    ],
)
def test_hypothetical_duplicate_substitute_and_missing_role_rosters_fail_closed(
    fixtures, mutation, match
):
    value = _regional_roster(fixtures)
    mutation(value)
    with pytest.raises(RosterValidationError, match=match):
        ExactRoster.from_mapping(value)


def test_new_roster_gets_wide_policy_synergy_prior_only_identified_total_is_published(fixtures):
    """A replacement five without identified policy/synergy publishes only
    the identified total roster strength; the components stay unavailable."""
    roster = ExactRoster.from_mapping(_regional_roster(fixtures))
    rating = aggregate_team_rating(
        roster, fixtures["regional"]["covariance"], scope="regional"
    )
    payload = rating.to_dict()
    # The identified total is the anchor plus the roster latent contribution;
    # the unidentified policy/synergy components never inflate it.
    assert payload["posterior_mean"] == pytest.approx(
        1500.0 + payload["roster_strength_component"]
    )
    assert payload["lineup_synergy_component"] is None
    assert payload["policy_component"] is None
    assert payload["component_availability"]["lineup_synergy"]["available"] is False
    assert payload["component_availability"]["policy"]["available"] is False


def test_global_scope_requires_eligible_league_rating_separation(fixtures):
    roster = ExactRoster.from_mapping(fixtures["global"]["roster"])
    league = LeagueRating.from_mapping(
        fixtures["global"]["league_rating"]
    )
    # A not-separately-identified League Rating must fail closed.


    broken_value = {
        "league_id": "SYN-LCK",
        "posterior_mean": 0.08,
        "posterior_variance": 0.004,
        "centered": True,
        "reference_constrained": True,
        "mobility_bridge_full_rank": True,
        "separately_identified": False,
        "transfer_continuity": True,
    }
    with pytest.raises(TeamRatingUnavailable):
        aggregate_team_rating(
            roster, fixtures["global"]["covariance"], scope="global",
            league_rating=LeagueRating.from_mapping(broken_value),
        )