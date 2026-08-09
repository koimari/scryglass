from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from lol_kills.v2.ratings.team import (
    CLAIM_CEILING,
    DISPLAY_SCALE,
    ArtifactIntegrityError,
    AuthorizationError,
    ExactRoster,
    LeagueRating,
    RosterValidationError,
    TeamRatingUnavailable,
    aggregate_team_rating,
    build_development_candidate,
    load_authorized_bundle,
    verify_development_candidate,
    verify_development_payload,
)
from lol_kills.v2.ratings.team.generate_artifacts import generate


DATA = (
    Path(__file__).parents[4]
    / "data"
    / "lol"
    / "v2"
    / "models"
    / "team"
)


@pytest.fixture
def fixtures():
    return json.loads((DATA / "team-rating-fixtures.json").read_text())


@pytest.fixture
def config():
    return json.loads((DATA / "team-rating-config.json").read_text())


def test_exact_five_accepts_ordered_active_official_roster(fixtures):
    roster = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    assert len(roster.players) == 5
    assert len(roster.roster_id) == 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["players"].pop(), "exactly five"),
        (
            lambda value: value["players"].__setitem__(
                1, deepcopy(value["players"][0])
            ),
            "duplicate",
        ),
        (
            lambda value: value["players"][0].__setitem__("role", "mid"),
            "role order",
        ),
        (
            lambda value: value["players"][0].__setitem__("active", False),
            "inactive",
        ),
        (lambda value: value.__setitem__("custom", True), "custom"),
        (lambda value: value.__setitem__("ambiguous", True), "ambiguous"),
        (lambda value: value.__setitem__("substitute", True), "substitute"),
    ],
)
def test_invalid_rosters_reject(fixtures, mutation, match):
    value = deepcopy(fixtures["regional"]["roster"])
    mutation(value)
    with pytest.raises(RosterValidationError, match=match):
        ExactRoster.from_mapping(value)


def test_roster_move_changes_identity_immediately_and_no_org_residual(fixtures):
    old_value = deepcopy(fixtures["regional"]["roster"])
    old = ExactRoster.from_mapping(old_value)
    old_rating = aggregate_team_rating(
        old, fixtures["regional"]["covariance"], scope="regional"
    )

    moved_value = deepcopy(old_value)
    moved_value["organization_id"] = fixtures["roster_change"]["new_organization_id"]
    moved_value["players"][2] = fixtures["roster_change"]["replacement"]
    moved = ExactRoster.from_mapping(moved_value)
    moved_rating = aggregate_team_rating(
        moved, fixtures["regional"]["covariance"], scope="regional"
    )

    assert moved.roster_id != old.roster_id
    expected_delta = DISPLAY_SCALE * (
        moved.players[2].posterior_mean - old.players[2].posterior_mean
    )
    assert moved_rating.posterior_mean - old_rating.posterior_mean == pytest.approx(
        expected_delta
    )
    assert "organization" not in moved_rating.to_dict()


def test_organization_change_alone_does_not_change_roster_identity_or_rating(fixtures):
    first_value = deepcopy(fixtures["regional"]["roster"])
    second_value = deepcopy(first_value)
    second_value["organization_id"] = "SYN-OTHER-ORG"
    first = ExactRoster.from_mapping(first_value)
    second = ExactRoster.from_mapping(second_value)
    assert first.roster_id == second.roster_id
    # Organization identity is part of the SOURCE receipt (it binds where the
    # roster came from) but never part of the strength identity.
    assert first.source_receipt_sha256 != second.source_receipt_sha256
    first_rating = aggregate_team_rating(
        first, fixtures["regional"]["covariance"], scope="regional"
    )
    second_rating = aggregate_team_rating(
        second, fixtures["regional"]["covariance"], scope="regional"
    )
    assert first_rating.posterior_mean == second_rating.posterior_mean
    assert first_rating.roster_id == second_rating.roster_id


def test_regional_has_zero_league_component_and_component_identity(fixtures):
    roster = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    rating = aggregate_team_rating(
        roster, fixtures["regional"]["covariance"], scope="regional"
    )
    assert rating.league_rating_component == 0.0
    assert rating.lineup_synergy_component is None
    assert rating.policy_component is None
    payload = rating.to_dict()
    assert rating.posterior_mean == pytest.approx(
        1500.0
        + rating.roster_strength_component
        + rating.league_rating_component
        + payload["reference_convention"]["computational_offset"]
    )


def test_global_adds_separate_league_rating(fixtures):
    roster = ExactRoster.from_mapping(fixtures["global"]["roster"])
    league = LeagueRating.from_mapping(fixtures["global"]["league_rating"])
    rating = aggregate_team_rating(
        roster,
        fixtures["global"]["covariance"],
        scope="global",
        league_rating=league,
    )
    assert rating.roster_latent_mean == pytest.approx(
        sum(player.posterior_mean for player in roster.players)
    )
    assert rating.league_rating_component == pytest.approx(
        DISPLAY_SCALE * league.posterior_mean
    )
    assert rating.posterior_mean == pytest.approx(
        1500.0 + rating.roster_strength_component + rating.league_rating_component
    )


def test_changing_league_rating_does_not_mutate_player_states(fixtures):
    roster = ExactRoster.from_mapping(fixtures["global"]["roster"])
    original_states = roster.players
    first = LeagueRating.from_mapping(fixtures["global"]["league_rating"])
    changed_value = deepcopy(fixtures["global"]["league_rating"])
    changed_value["posterior_mean"] = -0.12
    second = LeagueRating.from_mapping(changed_value)
    first_rating = aggregate_team_rating(
        roster, fixtures["global"]["covariance"], scope="global", league_rating=first
    )
    second_rating = aggregate_team_rating(
        roster, fixtures["global"]["covariance"], scope="global", league_rating=second
    )
    assert roster.players == original_states
    assert first_rating.player_posterior_means == second_rating.player_posterior_means
    assert second_rating.posterior_mean - first_rating.posterior_mean == pytest.approx(
        DISPLAY_SCALE * (second.posterior_mean - first.posterior_mean)
    )


@pytest.mark.parametrize("role_index", range(5))
def test_one_player_replacement_equality_every_role(fixtures, role_index):
    base_value = deepcopy(fixtures["regional"]["roster"])
    base = ExactRoster.from_mapping(base_value)
    base_rating = aggregate_team_rating(
        base, fixtures["regional"]["covariance"], scope="regional"
    )
    replacement_value = deepcopy(base_value)
    old_mean = replacement_value["players"][role_index]["posterior_mean"]
    replacement_value["players"][role_index]["player_id"] += "-replacement"
    replacement_value["players"][role_index]["posterior_mean"] = old_mean + 0.025
    replacement = ExactRoster.from_mapping(replacement_value)
    replacement_rating = aggregate_team_rating(
        replacement, fixtures["regional"]["covariance"], scope="regional"
    )
    assert replacement_rating.posterior_mean - base_rating.posterior_mean == pytest.approx(
        DISPLAY_SCALE * 0.025
    )


def test_joint_covariance_oracle_and_interval(fixtures):
    roster = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    covariance = fixtures["regional"]["covariance"]
    rating = aggregate_team_rating(roster, covariance, scope="regional")
    oracle_variance = sum(sum(row) for row in covariance)
    oracle_half_width = 1.96 * DISPLAY_SCALE * math.sqrt(oracle_variance)
    assert rating.roster_latent_variance == pytest.approx(oracle_variance)
    assert rating.posterior_interval_95 == pytest.approx(
        (
            rating.posterior_mean - oracle_half_width,
            rating.posterior_mean + oracle_half_width,
        )
    )


def test_global_and_regional_scope_separation_fail_closed(fixtures):
    regional = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    global_roster = ExactRoster.from_mapping(fixtures["global"]["roster"])
    league = LeagueRating.from_mapping(fixtures["global"]["league_rating"])
    with pytest.raises(TeamRatingUnavailable, match="global-scoped"):
        aggregate_team_rating(
            regional,
            fixtures["regional"]["covariance"],
            scope="global",
            league_rating=league,
        )
    with pytest.raises(TeamRatingUnavailable, match="regional-scoped"):
        aggregate_team_rating(
            global_roster, fixtures["global"]["covariance"], scope="regional"
        )
    deficient = deepcopy(fixtures["global"]["league_rating"])
    deficient["mobility_bridge_full_rank"] = False
    with pytest.raises(TeamRatingUnavailable, match="not separately identified"):
        aggregate_team_rating(
            global_roster,
            fixtures["global"]["covariance"],
            scope="global",
            league_rating=LeagueRating.from_mapping(deficient),
        )


def test_all_claim_ceilings_and_rank_eligibility_are_false(fixtures):
    roster = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    rating = aggregate_team_rating(
        roster, fixtures["regional"]["covariance"], scope="regional"
    )
    assert CLAIM_CEILING
    assert not any(CLAIM_CEILING.values())
    assert rating.rank_eligibility is False
    assert rating.missing_c2_dependency is True
    assert rating.development_only is True
    assert rating.to_dict()["schema_conformance"][
        "production_team_rating_schema"
    ] is False


@pytest.mark.parametrize("scope", ["regional", "global"])
def test_unidentified_policy_and_synergy_are_null_and_unavailable(fixtures, scope):
    roster = ExactRoster.from_mapping(fixtures[scope]["roster"])
    league = (
        LeagueRating.from_mapping(fixtures["global"]["league_rating"])
        if scope == "global"
        else None
    )
    payload = aggregate_team_rating(
        roster,
        fixtures[scope]["covariance"],
        scope=scope,
        league_rating=league,
    ).to_dict()
    assert payload["lineup_synergy_component"] is None
    assert payload["policy_component"] is None
    assert payload["estimand"]["gamma_q"] is None
    assert payload["estimand"]["policy_q"] is None
    for name in ("policy", "lineup_synergy"):
        component = payload["component_availability"][name]
        assert component["available"] is False
        assert component["status"] == "unavailable"
        assert component["reason"]
        assert component["blocker"]
        assert not any("interval" in key or "claim" in key for key in component)
    assert payload["reference_convention"] == {
        "status": "non_estimated",
        "computational_offset": 0.0,
        "contributes_exactly_zero": True,
        "covers": ["policy", "lineup_synergy"],
        "consumer_rule": "must_not_be_treated_as_an_estimate",
    }


def test_unavailable_components_survive_serialization_and_zero_mutation_rejects(
    fixtures,
):
    roster = ExactRoster.from_mapping(fixtures["regional"]["roster"])
    payload = aggregate_team_rating(
        roster, fixtures["regional"]["covariance"], scope="regional"
    ).to_dict()
    round_tripped = json.loads(json.dumps(payload, sort_keys=True))
    verify_development_payload(round_tripped)
    assert round_tripped["lineup_synergy_component"] is None
    forged = deepcopy(round_tripped)
    forged["lineup_synergy_component"] = 0.0
    with pytest.raises(ArtifactIntegrityError, match="synergy must be null"):
        verify_development_payload(forged)
    forged_estimand = deepcopy(round_tripped)
    forged_estimand["estimand"]["gamma_q"] = 0.0
    with pytest.raises(ArtifactIntegrityError, match="estimands must be null"):
        verify_development_payload(forged_estimand)


def test_candidate_hash_mutation_rejects_and_loader_fails_closed(config, fixtures):
    candidate = build_development_candidate(config, fixtures)
    verify_development_candidate(candidate, config, fixtures)
    forged = deepcopy(candidate)
    forged["candidate_id"] = "0" * 64
    with pytest.raises(ArtifactIntegrityError, match="identity mismatch"):
        verify_development_candidate(forged, config, fixtures)
    forged_claim = deepcopy(candidate)
    forged_claim["authorizing"] = True
    with pytest.raises(ArtifactIntegrityError, match="identity mismatch"):
        verify_development_candidate(forged_claim, config, fixtures)
    with pytest.raises(AuthorizationError, match="no independent"):
        load_authorized_bundle(candidate)


def test_deterministic_two_build_replay(config, fixtures, tmp_path):
    first = build_development_candidate(config, fixtures)
    second = build_development_candidate(deepcopy(config), deepcopy(fixtures))
    assert first == second
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    generate(
        DATA / "team-rating-config.json",
        DATA / "team-rating-fixtures.json",
        first_path,
    )
    generate(
        DATA / "team-rating-config.json",
        DATA / "team-rating-fixtures.json",
        second_path,
    )
    assert first_path.read_bytes() == second_path.read_bytes()

def test_display_scale_is_registered_400_point_logit_conversion():
    # Registered 1500/400 Elo display contract (mathematical-contract.md
    # sections 2-3): 400 display points map to ln(10) logits, so the
    # logit-to-point conversion is c_E = 400/log(10) -- the same scale as
    # Player Rating, never the legacy linear 400.0 per logit.
    assert DISPLAY_SCALE == pytest.approx(400.0 / math.log(10.0))
    assert DISPLAY_SCALE * math.log(10.0) == pytest.approx(400.0)
    assert not math.isclose(DISPLAY_SCALE, 400.0)


def test_one_logit_replacement_maps_through_registered_elo_scale(fixtures):
    # Replacing one player under the reference policy changes the roster
    # aggregation by exactly that player's displayed logit difference; on
    # the team display scale that one logit must be worth 400/log(10)
    # points, so the plug-in Elo odds ratio is e^1, not 10^1.
    base_value = deepcopy(fixtures["regional"]["roster"])
    base = ExactRoster.from_mapping(base_value)
    base_rating = aggregate_team_rating(
        base, fixtures["regional"]["covariance"], scope="regional"
    )
    one_logit_value = deepcopy(base_value)
    one_logit_value["players"][0]["player_id"] += "-one-logit"
    one_logit_value["players"][0]["posterior_mean"] += 1.0
    one_logit = ExactRoster.from_mapping(one_logit_value)
    one_logit_rating = aggregate_team_rating(
        one_logit, fixtures["regional"]["covariance"], scope="regional"
    )
    delta = one_logit_rating.posterior_mean - base_rating.posterior_mean
    assert delta == pytest.approx(DISPLAY_SCALE)
    assert delta == pytest.approx(400.0 / math.log(10.0))
    elo_odds = 10.0 ** (delta / 400.0)
    assert elo_odds == pytest.approx(math.e)


def test_400_point_team_difference_is_ln10_logits_not_one_logit(fixtures):
    # Through the registered Elo formula a 400-point difference must imply
    # an odds ratio of 10 (i.e. ln(10) logits).  If the legacy linear scale
    # had survived, 400 points would have meant only one logit.
    assert 400.0 / DISPLAY_SCALE == pytest.approx(math.log(10.0))
    for logit_delta in (0.25, 1.0, math.log(10.0)):
        points = DISPLAY_SCALE * logit_delta
        assert 10.0 ** (points / 400.0) == pytest.approx(math.exp(logit_delta))
