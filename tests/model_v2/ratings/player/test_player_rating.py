from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from lol_kills.v2.evaluation.contract_validation import validate_output_payload
from lol_kills.v2.ratings.player import model
from lol_kills.v2.ratings.player.model import (
    CLAIM_CEILING,
    DECAY_CANDIDATES,
    RESOURCE_CANDIDATES,
    ReplayResult,
    ValidationFailure,
    compare_candidates,
    canonical_public_output_sha256,
    evidence_components,
    latent_to_rating,
    load_bundle,
    load_authorized_bundle,
    plugin_expected_result,
    posterior_predictive_expected_result,
    public_unavailable_payload,
    rank_ratings,
    rating_payload,
    rating_to_latent,
    reference_roster_logit,
    replay,
    replay_authenticated,
    replacement_logit_delta,
)


ROOT = Path(__file__).resolve().parents[4]


def plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT)


def selected_replay(bundle, as_of="2026-03-01T14:00:00Z"):
    return replay_authenticated(
        bundle,
        bundle.selected_candidate_id,
        as_of,
    )


def test_bundle_is_deterministic_deep_frozen_and_pins_full_closure(bundle):
    second = load_bundle(ROOT)
    assert bundle.raw_sha256 == second.raw_sha256
    assert set(bundle.raw_sha256) == {
        "candidate_identity",
        "config",
        "fixtures",
        "report",
        "manifest",
        "package_source",
        "model_source",
        "generator_source",
        "tests",
        "player_schema",
        "common_schema",
        "provenance_schema",
        "model_manifest_schema",
        "c1_authority",
    }
    with pytest.raises(TypeError):
        bundle.config["claim_ceiling"]["production_eligible"] = True
    with pytest.raises(AttributeError):
        bundle.fixtures["matches"].append({})


def test_rehash_elevation_is_rejected_by_code_held_predicates(bundle):
    elevated = plain(bundle.candidate_identity)
    elevated["claim_ceiling"]["predictive_performance_authorized"] = True
    elevated["claim_ceiling"]["publication_authorized"] = True
    elevated["claim_ceiling"]["pass_b2"] = True
    elevated["claim_ceiling"]["c2"] = True
    with pytest.raises(ValidationFailure, match="claim ceiling"):
        model._validate_candidate_identity_semantics(elevated)
    assert plain(bundle.c1.payload["claim_boundary"])["pass_b2"] is False


@pytest.mark.parametrize("locator", [
    "data//lol/file.json",
    "data/./lol/file.json",
    "data/lol/../file.json",
    "/absolute/file.json",
    "data\\lol\\file.json",
])
def test_noncanonical_locators_fail_closed(locator):
    with pytest.raises(ValidationFailure):
        model._safe_file(ROOT, locator)


def test_symlink_hardlink_and_duplicate_inode_fail_closed(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(ValidationFailure, match="symlink"):
        model._safe_file(tmp_path, "symlink.json")
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValidationFailure, match="hard-linked"):
        model._safe_file(tmp_path, "hardlink.json")


def test_development_manifest_explicitly_does_not_claim_production_schema(bundle):
    assert bundle.manifest["production_manifest_conformance"] is False
    schema = json.loads(
        (ROOT / "docs/model-v2/contracts/model-manifest.schema.json").read_text()
    )
    common = json.loads((ROOT / "docs/model-v2/contracts/common.schema.json").read_text())
    registry = Registry().with_resource(common["$id"], Resource.from_contents(common))
    errors = list(
        jsonschema.Draft202012Validator(schema, registry=registry).iter_errors(
            plain(bundle.manifest)
        )
    )
    assert errors


def _audit_two_match_fixture(bundle, first_update_available):
    first = copy.deepcopy(plain(bundle.fixtures["matches"][0]))
    second = copy.deepcopy(plain(bundle.fixtures["matches"][1]))
    first.update(
        {
            "match_id": "audit-delayed-first",
            "event_start": "2026-04-01T12:00:00Z",
            "event_end": "2026-04-01T12:30:00Z",
            "outcome_available_at": "2026-04-01T14:00:00Z",
            "season_id": "2026",
            "calendar_year": 2026,
            "patch_id": "26.6",
        }
    )
    first["player_updates"] = [
        {
            "player_id": "a-mid",
            "channel_id": "role_normalized_nonresource_impact",
            "feature_provenance_id": "role-normalized-nonresource-impact-v1",
            "value": 2.0,
            "observation_variance": 0.1,
            "available_at": first_update_available,
            "source_context": "combat",
        }
    ]
    first["policy_updates"] = []
    second.update(
        {
            "match_id": "audit-later-forecast",
            "event_start": "2026-04-01T13:00:00Z",
            "event_end": "2026-04-01T13:30:00Z",
            "outcome_available_at": "2026-04-01T14:30:00Z",
            "season_id": "2026",
            "calendar_year": 2026,
            "patch_id": "26.6",
        }
    )
    second["player_updates"] = []
    second["policy_updates"] = []
    return [first, second]


def test_delayed_cross_boundary_does_not_change_1300_forecast(bundle):
    rows = _audit_two_match_fixture(bundle, "2026-04-01T14:00:00Z")
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2026-04-01T15:00:00Z",
        matches=rows,
    )
    later = {row["match_id"]: row for row in result.forecasts}["audit-later-forecast"]
    assert later["plugin_expected_result"] == 0.5
    assert later["plugin_expected_result"] != pytest.approx(0.6464499102)


def test_same_timestamp_availability_is_excluded_from_forecast(bundle):
    rows = _audit_two_match_fixture(bundle, "2026-04-01T13:00:00Z")
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2026-04-01T15:00:00Z",
        matches=rows,
    )
    later = {row["match_id"]: row for row in result.forecasts}["audit-later-forecast"]
    assert later["plugin_expected_result"] == 0.5


def test_early_forecast_hides_label_and_selection_count(bundle):
    one = copy.deepcopy(plain(bundle.fixtures["matches"][0]))
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T12:30:00Z",
        matches=[one],
    )
    assert result.forecasts[0]["label"] is None
    fixtures = plain(bundle.fixtures)
    fixtures["matches"] = [one]
    config = plain(bundle.config)
    config["selection"]["minimum_eligible_origins"] = 1
    decision = compare_candidates(config, fixtures, "2025-12-20T12:30:00Z")
    assert decision["selected_candidate_id"] is None
    assert decision["eligible_origin_count"] == 0
    assert all(row["eligible_origin_count"] == 0 for row in decision["diagnostics"])


def test_matches_override_receives_full_event_validation(bundle):
    hostile = [copy.deepcopy(plain(bundle.fixtures["matches"][0]))]
    hostile[0]["event_end"] = hostile[0]["event_start"]
    with pytest.raises(ValidationFailure, match="event_start"):
        replay(
            bundle.config,
            bundle.fixtures,
            "random_walk_no_reset+no_resource",
            "2026-01-01T00:00:00Z",
            matches=hostile,
        )


def test_forecast_hash_is_serialized_before_label_and_label_changes_do_not_change_it(bundle):
    late = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T14:00:00Z",
        matches=[bundle.fixtures["matches"][0]],
    ).forecasts[0]
    early = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T12:30:00Z",
        matches=[bundle.fixtures["matches"][0]],
    ).forecasts[0]
    assert early["label"] is None and late["label"] == 1
    assert early["forecast_sha256"] == late["forecast_sha256"]


def test_single_authenticated_estimator_and_full_payload_hash_identity(bundle):
    result = selected_replay(bundle)
    payload = rating_payload(bundle, result, "a-mid", "mid", "L02")
    assert payload["candidate_id"] == bundle.selected_candidate_id
    unsigned = plain(payload)
    output_hash = unsigned.pop("output_sha256")
    assert output_hash == hashlib.sha256(
        json.dumps(unsigned, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    arbitrary = replay(
        bundle.config,
        bundle.fixtures,
        "mean_reversion+no_resource",
        result.as_of,
    )
    with pytest.raises(ValidationFailure, match="loader-issued authenticated replay"):
        rating_payload(bundle, arbitrary, "a-mid", "mid", "L02")


def test_changed_value_changes_full_output_hash(bundle):
    result = selected_replay(bundle)
    key = ("league", "L02", "mid", "a-mid")
    changed_states = dict(result.states)
    changed_states[key] = replace(changed_states[key], mean=changed_states[key].mean + 0.1)
    changed = replace(result, states=changed_states)
    original_payload = rating_payload(bundle, result, "a-mid", "mid", "L02")
    assert original_payload["replay_identity_sha256"] == result.replay_identity_sha256
    with pytest.raises(ValidationFailure, match="loader-issued authenticated replay"):
        rating_payload(bundle, changed, "a-mid", "mid", "L02")


def test_honest_development_and_public_unavailable_semantics(bundle):
    result = selected_replay(bundle)
    payload = rating_payload(bundle, result, "a-mid", "mid", "L02")
    assert payload["status"] == "development_only"
    assert payload["rank_eligible"] is False
    assert payload["reliability"] == {
        "status": "unavailable",
        "label": "unrated",
        "real_sample_count": 0,
        "real_cluster_count": 0,
        "validation_stratum_id": None,
        "probability_wording_approved": False,
        "reason": "authoritative observed rows unavailable",
    }
    public = public_unavailable_payload(bundle, result, "a-mid", "mid", "L02")
    assert public["status"] == "unavailable"
    assert "posterior_mean" not in public
    assert rank_ratings([payload]) == []


def test_public_unavailable_is_structurally_valid_but_repository_semantics_reject_authority(bundle):
    result = selected_replay(bundle)
    payload = plain(public_unavailable_payload(bundle, result, "a-mid", "mid", "L02"))
    contracts = ROOT / "docs/model-v2/contracts"
    schemas = {
        name: json.loads((contracts / name).read_text())
        for name in (
            "player-rating.schema.json",
            "common.schema.json",
            "prediction-provenance.schema.json",
        )
    }
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    jsonschema.Draft202012Validator(
        schemas["player-rating.schema.json"], registry=registry
    ).validate(payload)
    with pytest.raises(ValidationFailure, match="production model authority is unavailable"):
        validate_output_payload("player_rating", payload)


def test_claim_ceiling_and_blockers_are_exact_everywhere(bundle):
    assert plain(bundle.config["claim_ceiling"]) == CLAIM_CEILING
    assert plain(bundle.report["claim_ceiling"]) == CLAIM_CEILING
    assert plain(bundle.manifest["claim_ceiling"]) == CLAIM_CEILING
    assert plain(bundle.candidate_identity["claim_ceiling"]) == CLAIM_CEILING
    assert bundle.report["production_eligible"] is False
    assert bundle.manifest["production_manifest_conformance"] is False


def test_mean_reversion_propagates_to_far_future(bundle):
    candidate = "mean_reversion+no_resource"
    near = replay(bundle.config, bundle.fixtures, candidate, "2026-03-01T14:00:00Z")
    far = replay(
        bundle.config,
        bundle.fixtures,
        candidate,
        "2036-03-01T14:00:00Z",
        target_context={"season_id": "2036", "calendar_year": 2036},
    )
    key = ("league", "L02", "mid", "a-mid")
    assert abs(far.states[key].mean) < abs(near.states[key].mean)
    assert far.states[key].mean != near.states[key].mean
    far_payload_result = replay_authenticated(
        bundle,
        bundle.selected_candidate_id,
        far.as_of,
        target_context={"season_id": "2036", "calendar_year": 2036},
    )
    payload = rating_payload(bundle, far_payload_result, "a-mid", "mid", "L02")
    assert payload["current"] is False
    assert payload["inputs_fresh_for_production"] is False


def test_opponent_only_organization_change_does_not_trigger_player_roster_shock(bundle):
    fixtures = plain(bundle.fixtures)
    changed = copy.deepcopy(fixtures)
    changed["matches"][1]["organization_ids"]["red"] = "OPPONENT-ONLY-CHANGE"
    candidate = "patch_roster_shock+no_resource"
    original = replay(bundle.config, fixtures, candidate, "2026-01-10T14:00:00Z")
    mutated = replay(bundle.config, changed, candidate, "2026-01-10T14:00:00Z")
    key = ("global", None, "mid", "a-mid")
    assert original.states[key].mean == mutated.states[key].mean


def test_cross_league_transfer_carries_player_history(bundle):
    result = selected_replay(bundle)
    forecast = {row["match_id"]: row for row in result.forecasts}["synthetic-004"]
    assert forecast["logit_mean"] != 0.0
    assert ("league", "L01", "mid", "a-mid") in result.states
    assert ("league", "L02", "mid", "a-mid") in result.states
    assert result.states[("league", "L02", "mid", "a-mid")].organization_id == "ORG-C"


def test_league_and_global_posteriors_are_separately_centered_and_bridged(bundle):
    result = selected_replay(bundle)
    league = result.states[("league", "L02", "mid", "a-mid")]
    global_state = result.states[("global", None, "mid", "a-mid")]
    assert (league.mean, league.variance) != (global_state.mean, global_state.variance)
    unavailable = rating_payload(bundle, result, "b-top", "top", "L02", "global")
    assert unavailable["status"] == "unavailable"


def test_all_resource_candidates_are_executed_and_double_count_is_not_selectable(bundle):
    decision = compare_candidates(
        bundle.config, bundle.fixtures, bundle.report["evaluation_cutoff"]
    )
    diagnostics = decision["diagnostics"]
    assert {row["resource_candidate_id"] for row in diagnostics} == set(RESOURCE_CANDIDATES)
    assert len(diagnostics) == len(DECAY_CANDIDATES) * len(RESOURCE_CANDIDATES)
    no_resource = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2026-01-10T12:30:00Z",
    )
    joint = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+joint_resource_to_performance",
        "2026-01-10T12:30:00Z",
    )
    assert no_resource.forecasts[1]["logit_mean"] != joint.forecasts[1]["logit_mean"]
    assert all(
        not row["selectable"]
        for row in diagnostics
        if row["resource_candidate_id"] == "player_policy_double_count_sensitivity"
    )


def test_same_map_resource_updates_policy_but_not_canonical_skill(bundle):
    no_resource = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T14:00:00Z",
    )
    sensitivity = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+player_policy_double_count_sensitivity",
        "2025-12-20T14:00:00Z",
    )
    assert no_resource.policy_states[("a-mid", "same_map_resource_share")] == 0.35
    key = ("league", "L01", "mid", "a-mid")
    assert no_resource.states[key].mean != sensitivity.states[key].mean


def test_team_outcome_anchor_updates_support_without_farm_penalty(bundle):
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T14:00:00Z",
    )
    support = result.states[("league", "L01", "support", "a-support")]
    assert support.mean > 0
    assert support.source_contexts == ("team_outcome_anchor",)


def test_full_reset_only_occurs_at_registered_season_boundary(bundle):
    config = plain(bundle.config)
    candidate = {
        item["candidate_id"]: item for item in config["decay_candidates"]
    }["full_reset"]
    state = model.PlayerState(
        "a-mid",
        "mid",
        "league",
        "L01",
        0.8,
        0.2,
        "2026-01-01T00:00:00Z",
        "26.1",
        "2026",
        2026,
        "ORG-A",
        "L01",
    )
    within = model._transition(
        state,
        candidate,
        model._parse_time("2026-02-01T00:00:00Z"),
        season_id="2026",
    )
    boundary = model._transition(
        state,
        candidate,
        model._parse_time("2027-01-01T00:00:00Z"),
        season_id="2027",
    )
    assert within.mean == state.mean
    assert boundary.mean == 0.0


def test_gauss_hermite_matches_independent_numerical_oracle():
    mean = 1.1
    variance = 2.4
    steps = 100000
    low, high = -9.0, 9.0
    width = (high - low) / steps
    total = 0.0
    for index in range(steps + 1):
        z = low + index * width
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        value = plugin_expected_result(mean + math.sqrt(variance) * z) * density
        total += value * (0.5 if index in (0, steps) else 1.0)
    oracle = total * width
    actual = posterior_predictive_expected_result(mean, variance)
    assert abs(actual - oracle) < 2e-7
    assert posterior_predictive_expected_result(0.0, variance) == pytest.approx(0.5)


@pytest.mark.parametrize("role", model.ROLES)
def test_end_to_end_reference_replacement_equality_on_emitted_ratings(bundle, role):
    result = selected_replay(bundle)
    player = f"a-{role}"
    payload = rating_payload(bundle, result, player, role, "L02")
    weights = bundle.config["reference_policy"]["role_weights"]
    incumbent = payload["posterior_mean"]
    replacement = incumbent + 37.0
    base = {item: 0.0 for item in model.ROLES}
    changed = dict(base)
    base[role] = rating_to_latent(incumbent)
    changed[role] = rating_to_latent(replacement)
    aggregate_delta = reference_roster_logit(changed, weights, weights) - reference_roster_logit(
        base, weights, weights
    )
    emitted_delta = replacement_logit_delta(role, incumbent, replacement, weights, weights)
    assert aggregate_delta == pytest.approx(emitted_delta)
    assert payload["reference_policy_id"] == "equal-role-reference-policy-v1"
    assert payload["reference_population_id"] == "synthetic-role-reference-population-v1"


def test_exact_1500_400_transform_and_interval_arithmetic(bundle):
    assert latent_to_rating(0.0) == 1500.0
    assert latent_to_rating(math.log(10.0)) == pytest.approx(1900.0)
    assert rating_to_latent(1900.0) == pytest.approx(math.log(10.0))
    result = selected_replay(bundle)
    payload = rating_payload(bundle, result, "a-mid", "mid", "L02")
    interval = payload["interval_95_model_range"]
    assert interval["lower"] <= payload["posterior_mean"] <= interval["upper"]
    assert interval["upper"] - payload["posterior_mean"] == pytest.approx(
        payload["posterior_mean"] - interval["lower"]
    )


def test_evidence_formula_is_pinned_and_uses_registered_denominator(bundle):
    result = selected_replay(bundle)
    state = result.states[("league", "L02", "mid", "a-mid")]
    evidence = evidence_components(state, bundle.config)
    coverage = evidence["source_context_coverage"]
    required = bundle.config["evidence"]["required_context_ids"]
    assert evidence["selection_status"] == "selected"
    assert coverage["required_context_count"] == len(required)
    assert coverage["coverage_fraction"] == pytest.approx(
        coverage["supported_context_count"] / len(required)
    )
    assert coverage["identity_terms_status"] == "unsupported"


def test_candidate_identity_is_not_independent_authorization(bundle):
    identity = bundle.candidate_identity
    assert identity["identity_kind"] == "l4_player_candidate_content_identity"
    assert identity["authorization_status"] == "absent"
    assert identity["independent_l4_authority_present"] is False
    assert identity["external_authority_expected_sha256"] is None
    with pytest.raises(ValidationFailure, match="independent L4 authority is absent"):
        load_authorized_bundle(ROOT)


def test_forged_rank_mapping_is_rejected(bundle):
    result = selected_replay(bundle)
    issued = rating_payload(bundle, result, "a-mid", "mid", "L02")
    forged = plain(issued)
    forged["rank_eligible"] = True
    forged["status"] = "ok"
    with pytest.raises(ValidationFailure, match="FORGED_RANK"):
        rank_ratings([forged])
    assert rank_ratings([issued]) == []


def test_public_unavailable_hash_covers_complete_payload(bundle):
    result = selected_replay(bundle)
    payload = plain(public_unavailable_payload(bundle, result, "a-mid", "mid", "L02"))
    original = payload["provenance"]["output_sha256"]
    assert canonical_public_output_sha256(payload) == original
    payload["error"]["message"] += " changed"
    assert canonical_public_output_sha256(payload) != original
    payload = plain(public_unavailable_payload(bundle, result, "a-mid", "mid", "L02"))
    payload["lineage"]["train_cutoff"] = "2026-03-01T14:00:01Z"
    assert canonical_public_output_sha256(payload) != original


def test_one_team_outcome_is_one_rank_one_update_with_covariance(bundle):
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T13:00:00Z",
        matches=[bundle.fixtures["matches"][0]],
    )
    assert result.joint_outcome_updates == 1
    top = ("league", "L01", "top", "a-top")
    jungle = ("league", "L01", "jungle", "a-jungle")
    pair = (top, jungle) if top <= jungle else (jungle, top)
    assert result.covariances[pair] != 0.0
    support = result.states[("league", "L01", "support", "a-support")]
    assert support.information == 0.0
    assert "team_outcome_anchor" in support.source_contexts


def test_explicit_final_cutoff_context_applies_distinct_boundary_laws(bundle):
    cases = {
        "season_shock+no_resource": 0.75,
        "calendar_boundary_shock+no_resource": 0.9,
        "full_reset+no_resource": 0.0,
    }
    for candidate, retention in cases.items():
        near = replay(
            bundle.config,
            bundle.fixtures,
            candidate,
            "2026-03-01T14:00:00Z",
            target_context={"season_id": "2026", "calendar_year": 2026},
        )
        far = replay(
            bundle.config,
            bundle.fixtures,
            candidate,
            "2036-03-01T14:00:00Z",
            target_context={"season_id": "2036", "calendar_year": 2036},
        )
        key = ("league", "L02", "mid", "a-mid")
        assert far.states[key].mean == pytest.approx(
            retention * near.states[key].mean
        )
    for candidate in cases:
        with pytest.raises(ValidationFailure, match="target temporal context"):
            replay(
                bundle.config,
                bundle.fixtures,
                candidate,
                "2036-03-01T14:00:00Z",
            )


def test_tier2_match_cannot_create_global_state(bundle):
    match = copy.deepcopy(plain(bundle.fixtures["matches"][0]))
    match["league_tier"] = 2
    result = replay(
        bundle.config,
        bundle.fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T14:00:00Z",
        matches=[match],
    )
    assert not any(key[0] == "global" for key in result.states)


def test_inactive_or_non_main_player_cannot_create_global_state(bundle):
    fixtures = plain(bundle.fixtures)
    for player in fixtures["players"]:
        if player["player_id"] == "a-mid":
            player["global_eligibility"]["active_status"] = "inactive"
            player["global_eligibility"]["roster_membership"] = "substitute"
    fixtures["matches"][0]["policy_updates"] = []
    result = replay(
        bundle.config,
        fixtures,
        "random_walk_no_reset+no_resource",
        "2025-12-20T14:00:00Z",
        matches=[fixtures["matches"][0]],
    )
    assert ("global", None, "mid", "a-mid") not in result.states


def test_no_policy_rows_make_resource_candidates_ineligible(bundle):
    fixtures = plain(bundle.fixtures)
    for match in fixtures["matches"]:
        match["policy_updates"] = []
    decision = compare_candidates(
        bundle.config, fixtures, bundle.report["evaluation_cutoff"]
    )
    selectable = [row for row in decision["diagnostics"] if row["selectable"]]
    assert selectable
    assert {row["resource_candidate_id"] for row in selectable} == {"no_resource"}
    assert decision["selected_candidate_id"].endswith("+no_resource")
    for row in decision["diagnostics"]:
        if row["resource_candidate_id"] != "no_resource":
            assert row["eligible_resource_observations"] == 0
            assert row["selectable"] is False


def test_resource_candidates_have_distinct_state_and_evidence_paths(bundle):
    results = {
        resource: replay(
            bundle.config,
            bundle.fixtures,
            f"random_walk_no_reset+{resource}",
            "2026-01-10T12:30:00Z",
        )
        for resource in (
            "joint_resource_to_performance",
            "lagged_pre_map_policy",
            "no_resource",
        )
    }
    assert results["joint_resource_to_performance"].resource_evidence["state_path"] == "joint_policy_latent"
    assert results["lagged_pre_map_policy"].resource_evidence["state_path"] == "lagged_observed_policy"
    assert results["no_resource"].resource_evidence["state_path"] == "excluded"
    means = {
        result.forecasts[1]["logit_mean"] for result in results.values()
    }
    assert len(means) == 3


def test_roster_role_mismatch_is_rejected(bundle):
    fixtures = plain(bundle.fixtures)
    for player in fixtures["players"]:
        if player["player_id"] == "a-mid":
            player["role"] = "top"
    with pytest.raises(ValidationFailure, match="metadata role"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


@pytest.mark.parametrize("variance", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_observation_variance_is_controlled_failure(bundle, variance):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"][0]["player_updates"][0]["observation_variance"] = variance
    with pytest.raises(
        ValidationFailure, match="finite and strictly positive"
    ):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


def test_feature_provenance_mismatch_is_rejected(bundle):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"][0]["player_updates"][0][
        "feature_provenance_id"
    ] = "forged-provenance"
    with pytest.raises(ValidationFailure, match="provenance mismatch"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


@pytest.mark.parametrize("mean,variance", [(5.0, 100.0), (20.0, 400.0)])
def test_adaptive_gauss_hermite_extreme_accuracy(mean, variance):
    from scipy.integrate import quad

    def integrand(z):
        return (
            plugin_expected_result(mean + math.sqrt(variance) * z)
            * math.exp(-0.5 * z * z)
            / math.sqrt(2.0 * math.pi)
        )

    oracle = quad(
        integrand, -12.0, 12.0, epsabs=1e-13, epsrel=1e-13, limit=500
    )[0]
    actual = posterior_predictive_expected_result(mean, variance)
    assert abs(actual - oracle) < 1e-7


def test_posterior_predictive_outside_accuracy_domain_fails_closed():
    with pytest.raises(ValidationFailure, match="safety domain"):
        posterior_predictive_expected_result(20.1, 1.0)
    with pytest.raises(ValidationFailure, match="safety domain"):
        posterior_predictive_expected_result(0.0, 400.1)


def test_replay_identity_commits_complete_replay_and_rejects_caller_objects(bundle):
    issued = selected_replay(bundle)
    assert issued.replay_identity_sha256
    assert issued.replay_input_sha256
    assert issued.ordered_events_sha256
    assert issued.replay_source_sha256 == bundle.raw_sha256["model_source"]
    forged = replace(issued, joint_outcome_updates=issued.joint_outcome_updates + 1)
    with pytest.raises(ValidationFailure, match="loader-issued authenticated replay"):
        rating_payload(bundle, forged, "a-mid", "mid", "L02")
    raw = replay(
        bundle.config,
        bundle.fixtures,
        bundle.selected_candidate_id,
        issued.as_of,
    )
    with pytest.raises(ValidationFailure, match="loader-issued authenticated replay"):
        rating_payload(bundle, raw, "a-mid", "mid", "L02")


@pytest.mark.parametrize(
    "schema_role",
    ["player_schema", "common_schema", "provenance_schema", "model_manifest_schema"],
)
def test_complete_schema_closure_is_code_pinned(bundle, schema_role):
    forged = plain(bundle.candidate_identity)
    schema = next(
        row for row in forged["artifacts"] if row["role"] == schema_role
    )
    schema["raw_sha256"] = "0" * 64
    schema["canonical_sha256"] = "1" * 64
    with pytest.raises(ValidationFailure, match="schema identity"):
        model._validate_candidate_identity_semantics(forged)


def test_resource_support_is_origin_local_and_post_forecast_rows_do_not_unlock(bundle):
    fixtures = plain(bundle.fixtures)
    for match in fixtures["matches"]:
        match["policy_updates"] = []
    late = copy.deepcopy(fixtures["matches"][-1]["player_updates"][0])
    late.pop("source_context")
    late.update(
        {
            "channel_id": "same_map_resource_share",
            "feature_provenance_id": "same-map-resource-share-policy-v1",
            "available_at": "2026-03-01T15:00:00Z",
        }
    )
    fixtures["matches"][-1]["policy_updates"] = [late]
    decision = compare_candidates(
        bundle.config, fixtures, "2026-03-01T16:00:00Z"
    )
    resource_rows = [
        row
        for row in decision["diagnostics"]
        if row["resource_candidate_id"] != "no_resource"
    ]
    assert all(row["eligible_resource_observations"] == 0 for row in resource_rows)
    assert all(row["selectable"] is False for row in resource_rows)


@pytest.mark.parametrize(
    "membership,status,ambiguous",
    [
        ("substitute", "active", False),
        ("main", "inactive", False),
        ("main", "active", True),
    ],
)
def test_policy_rows_require_active_exact_main_roster(
    bundle, membership, status, ambiguous
):
    fixtures = plain(bundle.fixtures)
    player = next(row for row in fixtures["players"] if row["player_id"] == "a-mid")
    player["global_eligibility"]["roster_membership"] = membership
    player["global_eligibility"]["active_status"] = status
    player["global_eligibility"]["roster_ambiguous"] = ambiguous
    with pytest.raises(ValidationFailure, match="active unambiguous main-roster"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


def test_policy_row_rejects_unknown_or_nonroster_player(bundle):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"][0]["policy_updates"][0]["player_id"] = "ghost"
    with pytest.raises(ValidationFailure, match="exact match roster"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


def test_delayed_old_season_update_never_regresses_context_or_reapplies_shock(bundle):
    fixtures = plain(bundle.fixtures)
    old = fixtures["matches"][0]
    old["outcome_available_at"] = "2026-01-10T14:00:00Z"
    for update in old["player_updates"]:
        update["available_at"] = "2026-01-10T14:00:00Z"
    for update in old["policy_updates"]:
        update["available_at"] = "2026-01-10T14:00:00Z"
    result = replay(
        bundle.config,
        fixtures,
        "season_shock+no_resource",
        "2026-01-10T15:00:00Z",
        matches=[fixtures["matches"][1], old],
        target_context={"season_id": "2026", "calendar_year": 2026},
    )
    state = result.states[("league", "L01", "mid", "a-mid")]
    assert state.season_id == "2026"
    assert state.calendar_year == 2026
    assert state.context_event_start == "2026-01-10T15:00:00Z"


def test_cross_league_carry_recenters_destination_common_mode(bundle):
    as_of = "2026-03-01T12:30:00Z"
    base = replay(
        bundle.config, bundle.fixtures, bundle.selected_candidate_id, as_of
    )
    config = plain(bundle.config)
    config["league_centers"]["L02"] = 0.25
    shifted = replay(config, bundle.fixtures, bundle.selected_candidate_id, as_of)
    keys = [
        ("league", "L02", role, f"a-{role}")
        for role in model.ROLES
    ]
    base_means = [base.states[key].mean for key in keys]
    shifted_means = [shifted.states[key].mean for key in keys]
    deltas = [new - old for old, new in zip(base_means, shifted_means)]
    assert max(deltas) - min(deltas) == pytest.approx(0.0)
    assert [
        base_means[i] - base_means[j]
        for i in range(len(keys))
        for j in range(i)
    ] == pytest.approx(
        [
            shifted_means[i] - shifted_means[j]
            for i in range(len(keys))
            for j in range(i)
        ]
    )
    weights = bundle.config["reference_policy"]["role_weights"]
    assert sum(
        weights[role]
        * (shifted.states[("league", "L02", role, f"a-{role}")].mean + 0.25)
        for role in model.ROLES
    ) == pytest.approx(
        sum(
            weights[role]
            * (base.states[("league", "L02", role, f"a-{role}")].mean + 0.05)
            for role in model.ROLES
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("player", float("nan")),
        ("player", float("inf")),
        ("policy", float("nan")),
        ("policy", float("-inf")),
    ],
)
def test_nonfinite_feature_values_are_rejected(bundle, field, value):
    fixtures = plain(bundle.fixtures)
    collection = "player_updates" if field == "player" else "policy_updates"
    fixtures["matches"][0][collection][0]["value"] = value
    with pytest.raises(ValidationFailure, match="feature value must be finite"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


@pytest.mark.parametrize("blue_win", [True, False, -1, 2, 0.5])
def test_blue_win_requires_exact_binary_integer(bundle, blue_win):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"][0]["blue_win"] = blue_win
    with pytest.raises(ValidationFailure, match="blue_win"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


@pytest.mark.parametrize("tier", [True, False, 1.0, 2.5])
def test_tier_requires_exact_integer(bundle, tier):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"][0]["league_tier"] = tier
    with pytest.raises(ValidationFailure, match="league_tier"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


def test_target_context_must_match_as_of_calendar(bundle):
    with pytest.raises(ValidationFailure, match="inconsistent with as_of"):
        replay(
            bundle.config,
            bundle.fixtures,
            "mean_reversion+no_resource",
            "2036-03-01T14:00:00Z",
            target_context={"season_id": "2036", "calendar_year": 2035},
        )


def test_adaptive_quadrature_nonconvergence_emits_no_numeric_result(monkeypatch):
    import scipy.special
    import numpy as np

    calls = {"count": 0}

    def alternating_roots(order):
        calls["count"] += 1
        estimate = 0.2 if calls["count"] % 2 else 0.8
        return (
            np.array([0.0]),
            np.array([math.sqrt(2.0 * math.pi) * estimate]),
        )

    monkeypatch.setattr(scipy.special, "roots_hermitenorm", alternating_roots)
    with pytest.raises(ValidationFailure, match="did not converge"):
        posterior_predictive_expected_result(5.0, 100.0)


def test_delayed_old_season_covariance_has_no_reverse_shock_and_is_shuffle_stable(bundle):
    fixtures = plain(bundle.fixtures)
    old = fixtures["matches"][0]
    old["player_updates"] = [old["player_updates"][0]]
    old["player_updates"][0]["available_at"] = "2026-01-10T14:00:00Z"
    old["policy_updates"] = []
    newer = fixtures["matches"][1]
    newer["player_updates"] = []
    newer["policy_updates"] = []
    rows = [old, newer]
    baseline = replay(
        bundle.config,
        fixtures,
        "season_shock+no_resource",
        "2026-01-10T13:00:00Z",
        matches=rows,
        target_context={"season_id": "2026", "calendar_year": 2026},
    )
    delayed = replay(
        bundle.config,
        fixtures,
        "season_shock+no_resource",
        "2026-01-10T15:00:00Z",
        matches=rows,
        target_context={"season_id": "2026", "calendar_year": 2026},
    )
    shuffled = replay(
        bundle.config,
        fixtures,
        "season_shock+no_resource",
        "2026-01-10T15:00:00Z",
        matches=list(reversed(rows)),
        target_context={"season_id": "2026", "calendar_year": 2026},
    )
    left = ("league", "L01", "top", "a-top")
    right = ("league", "L01", "jungle", "a-jungle")
    pair = (left, right) if left <= right else (right, left)
    delayed_ratio = delayed.covariances[pair] / baseline.covariances[pair]
    state = baseline.states[left]
    process_variance = next(
        row["process_variance_per_day"]
        for row in bundle.config["decay_candidates"]
        if row["candidate_id"] == "season_shock"
    )
    transitioned_variance = state.variance + process_variance / 24.0
    observation_variance = old["player_updates"][0]["observation_variance"]
    gain = transitioned_variance / (
        transitioned_variance + observation_variance
    )
    no_reverse_shock_ratio = 1.0 - gain
    assert delayed_ratio == pytest.approx(no_reverse_shock_ratio)
    assert delayed_ratio != pytest.approx(0.75 * no_reverse_shock_ratio)
    assert shuffled.covariances == delayed.covariances


def test_candidate_comparison_uses_one_persisted_common_origin_set(bundle):
    fixtures = plain(bundle.fixtures)
    fifth = copy.deepcopy(fixtures["matches"][-1])
    fifth["match_id"] = "synthetic-005"
    fifth["event_start"] = "2026-04-01T12:00:00Z"
    fifth["event_end"] = "2026-04-01T12:45:00Z"
    fifth["outcome_available_at"] = "2026-04-01T13:00:00Z"
    for update in fifth["player_updates"]:
        update["available_at"] = "2026-04-01T13:00:00Z"
    fifth["policy_updates"] = []
    fixtures["matches"].append(fifth)
    decision = compare_candidates(
        bundle.config, fixtures, "2026-04-01T14:00:00Z"
    )
    assert len(decision["common_origin_ids"]) == 4
    assert decision["common_origin_sha256"] == hashlib.sha256(
        json.dumps(
            plain(decision["common_origin_ids"]),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert {
        tuple(row["origin_ids"]) for row in decision["diagnostics"]
    } == {tuple(decision["common_origin_ids"])}
    assert {
        row["origin_set_sha256"] for row in decision["diagnostics"]
    } == {decision["common_origin_sha256"]}
    assert {
        row["eligible_origin_count"] for row in decision["diagnostics"]
    } == {4}
    random_walk = {
        row["resource_candidate_id"]: row
        for row in decision["diagnostics"]
        if row["decay_candidate_id"] == "random_walk_no_reset"
    }
    assert random_walk["no_resource"]["log_loss"] < random_walk[
        "lagged_pre_map_policy"
    ]["log_loss"]
    assert decision["selected_candidate_id"] == "random_walk_no_reset+no_resource"


@pytest.mark.parametrize(
    "field,value",
    [
        ("international_eligible", "false"),
        ("international_eligible", None),
        ("connectivity_supported", "false"),
        ("connectivity_supported", 1),
        ("bridge_component_id", ""),
        ("bridge_component_id", None),
        ("version", ""),
        ("version", None),
    ],
)
def test_malformed_global_structural_eligibility_is_rejected(bundle, field, value):
    fixtures = plain(bundle.fixtures)
    player = next(row for row in fixtures["players"] if row["player_id"] == "a-mid")
    player["global_eligibility"][field] = value
    with pytest.raises(ValidationFailure, match="global eligibility"):
        replay(
            bundle.config,
            fixtures,
            "random_walk_no_reset+no_resource",
            "2025-12-20T14:00:00Z",
        )


def test_generator_stdout_has_one_canonical_lf_and_explicit_digest_definition():
    from lol_kills.v2.ratings.player.generate_artifacts import serialize_stdout

    value = {
        "stdout_hash_definition": "sha256 of complete canonical compact JSON UTF-8 stdout bytes including exactly one trailing LF",
        "z": 1,
        "a": 2,
    }
    raw = serialize_stdout(value)
    assert raw == (
        b'{"a":2,"stdout_hash_definition":"sha256 of complete canonical compact '
        b'JSON UTF-8 stdout bytes including exactly one trailing LF","z":1}\n'
    )
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def _six_fixture_common_origin_case(bundle):
    fixtures = plain(bundle.fixtures)
    for number, month in ((5, 4), (6, 5)):
        row = copy.deepcopy(fixtures["matches"][-1])
        row["match_id"] = f"synthetic-00{number}"
        row["event_start"] = f"2026-{month:02d}-01T12:00:00Z"
        row["event_end"] = f"2026-{month:02d}-01T12:45:00Z"
        row["outcome_available_at"] = f"2026-{month:02d}-01T13:00:00Z"
        for update in row["player_updates"]:
            update["available_at"] = f"2026-{month:02d}-01T13:00:00Z"
        row["policy_updates"] = []
        fixtures["matches"].append(row)
    return fixtures


def test_missing_origin_candidate_is_unavailable_with_actual_identity(bundle, monkeypatch):
    fixtures = _six_fixture_common_origin_case(bundle)
    original_replay = model.replay
    calls = {"target": 0}

    def attacked_replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        if args[2] == "random_walk_no_reset+no_resource":
            calls["target"] += 1
            if calls["target"] == 2:
                result = replace(
                    result,
                    forecasts=tuple(
                        row
                        for row in result.forecasts
                        if row["match_id"] != "synthetic-002"
                    ),
                )
        return result

    monkeypatch.setattr(model, "replay", attacked_replay)
    decision = model.compare_candidates(
        bundle.config, fixtures, "2026-05-01T14:00:00Z"
    )
    row = next(
        item
        for item in decision["diagnostics"]
        if item["candidate_id"] == "random_walk_no_reset+no_resource"
    )
    assert tuple(decision["common_origin_ids"]) == (
        "synthetic-002",
        "synthetic-003",
        "synthetic-004",
        "synthetic-005",
        "synthetic-006",
    )
    assert tuple(row["origin_ids"]) == (
        "synthetic-003",
        "synthetic-004",
        "synthetic-005",
        "synthetic-006",
    )
    assert row["eligible_origin_count"] == 4
    assert row["origin_identity_status"] == "origin_identity_mismatch"
    assert row["selectable"] is False
    assert row["log_loss"] is None
    assert decision["status"] == "origin_identity_incomplete"
    assert decision["selected_candidate_id"] is None
    assert decision["verified_selectable_origin_sha256"] is None


def test_duplicate_replacement_origin_cannot_remain_selectable(bundle, monkeypatch):
    fixtures = _six_fixture_common_origin_case(bundle)
    original_replay = model.replay
    target = "random_walk_no_reset+lagged_pre_map_policy"

    def attacked_replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        if args[2] == target:
            forecasts = list(result.forecasts)
            index_two = next(
                index
                for index, row in enumerate(forecasts)
                if row["match_id"] == "synthetic-002"
            )
            replacement = next(
                row for row in forecasts if row["match_id"] == "synthetic-003"
            )
            forecasts[index_two] = replacement
            result = replace(result, forecasts=tuple(forecasts))
        return result

    monkeypatch.setattr(model, "replay", attacked_replay)
    decision = model.compare_candidates(
        bundle.config, fixtures, "2026-05-01T14:00:00Z"
    )
    row = next(
        item for item in decision["diagnostics"] if item["candidate_id"] == target
    )
    assert len(row["origin_ids"]) == 5
    assert tuple(row["origin_ids"]).count("synthetic-003") == 2
    assert row["origin_set_sha256"] != decision["common_origin_sha256"]
    assert row["origin_identity_status"] == "origin_identity_mismatch"
    assert row["selectable"] is False
    assert row["log_loss"] is None
    assert decision["status"] == "origin_identity_incomplete"
    assert decision["selected_candidate_id"] is None
    assert decision["verified_selectable_origin_sha256"] is None


def test_extra_eligible_origin_is_not_silently_filtered(bundle, monkeypatch):
    fixtures = _six_fixture_common_origin_case(bundle)
    original_replay = model.replay
    target = "mean_reversion+lagged_pre_map_policy"

    def attacked_replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        if args[2] == target:
            extra = plain(
                next(
                    row
                    for row in result.forecasts
                    if row["match_id"] == "synthetic-003"
                )
            )
            extra["match_id"] = "synthetic-extra"
            extra["event_start"] = "2026-03-15T12:00:00Z"
            result = replace(result, forecasts=result.forecasts + (extra,))
        return result

    monkeypatch.setattr(model, "replay", attacked_replay)
    decision = model.compare_candidates(
        bundle.config, fixtures, "2026-05-01T14:00:00Z"
    )
    row = next(
        item for item in decision["diagnostics"] if item["candidate_id"] == target
    )
    assert "synthetic-extra" in row["origin_ids"]
    assert row["eligible_origin_count"] == 6
    assert row["origin_identity_status"] == "origin_identity_mismatch"
    assert row["selectable"] is False


def test_reordered_candidate_origin_sequence_fails_closed(bundle, monkeypatch):
    fixtures = _six_fixture_common_origin_case(bundle)
    original_replay = model.replay
    target = "calendar_boundary_shock+no_resource"

    def attacked_replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        if args[2] == target:
            result = replace(result, forecasts=tuple(reversed(result.forecasts)))
        return result

    monkeypatch.setattr(model, "replay", attacked_replay)
    decision = model.compare_candidates(
        bundle.config, fixtures, "2026-05-01T14:00:00Z"
    )
    row = next(
        item for item in decision["diagnostics"] if item["candidate_id"] == target
    )
    assert tuple(row["origin_ids"]) == tuple(
        reversed(decision["common_origin_ids"])
    )
    assert row["origin_identity_status"] == "origin_identity_mismatch"
    assert row["selectable"] is False
    assert row["log_loss"] is None


def test_duplicate_match_ids_in_comparison_inputs_are_rejected(bundle):
    fixtures = plain(bundle.fixtures)
    fixtures["matches"].append(copy.deepcopy(fixtures["matches"][0]))
    with pytest.raises(ValidationFailure, match="match IDs must be unique"):
        compare_candidates(
            bundle.config, fixtures, bundle.report["evaluation_cutoff"]
        )
