from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lol_kills import private_rating_authority as ratings
from tools.live_fair_odds import model


NOW = datetime(2026, 8, 1, 20, 35, tzinfo=timezone.utc)
EVENT_START = "2026-08-01T20:30:00+00:00"
ROSTER_RECEIPT_SHA = "a" * 64
ROSTER_REGISTRY_SHA = "b" * 64
RECEIPT_LOCATOR = (
    "data/lol/private_rating_authority/receipts/event-1.json"
)
REGISTRY_LOCATOR = "data/lol/private_rating_authority/registry.json"
EVIDENCE_ROOT = "data/lol/private_rating_authority/evidence"
PREDICTION_LOCATOR = (
    "data/lol/v2/evaluation/multileague-v3/predictions/event-1.json"
)
ARTIFACT_PAYLOADS = {
    kind: f"frozen-{kind}-artifact\n".encode("utf-8")
    for kind in ratings.ARTIFACT_KINDS
}


@pytest.fixture(autouse=True)
def approved_semantic_rating_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing event-registry tests run beneath an explicit semantic gate."""

    monkeypatch.setattr(
        ratings.semantic_authority,
        "load_active_semantic_rating_authority_v1",
        lambda **_: {
            "receipt": {
                "authority_id": "semantic-rating-authority-test",
                "issued_at_utc": "2026-08-01T19:50:00+00:00",
                "valid_until_utc": "2026-08-01T22:30:00+00:00",
                "deployment_policy": {
                    "maximum_data_age_seconds": 14 * 24 * 60 * 60,
                },
            },
            "receipt_raw_sha256": "c" * 64,
            "deployment_artifacts": artifact_references(),
        },
    )
    monkeypatch.setattr(
        ratings.ratings_ledger,
        "replay_pre_event_prediction_receipt",
        lambda *_args, **_kwargs: prediction_payload(),
    )


def artifact_references() -> dict[str, dict[str, str]]:
    return {
        kind: {
            "locator": f"{EVIDENCE_ROOT}/{kind}.bin",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for kind, raw in ARTIFACT_PAYLOADS.items()
    }


def _runtime_roster_teams() -> list[dict]:
    result = []
    for side, organization in (("blue", "Alpha"), ("red", "Bravo")):
        result.append(
            {
                "side": side,
                "organization_id": f"org-{organization.lower()}",
                "organization_name": organization,
                "roster_id": f"roster-{organization.lower()}-event-1",
                "players": [
                    {
                        "role": role,
                        "player_id": f"{organization.lower()}-{role}",
                        "display_name": f"{organization} {role}",
                    }
                    for role in ratings.ROLES
                ],
            }
        )
    return result


def prediction_payload() -> dict:
    runtime_teams = _runtime_roster_teams()
    players = []
    diagnostic_teams = []
    starts = {"blue": 1500.0, "red": 1480.0}
    organization_means = {"blue": 15.0, "red": 10.0}
    for roster_team in runtime_teams:
        side = roster_team["side"]
        start = starts[side]
        side_players = [
            {
                "side": side,
                "role": role,
                "player_id": roster_team["players"][index]["player_id"],
                "display_name": roster_team["players"][index]["display_name"],
                "display_rating_mean": start + 10.0 * index,
                "display_rating_sd": 45.0 + index,
            }
            for index, role in enumerate(ratings.ROLES)
        ]
        players.extend(side_players)
        player_mean = sum(
            player["display_rating_mean"] for player in side_players
        ) / len(side_players)
        organization_mean = organization_means[side]
        joint_mean = player_mean + organization_mean
        diagnostic_teams.append(
            {
                "side": side,
                "organization_id": roster_team["organization_id"],
                "organization_name": roster_team["organization_name"],
                "components": {
                    "player_aggregate": {
                        "posterior_mean_logit": (
                            player_mean - ratings.ratings_ledger.rating.DISPLAY_ANCHOR
                        )
                        / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE,
                        "posterior_sd_logit": 35.0
                        / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE,
                    },
                    "organization_residual": {
                        "posterior_mean_logit": organization_mean
                        / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE,
                        "posterior_sd_logit": 20.0
                        / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE,
                    },
                },
                "joint_player_plus_organization": {
                    "display_rating_mean": joint_mean,
                    "display_rating_sd": 40.0,
                },
            }
        )
    difference_mean = (
        diagnostic_teams[0]["joint_player_plus_organization"]["display_rating_mean"]
        - diagnostic_teams[1]["joint_player_plus_organization"]["display_rating_mean"]
    )
    predictions = {
        model_id: {
            "p_blue": 0.5,
            "p_red": 0.5,
            "latent_mean": difference_mean
            / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE,
            "latent_variance": (
                50.0 / ratings.ratings_ledger.rating.DISPLAY_LOGIT_SCALE
            )
            ** 2,
        }
        for model_id in ratings.ratings_ledger.MODEL_IDS
    }
    return {
        "artifact_sha256": "d" * 64,
        "captured_at_utc": "2026-08-01T19:56:00+00:00",
        "source_snapshot": {
            "latest_observed_source_time": "2026-07-25T00:00:00"
        },
        "event": {
            "event_id": "event-1-map-1",
            "event_start_utc": EVENT_START,
            "league": "LCS",
        },
        "input_receipts": {
            "roster": {
                "canonical_sha256": ROSTER_RECEIPT_SHA,
                "receipt": {"teams": runtime_teams},
            }
        },
        "evaluation_predictions": predictions,
        "event_rating_diagnostics": {
            "players": players,
            "teams": diagnostic_teams,
        },
    }


def prediction_reference() -> dict[str, str]:
    raw = json.dumps(prediction_payload()).encode("utf-8")
    return {
        "locator": PREDICTION_LOCATOR,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_sha256": prediction_payload()["artifact_sha256"],
    }


def teams() -> list[dict]:
    expected, _ = ratings._expected_rating_outputs_from_prediction(
        prediction_payload()
    )
    return deepcopy(expected)


def strength(team_values: list[dict] | None = None) -> dict:
    values = team_values or teams()
    if team_values is None:
        _, expected = ratings._expected_rating_outputs_from_prediction(
            prediction_payload()
        )
        return deepcopy(expected)
    mean = values[0]["posterior_mean"] - values[1]["posterior_mean"]
    return {
        "orientation": "blue_minus_red",
        "estimand_scope": values[0]["estimand"]["scope"],
        "included_components": values[0]["estimand"]["included_components"],
        "posterior_mean": mean,
        "posterior_interval_95": [mean - 100, mean + 100],
    }


def receipt(**overrides) -> dict:
    supplied_teams = overrides.pop("teams", None)
    team_values = supplied_teams if supplied_teams is not None else teams()
    values = {
        "rating_record_id": "rating-event-1-v1",
        "producer_id": "rating-pipeline-1",
        "event_id": "event-1-map-1",
        "event_start": EVENT_START,
        "league": "LCS",
        "roster_receipt_sha256": ROSTER_RECEIPT_SHA,
        "roster_registry_sha256": ROSTER_REGISTRY_SHA,
        "data_cutoff_at": "2026-07-25T00:00:00+00:00",
        "produced_at": "2026-08-01T19:56:00+00:00",
        "valid_until": "2026-08-01T22:30:00+00:00",
        "maximum_data_age_seconds": 14 * 24 * 60 * 60,
        "artifacts": artifact_references(),
        "prediction_receipt": prediction_reference(),
        "teams": team_values,
        "strength_difference": (
            strength(team_values) if supplied_teams is not None else strength()
        ),
    }
    values.update(overrides)
    return ratings.build_event_rating_receipt(**values)


def registry(candidate: dict | None = None, **overrides) -> dict:
    values = {
        "receipts": [(RECEIPT_LOCATOR, candidate or receipt())],
        "registry_id": "rating-review-1",
        "independent_reviewer_id": "independent-reviewer-1",
        "issued_at": "2026-08-01T19:57:00+00:00",
    }
    values.update(overrides)
    return ratings.build_event_rating_registry(**values)


def registered_roster() -> dict:
    roster_teams = []
    for team in teams():
        roster_teams.append(
            {
                "side": team["side"],
                "organization_id": team["organization_id"],
                "organization_name": team["organization_name"],
                "roster_id": team["roster_id"],
                "players": [
                    {
                        "role": player["role"],
                        "player_id": player["player_id"],
                        "display_name": player["display_name"],
                    }
                    for player in team["players"]
                ],
            }
        )
    return {
        "status": "registered",
        "roster": {
            "event_id": "event-1-map-1",
            "event_start": EVENT_START,
            "league": "LCS",
            "teams": roster_teams,
        },
        "receipt_sha256": ROSTER_RECEIPT_SHA,
        "registry_sha256": ROSTER_REGISTRY_SHA,
    }


def write_json(root: Path, locator: str, value: dict) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_package(root: Path, candidate: dict, value: dict) -> None:
    for kind, raw in ARTIFACT_PAYLOADS.items():
        path = root / EVIDENCE_ROOT / f"{kind}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    prediction_path = root / PREDICTION_LOCATOR
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(prediction_payload()), encoding="utf-8")
    write_json(root, RECEIPT_LOCATOR, candidate)
    write_json(root, REGISTRY_LOCATOR, value)


def load(root: Path, value: dict, roster: dict | None = None, **overrides):
    args = {
        "registry_locator": REGISTRY_LOCATOR,
        "expected_registry_sha256": ratings.sha256_json(value),
        "registered_roster": roster or registered_roster(),
        "event_id": "event-1-map-1",
        "event_start": EVENT_START,
        "league": "LCS",
        "blue_organization_name": "Alpha",
        "red_organization_name": "Bravo",
        "as_of": NOW,
        "root": root,
    }
    args.update(overrides)
    return ratings.load_registered_event_rating(**args)


def test_receipt_replays_exact_players_components_and_difference() -> None:
    candidate = receipt()
    checked = ratings.validate_event_rating_receipt(candidate)
    assert checked["receipt_sha256"] == ratings.sha256_json(candidate)
    assert [team["side"] for team in checked["teams"]] == ["blue", "red"]
    assert [
        player["role"] for player in checked["teams"][0]["players"]
    ] == list(ratings.ROLES)
    assert checked["strength_difference"]["posterior_mean"] == pytest.approx(25.0)
    assert checked["teams"][0]["components"]["lineup_synergy"]["status"] == (
        "UNAVAILABLE"
    )
    assert checked["teams"][0]["components"]["lineup_synergy"]["posterior_mean"] is None


def test_receipt_accepts_only_the_exact_evaluated_runtime_outside_data_roots() -> None:
    references = artifact_references()
    references["player_model"] = {
        "locator": ratings.ratings_ledger.SOURCE_LOCATOR,
        "raw_sha256": "e" * 64,
    }
    references["team_model"] = dict(references["player_model"])
    checked = receipt(artifacts=references)
    assert checked["artifacts"]["player_model"]["locator"] == (
        ratings.ratings_ledger.SOURCE_LOCATOR
    )

    references["player_model"]["locator"] = (
        "lol_kills/v2/ratings/player/unreviewed_runtime.py"
    )
    with pytest.raises(ratings.RatingAuthorityError, match="outside the allowed"):
        receipt(artifacts=references)


def test_missing_component_or_wrong_player_aggregate_is_rejected() -> None:
    missing = teams()
    del missing[0]["components"]["lineup_synergy"]
    with pytest.raises(ratings.RatingAuthorityError, match="every frozen component"):
        receipt(teams=missing)
    mismatch = teams()
    mismatch[0]["components"]["player_aggregate"]["posterior_mean"] += 1
    mismatch[0]["posterior_mean"] += 1
    with pytest.raises(ratings.RatingAuthorityError, match="exact roster"):
        receipt(teams=mismatch)


def test_unavailable_component_cannot_be_smuggled_as_zero() -> None:
    values = teams()
    values[0]["components"]["lineup_synergy"]["posterior_mean"] = 0.0
    with pytest.raises(ratings.RatingAuthorityError, match="null rather than zero"):
        receipt(teams=values)


def test_strength_difference_must_preserve_component_estimand() -> None:
    values = teams()
    difference = strength(values)
    difference["included_components"] = ["player_aggregate"]
    with pytest.raises(ratings.RatingAuthorityError, match="identified-component estimand"):
        receipt(teams=values, strength_difference=difference)


def test_post_event_or_stale_rating_is_rejected() -> None:
    with pytest.raises(ratings.RatingAuthorityError, match="before event_start"):
        receipt(produced_at=EVENT_START)
    with pytest.raises(ratings.RatingAuthorityError, match="freshness"):
        receipt(maximum_data_age_seconds=60)


def test_registry_cannot_register_itself_without_external_digest() -> None:
    value = registry()
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        ratings.validate_event_rating_registry(
            value, expected_registry_sha256=None
        )
    assert error.value.code == "rating_registry_not_registered"


def test_registered_loader_requires_active_semantic_rating_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)

    def unavailable(**_):
        raise ratings.semantic_authority.SemanticRatingAuthorityError(
            "future evaluation unavailable"
        )

    monkeypatch.setattr(
        ratings.semantic_authority,
        "load_active_semantic_rating_authority_v1",
        unavailable,
    )
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value)
    assert error.value.code == "semantic_rating_authority_unavailable"


def test_reviewer_must_be_distinct_and_pre_event() -> None:
    with pytest.raises(ratings.RatingAuthorityError, match="cannot be the rating producer"):
        registry(independent_reviewer_id="rating-pipeline-1")
    with pytest.raises(ratings.RatingAuthorityError, match="before event_start"):
        registry(issued_at=EVENT_START)


def test_registered_loader_replays_roster_and_all_artifact_bindings(
    tmp_path: Path,
) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    loaded = load(tmp_path, value)
    assert loaded["status"] == "registered"
    assert loaded["player_rating_authorized"] is True
    assert loaded["team_rating_authorized"] is True
    assert loaded["match_probability_authorized"] is False
    assert loaded["betting_decision_authorized"] is False
    assert loaded["ratings"]["strength_difference"]["posterior_mean"] == pytest.approx(
        25.0
    )
    assert "rating_to_match_probability_calibration_unavailable" in loaded["blockers"]


def test_registered_loader_rejects_well_formed_but_fabricated_rating_numbers(
    tmp_path: Path,
) -> None:
    forged_teams = teams()
    blue = forged_teams[0]
    for player in blue["players"]:
        player["posterior_mean"] += 10.0
    blue["components"]["player_aggregate"]["posterior_mean"] += 10.0
    blue["posterior_mean"] += 10.0
    blue["posterior_interval_95"] = [
        bound + 10.0 for bound in blue["posterior_interval_95"]
    ]
    forged = receipt(teams=forged_teams)
    value = registry(forged)
    write_package(tmp_path, forged, value)
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value)
    assert error.value.code == "rating_outputs_do_not_replay_exact_prediction"


def test_registered_loader_rejects_prediction_receipt_byte_tamper(
    tmp_path: Path,
) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    (tmp_path / PREDICTION_LOCATOR).write_text("{}", encoding="utf-8")
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value)
    assert error.value.code == "rating_prediction_receipt_digest_mismatch"


def test_exact_prediction_builder_derives_numbers_and_lineage_without_caller_input(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / PREDICTION_LOCATOR
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(prediction_payload()), encoding="utf-8")
    built = ratings.build_event_rating_receipt_from_prediction(
        prediction_receipt_locator=PREDICTION_LOCATOR,
        rating_record_id="rating-event-1-derived",
        producer_id="rating-pipeline-1",
        roster_registry_sha256=ROSTER_REGISTRY_SHA,
        valid_until="2026-08-01T22:30:00+00:00",
        maximum_data_age_seconds=14 * 24 * 60 * 60,
        artifacts=artifact_references(),
        root=tmp_path,
    )
    assert built["teams"] == teams()
    assert built["strength_difference"] == strength()
    assert built["prediction_receipt"] == prediction_reference()


def test_roster_player_or_side_swap_cannot_bind(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    wrong_roster = registered_roster()
    wrong_roster["roster"]["teams"][0]["players"][0]["player_id"] = "attacker"
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value, wrong_roster)
    assert error.value.code == "rating_roster_blue_top_player_id_binding_mismatch"
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(
            tmp_path,
            value,
            blue_organization_name="Bravo",
            red_organization_name="Alpha",
        )
    assert error.value.code == "rating_blue_organization_name_binding_mismatch"


def test_post_registration_artifact_tamper_is_rejected(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    (tmp_path / EVIDENCE_ROOT / "evaluation.bin").write_bytes(b"tampered\n")
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value)
    assert error.value.code == "rating_evaluation_artifact_digest_mismatch"


def test_future_and_expired_rating_are_rejected(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(
            tmp_path,
            value,
            as_of=datetime(2026, 8, 1, 19, 56, 30, tzinfo=timezone.utc),
        )
    assert error.value.code == "rating_registry_from_future"
    with pytest.raises(ratings.RegisteredEventRatingUnavailable) as error:
        load(tmp_path, value, as_of=NOW + timedelta(hours=3))
    assert error.value.code == "registered_event_rating_expired"


def test_receipt_path_escape_is_rejected() -> None:
    with pytest.raises(ratings.RatingAuthorityError, match="receipt root"):
        ratings.build_event_rating_registry(
            receipts=[("../../rating.json", receipt())],
            registry_id="rating-review-1",
            independent_reviewer_id="independent-reviewer-1",
            issued_at="2026-08-01T19:57:00+00:00",
        )


def test_private_worksheet_loads_ratings_only_through_pinned_registry(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_package(tmp_path, candidate, value)
    monkeypatch.setattr(model, "ROOT", tmp_path)
    monkeypatch.setenv(
        model.RATING_REGISTRY_SHA_ENV, ratings.sha256_json(value)
    )
    loaded = model._registered_event_rating(
        event_id="event-1-map-1",
        event_start=EVENT_START,
        league="LCS",
        blue_team="Alpha",
        red_team="Bravo",
        roster_registration=registered_roster(),
        as_of=NOW,
    )
    assert loaded["status"] == "registered"
    assert loaded["team_rating_authorized"] is True
    assert loaded["match_probability_authorized"] is False


def test_private_worksheet_reports_missing_rating_pin(monkeypatch) -> None:
    monkeypatch.delenv(model.RATING_REGISTRY_SHA_ENV, raising=False)
    unavailable = model._registered_event_rating(
        event_id="event-1-map-1",
        event_start=EVENT_START,
        league="LCS",
        blue_team="Alpha",
        red_team="Bravo",
        roster_registration=registered_roster(),
        as_of=NOW,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["blockers"] == ["rating_registry_not_registered"]
