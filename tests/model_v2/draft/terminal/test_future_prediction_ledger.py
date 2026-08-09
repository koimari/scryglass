from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from lol_kills import pregame_roster_capture as roster_capture
from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.draft.terminal import capture_readiness_v1
from lol_kills.v2.draft.terminal import capture_readiness_registry_v1
from lol_kills.v2.draft.terminal import future_prediction_ledger as ledger
from lol_kills.v2.draft.terminal import side_neutral_prediction_v1 as side_neutral
from lol_kills.v2.market import phase_one_collection_v1 as phase_one
from lol_kills.v2.market import side_neutral_capture_bundle_v1 as side_bundle
from lol_kills.v2.market import side_neutral_ledger_v1 as neutral_ledger
from lol_kills.v2.market import fast_event_uncertainty_v1 as fast_uncertainty
from lol_kills.v2.market import full_pipeline_uncertainty_v1 as uncertainty
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)
from lol_kills.v2.ratings.player import pre_side_rating_envelope_v1 as envelope
from lol_kills.v2.ratings.player import pre_side_rating_binding_v1 as binding
from lol_kills.v2.ratings.player import post_validation_refit_v1 as rating_refit
from lol_kills.v2.ratings.player import side_neutral_protocol_review_v1 as neutral_review
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as NEUTRAL_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as NEUTRAL_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC as NEUTRAL_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256 as NEUTRAL_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_v2 import (
    INDEPENDENT_REVIEW_ENV as NEUTRAL_REVIEW_ENV,
)
from lol_kills.v2.ratings.player.side_neutral_collection_implementation_registry_v1 import (
    validate_registered_side_neutral_collection_implementation,
)


EVENT_ID = "future-lcs-evaluation-map-1"
SERIES_ID = "future-lcs-evaluation-series-1"
EVENT_START = "2026-08-08T20:00:00Z"
ACTUAL_MAP_START = "2026-08-08T20:30:00Z"
RATINGS_CAPTURED_AT = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
SIDE_NEUTRAL_RATINGS_CAPTURED_AT = datetime(
    2026, 8, 2, 8, 0, tzinfo=timezone.utc
)
DRAFT_CAPTURED_AT = datetime(2026, 8, 8, 20, 29, tzinfo=timezone.utc)
SIDE_CAPTURED_AT = datetime(2026, 8, 8, 20, 27, 30, tzinfo=timezone.utc)
MAP_START_CAPTURED_AT = datetime(2026, 8, 8, 20, 32, tzinfo=timezone.utc)
LEDGER_CREATED_AT = datetime(2026, 8, 8, 20, 33, tzinfo=timezone.utc)
BLUE_ORG_ID = "oe:team:8eb884e168f28402ce685bedebb5250"
RED_ORG_ID = "oe:team:fc8e90107dabb9a35c490b0d86adea0"
BLUE_ORG_NAME = "Team Liquid"
RED_ORG_NAME = "Cloud9"


def _json_raw(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _teams() -> list[dict]:
    values = (
        (
            "blue",
            BLUE_ORG_ID,
            BLUE_ORG_NAME,
            (
                ("top", "oe:player:f73dde2d8c7879ae8d1fcdc03c4ffeb", "Morgan"),
                ("jungle", "oe:player:52b4f060a577ff87c99118d97fa75f4", "Josedeodo"),
                ("mid", "oe:player:8f8eaac8e99c203ee76e57c5c81c91f", "Quid"),
                ("bot", "oe:player:f6b8af05a5af22004a67a78f3046b92", "Yeon"),
                ("support", "oe:player:05fafeaf23dc3285558a4e968970b9f", "CoreJJ"),
            ),
        ),
        (
            "red",
            RED_ORG_ID,
            RED_ORG_NAME,
            (
                ("top", "oe:player:1efc4e5732e91520d03385900b903f9", "Thanatos"),
                ("jungle", "oe:player:ed455f823b6586e9e3621dfb5aeb42e", "Blaber"),
                ("mid", "oe:player:0a53c1102dafce61e89decc517b3175", "Loki"),
                ("bot", "oe:player:4e0619e57cb50c4ab1230c2dd72df1b", "Tactical"),
                ("support", "oe:player:81acc6bcde5c927c78e0990e9887f84", "Vulcan"),
            ),
        ),
    )
    return [
        {
            "side": side,
            "organization_id": organization_id,
            "organization_name": organization_name,
            "roster_id": f"{EVENT_ID}-{side}",
            "players": [
                {"role": role, "player_id": player_id, "display_name": name}
                for role, player_id, name in players
            ],
        }
        for side, organization_id, organization_name, players in values
    ]


def _roster_raw() -> bytes:
    receipt = roster_capture.build_pregame_roster_receipt(
        raw_source_payload=b'{"kind":"synthetic-pre-event-test-roster"}',
        source="test-fixture",
        source_url="https://example.test/future-lcs-evaluation-map-1",
        source_record_id="future-lcs-evaluation-map-1-roster",
        source_updated_at="2026-08-01T23:56:00Z",
        available_at="2026-08-01T23:56:30Z",
        captured_at="2026-08-01T23:57:00Z",
        event_id=EVENT_ID,
        event_start=EVENT_START,
        league="LCS",
        teams=_teams(),
        capture_protocol_sha256="a" * 64,
    )
    return _json_raw(receipt)


@pytest.fixture(scope="module")
def fresh_rating_source_snapshot(evaluation_root: Path) -> str:
    relative_dir = Path(
        "data/lol/v2/snapshots/rating-deployment/post-validation-refit-test"
    )
    directory = evaluation_root / relative_dir
    directory.mkdir(parents=True)
    for filename in ("maps.parquet", "players.parquet"):
        shutil.copy2(
            evaluation_root / f"data/lol/warehouse/parquet/{filename}",
            directory / filename,
        )
    payload = rating_refit.build_source_snapshot_manifest_v1(
        snapshot_id="post-validation-refit-test",
        maps_locator=(relative_dir / "maps.parquet").as_posix(),
        players_locator=(relative_dir / "players.parquet").as_posix(),
        root=evaluation_root,
        clock=lambda: datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc),
    )
    locator = (relative_dir / "manifest.json").as_posix()
    rating_refit.write_no_clobber(evaluation_root / locator, payload)
    return locator


def _patch_raw() -> bytes:
    evidence = {
        "revision_timestamp": "2026-08-01T23:40:00Z",
        "revision_id": 123456,
        "source_kind": "leaguepedia_data_page_revision",
        "content_sha256": "b" * 64,
    }
    receipt = {
        "schema_version": ratings_ledger.PATCH_RECEIPT_SCHEMA,
        "fixture_id": EVENT_ID,
        "event_start": EVENT_START,
        "as_of": "2026-08-01T23:58:00Z",
        "patch": "26.15",
        "client_patch": "16.15",
        "authority_status": "pre_event_revision",
        "pregame_authorized": True,
        "blockers": [],
        "evidence": evidence,
        "evidence_hash": ratings_ledger._canonical_sha256(
            {"fixture_id": EVENT_ID, "evidence": evidence}
        ),
    }
    return _json_raw(receipt)


@pytest.fixture(scope="module")
def evaluation_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(".").resolve()
    root = tmp_path_factory.mktemp("draft-future-ledger-root")
    (root / "lol_kills").symlink_to(repo_root / "lol_kills", target_is_directory=True)
    (root / "docs").symlink_to(repo_root / "docs", target_is_directory=True)
    (root / "data/lol/v2").mkdir(parents=True)
    (root / "data/lol/v2/models").symlink_to(
        repo_root / "data/lol/v2/models",
        target_is_directory=True,
    )
    shutil.copytree(
        repo_root / "data/lol/v2/snapshots",
        root / "data/lol/v2/snapshots",
    )
    market_protocol = Path(
        "data/lol/v2/evaluation/match-winner-market-v1/future-protocol-v1.json"
    )
    (root / market_protocol.parent).mkdir(parents=True)
    shutil.copy2(repo_root / market_protocol, root / market_protocol)
    (root / "data/lol/warehouse/parquet").mkdir(parents=True)
    (root / "data/lol/warehouse/raw_grid").symlink_to(
        repo_root / "data/lol/warehouse/raw_grid",
        target_is_directory=True,
    )
    for filename in ("maps.parquet", "players.parquet"):
        os.link(
            repo_root / f"data/lol/warehouse/parquet/{filename}",
            root / f"data/lol/warehouse/parquet/{filename}",
        )
    (root / ledger.PREDICTION_PREFIX).mkdir(parents=True)
    (root / ledger.MAP_START_PREFIX).mkdir(parents=True)
    return root


@pytest.fixture(scope="module")
def ratings_receipt(evaluation_root: Path) -> dict:
    return ratings_ledger.build_pre_event_prediction_receipt(
        roster_receipt_raw=_roster_raw(),
        patch_receipt_raw=_patch_raw(),
        series_id=SERIES_ID,
        game_number=1,
        root=evaluation_root,
        clock=lambda: RATINGS_CAPTURED_AT,
    )


def test_post_validation_refit_uses_fresh_source_and_exact_roster_covariance(
    evaluation_root: Path,
    fresh_rating_source_snapshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_result = {
        "artifact_sha256": "e" * 64,
        "phase_one_models_passed": True,
    }
    phase_registry = {
        "receipt": {
            "registry_id": "independent-phase-one-pass-test",
            "registered_at_utc": "2026-08-02T08:30:00+00:00",
        },
        "receipt_raw_sha256": "f" * 64,
        "phase_one_models_independently_passed": True,
    }
    monkeypatch.setattr(
        rating_refit,
        "_registered_pass",
        lambda **_kwargs: (
            phase_registry,
            b'{"phase_one_models_passed":true}\n',
            phase_result,
        ),
    )
    payload = rating_refit.build_post_validation_refit_v1(
        phase_one_result_locator=(
            "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
            "outputs/registered-pass-test.json"
        ),
        source_snapshot_locator=fresh_rating_source_snapshot,
        roster_receipt_raw=_roster_raw(),
        patch_receipt_raw=_patch_raw(),
        root=evaluation_root,
        environment={},
        clock=lambda: datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
    )
    checked = rating_refit.validate_post_validation_refit_v1(
        payload,
        root=evaluation_root,
        environment={},
    )
    assert checked["result_state"] == rating_refit.RESULT_STATE
    assert checked["ratings"]["applied_series"] > 0
    assert checked["ratings"]["applied_maps"] > 0
    assert (
        checked["ratings"]["data_age_seconds_at_event"]
        <= rating_refit.MAXIMUM_DATA_AGE_SECONDS
    )
    assert checked["ratings"]["target_event_outcome_accessed"] is False
    assert all(value is False for value in checked["authority"].values())
    assert all(value is None for value in checked["decision_outputs"].values())

    teams = checked["ratings"]["teams"]
    assert [team["side"] for team in teams] == ["blue", "red"]
    assert all(len(team["players"]) == 5 for team in teams)
    assert len(
        {
            player["player_id"]
            for team in teams
            for player in team["players"]
        }
    ) == 10
    for team in teams:
        components = team["components"]
        expected_mean = (
            components["player_aggregate"]["posterior_mean_logit"]
            + components["organization_residual"]["posterior_mean_logit"]
            + components["league_adjustment"]["posterior_mean_logit"]
        )
        assert team["joint_identified_strength"][
            "posterior_mean_logit"
        ] == pytest.approx(expected_mean)
        for name in ("lineup_synergy", "team_policy"):
            assert components[name]["status"] == "UNAVAILABLE"
            assert components[name]["posterior_mean_logit"] is None
            assert components[name]["posterior_sd_logit"] is None
            assert components[name]["posterior_interval_95_logit"] is None

    difference = checked["ratings"]["strength_difference_blue_minus_red"]
    assert difference["cross_team_covariance_retained"] is True
    assert difference["blue_side_effect_included"] is False
    assert difference["posterior_mean_logit"] == pytest.approx(
        teams[0]["joint_identified_strength"]["posterior_mean_logit"]
        - teams[1]["joint_identified_strength"]["posterior_mean_logit"]
    )

    forged = deepcopy(checked)
    forged["ratings"]["teams"][0]["components"]["lineup_synergy"][
        "posterior_mean_logit"
    ] = 0.0
    forged["artifact_sha256"] = rating_refit._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        rating_refit.PostValidationRefitError,
        match="do not replay",
    ):
        rating_refit.validate_post_validation_refit_v1(
            forged,
            root=evaluation_root,
            environment={},
        )


def test_post_validation_refit_freshness_ceiling_is_fail_closed() -> None:
    cutoff = datetime(2026, 8, 20)
    assert rating_refit._data_age_seconds(
        cutoff,
        cutoff - timedelta(days=13),
    ) == 13 * 24 * 60 * 60
    with pytest.raises(
        rating_refit.PostValidationRefitError,
        match="data-age ceiling",
    ):
        rating_refit._data_age_seconds(
            cutoff,
            cutoff - timedelta(days=15),
        )


@pytest.fixture(scope="module")
def draft_source_raw() -> bytes:
    return _json_raw(
        {
            "eventId": EVENT_ID,
            "kind": "terminal-draft-source-test-fixture",
            "terminalDraftComplete": True,
        }
    )


def _draft_sides() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {
            "top": "Aatrox",
            "jungle": "Nidalee",
            "mid": "Ahri",
            "bot": "Jinx",
            "support": "Thresh",
        },
        {
            "top": "Gnar",
            "jungle": "Sejuani",
            "mid": "Orianna",
            "bot": "Aphelios",
            "support": "Rakan",
        },
    )


def _draft_actions() -> tuple[list[dict], list[dict]]:
    blue, red = _draft_sides()
    ordered = (
        ("blue", "ban", "Renekton", None),
        ("red", "ban", "Vi", None),
        ("blue", "ban", "Azir", None),
        ("red", "ban", "Kalista", None),
        ("blue", "ban", "Nautilus", None),
        ("red", "ban", "Poppy", None),
        ("blue", "pick", blue["top"], "top"),
        ("red", "pick", red["top"], "top"),
        ("red", "pick", red["jungle"], "jungle"),
        ("blue", "pick", blue["jungle"], "jungle"),
        ("blue", "pick", blue["mid"], "mid"),
        ("red", "pick", red["mid"], "mid"),
        ("red", "ban", "Xayah", None),
        ("blue", "ban", "Leona", None),
        ("red", "ban", "Rumble", None),
        ("blue", "ban", "Wukong", None),
        ("red", "pick", red["bot"], "bot"),
        ("blue", "pick", blue["bot"], "bot"),
        ("blue", "pick", blue["support"], "support"),
        ("red", "pick", red["support"], "support"),
    )
    actions = [
        {
            "slot": slot,
            "action_id": f"action-{slot}",
            "side": side,
            "kind": kind,
            "champion_id": champion,
            "role_set": [role] if role else [],
        }
        for slot, (side, kind, champion, role) in enumerate(ordered, 1)
    ]
    assignments = [
        {
            "action_id": action["action_id"],
            "side": action["side"],
            "champion_id": action["champion_id"],
            "role": action["role_set"][0],
        }
        for action in actions
        if action["kind"] == "pick"
    ]
    return actions, assignments


def _draft_metadata(draft_source_raw: bytes) -> dict:
    blue, red = _draft_sides()
    actions, assignments = _draft_actions()
    return {
        "schema_version": "scryglass:terminal-draft-capture-input:v1",
        "event_id": EVENT_ID,
        "series_id": SERIES_ID,
        "game_number": 1,
        "league": "LCS",
        "patch": "26.15",
        "blue_organization_id": BLUE_ORG_ID,
        "blue_organization_name": BLUE_ORG_NAME,
        "red_organization_id": RED_ORG_ID,
        "red_organization_name": RED_ORG_NAME,
        "source": {
            "source_id": "synthetic-terminal-draft-test-source",
            "source_url": "https://example.test/terminal-draft",
            "source_record_id": f"{EVENT_ID}-terminal-draft",
            "available_at_utc": "2026-08-08T20:28:00Z",
            "rights_status": "reviewed",
            "payload_raw_sha256": hashlib.sha256(draft_source_raw).hexdigest(),
        },
        "protocol_validation": {
            "protocol_id": "synthetic-terminal-draft-parser-v1",
            "validator_id": "synthetic-terminal-draft-validator-v1",
            "validator_sha256": "c" * 64,
            "validated_at_utc": "2026-08-08T20:28:30Z",
            "action_order_verified": True,
            "pick_ban_counts_verified": True,
            "blue_red_side_mapping_verified": True,
        },
        "blue": blue,
        "red": red,
        "actions": actions,
        "final_assignments": assignments,
    }


def _pre_side_input() -> dict:
    teams = []
    for index, team in enumerate(_teams(), 1):
        converted = deepcopy(team)
        converted["slot"] = f"team{index}"
        converted.pop("side")
        teams.append(converted)
    return {
        "schema_version": envelope.INPUT_SCHEMA_VERSION,
        "event": {
            "event_id": EVENT_ID,
            "series_id": SERIES_ID,
            "game_number": 1,
            "scheduled_series_start_utc": EVENT_START,
            "league": "LCS",
        },
        "roster_source": {
            "source": "test-fixture",
            "source_url": "https://example.test/future-lcs-evaluation-map-1",
            "source_record_id": f"{EVENT_ID}-roster",
            "source_updated_at_utc": "2026-08-01T23:56:00Z",
            "available_at_utc": "2026-08-01T23:56:30Z",
            "rights_status": "reviewed",
        },
        "teams": teams,
    }


def _side_source_raw() -> bytes:
    return _json_raw(
        {
            "match": {
                "blue": {"name": BLUE_ORG_NAME},
                "red": {"name": RED_ORG_NAME},
            },
            "state": "champion_select",
        }
    )


def _side_binding_input() -> dict:
    return {
        "schema_version": binding.INPUT_SCHEMA_VERSION,
        "event": {
            "event_id": EVENT_ID,
            "series_id": SERIES_ID,
            "game_number": 1,
        },
        "source": {
            "source": "test-public-side-feed",
            "source_url": "https://example.test/public-side-feed/map-1",
            "source_record_id": f"{EVENT_ID}-side-observation",
            "source_updated_at_utc": "2026-08-08T20:26:50Z",
            "available_at_utc": "2026-08-08T20:27:00Z",
            "rights_status": "reviewed",
        },
        "extraction": {
            "format": "strict_json_pointer_v1",
            "blue_organization_name_json_pointer": "/match/blue/name",
            "red_organization_name_json_pointer": "/match/red/name",
        },
    }


@pytest.fixture(scope="module")
def pre_side_envelope(evaluation_root: Path) -> dict:
    return envelope.build_pre_side_rating_envelope(
        input_raw=_json_raw(_pre_side_input()),
        roster_source_payload_raw=b'{"kind":"synthetic-pre-event-test-roster"}',
        patch_receipt_raw=_patch_raw(),
        root=evaluation_root,
        clock=lambda: SIDE_NEUTRAL_RATINGS_CAPTURED_AT,
    )


@pytest.fixture(scope="module")
def side_binding_receipt(evaluation_root: Path, pre_side_envelope: dict) -> dict:
    return binding.build_pre_side_rating_binding(
        envelope_raw=_json_raw(pre_side_envelope),
        binding_input_raw=_json_raw(_side_binding_input()),
        public_side_source_raw=_side_source_raw(),
        root=evaluation_root,
        clock=lambda: SIDE_CAPTURED_AT,
    )


@pytest.fixture(scope="module")
def side_neutral_prediction(
    evaluation_root: Path,
    side_binding_receipt: dict,
    draft_source_raw: bytes,
) -> dict:
    return side_neutral.build_side_neutral_draft_prediction(
        side_binding_raw=_json_raw(side_binding_receipt),
        draft_metadata_raw=_json_raw(_draft_metadata(draft_source_raw)),
        draft_source_payload_raw=draft_source_raw,
        root=evaluation_root,
        clock=lambda: DRAFT_CAPTURED_AT,
    )


@pytest.fixture(scope="module")
def side_neutral_bundle(
    evaluation_root: Path,
    side_neutral_prediction: dict,
    map_start_receipt: dict,
) -> dict:
    return side_bundle.build_side_neutral_capture_bundle(
        side_neutral_draft_raw=_json_raw(side_neutral_prediction),
        map_start_receipt_raw=_json_raw(map_start_receipt),
        root=evaluation_root,
    )


def _neutral_review_payload(root: Path) -> dict:
    protocol = validate_registered_side_neutral_protocol_v2(root=root)
    admission = validate_registered_side_neutral_collection_implementation(root=root)
    reviewed_at = "2026-08-02T07:40:00+00:00"
    return {
        "schema_version": neutral_review.SCHEMA_VERSION,
        "review_id": "independent-side-neutral-ledger-test-review",
        "reviewer": {
            "reviewer_id": "independent-human-ledger-test-reviewer",
            "reviewer_role": "independent-human-reviewer",
            "independent_from_implementation": True,
            "not_the_protocol_author": True,
            "conflicts_disclosed": True,
        },
        "reviewed_at_utc": reviewed_at,
        "protocol": {
            "locator": NEUTRAL_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": NEUTRAL_PROTOCOL_RAW_SHA256,
            "artifact_sha256": NEUTRAL_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": NEUTRAL_PROTOCOL_LOCKED_AT_UTC,
        },
        "reviewed_source_locks": protocol["source_locks"],
        "reviewed_admission_implementation": admission["records"],
        "findings": {
            "protocol_and_implementation_reviewed": True,
            "source_provenance_and_exact_roster_binding_reviewed": True,
            "side_selection_without_rating_refit_reviewed": True,
            "terminal_draft_and_actual_start_timing_reviewed": True,
            "duplicate_or_ambiguous_side_binding_policy_reviewed": True,
            "outcome_leakage_controls_reviewed": True,
            "no_clobber_persistence_reviewed": True,
            "model_source_boundary_stopping_evaluation_and_uncertainty_unchanged": True,
            "future_outcomes_accessed": False,
            "future_predictions_accessed": False,
            "unresolved_critical_findings": [],
        },
        "authorization": {
            "prospective_collection_authorized": True,
            "effective_at_utc": reviewed_at,
            "captures_before_effective_time_eligible": False,
            "retrospective_backfill_authorized": False,
            "outcome_opening_authorized": False,
            "rating_or_draft_authority_granted": False,
            "probability_odds_ev_or_recommendation_authorized": False,
            "betting_authorized": False,
        },
        "authority": {
            "prospective_collection_authority": True,
            "outcome_opening_authority": False,
            "model_validation_authority": False,
            "player_rating_authority": False,
            "team_rating_authority": False,
            "draft_validation_authority": False,
            "probability_authority": False,
            "odds_authority": False,
            "expected_value_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "Independent authorization for prospective outcome-free collection only, "
            "effective after this review. No retrospective evidence, outcome opening, "
            "rating, Draft, probability, odds, EV, recommendation, or betting authority."
        ),
    }


@pytest.fixture(scope="module")
def neutral_review_environment(evaluation_root: Path) -> dict[str, str]:
    path = evaluation_root / neutral_review.REVIEW_LOCATOR
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_raw(_neutral_review_payload(evaluation_root))
    path.write_bytes(raw)
    return {NEUTRAL_REVIEW_ENV: hashlib.sha256(raw).hexdigest()}


@pytest.fixture(scope="module")
def persisted_side_neutral_bundle(
    evaluation_root: Path, side_neutral_bundle: dict
) -> str:
    locator = side_bundle.bundle_locator(side_neutral_bundle)
    side_bundle.write_no_clobber(evaluation_root / locator, side_neutral_bundle)
    return locator


@pytest.fixture(scope="module")
def admitted_side_neutral_ledger(
    evaluation_root: Path,
    neutral_review_environment: dict[str, str],
    persisted_side_neutral_bundle: str,
) -> dict:
    return neutral_ledger.build_side_neutral_ledger(
        bundle_locators=[persisted_side_neutral_bundle],
        environment=neutral_review_environment,
        root=evaluation_root,
        clock=lambda: LEDGER_CREATED_AT,
    )


@pytest.fixture(scope="module")
def prediction_receipt(
    evaluation_root: Path,
    ratings_receipt: dict,
    draft_source_raw: bytes,
) -> dict:
    return ledger.build_draft_prediction_receipt(
        ratings_receipt_raw=_json_raw(ratings_receipt),
        draft_metadata_raw=_json_raw(_draft_metadata(draft_source_raw)),
        draft_source_payload_raw=draft_source_raw,
        root=evaluation_root,
        clock=lambda: DRAFT_CAPTURED_AT,
    )


def _map_start_metadata(
    source_raw: bytes, *, actual_map_start: str = ACTUAL_MAP_START
) -> dict:
    return {
        "schema_version": "scryglass:actual-map-start-capture-input:v1",
        "event_id": EVENT_ID,
        "series_id": SERIES_ID,
        "game_number": 1,
        "league": "LCS",
        "patch": "26.15",
        "actual_map_start_utc": actual_map_start,
        "source": {
            "source_id": "synthetic-actual-map-start-test-source",
            "source_url": "https://example.test/actual-map-start",
            "source_record_id": f"{EVENT_ID}-actual-map-start",
            "available_at_utc": "2026-08-08T20:31:00Z",
            "rights_status": "reviewed",
            "payload_raw_sha256": hashlib.sha256(source_raw).hexdigest(),
        },
    }


@pytest.fixture(scope="module")
def map_start_source_raw() -> bytes:
    return _json_raw(
        {
            "actualMapStartUtc": ACTUAL_MAP_START,
            "eventId": EVENT_ID,
            "kind": "actual-map-start-test-fixture",
        }
    )


@pytest.fixture(scope="module")
def map_start_receipt(evaluation_root: Path, map_start_source_raw: bytes) -> dict:
    return ledger.build_map_start_receipt(
        map_start_metadata_raw=_json_raw(
            _map_start_metadata(map_start_source_raw)
        ),
        map_start_source_payload_raw=map_start_source_raw,
        root=evaluation_root,
        clock=lambda: MAP_START_CAPTURED_AT,
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = sha256_canonical_object(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_capture_is_outcome_free_pending_and_uses_incremental_estimand(
    prediction_receipt: dict,
) -> None:
    assert prediction_receipt["result_state"] == ledger.PREDICTION_RESULT_STATE
    assert prediction_receipt["qualification"][
        "actual_map_start_authority_present"
    ] is False
    assert prediction_receipt["qualification"]["eligible_future_evidence"] is False
    assert prediction_receipt["draft_index"][
        "neutral_output_directly_outcome_calibrated"
    ] is False
    assert prediction_receipt["draft_index"]["canonical_side_a"] == "blue"
    ratings_only = prediction_receipt["evaluation_predictions"]["ratings_only"]
    combined = prediction_receipt["evaluation_predictions"]["ratings_plus_draft"]
    assert combined["logit_blue"] == pytest.approx(
        ratings_only["logit_blue"]
        + prediction_receipt["draft_index"]["scaled_logit_blue"]
    )
    assert combined["p_blue"] + combined["p_red"] == pytest.approx(1.0)
    assert all(value is False for value in prediction_receipt["authority"].values())


def test_side_neutral_adapter_uses_exact_selected_rating_without_refit(
    side_binding_receipt: dict,
    side_neutral_prediction: dict,
    prediction_receipt: dict,
) -> None:
    assert side_neutral_prediction["result_state"] == side_neutral.RESULT_STATE
    selected = side_binding_receipt["selection"]
    assert side_neutral_prediction["selected_rating_binding"] == {
        "scenario": "team1_blue",
        "rating_receipt_raw_sha256": selected[
            "selected_rating_receipt_raw_sha256"
        ],
        "rating_receipt_artifact_sha256": selected[
            "selected_rating_receipt_artifact_sha256"
        ],
        "rating_recomputed_after_side_observation": False,
    }
    child = side_neutral_prediction["terminal_draft_prediction"]["value"]
    assert child["evaluation_predictions"] == prediction_receipt[
        "evaluation_predictions"
    ]
    assert child["draft_index"] == prediction_receipt["draft_index"]
    assert side_neutral_prediction["qualification"][
        "actual_map_start_present"
    ] is False
    assert side_neutral_prediction["qualification"][
        "eligible_evaluation_map"
    ] is False
    assert all(
        value is False for value in side_neutral_prediction["authority"].values()
    )


def test_side_neutral_adapter_replays_and_rejects_resigned_child_swap(
    evaluation_root: Path, side_neutral_prediction: dict
) -> None:
    checked = side_neutral.validate_side_neutral_draft_prediction(
        json.loads(json.dumps(side_neutral_prediction, sort_keys=True)),
        root=evaluation_root,
    )
    assert checked["artifact_sha256"] == side_neutral_prediction[
        "artifact_sha256"
    ]

    forged = deepcopy(side_neutral_prediction)
    forged["selected_rating_binding"]["scenario"] = "team2_blue"
    forged["artifact_sha256"] = side_neutral._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        side_neutral.SideNeutralDraftPredictionError,
        match="selected rating binding",
    ):
        side_neutral.validate_side_neutral_draft_prediction(
            forged, root=evaluation_root
        )


def test_side_neutral_adapter_rejects_draft_not_after_side_binding(
    evaluation_root: Path,
    side_binding_receipt: dict,
    draft_source_raw: bytes,
) -> None:
    with pytest.raises(
        side_neutral.SideNeutralDraftPredictionError,
        match="after side binding",
    ):
        side_neutral.build_side_neutral_draft_prediction(
            side_binding_raw=_json_raw(side_binding_receipt),
            draft_metadata_raw=_json_raw(_draft_metadata(draft_source_raw)),
            draft_source_payload_raw=draft_source_raw,
            root=evaluation_root,
            clock=lambda: SIDE_CAPTURED_AT,
        )


def test_side_neutral_bundle_completes_timing_but_counts_zero(
    side_neutral_bundle: dict,
) -> None:
    assert side_neutral_bundle["result_state"] == side_bundle.RESULT_STATE
    assert side_neutral_bundle["timing"]["pre_side_before_side_binding"] is True
    assert side_neutral_bundle["timing"][
        "side_binding_before_terminal_draft"
    ] is True
    assert side_neutral_bundle["timing"][
        "terminal_draft_before_actual_map_start"
    ] is True
    qualification = side_neutral_bundle["qualification"]
    assert qualification["four_stage_capture_chain_complete"] is True
    assert qualification["side_binding_ambiguity_checked_across_registry"] is False
    assert qualification["side_neutral_protocol_independently_registered"] is False
    assert qualification["eligible_evaluation_map"] is False
    assert qualification["eligible_map_count_contribution"] == 0
    assert all(value is False for value in side_neutral_bundle["authority"].values())


def test_side_neutral_bundle_rejects_resigned_eligibility_forgery(
    evaluation_root: Path, side_neutral_bundle: dict
) -> None:
    forged = deepcopy(side_neutral_bundle)
    forged["qualification"]["eligible_evaluation_map"] = True
    forged["qualification"]["eligible_map_count_contribution"] = 1
    forged["artifact_sha256"] = side_bundle._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        side_bundle.SideNeutralCaptureBundleError, match="qualification"
    ):
        side_bundle.validate_side_neutral_capture_bundle(
            forged, root=evaluation_root
        )


def test_side_neutral_bundle_rejects_draft_at_actual_start(
    evaluation_root: Path,
    side_neutral_prediction: dict,
    map_start_source_raw: bytes,
) -> None:
    too_early_start = ledger.build_map_start_receipt(
        map_start_metadata_raw=_json_raw(
            _map_start_metadata(
                map_start_source_raw,
                actual_map_start=DRAFT_CAPTURED_AT.isoformat(),
            )
        ),
        map_start_source_payload_raw=map_start_source_raw,
        root=evaluation_root,
        clock=lambda: MAP_START_CAPTURED_AT,
    )
    with pytest.raises(
        side_bundle.SideNeutralCaptureBundleError,
        match="pre-side < side < Draft < actual map start",
    ):
        side_bundle.build_side_neutral_capture_bundle(
            side_neutral_draft_raw=_json_raw(side_neutral_prediction),
            map_start_receipt_raw=_json_raw(too_early_start),
            root=evaluation_root,
        )


def test_reviewed_side_neutral_ledger_admits_only_evaluation_denominator(
    admitted_side_neutral_ledger: dict,
) -> None:
    ledger_value = admitted_side_neutral_ledger
    assert ledger_value["result_state"] == neutral_ledger.RESULT_STATE
    assert ledger_value["qualification"]["eligible_map_count"] == 1
    assert ledger_value["qualification"][
        "independent_review_present_and_valid"
    ] is True
    assert ledger_value["qualification"]["outcomes_accessed"] is False
    assert ledger_value["qualification"]["opening_authority"] is False
    assert ledger_value["support"]["eligible_maps"] == 1
    assert ledger_value["support"]["support_met"] is False
    assert all(value is False for value in ledger_value["authority"].values())


def test_reviewed_side_neutral_ledger_replays_exact_bundle_bytes(
    evaluation_root: Path,
    neutral_review_environment: dict[str, str],
    admitted_side_neutral_ledger: dict,
) -> None:
    checked = neutral_ledger.validate_side_neutral_ledger(
        json.loads(json.dumps(admitted_side_neutral_ledger, sort_keys=True)),
        root=evaluation_root,
        environment=neutral_review_environment,
        as_of=LEDGER_CREATED_AT,
    )
    assert checked["artifact_sha256"] == admitted_side_neutral_ledger[
        "artifact_sha256"
    ]


def test_reviewed_side_neutral_ledger_rejects_duplicate_map_or_missing_review(
    evaluation_root: Path,
    neutral_review_environment: dict[str, str],
    persisted_side_neutral_bundle: str,
) -> None:
    with pytest.raises(neutral_ledger.SideNeutralLedgerError, match="duplicate bundle"):
        neutral_ledger.build_side_neutral_ledger(
            bundle_locators=[
                persisted_side_neutral_bundle,
                persisted_side_neutral_bundle,
            ],
            environment=neutral_review_environment,
            root=evaluation_root,
            clock=lambda: LEDGER_CREATED_AT,
        )
    with pytest.raises(neutral_review.SideNeutralProtocolReviewError, match="missing external"):
        neutral_ledger.build_side_neutral_ledger(
            bundle_locators=[persisted_side_neutral_bundle],
            environment={},
            root=evaluation_root,
            clock=lambda: LEDGER_CREATED_AT,
        )


def test_side_neutral_admission_rejects_pre_review_capture(
    side_neutral_bundle: dict,
) -> None:
    pre_side = side_neutral_bundle["timing"]["pre_side_captured_at_utc"]
    after_capture = datetime.fromisoformat(pre_side) + timedelta(seconds=1)
    with pytest.raises(
        neutral_ledger.SideNeutralLedgerError,
        match="does not follow independent review",
    ):
        neutral_ledger._entry(
            locator="data/lol/v2/evaluation/match-winner-market-v1/phase-one/side-neutral-bundles/2026-08-08/test.json",
            raw=_json_raw(side_neutral_bundle),
            bundle=side_neutral_bundle,
            review_effective=after_capture,
        )


def test_prediction_replays_exactly(
    prediction_receipt: dict, evaluation_root: Path
) -> None:
    assert (
        ledger.replay_draft_prediction_receipt(
            prediction_receipt, root=evaluation_root
        )
        == prediction_receipt
    )


def test_full_pipeline_uncertainty_refits_real_rating_and_draft_components(
    prediction_receipt: dict,
    evaluation_root: Path,
    fresh_rating_source_snapshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_result = {
        "artifact_sha256": "e" * 64,
        "phase_one_models_passed": True,
    }
    phase_registry = {
        "receipt": {
            "registry_id": "independent-phase-one-pass-bootstrap-test",
            "registered_at_utc": "2026-08-02T08:30:00+00:00",
        },
        "receipt_raw_sha256": "f" * 64,
        "phase_one_models_independently_passed": True,
    }
    monkeypatch.setattr(
        rating_refit,
        "_registered_pass",
        lambda **_kwargs: (
            phase_registry,
            b'{"phase_one_models_passed":true}\n',
            phase_result,
        ),
    )
    refit = rating_refit.build_post_validation_refit_v1(
        phase_one_result_locator=(
            "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
            "outputs/registered-pass-bootstrap-test.json"
        ),
        source_snapshot_locator=fresh_rating_source_snapshot,
        roster_receipt_raw=_roster_raw(),
        patch_receipt_raw=_patch_raw(),
        root=evaluation_root,
        environment={},
        clock=lambda: datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
    )
    refit_prepared = rating_refit.prepare_probability_replay_v1(
        refit, root=evaluation_root, environment={}
    )
    source_raw = ledger._decode_source_payload(
        prediction_receipt["input_receipts"]["draft_source_payload"],
        "draft source",
    )
    metadata = ledger._validate_draft_metadata(
        prediction_receipt["input_receipts"]["draft_metadata"]["value"],
        source_payload_raw=source_raw,
    )
    rating_probability = uncertainty._rating_target_probability(
        rating_refit_prepared=refit_prepared,
        sampled_indices=list(
            range(len(refit_prepared["input_data"].development_series))
        ),
    )
    assert 0.0 < rating_probability < 1.0
    assert rating_probability == pytest.approx(
        rating_refit.point_rating_probability_v1(refit_prepared)
    )

    draft_rows, _source = uncertainty.load_development_snapshot(evaluation_root)
    order, grouped = uncertainty._cluster_partition(draft_rows)
    calibration_count = max(20, len(order) // 10)
    train_order = order[:-calibration_count]
    calibration_order = order[-calibration_count:]
    baseline = uncertainty.pre_event_team_elo_logits(draft_rows)
    draft_logit, diagnostics = uncertainty._draft_target_scaled_logit(
        rows=draft_rows,
        baseline_logits=baseline,
        metadata=metadata,
        train_order=train_order,
        calibration_order=calibration_order,
        grouped=grouped,
        train_indices=list(range(len(train_order))),
        calibration_indices=list(range(len(calibration_order))),
    )
    assert math.isfinite(draft_logit)
    assert diagnostics["feature_count"] > 0

    phase_rows = [
        {
            "series_id": series_id,
            "blue_win": outcome,
            "ratings_plus_draft": 0.62 if outcome else 0.38,
            "ratings_only": 0.58 if outcome else 0.42,
        }
        for series_id in ("bootstrap-a", "bootstrap-b")
        for outcome in (0, 1)
    ]
    slow_prepared = {
        "rating_refit_prepared": refit_prepared,
        "target_metadata": metadata,
        "phase_one_rows": phase_rows,
        "draft_rows": draft_rows,
        "draft_baseline_logits": baseline,
        "draft_train_order": train_order,
        "draft_calibration_order": calibration_order,
        "draft_grouped": grouped,
    }
    slow_draw = uncertainty._draw_from_prepared(slow_prepared, 0)
    fast_prepared = {
        **{
            key: slow_prepared[key]
            for key in (
                "target_metadata",
                "phase_one_rows",
                "draft_rows",
                "draft_baseline_logits",
                "draft_train_order",
                "draft_calibration_order",
                "draft_grouped",
            )
        },
        "rating": {
            "draws": [
                {
                    "draw_id": 0,
                    "seed": slow_draw["seeds"]["ratings_development"],
                    "sample_digest": slow_draw["sample_digests"][
                        "ratings_development"
                    ],
                    "rating_probability_blue": slow_draw["refit"][
                        "rating_probability_blue"
                    ],
                }
            ]
        },
    }
    assert fast_uncertainty._fast_draw(fast_prepared, 0) == slow_draw


def test_capture_rejects_ratings_identity_mismatch(
    ratings_receipt: dict,
    draft_source_raw: bytes,
    evaluation_root: Path,
) -> None:
    metadata = _draft_metadata(draft_source_raw)
    metadata["red_organization_id"] = "oe:team:wrong-red-organization"
    with pytest.raises(ledger.DraftPredictionLedgerError, match="identity differ"):
        ledger.build_draft_prediction_receipt(
            ratings_receipt_raw=_json_raw(ratings_receipt),
            draft_metadata_raw=_json_raw(metadata),
            draft_source_payload_raw=draft_source_raw,
            root=evaluation_root,
            clock=lambda: DRAFT_CAPTURED_AT,
        )


def test_prediction_rejects_outcomes_forged_authority_and_bad_base64(
    prediction_receipt: dict, evaluation_root: Path
) -> None:
    leaked = deepcopy(prediction_receipt)
    leaked["event"]["winner"] = "blue"
    _resign(leaked)
    with pytest.raises(ledger.DraftPredictionLedgerError, match="outcome field"):
        ledger.validate_draft_prediction_receipt(leaked, root=evaluation_root)

    forged = deepcopy(prediction_receipt)
    forged["authority"]["betting_authority"] = True
    _resign(forged)
    with pytest.raises(ledger.DraftPredictionLedgerError, match="exceeds authority"):
        ledger.validate_draft_prediction_receipt(forged, root=evaluation_root)

    malformed = deepcopy(prediction_receipt)
    malformed["input_receipts"]["draft_source_payload"]["raw_base64"] = "%%%"
    _resign(malformed)
    with pytest.raises(ledger.DraftPredictionLedgerError, match="base64 is invalid"):
        ledger.validate_draft_prediction_receipt(malformed, root=evaluation_root)


def test_source_payloads_must_be_json_and_outcome_free(
    ratings_receipt: dict,
    draft_source_raw: bytes,
    map_start_source_raw: bytes,
    evaluation_root: Path,
) -> None:
    outcome_source = _json_raw({"eventId": EVENT_ID, "winner": "blue"})
    metadata = _draft_metadata(outcome_source)
    with pytest.raises(ledger.DraftPredictionLedgerError, match="outcome field"):
        ledger.build_draft_prediction_receipt(
            ratings_receipt_raw=_json_raw(ratings_receipt),
            draft_metadata_raw=_json_raw(metadata),
            draft_source_payload_raw=outcome_source,
            root=evaluation_root,
            clock=lambda: DRAFT_CAPTURED_AT,
        )

    invalid_json = b"provider returned non-json"
    metadata = _draft_metadata(invalid_json)
    with pytest.raises(ledger.DraftPredictionLedgerError, match="strict UTF-8 JSON"):
        ledger.build_draft_prediction_receipt(
            ratings_receipt_raw=_json_raw(ratings_receipt),
            draft_metadata_raw=_json_raw(metadata),
            draft_source_payload_raw=invalid_json,
            root=evaluation_root,
            clock=lambda: DRAFT_CAPTURED_AT,
        )

    outcome_map_start = _json_raw({"winnerTeamId": BLUE_ORG_ID})
    with pytest.raises(ledger.DraftPredictionLedgerError, match="outcome field"):
        ledger.build_map_start_receipt(
            map_start_metadata_raw=_json_raw(
                _map_start_metadata(outcome_map_start)
            ),
            map_start_source_payload_raw=outcome_map_start,
            root=evaluation_root,
            clock=lambda: MAP_START_CAPTURED_AT,
        )


def test_map_start_receipt_is_authority_only_for_actual_start(
    map_start_receipt: dict, evaluation_root: Path
) -> None:
    checked = ledger.validate_map_start_receipt(
        map_start_receipt, root=evaluation_root
    )
    assert checked["qualification"]["actual_map_start_authority_present"] is True
    assert checked["qualification"]["event_outcome_accessed"] is False
    assert all(value is False for value in checked["authority"].values())


def test_ledger_binds_receipt_bytes_and_counts_metadata_only(
    prediction_receipt: dict,
    map_start_receipt: dict,
    evaluation_root: Path,
) -> None:
    prediction_locator = (
        ledger.PREDICTION_PREFIX / "future-lcs-evaluation-map-1.json"
    ).as_posix()
    map_start_locator = (
        ledger.MAP_START_PREFIX / "future-lcs-evaluation-map-1.json"
    ).as_posix()
    ledger.write_no_clobber(
        evaluation_root / prediction_locator, prediction_receipt
    )
    ledger.write_no_clobber(
        evaluation_root / map_start_locator, map_start_receipt
    )
    value = ledger.build_prediction_ledger(
        receipts=[
            (
                prediction_locator,
                prediction_receipt,
                map_start_locator,
                map_start_receipt,
            )
        ],
        root=evaluation_root,
        clock=lambda: LEDGER_CREATED_AT,
    )
    assert value["status"] == "COLLECTING_OUTCOME_FREE_DRAFT_PREDICTIONS"
    assert value["metadata_support"]["eligible_maps"] == 1
    assert value["metadata_support"]["eligible_series"] == 1
    assert value["metadata_support"]["support_met"] is False
    assert value["outcomes_present"] is False
    assert value["opening_authority"] is False
    assert all(item is False for item in value["authority"].values())
    assert ledger.validate_prediction_ledger(value, root=evaluation_root) == value

    forged = deepcopy(value)
    forged["entries"][0]["prediction_artifact_sha256"] = "f" * 64
    _resign(forged)
    with pytest.raises(
        ledger.DraftPredictionLedgerError, match="differs from its bound receipts"
    ):
        ledger.validate_prediction_ledger(forged, root=evaluation_root)


def test_ledger_rejects_prediction_not_before_actual_map_start(
    prediction_receipt: dict,
    map_start_source_raw: bytes,
    evaluation_root: Path,
) -> None:
    start = ledger.build_map_start_receipt(
        map_start_metadata_raw=_json_raw(
            _map_start_metadata(
                map_start_source_raw,
                actual_map_start=DRAFT_CAPTURED_AT.isoformat(),
            )
        ),
        map_start_source_payload_raw=map_start_source_raw,
        root=evaluation_root,
        clock=lambda: MAP_START_CAPTURED_AT,
    )
    with pytest.raises(
        ledger.DraftPredictionLedgerError, match="before map start"
    ):
        ledger.build_prediction_ledger(
            receipts=[
                (
                    (ledger.PREDICTION_PREFIX / "bad-time.json").as_posix(),
                    prediction_receipt,
                    (ledger.MAP_START_PREFIX / "bad-time.json").as_posix(),
                    start,
                )
            ],
            root=evaluation_root,
            clock=lambda: LEDGER_CREATED_AT,
        )


def test_cli_exposes_no_user_capture_timestamp(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        ledger.main(["capture", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--captured-at" not in help_text
    assert "--timestamp" not in help_text


def test_write_no_clobber_refuses_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = {"kind": "outcome-free-test"}
    first_hash = ledger.write_no_clobber(target, payload)
    assert first_hash == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ledger.DraftPredictionLedgerError, match="overwrite"):
        ledger.write_no_clobber(target, payload)


def test_capture_readiness_is_empty_replayable_and_non_authorizing() -> None:
    repo_root = Path(".").resolve()
    payload = capture_readiness_v1.build_capture_readiness_v1(
        root=repo_root,
        clock=lambda: datetime(2026, 8, 2, 2, 0, 0, tzinfo=timezone.utc),
    )
    checked = capture_readiness_v1.validate_capture_readiness_v1(
        payload, root=repo_root
    )
    assert checked["implementation"][
        "ready_for_outcome_free_future_capture"
    ] is True
    assert checked["implementation"][
        "actual_future_prediction_evidence_present"
    ] is False
    assert checked["ledger_state"]["entries"] == 0
    assert checked["empty_ledger_template"]["entries"] == 0
    assert all(item is False for item in checked["authority"].values())


def test_capture_readiness_rejects_lock_at_future_boundary() -> None:
    with pytest.raises(
        capture_readiness_v1.DraftCaptureReadinessError,
        match="before the future boundary",
    ):
        capture_readiness_v1.build_capture_readiness_v1(
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc),
        )


def test_registered_capture_readiness_is_hash_pinned() -> None:
    checked = (
        capture_readiness_registry_v1.validate_registered_capture_readiness_v1(
            root=Path(".").resolve()
        )
    )
    assert (
        checked["artifact_sha256"]
        == capture_readiness_registry_v1.REGISTERED_CAPTURE_ARTIFACT_SHA256
    )
    assert (
        checked["locked_at_utc"]
        == capture_readiness_registry_v1.REGISTERED_CAPTURE_LOCKED_AT_UTC
    )


def _ensure_receipt(path: Path, payload: dict, writer) -> None:
    if path.exists():
        assert json.loads(path.read_text()) == payload
        return
    writer(path, payload)


def test_phase_one_collection_joins_exact_receipts_and_rebuilds_both_ledgers(
    evaluation_root: Path,
    ratings_receipt: dict,
    prediction_receipt: dict,
    map_start_receipt: dict,
) -> None:
    tail = Path(f"{EVENT_ID}.json")
    ratings_locator = (ratings_ledger.RECEIPT_PREFIX / tail).as_posix()
    draft_locator = (ledger.PREDICTION_PREFIX / tail).as_posix()
    start_locator = (ledger.MAP_START_PREFIX / tail).as_posix()
    _ensure_receipt(
        evaluation_root / ratings_locator,
        ratings_receipt,
        ratings_ledger.write_no_clobber,
    )
    _ensure_receipt(
        evaluation_root / draft_locator,
        prediction_receipt,
        ledger.write_no_clobber,
    )
    _ensure_receipt(
        evaluation_root / start_locator,
        map_start_receipt,
        ledger.write_no_clobber,
    )

    plan = phase_one.build_event_plan(
        ratings_prediction_locator=ratings_locator,
        root=evaluation_root,
        clock=lambda: datetime(2026, 8, 8, 19, 0, tzinfo=timezone.utc),
    )
    plan_locator = plan["locators"]["plan"]
    phase_one.write_no_clobber(evaluation_root / plan_locator, plan)
    bundle = phase_one.build_event_bundle(
        plan_locator=plan_locator,
        root=evaluation_root,
        clock=lambda: datetime(2026, 8, 8, 20, 33, tzinfo=timezone.utc),
    )
    bundle_locator = bundle["bundle_locator"]
    phase_one.write_no_clobber(evaluation_root / bundle_locator, bundle)
    snapshot_locator = (
        phase_one.SNAPSHOT_PREFIX / "future-lcs-evaluation-snapshot-1.json"
    ).as_posix()
    snapshot = phase_one.build_joint_ledger_snapshot(
        bundle_locators=[bundle_locator],
        snapshot_locator=snapshot_locator,
        root=evaluation_root,
        clock=lambda: datetime(2026, 8, 8, 20, 34, tzinfo=timezone.utc),
    )

    assert plan["result_state"] == phase_one.PLAN_RESULT_STATE
    assert bundle["result_state"] == phase_one.BUNDLE_RESULT_STATE
    assert bundle["timing"]["draft_before_actual_map_start"] is True
    assert bundle["receipt_bindings"]["ratings_prediction"]["raw_sha256"] == (
        prediction_receipt["input_receipts"]["ratings_prediction"]["raw_sha256"]
    )
    assert snapshot["status"] == "COLLECTING_OUTCOME_FREE_PHASE_ONE_EVIDENCE"
    assert snapshot["support"]["event_bundles"] == 1
    assert snapshot["ratings_ledger_candidate"]["metadata_support"][
        "overall_series"
    ] == 1
    assert snapshot["draft_ledger_candidate"]["metadata_support"][
        "eligible_maps"
    ] == 1
    assert snapshot["support"]["model_evaluation_passed"] is False
    assert snapshot["outcomes_accessed"] is False
    assert snapshot["opening_authority"] is False
    assert all(value is False for value in snapshot["authority"].values())
    assert (
        phase_one.validate_joint_ledger_snapshot(
            json.loads(json.dumps(snapshot)), root=evaluation_root
        )
        == snapshot
    )

    phase_one.write_no_clobber(evaluation_root / snapshot_locator, snapshot)
    repo_root = Path(".").resolve()
    completed = subprocess.run(
        [
            str(repo_root / "apps/scryglass/node_modules/.bin/tsx"),
            str(repo_root / "apps/scryglass/scripts/phaseOneDraftParity.mts"),
            str(evaluation_root),
            snapshot_locator,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    replay = json.loads(completed.stdout)
    assert replay["schema_version"] == (
        "scryglass:phase-one-draft-typescript-replay:v1"
    )
    assert replay["snapshot_artifact_sha256"] == snapshot["artifact_sha256"]
    assert len(replay["comparisons"]) == 1
    comparison = replay["comparisons"][0]
    assert comparison["event_id"] == EVENT_ID
    assert comparison["prediction_artifact_sha256"] == (
        prediction_receipt["artifact_sha256"]
    )
    assert comparison["draft_index_absolute_delta"] <= 1e-12
    assert comparison["combined_absolute_delta"] <= 1e-12


def test_phase_one_collection_rejects_outcomes_and_forged_authority(
    evaluation_root: Path,
    ratings_receipt: dict,
) -> None:
    ratings_locator = (
        ratings_ledger.RECEIPT_PREFIX / f"{EVENT_ID}-forgery-check.json"
    ).as_posix()
    _ensure_receipt(
        evaluation_root / ratings_locator,
        ratings_receipt,
        ratings_ledger.write_no_clobber,
    )
    plan = phase_one.build_event_plan(
        ratings_prediction_locator=ratings_locator,
        root=evaluation_root,
        clock=lambda: datetime(2026, 8, 8, 19, 0, tzinfo=timezone.utc),
    )
    leaked = deepcopy(plan)
    leaked["event"]["winner"] = "blue"
    leaked["artifact_sha256"] = phase_one._canonical_sha256(
        {key: value for key, value in leaked.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        phase_one.PhaseOneCollectionError, match="outcome field"
    ):
        phase_one.validate_event_plan(leaked, root=evaluation_root)

    forged = deepcopy(plan)
    forged["authority"]["betting_authority"] = True
    forged["artifact_sha256"] = phase_one._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        phase_one.PhaseOneCollectionError, match="exceeds authority"
    ):
        phase_one.validate_event_plan(forged, root=evaluation_root)


def test_phase_one_cli_exposes_no_user_timestamp(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as exc:
        phase_one.main(["plan", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--captured-at" not in help_text
    assert "--planned-at" not in help_text
    assert "--timestamp" not in help_text


def test_phase_one_rejects_symlinked_receipt_and_no_clobber(
    evaluation_root: Path,
    ratings_receipt: dict,
    tmp_path: Path,
) -> None:
    regular_locator = (
        ratings_ledger.RECEIPT_PREFIX / f"{EVENT_ID}-regular.json"
    ).as_posix()
    regular_path = evaluation_root / regular_locator
    _ensure_receipt(
        regular_path,
        ratings_receipt,
        ratings_ledger.write_no_clobber,
    )
    linked_locator = (
        ratings_ledger.RECEIPT_PREFIX / f"{EVENT_ID}-linked.json"
    ).as_posix()
    linked_path = evaluation_root / linked_locator
    linked_path.symlink_to(regular_path)
    with pytest.raises(
        phase_one.PhaseOneCollectionError, match="symlink is rejected"
    ):
        phase_one.build_event_plan(
            ratings_prediction_locator=linked_locator,
            root=evaluation_root,
            clock=lambda: datetime(2026, 8, 8, 19, 0, tzinfo=timezone.utc),
        )

    target = tmp_path / "phase-one.json"
    payload = {"kind": "outcome-free-phase-one-test"}
    first = phase_one.write_no_clobber(target, payload)
    assert first == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(
        phase_one.PhaseOneCollectionError, match="refusing to overwrite"
    ):
        phase_one.write_no_clobber(target, payload)
