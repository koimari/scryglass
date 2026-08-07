"""Artifact build, fail-closed, and persistence tests for L9 tier lists."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.data.competitions import CompetitionTaxonomy, CompetitionTaxonomyRow
from lol_kills.v2.draft.terminal.model import TerminalModel
from lol_kills.v2.tierlists.artifact import (
    build_tier_list_artifact,
    filter_rows,
    load_frozen_terminal_model,
    load_tier_list_artifact,
    verify_tier_list_payload,
    write_tier_list_artifact,
)
from lol_kills.v2.tierlists.appearances import AppearanceTable, league_scope, international_scope
from lol_kills.v2.tierlists.model import (
    CLAIM_CEILING,
    CROSSWALK_ARTIFACT,
    TERMINAL_MODEL_ARTIFACT,
    TierListIntegrityError,
    load_crosswalk_vocabulary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

ROWS = [
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m2", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Gnar", "event_end": "2026-07-21T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m3", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-21T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
]


def make_terminal_model(*, model_version: str | None = None, counter_logit=None, role_coefficients=None) -> TerminalModel:
    return TerminalModel(
        model_version=model_version or TERMINAL_MODEL_ARTIFACT["model_version"],
        model_as_of="2026-07-18T16:33:48Z",
        intercept=0.0,
        calibration_slope=0.8,
        calibration_intercept=0.0,
        uncertainty_logit_sd=0.12568233044362706,
        champion_role_logit=role_coefficients
        or {
            "top|Aatrox": 0.024087424395910325,
            "top|Gnar": 0.006858090014511986,
        },
        ally_synergy_logit={},
        counter_logit=counter_logit or {},
        artifact_sha256="a" * 64,
        authorizes_prediction=False,
    )


CROSSWALK = {"Aatrox": "riot:champion:266", "Gnar": "riot:champion:150"}


def build_cell(*, table=None, scope=None, model=None, crosswalk=None, as_of="2026-08-01T00:00:00Z", source_sha=None, taxonomy=None):
    appearances = (table or AppearanceTable.from_rows(ROWS)).filter(
        scope or league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1"),
        as_of=as_of,
    )
    return build_tier_list_artifact(
        scope=scope or league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1"),
        as_of=as_of,
        terminal_model=model or make_terminal_model(),
        crosswalk=crosswalk or CROSSWALK,
        appearances=appearances,
        appearance_source_sha256=source_sha or "b" * 64,
        created_at="2026-08-02T00:00:00Z",
        taxonomy=taxonomy,
    )


def test_numeric_payload_invariants() -> None:
    payload = build_cell()
    assert payload["status"] == "development_only"
    assert payload["fail_closed_status"] == "counterability_unavailable"
    assert payload["rank_eligibility"] is False
    assert payload["publication_eligible"] is False
    assert payload["claim_ceiling"] == CLAIM_CEILING
    assert payload["scope"]["league_id"] == "LEC"
    assert payload["patch_id"] == "16.14"
    assert payload["role"] == "top"
    assert payload["as_of"] == "2026-08-01T00:00:00Z"
    assert payload["created_at"] == "2026-08-02T00:00:00Z"
    assert {row["champion_name"] for row in payload["rows"]} == {"Aatrox", "Gnar"}
    for row in payload["rows"]:
        assert row["counterability"] is None  # unavailable serializes as null, never zero
        assert row["counterability_status"] == "unavailable"
        assert row["weighted_tier_value"] == pytest.approx(row["tier_value"], abs=1e-12)
        assert row["verified_appearance_count"] >= 1
    by_name = {row["champion_name"]: row for row in payload["rows"]}
    assert by_name["Aatrox"]["verified_appearance_count"] == 2
    assert by_name["Gnar"]["verified_appearance_count"] == 1
    verify_tier_list_payload(payload)


def _scan_keys(value, forbidden):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                yield key
            yield from _scan_keys(item, forbidden)
    elif isinstance(value, list):
        for item in value:
            yield from _scan_keys(item, forbidden)


def test_no_raw_win_rate_target_anywhere() -> None:
    payload = build_cell()
    # no JSON key anywhere may be a raw-win-rate value field; the estimand
    # block only carries the explicit false denial "raw_win_rate_target"
    forbidden = {"win_rate", "raw_win_rate", "wr", "wins", "losses"}
    assert list(_scan_keys(payload, forbidden)) == []
    assert payload["estimand"]["raw_win_rate_target"] is False


def test_no_zero_play_rows() -> None:
    payload = build_cell()
    assert payload["rows"]
    for row in payload["rows"]:
        assert row["verified_appearance_count"] >= 1
    assert len(payload["membership"]) == len(payload["rows"])


def test_fail_closed_empty_membership() -> None:
    table = AppearanceTable.from_rows([ROWS[2]])  # only Aatrox top
    payload = build_cell(table=table, scope=league_scope("LEC", role="mid", patch_id="16.14", competition_tier="tier1"))
    assert payload["status"] == "unavailable"
    assert payload["fail_closed_status"] == "unavailable"
    assert payload["error"]["reason"] == "no_played_champions_in_cell"
    assert payload["rows"] == [] and payload["membership"] == []
    verify_tier_list_payload(payload)


def test_fail_closed_missing_terminal_coverage() -> None:
    table = AppearanceTable.from_rows(
        ROWS + [
            {"map_id": "m9", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Yone", "event_end": "2026-07-22T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
        ]
    )
    payload = build_cell(table=table, crosswalk={**CROSSWALK, "Yone": "riot:champion:777"})
    assert payload["status"] == "unavailable"
    assert payload["error"]["reason"] == "played_champion_missing_terminal_coverage"
    assert "Yone" in payload["error"]["missing_fields"]


def test_fail_closed_unresolved_identity() -> None:
    table = AppearanceTable.from_rows(
        ROWS + [
            {"map_id": "m9", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "FutureChamp", "event_end": "2026-07-22T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
        ]
    )
    payload = build_cell(table=table)
    assert payload["status"] == "unavailable"
    assert payload["error"]["reason"] == "played_champion_unresolved_identity"


def test_fail_closed_wrong_terminal_model_version() -> None:
    payload = build_cell(model=make_terminal_model(model_version="draft-terminal-neutral-dev-v2.0.0"))
    assert payload["status"] == "unavailable"
    assert payload["error"]["reason"] == "terminal_model_version_mismatch"


def test_fail_closed_taxonomy_tier_conflict() -> None:
    taxonomy = CompetitionTaxonomy.empty()
    taxonomy = taxonomy.append(
        CompetitionTaxonomyRow(
            row_id="row:tier2-lec",
            league_id="LEC",
            competition_tier="tier2",
            structurally_globally_eligible=False,
            source_id="test",
            source_name="test",
            source_record_id="r1",
            source_snapshot_id="snap1",
            source_snapshot_row_id="snaprow1",
            source_snapshot_content_sha256="c" * 64,
            effective_from="2026-01-01T00:00:00Z",
            effective_to=None,
            internationally_connectable=False,
            qualification_rule_id="rule:test",
            precedence=10,
            observed_at="2026-01-02T00:00:00Z",
            source_updated_at="2026-01-01T00:00:00Z",
            available_at="2026-01-01T00:00:00Z",
        )
    )
    payload = build_cell(taxonomy=taxonomy)
    assert payload["status"] == "unavailable"
    assert payload["error"]["reason"].startswith("taxonomy_tier_conflict")


def test_round_trip_and_mutation_rejection(tmp_path: Path) -> None:
    payload = build_cell()
    path = tmp_path / "tierlist.json"
    digest = write_tier_list_artifact(path, payload)
    assert digest == payload["artifact_sha256"]
    loaded = load_tier_list_artifact(path)
    assert loaded["artifact_sha256"] == digest
    assert loaded == payload

    # semantic mutation is rejected
    mutated = copy.deepcopy(payload)
    mutated["rows"][0]["tier_value"] = 999.0
    with pytest.raises(TierListIntegrityError):
        verify_tier_list_payload(mutated)

    # byte-level mutation is rejected
    raw = path.read_bytes()
    path.write_bytes(raw + b"\n")
    with pytest.raises(TierListIntegrityError):
        load_tier_list_artifact(path)

    # expected-digest binding is enforced
    path.write_bytes(raw)
    with pytest.raises(TierListIntegrityError):
        load_tier_list_artifact(path, expected_sha256="d" * 64)


def test_write_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "tierlist.json"
    write_tier_list_artifact(path, build_cell())
    with pytest.raises(TierListIntegrityError, match="refusing to overwrite"):
        write_tier_list_artifact(path, build_cell())


def test_filter_rows_views() -> None:
    payload = build_cell()
    assert len(filter_rows(payload, role="top", patch="16.14")) == 2
    assert len(filter_rows(payload, region="europe", league="LEC")) == 2
    assert len(filter_rows(payload, league="LCS")) == 0
    assert len(filter_rows(payload, competition_tier="tier2")) == 0
    assert len(filter_rows(payload, played_maps_min=2)) == 1  # only Aatrox


def test_counterability_available_path_with_response_support() -> None:
    model = make_terminal_model(
        counter_logit={"top|Aatrox|Gnar": 0.12, "top|Aatrox|Renekton": -0.2, "top|Gnar|Renekton": 0.05}
    )
    payload = build_cell(model=model)
    assert payload["status"] == "development_only"
    rows = {row["champion_name"]: row for row in payload["rows"]}
    for champion in ("Aatrox", "Gnar"):
        assert rows[champion]["counterability_status"] == "available"
        assert rows[champion]["counterability"] >= 0.0
        assert rows[champion]["counterability_regret_support_size"] >= 1
    # weight stays zero without L2 validation; TV == IV
    assert payload["reference_convention"]["counterability_weight_lambda_c"] == 0.0
    for row in payload["rows"]:
        assert row["weighted_tier_value"] == pytest.approx(row["tier_value"], abs=1e-12)
    verify_tier_list_payload(payload)


def test_frozen_terminal_artifact_and_crosswalk_load() -> None:
    model = load_frozen_terminal_model(REPO_ROOT)
    assert model.model_version == TERMINAL_MODEL_ARTIFACT["model_version"]
    crosswalk = load_crosswalk_vocabulary(REPO_ROOT)
    assert crosswalk["aatrox"] == "riot:champion:266"


def test_real_end_to_end_cell() -> None:
    """Build the LEC 16.14 top cell from the real warehouse (dev-only)."""
    from lol_kills.v2.tierlists.appearances import AppearanceTable

    table = AppearanceTable.from_oe_player_games(REPO_ROOT)
    scope = league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1")
    cell = table.filter(scope, as_of="2026-08-01T00:00:00Z")
    assert cell.membership(), "LEC 16.14 top must have played champions"
    model = load_frozen_terminal_model(REPO_ROOT)
    crosswalk = load_crosswalk_vocabulary(REPO_ROOT)
    payload = build_tier_list_artifact(
        scope=scope,
        as_of="2026-08-01T00:00:00Z",
        terminal_model=model,
        crosswalk=crosswalk,
        appearances=cell,
        appearance_source_sha256=table.raw_sha256,
        appearance_source_locator=table.source_locator,
        created_at="2026-08-02T00:00:00Z",
    )
    # Yone and Twisted Fate played top on 16.14 but have no frozen coverage:
    # the cell must fail closed rather than silently drop played champions.
    assert payload["status"] == "unavailable"
    assert payload["error"]["reason"] == "played_champion_missing_terminal_coverage"
    verify_tier_list_payload(payload)


def test_generated_artifact_round_trip() -> None:
    path = REPO_ROOT / "data/lol/v2/tierlists/tierlist-lec-16.14-mid-development-v1.json"
    if not path.exists():
        pytest.skip("generated artifact not present in this worktree")
    payload = load_tier_list_artifact(path)
    assert payload["status"] == "development_only"
    assert payload["scope"]["scope_id"] == "LEC"
    assert payload["patch_id"] == "16.14"
    assert payload["role"] == "mid"
    assert payload["rank_eligibility"] is False
    assert payload["rows"]
    for row in payload["rows"]:
        assert row["verified_appearance_count"] >= 1
        assert row["counterability"] is None
