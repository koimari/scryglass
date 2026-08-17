"""Focused checks for the 26.15 LCC base slice and sparse 26.16 replay."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.mechanics.atom_ledger import (
    ATOM_CATEGORIES,
    AtomLedgerConflictError,
    AtomLedgerCoverageError,
    AtomLedgerFutureDataError,
    AtomLedgerIntegrityError,
    canonical_sha256,
    load_base_snapshot,
    load_delta_event,
    replay_events,
    resolve_model_ready_snapshot,
    stable_atom_id,
)
from lol_kills.v2.mechanics.atom_ledger.schema import DELTA_SCHEMA_ID

KNOWLEDGE_CUTOFF = "2026-08-17T00:00:00Z"


@pytest.fixture(scope="module")
def base():
    return load_base_snapshot()


@pytest.fixture(scope="module")
def pilot():
    return load_delta_event()


def _atoms(snapshot):
    return {atom["atom_id"]: atom for atom in snapshot["atoms"]}


def _sign_event(event):
    event["event_hash"] = canonical_sha256(
        {key: value for key, value in event.items() if key != "event_hash"}
    )
    return event


def test_base_is_exact_for_its_complete_source_corpus_and_partial_for_the_game(base):
    assert base["patch"] == "26.15"
    assert base["authority_status"] == "exact_patch_bound_source_corpus"
    assert "accuracy_label" not in base
    assert base["authority_scope"] == "exact_to_hash_bound_lcc_26.15_six_domain_corpus"
    assert base["coverage"]["source_corpus_ingestion"] == "complete"
    assert base["coverage"]["status"] == "measured_partial"
    assert base["coverage"]["full_wiki_game_coverage"] is False
    assert base["coverage"]["missing_game_domains"]
    assert base["atom_count"] == 19_852
    assert base["domain_counts"] == {
        "abilities": 5093,
        "champions": 5372,
        "economics": 817,
        "items": 1664,
        "runes": 127,
        "stats": 6779,
    }
    assert set(base["category_counts"]) == set(ATOM_CATEGORIES)
    assert len(base["source_binding"]["file_sha256"]) == 7
    assert all(
        len(value) == 64 for value in base["source_binding"]["file_sha256"].values()
    )


def test_every_field_has_source_unit_confidence_and_missing_mask(base):
    for atom in base["atoms"]:
        assert atom["fields"]
        assert atom["missing_mask"] == {
            field_name: cell["missing"] for field_name, cell in atom["fields"].items()
        }
        for cell in atom["fields"].values():
            assert set(cell) == {
                "value",
                "source",
                "unit",
                "confidence",
                "missing",
                "authority",
            }
            assert cell["source"]
            assert cell["confidence"] == 1.0
            assert cell["authority"].endswith("patch_bound")


def test_stable_ids_support_every_required_category():
    identity = {
        "domain": "test",
        "entity": "Test Entity",
        "source_atom_id": "test.value",
        "source_locator": "Test.value",
        "source_slot": "slot:000",
    }
    atom_ids = {stable_atom_id(category, identity) for category in ATOM_CATEGORIES}
    assert len(atom_ids) == len(ATOM_CATEGORIES)
    assert all(atom_id.startswith("lolatom:v1:") for atom_id in atom_ids)


def test_behavior_and_formula_changes_preserve_identity_and_change_record_hash():
    identity = {
        "domain": "abilities",
        "entity": "Test Champion",
        "source_atom_id": "ability.damage",
        "source_locator": "TestChampion.Q.effects[0]",
        "source_slot": "slot:000",
    }
    atom_id = stable_atom_id("spell", identity)
    before = {
        "atom_id": atom_id,
        "identity": identity,
        "behavior": "old behavior",
        "formula": "10 + 0.5 AP",
    }
    after = deepcopy(before)
    after["behavior"] = "new behavior"
    after["formula"] = "20 + 0.6 AP"
    assert stable_atom_id("spell", identity) == atom_id
    assert canonical_sha256(before) != canonical_sha256(after)


def test_26_16_changes_selected_champions_and_items_and_carries_others(base, pilot):
    result, receipt = replay_events(
        base,
        [pilot],
        as_of_patch="26.16",
        knowledge_cutoff=KNOWLEDGE_CUTOFF,
    )
    before = _atoms(base)
    after = _atoms(result)
    expected_values = {
        "lolatom:v1:champion:4cac67468105dcdc7b344e2ee2602270": 33.0,
        "lolatom:v1:champion:7d883539a29a8bc37433d417d6d90284": 1.1,
        "lolatom:v1:champion:412dce467cceb3cdc2bdf7201d4d5bb9": 33.0,
        "lolatom:v1:champion:75185ba49e582a265904b7c19df532ec": 1.1,
        "lolatom:v1:champion:21a227f3f451129ce7507f6b20ca878b": 56.0,
        "lolatom:v1:item:9132a7d4ddeaba399b9fc160348194b8": 30.0,
        "lolatom:v1:item:71e01631e7825f3e2ee23e9ecec6a782": 45.0,
        "lolatom:v1:item:fae2b47c6d4767f9f1fb954614eb7842": 25.0,
        "lolatom:v1:item:86f53986659096221a0a303b0c8bef4b": 2800.0,
    }
    for atom_id, value in expected_values.items():
        assert after[atom_id]["fields"]["value:000"]["value"] == value
        assert after[atom_id]["atom_id"] == before[atom_id]["atom_id"]
        assert after[atom_id]["record_hash"] != before[atom_id]["record_hash"]
    unchanged_champion = "lolatom:v1:champion:81fd667bb42b5713daef8f764d213d06"
    unchanged_item = "lolatom:v1:item:b4eef31315795ec36d7cdf999c531ec2"
    assert after[unchanged_champion] == before[unchanged_champion]
    assert after[unchanged_item] == before[unchanged_item]
    assert receipt["changed_atom_count"] == 12
    assert receipt["unchanged_atom_count"] == 19_840
    assert (
        receipt["binary_only_unchanged_field_status"]
        == "unchanged_with_prior_authority"
    )
    assert receipt["binary_only_unchanged_field_count"] > 0
    assert receipt["authority_status"] == "partial_delta_pilot"
    assert "accuracy_label" not in result
    assert receipt["coverage"]["status"] == "partial"
    assert receipt["coverage"]["parsed_change_count"] == 12
    assert receipt["coverage"]["unparsed_or_unsupported_change_count"] == 106
    assert (
        receipt["field_status_index"]["default_status"]
        == "unchanged_with_prior_authority"
    )
    assert len(receipt["field_status_index"]["overrides"]) == 12


def test_add_and_deactivate_keep_history(base):
    prior = _atoms(base)
    removed_id = "lolatom:v1:item:b4eef31315795ec36d7cdf999c531ec2"
    identity = {
        "domain": "items",
        "entity": "test:future-item",
        "source_atom_id": "effect.test",
        "source_locator": "Future Item.effect.test",
        "source_slot": "slot:000",
    }
    added_id = stable_atom_id("item", identity)
    record = {
        "atom_id": added_id,
        "identity": identity,
        "primary_category": "item",
        "behavior": "effect",
        "categories": ["item", "effect"],
        "name": "Future Item",
        "fields": {
            "value:000": {
                "value": 10.0,
                "source": "test revision",
                "unit": "flat",
                "confidence": 1.0,
                "missing": False,
                "authority": "wiki_revision",
            }
        },
        "missing_mask": {"value:000": False},
        "evidence": ["test addition"],
        "source_record_hash": "test",
        "active": True,
    }
    event = _sign_event(
        {
            "schema_id": DELTA_SCHEMA_ID,
            "event_id": "test-add-deactivate",
            "base_patch": "26.15",
            "target_patch": "26.16",
            "effective_at": "2026-08-12T00:00:00Z",
            "previous_event_hash": None,
            "previous_snapshot_hash": base["snapshot_hash"],
            "authority_status": "complete_patch_delta",
            "coverage": {
                "status": "complete",
                "model_ready": True,
                "patch_page_candidate_change_count": 2,
                "parsed_change_count": 2,
                "unparsed_or_unsupported_change_count": 0,
                "unparsed_or_unsupported_changes": [],
            },
            "source_receipt": {
                "title": "Test",
                "revision_id": 1,
                "revision_timestamp": "2026-08-12T00:00:00Z",
                "source_url": "https://example.invalid/test",
            },
            "operations": [
                {
                    "operation_id": "test:add",
                    "op": "add",
                    "atom_id": added_id,
                    "record": record,
                },
                {
                    "operation_id": "test:deactivate",
                    "op": "deactivate",
                    "atom_id": removed_id,
                    "expected_record_hash": prior[removed_id]["record_hash"],
                },
            ],
        }
    )
    result, receipt = replay_events(
        base,
        [event],
        as_of_patch="26.16",
        knowledge_cutoff=KNOWLEDGE_CUTOFF,
    )
    after = _atoms(result)
    assert after[added_id]["active"] is True
    assert after[removed_id]["active"] is False
    assert removed_id in after
    assert receipt["atom_count"] == 19_853
    assert receipt["active_atom_count"] == 19_852


def test_replay_is_deterministic(base, pilot):
    first = replay_events(
        base, [pilot], as_of_patch="26.16", knowledge_cutoff=KNOWLEDGE_CUTOFF
    )
    second = replay_events(
        base, [pilot], as_of_patch="26.16", knowledge_cutoff=KNOWLEDGE_CUTOFF
    )
    assert first == second


def test_model_ready_resolver_rejects_incomplete_delta(base, pilot):
    with pytest.raises(AtomLedgerCoverageError, match="incomplete for model use"):
        resolve_model_ready_snapshot(
            base,
            [pilot],
            as_of_patch="26.16",
            knowledge_cutoff=KNOWLEDGE_CUTOFF,
        )


def test_double_applied_and_conflicting_deltas_fail(base, pilot):
    with pytest.raises(AtomLedgerConflictError, match="applied twice"):
        replay_events(
            base,
            [pilot, pilot],
            as_of_patch="26.16",
            knowledge_cutoff=KNOWLEDGE_CUTOFF,
        )
    conflict = deepcopy(pilot)
    conflict["operations"][0]["expected_record_hash"] = "f" * 64
    _sign_event(conflict)
    with pytest.raises(AtomLedgerConflictError, match="prior hash differs"):
        replay_events(
            base, [conflict], as_of_patch="26.16", knowledge_cutoff=KNOWLEDGE_CUTOFF
        )


def test_missing_base_and_broken_chain_fail(base, pilot, tmp_path):
    with pytest.raises(AtomLedgerIntegrityError, match="missing base snapshot"):
        load_base_snapshot(Path(tmp_path) / "absent.json.gz")
    broken = deepcopy(pilot)
    broken["previous_snapshot_hash"] = "f" * 64
    _sign_event(broken)
    with pytest.raises(AtomLedgerIntegrityError, match="breaks the chain"):
        replay_events(
            base, [broken], as_of_patch="26.16", knowledge_cutoff=KNOWLEDGE_CUTOFF
        )


def test_future_patch_and_future_source_revision_are_rejected(base, pilot):
    with pytest.raises(AtomLedgerFutureDataError, match="target patch"):
        replay_events(
            base, [pilot], as_of_patch="26.15", knowledge_cutoff=KNOWLEDGE_CUTOFF
        )
    with pytest.raises(AtomLedgerFutureDataError, match="source revision"):
        replay_events(
            base,
            [pilot],
            as_of_patch="26.16",
            knowledge_cutoff="2026-08-15T23:59:59Z",
        )
