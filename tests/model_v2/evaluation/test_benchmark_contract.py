"""Focused hostile and oracle tests for the fail-closed v1.4 engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lol_kills.v2.evaluation.benchmark_contract as bc


REPO_ROOT = Path(__file__).parents[3]
PACKAGE_ROOT = REPO_ROOT / bc.REAL_V1_ROOT
MAIN_CANDIDATE = "candidate:scryglass-v2"


def _read(name: str) -> dict:
    return bc.load_canonical_json(PACKAGE_ROOT / name)


def _rows(
    count: int = 50,
    *,
    score_kind: str = "LOG_LOSS",
    folds: int = 1,
    stratum_id: str = "overall",
) -> list[dict]:
    """Rows with exact v1.4 identity and a bound-probability score oracle."""

    assert count % (5 * folds) == 0
    series_ids = [f"series-{index:03d}" for index in range(count)]
    row_order = bc.stable_digest(series_ids)
    leagues = ("LEC", "LCK", "LPL", "LCS", "LCP")
    per_league_fold = count // (5 * folds)
    rows: list[dict] = []
    for index, series_id in enumerate(series_ids):
        outcome = index % 2
        calibrated_edge = 0.55 + 0.01 * (index % 5)
        candidate_probability = calibrated_edge if outcome else 1.0 - calibrated_edge
        row = {
                "series_id": series_id,
                "fold_id": f"fold-{index % folds}",
                "league_id": leagues[(index // folds) % len(leagues)],
                "candidate_id": "candidate:test",
                "baseline_id": "baseline:test",
                "output_id": "player_rating",
                "stratum_id": stratum_id,
                "score_kind": score_kind,
                "map_ids": [f"map-{index:03d}"],
                "candidate_probabilities": [candidate_probability],
                "baseline_probabilities": [0.50],
                "outcome": [outcome],
                "macro_weight": 1.0 / count,
                "registered_fold_ids": [
                    f"fold-{fold}" for fold in range(folds)
                ],
                "game_side": "BLUE" if index % 2 else "RED",
                "roster_change": "STABLE_EXACT_ROSTER",
                "international_event": "NONE",
                "draft_depth": -1,
                "exact_roster_id": f"roster-{index:03d}",
                "series_order_within_exact_roster_tournament": 1,
                "strength_source_id": "strength:pre-outcome-v1",
                "pre_outcome_candidate_strength": 0.5,
                "pre_outcome_baseline_strength": 0.5,
                bc._P: f"p-{index:03d}",
                bc._T: f"t-{index:03d}",
                bc._H: f"h-{index:03d}",
                "resolved": True,
                "input_sha256": "0" * 64,
                "candidate_prediction_sha256": "0" * 64,
                "baseline_prediction_sha256": "0" * 64,
                "row_order_sha256": row_order,
            }
        row["input_sha256"] = bc._derived_input_sha256(row)
        row["candidate_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["candidate_probabilities"]
        )
        row["baseline_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["baseline_probabilities"]
        )
        rows.append(row)
    return rows


def _with_order(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["series_id"])
    digest = bc.stable_digest([row["series_id"] for row in ordered])
    rebound = [{**row, "row_order_sha256": digest} for row in ordered]
    for row in rebound:
        row["input_sha256"] = bc._derived_input_sha256(row)
        row["candidate_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["candidate_probabilities"]
        )
        row["baseline_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["baseline_probabilities"]
        )
    return rebound


def _rebind_identities(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["input_sha256"] = bc._derived_input_sha256(row)
        row["candidate_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["candidate_probabilities"]
        )
        row["baseline_prediction_sha256"] = bc._derived_prediction_sha256(
            row, row["baseline_probabilities"]
        )
    return rows


def _macro_uneven_rows() -> list[dict]:
    """A deliberately uneven league mix to distinguish macro from raw means."""

    rows = _rows()
    extra = [{**row} for row in rows[:20]]
    for index, row in enumerate(extra):
        row["series_id"] = f"extra-{index:03d}"
        row["map_ids"] = [f"extra-map-{index:03d}"]
        row["league_id"] = "LEC"
        outcome = row["outcome"][0]
        row["candidate_probabilities"] = [0.55 if outcome else 0.45]
        row[bc._P] = f"extra-p-{index:03d}"
        row[bc._T] = f"extra-t-{index:03d}"
        row[bc._H] = f"extra-h-{index:03d}"
    rows = _with_order(rows + extra)
    weights = bc.macro_regional_series_weights(rows)
    return [
        {**row, "macro_weight": weights[row["series_id"]]}
        for row in rows
    ]


def _pair_context(pair: dict) -> SimpleNamespace:
    return SimpleNamespace(
        pair_registry={
            "families": [
                {
                    "pair_family_id": pair["pair_family_id"],
                    "pair_ids": [pair["pair_id"]],
                }
            ],
            "pairs": [pair],
        },
        candidate_registry={
            "candidates": [
                {"candidate_id": "candidate:a", "simplicity_rank": 1},
                {"candidate_id": "candidate:b", "simplicity_rank": 2},
            ],
            "slots": [],
            "families": [],
        },
    )


def _pair_record() -> dict:
    return {
        "pair_id": "pair:player:test",
        "pair_family_id": "complexity:player_rating",
        "candidate_a_id": "candidate:a",
        "candidate_b_id": "candidate:b",
        "output_id": "player_rating",
        "orientation": "candidate_a_minus_candidate_b",
        "score_kind": "LOG_LOSS",
        "status": "RESOLVED",
        "aligned_row_ids_sha256": "a" * 64,
        "bootstrap_plan_sha256": "b" * 64,
        "registered_fold_ids_sha256": "1" * 64,
        "league_ids_sha256": "2" * 64,
        "macro_weights_sha256": "c" * 64,
        "outcomes_sha256": "d" * 64,
        "pth_assignments_sha256": "3" * 64,
        "critical_selectors_sha256": "4" * 64,
        "cluster_assignments_sha256": "e" * 64,
        "candidate_a_prediction_rows_sha256": "f" * 64,
        "candidate_b_prediction_rows_sha256": "0" * 64,
        "input_rows_sha256": "1" * 64,
    }


def _bound_pair_evidence(
    prepared: bc._PreparedWCR,
    pair: dict,
) -> dict:
    """Build the complete, independently re-derived pair identity surface."""

    pair["candidate_a_id"] = prepared.candidate_id
    pair["candidate_b_id"] = prepared.baseline_id
    pair["output_id"] = prepared.output_id
    rows = prepared.rows
    derived = {
        "aligned_row_ids_sha256": bc.stable_digest(
            list(prepared.canonical_series_ids)
        ),
        "difference_rows_sha256": bc.stable_digest(
            [
                {
                    "series_id": series_id,
                    "candidate_a_minus_candidate_b": float(difference),
                }
                for series_id, difference in zip(
                    prepared.canonical_series_ids,
                    prepared.differences,
                )
            ]
        ),
        "registered_fold_ids_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    "fold_id": row["fold_id"],
                    "registered_fold_ids": row["registered_fold_ids"],
                }
                for row in rows
            ]
        ),
        "league_ids_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    "league_id": row["league_id"],
                }
                for row in rows
            ]
        ),
        "macro_weights_sha256": bc.stable_digest(
            [
                {"series_id": series_id, "macro_weight": float(weight)}
                for series_id, weight in zip(
                    prepared.canonical_series_ids,
                    prepared.weights,
                )
            ]
        ),
        "outcomes_sha256": bc.stable_digest(
            [
                {"series_id": row["series_id"], "outcome": row["outcome"]}
                for row in rows
            ]
        ),
        "pth_assignments_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    bc._P: row[bc._P],
                    bc._T: row[bc._T],
                    bc._H: row[bc._H],
                }
                for row in rows
            ]
        ),
        "critical_selectors_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    "game_side": row["game_side"],
                    "roster_change": row["roster_change"],
                    "international_event": row["international_event"],
                    "draft_depth": row["draft_depth"],
                }
                for row in rows
            ]
        ),
        "cluster_assignments_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    bc._P: row[bc._P],
                    bc._T: row[bc._T],
                    bc._H: row[bc._H],
                    "P_T": bc.canonical_intersection_id(
                        [row[bc._P], row[bc._T]]
                    ),
                    "P_H": bc.canonical_intersection_id(
                        [row[bc._P], row[bc._H]]
                    ),
                    "T_H": bc.canonical_intersection_id(
                        [row[bc._T], row[bc._H]]
                    ),
                    "P_T_H": bc.canonical_intersection_id(
                        [row[bc._P], row[bc._T], row[bc._H]]
                    ),
                }
                for row in rows
            ]
        ),
        "candidate_a_prediction_rows_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    "prediction_sha256": row["candidate_prediction_sha256"],
                }
                for row in rows
            ]
        ),
        "candidate_b_prediction_rows_sha256": bc.stable_digest(
            [
                {
                    "series_id": row["series_id"],
                    "prediction_sha256": row["baseline_prediction_sha256"],
                }
                for row in rows
            ]
        ),
        "input_rows_sha256": bc._derived_input_rows_sha256(rows),
        "analysis_rows_sha256": prepared.analysis_rows_sha256,
        "bootstrap_plan_sha256": bc._registered_pair_wcr_plan_sha256(),
    }
    pair.update(derived)
    return {
        "schema_version": "g0-pair-evidence-v1.4",
        "pair_id": pair["pair_id"],
        "pair_family_id": pair["pair_family_id"],
        "candidate_a_id": pair["candidate_a_id"],
        "candidate_b_id": pair["candidate_b_id"],
        "output_id": pair["output_id"],
        "orientation": pair["orientation"],
        "score_kind": pair["score_kind"],
        "wcr_execution_id": bc._WCR_EXECUTION_ID,
        "source_code_raw_sha256": bc.raw_digest(
            (REPO_ROOT / "lol_kills/v2/evaluation/benchmark_contract.py").read_bytes()
        ),
        "rows": [dict(row) for row in rows],
        **derived,
    }


def _run(
    prepared: bc._PreparedWCR,
    *,
    theta0: float = 0.0,
    attempts: int = 31,
) -> bc.WCRBootclusterResult:
    return bc._run_wcr_bootcluster_core(
        prepared,
        active_dimensions=bc._PRIMARY_ACTIVE,
        bootcluster_dimension=bc._P,
        theta0=theta0,
        attempts=attempts,
        invert_endpoint=True,
    )


def test_v14_package_and_exact_complete_candidate_inventory() -> None:
    digests = bc.validate_real_v1(PACKAGE_ROOT, repo_root=REPO_ROOT)
    assert len(digests) == 12
    baselines = _read(bc.BASELINE_REGISTRY)
    registry = _read(bc.CANDIDATE_REGISTRY)
    pairs = _read(bc.PAIR_REGISTRY)
    assert registry["schema_version"] == "g0-candidate-registry-v1.4"
    assert len(baselines["baselines"]) == 61
    assert len(registry["candidates"]) == 2
    assert len(registry["families"]) == 6
    assert len(registry["slots"]) == 2182
    assert sum(
        slot["family_id"].startswith("primary:")
        for slot in registry["slots"]
    ) == 122
    assert sum(
        slot["family_id"].startswith("harm:")
        for slot in registry["slots"]
    ) == 1816
    assert sum(
        slot["family_id"].startswith("secondary:")
        for slot in registry["slots"]
    ) == 244
    assert {family["kind"] for family in registry["families"]} == {
        "PRIMARY_HOLM",
        "HARM_HOLM",
        "SECONDARY_HOLM",
    }
    assert len(pairs["families"]) == 1
    assert len(pairs["pairs"]) == 4


def test_g1_human_authority_handoff_binds_and_reopens_exact_artifact() -> None:
    registry = _read(bc.CANDIDATE_REGISTRY)
    handoff = registry["g1_unified_authority_handoff"]
    binding = handoff["authority_artifact"]
    raw = (REPO_ROOT / binding["relative_path"]).read_bytes()
    payload = bc.load_canonical_json(REPO_ROOT / binding["relative_path"])

    assert handoff["reviewer_identity"] == "KOI_MARI"
    assert handoff["approval_scope"] == "private_retrospective_oe_target_v1"
    assert handoff["approved_actions"] == ["model_fit", "rank_selection"]
    assert binding == {
        "relative_path": (
            "data/lol/v2/models/draft-interactions/"
            "oe-private-target-authority-2026-07-29.json"
        ),
        "raw_sha256": (
            "b1d0a6e37abb9a74dee8689dc19ab54d"
            "30fd15516bd4ee454906a075d8f20788"
        ),
    }
    assert bc.raw_digest(raw) == binding["raw_sha256"]
    assert payload["reviewer_identity"] == handoff["reviewer_identity"]
    assert payload["approval_scope"] == handoff["approval_scope"]
    assert payload["approved_actions"] == handoff["approved_actions"]
    assert payload["final_temporal_holdout_sealed"] is True
    bc.validate_candidate_registry(
        registry,
        _read(bc.BASELINE_REGISTRY),
        repo_root=REPO_ROOT,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reviewer_identity", "FABRICATED"),
        ("approval_scope", "broadened_scope"),
        ("approved_actions", ["model_fit", "rank_selection", "publication"]),
        ("authority_artifact", {
            "relative_path": (
                "data/lol/v2/models/draft-interactions/"
                "oe-private-target-authority-2026-07-29.json"
            ),
            "raw_sha256": "f" * 64,
        }),
    ),
)
def test_g1_handoff_mutation_rejects_before_authority_artifact_read(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    registry = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    registry["g1_unified_authority_handoff"][field] = value
    monkeypatch.setattr(
        bc,
        "_validate_g1_human_authority_artifact",
        lambda *_args, **_kwargs: pytest.fail(
            "mutated handoff must reject before artifact read"
        ),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_candidate_registry(
            registry,
            _read(bc.BASELINE_REGISTRY),
            repo_root=REPO_ROOT,
        )
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reviewer_identity", "FABRICATED"),
        ("approval_scope", "broadened_scope"),
        ("approved_actions", ["model_fit", "publication"]),
        ("final_temporal_holdout_sealed", False),
    ),
)
def test_g1_authority_artifact_semantic_drift_rejects_after_raw_reopen(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    handoff = deepcopy(bc._G1_CANDIDATE_AUTHORITY_HANDOFF)
    payload = bc.load_canonical_json(
        REPO_ROOT / handoff["authority_artifact"]["relative_path"]
    )
    payload[field] = value
    raw = bc.canonical_json(payload).encode("utf-8")
    handoff["authority_artifact"]["raw_sha256"] = bc.raw_digest(raw)
    monkeypatch.setattr(
        bc,
        "_read_regular_under_root",
        lambda *_args, **_kwargs: raw,
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._validate_g1_human_authority_artifact(handoff, REPO_ROOT)
    assert error.value.code == "G1_HUMAN_AUTHORITY_ARTIFACT_MISMATCH"


def test_typed_unresolved_authority_remains_blocked() -> None:
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.compute_registered_holm(
            MAIN_CANDIDATE,
            f"primary:{MAIN_CANDIDATE}:player_rating",
        )
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_resolved_candidate_is_rejected_before_slot_evidence_or_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_id = "primary:candidate:scryglass-v2"
    slot = {"slot_id": "slot:test", "family_id": family_id, "candidate_id": MAIN_CANDIDATE, "baseline_id": "rating.classical_elo", "output_id": "player_rating", "stratum_id": "overall", "decision_kind": "SUPERIORITY", "status": "TYPED_UNRESOLVED"}
    context = SimpleNamespace(
        candidate_registry={"candidates": [{"candidate_id": MAIN_CANDIDATE, "status": "RESOLVED"}], "families": [{"family_id": family_id, "candidate_id": MAIN_CANDIDATE, "kind": "PRIMARY_HOLM", "slot_ids": ["slot:test"]}], "slots": [slot]},
        baseline_registry={"baselines": [{"id": "rating.classical_elo", "status": "EXECUTABLE_PREBOUND"}]},
    )
    monkeypatch.setattr(
        bc,
        "_read_registry_bound_json",
        lambda *_args, **_kwargs: pytest.fail("evidence must not be read"),
    )
    monkeypatch.setattr(
        bc,
        "_run_wcr_bootcluster_core",
        lambda *_args, **_kwargs: pytest.fail("inference must not run"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._compute_registered_holm_at(context, MAIN_CANDIDATE, family_id, attempts=11)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


@pytest.mark.parametrize("mutation", ("family", "slot"))
def test_candidate_registry_omitted_family_or_slot_fails_closed(
    mutation: str,
) -> None:
    registry = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    if mutation == "family":
        registry["families"].pop()
    else:
        registry["slots"].pop()
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_candidate_registry(registry, _read(bc.BASELINE_REGISTRY))
    assert error.value.code in {
        "FAMILY_DERIVATION_MISMATCH",
        "CANDIDATE_SLOT_MISSING",
    }


def test_unavailable_baseline_cannot_be_forged_into_a_resolved_slot() -> None:
    registry = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    registry.pop("opening_candidate_id", None)
    slot = registry["slots"][0]
    slot.pop("unavailable")
    slot["status"] = "RESOLVED"
    slot["evidence"] = {
        "relative_path": "nope.json",
        "raw_sha256": "0" * 64,
        "semantic_sha256": "0" * 64,
    }
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_candidate_registry(registry, _read(bc.BASELINE_REGISTRY))
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_baseline_acceptance_cannot_claim_prebound_while_all_are_unavailable() -> None:
    registry = deepcopy(_read(bc.BASELINE_REGISTRY))
    registry["acceptance"]["source_dependent_execution_bindings_status"] = (
        "PREBOUND_EXECUTABLES_REGISTERED"
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_baseline_registry(registry)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_executable_baseline_rejects_before_adapter_or_fixture_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = deepcopy(_read(bc.BASELINE_REGISTRY))
    entry = registry["baselines"][0]
    entry["status"] = "EXECUTABLE_PREBOUND"
    entry.pop("unavailable")
    entry["execution"] = {
        "entry_point": "fabricated:run",
        "execution_authority": {"receipt_sha256": "f" * 64},
    }
    monkeypatch.setattr(
        bc,
        "_read_regular_under_root",
        lambda *_args, **_kwargs: pytest.fail("adapter must not be read"),
    )
    monkeypatch.setattr(
        bc,
        "_invoke_bound_fixture",
        lambda *_args, **_kwargs: pytest.fail("fixture must not run"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_baseline_registry(registry, repo_root=REPO_ROOT)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_patch_child_inventory_derives_children_and_rejects_mutation() -> None:
    baseline = _read(bc.BASELINE_REGISTRY)
    _, children = bc._expected_candidate_inventory(
        baseline, held_out_patches=("14.1", "14.2")
    )
    assert all(
        "patch:each-held-out-major-minor" not in slot_id
        for slot_id in children
    )
    assert any(":patch:14.1" in slot_id for slot_id in children)
    registry = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    registry["held_out_patch_inventory"] = {
        "status": "RESOLVED",
        "patches": ["14.1"],
        "patches_sha256": "0" * 64,
    }
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_candidate_registry(registry, baseline)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_typed_unresolved_patch_inventory_preserves_template_and_blocks() -> None:
    registry = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    bc.validate_candidate_registry(registry, _read(bc.BASELINE_REGISTRY))
    registry["held_out_patch_inventory"]["unavailable"].pop("claim_effect")
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc.validate_candidate_registry(registry, _read(bc.BASELINE_REGISTRY))
    assert error.value.code == "CANDIDATE_SLOT_UNRESOLVED"


def test_global_pair_family_is_exactly_four_outputs() -> None:
    pairs = _read(bc.PAIR_REGISTRY)
    bc.validate_pair_registry(pairs, _read(bc.CANDIDATE_REGISTRY))
    assert pairs["families"] == [
        {
            "pair_family_id": "complexity:global",
            "kind": "SIMULTANEOUS_COMPLEXITY",
            "pair_ids": sorted(pair["pair_id"] for pair in pairs["pairs"]),
        }
    ]


@pytest.mark.parametrize(
    "transition",
    (
        "baseline",
        "candidate",
        "slot",
        "secondary_equal_strength",
        "secondary_transfer",
        "held_out_patch",
        "pair",
    ),
)
def test_every_source_bound_registry_transition_rejects_before_read(
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    baseline = deepcopy(_read(bc.BASELINE_REGISTRY))
    candidate = deepcopy(_read(bc.CANDIDATE_REGISTRY))
    pair = deepcopy(_read(bc.PAIR_REGISTRY))
    monkeypatch.setattr(
        bc,
        "_read_regular_under_root",
        lambda *_args, **_kwargs: pytest.fail("source bytes must not be read"),
    )
    monkeypatch.setattr(
        bc,
        "_read_registry_bound_json",
        lambda *_args, **_kwargs: pytest.fail("evidence must not be read"),
    )
    if transition == "baseline":
        baseline["baselines"][0]["status"] = "EXECUTABLE_PREBOUND"
        validate = lambda: bc.validate_baseline_registry(
            baseline,
            repo_root=REPO_ROOT,
        )
    elif transition == "candidate":
        candidate["candidates"][0]["status"] = "RESOLVED"
        validate = lambda: bc.validate_candidate_registry(candidate, baseline)
    elif transition == "slot":
        candidate["slots"][0]["status"] = "RESOLVED"
        validate = lambda: bc.validate_candidate_registry(candidate, baseline)
    elif transition == "secondary_equal_strength":
        candidate["secondary_authority"][
            "equal_strength_draft_increment"
        ]["status"] = "RESOLVED"
        validate = lambda: bc.validate_candidate_registry(candidate, baseline)
    elif transition == "secondary_transfer":
        candidate["secondary_authority"]["transfer_new_roster"][
            "status"
        ] = "RESOLVED"
        validate = lambda: bc.validate_candidate_registry(candidate, baseline)
    elif transition == "held_out_patch":
        candidate["held_out_patch_inventory"]["status"] = "RESOLVED"
        validate = lambda: bc.validate_candidate_registry(candidate, baseline)
    else:
        pair["pairs"][0]["status"] = "RESOLVED"
        validate = lambda: bc.validate_pair_registry(pair, candidate)
    with pytest.raises(bc.BenchmarkContractError) as error:
        validate()
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_v14_schemas_make_every_source_bound_status_unsatisfiable() -> None:
    baseline_schema = _read(bc.BASELINE_SCHEMA)
    candidate_schema = _read(bc.CANDIDATE_REGISTRY_SCHEMA)
    pair_schema = _read(bc.PAIR_REGISTRY_SCHEMA)

    assert (
        baseline_schema["properties"]["baselines"]["items"]["oneOf"][1][
            "not"
        ]
        == {}
    )
    assert (
        candidate_schema["properties"]["held_out_patch_inventory"]["oneOf"][
            1
        ]["properties"]["status"]["not"]
        == {}
    )
    assert (
        candidate_schema["properties"]["secondary_authority"]["properties"][
            "equal_strength_draft_increment"
        ]["oneOf"][1]["properties"]["status"]["not"]
        == {}
    )
    assert (
        candidate_schema["properties"]["secondary_authority"]["properties"][
            "transfer_new_roster"
        ]["properties"]["status"]["const"]
        == "TYPED_UNRESOLVED"
    )
    assert (
        candidate_schema["properties"]["candidates"]["items"]["oneOf"][1][
            "properties"
        ]["status"]["not"]
        == {}
    )
    assert (
        candidate_schema["properties"]["slots"]["items"]["oneOf"][1][
            "properties"
        ]["status"]["not"]
        == {}
    )
    assert (
        pair_schema["properties"]["pairs"]["items"]["oneOf"][1][
            "properties"
        ]["status"]["not"]
        == {}
    )


def test_wcr_derives_log_loss_from_probabilities_and_outcomes() -> None:
    prepared = bc._prepare_wcr_rows(_rows())
    expected = -math.log(0.55) + math.log(0.50)
    assert prepared.differences[0] == pytest.approx(expected)
    assert np.all(prepared.differences < 0.0)
    result = _run(prepared)
    assert result.threshold == 0.0
    assert math.isfinite(result.unadjusted_one_sided_95_upper_bound)


def test_wcr_derives_brier_from_probabilities_and_outcomes() -> None:
    prepared = bc._prepare_wcr_rows(_rows(score_kind="BRIER"))
    assert prepared.differences[0] == pytest.approx(0.45**2 - 0.50**2)
    assert np.all(prepared.differences < 0.0)


def test_wcr_rejects_opaque_forecast_labels_after_probability_drift() -> None:
    rows = _rows()
    rows[0]["candidate_probabilities"] = [0.61]
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == "FORECAST_IDENTITY_MISMATCH"


def test_wcr_rejects_opaque_common_input_label_after_outcome_drift() -> None:
    rows = _rows()
    rows[0]["outcome"] = [1 - rows[0]["outcome"][0]]
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == "FORECAST_IDENTITY_MISMATCH"


def test_fabricated_pair_evidence_rejects_before_evidence_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair_record()
    pair["evidence"] = {}
    context = SimpleNamespace(repo_root=REPO_ROOT)
    monkeypatch.setattr(
        bc,
        "_read_registry_bound_json",
        lambda *_args, **_kwargs: pytest.fail("pair evidence must not be read"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._validated_pair_evidence(context, pair)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_primary_forecast_cross_bind_rejects_two_primary_vectors_and_third_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_rows, second_rows, pair_rows = _rows(), _rows(), _rows()
    for rows, candidate, baseline in (
        (first_rows, "candidate:a", "base:one"),
        (second_rows, "candidate:a", "base:two"),
        (pair_rows, "candidate:a", "candidate:b"),
    ):
        for row in rows:
            row["candidate_id"], row["baseline_id"] = candidate, baseline
        _rebind_identities(rows)
    second_rows[0]["candidate_probabilities"] = [0.63]
    pair_rows[0]["candidate_probabilities"] = [0.67]
    _rebind_identities(second_rows)
    _rebind_identities(pair_rows)
    prepared = [
        bc._prepare_wcr_rows(rows)
        for rows in (first_rows, second_rows, pair_rows)
    ]
    primary_slots = [
        {"slot_id": f"p:{index}", "candidate_id": "candidate:a", "output_id": "player_rating", "status": "RESOLVED", "family_id": "primary:a"}
        for index in range(2)
    ]
    pair = _pair_record()
    pair.update(
        candidate_a_id="candidate:a", candidate_b_id="candidate:b",
        input_rows_sha256=bc._derived_input_rows_sha256(pair_rows),
    )
    context = SimpleNamespace(
        candidate_registry={"slots": primary_slots, "families": [{"family_id": "primary:a", "kind": "PRIMARY_HOLM"}], "candidates": [{"candidate_id": "candidate:a", "simplicity_rank": 1}, {"candidate_id": "candidate:b", "simplicity_rank": 2}]},
        pair_registry={"families": [{"pair_family_id": pair["pair_family_id"], "pair_ids": [pair["pair_id"]]}], "pairs": [pair]},
    )
    evidence = iter(prepared[:2])
    monkeypatch.setattr(bc, "_holm_evidence", lambda *_: next(evidence))
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared[2])
    monkeypatch.setattr(bc, "_candidate_gate_status_at", lambda *_args, **_kwargs: "PASS")
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._compute_registered_pairwise_intervals_at(
            context, pair["pair_family_id"], attempts=11
        )
    assert error.value.code == "PAIR_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    "stratum_id",
    (
        "benefit:brier",
        "benefit:transfer_new_roster",
        "benefit:equal_strength_draft_increment",
    ),
)
def test_resolved_secondary_evidence_rejects_before_primary_evidence_read(
    monkeypatch: pytest.MonkeyPatch,
    stratum_id: str,
) -> None:
    slot = {
        "slot_id": f"slot:test:{stratum_id}",
        "stratum_id": stratum_id,
    }
    monkeypatch.setattr(
        bc,
        "_holm_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "primary evidence must not be read"
        ),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._validate_secondary_evidence(
            SimpleNamespace(candidate_registry={"slots": []}),
            slot,
            {"fabricated": "receipt"},
            SimpleNamespace(),
        )
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_resolved_candidate_provenance_rejects_before_binding_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "candidate_id": "candidate:test",
        "status": "RESOLVED",
        "execution_authority": {"receipt_sha256": "f" * 64},
    }
    monkeypatch.setattr(
        bc,
        "_read_regular_under_root",
        lambda *_args, **_kwargs: pytest.fail(
            "candidate provenance must not be read"
        ),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._validate_resolved_candidate_provenance(
            SimpleNamespace(package_root=PACKAGE_ROOT),
            candidate,
        )
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_opening_rejects_before_permit_parse_or_ledger_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        package_root=PACKAGE_ROOT,
        candidate_registry={},
        verified=SimpleNamespace(preflight_sha256="c" * 64, manifest_contract_set_sha256="d" * 64, evidence_generator_identity="generator", permit_ledger_relative_path="ledger"),
    )
    monkeypatch.setattr(
        bc,
        "_parse_json_bytes",
        lambda *_args, **_kwargs: pytest.fail("permit must not be parsed"),
    )
    monkeypatch.setattr(
        bc,
        "_AtomicFileExactOnceStore",
        lambda *_args, **_kwargs: pytest.fail("ledger must not be opened"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._consume_bound_opening_permit_at(context, b"permit")
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_authority_bundle_id_must_equal_contract_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = {"authority_id": "self-consistent-but-wrong", "contract_set_sha256": "a" * 64}
    contract = {"label_access": {"boundary_id": "scryglass:authority-derived-preflight:v1.4"}}
    values = iter((bundle, {}, contract))
    monkeypatch.setattr(bc, "validate_real_v1", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bc, "_read_regular_under_root", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(bc, "_parse_json_bytes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bc, "_schema_validate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bc, "_manifest_bound_json", lambda *_args, **_kwargs: (b"{}", next(values)))
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._verify_authoritative_preflight_at(PACKAGE_ROOT, REPO_ROOT)
    assert error.value.code == "AUTHORITY_BUNDLE_MISMATCH"

def test_wcr_requires_both_outcomes_per_inferential_cell() -> None:
    rows = _rows()
    for row in rows:
        row["outcome"] = [1]
        row["candidate_probabilities"] = [0.60]
    _rebind_identities(rows)
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == "OUTCOME_SUPPORT_INVALID"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("candidate_probabilities", [0.0], "PROBABILITY_INVALID"),
        ("map_ids", [], "PROBABILITY_INVALID"),
        ("row_order_sha256", "0" * 64, "FAMILY_DERIVATION_MISMATCH"),
    ],
)
def test_wcr_probability_map_and_order_boundaries_fail_closed(
    field: str,
    value: object,
    code: str,
) -> None:
    rows = _rows()
    rows[0][field] = value
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == code


def test_wcr_requires_exact_registered_folds_and_removal_preserves_them() -> None:
    rows = _rows(folds=5)
    rows[0]["registered_fold_ids"] = ["fold-0"]
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == "FAMILY_DERIVATION_MISMATCH"

    rows = _rows(folds=5)
    reduced = [row for row in rows if row["fold_id"] != "fold-0"]
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(_with_order(reduced))
    assert error.value.code == "FAMILY_DERIVATION_MISMATCH"


def test_wcr_internal_macro_weights_reject_caller_supplied_drift() -> None:
    rows = _rows()
    rows[0]["macro_weight"] *= 2.0
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(rows)
    assert error.value.code == "MACRO_WEIGHT_MISMATCH"


def test_wcr_low_cluster_support_blocks_before_inference() -> None:
    rows = _rows()
    for row in rows:
        row[bc._P] = "one-cluster"
    _rebind_identities(rows)
    prepared = bc._prepare_wcr_rows(rows)
    with pytest.raises(bc.BenchmarkContractError) as error:
        _run(prepared)
    assert error.value.code == "unavailable_dependence_support"


def test_registered_nonzero_theta0_is_used_by_the_wcr_test() -> None:
    prepared = bc._prepare_wcr_rows(_rows())
    result = _run(prepared, theta0=0.017)
    assert result.threshold == pytest.approx(0.017)


def test_unadjusted_endpoint_and_type7_diagnostics_are_named_honestly() -> None:
    names = {field.name for field in fields(bc.WCRBootclusterResult)}
    assert "unadjusted_one_sided_95_upper_bound" in names
    assert "type7_linearized_upper_bound_diagnostic" in names
    assert "type7_bootstrap_t_q05_diagnostic" in names
    assert "inverted_upper_bound" not in names
    assert "holm_endpoint_agree" not in {
        field.name for field in fields(bc.HolmDecision)
    }


def test_legacy_wcr_helpers_cannot_silently_default_to_zero_theta() -> None:
    for helper in (
        bc._run_wcr_bootcluster,
        bc._run_wcr_suite,
        bc._run_wcr_consensus,
    ):
        assert inspect.signature(helper).parameters["theta0"].default is inspect.Parameter.empty


def test_holm_local_alpha_and_unadjusted_endpoint_are_separate_gates() -> None:
    selected = [
        {"slot_id": "a", "decision_kind": "SUPERIORITY"},
        {"slot_id": "b", "decision_kind": "SUPERIORITY"},
    ]
    runs = [
        SimpleNamespace(
            variant_id="full|active=P,T|bootcluster=P",
            sample_variant="full",
            active_dimensions=bc._PRIMARY_ACTIVE,
            bootcluster_dimension=bc._P,
            result=SimpleNamespace(
                lower_tail_p=0.06,
                unadjusted_one_sided_95_upper_bound=-0.01,
            ),
        ),
        SimpleNamespace(
            variant_id="full|active=P,T|bootcluster=P",
            sample_variant="full",
            active_dimensions=bc._PRIMARY_ACTIVE,
            bootcluster_dimension=bc._P,
            result=SimpleNamespace(
                lower_tail_p=0.001,
                unadjusted_one_sided_95_upper_bound=0.01,
            ),
        ),
    ]
    report = bc._holm_variant_decisions(
        family_kind="PRIMARY_HOLM",
        selected=selected,
        thresholds=[0.0, 0.0],
        runs=runs,
    )
    by_slot = {decision.slot_id: decision for decision in report.decisions}
    assert by_slot["a"].holm_reject is False
    assert by_slot["a"].unadjusted_endpoint_pass is True
    assert by_slot["a"].passes is False
    assert by_slot["b"].holm_reject is True
    assert by_slot["b"].unadjusted_endpoint_pass is False
    assert by_slot["b"].passes is False
    assert by_slot["a"].local_alpha == pytest.approx(0.05)
    assert by_slot["b"].local_alpha == pytest.approx(0.025)


def test_holm_boundary_p_values_use_the_registered_inclusive_step_down() -> None:
    assert bc._holm_rejections([0.025, 0.05]) == (True, True)


def test_each_of_twenty_holm_variants_reorders_from_its_own_p_values() -> None:
    selected = [
        {"slot_id": "a", "decision_kind": "SUPERIORITY"},
        {"slot_id": "b", "decision_kind": "SUPERIORITY"},
    ]
    ranks: list[int] = []
    for index in range(20):
        p_a, p_b = (0.001, 0.02) if index % 2 == 0 else (0.02, 0.001)
        variant_id = f"variant-{index}"
        runs = [
            SimpleNamespace(
                variant_id=variant_id,
                sample_variant="full",
                active_dimensions=bc._PRIMARY_ACTIVE,
                bootcluster_dimension=bc._P,
                result=SimpleNamespace(
                    lower_tail_p=p_a,
                    unadjusted_one_sided_95_upper_bound=-0.1,
                ),
            ),
            SimpleNamespace(
                variant_id=variant_id,
                sample_variant="full",
                active_dimensions=bc._PRIMARY_ACTIVE,
                bootcluster_dimension=bc._P,
                result=SimpleNamespace(
                    lower_tail_p=p_b,
                    unadjusted_one_sided_95_upper_bound=-0.1,
                ),
            ),
        ]
        report = bc._holm_variant_decisions(
            family_kind="PRIMARY_HOLM",
            selected=selected,
            thresholds=[0.0, 0.0],
            runs=runs,
        )
        ranks.append(
            next(item.holm_rank for item in report.decisions if item.slot_id == "a")
        )
    assert ranks == [1 if index % 2 == 0 else 2 for index in range(20)]


def test_secondary_benefit_cannot_swap_passing_slots_between_sensitivities() -> None:
    variants: list[bc.HolmVariantReport] = []
    for index in range(20):
        decisions = tuple(
            bc.HolmDecision(
                slot_id=slot_id,
                raw_one_sided_p=0.001,
                holm_adjusted_p=0.001,
                holm_rank=1,
                local_alpha=0.05,
                registered_threshold=0.0,
                unadjusted_one_sided_95_upper_bound=-0.1,
                endpoint_label="UNADJUSTED_ONE_SIDED_95_PERCENT_WCR_INVERTED_UPPER_BOUND",
                holm_reject=passes,
                unadjusted_endpoint_pass=passes,
                passes=passes,
            )
            for slot_id, passes in (
                ("benefit:a", index < 10),
                ("benefit:b", index >= 10),
            )
        )
        variants.append(
            bc.HolmVariantReport(
                variant_id=f"variant-{index}",
                sample_variant="full",
                active_dimensions=bc._PRIMARY_ACTIVE,
                bootcluster_dimension=bc._P,
                decisions=decisions,
                all_members_pass=False,
                family_gate_pass=True,
            )
        )
    assert not bc._holm_family_gate_across_variants(
        "SECONDARY_HOLM",
        variants,
    )


def test_holm_replays_twenty_independent_variant_inventory() -> None:
    prepared = bc._prepare_wcr_rows(_rows())
    runs = bc._registered_wcr_variants(
        prepared,
        theta0=0.0,
        attempts=11,
        invert_endpoint=False,
    )
    assert len(runs) == len({run.variant_id for run in runs}) == 20
    assert {run.sample_variant for run in runs} == {
        "full",
        f"leave_largest:{bc._P}",
        f"leave_largest:{bc._T}",
        f"leave_largest:{bc._H}",
    }


def test_leave_largest_removal_that_loses_outcomes_blocks() -> None:
    rows = _rows(100)
    for index, row in enumerate(rows):
        row[bc._P] = "dominant" if row["outcome"] == [0] else f"p-{index}"
    _rebind_identities(rows)
    prepared = bc._prepare_wcr_rows(rows)
    dominant, _ = bc.select_largest_cluster(prepared.rows, bc._P)
    reduced = [row for row in prepared.rows if row[bc._P] != dominant]
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._prepare_wcr_rows(bc._reweight_reduced_rows(reduced))
    assert error.value.code == "OUTCOME_SUPPORT_INVALID"


def test_wcr_rows_are_canonical_and_candidate_free_for_stream_identity() -> None:
    rows = _rows()
    first = _run(bc._prepare_wcr_rows(rows), attempts=31)
    reordered = _run(bc._prepare_wcr_rows(list(reversed(rows))), attempts=31)
    assert first.bootstrap_t_sha256 == reordered.bootstrap_t_sha256


def test_pair_center_is_macro_chronological_not_plain_row_mean() -> None:
    prepared = bc._prepare_wcr_rows(_macro_uneven_rows())
    macro_center = float(np.sum(prepared.weights * prepared.differences))
    plain_center = float(np.mean(prepared.differences))
    assert macro_center != pytest.approx(plain_center)
    runs = bc._pair_wcr_runs(prepared, attempts=11)
    assert len(runs) == 20
    assert all(run.result.threshold == pytest.approx(macro_center) for run in runs)


def test_pair_leave_largest_runs_recenter_each_reduced_sample() -> None:
    rows = _macro_uneven_rows()
    rows[0][bc._P] = "largest-p"
    rows[1][bc._P] = "largest-p"
    rows[0]["candidate_probabilities"] = [0.49]
    rows[1]["candidate_probabilities"] = [0.51]
    _rebind_identities(rows)
    prepared = bc._prepare_wcr_rows(rows)
    runs = bc._pair_wcr_runs(prepared, attempts=11)
    thresholds = {
        run.sample_variant: run.result.threshold
        for run in runs
    }
    assert len(thresholds) == 4
    assert len({round(value, 12) for value in thresholds.values()}) > 1
    assert all(
        run.result.threshold == pytest.approx(run.result.point_estimate)
        for run in runs
    )


def test_pair_interval_authority_requires_both_candidate_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair_record()
    context = _pair_context(pair)
    prepared = bc._prepare_wcr_rows(_rows())
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared)
    monkeypatch.setattr(
        bc,
        "_candidate_gate_status_at",
        lambda _context, candidate_id, **_kwargs: (
            "PASS" if candidate_id == "candidate:a" else "BLOCKED_FAMILY_DECISION"
        ),
    )
    report = bc._compute_registered_pairwise_intervals_at(
        context,
        pair["pair_family_id"],
        attempts=11,
    )
    assert report.selection_status == "BLOCKED_CANDIDATE_FAMILIES"
    assert report.intervals == ()
    assert report.candidate_gate_status == (
        ("candidate:a", "PASS"),
        ("candidate:b", "BLOCKED_FAMILY_DECISION"),
    )


def test_pair_forecasts_must_cross_bind_to_independent_candidate_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair_record()
    context = _pair_context(pair)
    prepared = bc._prepare_wcr_rows(_rows())
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared)
    monkeypatch.setattr(bc, "_candidate_gate_status_at", lambda *_args, **_kwargs: "PASS")
    monkeypatch.setattr(
        bc,
        "_candidate_primary_prediction_digest_at",
        lambda *_args, **_kwargs: "9" * 64,
    )
    monkeypatch.setattr(
        bc,
        "_candidate_primary_input_digest_at",
        lambda *_args, **_kwargs: pair["input_rows_sha256"],
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._compute_registered_pairwise_intervals_at(
            context,
            pair["pair_family_id"],
            attempts=11,
        )
    assert error.value.code == "PAIR_IDENTITY_MISMATCH"


def test_pair_sensitivity_disagreement_returns_no_winner_remand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair_record()
    context = _pair_context(pair)
    prepared = bc._prepare_wcr_rows(_rows())
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared)
    monkeypatch.setattr(bc, "_candidate_gate_status_at", lambda *_args, **_kwargs: "PASS")
    monkeypatch.setattr(
        bc,
        "_candidate_primary_prediction_digest_at",
        lambda _context, candidate_id, _output: (
            pair["candidate_a_prediction_rows_sha256"]
            if candidate_id == "candidate:a"
            else pair["candidate_b_prediction_rows_sha256"]
        ),
    )
    monkeypatch.setattr(
        bc,
        "_candidate_primary_input_digest_at",
        lambda *_args, **_kwargs: pair["input_rows_sha256"],
    )

    runs: list[bc._RegisteredSlotRun] = []
    for index in range(20):
        sample = (
            "full"
            if index < 5
            else f"leave_largest:{(bc._P, bc._T, bc._H)[(index - 5) // 5]}"
        )
        active = bc._PRIMARY_ACTIVE if index % 5 < 2 else bc._PATCH_ACTIVE
        bootcluster = active[index % len(active)]
        point = -1.0 if index == 0 else 1.0 if index == 1 else 0.0
        runs.append(
            bc._RegisteredSlotRun(
                sample_variant=sample,
                active_dimensions=active,
                bootcluster_dimension=bootcluster,
                removed_dimension=None if sample == "full" else sample.rsplit(":", 1)[1],
                removed_cluster_id=None if sample == "full" else "largest",
                result=SimpleNamespace(
                    bootstrap_t=(1.0,) * 11,
                    point_estimate=point,
                    standard_error=0.1,
                ),
            )
        )
    monkeypatch.setattr(bc, "_pair_wcr_runs", lambda *_args, **_kwargs: tuple(runs))
    report = bc._compute_registered_pairwise_intervals_at(
        context,
        pair["pair_family_id"],
        attempts=11,
    )
    assert len(report.sensitivities) == 20
    assert report.sensitivity_conclusions_agree is False
    assert report.selection_status == "NO_WINNER_REMAND"
    assert report.winner_candidate_id is None


def test_pair_max_t_uses_aligned_replicates_across_every_family_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _pair_record(), _pair_record()
    first["pair_id"], second["pair_id"] = "pair:one", "pair:two"
    family_id = first["pair_family_id"]
    context = SimpleNamespace(
        pair_registry={
            "families": [
                {"pair_family_id": family_id, "pair_ids": ["pair:one", "pair:two"]}
            ],
            "pairs": [first, second],
        },
        candidate_registry={
            "candidates": [
                {"candidate_id": "candidate:a", "simplicity_rank": 1},
                {"candidate_id": "candidate:b", "simplicity_rank": 2},
            ],
            "slots": [],
            "families": [],
        },
    )
    prepared = bc._prepare_wcr_rows(_rows())
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared)
    monkeypatch.setattr(bc, "_candidate_gate_status_at", lambda *_args, **_kwargs: "PASS")
    monkeypatch.setattr(
        bc,
        "_candidate_primary_prediction_digest_at",
        lambda _context, candidate_id, _output: (
            "f" * 64 if candidate_id == "candidate:a" else "0" * 64
        ),
    )
    monkeypatch.setattr(
        bc,
        "_candidate_primary_input_digest_at",
        lambda *_args, **_kwargs: first["input_rows_sha256"],
    )

    def runs(scale: float) -> tuple[bc._RegisteredSlotRun, ...]:
        result: list[bc._RegisteredSlotRun] = []
        for index in range(20):
            sample = (
                "full"
                if index < 5
                else f"leave_largest:{(bc._P, bc._T, bc._H)[(index - 5) // 5]}"
            )
            active = bc._PRIMARY_ACTIVE if index % 5 < 2 else bc._PATCH_ACTIVE
            result.append(
                bc._RegisteredSlotRun(
                    sample_variant=sample,
                    active_dimensions=active,
                    bootcluster_dimension=active[index % len(active)],
                    removed_dimension=None if sample == "full" else sample.rsplit(":", 1)[1],
                    removed_cluster_id=None if sample == "full" else "largest",
                    result=SimpleNamespace(
                        bootstrap_t=(scale,) * 11,
                        point_estimate=0.0,
                        standard_error=0.1,
                    ),
                )
            )
        return tuple(result)

    pair_runs = iter((runs(1.0), runs(2.0)))
    monkeypatch.setattr(
        bc,
        "_pair_wcr_runs",
        lambda *_args, **_kwargs: next(pair_runs),
    )
    report = bc._compute_registered_pairwise_intervals_at(
        context,
        family_id,
        attempts=11,
    )
    assert len(report.sensitivities) == 20
    assert all(len(item.intervals) == 2 for item in report.sensitivities)
    assert all(item.critical_value == pytest.approx(2.0) for item in report.sensitivities)


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("aligned_row_ids_sha256", "PAIR_ROWS_UNALIGNED"),
        ("bootstrap_plan_sha256", "PAIR_BOOTSTRAP_PLAN_MISMATCH"),
        ("macro_weights_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("registered_fold_ids_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("league_ids_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("outcomes_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("pth_assignments_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("critical_selectors_sha256", "PAIR_IDENTITY_MISMATCH"),
        ("cluster_assignments_sha256", "PAIR_IDENTITY_MISMATCH"),
    ],
)
def test_unit_pair_family_rejects_every_aligned_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    code: str,
) -> None:
    """Unit scope: isolate family cross-alignment after per-pair validation."""

    first, second = _pair_record(), _pair_record()
    first["pair_id"], second["pair_id"] = "pair:one", "pair:two"
    second[field] = "9" * 64
    family_id = first["pair_family_id"]
    context = SimpleNamespace(
        pair_registry={
            "families": [
                {"pair_family_id": family_id, "pair_ids": ["pair:one", "pair:two"]}
            ],
            "pairs": [first, second],
        },
        candidate_registry={"candidates": [], "slots": [], "families": []},
    )
    prepared = bc._prepare_wcr_rows(_rows())
    monkeypatch.setattr(bc, "_candidate_record", lambda *_: {})
    monkeypatch.setattr(bc, "_validated_pair_evidence", lambda *_: prepared)
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._compute_registered_pairwise_intervals_at(
            context,
            family_id,
            attempts=11,
        )
    assert error.value.code == code


@pytest.mark.parametrize(
    "field",
    [
        "aligned_row_ids_sha256",
        "difference_rows_sha256",
        "registered_fold_ids_sha256",
        "league_ids_sha256",
        "macro_weights_sha256",
        "outcomes_sha256",
        "pth_assignments_sha256",
        "critical_selectors_sha256",
        "cluster_assignments_sha256",
        "bootstrap_plan_sha256",
        "candidate_a_prediction_rows_sha256",
        "candidate_b_prediction_rows_sha256",
        "analysis_rows_sha256",
    ],
)
def test_fabricated_pair_identity_is_rejected_before_any_evidence_read(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    pair = _pair_record()
    pair["evidence"] = {}
    pair[field] = "9" * 64
    context = SimpleNamespace(repo_root=REPO_ROOT)
    monkeypatch.setattr(
        bc,
        "_read_registry_bound_json",
        lambda *_args, **_kwargs: pytest.fail("pair evidence must not be read"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._validated_pair_evidence(context, pair)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"


def test_holm_evidence_rejects_before_any_evidence_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot = {
        "slot_id": "slot:fabricated",
        "status": "RESOLVED",
        "evidence": {"relative_path": "fabricated.json"},
    }
    monkeypatch.setattr(
        bc,
        "_read_registry_bound_json",
        lambda *_args, **_kwargs: pytest.fail("Holm evidence must not be read"),
    )
    with pytest.raises(bc.BenchmarkContractError) as error:
        bc._holm_evidence(SimpleNamespace(repo_root=REPO_ROOT), slot)
    assert error.value.code == "G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED"
