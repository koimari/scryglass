from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions import (
    representation_rank_2026_support_gate as gate,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / gate.SOURCE_LOCATOR
ARTIFACT = ROOT / gate.OUTPUT_LOCATOR

EXPECTED_MONTHS = [
    ("development", "2026-01", 342, 230, 190, 152),
    ("development", "2026-02", 461, 421, 235, 221),
    ("development", "2026-03", 223, 202, 84, 83),
    ("validation", "2026-04", 581, 515, 297, 288),
    ("validation", "2026-05", 648, 569, 244, 243),
]
EXPECTED_LEAGUES = [
    ("CBLOL", 169, 150, 78, 71),
    ("EWC", 151, 133, 63, 63),
    ("FST", 45, 41, 13, 13),
    ("LCK", 329, 284, 130, 129),
    ("LCP", 191, 158, 69, 66),
    ("LCS", 144, 126, 53, 53),
    ("LEC", 235, 195, 130, 115),
    ("LJL", 308, 264, 253, 221),
    ("LPL", 421, 363, 155, 155),
    ("PCS", 33, 32, 15, 15),
    ("TCL", 140, 112, 55, 50),
    ("VCS", 89, 79, 36, 36),
]


def _source_payload() -> dict:
    return json.loads(SOURCE.read_bytes())


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_bytes())


def _rehashed_artifact(payload: dict) -> dict:
    unsigned = deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    return gate.with_artifact_sha256(unsigned)


def test_pinned_source_and_checked_in_artifact_replay_exactly() -> None:
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == gate.SOURCE_RAW_SHA256
    source = json.loads(raw)
    assert source["schema_id"] == gate.SOURCE_SCHEMA_ID
    assert source["artifact_sha256"] == gate.SOURCE_ARTIFACT_SHA256
    assert source["aggregate_only"] is True
    assert source["final_target_loaded"] is False

    expected = gate.build_support_gate(source)
    assert _artifact() == expected
    assert gate.canonical_json_bytes(expected) + b"\n" == ARTIFACT.read_bytes()
    gate.validate_support_gate(expected, source)
    assert gate.build_support_gate(source) == expected


def test_exact_2026_overall_month_and_league_aggregates() -> None:
    artifact = _artifact()
    assert artifact["terminal_status"] == "PASS"
    assert artifact["overall"] == {
        "maps": 2255,
        "eligible_maps": 1937,
        "clusters": 1050,
        "eligible_clusters": 987,
        "component_pass": {
            "eligible_maps_at_least_four_fifths": True,
            "eligible_clusters_at_least_four_fifths": True,
        },
        "passed": True,
    }
    assert [
        (
            row["split"],
            row["calendar_month"],
            row["maps"],
            row["eligible_maps"],
            row["clusters"],
            row["eligible_clusters"],
        )
        for row in artifact["months"]
    ] == EXPECTED_MONTHS
    assert all(
        row["component_pass"]
        == {
            "eligible_maps_at_least_two_thirds": True,
            "eligible_clusters_at_least_15": True,
        }
        and row["passed"] is True
        for row in artifact["months"]
    )
    assert [
        (
            row["league"],
            row["maps"],
            row["eligible_maps"],
            row["clusters"],
            row["eligible_clusters"],
        )
        for row in artifact["leagues"]
    ] == EXPECTED_LEAGUES
    assert all(row["required"] and row["passed"] for row in artifact["leagues"])


def test_every_count_identity_recomputes_from_five_blocks() -> None:
    source = gate.project_pinned_source(_source_payload())
    rows = source["coverage_diagnostics"][len(gate.DIAGNOSTIC_ONLY_BLOCKS):]
    for row, expected in zip(rows, EXPECTED_MONTHS):
        assert (
            row["split"],
            row["calendar_month"],
            row["maps"],
            row["eligible_maps"],
            row["clusters"],
            row["eligible_clusters"],
        ) == expected
        for field in gate.COUNT_FIELDS:
            assert sum(league[field] for league in row["leagues"]) == row[field]

    artifact = _artifact()
    for field in gate.COUNT_FIELDS:
        assert sum(row[field] for row in rows) == artifact["overall"][field]
        assert sum(row[field] for row in artifact["months"]) == artifact["overall"][
            field
        ]
    for expected in EXPECTED_LEAGUES:
        name = expected[0]
        observed = next(row for row in artifact["leagues"] if row["league"] == name)
        for offset, field in enumerate(gate.COUNT_FIELDS, start=1):
            assert sum(
                league[field]
                for row in rows
                for league in row["leagues"]
                if league["league"] == name
            ) == expected[offset] == observed[field]


def test_gate_edges_use_integer_cross_products_without_rounding() -> None:
    assert gate._fraction_passed(
        eligible=4, total=5, numerator=4, denominator=5
    )
    assert not gate._fraction_passed(
        eligible=799, total=1000, numerator=4, denominator=5
    )
    assert gate._fraction_passed(
        eligible=2, total=3, numerator=2, denominator=3
    )
    assert not gate._fraction_passed(
        eligible=666_666, total=1_000_000, numerator=2, denominator=3
    )
    assert gate._fraction_passed(
        eligible=3, total=4, numerator=3, denominator=4
    )
    assert not gate._fraction_passed(
        eligible=749_999, total=1_000_000, numerator=3, denominator=4
    )


@pytest.mark.parametrize(
    ("eligible_maps", "eligible_clusters", "expected"),
    [(24, 10, True), (23, 10, False), (24, 9, True)],
)
def test_pooled_league_threshold_and_fraction_edges(
    eligible_maps: int,
    eligible_clusters: int,
    expected: bool,
) -> None:
    rows = [
        {
            "leagues": [
                {
                    "league": "EDGE",
                    "maps": 32,
                    "eligible_maps": eligible_maps,
                    "clusters": 10,
                    "eligible_clusters": eligible_clusters,
                }
            ]
        }
    ]
    result = gate._league_results(rows)[0]
    assert result["required"] is True
    assert result["passed"] is expected
    rows[0]["leagues"][0]["maps"] = 29
    result = gate._league_results(rows)[0]
    assert result["required"] is False
    assert result["passed"] is True
    assert result["component_pass"]["eligible_maps_at_least_three_fourths"] is None


@pytest.mark.parametrize("mutation", ["missing", "order", "extra_2026", "final"])
def test_exact_2026_block_completeness_and_no_final_split(mutation: str) -> None:
    source = _source_payload()
    coverage = source["coverage_diagnostics"]
    if mutation == "missing":
        coverage.pop(7)
    elif mutation == "order":
        coverage[7], coverage[8] = coverage[8], coverage[7]
    elif mutation == "extra_2026":
        extra = deepcopy(coverage[-1])
        extra["calendar_month"] = "2026-06"
        extra["month"]["calendar_month"] = "2026-06"
        coverage.append(extra)
    else:
        extra = deepcopy(coverage[-1])
        extra["split"] = "final"
        coverage.append(extra)
    with pytest.raises(gate.SupportGateError):
        gate.build_support_gate(source)


@pytest.mark.parametrize(
    "mutation",
    ["block_month", "block_league", "eligible_excess", "boolean_count"],
)
def test_count_mutations_fail_closed(mutation: str) -> None:
    source = _source_payload()
    row = source["coverage_diagnostics"][7]
    if mutation == "block_month":
        row["maps"] += 1
    elif mutation == "block_league":
        row["maps"] += 1
        row["month"]["maps"] += 1
    elif mutation == "eligible_excess":
        row["eligible_maps"] = row["maps"] + 1
        row["month"]["eligible_maps"] = row["maps"] + 1
    else:
        row["maps"] = True
    with pytest.raises(gate.SupportGateError):
        gate.build_support_gate(source)


def test_stored_coverage_pass_booleans_are_ignored() -> None:
    source = _source_payload()
    expected = gate.build_support_gate(source)
    for row in source["coverage_diagnostics"]:
        row["passed"] = not row["passed"]
    assert gate.build_support_gate(source) == expected


def test_coordinated_consistent_count_mutation_cannot_reuse_stale_claim() -> None:
    source = _source_payload()
    row = source["coverage_diagnostics"][7]
    row["maps"] += 3
    row["month"]["maps"] += 3
    row["leagues"][0]["maps"] += 3
    with pytest.raises(gate.SupportGateError, match="source projection"):
        gate.build_support_gate(source)


@pytest.mark.parametrize(
    "mutation",
    ["membership", "league_order", "extra_field", "omitted_field"],
)
def test_changed_or_nonexact_authorized_projection_fails_closed(
    mutation: str,
) -> None:
    source = _source_payload()
    row = source["coverage_diagnostics"][7]
    if mutation == "membership":
        row["membership_sha256"] = "0" * 64
        expected = "source projection"
    elif mutation == "league_order":
        row["leagues"][0], row["leagues"][1] = (
            row["leagues"][1],
            row["leagues"][0],
        )
        expected = "league order"
    elif mutation == "extra_field":
        row["y"] = [0, 1]
        expected = "source schema"
    else:
        row.pop("membership_sha256")
        expected = "source schema"
    with pytest.raises(gate.SupportGateError, match=expected):
        gate.build_support_gate(source)


def test_source_identity_and_raw_file_hash_are_pinned(tmp_path: Path) -> None:
    source = _source_payload()
    source["artifact_sha256"] = "0" * 64
    with pytest.raises(gate.SupportGateError, match="source identity"):
        gate.build_support_gate(source)

    changed_raw = bytearray(SOURCE.read_bytes())
    changed_raw[-2] = ord(" ")
    changed = tmp_path / "changed-source.json"
    changed.write_bytes(bytes(changed_raw))
    with pytest.raises(gate.SupportGateError, match="raw-file"):
        gate.load_pinned_source_projection(changed)


def test_artifact_tamper_fails_even_when_rehashed() -> None:
    source = _source_payload()
    tampered = deepcopy(_artifact())
    tampered["overall"]["eligible_maps"] -= 1
    with pytest.raises(gate.SupportGateError, match="artifact hash"):
        gate.validate_support_gate(tampered, source)
    rehashed = _rehashed_artifact(tampered)
    with pytest.raises(gate.SupportGateError, match="deterministic replay"):
        gate.validate_support_gate(rehashed, source)


def test_claim_ceiling_is_support_only() -> None:
    artifact = _artifact()
    ceiling = artifact["claim_ceiling"]
    assert ceiling["statement"].startswith("support sufficiency only")
    assert all(
        ceiling[key] is False
        for key in (
            "rank_authority",
            "model_fit_authority",
            "prediction_authority",
            "publication_authority",
            "production_authority",
            "reliability_authority",
            "sota_claim_authority",
        )
    )
    assert artifact["development_only"] is True
    assert artifact["outcome_free"] is True
    assert artifact["aggregate_only"] is True


class _ReadAudit(dict):
    def __init__(self, value: dict) -> None:
        super().__init__(value)
        self.read_keys: set[str] = set()

    def get(self, key: str, default=None):
        self.read_keys.add(key)
        if key in {
            "y",
            "target",
            "targets",
            "p0",
            "p_blue_win_m0",
            "prediction",
            "predictions",
            "fit_plan",
            "development_diagnostics",
            "validation_diagnostics",
            "target_domain_sha256",
        }:
            raise AssertionError(f"outcome or fit field was read: {key}")
        return super().get(key, default)


def test_projection_reads_no_outcome_fit_or_prediction_fields() -> None:
    source = _source_payload()
    audited = _ReadAudit(source)
    artifact = gate.build_support_gate(audited)
    assert artifact["terminal_status"] == "PASS"
    assert audited.read_keys == {
        "schema_id",
        "artifact_sha256",
        "aggregate_only",
        "final_target_loaded",
        "coverage_diagnostics",
    }
    assert (
        artifact["source_identity"]["authorized_projection_sha256"]
        == gate.SOURCE_PROJECTION_SHA256
    )
    assert (
        artifact["policy"]["authorized_source_projection"]["sha256"]
        == gate.SOURCE_PROJECTION_SHA256
    )
