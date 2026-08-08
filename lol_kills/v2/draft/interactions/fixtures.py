"""Synthetic development fixtures for L6 neutral draft interactions."""

from __future__ import annotations

from typing import Any

from .types import CANONICAL_ROLES, DraftCompositionRow, normalize_side


SEED_TAG = "scryglass-l6-draft-interactions-synthetic-seed-v1"


def reveal_synthetic_seed() -> str:
    return SEED_TAG


def _rows_payloads() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "l6-synth-01",
            "league_id": "LEC",
            "patch_id": "26.14",
            "source_id": "synth-01",
            "source_patch_pool": "26.x",
            "label": 1,
            "side_a": {
                "top": "riot:champion:115",
                "jungle": "riot:champion:101",
                "mid": "riot:champion:161",
                "bot": "riot:champion:266",
                "support": "riot:champion:114",
            },
            "side_b": {
                "top": "riot:champion:99911",
                "jungle": "riot:champion:99912",
                "mid": "riot:champion:99913",
                "bot": "riot:champion:99914",
                "support": "riot:champion:99915",
            },
        },
        {
            "row_id": "l6-synth-02",
            "league_id": "LEC",
            "patch_id": "26.14",
            "source_id": "synth-01",
            "source_patch_pool": "26.x",
            "label": 0,
            "side_a": {
                "top": "riot:champion:99911",
                "jungle": "riot:champion:114",
                "mid": "riot:champion:115",
                "bot": "riot:champion:101",
                "support": "riot:champion:161",
            },
            "side_b": {
                "top": "riot:champion:266",
                "jungle": "riot:champion:99913",
                "mid": "riot:champion:99914",
                "bot": "riot:champion:99915",
                "support": "riot:champion:99912",
            },
        },
        {
            "row_id": "l6-synth-03",
            "league_id": "LCS",
            "patch_id": "26.14",
            "source_id": "synth-02",
            "source_patch_pool": "26.x",
            "label": 1,
            "side_a": {
                "top": "riot:champion:161",
                "jungle": "riot:champion:266",
                "mid": "riot:champion:114",
                "bot": "riot:champion:99911",
                "support": "riot:champion:99912",
            },
            "side_b": {
                "top": "riot:champion:115",
                "jungle": "riot:champion:101",
                "mid": "riot:champion:99913",
                "bot": "riot:champion:99914",
                "support": "riot:champion:99915",
            },
        },
        {
            "row_id": "l6-synth-04",
            "league_id": "LEC",
            "patch_id": "26.15",
            "source_id": "synth-03",
            "source_patch_pool": "26.x",
            "label": 0,
            "side_a": {
                "top": "riot:champion:115",
                "jungle": "riot:champion:101",
                "mid": "riot:champion:99911",
                "bot": "riot:champion:266",
                "support": "riot:champion:161",
            },
            "side_b": {
                "top": "riot:champion:99912",
                "jungle": "riot:champion:99913",
                "mid": "riot:champion:99914",
                "bot": "riot:champion:99915",
                "support": "riot:champion:114",
            },
        },
        {
            "row_id": "l6-synth-05",
            "league_id": "LCS",
            "patch_id": "26.15",
            "source_id": "synth-04",
            "source_patch_pool": "26.x",
            "label": 1,
            "side_a": {
                "top": "riot:champion:266",
                "jungle": "riot:champion:115",
                "mid": "riot:champion:99911",
                "bot": "riot:champion:99912",
                "support": "riot:champion:101",
            },
            "side_b": {
                "top": "riot:champion:99913",
                "jungle": "riot:champion:99914",
                "mid": "riot:champion:99915",
                "bot": "riot:champion:114",
                "support": "riot:champion:161",
            },
        },
        {
            "row_id": "l6-synth-06",
            "league_id": "LEC",
            "patch_id": "26.15",
            "source_id": "synth-05",
            "source_patch_pool": "26.x",
            "label": 0,
            "side_a": {
                "top": "riot:champion:99911",
                "jungle": "riot:champion:115",
                "mid": "riot:champion:114",
                "bot": "riot:champion:266",
                "support": "riot:champion:161",
            },
            "side_b": {
                "top": "riot:champion:101",
                "jungle": "riot:champion:99912",
                "mid": "riot:champion:99913",
                "bot": "riot:champion:99914",
                "support": "riot:champion:99915",
            },
        },
        {
            "row_id": "l6-synth-07",
            "league_id": "LCS",
            "patch_id": "26.16",
            "source_id": "synth-06",
            "source_patch_pool": "26.x",
            "label": 1,
            "side_a": {
                "top": "riot:champion:99915",
                "jungle": "riot:champion:115",
                "mid": "riot:champion:101",
                "bot": "riot:champion:161",
                "support": "riot:champion:266",
            },
            "side_b": {
                "top": "riot:champion:99914",
                "jungle": "riot:champion:114",
                "mid": "riot:champion:99911",
                "bot": "riot:champion:99912",
                "support": "riot:champion:99913",
            },
        },
        {
            "row_id": "l6-synth-08",
            "league_id": "LCS",
            "patch_id": "26.16",
            "source_id": "synth-06",
            "source_patch_pool": "26.x",
            "label": 0,
            "side_a": {
                "top": "riot:champion:99911",
                "jungle": "riot:champion:99914",
                "mid": "riot:champion:115",
                "bot": "riot:champion:266",
                "support": "riot:champion:101",
            },
            "side_b": {
                "top": "riot:champion:99913",
                "jungle": "riot:champion:99915",
                "mid": "riot:champion:114",
                "bot": "riot:champion:161",
                "support": "riot:champion:99912",
            },
        },
    ]


def load_synthetic_rows() -> list[DraftCompositionRow]:
    return [
        DraftCompositionRow.from_payload(row)
        for row in _rows_payloads_with_scopes()
    ]


def _rows_payloads_with_scopes() -> list[dict[str, Any]]:
    rows = _rows_payloads()
    for row_index, label in enumerate((0, 1), start=9):
        offset = (row_index - 9) * 10
        champions = [
            f"riot:champion:{88001 + offset + index}" for index in range(10)
        ]
        rows.append(
            {
                "row_id": f"l6-synth-{row_index:02d}",
                "league_id": "LEC" if label == 0 else "LCS",
                "patch_id": "26.16",
                "source_id": f"synth-{row_index - 2:02d}",
                "source_patch_pool": "26.x",
                "label": label,
                "side_a": dict(zip(CANONICAL_ROLES, champions[:5])),
                "side_b": dict(zip(CANONICAL_ROLES, champions[5:])),
            }
        )
    for row in rows:
        row["competition_scope_id"] = (
            "emea" if row["league_id"] == "LEC" else "americas"
        )
    return rows


def load_synthetic_fixture() -> dict[str, Any]:
    return {
        "artifact_id": "draft-interactions-synthetic-fixtures-l6-v1",
        "kind": "draft-interactions-l6-synthetic-rows",
        "seed": SEED_TAG,
        "source_scope": "synthetic",
        "rows": _rows_payloads_with_scopes(),
    }


def load_synthetic_row_payloads_for_validation() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _rows_payloads():
        normalized_side_a = normalize_side(row["side_a"])
        normalized_side_b = normalize_side(row["side_b"])
        out.append(
            {
                "row_id": row["row_id"],
                "league_id": row["league_id"],
                "patch_id": row["patch_id"],
                "source_id": row["source_id"],
                "side_a": {role: champ for role, champ in normalized_side_a},
                "side_b": {role: champ for role, champ in normalized_side_b},
                "canonical_roles": list(CANONICAL_ROLES),
                "canonical_roles_signature": ",".join(role for role, _ in normalized_side_a),
                "role_hash": ",".join(
                    f"{role}:{champ}" for role, champ in normalized_side_a + normalized_side_b
                ),
            }
        )
    return out
