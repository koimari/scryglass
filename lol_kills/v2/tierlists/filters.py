"""User-facing tier-list filters over the scope index + canonical cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.v2.data.common import ROLES

from .schema import (
    COMPETITION_TIERS,
    INTERNATIONAL_SCOPES,
    REGIONS,
    TierListError,
    validate_patch,
    validate_role,
)

DEFAULT_INDEX_PATH = Path("data/lol/v2/tierlists/index-v1.json")
TIER_BUCKETS = ("Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D")


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class TierListIndex:
    """Validated index + merged cell rows with filter views."""

    def __init__(self, index_payload: Mapping[str, Any], root: Path) -> None:
        if not isinstance(index_payload, Mapping):
            raise TierListError("tier-list index must be an object")
        submitted = index_payload.get("artifact_sha256")
        if not isinstance(submitted, str) or len(submitted) != 64:
            raise TierListError("index artifact_sha256 is invalid")
        unsigned = {k: v for k, v in index_payload.items() if k != "artifact_sha256"}
        if _raw_sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()) != submitted:
            raise TierListError("index artifact_sha256 does not match canonical payload")
        self._index = dict(index_payload)
        self._root = root
        self._cells: dict[str, dict[str, Any]] = {}
        for cell in index_payload["cells"]:
            locator = cell["locator"]
            raw = (root / locator).read_bytes()
            if _raw_sha256(raw) != cell["raw_sha256"]:
                raise TierListError(f"cell {locator} bytes changed since index build")
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("status") != "development_only":
                continue
            self._cells[cell["artifact_id"]] = payload

    @classmethod
    def load(cls, root: Path, index_path: Path = DEFAULT_INDEX_PATH) -> "TierListIndex":
        try:
            payload = json.loads((root / index_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TierListError(f"cannot load tier-list index at {index_path}: {exc}") from exc
        return cls(payload, root)

    @property
    def generated_at(self) -> str:
        return self._index["generated_at"]

    @property
    def cells(self) -> list[dict[str, Any]]:
        return list(self._index["cells"])

    def available_options(self) -> dict[str, list[str]]:
        cells = [c for c in self._index["cells"] if c["status"] == "development_only"]
        return {
            "regions": sorted(REGIONS),
            "leagues": sorted({c["league"] for c in cells if c["league"]}),
            "event_kinds": sorted({c["event_kind"] for c in cells if c["event_kind"]}),
            "competition_tiers": sorted({c["competition_tier"] for c in cells if c["competition_tier"]}),
            "roles": list(ROLES),
            "patches": sorted({c["patch_id"] for c in cells}),
        }

    def merged_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for meta in self._index["cells"]:
            payload = self._cells.get(meta["artifact_id"])
            if payload is None:
                continue
            role = meta["role"]
            cell_rows = sorted(
                payload["rows"],
                key=lambda row: row["tier_value"],
                reverse=True,
            )
            total = len(cell_rows)
            for rank, row in enumerate(cell_rows, start=1):
                quantile = (rank - 1) / max(1, total)
                fallback_buckets = ("A", "B", "C", "D")
                bucket = row.get(
                    "tier_bucket",
                    fallback_buckets[min(len(fallback_buckets) - 1, int(quantile * len(fallback_buckets)))],
                )
                rows.append(
                    {
                        "scope_id": meta["artifact_id"],
                        "league": meta.get("league"),
                        "event_kind": meta.get("event_kind"),
                        "competition_tier": meta.get("competition_tier"),
                        "role": role,
                        "patch": meta["patch_id"],
                        "champion": row["champion_name"],
                        "champion_id": row["champion_id"],
                        "tier_value_pp": row["tier_value"],
                        "rank": rank,
                        "tier_bucket": bucket,
                        "played_maps": row["verified_appearance_count"],
                        "counterability_status": row.get("counterability_status", "unavailable"),
                        "cell_status": meta["status"],
                    }
                )
        return rows

    def filter_rows(
        self,
        *,
        region: str | None = None,
        league: str | None = None,
        international: str | None = None,
        competition_tier: str | None = None,
        role: str | None = None,
        patch: str | None = None,
        played_maps_min: int = 1,
    ) -> list[dict[str, Any]]:
        if region is not None and region not in REGIONS:
            raise TierListError(f"unknown region: {region}")
        if league is not None and league not in self._index["options"]["leagues"]:
            raise TierListError(f"unknown league: {league}")
        allowed_events: tuple[str, ...] | None = None
        if international is not None:
            if international == "international":
                allowed_events = tuple(e.lower() for e in INTERNATIONAL_SCOPES)
            elif international in INTERNATIONAL_SCOPES or international in tuple(e.lower() for e in INTERNATIONAL_SCOPES):
                allowed_events = (international.lower(),)
            else:
                raise TierListError(f"unknown international scope: {international}")
        if competition_tier is not None and competition_tier not in COMPETITION_TIERS + ("international",):
            raise TierListError(f"unknown competition tier: {competition_tier}")
        if role is not None:
            validate_role(role)
        if patch is not None:
            validate_patch(patch)
        out = []
        for row in self.merged_rows():
            if region == "international":
                if not row["event_kind"]:
                    continue
            elif region is not None:
                if row["league"] not in REGIONS[region]:
                    continue
            if league is not None and row["league"] != league:
                continue
            if allowed_events is not None and row["event_kind"] not in allowed_events:
                continue
            if competition_tier is not None and row["competition_tier"] != competition_tier:
                continue
            if role is not None and row["role"] != role:
                continue
            if patch is not None and row["patch"] != patch:
                continue
            if row["played_maps"] < played_maps_min:
                continue
            out.append(row)
        return out
