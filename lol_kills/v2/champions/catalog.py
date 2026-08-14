"""Champion ontology catalog loader and archetype-prior builder."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .paths import (
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH,
)
from .schema import (
    DIMENSION_LABEL_ORDER,
    REQUIRED_DIMENSIONS,
    DIMENSION_LABELS,
    ROLES,
    validate_dimension_map,
    validate_uncertainty_map,
)
from ..patch_identity import client_patch, public_patch

__all__ = [
    "ChampionOntologyError",
    "ChampionOntology",
    "load_champion_ontology",
    "canonical_serialization",
    "canonical_sha256",
]


class ChampionOntologyError(ValueError):
    """Raised when champion ontology inputs are not valid."""


STABLE_ID_RE = re.compile(r"^riot:champion:[0-9]+$")
REVIEW_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,63})?$")
PATCH_RE = re.compile(r"^[0-9]+\.[0-9]+$")
REPO_PATH_RE = re.compile(r"^[A-Za-z0-9._/^-]+$")
SOURCE_RELATIVE_RE = re.compile(r"^data/lol/v2/champions/.+")

REVIEW_EFFECT_RULE_ID = "review_effective_prior_v1"
FUTURE_PATCH_FALLBACK_RULE_ID = "future_patch_fallback_v1"
DIMENSION_REVIEW_FACTORS: dict[str, dict[str, float]] = {
    "accepted": {"value_weight": 1.0, "uncertainty_multiplier": 1.0},
    "disputed": {"value_weight": 0.65, "uncertainty_multiplier": 1.35},
    "unreviewed": {"value_weight": 0.35, "uncertainty_multiplier": 1.75},
}
FUTURE_PATCH_RESIDUAL_SIGMA_WIDEN = 0.5
FUTURE_PATCH_UNCERTAINTY_MULTIPLIER = 1.5
EMPIRICAL_REQUIRED_FIELDS = ("champion_id", "patch_id", "role", "league_id", "residual", "verified_appearance_count")


def canonical_serialization(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_serialization(payload).encode("utf-8")).hexdigest()


def _parse_patch(patch_id: str) -> tuple[int, int]:
    if not PATCH_RE.match(patch_id):
        raise ChampionOntologyError(f"invalid patch id: {patch_id}")
    major, minor = patch_id.split(".")
    return int(major), int(minor)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ChampionOntologyError(f"missing file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as err:
        raise ChampionOntologyError(f"invalid JSON in {path}: {err}") from err


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ChampionOntologyError(
                    f"invalid jsonl in {path}:{line_no}: {err}"
                ) from err
            if not isinstance(row, dict):
                raise ChampionOntologyError(
                    f"invalid review row in {path}:{line_no}: expected object"
                )
            rows.append(row)
    return rows


def _load_json_optional(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    return _load_json(path), str(path)


def _repo_root() -> Path:
    return DEFAULT_ONTOLOGY_PATH.resolve().parents[4]


def _to_repo_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return str(resolved.relative_to(resolved_root).as_posix())


def _validate_repository_locator(locator: str) -> str:
    if not isinstance(locator, str) or not locator.strip():
        raise ChampionOntologyError("locator must be a non-empty string")
    if locator.startswith("/"):
        raise ChampionOntologyError("repository locator must be relative, not absolute")
    if locator.startswith("~"):
        raise ChampionOntologyError("repository locator must be relative")
    if ".." in Path(locator).parts:
        raise ChampionOntologyError("repository locator must not contain path traversal")
    if not REPO_PATH_RE.match(locator):
        raise ChampionOntologyError("repository locator contains invalid characters")
    if not locator.startswith("data/"):
        raise ChampionOntologyError("repository locator must be under data/")
    return locator


def _extract_datadragon_patch(url: str) -> str | None:
    parsed = urlparse(url)
    for segment in parsed.path.split("/"):
        if not segment:
            continue
        if PATCH_RE.fullmatch(segment):
            try:
                return public_patch(segment)
            except ValueError:
                return segment
        if segment.count(".") >= 2:
            parts = segment.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                try:
                    return public_patch(f"{parts[0]}.{parts[1]}")
                except ValueError:
                    return f"{parts[0]}.{parts[1]}"
    return None


def _validate_timestamp(ts: object) -> None:
    if not isinstance(ts, str):
        raise ChampionOntologyError("timestamp must be ISO-8601 string")
    value = ts.rstrip("Z")
    try:
        datetime.fromisoformat(value)
    except ValueError as err:
        raise ChampionOntologyError(f"invalid timestamp: {ts}") from err


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    champion_id: str
    patch_id: str
    role: str
    dimension: str
    reviewer: str
    label: str
    decision: str
    confidence: float
    revision_of: str | None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class EmpiricalCell:
    champion_id: str
    patch_id: str
    role: str
    league_id: str
    residual: dict[str, Any]
    verified_appearance_count: int


def _parse_review_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.rstrip("Z"))


def _iter_revision_index(reviews: list[ReviewRecord]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for review in reviews:
        if review.revision_of is None:
            continue
        index.setdefault(review.revision_of, []).append(review.review_id)
    return index


def _assert_acyclic_revisions(adjacency: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ChampionOntologyError(f"revision cycle detected at {node}")

        visiting.add(node)
        for child in adjacency.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def _read_reviews(path: Path, as_of: str | None = None) -> list[ReviewRecord]:
    out: list[ReviewRecord] = []
    by_id: dict[str, ReviewRecord] = {}
    cutoff = _parse_review_timestamp(as_of) if as_of is not None else None
    for row in _load_jsonl(path):
        required = (
            "review_id",
            "champion_id",
            "patch_id",
            "role",
            "dimension",
            "reviewer",
            "label",
            "decision",
            "confidence",
        )
        for field in required:
            if field not in row:
                raise ChampionOntologyError(f"missing review field: {field}")

        if not REVIEW_ID_RE.match(row["review_id"]):
            raise ChampionOntologyError(f"invalid review_id: {row['review_id']}")
        reviewed_at = row.get("reviewed_at")
        if reviewed_at is None:
            raise ChampionOntologyError(f"review row missing reviewed_at: {row['review_id']}")
        _validate_timestamp(reviewed_at)
        review_time = _parse_review_timestamp(reviewed_at)
        if cutoff is not None and review_time is not None and review_time > cutoff:
            continue

        review_id = row["review_id"]
        if review_id in by_id:
            raise ChampionOntologyError(f"duplicated review_id: {row['review_id']}")

        if row["decision"] not in {"proposed", "accepted", "disputed", "revoked"}:
            raise ChampionOntologyError(f"invalid decision: {row['decision']}")

        champion_id = row["champion_id"]
        if not STABLE_ID_RE.match(champion_id):
            raise ChampionOntologyError(f"invalid champion id in review: {champion_id}")
        if row["role"] not in ROLES:
            raise ChampionOntologyError(f"invalid role in review: {row['role']}")
        if row["dimension"] not in REQUIRED_DIMENSIONS:
            raise ChampionOntologyError(f"invalid dimension in review: {row['dimension']}")
        if row["label"] not in DIMENSION_LABELS[row["dimension"]]:
            raise ChampionOntologyError(
                f"invalid label '{row['label']}' for dimension {row['dimension']}"
            )
        if not (0 <= float(row["confidence"]) <= 1):
            raise ChampionOntologyError(f"invalid review confidence: {row['confidence']}")
        _parse_patch(row["patch_id"])
        if not row["reviewer"]:
            raise ChampionOntologyError("reviewer must not be empty")
        if row.get("revision_of") is not None and not REVIEW_ID_RE.match(row["revision_of"]):
            raise ChampionOntologyError(f"invalid revision_of: {row['revision_of']}")

        out.append(
            ReviewRecord(
                review_id=review_id,
                champion_id=row["champion_id"],
                patch_id=row["patch_id"],
                role=row["role"],
                dimension=row["dimension"],
                reviewer=row["reviewer"],
                label=row["label"],
                decision=row["decision"],
                confidence=float(row["confidence"]),
                revision_of=row.get("revision_of"),
                reviewed_at=reviewed_at,
            )
        )
        by_id[review_id] = out[-1]

    for record in out:
        if record.revision_of is not None and record.revision_of not in by_id:
            raise ChampionOntologyError(f"unknown revision_of: {record.revision_of}")

        if record.revision_of is not None:
            parent = by_id[record.revision_of]
            if (
                parent.champion_id != record.champion_id
                or parent.patch_id != record.patch_id
                or parent.role != record.role
                or parent.dimension != record.dimension
            ):
                raise ChampionOntologyError(
                    "revision_of must target same champion/patch/role/dimension: "
                    f"{record.review_id}"
                )

            parent_time = _parse_review_timestamp(parent.reviewed_at)
            current_time = _parse_review_timestamp(record.reviewed_at)
            if current_time is not None and parent_time is not None and current_time < parent_time:
                raise ChampionOntologyError(
                    f"revision chronology invalid for {record.review_id}"
                )

    adjacency = _iter_revision_index(out)
    _assert_acyclic_revisions(adjacency)

    return out


def _validate_empirical_cell(row: dict[str, Any]) -> EmpiricalCell:
    for field in EMPIRICAL_REQUIRED_FIELDS:
        if field not in row:
            raise ChampionOntologyError(f"empirical row missing field: {field}")

    champion_id = row["champion_id"]
    if not isinstance(champion_id, str) or not STABLE_ID_RE.match(champion_id):
        raise ChampionOntologyError(f"invalid empirical champion_id: {champion_id}")
    patch_id = row["patch_id"]
    _parse_patch(patch_id)
    role = row["role"]
    if role not in ROLES:
        raise ChampionOntologyError(f"invalid empirical role: {role}")
    league_id = row["league_id"]
    if not isinstance(league_id, str) or not league_id:
        raise ChampionOntologyError("empirical league_id must be non-empty string")

    residual = row["residual"]
    if not isinstance(residual, dict):
        raise ChampionOntologyError("empirical residual must be object")
    for field in ("status", "mean", "sigma", "observation_count"):
        if field not in residual:
            raise ChampionOntologyError(f"empirical residual missing field {field}")
    if residual["status"] not in {"observed", "masked", "prior_only", "missing_ontology"}:
        raise ChampionOntologyError(f"invalid empirical residual status: {residual['status']}")
    if not isinstance(residual["mean"], (int, float)):
        raise ChampionOntologyError("empirical residual mean must be numeric")
    if not isinstance(residual["sigma"], (int, float)) or residual["sigma"] <= 0:
        raise ChampionOntologyError("empirical residual sigma must be positive")
    if not isinstance(residual["observation_count"], int) or residual["observation_count"] < 0:
        raise ChampionOntologyError("empirical residual observation_count must be non-negative int")

    appearance_count = row["verified_appearance_count"]
    if not isinstance(appearance_count, int) or appearance_count < 0:
        raise ChampionOntologyError("empirical verified_appearance_count must be non-negative int")

    return EmpiricalCell(
        champion_id=champion_id,
        patch_id=patch_id,
        role=role,
        league_id=league_id,
        residual=dict(residual),
        verified_appearance_count=appearance_count,
    )


def _load_empirical(
    path: Path | None,
    as_of: str | None = None,
) -> tuple[dict[tuple[str, str, str, str], EmpiricalCell], dict[str, Any] | None]:
    if path is None:
        return {}, None

    payload = _load_json(path)
    if not isinstance(payload.get("cells"), list):
        raise ChampionOntologyError("empirical payload missing cells list")
    if not isinstance(payload.get("schema_version"), str) or not payload["schema_version"]:
        raise ChampionOntologyError("empirical payload missing schema_version")
    if not isinstance(payload.get("as_of"), str) or not payload["as_of"]:
        raise ChampionOntologyError("empirical payload missing as_of")
    _validate_timestamp(payload["as_of"])
    if as_of is not None:
        empirical_as_of = _parse_review_timestamp(payload["as_of"])
        requested_as_of = _parse_review_timestamp(as_of)
        if empirical_as_of > requested_as_of:
            raise ChampionOntologyError(
                f"empirical snapshot as_of {payload['as_of']} exceeds requested as_of {as_of}"
            )

    index: dict[tuple[str, str, str, str], EmpiricalCell] = {}
    for row in payload["cells"]:
        if not isinstance(row, dict):
            raise ChampionOntologyError("empirical row must be object")
        cell = _validate_empirical_cell(row)
        key = (cell.champion_id, cell.patch_id, cell.role, cell.league_id)
        if key in index:
            raise ChampionOntologyError(f"duplicated empirical cell: {key}")
        index[key] = cell
    return index, payload


def _validate_sources(
    sources_payload: dict[str, Any],
    root: Path,
    expected_manual_locator: str | None,
    as_of: str | None = None,
) -> str:
    if "as_of" not in sources_payload:
        raise ChampionOntologyError("sources payload missing as_of")
    if not isinstance(sources_payload["as_of"], str) or not sources_payload["as_of"]:
        raise ChampionOntologyError("sources payload as_of must be a non-empty string")
    _validate_timestamp(sources_payload["as_of"])
    source_as_of = sources_payload["as_of"]
    source_as_of_time = _parse_review_timestamp(source_as_of)
    if as_of is not None:
        cutoff = _parse_review_timestamp(as_of)
        if source_as_of_time > cutoff:
            raise ChampionOntologyError(
                f"source metadata as_of {source_as_of} exceeds requested as_of {as_of}"
            )

    if not isinstance(sources_payload.get("sources"), list):
        raise ChampionOntologyError("sources payload requires list at 'sources'")
    required = (
        "source_id",
        "kind",
        "publication_decision",
        "reviewed_by",
        "reviewed_at",
    )
    allowed_decisions = {
        "public",
        "authenticated",
        "private",
        "private_pending_review",
        "prohibited",
    }
    allowed_kinds = {
        "riot_datadragon",
        "communitydragon",
        "manual_labels",
        "manual_review",
        "official_note",
        "atom_bridge",
    }
    seen: set[str] = set()
    for row in sources_payload["sources"]:
        if not isinstance(row, dict):
            raise ChampionOntologyError("source row must be object")
        for field in required:
            if field not in row:
                raise ChampionOntologyError(f"source row missing field {field}")
        source_id = row["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise ChampionOntologyError("source_id must be a non-empty string")
        if source_id in seen:
            raise ChampionOntologyError(f"duplicated source_id: {source_id}")
        seen.add(source_id)

        if row["kind"] not in allowed_kinds:
            raise ChampionOntologyError(f"invalid source kind: {row['kind']}")
        kind = row["kind"]
        locator_kind = row.get("locator_kind")
        locator = row.get("locator")
        url = row.get("url")

        if kind == "manual_labels":
            if locator_kind != "repository_path":
                raise ChampionOntologyError(
                    f"manual_labels source must declare repository_path locator: {source_id}"
                )
            _validate_repository_locator(locator)
            if not SOURCE_RELATIVE_RE.match(locator):
                raise ChampionOntologyError(
                    f"manual_labels source locator must point to champion review log: {source_id}"
                )
            if not locator.endswith(".jsonl"):
                raise ChampionOntologyError(
                    f"manual_labels source locator must target a JSONL file: {source_id}"
                )
            resolved_locator = _to_repo_relative(root / locator, root)
            if expected_manual_locator is None:
                raise ChampionOntologyError(
                    "manual_labels locator cannot be resolved without review source path"
                )
            if resolved_locator != expected_manual_locator:
                raise ChampionOntologyError(
                    f"manual_labels locator {resolved_locator} does not match review path {expected_manual_locator}"
                )
            locator_path = (root / locator).resolve()
            if not locator_path.exists():
                raise ChampionOntologyError(
                    f"manual_labels locator path does not exist: {locator}"
                )
            if url is not None:
                raise ChampionOntologyError(
                    f"manual_labels source must not use url: {source_id}"
                )
        elif kind in {"official_note", "manual_review"}:
            if locator is not None:
                raise ChampionOntologyError(f"{kind} source cannot include locator: {source_id}")
            if locator_kind is not None and locator_kind != "":
                raise ChampionOntologyError(f"{kind} source must not include locator: {source_id}")
            if not isinstance(url, str) or not url:
                raise ChampionOntologyError(f"{kind} source must include https URL: {source_id}")
            if not url.startswith("https://"):
                raise ChampionOntologyError(f"source url must be https: {source_id}")
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ChampionOntologyError(f"source row missing netloc: {source_id}")
            if ".example" in parsed.netloc:
                raise ChampionOntologyError(f"source url cannot use .example domain: {source_id}")
            if ".example." in parsed.netloc or parsed.netloc.endswith(".example"):
                raise ChampionOntologyError(f"source url cannot be private .example placeholder: {source_id}")
        elif kind == "atom_bridge":
            # Automated mechanistic prior derived from the League Combat
            # Calculator atom cache through scryglass.lcc-atom-bridge.v1.
            if locator_kind != "repository_path":
                raise ChampionOntologyError(
                    f"atom_bridge source must declare repository_path locator: {source_id}"
                )
            _validate_repository_locator(locator)
            if not SOURCE_RELATIVE_RE.match(locator):
                raise ChampionOntologyError(
                    f"atom_bridge source locator must point under data/lol/v2/champions: {source_id}"
                )
            if not locator.endswith(".json"):
                raise ChampionOntologyError(
                    f"atom_bridge source locator must target a JSON artifact: {source_id}"
                )
            locator_path = (root / locator).resolve()
            if not locator_path.exists():
                raise ChampionOntologyError(
                    f"atom_bridge source locator path does not exist: {source_id}"
                )
            if url is not None:
                raise ChampionOntologyError(
                    f"atom_bridge source must not use url: {source_id}"
                )
        elif kind == "communitydragon":
            if locator_kind != "receipt":
                raise ChampionOntologyError(
                    f"communitydragon source must declare a receipt locator: {source_id}"
                )
            _validate_repository_locator(locator)
            if not locator.endswith("-receipt.json"):
                raise ChampionOntologyError(
                    f"communitydragon source locator must target a receipt: {source_id}"
                )
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ChampionOntologyError(
                    f"communitydragon source must include an https URL: {source_id}"
                )
            parsed = urlparse(url)
            if parsed.netloc != "raw.communitydragon.org":
                raise ChampionOntologyError(
                    f"communitydragon source host is not allowlisted: {source_id}"
                )
            public = row.get("public_patch") or row.get("patch")
            client = row.get("client_patch")
            if public is None or client is None:
                raise ChampionOntologyError(
                    f"communitydragon source must declare public and client patches: {source_id}"
                )
            try:
                if public_patch(public) != str(public) or client_patch(public) != str(client):
                    raise ChampionOntologyError(
                        f"communitydragon source patch namespaces do not match: {source_id}"
                    )
            except ValueError as exc:
                raise ChampionOntologyError(
                    f"communitydragon source patch is invalid: {source_id}"
                ) from exc
        elif kind == "riot_datadragon":
            if locator is not None:
                raise ChampionOntologyError(f"riot_datadragon source cannot include locator: {source_id}")
            if locator_kind is not None and locator_kind != "":
                raise ChampionOntologyError(f"riot_datadragon source cannot include locator_kind: {source_id}")
            if not isinstance(url, str) or not url:
                raise ChampionOntologyError(f"source missing url: {source_id}")
            if not url.startswith("https://"):
                raise ChampionOntologyError(f"source url must be https: {source_id}")
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ChampionOntologyError(f"source row missing netloc: {source_id}")
            if ".example" in parsed.netloc:
                raise ChampionOntologyError(f"source url cannot use .example domain: {source_id}")
            if ".example." in parsed.netloc or parsed.netloc.endswith(".example"):
                raise ChampionOntologyError(f"source url cannot be private .example placeholder: {source_id}")
            patch_id = row.get("patch_id")
            if patch_id is None:
                raise ChampionOntologyError(f"source row missing patch_id for riot_datadragon: {source_id}")
            _parse_patch(patch_id)
            discovered_patch = _extract_datadragon_patch(url)
            if discovered_patch is None:
                raise ChampionOntologyError(f"source row missing datadragon patch in url: {source_id}")
            if discovered_patch != row["patch_id"]:
                raise ChampionOntologyError(
                    f"riot_datadragon source patch mismatch for {source_id}: "
                    f"{row['patch_id']} vs {discovered_patch}"
                )

        decision = row["publication_decision"]
        if decision not in allowed_decisions:
            raise ChampionOntologyError(f"invalid publication_decision: {decision}")

        _validate_timestamp(row["reviewed_at"])
        row_reviewed_at = _parse_review_timestamp(row["reviewed_at"])
        if row_reviewed_at is not None and row_reviewed_at > source_as_of_time:
            raise ChampionOntologyError(
                f"source row reviewed_at after source as_of: {source_id}"
            )
        if row["reviewed_by"] == "":
            raise ChampionOntologyError(f"source row missing reviewer: {source_id}")

    return source_as_of


def _validate_appearance_counts(appearances: Any) -> dict[str, int]:
    if not isinstance(appearances, dict):
        raise ChampionOntologyError("verified_appearances must be object")
    output: dict[str, int] = {}
    for league_id, value in appearances.items():
        if not isinstance(league_id, str) or not league_id:
            raise ChampionOntologyError("league_id in verified_appearances must be non-empty string")
        if isinstance(value, dict):
            raw = value.get("matches")
            if raw is None:
                raise ChampionOntologyError(f"league {league_id} verified_appearances requires matches")
        else:
            raw = value
        if not isinstance(raw, int) or raw < 0:
            raise ChampionOntologyError(f"verified_appearances count must be non-negative int for {league_id}")
        output[league_id] = raw
    return output


def _validate_residual(residual: Any, *, allow_observed: bool = True) -> None:
    if not isinstance(residual, dict):
        raise ChampionOntologyError("role profile residual must be object")
    for field in ("status", "mean", "sigma", "observation_count"):
        if field not in residual:
            raise ChampionOntologyError(f"role profile residual missing {field}")
    allowed_statuses = {"masked", "missing_ontology", "prior_only"}
    if allow_observed:
        allowed_statuses.add("observed")
    if residual["status"] not in allowed_statuses:
        raise ChampionOntologyError(f"invalid residual status: {residual['status']}")
    if not isinstance(residual["mean"], (int, float)):
        raise ChampionOntologyError("residual mean must be numeric")
    if not isinstance(residual["sigma"], (int, float)) or residual["sigma"] <= 0:
        raise ChampionOntologyError("residual sigma must be positive number")
    if not isinstance(residual["observation_count"], int) or residual["observation_count"] < 0:
        raise ChampionOntologyError("residual observation_count must be non-negative int")


def _validate_role_profile(role_profile: dict[str, Any]) -> None:
    for required in ("dimensions", "residual", "verified_appearances", "source_ids"):
        if required not in role_profile:
            raise ChampionOntologyError(f"role profile missing field {required}")
    if not isinstance(role_profile["source_ids"], list) or not role_profile["source_ids"]:
        raise ChampionOntologyError("role profile source_ids must be non-empty list")
    for source_id in role_profile["source_ids"]:
        if not isinstance(source_id, str) or not source_id:
            raise ChampionOntologyError("role profile source_id must be non-empty string")

    dimensions = role_profile["dimensions"]
    if not isinstance(dimensions, dict):
        raise ChampionOntologyError("role profile dimensions must be object")
    for dimension in REQUIRED_DIMENSIONS:
        if dimension not in dimensions:
            raise ChampionOntologyError(f"role profile missing dimension {dimension}")
        profile = dimensions[dimension]
        if not isinstance(profile, dict):
            raise ChampionOntologyError(f"dimension '{dimension}' must be object")
        if "labels" not in profile:
            raise ChampionOntologyError(f"dimension '{dimension}' missing labels")
        validate_dimension_map(dimension, profile["labels"])
        uncertainty = profile.get("uncertainty", {})
        validate_uncertainty_map(dimension, uncertainty)

    _validate_residual(role_profile["residual"], allow_observed=False)
    _validate_appearance_counts(role_profile["verified_appearances"])


def _collect_patch_profiles(roles_payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(roles_payload, dict):
        raise ChampionOntologyError("patch_profiles must be object")
    if not roles_payload:
        raise ChampionOntologyError("champion has no patch profiles")

    patch_profiles: dict[str, dict[str, Any]] = {}
    for patch_id, patch in roles_payload.items():
        if not isinstance(patch_id, str) or not patch_id:
            raise ChampionOntologyError("patch_id must be non-empty string")
        _parse_patch(patch_id)
        if not isinstance(patch, dict):
            raise ChampionOntologyError(f"patch profile {patch_id} must be object")
        if "role_profiles" not in patch:
            raise ChampionOntologyError(f"patch {patch_id} missing role_profiles")
        role_profiles = patch["role_profiles"]
        if not isinstance(role_profiles, dict) or not role_profiles:
            raise ChampionOntologyError(f"patch {patch_id} role_profiles must be non-empty object")
        for role, profile in role_profiles.items():
            if role not in ROLES:
                raise ChampionOntologyError(f"invalid role in patch {patch_id}: {role}")
            if not isinstance(profile, dict):
                raise ChampionOntologyError(f"role profile for {role} in patch {patch_id} must be object")
            _validate_role_profile(profile)
        league_profiles = patch.get("league_role_profiles", {})
        if not isinstance(league_profiles, dict):
            raise ChampionOntologyError(f"patch {patch_id} league_role_profiles must be object")
        for league_id, league_payload in league_profiles.items():
            if not isinstance(league_id, str) or not league_id:
                raise ChampionOntologyError(f"league id in patch {patch_id} must be non-empty string")
            if not isinstance(league_payload, dict):
                raise ChampionOntologyError(f"league_role_profiles.{league_id} in patch {patch_id} must be object")
            for role, profile in league_payload.items():
                if role not in ROLES:
                    raise ChampionOntologyError(f"invalid league role in patch {patch_id}: {role}")
                if not isinstance(profile, dict):
                    raise ChampionOntologyError(f"league role profile for {role} in patch {patch_id} must be object")
                _validate_role_profile(profile)
        patch_profiles[patch_id] = patch

    return patch_profiles


def _validate_ontology(
    payload: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    for field in ("schema_version", "snapshot_id", "as_of", "champions"):
        if field not in payload:
            raise ChampionOntologyError(f"ontology payload missing required field: {field}")
    if not isinstance(payload["schema_version"], str) or not payload["schema_version"]:
        raise ChampionOntologyError("schema_version must be non-empty string")
    if not isinstance(payload["snapshot_id"], str) or not payload["snapshot_id"]:
        raise ChampionOntologyError("snapshot_id must be non-empty string")
    if not isinstance(payload["as_of"], str) or not payload["as_of"]:
        raise ChampionOntologyError("as_of must be non-empty string")
    _validate_timestamp(payload["as_of"])
    if not isinstance(payload["champions"], list):
        raise ChampionOntologyError("champions must be a list")
    if not payload["champions"]:
        raise ChampionOntologyError("ontology must contain at least one champion")

    source_ids = set(source_by_id)

    seen_champions: set[str] = set()
    seen_aliases: dict[str, str] = {}
    for row in payload["champions"]:
        if not isinstance(row, dict):
            raise ChampionOntologyError("champion row must be object")
        for field in ("champion_id", "display_name", "aliases", "role_legalities", "patch_profiles"):
            if field not in row:
                raise ChampionOntologyError(f"champion row missing required field: {field}")

        champion_id = row["champion_id"]
        if not STABLE_ID_RE.match(champion_id):
            raise ChampionOntologyError(f"invalid champion_id: {champion_id}")
        if champion_id in seen_champions:
            raise ChampionOntologyError(f"duplicate champion_id: {champion_id}")
        seen_champions.add(champion_id)

        display_name = row["display_name"]
        if not isinstance(display_name, str) or not display_name:
            raise ChampionOntologyError(f"{champion_id} display_name must be non-empty string")

        aliases = row["aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise ChampionOntologyError(f"{champion_id} aliases must be non-empty list")
        for item in aliases:
            if not isinstance(item, dict) or "value" not in item:
                raise ChampionOntologyError(f"{champion_id} alias entries require dict with value")
            alias = item["value"].strip().lower()
            if not alias:
                raise ChampionOntologyError(f"{champion_id} has empty alias")
            if alias in seen_aliases and seen_aliases[alias] != champion_id:
                raise ChampionOntologyError(
                    f"alias collision on {alias}: {champion_id} and {seen_aliases[alias]}"
                )
            seen_aliases[alias] = champion_id

        legal_roles = row["role_legalities"]
        if not isinstance(legal_roles, list) or not legal_roles:
            raise ChampionOntologyError(f"{champion_id} role_legalities must be non-empty list")
        for role in legal_roles:
            if role not in ROLES:
                raise ChampionOntologyError(f"{champion_id} has invalid legal role: {role}")

        patch_profiles = _collect_patch_profiles(row["patch_profiles"])
        source_references = set()
        for patch_profile in patch_profiles.values():
            for role_profile in patch_profile["role_profiles"].values():
                source_references.update(role_profile["source_ids"])
            for league_payload in patch_profile.get("league_role_profiles", {}).values():
                for role_profile in league_payload.values():
                    source_references.update(role_profile["source_ids"])

        unknown_source = source_references - source_ids
        if unknown_source:
            unknown = sorted(unknown_source)[0]
            raise ChampionOntologyError(f"{champion_id} references unknown source_id: {unknown}")

        for patch_id, patch_profile in patch_profiles.items():
            for role_profile in patch_profile["role_profiles"].values():
                for source_id in role_profile["source_ids"]:
                    source = source_by_id[source_id]
                    if source["kind"] == "riot_datadragon" and source["patch_id"] != patch_id:
                        raise ChampionOntologyError(
                            f"{champion_id} references riot_datadragon source {source_id} "
                            f"for patch {patch_id} but source patch is {source['patch_id']}"
                        )

            for league_payload in patch_profile.get("league_role_profiles", {}).values():
                for role_profile in league_payload.values():
                    for source_id in role_profile["source_ids"]:
                        source = source_by_id[source_id]
                        if source["kind"] == "riot_datadragon" and source["patch_id"] != patch_id:
                            raise ChampionOntologyError(
                                f"{champion_id} references riot_datadragon source {source_id} "
                                f"for patch {patch_id} but source patch is {source['patch_id']}"
                            )


def _resolve_patch_profile(
    patch_profiles: dict[str, dict[str, Any]],
    requested_patch: str | None,
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    if not patch_profiles:
        raise ChampionOntologyError("no patch profiles available")
    resolved_patches = sorted(patch_profiles, key=_parse_patch)
    if requested_patch is None:
        return resolved_patches[-1], patch_profiles[resolved_patches[-1]], ["patch_default_latest"]

    target = _parse_patch(requested_patch)
    if requested_patch in patch_profiles:
        return requested_patch, patch_profiles[requested_patch], []

    earliest_patch = resolved_patches[0]
    if _parse_patch(earliest_patch) > target:
        return None, None, [f"patch_no_prior:{requested_patch}:{earliest_patch}"]

    for patch_id in reversed(resolved_patches):
        if _parse_patch(patch_id) <= target:
            return patch_id, patch_profiles[patch_id], [f"patch_fallback:{patch_id}"]

    return None, None, [f"patch_no_prior:{requested_patch}:{earliest_patch}"]


def _normalize_feature_value(value: object) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ChampionOntologyError(f"invalid feature value: {normalized}")
    return normalized


def _flatten_dimension_values(
    role_profile: dict[str, Any],
) -> tuple[list[float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    dimension_values: dict[str, dict[str, float]] = {}
    dimension_uncertainty: dict[str, dict[str, float]] = {}
    vector: list[float] = []
    for dimension, label in DIMENSION_LABEL_ORDER:
        profile = role_profile["dimensions"][dimension]
        values = validate_dimension_map(dimension, profile["labels"])
        uncertainty = validate_uncertainty_map(dimension, profile.get("uncertainty", {}))
        dimension_values[dimension] = values
        dimension_uncertainty[dimension] = uncertainty
        vector.append(values[label])
    return vector, dimension_values, dimension_uncertainty


def _flatten_vector_from_dimension_values(
    dimension_values: dict[str, dict[str, float]]
) -> tuple[list[float], dict[str, dict[str, float]]]:
    vector: list[float] = []
    ordered_values: dict[str, dict[str, float]] = {}
    for dimension, label in DIMENSION_LABEL_ORDER:
        values = dimension_values.get(dimension, {})
        ordered_values[dimension] = values
        vector.append(values[label])
    return vector, ordered_values


def _apply_review_rules(
    dimension_values: dict[str, dict[str, float]],
    dimension_uncertainty: dict[str, dict[str, float]],
    review_summary: dict[str, Any],
    fallback_for_patch: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[str]]:
    effective_values: dict[str, dict[str, float]] = {}
    effective_uncertainty: dict[str, dict[str, float]] = {}
    applied_rules: list[str] = [REVIEW_EFFECT_RULE_ID]
    for dimension in REQUIRED_DIMENSIONS:
        values = dimension_values[dimension]
        uncertainty = dimension_uncertainty[dimension]
        status = review_summary.get(dimension, {}).get("status", "unreviewed")
        rule = DIMENSION_REVIEW_FACTORS.get(status, DIMENSION_REVIEW_FACTORS["unreviewed"])
        value_weight = rule["value_weight"]
        uncertainty_mult = rule["uncertainty_multiplier"]
        if fallback_for_patch:
            uncertainty_mult = uncertainty_mult * FUTURE_PATCH_UNCERTAINTY_MULTIPLIER
        applied_rules.append(f"dimension_review:{dimension}:{status}:{REVIEW_EFFECT_RULE_ID}")
        if fallback_for_patch:
            applied_rules.append(FUTURE_PATCH_FALLBACK_RULE_ID)
        effective_values[dimension] = {}
        effective_uncertainty[dimension] = {}
        k = len(values)
        if k == 0:
            raise ChampionOntologyError("dimension values missing labels")
        uniform = 1.0 / k
        for label, value in values.items():
            blended = value_weight * value + (1.0 - value_weight) * uniform
            effective_values[dimension][label] = round(blended, 8)
            scaled_uncertainty = uncertainty[label] * uncertainty_mult
            effective_uncertainty[dimension][label] = round(
                max(0.0, scaled_uncertainty), 8
            )
    return effective_values, effective_uncertainty, sorted(set(applied_rules))


def _build_residual(
    residual: dict[str, Any],
    observed_matches: int | None = None,
    *,
    widen_for_patch_fallback: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    resolved = dict(residual)
    issues: list[str] = []
    status = resolved.get("status", "masked")
    sigma = max(float(residual.get("sigma", 1.0)), 1.0)
    mean = float(residual.get("mean", 0.0))
    observations = int(residual.get("observation_count", 0))
    if status == "observed" and observations <= 0:
        status = "prior_only"
        sigma = max(sigma, 2.0)
        mean = 0.0
        issues.append("zero_observation_count")
    elif status == "missing_ontology":
        mean = 0.0
        sigma = max(sigma, 2.0)
    elif status == "prior_only":
        mean = 0.0
        sigma = max(sigma, 2.0)
    elif status != "observed":
        mean = 0.0
        sigma = max(sigma, 1.2)

    if widen_for_patch_fallback and status != "missing_ontology":
        sigma = max(2.0, sigma) + FUTURE_PATCH_RESIDUAL_SIGMA_WIDEN
        issues.append(f"patch_fallback_unseen:{FUTURE_PATCH_FALLBACK_RULE_ID}")

    if observed_matches == 0:
        if status in {"observed", "masked", "prior_only"}:
            status = "prior_only" if status != "missing_ontology" else status
        mean = 0.0
        sigma = max(sigma, 2.0)
        if "zero_appearance" not in issues:
            issues.append("zero_appearance")
    return {
        "status": status,
        "mean": round(mean, 8),
        "sigma": round(float(sigma), 8),
        "observation_count": observations,
    }, issues


def _league_matches(verified_appearances: dict[str, int], league_id: str) -> int:
    return int(verified_appearances.get(league_id, 0))


def _effective_review_rows(rows: list[ReviewRecord]) -> list[ReviewRecord]:
    if not rows:
        return []
    superseded: set[str] = {row.revision_of for row in rows if row.revision_of is not None}
    return [
        row
        for row in rows
        if row.review_id not in superseded and row.decision != "revoked"
    ]


def _fallback_level(issues: list[str], *, has_ontology: bool, has_role_profile: bool) -> str:
    if any(issue.startswith("patch_no_prior") for issue in issues):
        return "no_prior_patch"
    if any(issue.startswith("patch_fallback") for issue in issues):
        return "future_patch_fallback"
    if any(issue.startswith("league_profile_only") for issue in issues):
        return "league_profile_only"
    if not has_ontology:
        return "missing_ontology"
    if not has_role_profile:
        return "missing_role_profile"
    return "none"


def _review_coverage(review_summary: dict[str, Any]) -> dict[str, Any]:
    statuses = {dimension: summary["status"] for dimension, summary in review_summary.items()}
    disputed = sorted(dimension for dimension, status in statuses.items() if status == "disputed")
    unreviewed = sorted(dimension for dimension, status in statuses.items() if status == "unreviewed")
    accepted = sorted(dimension for dimension, status in statuses.items() if status == "accepted")
    total = len(REQUIRED_DIMENSIONS)
    reviewed = total - sum(1 for status in statuses.values() if status == "unreviewed")
    return {
        "dimensions_total": total,
        "dimensions_reviewed": reviewed,
        "disputed_dimensions": disputed,
        "unreviewed_dimensions": unreviewed,
        "accepted_dimensions": accepted,
    }


class ChampionOntology:
    """Immutable access layer for archetype priors and review trail."""

    def __init__(
        self,
        *,
        ontology_payload: dict[str, Any],
        source_payload: dict[str, Any],
        review_records: list[ReviewRecord],
        review_cutoff: str | None = None,
        empirical_cells: dict[tuple[str, str, str, str], EmpiricalCell] | None = None,
        empirical_payload: dict[str, Any] | None = None,
        expected_manual_locator: str | None = None,
        ontology_as_of: str | None = None,
        source_as_of: str | None = None,
    ) -> None:
        validated_source_as_of = _validate_sources(
            source_payload,
            _repo_root(),
            expected_manual_locator,
            as_of=review_cutoff or ontology_payload["as_of"],
        )
        if source_as_of is not None and source_as_of != validated_source_as_of:
            raise ChampionOntologyError(
                f"source as_of mismatch: payload {source_payload['as_of']} vs validated {validated_source_as_of}"
            )
        source_by_id = {row["source_id"]: row for row in source_payload["sources"]}
        _validate_ontology(ontology_payload, source_by_id)
        self._ontology_payload = ontology_payload
        self._source_payload = source_payload
        self._source_by_id = source_by_id
        self._by_id = {row["champion_id"]: row for row in ontology_payload["champions"]}
        self._reviews = sorted(
            review_records,
            key=lambda row: (row.reviewed_at or "", row.review_id),
        )
        self._review_as_of = review_cutoff or ontology_payload["as_of"]
        self._as_of = self._review_as_of
        self._ontology_as_of = ontology_as_of or ontology_payload["as_of"]
        self._source_as_of = validated_source_as_of
        self._review_rows = [
            {
                "review_id": row.review_id,
                "champion_id": row.champion_id,
                "patch_id": row.patch_id,
                "role": row.role,
                "dimension": row.dimension,
                "reviewer": row.reviewer,
                "label": row.label,
                "decision": row.decision,
                "confidence": row.confidence,
                "revision_of": row.revision_of,
                "reviewed_at": row.reviewed_at,
            }
            for row in self._reviews
        ]
        self._review_rows.sort(key=lambda row: (row["reviewed_at"] or "", row["review_id"]))

        self._review_index: dict[tuple[str, str, str, str], list[ReviewRecord]] = {}
        for review in review_records:
            key = (review.champion_id, review.patch_id, review.role, review.dimension)
            self._review_index.setdefault(key, []).append(review)

        self._empirical_cells = empirical_cells or {}
        self._empirical_payload = empirical_payload
        self._empirical_snapshot_hash = (
            canonical_sha256(empirical_payload) if empirical_payload is not None else None
        )

        self._expected_manual_locator = expected_manual_locator

    @property
    def snapshot_id(self) -> str:
        return self._ontology_payload["snapshot_id"]

    @property
    def as_of(self) -> str:
        return self._as_of

    @property
    def ontology_as_of(self) -> str:
        return self._ontology_as_of

    @property
    def source_as_of(self) -> str:
        return self._source_as_of

    @property
    def empirical_as_of(self) -> str | None:
        return self._empirical_payload["as_of"] if self._empirical_payload is not None else None

    @property
    def schema_version(self) -> str:
        return self._ontology_payload["schema_version"]

    @property
    def source_metadata_sha256(self) -> str:
        return canonical_sha256(self._source_payload)

    @property
    def source_snapshot_hash(self) -> None:
        return None

    @property
    def ontology_snapshot_hash(self) -> str:
        return canonical_sha256(self._ontology_payload)

    @property
    def empirical_snapshot_hash(self) -> str | None:
        return self._empirical_snapshot_hash

    @property
    def review_snapshot_hash(self) -> str:
        return canonical_sha256({"as_of": self._review_as_of, "reviews": self._review_rows})

    def _lookup_empirical_cell(
        self,
        champion_id: str,
        patch_id: str,
        role: str,
        league_id: str,
    ) -> EmpiricalCell | None:
        if league_id is None:
            return None
        return self._empirical_cells.get((champion_id, patch_id, role, league_id))

    def champion_ids(self) -> list[str]:
        return sorted(self._by_id)

    def build_alias_index(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for champ_id, row in self._by_id.items():
            for alias_row in row["aliases"]:
                alias = alias_row["value"].strip().lower()
                aliases[alias] = champ_id
        return aliases

    def resolve_by_alias(self, alias: str) -> str | None:
        return self.build_alias_index().get(alias.strip().lower())

    def _resolve_profile(
        self,
        champion_id: str,
        patch_id: str | None,
        role: str,
        league_id: str | None = None,
    ) -> tuple[
        str | None,
        str | None,
        list[str],
        list[str],
        dict[str, Any] | None,
        list[str],
    ]:
        if champion_id not in self._by_id:
            return (
                None,
                None,
                [],
                [],
                None,
                [f"unknown_champion:{champion_id}"],
            )
        entry = self._by_id[champion_id]
        resolved_patch_id, patch_profile, issues = _resolve_patch_profile(
            entry["patch_profiles"],
            patch_id,
        )
        requested_profile_roles: list[str] = []
        if patch_id is not None and patch_id in entry["patch_profiles"]:
            requested_profile_roles = sorted(
                entry["patch_profiles"][patch_id].get("role_profiles", {}).keys()
            )

        if patch_profile is None:
            return (
                None,
                patch_id,
                [],
                requested_profile_roles,
                None,
                issues,
            )

        role_profiles = patch_profile["role_profiles"]
        chosen_profile = role_profiles.get(role)
        exact_patch_roles = self._role_legal_for_patch(
            champion_id=champion_id,
            patch_id=resolved_patch_id,
        )
        issue_list: list[str] = list(issues)

        league_profiles = patch_profile.get("league_role_profiles", {})
        if league_id and league_id in league_profiles and role in league_profiles[league_id]:
            if chosen_profile is None:
                issue_list.append("league_profile_only")
            else:
                issue_list.append("league_profile_only_ignored")
        if chosen_profile is None:
            return (
                resolved_patch_id,
                patch_id,
                exact_patch_roles,
                requested_profile_roles,
                None,
                issue_list + [f"missing_role_profile:{role}"],
            )

        return (
            resolved_patch_id,
            patch_id,
            self._role_legal_for_patch(
                champion_id=champion_id,
                patch_id=resolved_patch_id,
            ),
            requested_profile_roles,
            chosen_profile,
            issue_list,
        )

    def _review_summary(
        self,
        champion_id: str,
        patch_id: str,
        role: str,
        dimension: str,
        authored_dimension: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        key = (champion_id, patch_id, role, dimension)
        rows = self._review_index.get(key, [])
        if not rows:
            return {
                "status": "unreviewed",
                "review_count": 0,
                "disagreement": False,
                "label_uncertainty": {},
                "top_label": None,
                "top_weight": 0.0,
                "effective_review_count": 0,
                "revisions": [],
                "reviewers": [],
            }

        effective_rows = _effective_review_rows(rows)
        if not effective_rows:
            return {
                "status": "unreviewed",
                "review_count": len(rows),
                "disagreement": False,
                "label_uncertainty": {},
                "top_label": None,
                "top_weight": 0.0,
                "effective_review_count": 0,
                "revisions": [],
                "reviewers": sorted({row.reviewer for row in rows}),
            }

        latest = max(effective_rows, key=lambda r: (r.reviewed_at or "", r.review_id))
        total_weight = 0.0
        weights: dict[str, float] = {}
        accepted_only: dict[str, float] = {}
        disputed = False
        for row in effective_rows:
            weights[row.label] = weights.get(row.label, 0.0) + row.confidence
            total_weight += row.confidence
            if row.decision == "accepted":
                accepted_only[row.label] = accepted_only.get(row.label, 0.0) + row.confidence
            elif row.decision != "accepted":
                disputed = True

        if not total_weight:
            label_uncertainty: dict[str, float] = {}
        else:
            label_uncertainty = {
                label: round(1.0 - (weight / total_weight), 6) for label, weight in weights.items()
            }

        if not accepted_only or disputed or len(accepted_only) != 1:
            status = "disputed"
            top_label = max(weights, key=weights.get)
            top_weight = 0.0
        else:
            top_label = max(accepted_only, key=accepted_only.get)
            accepted_weight = sum(accepted_only.values()) or 1.0
            top_weight = round(accepted_only[top_label] / accepted_weight, 6)
            status = "accepted"

            if authored_dimension is not None:
                top_authored = max(
                    authored_dimension.items(),
                    key=lambda item: (item[1], item[0]),
                )[0]
                if top_label != top_authored:
                    status = "disputed"
                    top_weight = 0.0
        disagreement = status == "disputed"

        return {
            "status": status,
            "review_count": len(rows),
            "disagreement": disagreement,
            "label_uncertainty": label_uncertainty,
            "top_label": top_label,
            "top_weight": top_weight,
            "latest_review_id": latest.review_id,
            "latest_decision": latest.decision,
            "effective_review_count": len(effective_rows),
            "revisions": [
                {
                    "review_id": row.review_id,
                    "revision_of": row.revision_of,
                    "reviewer": row.reviewer,
                    "label": row.label,
                    "decision": row.decision,
                    "confidence": round(row.confidence, 6),
                }
                for row in sorted(rows, key=lambda r: (r.reviewed_at or "", r.review_id))
            ],
            "reviewers": sorted({row.reviewer for row in rows}),
        }

    def build_feature_vector(
        self,
        champion_id: str,
        role: str,
        patch_id: str,
        league_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in ROLES:
            raise ChampionOntologyError(f"invalid role: {role}")
        if not STABLE_ID_RE.match(champion_id):
            raise ChampionOntologyError(f"invalid champion id: {champion_id}")
        _parse_patch(patch_id)

        resolved_patch_id, requested_patch, exact_patch_roles, requested_patch_roles, role_profile, issues = self._resolve_profile(
            champion_id=champion_id,
            patch_id=patch_id,
            role=role,
            league_id=league_id,
        )
        is_exact_patch = patch_id == resolved_patch_id
        requested_patch_role_legal_available = bool(requested_patch_roles)
        requested_patch_role_legal = (
            bool(requested_patch_roles) and role in requested_patch_roles
            if requested_patch is not None
            else False
        )
        is_exact_patch_role_legal = role in exact_patch_roles
        fallback_for_patch = any(issue.startswith("patch_fallback") for issue in issues)
        if requested_patch is not None and not is_exact_patch and not requested_patch_role_legal_available:
            issues.append(f"requested_patch_role_legality_unavailable:{requested_patch}")
        if requested_patch is not None and requested_patch_role_legal_available and not requested_patch_role_legal:
            issues.append(f"requested_patch_role_illegal:{requested_patch}:{role}")

        if requested_patch is None:
            fallback_level = "missing_ontology"
            exact_cell = {
                "patch_id": None,
                "requested_patch_id": patch_id,
                "league_id": league_id,
                "role": role,
                "verified_appearance_count": 0,
                "is_exact_patch": False,
                "is_eligible": False,
                "is_role_legal": False,
                "is_requested_patch_role_legal": False,
                "is_requested_patch_role_legal_available": False,
                "is_exact_patch_role_legal": False,
                "requested_patch_role_legal": False,
                "requested_patch_roles": [],
                "exact_patch_roles": [],
            }
            residual_evidence = {
                "status": "missing_ontology",
                "mean": 0.0,
                "sigma": 2.0,
                "observation_count": 0,
                "exact_cell_observations": 0,
                "residual_source": "seed",
                "notes": sorted(set(issues)),
            }
            return {
                "champion_id": champion_id,
                "role": role,
                "patch_id": patch_id,
                "resolved_patch_id": None,
                "display_name": None,
                "residual": {
                    "status": residual_evidence["status"],
                    "mean": residual_evidence["mean"],
                    "sigma": residual_evidence["sigma"],
                    "observation_count": residual_evidence["observation_count"],
                },
                "ontology_coverage": {
                    "has_ontology": False,
                    "has_role_profile": False,
                    "requested_role": role,
                    "resolved_patch_id": None,
                },
                "residual_evidence": residual_evidence,
                "exact_cell_appearances": exact_cell,
                "source_ids": [],
                "vector": [0.0 for _ in DIMENSION_LABEL_ORDER],
                "dimension_values": {},
                "authored_dimension_values": {},
                "dimension_uncertainty": {},
                "authored_dimension_uncertainty": {},
                "review_summary": {},
                "dimension_rules": [],
                "exact_patch_roles": exact_patch_roles,
                "requested_patch_roles": requested_patch_roles,
                "review_coverage": _review_coverage({}),
                "tier_list_eligible": False,
                "tier_list_eligibility_reason": "missing_champion",
                "fallback_level": fallback_level,
                "issues": issues,
            }

        if role_profile is None:
            has_ontology = champion_id in self._by_id
            fallback_level = _fallback_level(
                issues,
                has_ontology=has_ontology,
                has_role_profile=False,
            )
            exact_cell = {
                    "patch_id": resolved_patch_id,
                    "requested_patch_id": patch_id,
                    "league_id": league_id,
                    "role": role,
                "verified_appearance_count": 0,
                "is_exact_patch": is_exact_patch,
                "is_eligible": False,
                "is_role_legal": False,
                "is_requested_patch_role_legal": requested_patch_role_legal,
                "is_requested_patch_role_legal_available": requested_patch_role_legal_available,
                "is_exact_patch_role_legal": False,
                "requested_patch_role_legal": requested_patch_role_legal,
                "requested_patch_roles": requested_patch_roles,
                "exact_patch_roles": exact_patch_roles,
            }
            residual = {
                "status": "masked",
                "mean": 0.0,
                "sigma": 2.0,
                "observation_count": 0,
            }
            residual_evidence = {
                "status": "masked",
                "mean": 0.0,
                "sigma": 2.0,
                "observation_count": 0,
                "exact_cell_observations": 0,
                "residual_source": "seed",
                "notes": issues,
            }
            return {
                    "champion_id": champion_id,
                    "role": role,
                    "patch_id": patch_id,
                    "resolved_patch_id": resolved_patch_id,
                    "display_name": self._by_id[champion_id]["display_name"],
                    "requested_patch_id": requested_patch,
                    "residual": residual,
                    "fallback_level": fallback_level,
                    "ontology_coverage": {
                        "has_ontology": True,
                        "has_role_profile": False,
                        "requested_role": role,
                        "resolved_patch_id": resolved_patch_id,
                        "source_ids": [],
                    },
                    "residual_evidence": residual_evidence,
                    "exact_cell_appearances": exact_cell,
                    "source_ids": [],
                    "vector": [0.0 for _ in DIMENSION_LABEL_ORDER],
                    "dimension_values": {},
                    "authored_dimension_values": {},
                "dimension_uncertainty": {},
                "authored_dimension_uncertainty": {},
                "review_summary": {},
                "dimension_rules": [],
                "exact_patch_roles": exact_patch_roles,
                "requested_patch_roles": requested_patch_roles,
                "review_coverage": _review_coverage({}),
                    "tier_list_eligible": False,
                    "tier_list_eligibility_reason": next(
                        (
                            issue
                            for issue in issues
                            if issue.startswith("patch_no_prior")
                        ),
                        f"missing_role_profile:{patch_id}" if has_ontology else f"patch_no_prior:{patch_id}",
                    ),
                    "issues": issues + [f"missing_role_profile:{role}"],
                }

        vector, authored_dimension_values, authored_dimension_uncertainty = _flatten_dimension_values(role_profile)
        for value in vector:
            _normalize_feature_value(value)

        review_summary: dict[str, Any] = {}
        for dimension in REQUIRED_DIMENSIONS:
            review_summary[dimension] = self._review_summary(
                champion_id=champion_id,
                patch_id=resolved_patch_id,
                role=role,
                dimension=dimension,
                authored_dimension=authored_dimension_values[dimension],
            )

        dimension_values, dimension_uncertainty, dimension_rules = _apply_review_rules(
            authored_dimension_values,
            authored_dimension_uncertainty,
            review_summary,
            fallback_for_patch=fallback_for_patch,
        )
        vector, dimension_values = _flatten_vector_from_dimension_values(dimension_values)
        dimension_values = dict(dimension_values)
        dimension_uncertainty = dict(dimension_uncertainty)

        residual_source = "seed"
        empirical_cell = None
        if league_id is not None:
            empirical_cell = self._lookup_empirical_cell(
                champion_id=champion_id,
                patch_id=resolved_patch_id,
                role=role,
                league_id=league_id,
            )

        if empirical_cell is not None:
            residual_source_payload = empirical_cell.residual
            residual_source = "empirical"
        else:
            residual_source_payload = role_profile["residual"]
        verified_appearances = role_profile["verified_appearances"]
        if empirical_cell is not None:
            exact_cell_matches = empirical_cell.verified_appearance_count
            exact_cell_match_source = "empirical"
        else:
            exact_cell_matches = _league_matches(verified_appearances, league_id) if league_id else 0
            exact_cell_match_source = "seed"

        residual, residual_issues = _build_residual(
            residual_source_payload,
            observed_matches=exact_cell_matches if league_id else None,
            widen_for_patch_fallback=fallback_for_patch,
        )
        all_issues = issues + residual_issues
        league_reason = "tier_not_requested"
        tier_ok = False
        if league_id is None:
            league_reason = "league_required"
        elif not is_exact_patch_role_legal:
            league_reason = "role_not_legal_for_patch"
        elif not is_exact_patch:
            league_reason = f"tier_requires_exact_patch:{resolved_patch_id}"
        elif exact_cell_matches <= 0:
            league_reason = f"no_verified_appearances:{league_id}"
        else:
            tier_ok = True
            league_reason = "ok"

        return {
            "champion_id": champion_id,
            "role": role,
            "patch_id": patch_id,
            "resolved_patch_id": resolved_patch_id,
            "requested_patch_id": requested_patch,
            "display_name": self._by_id[champion_id]["display_name"],
            "source_ids": list(role_profile["source_ids"]),
            "exact_patch_roles": exact_patch_roles,
            "residual": residual,
            "vector": vector,
            "vector_dimension_count": len(vector),
            "dimension_values": dimension_values,
            "authored_dimension_values": authored_dimension_values,
            "dimension_uncertainty": dimension_uncertainty,
            "authored_dimension_uncertainty": authored_dimension_uncertainty,
            "review_summary": review_summary,
            "dimension_rules": dimension_rules,
            "tier_list_eligible": tier_ok,
            "tier_list_eligibility_reason": league_reason,
            "fallback_level": _fallback_level(
                issues,
                has_ontology=True,
                has_role_profile=True,
            ),
            "ontology_coverage": {
                "has_ontology": True,
                "has_role_profile": True,
                "requested_role": role,
                "resolved_patch_id": resolved_patch_id,
                "source_ids": list(role_profile["source_ids"]),
                "requested_patch_id": requested_patch,
                "has_vector": True,
            },
            "residual_evidence": {
                "status": residual["status"],
                "observation_count": residual["observation_count"],
                "sigma": residual["sigma"],
                "mean": residual["mean"],
                "exact_cell_observations": exact_cell_matches,
                "residual_source": residual_source,
                "notes": sorted(set(all_issues)),
            },
            "exact_cell_appearances": {
                "patch_id": resolved_patch_id,
                "requested_patch_id": patch_id,
                "league_id": league_id,
                "role": role,
                "verified_appearance_count": exact_cell_matches,
                "is_exact_patch": is_exact_patch,
                "is_eligible": tier_ok,
                "is_role_legal": is_exact_patch_role_legal,
                "is_requested_patch_role_legal": requested_patch_role_legal,
                "is_requested_patch_role_legal_available": requested_patch_role_legal_available,
                "is_exact_patch_role_legal": is_exact_patch_role_legal,
                "requested_patch_roles": requested_patch_roles,
                "exact_patch_roles": exact_patch_roles,
                "exact_cell_match_source": exact_cell_match_source,
                "requested_patch_role_legal": requested_patch_role_legal,
            },
            "review_coverage": _review_coverage(review_summary),
            "issues": sorted(set(all_issues)),
        }

    def role_legal(self, champion_id: str, patch_id: str | None = None) -> list[str]:
        entry = self._by_id.get(champion_id)
        if not entry:
            return []
        role_set: set[str] = set(entry["role_legalities"])
        if patch_id is not None:
            patch_profile = entry["patch_profiles"].get(patch_id)
            if patch_profile is not None:
                role_set.update(patch_profile["role_profiles"].keys())
        return sorted(role_set)

    def _role_legal_for_patch(self, champion_id: str, patch_id: str | None) -> list[str]:
        entry = self._by_id.get(champion_id)
        if not entry or patch_id is None:
            return []
        patch_profile = entry["patch_profiles"].get(patch_id)
        if patch_profile is None:
            return []
        return sorted(patch_profile["role_profiles"].keys())

    def tier_list_eligible(
        self,
        *,
        champion_id: str,
        patch_id: str | None,
        role: str,
        league_id: str,
    ) -> bool:
        if not league_id:
            return False
        feature = self.build_feature_vector(
            champion_id=champion_id,
            role=role,
            patch_id=patch_id,
            league_id=league_id,
        )
        return bool(feature["tier_list_eligible"])

    def build_archetype_prior(
        self,
        *,
        champion_id: str,
        role: str,
        patch_id: str,
        league_id: str | None = None,
    ) -> dict[str, Any]:
        feature = self.build_feature_vector(
            champion_id=champion_id,
            role=role,
            patch_id=patch_id,
            league_id=league_id,
        )
        declared_role_legal: list[str] = []
        champion_entry = self._by_id.get(champion_id)
        if champion_entry is not None:
            declared_role_legal = sorted(champion_entry["role_legalities"])
        return {
            "champion_id": champion_id,
            "role": role,
            "patch_id": patch_id,
            "resolved_patch_id": feature["resolved_patch_id"],
            "display_name": feature["display_name"],
            "role_legal": declared_role_legal,
            "role_legal_requested_patch": feature.get("requested_patch_roles", []),
            "role_legal_exact_patch": feature.get("exact_patch_roles", []),
            "requested_patch_id": feature.get("requested_patch_id"),
            "requested_patch_roles": feature.get("requested_patch_roles", []),
            "exact_patch_roles": feature.get("exact_patch_roles", []),
            "vector": feature["vector"],
            "vector_dimension_count": len(feature["vector"]),
            "authored_dimension_values": feature.get("authored_dimension_values", {}),
            "dimension_values": feature.get("dimension_values", {}),
            "authored_dimension_uncertainty": feature.get("authored_dimension_uncertainty", {}),
            "residual": feature["residual"],
            "source_ids": feature["source_ids"],
            "dimension_rules": feature.get("dimension_rules", []),
            "tier_list_eligible": feature["tier_list_eligible"],
            "tier_list_eligibility_reason": feature["tier_list_eligibility_reason"],
            "issues": feature["issues"],
            "artifact_ids": {
                "ontology_snapshot_id": self.snapshot_id,
                "review_snapshot_id": self.review_snapshot_hash,
                "review_snapshot_status": "derived",
                "review_snapshot_as_of": self._review_as_of,
                "empirical_snapshot_id": self.empirical_snapshot_hash,
                "empirical_snapshot_status": (
                    "available" if self._empirical_payload is not None else "pending_l1_snapshot"
                ),
                "empirical_snapshot_as_of": self.empirical_as_of,
                "source_snapshot_id": None,
                "source_snapshot_status": "pending_l1_snapshot",
                "source_metadata_sha256": self.source_metadata_sha256,
                "ontology_as_of": self.ontology_as_of,
                "source_as_of": self.source_as_of,
                "source_ids": sorted(self._source_by_id),
            },
            "review_summary": feature["review_summary"],
            "snapshot_hashes": {
                "ontology_sha256": self.ontology_snapshot_hash,
                "source_metadata_sha256": self.source_metadata_sha256,
                "reviews_sha256": self.review_snapshot_hash,
                "empirical_sha256": self.empirical_snapshot_hash,
            },
            "ontology_as_of": self.ontology_as_of,
            "source_as_of": self.source_as_of,
            "empirical_as_of": self.empirical_as_of,
            "as_of": self.as_of,
            "review_as_of": self._review_as_of,
            "dimension_uncertainty": feature.get("dimension_uncertainty", {}),
            "review_coverage": feature["review_coverage"],
            "ontology_coverage": feature["ontology_coverage"],
            "residual_evidence": feature["residual_evidence"],
            "exact_cell_appearances": feature["exact_cell_appearances"],
            "fallback_level": feature["fallback_level"],
            "review_status": {
                "status": {
                    dimension: values["status"] for dimension, values in feature["review_summary"].items()
                }
            },
            "source_payload": {
                "decision": {source_id: self._source_by_id[source_id]["publication_decision"] for source_id in feature["source_ids"]},
            },
            "source_dependency": {
                "source_snapshot_id": None,
                "source_snapshot_status": "pending_l1_snapshot",
                "source_metadata_sha256": self.source_metadata_sha256,
                "source_as_of": self.source_as_of,
                "ontology_as_of": self.ontology_as_of,
                "empirical_as_of": self.empirical_as_of,
                "review_as_of": self._review_as_of,
            },
        }

    def profile_distance(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ChampionOntologyError("feature vectors must have equal length")
        if len(a) != len(DIMENSION_LABEL_ORDER):
            raise ChampionOntologyError("feature vector length does not match ontology schema")
        return sum((left - right) ** 2 for left, right in zip(a, b)) ** 0.5


def load_champion_ontology(
    *,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    source_path: Path = DEFAULT_SOURCE_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
    as_of: str | None = None,
    empirical_path: Path | None = None,
) -> ChampionOntology:
    ontology_payload = _load_json(ontology_path)
    source_payload = _load_json(source_path)
    _validate_timestamp(ontology_payload["as_of"])
    if not isinstance(source_payload.get("as_of"), str) or not source_payload["as_of"]:
        raise ChampionOntologyError("sources payload missing as_of")
    _validate_timestamp(source_payload["as_of"])

    root = _repo_root()
    try:
        expected_manual_locator = _to_repo_relative(review_path, root)
    except ValueError as err:
        raise ChampionOntologyError(
            f"review_path must be inside repository root for locator validation: {review_path}"
        ) from err

    if as_of is None:
        as_of = ontology_payload["as_of"]
    else:
        _validate_timestamp(as_of)
    if _parse_review_timestamp(as_of) < _parse_review_timestamp(ontology_payload["as_of"]):
        raise ChampionOntologyError("as_of cutoff cannot be earlier than ontology as_of")
    if _parse_review_timestamp(source_payload["as_of"]) > _parse_review_timestamp(as_of):
        raise ChampionOntologyError(
            f"source metadata as_of {source_payload['as_of']} exceeds requested as_of {as_of}"
        )

    empirical_cells, empirical_payload = _load_empirical(empirical_path, as_of=as_of)

    requested_cutoff = as_of
    review_cutoff = requested_cutoff
    if _parse_review_timestamp(source_payload["as_of"]) < _parse_review_timestamp(requested_cutoff):
        review_cutoff = source_payload["as_of"]

    reviews = _read_reviews(review_path, as_of=review_cutoff)
    return ChampionOntology(
        ontology_payload=ontology_payload,
        source_payload=source_payload,
        review_records=reviews,
        review_cutoff=as_of,
        empirical_cells=empirical_cells,
        empirical_payload=empirical_payload,
        expected_manual_locator=expected_manual_locator,
        ontology_as_of=ontology_payload["as_of"],
        source_as_of=source_payload["as_of"],
    )
