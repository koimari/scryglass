"""Internal mechanics-first draft, roster, GRID, and winner-gate contracts.

The module is intentionally independent of the public Draft Score API.  It
provides deterministic feature keys and fail-closed evaluation primitives for
the new source-backed engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.v2.data.common import parse_rfc3339, to_rfc3339


SCHEMA_VERSION = "scryglass:mechanics-composite:v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_ALIASES = {
    "top": "top",
    "jungle": "jng",
    "jng": "jng",
    "jg": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "support": "sup",
    "sup": "sup",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROSTER_ACTIVE_STATUSES = frozenset({"starter", "substitute", "confirmed_starter", "confirmed_substitute"})


class CompositeEngineError(ValueError):
    """Raised when a composite contract is malformed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositeEngineError(f"{name} must be a non-empty string")
    return value.strip()


def _role(value: Any) -> str:
    raw = _text(value, "role").casefold()
    if raw not in ROLE_ALIASES:
        raise CompositeEngineError(f"unsupported role: {value}")
    return ROLE_ALIASES[raw]


def _timestamp(value: Any, name: str) -> datetime:
    try:
        return parse_rfc3339(_text(value, name))
    except Exception as exc:
        raise CompositeEngineError(f"{name} must be RFC-3339") from exc


def _sha256(value: Any, name: str) -> str:
    candidate = _text(value, name)
    if not SHA256_RE.fullmatch(candidate):
        raise CompositeEngineError(f"{name} must be a lowercase SHA-256")
    return candidate


@dataclass(frozen=True)
class RosterInterval:
    team_id: str
    player_id: str
    role: str
    effective_from: str
    effective_until: str | None
    available_at: str
    roster_status: str
    source_snapshot_hash: str
    authority_status: str = "confirmed"

    def __post_init__(self) -> None:
        _text(self.team_id, "team_id")
        _text(self.player_id, "player_id")
        _role(self.role)
        start = _timestamp(self.effective_from, "effective_from")
        end = _timestamp(self.effective_until, "effective_until") if self.effective_until else None
        available = _timestamp(self.available_at, "available_at")
        if end is not None and end <= start:
            raise CompositeEngineError("roster interval must move forward")
        # A transaction may become public after its effective timestamp but
        # still before a later match.  Availability is therefore checked at
        # resolution time against the event cutoff, not against the interval
        # start.
        _sha256(self.source_snapshot_hash, "source_snapshot_hash")
        if self.roster_status not in ROSTER_ACTIVE_STATUSES | {"inactive", "leave"}:
            raise CompositeEngineError(f"unsupported roster_status: {self.roster_status}")
        if self.authority_status not in {"confirmed", "candidate", "unavailable"}:
            raise CompositeEngineError(f"unsupported authority_status: {self.authority_status}")

    def active_at(self, event_start: str | datetime, as_of: str | datetime) -> bool:
        event = event_start if isinstance(event_start, datetime) else _timestamp(event_start, "event_start")
        cutoff = as_of if isinstance(as_of, datetime) else _timestamp(as_of, "as_of")
        start = _timestamp(self.effective_from, "effective_from")
        end = _timestamp(self.effective_until, "effective_until") if self.effective_until else None
        available = _timestamp(self.available_at, "available_at")
        return available <= cutoff and start <= event and (end is None or event < end)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "player_id": self.player_id,
            "role": _role(self.role),
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "available_at": self.available_at,
            "roster_status": self.roster_status,
            "source_snapshot_hash": self.source_snapshot_hash,
            "authority_status": self.authority_status,
        }


@dataclass(frozen=True)
class LineupSnapshot:
    fixture_id: str
    as_of: str
    event_start: str
    team_id: str
    players: tuple[tuple[str, str], ...]
    evidence_hash: str | None
    authority_status: str
    blockers: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.authority_status == "confirmed" and not self.blockers

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "as_of": self.as_of,
            "event_start": self.event_start,
            "team_id": self.team_id,
            "players": [{"role": role, "player_id": player} for role, player in self.players],
            "evidence_hash": self.evidence_hash,
            "authority_status": self.authority_status,
            "blockers": list(self.blockers),
        }


class TemporalRosterRegistry:
    """Append-only interval registry with exact pre-event resolution."""

    def __init__(self, intervals: Iterable[RosterInterval] = ()):
        self._intervals: list[RosterInterval] = []
        for interval in intervals:
            self.add(interval)

    def add(self, interval: RosterInterval) -> None:
        if not isinstance(interval, RosterInterval):
            raise CompositeEngineError("roster registry accepts RosterInterval values")
        self._intervals.append(interval)
        self._intervals.sort(key=lambda row: (row.team_id, row.role, row.effective_from, row.player_id))

    def resolve(
        self,
        *,
        fixture_id: str,
        team_id: str,
        as_of: str,
        event_start: str,
    ) -> LineupSnapshot:
        _text(fixture_id, "fixture_id")
        _text(team_id, "team_id")
        cutoff = _timestamp(as_of, "as_of")
        event = _timestamp(event_start, "event_start")
        blockers: list[str] = []
        if cutoff >= event:
            blockers.append("roster_as_of_not_before_event_start")
        active = [
            row
            for row in self._intervals
            if row.team_id == team_id and row.active_at(event, cutoff)
        ]
        by_role: dict[str, list[RosterInterval]] = {role: [] for role in ROLES}
        for row in active:
            by_role[_role(row.role)].append(row)
        selected: list[tuple[str, str]] = []
        evidence: list[str] = []
        for role in ROLES:
            candidates = by_role[role]
            if len(candidates) != 1:
                blockers.append(f"roster_role_{role}_arity_{len(candidates)}")
                continue
            row = candidates[0]
            if row.authority_status != "confirmed":
                blockers.append(f"roster_role_{role}_not_confirmed")
            if row.roster_status in {"inactive", "leave"}:
                blockers.append(f"roster_role_{role}_{row.roster_status}")
            selected.append((role, row.player_id))
            evidence.append(row.source_snapshot_hash)
        players = [player for _, player in selected]
        if len(players) != len(set(players)):
            blockers.append("roster_player_ids_not_unique")
        evidence_hash = _sha(sorted(evidence)) if evidence and not blockers else None
        return LineupSnapshot(
            fixture_id=fixture_id,
            as_of=to_rfc3339(cutoff),
            event_start=to_rfc3339(event),
            team_id=team_id,
            players=tuple(selected),
            evidence_hash=evidence_hash,
            authority_status="confirmed" if not blockers else "unavailable",
            blockers=tuple(sorted(set(blockers))),
        )


@dataclass(frozen=True)
class InteractionKey:
    focal_role: str
    focal_champion: str
    order: int
    allies: tuple[tuple[str, str], ...]
    enemies: tuple[tuple[str, str], ...]
    patch: str | None = None
    timing_bucket: str = "pregame"
    effect_type: str = "composition"
    target_filter: str = "all"

    def __post_init__(self) -> None:
        _role(self.focal_role)
        _text(self.focal_champion, "focal_champion")
        if type(self.order) is not int or not 1 <= self.order <= 9:
            raise CompositeEngineError("interaction order must be in [1, 9]")
        if self.order != len(self.allies) + len(self.enemies) + 1:
            raise CompositeEngineError("interaction order does not match context arity")
        if self.patch is not None:
            _text(self.patch, "patch")
        _text(self.timing_bucket, "timing_bucket")
        _text(self.effect_type, "effect_type")
        _text(self.target_filter, "target_filter")

    @property
    def key(self) -> str:
        context = [f"A:{role}:{champion}" for role, champion in self.allies]
        context.extend(f"E:{role}:{champion}" for role, champion in self.enemies)
        patch = self.patch or "unbound"
        prefix = f"IH|patch={patch}|time={self.timing_bucket}|effect={self.effect_type}|target={self.target_filter}"
        return f"{prefix}|role={_role(self.focal_role)}|champion={self.focal_champion}|order={self.order}|{'|'.join(sorted(context))}"

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key, "focal_role": _role(self.focal_role), "focal_champion": self.focal_champion, "order": self.order, "allies": [{"role": role, "champion": champion} for role, champion in self.allies], "enemies": [{"role": role, "champion": champion} for role, champion in self.enemies], "patch": self.patch, "timing_bucket": self.timing_bucket, "effect_type": self.effect_type, "target_filter": self.target_filter}


def _validate_draft(draft: Mapping[str, Mapping[str, str]]) -> None:
    if set(draft) != {"blue", "red"}:
        raise CompositeEngineError("draft must contain blue and red sides")
    all_champions: list[str] = []
    for side in ("blue", "red"):
        if set(draft[side]) != set(ROLES):
            raise CompositeEngineError(f"{side} draft must contain all canonical roles")
        for role in ROLES:
            all_champions.append(_text(draft[side][role], f"{side}.{role}"))
    if len(all_champions) != len(set(all_champions)):
        raise CompositeEngineError("draft cannot contain duplicate champions")


def iter_interaction_keys(
    draft: Mapping[str, Mapping[str, str]],
    *,
    max_order: int = 9,
    patch: str | None = None,
    timing_bucket: str = "pregame",
    effect_type: str = "composition",
    target_filter: str = "all",
) -> tuple[InteractionKey, ...]:
    """Enumerate focal-champion interactions for every order 1 through 9."""

    _validate_draft(draft)
    if type(max_order) is not int or not 1 <= max_order <= 9:
        raise CompositeEngineError("max_order must be in [1, 9]")
    result: list[InteractionKey] = []
    for side in ("blue", "red"):
        enemy_side = "red" if side == "blue" else "blue"
        for role in ROLES:
            allies = tuple(
                (other_role, draft[side][other_role])
                for other_role in ROLES
                if other_role != role
            )
            enemies = tuple((other_role, draft[enemy_side][other_role]) for other_role in ROLES)
            contexts = tuple(("A", item) for item in allies) + tuple(("E", item) for item in enemies)
            for context_order in range(0, max_order):
                for subset in combinations(contexts, context_order):
                    selected_allies = tuple(sorted(item for relation, item in subset if relation == "A"))
                    selected_enemies = tuple(sorted(item for relation, item in subset if relation == "E"))
                    result.append(InteractionKey(role, draft[side][role], context_order + 1, selected_allies, selected_enemies, patch, timing_bucket, effect_type, target_filter))
    return tuple(sorted(result, key=lambda item: item.key))


def interaction_backoff_chain(item: InteractionKey) -> tuple[str, ...]:
    """Return exact-to-lower-order keys without inventing a coefficient."""

    contexts = tuple(("A", value) for value in item.allies) + tuple(("E", value) for value in item.enemies)
    keys: list[str] = [item.key]
    for retained in range(len(contexts) - 1, -1, -1):
        for subset in combinations(contexts, retained):
            allies = tuple(sorted(value for relation, value in subset if relation == "A"))
            enemies = tuple(sorted(value for relation, value in subset if relation == "E"))
            candidate = InteractionKey(item.focal_role, item.focal_champion, retained + 1, allies, enemies, item.patch, item.timing_bucket, item.effect_type, item.target_filter).key
            if candidate not in keys:
                keys.append(candidate)
    return tuple(keys)


@dataclass(frozen=True)
class InteractionTerm:
    key: str
    order: int
    family: str
    value: float | None
    source: str
    support: int
    available: bool
    blocker: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key, "order": self.order, "family": self.family, "value": self.value, "source": self.source, "support": self.support, "available": self.available, "blocker": self.blocker}


@dataclass(frozen=True)
class InteractionLedger:
    terms: tuple[InteractionTerm, ...]
    total_edge: float | None
    blockers: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.total_edge is not None and not self.blockers

    def to_mapping(self) -> dict[str, Any]:
        return {"terms": [term.to_mapping() for term in self.terms], "total_edge": self.total_edge, "blockers": list(self.blockers), "available": self.available}


def _interaction_family(item: InteractionKey) -> str:
    if item.order == 1:
        return "champion_baseline"
    if item.allies and not item.enemies:
        return "ally_synergy"
    if item.enemies and not item.allies:
        return "enemy_counter"
    return "mixed_interaction"


def score_interactions(
    draft: Mapping[str, Mapping[str, str]],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    max_order: int = 9,
    allow_backoff: bool = True,
    allow_neutral: bool = False,
    patch: str | None = None,
    timing_bucket: str = "pregame",
    effect_type: str = "composition",
    target_filter: str = "all",
) -> InteractionLedger:
    """Resolve interaction coefficients with explicit support provenance."""

    terms: list[InteractionTerm] = []
    blockers: list[str] = []
    total = 0.0
    for item in iter_interaction_keys(
        draft,
        max_order=max_order,
        patch=patch,
        timing_bucket=timing_bucket,
        effect_type=effect_type,
        target_filter=target_filter,
    ):
        selected: Mapping[str, Any] | None = None
        selected_source = "exact"
        for candidate in interaction_backoff_chain(item) if allow_backoff else (item.key,):
            row = evidence.get(candidate)
            if not isinstance(row, Mapping):
                continue
            status = str(row.get("status", "blocked"))
            value = row.get("value")
            if status in {"exact", "reconciled", "supported"} and isinstance(value, (int, float)) and math.isfinite(float(value)):
                selected = row
                selected_source = "exact" if candidate == item.key else "backoff"
                break
        if selected is None:
            if allow_neutral:
                terms.append(InteractionTerm(item.key, item.order, _interaction_family(item), 0.0, "neutral", 0, True, None))
                continue
            blocker = f"interaction_unavailable:{item.key}"
            blockers.append(blocker)
            terms.append(InteractionTerm(item.key, item.order, _interaction_family(item), None, "blocked", 0, False, blocker))
            continue
        value = float(selected["value"])
        support = int(selected.get("support", 0))
        total += value
        terms.append(InteractionTerm(item.key, item.order, _interaction_family(item), value, selected_source, support, True, None))
    return InteractionLedger(tuple(terms), total if not blockers else None, tuple(sorted(set(blockers))))


@dataclass(frozen=True)
class GridCheckpointReceipt:
    series_id: str
    game_id: str
    provider_game_time_ms: int
    source_sequence: int
    observed_at: str
    state_sha256: str
    source_sha256: str
    complete: bool
    sequence_gap: bool = False
    revision_after_cutoff: bool = False
    provider_updated_at: str | None = None

    def __post_init__(self) -> None:
        _text(self.series_id, "series_id")
        _text(self.game_id, "game_id")
        if self.provider_game_time_ms < 0 or self.source_sequence < 0:
            raise CompositeEngineError("GRID time and sequence must be non-negative")
        _timestamp(self.observed_at, "observed_at")
        _sha256(self.state_sha256, "state_sha256")
        _sha256(self.source_sha256, "source_sha256")
        if self.provider_updated_at is not None:
            _timestamp(self.provider_updated_at, "provider_updated_at")

    @property
    def receipt_sha256(self) -> str:
        return _sha(self.to_mapping())

    def validate_checkpoint(
        self,
        *,
        cutoff_game_time_ms: int,
        maximum_age_ms: int,
        cutoff_observed_at: str | None = None,
        maximum_observed_age_ms: int | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        blockers: list[str] = []
        if type(cutoff_game_time_ms) is not int or cutoff_game_time_ms < 0:
            blockers.append("grid_cutoff_time_invalid")
        if type(maximum_age_ms) is not int or maximum_age_ms < 0:
            blockers.append("grid_maximum_age_invalid")
        if self.provider_game_time_ms > cutoff_game_time_ms:
            blockers.append("grid_state_after_cutoff")
        if cutoff_game_time_ms >= self.provider_game_time_ms and cutoff_game_time_ms - self.provider_game_time_ms > maximum_age_ms:
            blockers.append("grid_state_stale")
        if cutoff_observed_at is not None:
            cutoff = _timestamp(cutoff_observed_at, "cutoff_observed_at")
            observed = _timestamp(self.observed_at, "observed_at")
            if observed > cutoff:
                blockers.append("grid_observation_after_cutoff")
            if maximum_observed_age_ms is not None:
                if type(maximum_observed_age_ms) is not int or maximum_observed_age_ms < 0:
                    blockers.append("grid_maximum_observed_age_invalid")
                elif observed <= cutoff and (cutoff - observed).total_seconds() * 1000 > maximum_observed_age_ms:
                    blockers.append("grid_observation_stale")
            if self.provider_updated_at is not None and _timestamp(self.provider_updated_at, "provider_updated_at") > cutoff:
                blockers.append("grid_provider_revision_after_cutoff")
        if not self.complete:
            blockers.append("grid_state_incomplete")
        if self.sequence_gap:
            blockers.append("grid_sequence_gap")
        if self.revision_after_cutoff:
            blockers.append("grid_revision_after_cutoff")
        return not blockers, tuple(sorted(set(blockers)))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "series_id": self.series_id, "game_id": self.game_id, "provider_game_time_ms": self.provider_game_time_ms, "source_sequence": self.source_sequence, "observed_at": self.observed_at, "state_sha256": self.state_sha256, "source_sha256": self.source_sha256, "complete": self.complete, "sequence_gap": self.sequence_gap, "revision_after_cutoff": self.revision_after_cutoff, "provider_updated_at": self.provider_updated_at}


@dataclass(frozen=True)
class CompositePrediction:
    fixture_id: str
    pre_event_cutoff: str
    predicted_winner: str | None
    p_blue: float | None
    mechanics_score: float | None
    player_context_score: float | None
    synergy_counter_ledger: Mapping[str, Any]
    availability: str
    blockers: tuple[str, ...]
    model_version: str
    input_manifest_hash: str

    def __post_init__(self) -> None:
        if self.predicted_winner not in {None, "blue", "red"}:
            raise CompositeEngineError("predicted_winner must be blue, red, or null")
        if self.p_blue is not None and not 0.0 <= self.p_blue <= 1.0:
            raise CompositeEngineError("p_blue must be in [0, 1]")
        if self.availability not in {"available", "unavailable", "partial"}:
            raise CompositeEngineError("unsupported prediction availability")
        _sha256(self.input_manifest_hash, "input_manifest_hash")

    def to_mapping(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "fixture_id": self.fixture_id, "pre_event_cutoff": self.pre_event_cutoff, "predicted_winner": self.predicted_winner, "p_blue": self.p_blue, "mechanics_score": self.mechanics_score, "player_context_score": self.player_context_score, "synergy_counter_ledger": dict(self.synergy_counter_ledger), "availability": self.availability, "blockers": list(self.blockers), "model_version": self.model_version, "input_manifest_hash": self.input_manifest_hash}


def compose_prediction(
    *,
    fixture_id: str,
    pre_event_cutoff: str,
    mechanics_score: float | None,
    player_context_score: float | None,
    ledger: InteractionLedger,
    lineups: Sequence[LineupSnapshot],
    model_version: str = "mechanics-composite-v1.0.0",
    input_manifest: Mapping[str, Any] | None = None,
) -> CompositePrediction:
    blockers = list(ledger.blockers)
    if len(lineups) != 2:
        blockers.append("lineup_pair_missing")
    for lineup in lineups:
        blockers.extend(lineup.blockers)
        if not lineup.available:
            blockers.append(f"lineup_unavailable:{lineup.team_id}")
    if mechanics_score is None:
        blockers.append("mechanics_score_unavailable")
    if player_context_score is None:
        blockers.append("player_context_unavailable")
    edge: float | None = None
    p_blue: float | None = None
    winner: str | None = None
    if not blockers and ledger.total_edge is not None and mechanics_score is not None and player_context_score is not None:
        edge = float(mechanics_score) + float(player_context_score) + float(ledger.total_edge)
        p_blue = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, edge))))
        winner = "blue" if p_blue >= 0.5 else "red"
    manifest_hash = _sha(input_manifest or {"fixture_id": fixture_id, "cutoff": pre_event_cutoff, "ledger": ledger.to_mapping()})
    return CompositePrediction(
        fixture_id=_text(fixture_id, "fixture_id"),
        pre_event_cutoff=_text(pre_event_cutoff, "pre_event_cutoff"),
        predicted_winner=winner,
        p_blue=p_blue,
        mechanics_score=mechanics_score,
        player_context_score=player_context_score,
        synergy_counter_ledger=ledger.to_mapping(),
        availability="available" if not blockers else "unavailable",
        blockers=tuple(sorted(set(blockers))),
        model_version=_text(model_version, "model_version"),
        input_manifest_hash=manifest_hash,
    )


def wilson_interval(correct: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float, float] | None:
    if type(correct) is not int or type(total) is not int or total <= 0 or not 0 <= correct <= total:
        return None
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half), half)


def evaluate_winner_gate(
    predictions: Iterable[CompositePrediction],
    outcomes: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate every verified outcome, counting unavailable as non-success."""

    rows = list(predictions)
    if not rows:
        return {"schema_version": SCHEMA_VERSION, "total_verified_games": 0, "status": "unavailable"}
    correct = 0
    available = 0
    brier: list[float] = []
    logloss: list[float] = []
    by_availability = {"available": 0, "partial": 0, "unavailable": 0}
    missing_outcomes: list[str] = []
    for prediction in rows:
        outcome = outcomes.get(prediction.fixture_id)
        if outcome not in {"blue", "red"}:
            missing_outcomes.append(prediction.fixture_id)
            continue
        by_availability[prediction.availability] = by_availability.get(prediction.availability, 0) + 1
        if prediction.availability == "available" and prediction.predicted_winner == outcome:
            correct += 1
        if prediction.availability == "available" and prediction.p_blue is not None:
            available += 1
            y = 1.0 if outcome == "blue" else 0.0
            p = max(1e-9, min(1.0 - 1e-9, prediction.p_blue))
            brier.append((p - y) ** 2)
            logloss.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
    total = len(rows) - len(missing_outcomes)
    interval = wilson_interval(correct, total)
    primary_accuracy = correct / total if total else None
    available_correct = sum(
        1
        for prediction in rows
        if prediction.availability == "available"
        and outcomes.get(prediction.fixture_id) == prediction.predicted_winner
    )
    available_accuracy = available_correct / available if available else None
    half_width = interval[2] if interval else None
    return {
        "schema_version": SCHEMA_VERSION,
        "total_verified_games": total,
        "correct_primary_gate": correct,
        "primary_accuracy": primary_accuracy,
        "available_predictions": available,
        "coverage": available / total if total else None,
        "available_accuracy": available_accuracy,
        "wilson_interval_95": {"lower": interval[0], "upper": interval[1], "half_width": interval[2]} if interval else None,
        "brier_available_only": sum(brier) / len(brier) if brier else None,
        "logloss_available_only": sum(logloss) / len(logloss) if logloss else None,
        "availability_counts": by_availability,
        "missing_outcomes": sorted(missing_outcomes),
        "gate": {
            "accuracy_at_least_80_percent": bool(primary_accuracy is not None and primary_accuracy >= 0.80),
            "wilson_half_width_at_most_5_percent": bool(half_width is not None and half_width <= 0.05),
            "passed": bool(primary_accuracy is not None and primary_accuracy >= 0.80 and half_width is not None and half_width <= 0.05),
        },
    }


__all__ = [
    "CompositeEngineError",
    "CompositePrediction",
    "GridCheckpointReceipt",
    "InteractionKey",
    "InteractionLedger",
    "InteractionTerm",
    "LineupSnapshot",
    "ROLES",
    "RosterInterval",
    "TemporalRosterRegistry",
    "compose_prediction",
    "evaluate_winner_gate",
    "interaction_backoff_chain",
    "iter_interaction_keys",
    "score_interactions",
    "wilson_interval",
]
