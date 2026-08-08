"""Draft protocol and state validation foundations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .common import ContractError, ROLES, canonicalize_role, parse_rfc3339


class DraftProtocolError(ContractError):
    """Raised for invalid draft protocol definitions."""


class DraftActionError(ContractError):
    """Raised for invalid terminal draft states."""


CANONICAL_DRAFT_SIDES = ("A", "B")
GAME_DRAFT_SIDES = ("blue", "red")
DRAFT_ORDER_POSITIONS = ("first", "second")


@dataclass(frozen=True)
class DraftProtocolStep:
    slot: int
    kind: str
    side: str


@dataclass(frozen=True)
class DraftProtocol:
    protocol_id: str
    steps: tuple[DraftProtocolStep, ...]


@dataclass(frozen=True)
class DraftAction:
    slot: int
    kind: str
    side: str
    champion_id: str
    role_set: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DraftSideMapping:
    canonical_side: str
    game_side: str
    draft_order: str
    source_id: str
    observed_at: str
    available_at: str
    source_record_id: str = ""
    source_updated_at: str = ""
    is_observed: bool = True
    is_reconstructed: bool = False


@dataclass(frozen=True)
class DraftRoleSetRevision:
    action_slot: int
    canonical_side: str
    sequence_position: int
    previous_roles: tuple[str, ...]
    revised_roles: tuple[str, ...]
    reason: str
    source_id: str
    observed_at: str
    available_at: str
    source_record_id: str = ""
    source_updated_at: str = ""
    is_observed: bool = True
    is_reconstructed: bool = False


@dataclass(frozen=True)
class DraftStateValidation:
    is_valid: bool
    is_terminal: bool
    action_count: int
    missing_actions: tuple[int, ...]
    champion_conflicts: tuple[str, ...]
    errors: tuple[str, ...]
    protocol_id: str | None = None
    protocol_step_sides: tuple[tuple[int, str], ...] = ()
    action_order: tuple[int, ...] = ()
    side_assignment: tuple[tuple[str, str], ...] = ()
    game_side_assignment: tuple[tuple[str, str], ...] = ()
    draft_order_assignment: tuple[tuple[str, str], ...] = ()
    side_mappings: tuple[DraftSideMapping, ...] = ()
    role_set_revisions: tuple[DraftRoleSetRevision, ...] = ()
    revision_count: int = 0


def validate_protocol(protocol: DraftProtocol) -> None:
    if not protocol.protocol_id:
        raise DraftProtocolError("protocol_id required")
    if not protocol.steps:
        raise DraftProtocolError("protocol must have at least one step")

    ordered = sorted(protocol.steps, key=lambda step: step.slot)
    for index, step in enumerate(ordered, start=1):
        if not isinstance(step.slot, int) or step.slot < 1:
            raise DraftProtocolError("slot must be a positive integer")
        if step.slot != index:
            raise DraftProtocolError("protocol slots must be continuous positive integers")

    seen_slots: set[int] = set()
    for step in ordered:
        if step.slot in seen_slots:
            raise DraftProtocolError(f"duplicate step slot: {step.slot}")
        seen_slots.add(step.slot)

        if step.kind not in {"pick", "ban"}:
            raise DraftProtocolError(f"invalid step kind: {step.kind}")
        if step.side not in CANONICAL_DRAFT_SIDES:
            raise DraftProtocolError(f"invalid step side: {step.side}")


def validate_actions(
    protocol: DraftProtocol,
    actions: Iterable[DraftAction],
    *,
    side_mappings: Iterable[DraftSideMapping] = (),
    role_set_revisions: Iterable[DraftRoleSetRevision] = (),
    require_terminal: bool = False,
) -> DraftStateValidation:
    validate_protocol(protocol)

    actions_list = tuple(actions)
    mapping_list = tuple(side_mappings)
    revision_list = tuple(role_set_revisions)
    errors: list[str] = []

    for action in actions_list:
        if not isinstance(action.slot, int) or action.slot < 1:
            raise DraftActionError("slots must be positive integers")

    protocol_steps = {step.slot: step for step in protocol.steps}
    protocol_slots = tuple(sorted(protocol_steps))
    protocol_step_profile = tuple((slot, protocol_steps[slot].side) for slot in protocol_slots)

    action_by_slot: dict[int, DraftAction] = {}
    duplicate_slots: list[str] = []
    for action in actions_list:
        if action.slot in action_by_slot:
            duplicate_slots.append(str(action.slot))
            continue
        action_by_slot[action.slot] = action

    if duplicate_slots:
        errors.append("duplicate action slots: " + ",".join(sorted(set(duplicate_slots))))

    observed_slots = tuple(action.slot for action in actions_list)
    expected_prefix = protocol_slots[: len(observed_slots)]
    if observed_slots != expected_prefix:
        if protocol_slots:
            errors.append("action sequence must follow protocol slot order")

    missing = tuple(slot for slot in protocol_slots if slot not in action_by_slot)

    side_assignments: dict[str, list[str]] = {side: [] for side in CANONICAL_DRAFT_SIDES}
    pick_actions_by_side: dict[str, list[DraftAction]] = {side: [] for side in CANONICAL_DRAFT_SIDES}
    champion_counts: dict[str, int] = {}

    for action in actions_list:
        step = protocol_steps.get(action.slot)
        if step is None:
            errors.append(f"unknown slot in action: {action.slot}")
            continue

        if action.kind not in {"pick", "ban"}:
            errors.append(f"invalid action kind {action.kind} on slot {action.slot}")
        elif action.kind != step.kind:
            errors.append(f"slot {action.slot} kind mismatch: expected {step.kind}, got {action.kind}")

        if action.side not in CANONICAL_DRAFT_SIDES:
            errors.append(f"invalid side {action.side} in action slot {action.slot}")
        elif action.side != step.side:
            errors.append(f"slot {action.slot} side mismatch: expected {step.side}, got {action.side}")
        else:
            side_assignments[action.side].append(action.champion_id)

        if not action.champion_id:
            errors.append(f"slot {action.slot} requires champion_id")
        else:
            champion_counts[action.champion_id] = champion_counts.get(action.champion_id, 0) + 1

        if action.kind == "pick":
            if action.role_set is None or not action.role_set:
                errors.append(f"pick action {action.slot} must include role_set")
                continue
            normalized_roles = _normalize_role_set(action.role_set)
            pick_actions_by_side[action.side].append(action)

    normalized_revisions, revision_errors = _normalize_role_set_revisions(
        revision_list,
        actions=tuple(action_by_slot.values()),
    )
    errors.extend(revision_errors)

    mapping_result, mapping_errors = _normalize_side_mappings(mapping_list)
    errors.extend(mapping_errors)

    if require_terminal:
        if missing:
            errors.append("state is not terminal: draft missing actions")
        for side in CANONICAL_DRAFT_SIDES:
            terminal_roles = _terminal_side_roles(side, protocol, action_by_slot, normalized_revisions)
            if len(terminal_roles) != 5:
                errors.append(f"side {side} terminal requires exactly five single-role picks")
                continue
            if sorted(terminal_roles) != sorted(ROLES):
                errors.append(f"side {side} terminal roles must be top/jungle/mid/bot/support")

    if not mapping_errors and require_terminal and len(mapping_result) != len(CANONICAL_DRAFT_SIDES):
        errors.append("terminal draft requires both canonical sides mapped to game side and draft order")

    champion_conflicts = tuple(
        sorted(
            {
                f"{champion_id}"
                for champion_id, count in champion_counts.items()
                if count > 1
            }
        )
    )

    is_terminal = len(missing) == 0 and not errors and not champion_conflicts
    is_valid = len(errors) == 0 and not champion_conflicts

    return DraftStateValidation(
        is_valid=is_valid,
        is_terminal=is_terminal,
        action_count=len(actions_list),
        missing_actions=missing,
        champion_conflicts=champion_conflicts,
        errors=tuple(errors),
        protocol_id=protocol.protocol_id,
        protocol_step_sides=protocol_step_profile,
        action_order=observed_slots,
        side_assignment=tuple(
            (side, _first(value)) for side, value in side_assignments.items() if value
        ),
        game_side_assignment=tuple(
            (mapping.canonical_side, mapping.game_side) for mapping in mapping_result
        ),
        draft_order_assignment=tuple(
            (mapping.canonical_side, mapping.draft_order) for mapping in mapping_result
        ),
        side_mappings=mapping_result,
        role_set_revisions=normalized_revisions,
        revision_count=len(normalized_revisions),
    )


def finalize_role_tuple(action_map: dict[str, DraftAction]) -> tuple[str, str, str, str, str]:
    if len(action_map) != 5:
        raise DraftActionError("need exactly 5 roles for terminal map")

    for role in ROLES:
        if role not in action_map:
            raise DraftActionError(f"missing exact role: {role}")

    fixed_roles = []
    for role in ROLES:
        action = action_map[role]
        normalized_role_set = _normalize_role_set(action.role_set or ())
        if len(normalized_role_set) != 1:
            raise DraftActionError(f"role {role} not fixed in terminal state")
        fixed_roles.append(_normalize_role_name(normalized_role_set[0]))

    normalized = tuple(fixed_roles)
    if tuple(sorted(normalized)) != tuple(sorted(ROLES)):
        raise DraftActionError("role tuple must contain every legal role exactly once")
    return normalized


def _terminal_side_roles(
    side: str,
    protocol: DraftProtocol,
    actions_by_slot: dict[int, DraftAction],
    revisions: tuple[DraftRoleSetRevision, ...],
) -> tuple[str, ...]:
    picks = sorted(
        (
            action
            for action in actions_by_slot.values()
            if action.kind == "pick" and action.side == side and action.slot in protocol_slot_map(protocol).keys()
        ),
        key=lambda action: action.slot,
    )
    if len(picks) != 5:
        return ()

    role_map = {action.slot: _normalize_role_set(action.role_set or ()) for action in picks}
    for revision in sorted(revisions, key=lambda item: item.sequence_position):
        action = actions_by_slot.get(revision.action_slot)
        if action is None or action.side != side or action.kind != "pick":
            continue
        role_map[revision.action_slot] = tuple(_normalize_role_set(revision.revised_roles))

    flat_roles: list[str] = []
    for slot in sorted(role_map):
        current = role_map[slot]
        if len(current) != 1:
            return ()
        flat_roles.append(current[0])

    return tuple(flat_roles)


def protocol_slot_map(protocol: DraftProtocol) -> dict[int, DraftProtocolStep]:
    return {step.slot: step for step in protocol.steps}


def _normalize_side_mappings(
    mappings: Iterable[DraftSideMapping],
) -> tuple[tuple[DraftSideMapping, ...], list[str]]:
    normalized: list[DraftSideMapping] = []
    errors: list[str] = []

    if not mappings:
        return (), []

    seen_canonical: set[str] = set()
    seen_game: set[str] = set()
    seen_order: set[str] = set()

    for mapping in mappings:
        if mapping.canonical_side not in CANONICAL_DRAFT_SIDES:
            errors.append(f"invalid canonical_side in mapping: {mapping.canonical_side}")
            continue
        if mapping.game_side not in GAME_DRAFT_SIDES:
            errors.append(f"invalid game_side in mapping for side {mapping.canonical_side}: {mapping.game_side}")
            continue
        if mapping.draft_order not in DRAFT_ORDER_POSITIONS:
            errors.append(
                f"invalid draft_order in mapping for side {mapping.canonical_side}: {mapping.draft_order}"
            )
            continue
        if not isinstance(mapping.is_observed, bool) or not isinstance(mapping.is_reconstructed, bool):
            errors.append("mapping is_observed/is_reconstructed must be booleans")
            invalid_timing = True
        else:
            invalid_timing = False
        if mapping.is_observed and mapping.is_reconstructed:
            errors.append(f"mapping side {mapping.canonical_side} cannot be both observed and reconstructed")
            invalid_timing = True

        if not mapping.source_id:
            errors.append(f"mapping source_id is required for side {mapping.canonical_side}")
            invalid_timing = True

        try:
            mapping_observed = parse_rfc3339(mapping.observed_at)
            mapping_available = parse_rfc3339(mapping.available_at)
        except Exception:
            errors.append(f"mapping timestamps invalid for side {mapping.canonical_side}")
            invalid_timing = True
        else:
            if mapping_available > mapping_observed:
                errors.append(f"mapping available_at cannot exceed observed_at for side {mapping.canonical_side}")
                invalid_timing = True

        if mapping.canonical_side in seen_canonical:
            errors.append(f"duplicate canonical_side mapping: {mapping.canonical_side}")
        else:
            seen_canonical.add(mapping.canonical_side)

        if mapping.game_side in seen_game:
            errors.append(f"duplicate game_side mapping: {mapping.game_side}")
        if mapping.draft_order in seen_order:
            errors.append(f"duplicate draft_order mapping: {mapping.draft_order}")
        if not invalid_timing:
            seen_game.add(mapping.game_side)
            seen_order.add(mapping.draft_order)
            normalized.append(mapping)

    if normalized and errors:
        return (), errors

    if seen_canonical and len(seen_canonical) != 1 and len(seen_canonical) != 2:
        errors.append("mapping must contain at most two canonical-side records")
    if seen_canonical and len(seen_game) != len(seen_canonical):
        errors.append("game_side mapping must be bijective with canonical_side mapping")
    if seen_order and len(seen_order) != len(seen_canonical):
        errors.append("draft_order mapping must be bijective with canonical_side mapping")

    return tuple(sorted(normalized, key=lambda mapping: mapping.canonical_side)), errors


def _normalize_role_set(roles: tuple[str, ...]) -> tuple[str, ...]:
    if not roles:
        raise DraftActionError("empty role_set is not allowed")
    normalized = tuple(_normalize_role_name(role) for role in roles)
    if len(set(normalized)) != len(normalized):
        raise DraftActionError("role_set must contain unique roles")
    return normalized


def _normalize_role_name(role: str) -> str:
    try:
        return canonicalize_role(role)
    except ContractError as err:
        raise DraftActionError(f"invalid role {role!r}") from err


def _first(items: list[str] | tuple[str, ...]) -> str | None:
    if not items:
        return None
    return items[0]


def _normalize_role_set_revisions(
    revisions: tuple[DraftRoleSetRevision, ...],
    actions: tuple[DraftAction, ...],
) -> tuple[tuple[DraftRoleSetRevision, ...], list[str]]:
    errors: list[str] = []
    if not revisions:
        return (), []

    action_by_slot = {action.slot: action for action in actions}
    role_state: dict[int, tuple[str, ...]] = {}
    seen_positions: set[int] = set()
    normalized: list[DraftRoleSetRevision] = []
    last_position = -1

    for revision in sorted(revisions, key=lambda item: item.sequence_position):
        if not isinstance(revision.sequence_position, int) or revision.sequence_position < 0:
            errors.append("revision sequence_position must be a non-negative integer")
            continue
        if revision.sequence_position <= last_position:
            errors.append("revision sequence positions must be strictly increasing")
            continue
        if revision.sequence_position in seen_positions:
            errors.append(f"duplicate revision sequence_position: {revision.sequence_position}")
            continue
        seen_positions.add(revision.sequence_position)
        last_position = revision.sequence_position

        action = action_by_slot.get(revision.action_slot)
        if action is None:
            errors.append(f"revision references unknown action slot: {revision.action_slot}")
            continue
        if action.kind != "pick":
            errors.append(f"revision cannot target non-pick action slot: {revision.action_slot}")
            continue
        if action.side != revision.canonical_side:
            errors.append(
                f"revision side mismatch for slot {revision.action_slot}: action side is {action.side}"
            )

        try:
            previous_roles = _normalize_role_set(revision.previous_roles)
            revised_roles = _normalize_role_set(revision.revised_roles)
        except DraftActionError as err:
            errors.append(f"invalid role-set revision roles at slot {revision.action_slot}: {err}")
            continue

        if len(set(previous_roles)) != len(previous_roles):
            errors.append(f"revision at sequence_position {revision.sequence_position} has duplicated previous roles")

        if len(set(revised_roles)) != len(revised_roles):
            errors.append(f"revision at sequence_position {revision.sequence_position} has duplicated revised roles")

        previous_expected = role_state.get(revision.action_slot, _normalize_role_set(action.role_set or ()) )
        if previous_roles != previous_expected:
            errors.append(
                f"revision at sequence_position {revision.sequence_position} has non-monotonic previous_roles for slot {revision.action_slot}"
            )

        if previous_roles == revised_roles:
            errors.append(
                f"revision at sequence_position {revision.sequence_position} has no effective role change"
            )

        try:
            parse_rfc3339(revision.observed_at)
            parse_rfc3339(revision.available_at)
        except Exception as err:
            errors.append(f"invalid revision timestamp for slot {revision.action_slot}: {err}")
            continue

        revision_available = parse_rfc3339(revision.available_at)
        revision_observed = parse_rfc3339(revision.observed_at)
        if revision_available > revision_observed:
            errors.append(f"revision available_at cannot exceed observed_at for slot {revision.action_slot}")
            continue

        role_state[revision.action_slot] = tuple(revised_roles)
        normalized.append(
            DraftRoleSetRevision(
                action_slot=revision.action_slot,
                canonical_side=revision.canonical_side,
                sequence_position=revision.sequence_position,
                previous_roles=tuple(previous_roles),
                revised_roles=tuple(revised_roles),
                reason=revision.reason,
                source_id=revision.source_id,
                observed_at=revision.observed_at,
                available_at=revision.available_at,
                source_record_id=revision.source_record_id,
                source_updated_at=revision.source_updated_at,
                is_observed=revision.is_observed,
                is_reconstructed=revision.is_reconstructed,
            )
        )

    if errors:
        return (), errors

    return tuple(normalized), []


__all__ = [
    "CANONICAL_DRAFT_SIDES",
    "DRAFT_ORDER_POSITIONS",
    "GAME_DRAFT_SIDES",
    "DraftAction",
    "DraftActionError",
    "DraftProtocol",
    "DraftProtocolError",
    "DraftProtocolStep",
    "DraftRoleSetRevision",
    "DraftSideMapping",
    "DraftStateValidation",
    "finalize_role_tuple",
    "validate_actions",
    "validate_protocol",
]
