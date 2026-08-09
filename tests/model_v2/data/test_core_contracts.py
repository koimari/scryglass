from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import pytest

from lol_kills.v2.data.competitions import (
    CompetitionTaxonomy,
    CompetitionTaxonomyRow,
    LeaguePatchRecord,
    MapPatchRecord,
    PatchConflictError,
    PatchResolver,
)
from lol_kills.v2.data.common import (
    TimestampError,
    StalenessError,
    enforce_as_of_order,
    enforce_contract_time_axis,
    sha256_canonical_object_hash,
    sha256_raw_bytes_hash,
    validate_freshness_window,
)
from lol_kills.v2.data.identity import IdentityCrosswalkRow, IdentityRegistry, IdentityRegistryError
from lol_kills.v2.provenance.snapshots import CONTRACT_TREE_SHA256
from lol_kills.v2.data.protocols import (
    CANONICAL_DRAFT_SIDES,
    DraftAction,
    DraftProtocol,
    DraftProtocolStep,
    DraftRoleSetRevision,
    DraftSideMapping,
    DraftStateValidation,
    validate_actions,
)
from lol_kills.v2.data.rosters import RosterRegistry, RosterRow, AmbiguousRosterError, RosterUnavailableError
from lol_kills.v2.data.series import SeriesCrosswalkRow, SeriesRegistry, MapRecord, SeriesError
from lol_kills.v2.data.source_tree import normalize_source_tree_path, canonical_source_tree_sha256
from lol_kills.v2.data import parse_rfc3339


def _ts(iso: str) -> str:
    return iso


def _dt(iso: str) -> datetime:
    return parse_rfc3339(iso)


class _NonFiniteDateTime(datetime):
    def timestamp(self) -> float:  # pragma: no cover - synthetic for invalid timestamp path
        raise OverflowError("non-finite")



def test_contract_time_axis_and_hash_primitives_cover_core_rules() -> None:
    base = _dt("2026-07-27T12:00:00Z")

    with pytest.raises(TimestampError):
        enforce_as_of_order(
            as_of=base,
            source_updated_at=_NonFiniteDateTime(2026, 7, 27, tzinfo=timezone.utc),
            observed_at=base,
            available_at=base,
        )

    with pytest.raises(TimestampError):
        enforce_contract_time_axis(
            source_updated_at=base,
            observed_at=base + timedelta(hours=1),
            available_at=base,
            as_of=base,
        )

    with pytest.raises(StalenessError):
        validate_freshness_window(
            available_at=base + timedelta(hours=1),
            as_of=base,
            limit_seconds=60,
        )

    canonical = sha256_canonical_object_hash({"b": 2, "a": 1})
    raw = sha256_raw_bytes_hash(b'{"b":2,"a":1}')
    assert canonical != raw

    assert CONTRACT_TREE_SHA256 == "8748bbe48b273593b09304ac80923f11384de808b835f6e83e97c6fef48661dd"


def test_source_tree_path_semantics_reject_traversal_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path
    (root / "alpha.txt").write_text("alpha", encoding="utf-8")
    (root / "beta.txt").write_text("beta", encoding="utf-8")

    with pytest.raises(ValueError):
        normalize_source_tree_path("../alpha.txt")
    with pytest.raises(ValueError):
        normalize_source_tree_path("./alpha.txt")
    with pytest.raises(ValueError):
        normalize_source_tree_path("/tmp/alpha.txt")
    with pytest.raises(ValueError):
        normalize_source_tree_path("alpha\\beta.txt")
    with pytest.raises(ValueError):
        canonical_source_tree_sha256(root, ("alpha.txt", "alpha.txt"))

    symlink = root / "alpha-link.txt"
    symlink.symlink_to(root / "alpha.txt")
    with pytest.raises(ValueError):
        canonical_source_tree_sha256(root, ("alpha-link.txt",))


def test_source_tree_raw_and_object_hash_divergence() -> None:
    payload = {"b": 2, "a": 1}
    assert sha256_canonical_object_hash(payload) != sha256_raw_bytes_hash(b'{"a":1, "b": 2}')


def _identity_row(
    *,
    row_id: str,
    canonical_id: str,
    alias: str,
    observed_at: str = "2026-07-01T00:00:00Z",
    available_at: str,
    precedence: int = 0,
) -> IdentityCrosswalkRow:
    effective_at = "2026-07-01T00:00:00Z"
    return IdentityCrosswalkRow(
        row_id=row_id,
        entity_type="player",
        canonical_id=canonical_id,
        canonical_name=canonical_id,
        source_name="official",
        source_id="riot",
        source_snapshot_id="snapshot-v1",
        source_snapshot_row_id="row-v1",
        source_snapshot_content_sha256="a" * 64,
        source_record_id="record-v1",
        alias=alias,
        effective_from="2026-07-01T00:00:00Z",
        effective_to=None,
        precedence=precedence,
        observed_at=observed_at,
        source_updated_at=effective_at,
        available_at=available_at,
    )


def test_identity_collision_and_availability_safe_lookup() -> None:
    registry = IdentityRegistry.from_rows(
        (
            _identity_row(
                row_id="id-a",
                canonical_id="riot:player:1",
                alias="alias-1",
                observed_at="2026-07-05T00:00:00Z",
                available_at="2026-07-05T00:00:00Z",
                precedence=1,
            ),
            _identity_row(
                row_id="id-b",
                canonical_id="riot:player:2",
                alias="alias-1",
                observed_at="2026-07-15T00:00:00Z",
                available_at="2026-07-15T00:00:00Z",
                precedence=1,
            ),
        )
    )

    assert registry.lookup("missing", as_of=_dt("2026-07-20T00:00:00Z")) is None
    assert registry.lookup("alias-1", as_of=_dt("2026-07-07T00:00:00Z")) == "riot:player:1"
    assert registry.lookup("alias-1", as_of=_dt("2026-07-16T00:00:00Z"), fail_on_collision=False) is None
    with pytest.raises(IdentityRegistryError):
        _ = registry.lookup("alias-1", as_of=_dt("2026-07-16T00:00:00Z"), fail_on_collision=True)

    collisions = registry.audit_collisions(as_of=_dt("2026-07-16T00:00:00Z"))
    assert len(collisions) == 1
    assert collisions[0].alias == "alias-1"


def _roster_row(
    *,
    row_id: str,
    roster_id: str,
    organization_id: str,
    role: str,
    player_id: str,
    organization_name: str = "Team X",
    is_substitute: bool = False,
) -> RosterRow:
    return RosterRow(
        row_id=row_id,
        roster_id=roster_id,
        organization_id=organization_id,
        organization_name=organization_name,
        role=role,
        player_id=player_id,
        player_name=f"Player {player_id}",
        source_id="official",
        source_name="official",
        source_record_id="record-1",
        source_snapshot_id="snapshot-1",
        source_snapshot_row_id="row-1",
        source_snapshot_content_sha256="a" * 64,
        effective_from="2026-07-01T00:00:00Z",
        effective_to=None,
        precedence=1,
        source_updated_at="2026-07-01T00:00:00Z",
        observed_at="2026-07-01T00:00:00Z",
        available_at="2026-07-01T00:00:00Z",
        is_substitute=is_substitute,
        is_provisional=False,
    )


def test_roster_exact_five_main_roles_and_no_substitute_rewrite() -> None:
    registry = RosterRegistry(
        rows=(
            _roster_row(
                row_id="r-a",
                roster_id="roster-main",
                organization_id="org-1",
                role="top",
                player_id="p-top",
            ),
            _roster_row(
                row_id="r-b",
                roster_id="roster-main",
                organization_id="org-1",
                role="jungle",
                player_id="p-jungle",
            ),
            _roster_row(
                row_id="r-c",
                roster_id="roster-main",
                organization_id="org-1",
                role="mid",
                player_id="p-mid",
            ),
            _roster_row(
                row_id="r-d",
                roster_id="roster-main",
                organization_id="org-1",
                role="bot",
                player_id="p-bot",
            ),
            _roster_row(
                row_id="r-e",
                roster_id="roster-main",
                organization_id="org-1",
                role="support",
                player_id="p-support",
            ),
            _roster_row(
                row_id="r-sub",
                roster_id="roster-sub",
                organization_id="org-1",
                role="mid",
                player_id="p-sub-mid",
                is_substitute=True,
            ),
        )
    )

    resolved = registry.resolve_exact_roster("org-1", as_of=_dt("2026-07-20T00:00:00Z"))
    assert resolved.status == "ok"
    assert resolved.roster is not None
    assert len(set(resolved.roster)) == 5


def test_roster_ambiguity_or_only_substitute_fails_closed() -> None:
    registry = RosterRegistry(
        rows=(
            _roster_row(
                row_id="r-a-1",
                roster_id="roster-main",
                organization_id="org-2",
                role="top",
                player_id="p-top-1",
            ),
            _roster_row(
                row_id="r-a-2",
                roster_id="roster-main",
                organization_id="org-2",
                role="top",
                player_id="p-top-2",
            ),
            _roster_row(
                row_id="r-b",
                roster_id="roster-main",
                organization_id="org-2",
                role="jungle",
                player_id="p-jungle",
            ),
            _roster_row(
                row_id="r-c",
                roster_id="roster-main",
                organization_id="org-2",
                role="mid",
                player_id="p-mid",
            ),
            _roster_row(
                row_id="r-d",
                roster_id="roster-main",
                organization_id="org-2",
                role="bot",
                player_id="p-bot",
            ),
            _roster_row(
                row_id="r-e",
                roster_id="roster-main",
                organization_id="org-2",
                role="support",
                player_id="p-support",
            ),
        )
    )

    with pytest.raises(AmbiguousRosterError):
        registry.resolve_exact_roster("org-2", as_of=_dt("2026-07-20T00:00:00Z"))

    empty_roster = RosterRegistry(
        rows=(
            _roster_row(
                row_id="s-a",
                roster_id="roster-sub-only",
                organization_id="org-3",
                role="top",
                player_id="s-top",
                is_substitute=True,
            ),
        )
    )
    with pytest.raises(RosterUnavailableError):
        empty_roster.resolve_exact_roster("org-3", as_of=_dt("2026-07-20T00:00:00Z"))


def _competition_taxonomy_row(
    *,
    row_id: str,
    league_id: str,
    tier: str,
    structurally_globally_eligible: bool,
    competition_qualification: str,
    precedence: int,
    observed_at: str,
) -> CompetitionTaxonomyRow:
    return CompetitionTaxonomyRow(
        row_id=row_id,
        league_id=league_id,
        competition_tier=tier,
        structurally_globally_eligible=structurally_globally_eligible,
        source_id="manual",
        source_name="manual",
        source_record_id="src-record",
        source_snapshot_id="src-snap",
        source_snapshot_row_id="src-snap-row",
        source_snapshot_content_sha256="a" * 64,
        effective_from="2026-06-01T00:00:00Z",
        effective_to=None,
        internationally_connectable=True,
        qualification_rule_id=competition_qualification,
        precedence=precedence,
        observed_at=observed_at,
        source_updated_at="2026-06-01T00:00:00Z",
        available_at=observed_at,
        taxonomy_version="v1",
    )


def test_competition_taxonomy_conflict_and_patch_precedence_rules() -> None:
    taxonomy = CompetitionTaxonomy(
        version="v1",
        rows=(
            _competition_taxonomy_row(
                row_id="tax-1",
                league_id="LPL",
                tier="tier1",
                structurally_globally_eligible=True,
                competition_qualification="qual-tier1",
                precedence=2,
                observed_at="2026-07-27T12:00:00Z",
            ),
            _competition_taxonomy_row(
                row_id="tax-2",
                league_id="LPL",
                tier="tier1",
                structurally_globally_eligible=False,
                competition_qualification="qual-tier1-alt",
                precedence=2,
                observed_at="2026-07-27T12:00:00Z",
            ),
        ),
    )

    resolver = PatchResolver(
        taxonomy=taxonomy,
        patch_records=(),
        map_records=(),
    )
    conflict = resolver.resolve_current_patch("LPL", as_of=_dt("2026-07-27T13:00:00Z"))
    assert conflict.status == "conflict"
    assert "conflicting_tier_profile" in conflict.reason

    resolver = PatchResolver(
        taxonomy=CompetitionTaxonomy(
            version="v1",
            rows=(
                _competition_taxonomy_row(
                    row_id="tax-1",
                    league_id="LCK",
                    tier="tier1",
                    structurally_globally_eligible=True,
                    competition_qualification="qual-tier1",
                    precedence=2,
                    observed_at="2026-07-27T12:00:00Z",
                ),
            ),
        ),
        patch_records=(
            LeaguePatchRecord(
                row_id="patch-low",
                league_id="LCK",
                patch_id="26.13",
                source_id="manual",
                source_name="manual",
                source_record_id="row-lp-1",
                source_snapshot_id="src-snap",
                source_snapshot_row_id="src-snap-row",
                source_snapshot_content_sha256="a" * 64,
                source_updated_at="2026-07-27T11:00:00Z",
                observed_at="2026-07-27T11:05:00Z",
                available_at="2026-07-27T11:05:00Z",
                announced_at="2026-07-27T11:00:00Z",
                is_authoritative=True,
                effective_from="2026-07-27T00:00:00Z",
                effective_to=None,
                precedence=1,
            ),
            LeaguePatchRecord(
                row_id="patch-high",
                league_id="LCK",
                patch_id="26.14",
                source_id="manual",
                source_name="manual",
                source_record_id="row-lp-2",
                source_snapshot_id="src-snap",
                source_snapshot_row_id="src-snap-row",
                source_snapshot_content_sha256="a" * 64,
                source_updated_at="2026-07-27T12:00:00Z",
                observed_at="2026-07-27T12:05:00Z",
                available_at="2026-07-27T12:05:00Z",
                announced_at="2026-07-27T12:00:00Z",
                is_authoritative=True,
                effective_from="2026-07-27T00:00:00Z",
                effective_to=None,
                precedence=3,
            ),
        ),
        map_records=(),
    )
    resolved = resolver.resolve_current_patch("LCK", as_of=_dt("2026-07-27T13:00:00Z"))
    assert resolved.status == "authoritative"
    assert resolved.patch_id == "26.14"


def _map_row_for_series(
    *,
    map_id: str,
    participants=("team-a", "team-b"),
    league_id: str = "LPL",
    tournament_id: str = "split-a",
    result: int | None = 1,
    season_id: str = "scryglass:season:lpl-2026",
    calendar_year: int = 2026,
    event_start: str | None = "2026-07-26T17:00:00Z",
    event_end: str | None = "2026-07-26T19:00:00Z",
) -> MapRecord:
    return MapRecord(
        map_id=map_id,
        league_id=league_id,
        tournament_id=tournament_id,
        participants=participants,
        source_series_id=None,
        source_id="riot",
        source_record_id="row-" + map_id,
        source_snapshot_id="snapshot",
        source_snapshot_row_id="snapshot-row",
        source_snapshot_content_sha256="a" * 64,
        source_updated_at="2026-07-26T20:00:00Z",
        observed_at="2026-07-26T20:00:00Z",
        available_at="2026-07-26T20:00:00Z",
        scheduled_start=None,
        event_start=event_start,
        event_end=event_end,
        result=result,
        patch_id="26.14",
        source_updated_by="official",
        season_id=season_id,
        calendar_year=calendar_year,
    )


def test_series_unresolved_and_incomplete_outcome_stays_unresolved() -> None:
    registry = SeriesRegistry(rows=(
        SeriesCrosswalkRow(
            row_id="cw-1",
            league_id="LPL",
            tournament_id="split-a",
            source_id="riot",
            source_record_id="r1",
            source_snapshot_id="snap",
            source_snapshot_row_id="snap-row",
            source_snapshot_content_sha256="a" * 64,
            source_series_id="s-id-1",
            series_id="S-1",
            participants=("team-a", "team-b"),
            effective_from="2026-07-20T00:00:00Z",
            effective_to=None,
            precedence=1,
            observed_at="2026-07-20T00:00:00Z",
            available_at="2026-07-20T00:00:00Z",
            source_updated_at="2026-07-20T00:00:00Z",
        ),
    ))

    unresolved = registry.resolve(
        _map_row_for_series(map_id="m-1", league_id="LCS"),
        as_of=_dt("2026-07-27T20:00:00Z"),
    )
    assert unresolved.series_id is None
    assert unresolved.resolution == "unresolved"

    with pytest.raises(Exception):
        # unresolved maps are excluded from primary resolved inference
        from lol_kills.v2.data.series import require_series_resolved_for_primary

        require_series_resolved_for_primary((unresolved,))

    no_result = registry.resolve(
        _map_row_for_series(
            map_id="m-2",
            league_id="LPL",
            participants=("team-a", "team-b"),
            result=None,
        ),
        as_of=_dt("2026-07-27T20:00:00Z"),
    )
    assert no_result.series_id == "S-1"


def _draft_protocol() -> DraftProtocol:
    return DraftProtocol(
        protocol_id="p-5v5",
        steps=tuple(
            DraftProtocolStep(slot=i + 1, kind="pick", side=CANONICAL_DRAFT_SIDES[i % 2])
            for i in range(10)
        ),
    )


def _draft_actions() -> tuple[DraftAction, ...]:
    roles = ["top", "jungle", "mid", "bot", "support", "top", "jungle", "mid", "bot", "support"]
    return tuple(
        DraftAction(
            slot=slot + 1,
            kind="pick",
            side=CANONICAL_DRAFT_SIDES[slot % 2],
            champion_id=f"riot:champion:{10 + slot}",
            role_set=(role,),
        )
        for slot, role in enumerate(roles)
    )


def _draft_mappings() -> tuple[DraftSideMapping, ...]:
    return (
        DraftSideMapping(
            canonical_side="A",
            game_side="blue",
            draft_order="first",
            source_id="provider:1",
            observed_at="2026-07-27T14:00:00Z",
            available_at="2026-07-27T14:00:00Z",
            source_record_id="r1",
            source_updated_at="2026-07-27T14:00:00Z",
        ),
        DraftSideMapping(
            canonical_side="B",
            game_side="red",
            draft_order="second",
            source_id="provider:1",
            observed_at="2026-07-27T14:00:00Z",
            available_at="2026-07-27T14:00:00Z",
            source_record_id="r2",
            source_updated_at="2026-07-27T14:00:00Z",
        ),
    )


def test_draft_protocol_positive_terminal_validation_and_role_revision_history() -> None:
    draft_validation: DraftStateValidation = validate_actions(
        protocol=_draft_protocol(),
        actions=_draft_actions(),
        side_mappings=_draft_mappings(),
        role_set_revisions=(
            DraftRoleSetRevision(
                action_slot=1,
                canonical_side="A",
                sequence_position=0,
                previous_roles=("top",),
                revised_roles=("top", "mid"),
                reason="role_resolution",
                source_id="provider:1",
                observed_at="2026-07-27T14:05:00Z",
                available_at="2026-07-27T14:05:00Z",
            ),
            DraftRoleSetRevision(
                action_slot=1,
                canonical_side="A",
                sequence_position=1,
                previous_roles=("top", "mid"),
                revised_roles=("top",),
                reason="role_resolution",
                source_id="provider:1",
                observed_at="2026-07-27T14:10:00Z",
                available_at="2026-07-27T14:10:00Z",
            ),
        ),
        require_terminal=True,
    )
    assert draft_validation.is_valid
    assert draft_validation.is_terminal
    assert draft_validation.revision_count == 2


def test_draft_protocol_rejects_action_reorder_and_champion_conflicts() -> None:
    protocol = _draft_protocol()

    out_of_order = (
        DraftAction(2, "pick", "B", "riot:champion:11", ("jungle",)),
        DraftAction(1, "pick", "A", "riot:champion:11", ("top",)),
    )

    result = validate_actions(protocol, out_of_order)
    assert result.is_valid is False
    assert "action sequence must follow protocol slot order" in result.errors
    assert "riot:champion:11" in result.champion_conflicts[0]


def test_draft_side_game_and_order_mappings_cannot_conflate() -> None:
    protocol = _draft_protocol()
    actions = _draft_actions()
    bad_mappings = (
        DraftSideMapping(
            canonical_side="A",
            game_side="blue",
            draft_order="first",
            source_id="provider:1",
            observed_at="2026-07-27T14:00:00Z",
            available_at="2026-07-27T14:00:00Z",
        ),
        DraftSideMapping(
            canonical_side="B",
            game_side="blue",
            draft_order="first",
            source_id="provider:1",
            observed_at="2026-07-27T14:00:00Z",
            available_at="2026-07-27T14:00:00Z",
        ),
    )
    result = validate_actions(protocol, actions, side_mappings=bad_mappings, require_terminal=True)
    assert result.is_valid is False
    assert any("duplicate game_side" in err for err in result.errors)
    assert any("duplicate draft_order" in err for err in result.errors)


def test_series_time_and_calendar_rules_are_enforced() -> None:
    registry = SeriesRegistry()
    with pytest.raises(SeriesError):
        registry.resolve(
            _map_row_for_series(
                map_id="bad-calendar",
                participants=("same", "same"),
            ),
            as_of=_dt("2026-07-27T20:00:00Z"),
        )
    with pytest.raises(SeriesError):
        registry.resolve(
            _map_row_for_series(
                map_id="bad-cross",
                season_id="2026",
                calendar_year=2026,
            ),
            as_of=_dt("2026-07-27T20:00:00Z"),
        )

def test_series_record_resolution_rejects_future_scheduled_start() -> None:
    registry = SeriesRegistry()
    record = _map_row_for_series(
        map_id="fut",
        event_start="2026-07-30T17:00:00Z",
    )
    with pytest.raises(SeriesError):
        registry.resolve(record, as_of=_dt("2026-07-29T00:00:00Z"))
