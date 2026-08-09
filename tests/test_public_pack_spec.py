import pyarrow as pa

from lol_kills.export import pack_spec
from lol_kills.export.public_pack import _ensure_year_column, _filter_years


def test_public_pack_does_not_pin_unreviewed_draft_artifacts() -> None:
    assert set(pack_spec.PINNED_MODEL_FILES).isdisjoint(pack_spec.WITHHELD_MODEL_FILES)
    assert "draft_wr_calibration.json" not in pack_spec.PINNED_MODEL_FILES
    assert "draft_recommendation.json" not in pack_spec.PINNED_MODEL_FILES


def test_pack_readme_declares_withheld_draft_artifacts() -> None:
    assert "Draft Score calibration" in pack_spec.PACK_README


def test_public_reproduction_contract_cites_only_available_public_inputs() -> None:
    required = set(pack_spec.PUBLIC_REPRODUCTION_REQUIRED_FILES)
    assert "features/major_teams.json" not in required
    assert "studies/grubs/void_grubs_scrap_value_and_contest_rationality.pdf" in required
    assert required.isdisjoint(pack_spec.WITHHELD_PUBLIC_FILES)


def test_public_pack_withholds_draft_context_as_well_as_draft_models() -> None:
    assert "features/draft_context.json" in pack_spec.WITHHELD_PUBLIC_FILES
    assert set(pack_spec.WITHHELD_MODEL_FILES).issubset(
        {path.rsplit("/", 1)[-1] for path in pack_spec.WITHHELD_PUBLIC_FILES}
    )


def test_live_map_overlay_gets_partition_year_from_date() -> None:
    table = pa.table(
        {
            "game_uid": ["g-2025", "g-2026"],
            "date": ["2025-12-31T23:00:00Z", "2026-01-01T01:00:00Z"],
        }
    )

    enriched = _ensure_year_column(table)

    assert enriched["year"].to_pylist() == [2025, 2026]


def test_live_overlay_prefers_normalized_oe_year_when_columns_disagree() -> None:
    table = pa.table(
        {
            "year": [2025, 2025, 2026],
            "oe_year": [2025, 2026, 2027],
        }
    )

    filtered = _filter_years(table, (2025, 2026), ("year", "oe_year"))

    assert filtered.to_pydict() == {
        "year": [2025, 2025],
        "oe_year": [2025, 2026],
    }
