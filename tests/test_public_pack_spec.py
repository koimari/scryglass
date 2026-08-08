from lol_kills.export import pack_spec


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
