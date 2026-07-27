from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.export import pack_spec
from lol_kills.research import grubs_article_publication as article
from lol_kills.research.grubs_action_graph import _article_ladder


def test_current_mechanics_headline_and_parity() -> None:
    document = article.build_article_publication()

    assert document["schema_version"] == "scryglass.grubs.article.v1"
    assert document["mechanics"]["patch"] == "26.11+"
    assert document["mechanics"]["touch_true_damage"] == 256
    assert (
        document["mechanics"]["brief_touch_progress_gold_equivalent"]
        == 34.13
    )
    assert document["mechanics"]["objective_gold_equivalent"] == 124.13
    assert (
        article.GRUB_CASH_GOLD
        + round(
            article.TOUCH_TRUE_DAMAGE
            / article.FIRST_PLATE_HP
            * article.FIRST_PLATE_GOLD,
            2,
        )
        == 124.13
    )
    assert document["p_star"] == pytest.approx(0.582420118348, abs=5e-13)
    assert document["p_star_pct"] == 58.24
    assert document["edge_at_50_pp"] == -1.94

    at_fifty = next(
        row for row in document["curve"] if row["p_win_fight"] == 0.5
    )
    assert at_fifty["model_preference"] == "LEAVE"
    assert at_fifty["edge_contest_minus_leave_pp"] == -1.94


def test_writer_is_deterministic_and_checkable(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    article.write_article_publication(first)
    article.write_article_publication(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")
    loaded = json.loads(first.read_text(encoding="utf-8"))
    article.validate_article_publication(loaded)


def test_schema_and_current_mechanics_drift_fail_closed() -> None:
    document = article.build_article_publication()

    extra = dict(document)
    extra["auxiliary"] = {}
    with pytest.raises(ValueError, match="schema keys differ"):
        article.validate_article_publication(extra)

    stale = json.loads(article.canonical_json(document))
    stale["mechanics"]["objective_gold_equivalent"] = 115.6
    with pytest.raises(ValueError, match="objective_gold_equivalent"):
        article.validate_article_publication(stale)


def test_public_article_contains_no_oe_action_artifact() -> None:
    document = article.build_article_publication()
    payload = article.canonical_json(document).casefold()

    assert "oe_sister" not in payload
    assert "leave_mix" not in payload
    assert "breakeven_p_win_fight" not in payload
    assert "not an identified action threshold" in payload


def test_research_tree_has_one_article_writer() -> None:
    research_root = Path(article.__file__).resolve().parent
    writers = []
    for source in sorted(research_root.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        if (
            "grubs_article_contest_ev.json" in text
            and ("os.replace(" in text or ".write_text(" in text)
        ):
            writers.append(source.name)

    assert writers == ["grubs_article_publication.py"]


def test_internal_action_graph_consumes_current_article_ladder() -> None:
    ladder = _article_ladder()
    parity = next(
        row
        for row in ladder["by_precontest_gold_B_two_wave_leave"]
        if row["B_gold"] == 0
    )

    assert ladder["schema_version"] == article.SCHEMA_VERSION
    assert ladder["mechanics"]["objective_gold_equivalent"] == 124.13
    assert parity["p_star_pct"] == 58.24


def test_public_pack_allows_only_the_canonical_article_json() -> None:
    assert pack_spec.GRUBS_MODEL_FILES == ("grubs_article_contest_ev.json",)
    assert pack_spec.GRUBS_PDF_FILES == ()
    assert "58.24%" in pack_spec.GRUBS_STUDY_NOTE
    assert "identified action policy" in pack_spec.GRUBS_STUDY_NOTE

    exporter = Path(pack_spec.__file__).with_name("public_pack.py")
    exporter_source = exporter.read_text(encoding="utf-8")
    assert "STUDY_NOTE.txt" not in exporter_source
    assert "grubs_decision_numbers.json" not in exporter_source
    assert "missing required grubs article artifact" in exporter_source
    assert (
        "void_grubs_scrap_value_and_contest_rationality.pdf"
        not in exporter_source
    )
