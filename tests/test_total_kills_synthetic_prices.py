from __future__ import annotations

import json
from pathlib import Path

import pytest

import lol_kills.total_kills_synthetic_prices as prices


MANIFEST = prices.DEFAULT_MANIFEST


def test_frozen_grid_cohort_builds_research_only_synthetic_prices() -> None:
    artifact = prices.build_artifact(MANIFEST, lines=(24.5, 32.5))

    assert artifact["scope"]["historical_replay_authority"] is True
    assert artifact["scope"]["prospective_live_latency_authority"] is False
    assert artifact["feature_selection"]["selected_family"] == "objective_state"
    assert artifact["heldout"]["selected"]["rmse"] < artifact["heldout"]["prior"]["rmse"]
    assert artifact["heldout"]["calibration_residual_cdf"]["status"] == "passed"
    assert artifact["authority"]["betting_classification_authorized"] is False
    assert "external_market_benchmark_missing" in artifact["authority"]["blockers"]
    assert artifact["protocol"]["gold_difference"].startswith("unavailable")

    line = artifact["synthetic_reference_prices"]["20"]["lines"][1]
    assert line["line"] == 32.5
    assert line["price_type"] == "synthetic_no_vig_research_benchmark"
    assert line["under_synthetic_fair_odds"] == pytest.approx(
        1.0 / line["under_probability"]
    )
    assert line["over_synthetic_fair_odds"] == pytest.approx(
        1.0 / line["over_probability"]
    )


def test_integer_or_nonpositive_lines_fail_closed() -> None:
    with pytest.raises(prices.SyntheticPriceError, match="half-kill"):
        prices.build_artifact(MANIFEST, lines=(32.0,))
    with pytest.raises(prices.SyntheticPriceError, match="half-kill"):
        prices.build_artifact(MANIFEST, lines=(0.0,))


def test_manual_calculator_prices_only_exact_validated_checkpoint() -> None:
    artifact = prices.build_artifact(MANIFEST, lines=(32.5,))
    result = prices.price_synthetic_lines(
        artifact,
        league="LCK",
        checkpoint=20,
        current_kills=11,
        total_dragons_now=3,
        total_barons_now=0,
        total_inhibitors_now=0,
        lines=(32.5,),
    )
    assert result["status"] == "synthetic_research_only"
    assert result["lines"][0]["classification"] == "SYNTHETIC_RESEARCH_ONLY"
    assert result["lines"][0]["under_synthetic_fair_odds"] is not None
    assert "external_market_benchmark_missing" in result["blockers"]

    withheld = prices.price_synthetic_lines(
        artifact,
        league="LCK",
        checkpoint=22,
        current_kills=14,
        total_dragons_now=3,
        total_barons_now=0,
        total_inhibitors_now=0,
        lines=(32.5,),
    )
    assert withheld["status"] == "unavailable"
    assert "checkpoint_not_validated:22" in withheld["blockers"]
    assert withheld["lines"][0]["under_probability"] is None


def test_manual_calculator_withholds_failed_or_unevaluated_lines() -> None:
    artifact = prices.build_artifact(MANIFEST, lines=(24.5, 32.5))
    failed = prices.price_synthetic_lines(
        artifact,
        league="LCS",
        checkpoint=10,
        current_kills=1,
        total_dragons_now=1,
        total_barons_now=0,
        total_inhibitors_now=0,
        lines=(24.5,),
    )
    assert failed["status"] == "unavailable"
    assert failed["lines"][0]["classification"] == "WITHHELD"
    assert failed["lines"][0]["under_probability"] is None
    assert "line_calibration_unavailable:24.5" in failed["blockers"]

    unevaluated = prices.price_synthetic_lines(
        artifact,
        league="LCS",
        checkpoint=10,
        current_kills=1,
        total_dragons_now=1,
        total_barons_now=0,
        total_inhibitors_now=0,
        lines=(23.5,),
    )
    assert unevaluated["status"] == "unavailable"
    assert unevaluated["lines"][0]["under_probability"] is None
    assert "line_not_evaluated:23.5" in unevaluated["blockers"]


def test_current_model_comparison_is_labeled_development_only() -> None:
    summary = prices._read_current_model(prices.DEFAULT_CURRENT_MODEL)
    assert summary["status"] == "observed_development_artifact"
    assert summary["authority"].endswith("independent_grid_scope")
    assert summary["selected_families"] == ["league", "gold_difference"]


def test_written_artifact_is_hash_addressed_and_replayable(tmp_path: Path) -> None:
    artifact, path = prices.write_artifact(
        MANIFEST,
        output_root=tmp_path,
        lines=(32.5,),
    )
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    digest_input = dict(loaded)
    digest_input.pop("artifact_sha256")
    assert loaded["artifact_sha256"] == prices._hash(digest_input)
    assert loaded["artifact_sha256"] == artifact["artifact_sha256"]
