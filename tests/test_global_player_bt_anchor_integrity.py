"""Integrity of the global BT performance anchor.

One test module per confirmed P1 finding on the anchor:

* P1-1 the anchor must not be inert in the release path, and a release that
  anchors zero players must fail loudly instead of publishing teammate ties;
* P1-2 a single malformed statistic must not dominate the composite;
* P1-3 baselines must be built from strictly earlier maps only;
* P1-4 a baseline with fewer than 20 prior observations must stay neutral and
  be counted.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from lol_kills.export.public_pack import (
    PLAYER_CONTRIBUTION_COLUMNS,
    PublicPlayerAnchorError,
    require_player_performance_anchor,
)
from lol_kills.ratings import player_elo
from lol_kills.ratings.global_player_bt import (
    ANCHOR_METRIC_Z_CLIP,
    ANCHOR_MIN_BASELINE_OBS,
    GlobalPlayerRatingError,
    PrefixBaselineCache,
    PERFORMANCE_ANCHOR_SOURCE_COLUMNS,
    _contribution_metrics,
    _prior_baseline_z,
    _role_normalized_composite,
    fit_global_player_bt,
)

from tests.test_global_player_bt import (
    ROLES,
    _config,
    _fixture,
    _locked_roster_games,
    _roster_profiles,
    _with_metrics,
)


# --------------------------------------------------------------------------
# P1-1  the anchor must not be inert, and an unanchored release must fail
# --------------------------------------------------------------------------


def test_public_pack_projects_every_anchor_source_column() -> None:
    """The release projection must carry every column the anchor reads.

    This is the defect itself: `player_rating_columns` selected identity and
    outcome columns only, so `_contribution_metrics` saw all-NaN and the
    published ladder kept every teammate tie.
    """

    missing = set(PERFORMANCE_ANCHOR_SOURCE_COLUMNS) - set(PLAYER_CONTRIBUTION_COLUMNS)
    assert missing == set(), f"release projection drops anchor inputs: {sorted(missing)}"


def test_release_grade_fit_refuses_a_ladder_that_anchors_no_player() -> None:
    """A validated fit that anchors nobody must fail, never publish."""

    maps, players = _fixture(_locked_roster_games())
    # Identity and outcome only, which is exactly what the broken projection
    # handed to the fit.
    bare = players[["gameid", "date", "side", "position", "playername"]].copy()

    with pytest.raises(GlobalPlayerRatingError, match="anchored 0 of"):
        fit_global_player_bt(
            maps, bare, _config(minimum_holdout_gain=-1.0), validate=True
        )


def test_release_check_reports_the_missing_contribution_columns() -> None:
    maps, players = _fixture(_locked_roster_games())
    bare = players[["gameid", "date", "side", "position", "playername"]].copy()

    with pytest.raises(GlobalPlayerRatingError) as raised:
        fit_global_player_bt(
            maps, bare, _config(minimum_holdout_gain=-1.0), validate=True
        )
    message = str(raised.value)
    for column in PERFORMANCE_ANCHOR_SOURCE_COLUMNS:
        assert column in message


def test_disabled_anchor_is_not_caught_by_the_release_check() -> None:
    """`performance_anchor_enabled = False` must still reproduce old behaviour."""

    maps, players = _fixture(_locked_roster_games())
    bare = players[["gameid", "date", "side", "position", "playername"]].copy()

    snapshot, meta = fit_global_player_bt(
        maps,
        bare,
        _config(minimum_holdout_gain=-1.0, performance_anchor_enabled=False),
        validate=True,
    )
    assert meta["performance_anchor"]["enabled"] is False
    assert "global_performance_anchor_logit" not in snapshot.columns


def _meta_file(tmp_path: Path, anchor: dict[str, object] | None) -> Path:
    path = tmp_path / "player_ratings_meta.json"
    payload: dict[str, object] = {"global_rating": {}}
    if anchor is not None:
        payload["global_rating"] = {"performance_anchor": anchor}
    path.write_text(json.dumps(payload))
    return path


def test_public_pack_release_check_refuses_an_unanchored_ladder(tmp_path: Path) -> None:
    path = _meta_file(tmp_path, {"enabled": True, "players_anchored": 0})

    with pytest.raises(PublicPlayerAnchorError, match="anchored 0 players"):
        require_player_performance_anchor(path)


def test_public_pack_release_check_requires_anchor_evidence(tmp_path: Path) -> None:
    with pytest.raises(PublicPlayerAnchorError, match="no performance anchor"):
        require_player_performance_anchor(_meta_file(tmp_path, None))

    with pytest.raises(PublicPlayerAnchorError, match="meta is missing"):
        require_player_performance_anchor(tmp_path / "absent.json")


def test_public_pack_release_check_accepts_an_anchored_ladder(tmp_path: Path) -> None:
    path = _meta_file(tmp_path, {"enabled": True, "players_anchored": 3797})
    assert require_player_performance_anchor(path)["players_anchored"] == 3797

    off = _meta_file(tmp_path, {"enabled": False, "players_anchored": 0})
    assert require_player_performance_anchor(off)["enabled"] is False


# --------------------------------------------------------------------------
# P1-3  baselines are built from strictly earlier maps only
# --------------------------------------------------------------------------


def test_prior_baseline_z_matches_the_elo_implementation() -> None:
    """The mirror must not drift from `player_elo._prior_baseline_z`."""

    rng = np.random.default_rng(20260818)
    size = 400
    values = pd.Series(rng.normal(size=size))
    values[rng.integers(0, size, 40)] = np.nan
    group = pd.Series(rng.choice(["top", "jng", "mid"], size=size))
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 60, size), unit="D")
    )

    mine, prior_obs = _prior_baseline_z(values, group, date, ANCHOR_MIN_BASELINE_OBS)
    theirs = player_elo._prior_baseline_z(
        values, group, date, ANCHOR_MIN_BASELINE_OBS
    )

    pd.testing.assert_series_equal(mine, theirs, check_names=False)
    assert mine.notna().any(), "fixture must actually exercise the usable branch"
    # The extra return is only the prior count the floor is measured against.
    assert (prior_obs[mine.notna()] >= ANCHOR_MIN_BASELINE_OBS).all()


def _robust_z(sample: np.ndarray, value: float) -> float | None:
    """Independent median/MAD reference, written straight from the definition."""

    if len(sample) == 0:
        return None
    centre = float(np.median(sample))
    spread = 1.4826 * float(np.median(np.abs(sample - centre)))
    if spread <= 0.0:
        low, high = np.quantile(sample, (0.25, 0.75))
        spread = float(high - low) / 1.349
    if spread <= 0.0:
        return None
    return (value - centre) / spread


def test_a_map_never_contributes_to_its_own_baseline() -> None:
    """Direct proof: each z equals the z against strictly earlier rows only."""

    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(40), unit="D")
    values = pd.Series(np.arange(1.0, 41.0))
    group = pd.Series(["mid"] * 40)
    date = pd.Series(dates)

    z, prior_obs = _prior_baseline_z(values, group, date, min_obs=2)

    for position in range(40):
        earlier = values.iloc[:position].to_numpy(dtype=float)
        assert prior_obs.iloc[position] == float(position)
        expected = (
            None
            if position < 2
            else _robust_z(earlier, float(values.iloc[position]))
        )
        if expected is None:
            assert pd.isna(z.iloc[position])
            continue
        assert z.iloc[position] == pytest.approx(expected, rel=1e-12, abs=1e-12)
        # The row's own value is excluded: including it would give a different
        # number, so this is a real exclusion and not a coincidence.
        including_self = values.iloc[: position + 1].to_numpy(dtype=float)
        with_self = _robust_z(including_self, float(values.iloc[position]))
        assert z.iloc[position] != pytest.approx(with_self, rel=1e-9, abs=1e-9)


def test_expanding_median_is_bit_identical_to_numpy_median() -> None:
    """`_robust_block_baseline` reads location from pandas' expanding median.

    That is only legitimate if it is exactly `np.median` of the same prefix,
    including under heavy ties and an all-constant pool, so pin it.
    """

    rng = np.random.default_rng(20260818)
    for sample in (
        rng.normal(size=400),
        np.round(rng.normal(size=400), 1),
        np.zeros(400),
        rng.choice([0.0, 0.0, 0.0, 1.0, 2.5], size=400),
    ):
        running = pd.Series(sample).expanding().median().to_numpy()
        reference = np.array([np.median(sample[: n + 1]) for n in range(len(sample))])
        assert (running == reference).all()


def test_rows_sharing_a_timestamp_cannot_see_each_other() -> None:
    """Two maps on the same date form one block with a shared prior baseline."""

    values = pd.Series(np.arange(1.0, 41.0))
    group = pd.Series(["mid"] * 40)
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(40) // 2, unit="D")
    )

    z, prior_obs = _prior_baseline_z(values, group, date, min_obs=2)

    for position in range(0, 40, 2):
        # Both rows in a block see the same prior count, which excludes the
        # block itself.
        assert prior_obs.iloc[position] == prior_obs.iloc[position + 1] == float(position)


def test_shared_prefix_cache_is_exact_for_full_and_historical_prefixes() -> None:
    """The cache reuses reference outputs for an immutable date prefix."""

    size = 40
    values = pd.Series(np.arange(1.0, size + 1))
    group = pd.Series(["mid"] * size)
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size), unit="D")
    )
    row_key = pd.Series(
        [("game", "Blue", f"player-{index}", "mid") for index in range(size)],
        dtype=object,
    )
    cache = PrefixBaselineCache(source_identity="fixture")

    reference, reference_prior = _prior_baseline_z(values, group, date, min_obs=2)
    first, first_prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    second, second_prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )

    pd.testing.assert_series_equal(first, reference, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(first_prior, reference_prior, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(second, reference, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(second_prior, reference_prior, check_names=False, check_exact=True)
    assert cache.stores == 1
    assert cache.hits == 1

    prefix = slice(0, 24)
    prefix_values = values.iloc[prefix].reset_index(drop=True)
    prefix_group = group.iloc[prefix].reset_index(drop=True)
    prefix_date = date.iloc[prefix].reset_index(drop=True)
    prefix_keys = row_key.iloc[prefix].reset_index(drop=True)
    prefix_cached, prefix_prior = _prior_baseline_z(
        prefix_values,
        prefix_group,
        prefix_date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=prefix_keys,
    )
    prefix_reference, prefix_reference_prior = _prior_baseline_z(
        prefix_values, prefix_group, prefix_date, min_obs=2
    )
    pd.testing.assert_series_equal(
        prefix_cached, prefix_reference, check_names=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        prefix_prior,
        prefix_reference_prior,
        check_names=False,
        check_exact=True,
    )
    assert cache.hits == 2


def test_shared_prefix_cache_keeps_same_timestamp_rows_and_rejects_drift() -> None:
    """Partial final blocks are safe; gaps and changed source rows miss closed."""

    size = 40
    values = pd.Series(np.arange(1.0, size + 1))
    group = pd.Series(["mid"] * size)
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size) // 2, unit="D")
    )
    row_key = pd.Series(
        [("game", "Blue", f"player-{index}", "mid") for index in range(size)],
        dtype=object,
    )
    cache = PrefixBaselineCache()
    _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )

    partial = slice(0, 21)
    partial_values = values.iloc[partial].reset_index(drop=True)
    partial_group = group.iloc[partial].reset_index(drop=True)
    partial_date = date.iloc[partial].reset_index(drop=True)
    partial_keys = row_key.iloc[partial].reset_index(drop=True)
    cached, cached_prior = _prior_baseline_z(
        partial_values,
        partial_group,
        partial_date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=partial_keys,
    )
    reference, reference_prior = _prior_baseline_z(
        partial_values, partial_group, partial_date, min_obs=2
    )
    pd.testing.assert_series_equal(cached, reference, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(
        cached_prior, reference_prior, check_names=False, check_exact=True
    )
    assert cache.hits == 1

    # Omitting an earlier row while retaining a later date is outside the
    # proven prefix contract. The caller computes the reference path again.
    gap = [0, 2, 3, 4, 5, 6]
    before_misses = cache.misses
    gap_values = values.iloc[gap].reset_index(drop=True)
    gap_group = group.iloc[gap].reset_index(drop=True)
    gap_date = date.iloc[gap].reset_index(drop=True)
    gap_keys = row_key.iloc[gap].reset_index(drop=True)
    gap_result, gap_prior = _prior_baseline_z(
        gap_values,
        gap_group,
        gap_date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=gap_keys,
    )
    gap_reference, gap_reference_prior = _prior_baseline_z(
        gap_values, gap_group, gap_date, min_obs=2
    )
    pd.testing.assert_series_equal(
        gap_result, gap_reference, check_names=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        gap_prior, gap_reference_prior, check_names=False, check_exact=True
    )
    assert cache.misses > before_misses
    assert cache.last_miss_reason == "source_drift_or_non_prefix"

    # A changed earlier value has the same safe result. The old entry never
    # supplies stale output for the changed source.
    changed = values.copy()
    changed.iloc[3] = 900.0
    changed_result, changed_prior = _prior_baseline_z(
        changed,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    changed_reference, changed_reference_prior = _prior_baseline_z(
        changed, group, date, min_obs=2
    )
    pd.testing.assert_series_equal(
        changed_result, changed_reference, check_names=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        changed_prior,
        changed_reference_prior,
        check_names=False,
        check_exact=True,
    )


def test_shared_prefix_cache_is_usable_by_both_baseline_implementations() -> None:
    """The player and global wrappers consume one exact cached result."""

    from lol_kills.ratings.player_elo import _prior_baseline_z as elo_prior_baseline_z

    size = 40
    values = pd.Series(np.arange(1.0, size + 1))
    group = pd.Series(["mid"] * size)
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size), unit="D")
    )
    row_key = pd.Series(
        [("game", "Blue", f"player-{index}", "mid") for index in range(size)],
        dtype=object,
    )
    cache = PrefixBaselineCache()
    global_z, global_prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    player_z = elo_prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )

    pd.testing.assert_series_equal(global_z, player_z, check_names=False, check_exact=True)
    assert global_prior.notna().all()
    assert cache.stores == 1
    assert cache.hits == 1


def test_persistent_cache_round_trip_extends_only_new_timestamp_blocks(tmp_path: Path) -> None:
    """A changed source watermark can extend an exact cached prefix."""

    values, group, date, row_key = _persistent_cache_fixture()
    path = tmp_path / "prefix-baseline"
    first = PrefixBaselineCache(storage_path=path, source_identity="source-v1")
    old = slice(0, 20)
    _prior_baseline_z(
        values.iloc[old].reset_index(drop=True),
        group.iloc[old].reset_index(drop=True),
        date.iloc[old].reset_index(drop=True),
        min_obs=2,
        baseline_cache=first,
        metric_key="cs_per_min",
        row_key=row_key.iloc[old].reset_index(drop=True),
    )
    first.flush()

    second = PrefixBaselineCache(storage_path=path, source_identity="source-v2")
    cached, cached_prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=second,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    reference, reference_prior = _prior_baseline_z(values, group, date, min_obs=2)
    pd.testing.assert_series_equal(cached, reference, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(
        cached_prior, reference_prior, check_names=False, check_exact=True
    )
    assert second.source_identity_changed is True
    assert second.hits == 1
    assert second.stores == 1
    assert second.invalidated is False
    second.flush()

    reloaded = PrefixBaselineCache(storage_path=path, source_identity="source-v2")
    prefix, prefix_prior = _prior_baseline_z(
        values.iloc[old].reset_index(drop=True),
        group.iloc[old].reset_index(drop=True),
        date.iloc[old].reset_index(drop=True),
        min_obs=2,
        baseline_cache=reloaded,
        metric_key="cs_per_min",
        row_key=row_key.iloc[old].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        prefix,
        reference.iloc[old].reset_index(drop=True),
        check_names=False,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        prefix_prior,
        reference_prior.iloc[old].reset_index(drop=True),
        check_names=False,
        check_exact=True,
    )
    assert reloaded.hits == 1


def test_persistent_cache_serialization_integrity_fails_closed(tmp_path: Path) -> None:
    """A tampered payload is rejected before any cached value is served."""

    values, group, date, row_key = _persistent_cache_fixture()
    path = tmp_path / "prefix-baseline"
    cache = PrefixBaselineCache(storage_path=path, source_identity="source-v1")
    _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    cache.flush()
    payload = path.with_suffix(".npz")
    raw = payload.read_bytes()
    payload.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))

    rejected = PrefixBaselineCache(storage_path=path, source_identity="source-v1")
    assert rejected.invalidated is True
    assert rejected.invalidated_reason is not None
    assert rejected._entries == {}
    fresh, _prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=rejected,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    reference, _reference_prior = _prior_baseline_z(values, group, date, min_obs=2)
    pd.testing.assert_series_equal(fresh, reference, check_names=False, check_exact=True)


def test_persistent_cache_invalidates_corrections_deletions_and_schema_drift(
    tmp_path: Path,
) -> None:
    """Corrections, deletions, and schema changes clear the old source."""

    values, group, date, row_key = _persistent_cache_fixture()
    path = tmp_path / "prefix-baseline"
    seed = PrefixBaselineCache(storage_path=path, source_identity="source-v1")
    _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=seed,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    seed.flush()

    corrected = values.copy()
    corrected.iloc[3] = 900.0
    corrected_cache = PrefixBaselineCache(storage_path=path, source_identity="source-v2")
    _prior_baseline_z(
        corrected,
        group,
        date,
        min_obs=2,
        baseline_cache=corrected_cache,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    assert corrected_cache.invalidated is True
    assert corrected_cache.invalidated_reason == "source_drift_or_non_prefix"

    seed.flush()
    deleted_cache = PrefixBaselineCache(storage_path=path, source_identity="source-v3")
    keep = [index for index in range(len(values)) if index != 3]
    _prior_baseline_z(
        values.iloc[keep].reset_index(drop=True),
        group.iloc[keep].reset_index(drop=True),
        date.iloc[keep].reset_index(drop=True),
        min_obs=2,
        baseline_cache=deleted_cache,
        metric_key="cs_per_min",
        row_key=row_key.iloc[keep].reset_index(drop=True),
    )
    assert deleted_cache.invalidated is True
    assert deleted_cache.invalidated_reason == "source_drift_or_non_prefix"

    schema_cache = PrefixBaselineCache(
        storage_path=path,
        source_identity="source-v1",
        schema_fingerprint="future-schema",
    )
    assert schema_cache.invalidated is True
    assert schema_cache._entries == {}


def test_persistent_cache_is_reused_by_a_new_python_process(tmp_path: Path) -> None:
    """The serialized prefix is usable by a separate interpreter process."""

    values, group, date, row_key = _persistent_cache_fixture()
    path = tmp_path / "prefix-baseline"
    seed = PrefixBaselineCache(storage_path=path, source_identity="source-v1")
    old = slice(0, 20)
    _prior_baseline_z(
        values.iloc[old].reset_index(drop=True),
        group.iloc[old].reset_index(drop=True),
        date.iloc[old].reset_index(drop=True),
        min_obs=2,
        baseline_cache=seed,
        metric_key="cs_per_min",
        row_key=row_key.iloc[old].reset_index(drop=True),
    )
    seed.flush()

    script = """
import sys
import numpy as np
import pandas as pd
from lol_kills.ratings.global_player_bt import PrefixBaselineCache, _prior_baseline_z

path = sys.argv[1]
size = 40
values = pd.Series(np.arange(1.0, size + 1))
group = pd.Series(["mid"] * size)
date = pd.Series(pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size), unit="D"))
row_key = pd.Series([("game", "Blue", f"player-{index}", "mid") for index in range(size)], dtype=object)
cache = PrefixBaselineCache(storage_path=path, source_identity="source-v2")
z, prior = _prior_baseline_z(
    values, group, date, min_obs=2, baseline_cache=cache,
    metric_key="cs_per_min", row_key=row_key,
)
print(cache.hits, cache.stores, cache.misses, cache.invalidated, float(z.iloc[-1]), float(prior.iloc[-1]))
cache.flush()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().split()[:4] == ["1", "1", "0", "False"]

    reloaded = PrefixBaselineCache(storage_path=path, source_identity="source-v2")
    reference, reference_prior = _prior_baseline_z(values, group, date, min_obs=2)
    cached, cached_prior = _prior_baseline_z(
        values,
        group,
        date,
        min_obs=2,
        baseline_cache=reloaded,
        metric_key="cs_per_min",
        row_key=row_key,
    )
    pd.testing.assert_series_equal(cached, reference, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(
        cached_prior, reference_prior, check_names=False, check_exact=True
    )
    assert reloaded.hits == 1


def test_a_later_map_cannot_change_an_earlier_maps_normalized_metric() -> None:
    """No future leakage: mutating the last map leaves earlier rows identical."""

    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(players, _roster_profiles())
    map_dates = pd.Series(
        pd.to_datetime(maps["date"]).dt.tz_localize(None).to_numpy(),
        index=pd.Index(maps["gameid"].astype(str)),
    )

    metrics = _contribution_metrics(frame, map_dates)
    baseline, _normalization, _diagnostics = _role_normalized_composite(metrics)

    last_map = str(metrics.sort_values("_date")["_game_id"].iloc[-1])
    mutated = frame.copy()
    target = mutated["gameid"].astype(str).eq(last_map)
    assert target.any()
    mutated.loc[target, "cspm"] = mutated.loc[target, "cspm"] * 7.5
    mutated.loc[target, "dpm"] = mutated.loc[target, "dpm"] * 0.1

    after, _normalization, _diagnostics = _role_normalized_composite(
        _contribution_metrics(mutated, map_dates)
    )

    earlier = metrics["_game_id"].astype(str).ne(last_map)
    pd.testing.assert_series_equal(
        baseline[earlier], after[earlier], check_names=False
    )
    # The mutated map itself does move, so the comparison is not vacuous.
    assert not baseline[~earlier].equals(after[~earlier])


# --------------------------------------------------------------------------
# P1-4  fewer than 20 prior observations stays neutral, and is counted
# --------------------------------------------------------------------------


def test_anchor_baseline_floor_matches_the_elo_constant() -> None:
    assert ANCHOR_MIN_BASELINE_OBS == player_elo.ATTRIBUTION_MIN_BASELINE_OBS == 20


def test_a_thin_baseline_yields_exactly_neutral_counted_anchors() -> None:
    """One round robin gives every role pool at most 10 prior observations."""

    maps, players = _fixture(_locked_roster_games(rounds=1))
    frame = _with_metrics(players, _roster_profiles())

    snapshot, meta = fit_global_player_bt(
        maps, players=frame, cfg=_config(), validate=False
    )
    anchor = meta["performance_anchor"]

    assert anchor["metric_cells_present"] > 0
    assert anchor["metric_cells_observed"] == 0
    assert anchor["metric_cells_withheld_below_baseline_floor"] == anchor["metric_cells_present"]
    # Unavailable means neutral AND counted.
    assert anchor["players_anchored"] == 0
    assert anchor["players_without_metrics"] == len(snapshot)
    assert (snapshot["global_performance_anchor_logit"] == 0.0).all()
    assert (snapshot["global_performance_anchored"] == 0).all()

    # Neutral means byte-for-byte the old shrink-to-zero behaviour.
    plain, _ = fit_global_player_bt(
        maps,
        players=frame,
        cfg=_config(performance_anchor_enabled=False),
        validate=False,
    )
    pd.testing.assert_series_equal(
        snapshot.set_index("player")["global_rating"].sort_index(),
        plain.set_index("player")["global_rating"].sort_index(),
    )


def test_the_floor_withholds_early_cells_but_not_the_whole_history() -> None:
    maps, players = _fixture(_locked_roster_games(rounds=6))
    frame = _with_metrics(players, _roster_profiles())

    _snapshot, meta = fit_global_player_bt(
        maps, players=frame, cfg=_config(), validate=False
    )
    anchor = meta["performance_anchor"]

    assert anchor["baseline_min_prior_observations"] == 20
    assert anchor["metric_cells_withheld_below_baseline_floor"] > 0
    assert anchor["metric_cells_observed"] > 0
    assert anchor["players_anchored"] == 20


# --------------------------------------------------------------------------
# P1-2  a single malformed statistic cannot dominate
# --------------------------------------------------------------------------


def _malformed_row(frame: pd.DataFrame, player: str = "alpha-mid") -> pd.Series:
    """Mask for the player's LAST map, whose baseline is thickest and usable."""

    seats = frame["playername"].astype(str).eq(player)
    assert seats.any()
    last_map = str(frame.loc[seats, "gameid"].astype(str).iloc[-1])
    return seats & frame["gameid"].astype(str).eq(last_map)


def test_normalized_metrics_are_bounded_by_the_named_clip() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(players, _roster_profiles())
    # Finite, nonnegative, and absurd: it clears every check at
    # lol_kills/etl/oe_database.py:548-559.
    victim = _malformed_row(frame)
    frame.loc[victim, "cspm"] = 1.0e12

    _snapshot, meta = fit_global_player_bt(
        maps, players=frame, cfg=_config(), validate=False
    )
    anchor = meta["performance_anchor"]

    assert anchor["normalized_metric_clip"] == ANCHOR_METRIC_Z_CLIP
    assert anchor["metric_cells_clipped"] >= 1
    assert anchor["normalized_metric_min"] >= -ANCHOR_METRIC_Z_CLIP
    assert anchor["normalized_metric_max"] <= ANCHOR_METRIC_Z_CLIP


def test_one_malformed_row_cannot_reorder_the_ladder() -> None:
    maps, players = _fixture(_locked_roster_games())
    clean = _with_metrics(players, _roster_profiles())

    before, _meta = fit_global_player_bt(
        maps, players=clean, cfg=_config(), validate=False
    )

    malformed = clean.copy()
    malformed.loc[_malformed_row(malformed), "cspm"] = 1.0e12

    after, _meta = fit_global_player_bt(
        maps, players=malformed, cfg=_config(), validate=False
    )

    assert before["player"].tolist() == after["player"].tolist()
    moved = (
        after.set_index("player")["global_rating"]
        - before.set_index("player")["global_rating"]
    ).abs()
    # Without the clip this single row drives the composite to ~1e12 standard
    # deviations, which pins its player at one end of the standardized anchor
    # and moves the whole ladder.
    assert moved.max() < 1.0


def test_the_clip_is_what_bounds_the_composite() -> None:
    """Same malformed row, measured on the normalized metric directly."""

    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(players, _roster_profiles())
    map_dates = pd.Series(
        pd.to_datetime(maps["date"]).dt.tz_localize(None).to_numpy(),
        index=pd.Index(maps["gameid"].astype(str)),
    )
    frame.loc[_malformed_row(frame), "cspm"] = 1.0e12

    metrics = _contribution_metrics(frame, map_dates)
    raw, _obs = _prior_baseline_z(
        metrics["cs_per_min"],
        metrics["_role"].astype(str) + "\x1f" + metrics["_tier"].astype(str),
        metrics["_date"],
        ANCHOR_MIN_BASELINE_OBS,
    )
    composite, _normalization, diagnostics = _role_normalized_composite(metrics)

    assert raw.max() > 1.0e6, "the malformed z must really be astronomical"
    assert diagnostics["normalized_metric_max"] <= ANCHOR_METRIC_Z_CLIP
    assert composite.dropna().abs().max() <= ANCHOR_METRIC_Z_CLIP


def _daily(values: list[float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    size = len(values)
    return (
        pd.Series(values, dtype=float),
        pd.Series(["mid"] * size),
        pd.Series(pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size), unit="D")),
    )


def _persistent_cache_fixture(size: int = 40) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    values = pd.Series(np.arange(1.0, size + 1))
    group = pd.Series(["mid"] * size)
    date = pd.Series(
        pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(size), unit="D")
    )
    row_key = pd.Series(
        [("game", "Blue", f"player-{index}", "mid") for index in range(size)],
        dtype=object,
    )
    return values, group, date, row_key


def test_a_malformed_row_cannot_poison_the_baseline_of_later_rows() -> None:
    """The masking failure the robust baseline exists to prevent.

    Under a mean/std baseline the single 1e12 row drags the pool mean and std
    so far that every CLEAN row after it reads about -1/sqrt(n) instead of its
    true z, and clipping the z cannot repair that because the damage is in the
    statistic the z is measured against.  Median and MAD break down only at
    50%, so the same row is measured against an essentially unchanged baseline.
    """

    rng = np.random.default_rng(20260818)
    clean = [float(value) for value in rng.normal(5.0, 0.4, 40)]
    # The malformed row is INSERTED, so the ten rows after it are the same ten
    # clean values in both runs and only their baseline differs.
    contaminated = clean[:30] + [1.0e12] + clean[30:]

    z_clean, _obs = _prior_baseline_z(*_daily(clean), ANCHOR_MIN_BASELINE_OBS)
    z_dirty, prior_obs = _prior_baseline_z(
        *_daily(contaminated), ANCHOR_MIN_BASELINE_OBS
    )

    assert (prior_obs.iloc[31:] >= ANCHOR_MIN_BASELINE_OBS).all()
    robust = [
        abs(float(z_dirty.iloc[31 + step]) - float(z_clean.iloc[30 + step]))
        for step in range(10)
    ]
    # Every one of these rows carries the 1e12 row in its baseline, and it
    # moves them only by the ordinary effect of one extra observation.
    assert max(robust) < 0.15

    # The malformed row itself is still reported as astronomical; it is the
    # +/-3 clip, not the baseline, that bounds it downstream.
    assert z_dirty.iloc[30] > 1.0e6

    # The same ten rows under the OLD mean/std baseline.  One row inflates the
    # std enough to crush every later z toward -1/sqrt(n), which is a masking
    # failure the clip cannot undo because it is in the baseline, not the z.
    dirty = np.asarray(contaminated, dtype=float)
    pure = np.asarray(clean, dtype=float)
    legacy = [
        abs(
            (dirty[31 + step] - dirty[: 31 + step].mean())
            / dirty[: 31 + step].std(ddof=1)
            - (pure[30 + step] - pure[: 30 + step].mean())
            / pure[: 30 + step].std(ddof=1)
        )
        for step in range(10)
    ]
    assert max(legacy) > 1.0
    assert max(legacy) > 10.0 * max(robust)


def test_mad_zero_falls_back_to_the_iqr_scale() -> None:
    """More than half the pool ties, so the MAD is exactly 0."""

    # 18 zeros then 12 ones: 60% of the baseline sits exactly on the median, so
    # more than half the deviations are 0 and the scaled MAD is 0.  The upper
    # quartile is still 1, so the IQR fallback has a usable scale.
    values, group, date = _daily([0.0] * 18 + [1.0] * 22)
    z, prior_obs = _prior_baseline_z(values, group, date, ANCHOR_MIN_BASELINE_OBS)

    row = 30
    sample = values.iloc[:row].to_numpy(dtype=float)
    assert float(np.median(np.abs(sample - np.median(sample)))) == 0.0
    low, high = np.quantile(sample, (0.25, 0.75))
    assert high - low > 0.0
    expected = (float(values.iloc[row]) - float(np.median(sample))) / (
        float(high - low) / 1.349
    )
    assert prior_obs.iloc[row] == float(row)
    assert z.iloc[row] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert np.isfinite(z.iloc[row])


def test_an_all_identical_baseline_is_withheld_not_scored() -> None:
    """FAIL CLOSED on the owner's exact synthetic case.

    30 identical clean values, one 1e12 row, then 10 more identical clean
    rows.  A mean/std baseline gives the later clean rows a spurious -0.18
    because the malformed row is the ONLY source of spread.  Median/MAD sees
    no spread at all, so the metric is unavailable: NaN, neutral, and counted.
    """

    values, group, date = _daily([5.0] * 30 + [1.0e12] + [5.0] * 10)
    z, prior_obs = _prior_baseline_z(values, group, date, ANCHOR_MIN_BASELINE_OBS)

    assert z.iloc[31:].isna().all()
    assert (prior_obs.iloc[31:] >= ANCHOR_MIN_BASELINE_OBS).all()

    # What the old estimator produced for exactly these rows.
    x = np.asarray([5.0] * 30 + [1.0e12] + [5.0] * 10, dtype=float)
    legacy = [
        (x[row] - x[:row].mean()) / x[:row].std(ddof=1) for row in range(31, 41)
    ]
    assert max(legacy) < -0.15 and min(legacy) > -0.25


def test_a_degenerate_pool_is_withheld_and_never_divided_by_zero() -> None:
    """FAIL CLOSED: MAD and IQR both zero means unavailable, not imputed."""

    values, group, date = _daily([0.0] * 39 + [7.0])
    z, prior_obs = _prior_baseline_z(values, group, date, ANCHOR_MIN_BASELINE_OBS)

    # The pool is entirely constant, so both robust scales are 0.
    assert z.isna().all()
    # Unavailable is still COUNTED: the floor is not what withheld these.
    assert prior_obs.iloc[39] == 39.0
    assert not np.isinf(z.to_numpy(dtype=float)).any()


def test_every_role_pool_is_bounded_on_clean_data_too() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(players, _roster_profiles())

    _snapshot, meta = fit_global_player_bt(
        maps, players=frame, cfg=_config(), validate=False
    )
    anchor = meta["performance_anchor"]
    assert -ANCHOR_METRIC_Z_CLIP <= anchor["normalized_metric_min"]
    assert anchor["normalized_metric_max"] <= ANCHOR_METRIC_Z_CLIP


def test_anchor_stays_zero_mean_under_the_new_baselines() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(players, _roster_profiles())

    snapshot, meta = fit_global_player_bt(
        maps, players=frame, cfg=_config(), validate=False
    )

    assert abs(float(snapshot["global_performance_anchor_logit"].sum())) < 1e-12
    assert abs(float(meta["performance_anchor"]["anchor_mean_logit"])) < 1e-12
    assert abs(float(snapshot["global_rating"].mean()) - 1500.0) < 1e-6
    # Always-together teammates still separate.  The count is 3 and not 5
    # because the estimator is now rank-based: this fixture has only FOUR
    # players per (role, tier) pool, and alpha holds the same rank in the
    # top/jng pools and again in the mid/bot pools, so a median/MAD z -- which
    # deliberately reads order rather than exact spacing -- returns the same
    # number for those seats.  That is the price of a 50%-breakdown estimator
    # at n=4 and it does not appear at census scale, where the real ladder
    # keeps the same 3506 distinct ratings over 3797 players it had under
    # mean/std.  What matters here is that the pre-anchor teammate tie of
    # exactly ONE rating is broken.
    ratings = snapshot.set_index("player")["global_rating"]
    alpha = [float(ratings[f"alpha-{role}"]) for role in ROLES]
    assert len(set(alpha)) >= 3
    # Every other roster, whose seats do not share a rank, still separates all
    # five ways, so the tie above is a rank collision and not a lost signal.
    for team in ("beta", "gamma", "delta"):
        assert len({float(ratings[f"{team}-{role}"]) for role in ROLES}) == 5
