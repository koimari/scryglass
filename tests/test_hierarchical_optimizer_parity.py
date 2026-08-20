"""Parity and callback-count tests for the hierarchical BT optimizer."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import lol_kills.ratings.hierarchical_bt as hierarchical_bt


def _map(index: int, blue: str, red: str, blue_win: int) -> dict[str, object]:
    return {
        "date": f"2026-01-{index + 1:02d} 10:00:00",
        "league": "LEC",
        "blue_team": blue,
        "red_team": red,
        "y_blue_win": blue_win,
        "game_uid": f"optimizer-fixture-{index}",
    }


def _fixture() -> pd.DataFrame:
    rows = [
        _map(0, "Team A", "Team B", 1),
        _map(1, "Team B", "Team C", 0),
        _map(2, "Team C", "Team A", 1),
        _map(3, "Team A", "Team C", 1),
        _map(4, "Team B", "Team A", 0),
        _map(5, "Team C", "Team B", 1),
        _map(6, "Team A", "Team B", 0),
        _map(7, "Team B", "Team C", 1),
        _map(8, "Team C", "Team A", 0),
        _map(9, "Team A", "Team C", 0),
        _map(10, "Team B", "Team A", 1),
        _map(11, "Team C", "Team B", 0),
    ]
    return pd.DataFrame(rows)


def _fit_with_callback_mode(
    maps: pd.DataFrame,
    monkeypatch: Any,
    *,
    split: bool,
) -> tuple[pd.DataFrame, dict[str, Any], Any, int]:
    real_minimize = hierarchical_bt.minimize
    result_box: list[Any] = []
    objective_calls: list[None] = []

    def patched_minimize(
        objective: Callable[..., Any],
        x0: np.ndarray,
        *,
        jac: Any,
        **kwargs: Any,
    ) -> Any:
        if split:
            def value(beta: np.ndarray) -> float:
                objective_calls.append(None)
                return objective(beta)[0]

            def gradient(beta: np.ndarray) -> np.ndarray:
                objective_calls.append(None)
                return objective(beta)[1]

            result = real_minimize(value, x0, jac=gradient, **kwargs)
        else:
            def fused_objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                objective_calls.append(None)
                return objective(beta)

            result = real_minimize(
                fused_objective,
                x0,
                jac=jac,
                **kwargs,
            )
        result_box.append(result)
        return result

    monkeypatch.setattr(hierarchical_bt, "minimize", patched_minimize)
    try:
        snapshot, meta = hierarchical_bt.fit_hierarchical_bt(maps, write=False)
    finally:
        monkeypatch.setattr(hierarchical_bt, "minimize", real_minimize)
    return snapshot, meta, result_box[0], len(objective_calls)


def test_fused_hierarchical_optimizer_preserves_fit_and_reduces_callbacks(monkeypatch: Any) -> None:
    maps = _fixture()
    fused_snapshot, fused_meta, fused_result, fused_calls = _fit_with_callback_mode(
        maps,
        monkeypatch,
        split=False,
    )
    split_snapshot, split_meta, split_result, split_calls = _fit_with_callback_mode(
        maps,
        monkeypatch,
        split=True,
    )

    assert fused_result.success is split_result.success
    assert fused_result.nit == split_result.nit
    assert fused_result.nfev == split_result.nfev
    assert fused_result.njev == split_result.njev
    assert fused_result.fun == split_result.fun
    np.testing.assert_array_equal(fused_result.x, split_result.x)
    assert fused_calls == fused_result.nfev
    assert split_calls == split_result.nfev + split_result.njev
    assert fused_calls < split_calls

    pd.testing.assert_frame_equal(fused_snapshot, split_snapshot, check_exact=True)
    assert fused_meta["optimizer_success"] == split_meta["optimizer_success"]
    assert fused_meta["optimizer_message"] == split_meta["optimizer_message"]


def test_hierarchical_cache_reuses_current_and_previous_source_bound_fits(
    tmp_path,
    monkeypatch: Any,
) -> None:
    maps = _fixture()
    source_identity = "source-a"
    uncached_current, uncached_meta = hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=False,
    )
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    current, cold_meta = hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256=source_identity,
    )
    first_weekly = hierarchical_bt.build_team_weekly_ranks(
        maps,
        as_of=pd.Timestamp("2026-01-12"),
        min_series=1,
        current=current,
        cache_dir=tmp_path,
        source_identity_sha256=source_identity,
    )
    assert len(optimizer_calls) == 2

    cached_current, cached_meta = hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256=source_identity,
    )
    cached_weekly = hierarchical_bt.build_team_weekly_ranks(
        maps,
        as_of=pd.Timestamp("2026-01-12"),
        min_series=1,
        current=cached_current,
        cache_dir=tmp_path,
        source_identity_sha256=source_identity,
    )
    assert len(optimizer_calls) == 2
    pd.testing.assert_frame_equal(uncached_current, current, check_exact=True)
    pd.testing.assert_frame_equal(current, cached_current, check_exact=True)
    assert cold_meta == uncached_meta
    assert cached_meta == uncached_meta
    assert first_weekly == cached_weekly

    manifest = json.loads(
        (tmp_path / hierarchical_bt.HIERARCHICAL_CACHE_MANIFEST).read_text(encoding="utf-8")
    )
    assert set(manifest) == {"current", "previous"}
    assert manifest["current"]["key"]["source_identity_sha256"] == source_identity
    assert manifest["current"]["key"]["implementation_sha256"] == hierarchical_bt.HIERARCHICAL_IMPLEMENTATION_SHA256
    assert manifest["current"]["snapshot"]["schema"] == hierarchical_bt.HIERARCHICAL_SNAPSHOT_SCHEMA
    assert manifest["current"]["snapshot"]["columns"] == list(
        hierarchical_bt._HIERARCHICAL_SNAPSHOT_COLUMNS
    )
    assert manifest["current"]["snapshot"]["byte_count"] > 0
    assert len(manifest["current"]["snapshot"]["sha256"]) == 64
    assert manifest["previous"]["key"]["slot"] == "previous"


def test_hierarchical_cache_rejects_changed_source_rows_and_binding(
    tmp_path,
    monkeypatch: Any,
) -> None:
    maps = _fixture()
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    changed = maps.copy()
    changed.loc[0, "y_blue_win"] = 0
    hierarchical_bt.fit_hierarchical_bt(
        changed,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-b",
    )
    assert len(optimizer_calls) == 3


def test_previous_cache_reuses_the_cutoff_prefix_after_an_append(
    tmp_path,
    monkeypatch: Any,
) -> None:
    base_maps = _fixture()
    appended_maps = pd.concat(
        [base_maps, pd.DataFrame([_map(12, "Team A", "Team B", 1)])],
        ignore_index=True,
    )
    cutoff = pd.Timestamp("2026-01-12")
    base_current, _ = hierarchical_bt.fit_hierarchical_bt(base_maps, write=False)
    appended_current, _ = hierarchical_bt.fit_hierarchical_bt(appended_maps, write=False)
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    hierarchical_bt.build_team_weekly_ranks(
        base_maps,
        as_of=cutoff,
        min_series=1,
        current=base_current,
        cache_dir=tmp_path,
        source_identity_sha256="source-base",
    )
    assert len(optimizer_calls) == 1
    hierarchical_bt.build_team_weekly_ranks(
        appended_maps,
        as_of=cutoff,
        min_series=1,
        current=appended_current,
        cache_dir=tmp_path,
        source_identity_sha256="source-current",
    )
    assert len(optimizer_calls) == 1

    previous_cutoff = pd.Timestamp("2026-01-03 23:59:59.999999")
    uncached_previous, _ = hierarchical_bt.fit_hierarchical_bt(
        appended_maps,
        as_of=previous_cutoff,
        write=False,
    )
    cached_previous = pd.read_parquet(
        tmp_path / "ratings_hierarchical_previous_snapshot.parquet"
    )
    pd.testing.assert_frame_equal(uncached_previous, cached_previous, check_exact=True)
    manifest = json.loads(
        (tmp_path / hierarchical_bt.HIERARCHICAL_CACHE_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["previous"]["key"]["source_identity_sha256"] != "source-current"
    assert manifest["previous"]["key"]["source_game_count"] == 3


def test_hierarchical_cache_rejects_tournament_and_league_source_corrections(
    tmp_path,
    monkeypatch: Any,
) -> None:
    maps = _fixture()
    maps["tournament"] = "LEC 2026"
    maps["league_source"] = "LEC"
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    changed_tournament = maps.copy()
    changed_tournament.loc[0, "tournament"] = "MSI 2026"
    hierarchical_bt.fit_hierarchical_bt(
        changed_tournament,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    changed_league_source = maps.copy()
    changed_league_source.loc[0, "league_source"] = "UNKNOWN"
    hierarchical_bt.fit_hierarchical_bt(
        changed_league_source,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    assert len(optimizer_calls) == 3


def test_hierarchical_cache_rejects_a_readable_tampered_snapshot(
    tmp_path,
    monkeypatch: Any,
) -> None:
    maps = _fixture()
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    snapshot_path = tmp_path / "ratings_hierarchical_snapshot.parquet"
    tampered = pd.read_parquet(snapshot_path)
    tampered.loc[0, "mu_total"] = float(tampered.loc[0, "mu_total"]) + 1.0
    tampered.to_parquet(snapshot_path, index=False)
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    assert len(optimizer_calls) == 2


def test_hierarchical_cache_rejects_changed_implementation_or_schema(
    tmp_path,
    monkeypatch: Any,
) -> None:
    maps = _fixture()
    real_minimize = hierarchical_bt.minimize
    optimizer_calls: list[None] = []

    def counted_minimize(*args: Any, **kwargs: Any) -> Any:
        optimizer_calls.append(None)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(hierarchical_bt, "minimize", counted_minimize)
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    monkeypatch.setattr(hierarchical_bt, "HIERARCHICAL_IMPLEMENTATION_SHA256", "changed")
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    assert len(optimizer_calls) == 2

    manifest_path = tmp_path / hierarchical_bt.HIERARCHICAL_CACHE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current"]["snapshot"]["columns"] = ["tampered"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hierarchical_bt.fit_hierarchical_bt(
        maps,
        write=False,
        cache_dir=tmp_path,
        source_identity_sha256="source-a",
    )
    assert len(optimizer_calls) == 3
