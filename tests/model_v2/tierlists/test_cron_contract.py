"""Contract checks for the Vercel cron to durable OE worker handoff."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_vercel_cron_targets_the_live_refresh_route() -> None:
    config = json.loads((ROOT / "apps/scryglass/vercel.json").read_text(encoding="utf-8"))
    assert config["crons"] == [
        {"path": "/api/cron/source-refresh", "schedule": "0 */6 * * *"},
        {"path": "/api/cron/ratings-refresh", "schedule": "15 */6 * * *"},
        {"path": "/api/cron/tierlist-refresh", "schedule": "30 */6 * * *"},
        {"path": "/api/cron/tierlist-authority-refresh", "schedule": "45 */6 * * *"},
        {"path": "/api/cron/pack-refresh", "schedule": "50 */6 * * *"},
    ]


def test_cron_worker_is_a_direct_python_function_with_a_distributed_lease() -> None:
    worker = (
        ROOT / "apps/scryglass/api/cron/tierlist-refresh.py"
    ).read_text(encoding="utf-8")
    assert "class handler" in worker
    assert "CRON_SECRET" in worker
    assert "PACK_LATEST" in worker
    assert "_publish_public_pack" in worker
    assert "RetentionPlan" in worker
    assert "_RefreshLease" in worker
    assert "LOCK_PATH" in worker
    assert "_download_source_bundle" in worker
    assert "refresh_candidate" in worker
    assert "_publish_candidate" in worker
    assert "promote=False" in worker


def test_source_refresh_is_a_separate_six_hour_job() -> None:
    worker = (ROOT / "apps/scryglass/api/cron/source-refresh.py").read_text(encoding="utf-8")
    assert "ORACLES_ELIXIR_API_KEY" in worker
    assert "_refresh_source_inputs" in worker
    assert "_publish_source_bundle" in worker
    assert "SOURCE_LOCK_PATH" in worker


def test_production_source_refresh_uses_the_deployed_atom_bridge() -> None:
    worker = (ROOT / "apps/scryglass/api/cron/tierlist-refresh.py").read_text(encoding="utf-8")
    assert "atom_step = _verify_prebuilt_atom_bridge(runtime_root)" in worker
    assert "lol_kills.v2.champions.atoms.bridge_v1" not in worker


def test_ratings_refresh_is_a_separate_six_hour_job() -> None:
    worker = (ROOT / "apps/scryglass/api/cron/ratings-refresh.py").read_text(encoding="utf-8")
    assert "RAW_SOURCE_POINTER_PATH" in worker
    assert "_refresh_rating_inputs" in worker
    assert "_publish_source_bundle" in worker
    assert "RATINGS_LOCK_PATH" in worker
    assert "restore_baseline" not in worker
    assert "include_baseline_pack=False" in worker


def test_pack_refresh_is_a_separate_six_hour_job() -> None:
    worker = (ROOT / "apps/scryglass/api/cron/pack-refresh.py").read_text(encoding="utf-8")
    assert "tierlist-refresh.py" in worker
    assert "_download_source_bundle" in worker
    assert "_publish_public_pack" in worker
    assert "PACK_LOCK_PATH" in worker


def test_tier_authority_refresh_is_a_separate_six_hour_job() -> None:
    worker = (
        ROOT / "apps/scryglass/api/cron/tierlist-authority-refresh.py"
    ).read_text(encoding="utf-8")
    assert "_download_candidate" in worker
    assert "forward_evaluation" in worker
    assert "independent_authority" in worker
    assert "production_bundle" in worker
    assert "publish_production_bundle" in worker
    assert "AUTHORITY_LOCK_PATH" in worker
