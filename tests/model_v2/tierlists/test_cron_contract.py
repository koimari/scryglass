"""Contract checks for the Vercel cron to durable OE worker handoff."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_vercel_cron_targets_the_live_refresh_route() -> None:
    config = json.loads((ROOT / "apps/scryglass/vercel.json").read_text(encoding="utf-8"))
    assert config["crons"] == [
        {"path": "/api/cron/tierlist-refresh", "schedule": "0 */6 * * *"},
        {"path": "/api/cron/pack-refresh", "schedule": "15 */6 * * *"},
    ]


def test_cron_worker_is_a_direct_python_function_with_a_distributed_lease() -> None:
    worker = (
        ROOT / "apps/scryglass/api/cron/tierlist-refresh.py"
    ).read_text(encoding="utf-8")
    assert "class handler" in worker
    assert "CRON_SECRET" in worker
    assert "ORACLES_ELIXIR_API_KEY" in worker
    assert "restore_baseline" in worker
    assert "_publish_public_pack" in worker
    assert "RetentionPlan" in worker
    assert "_RefreshLease" in worker
    assert "LOCK_PATH" in worker
    assert "skip_atom_bridge=True" in worker
    assert "refresh_candidate" in worker


def test_pack_refresh_is_a_separate_six_hour_job() -> None:
    worker = (ROOT / "apps/scryglass/api/cron/pack-refresh.py").read_text(encoding="utf-8")
    assert "tierlist-refresh.py" in worker
    assert "_publish_public_pack" in worker
    assert "PACK_LOCK_PATH" in worker
