"""Contract checks for the Vercel cron to durable OE worker handoff."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_vercel_cron_targets_the_live_refresh_route() -> None:
    config = json.loads((ROOT / "apps/scryglass/vercel.json").read_text(encoding="utf-8"))
    assert config["crons"] == [
        {"path": "/api/cron/tierlist-refresh", "schedule": "*/5 * * * *"}
    ]


def test_cron_route_requires_worker_auth_and_preserves_source_windows() -> None:
    route = (
        ROOT / "apps/scryglass/src/app/api/cron/tierlist-refresh/route.ts"
    ).read_text(encoding="utf-8")
    assert "CRON_SECRET" in route
    assert "SCRYGLASS_TIERLIST_INGEST_URL" in route
    assert "SCRYGLASS_TIERLIST_INGEST_TOKEN" in route
    assert 'protocol !== "https:"' in route
    assert 'window_start: START_DATE' in route
    assert 'live_window_start: LIVE_WINDOW_START' in route
    assert 'source_mode: "oe_only"' in route
    assert 'trigger: "vercel_cron"' in route
    assert "controller.abort()" in route
    assert "dispatchGitHubWorkflow" not in route
    assert "api.github.com" not in route
    assert "refresh_worker_not_configured" in route
