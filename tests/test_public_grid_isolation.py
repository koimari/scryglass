from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PUBLIC_GRID_TOKENS = (
    "GRID_API_KEY",
    "--download-grid",
    "--grid-required",
)


def test_production_workflows_have_no_grid_requirement() -> None:
    workflow_root = ROOT / ".github/workflows"
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(workflow_root.glob("*.y*ml"))
    )

    for token in FORBIDDEN_PUBLIC_GRID_TOKENS:
        assert token not in workflow_text


def test_retired_grid_snapshot_workflow_stays_absent() -> None:
    assert not (ROOT / ".github/workflows/live-grid-snapshots.yml").exists()


def test_public_pack_refresh_forces_an_oe_only_warehouse() -> None:
    source = (ROOT / "lol_kills/update_public_pack.py").read_text(encoding="utf-8")

    assert 'refresh_args.append("--skip-grid")' in source
    assert 'parser.add_argument("--download-grid"' not in source
    assert 'parser.add_argument("--grid-required"' not in source


def test_public_tier_refresh_has_no_grid_command() -> None:
    source = (ROOT / "lol_kills/v2/tierlists/live_refresh.py").read_text(encoding="utf-8")

    assert '"--download-grid"' not in source
    assert 'choices=("oe_only",)' in source
