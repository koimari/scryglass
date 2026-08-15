from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lol_kills.research.private_draft_recipe import (
    R9E_CANDIDATE_ID,
    R9E_RECORDED_AUC,
    PrivateDraftRecipeError,
    build_r9e_query,
    load_private_descriptive_r9e_binding,
    score_private_descriptive_r9e,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = {
        "schema_version": "scryglass:r9e-fast-path:v1",
        "candidate_id": R9E_CANDIDATE_ID,
        "status": "development_only",
        "recorded_metrics": {"auc": 0.70681, "brier": 0.21708, "log_loss": 0.62330},
        "source_policy": {"public_authority": "unavailable"},
        "training": {"fit_through": "2026-08-12 06:56:48+00:00", "patch_token": "16.15"},
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module_path = tmp_path / "r9e_module.py"
    module_path.write_text(
        "class Checkpoint:\n"
        "    manifest = {'candidate_id': 'R9E_d4_ss'}\n"
        "def load_checkpoint(cache_dir):\n"
        "    return Checkpoint()\n"
        "def score_query(cache_dir, query):\n"
        "    return {'candidate_id': 'R9E_d4_ss', 'status': 'development_only', 'authority': 'unavailable',\n"
        "            'blue_probability': 0.58, 'red_probability': 0.42, 'query': query}\n",
        encoding="utf-8",
    )
    binding = {
        "schema_version": "scryglass:private-r9e-development-composite:v1",
        "candidate_id": R9E_CANDIDATE_ID,
        "mode": "development_composite",
        "status": "development_only",
        "authority": "unavailable",
        "claim_ceiling": {
            "descriptive_draft_score": False,
            "composite_diagnostic": True,
            "public_probability": False,
            "recommendation": False,
            "betting": False,
        },
        "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "cache_dir": str(cache),
    }
    binding["artifact_sha256"] = _artifact(binding)
    binding_path = tmp_path / "r9e-binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    return binding_path, binding


def test_private_binding_requires_descriptive_claim_ceiling(tmp_path: Path) -> None:
    binding_path, binding = _fixture(tmp_path)
    binding["claim_ceiling"]["descriptive_draft_score"] = True
    binding["artifact_sha256"] = _artifact(binding)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(PrivateDraftRecipeError, match="claim ceiling"):
        load_private_descriptive_r9e_binding(binding_path)


def test_private_binding_checks_manifest_hash_and_cache_identity(tmp_path: Path) -> None:
    binding_path, binding = _fixture(tmp_path)
    manifest_path = Path(binding["manifest"]["path"])
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(PrivateDraftRecipeError, match="manifest hash"):
        load_private_descriptive_r9e_binding(binding_path)


def test_private_binding_rejects_a_binding_symlink(tmp_path: Path) -> None:
    binding_path, _ = _fixture(tmp_path)
    link_path = tmp_path / "binding-link.json"
    os.symlink(binding_path, link_path)

    with pytest.raises(PrivateDraftRecipeError, match="binding is a symlink"):
        load_private_descriptive_r9e_binding(link_path)


def test_private_r9e_score_stays_a_composite_diagnostic(tmp_path: Path) -> None:
    binding_path, _ = _fixture(tmp_path)
    result = score_private_descriptive_r9e(binding_path, {"date": "2026-08-14T12:00:00Z"})

    assert result["candidate_id"] == R9E_CANDIDATE_ID
    assert result["mode"] == "development_composite"
    assert result["estimand"] == "composite_map_probability"
    assert result["status"] == "development_only"
    assert result["authority"] == "unavailable"
    assert result["recorded_metrics"]["auc"] == pytest.approx(R9E_RECORDED_AUC)
    assert result["claim_ceiling"]["descriptive_draft_score"] is False
    assert result["claim_ceiling"]["composite_diagnostic"] is True
    assert result["development_composite"]["edge_pp"] == pytest.approx(16.0)
    assert "draft_contribution" not in result


def test_build_r9e_query_preserves_exact_roster_and_role_order() -> None:
    registration = {
        "teams": [
            {
                "side": "blue",
                "organization_name": "Blue",
                "players": [
                    {"role": role, "display_name": f"B-{role}"} for role in ("top", "jungle", "mid", "bot", "support")
                ],
            },
            {
                "side": "red",
                "organization_name": "Red",
                "players": [
                    {"role": role, "display_name": f"R-{role}"} for role in ("top", "jungle", "mid", "bot", "support")
                ],
            },
        ]
    }
    draft = {
        "blue": {role: f"B-Champ-{role}" for role in ("top", "jungle", "mid", "bot", "support")},
        "red": {role: f"R-Champ-{role}" for role in ("top", "jungle", "mid", "bot", "support")},
    }
    query = build_r9e_query(
        registration,
        draft,
        date="2026-08-14T12:00:00Z",
        patch="16.15",
        mu_diff=25.0,
        sigma_pair=80.0,
        player_elo={"p": 0.54, "sigma": 40.0},
    )

    assert query["blue"]["jng"] == {"player": "B-jungle", "champion": "B-Champ-jungle"}
    assert query["red"]["sup"] == {"player": "R-support", "champion": "R-Champ-support"}
    assert query["mu_diff"] == 25.0
    assert query["player_elo"]["p"] == 0.54
