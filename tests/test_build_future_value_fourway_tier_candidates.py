from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import benchmarks.build_future_value_fourway_tier_candidates as adapter
from lol_kills.v2.tierlists.accepted_census import identity_sha256


IDS = ("g1", "g2")
ELIGIBLE = ("g1",)
SOURCE_AS_OF = "2026-08-20T14:51:29Z"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, locator: str | None = None) -> dict[str, object]:
    return {
        "locator": locator or path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(adapter.canonical_json_bytes(value) + b"\n")


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, str, str, Path]:
    root = tmp_path / "baseline"
    (root / "source").mkdir(parents=True)
    maps = root / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": list(IDS),
            "date": ["2026-08-19T00:00:00Z", SOURCE_AS_OF],
            "y_blue_win": [1, 0],
        }
    ).to_parquet(maps, index=False)
    players = root / "oe_player_games.parquet"
    teams = root / "oe_team_games.parquet"
    players.write_bytes(b"players")
    teams.write_bytes(b"teams")
    census = root / "accepted-census.json"
    _write_json(census, {"game_ids": list(IDS)})
    meta = root / "source/meta.json"
    _write_json(meta, {"source": "fixture"})

    source_receipt = root / "future-value-source-receipt.json"
    source_payload = {
        "source_as_of": SOURCE_AS_OF,
        "source_game_count": len(IDS),
        "source_identity_sha256": identity_sha256(IDS),
        "accepted_game_ids": list(IDS),
        "model_eligible_game_count": len(ELIGIBLE),
        "model_eligible_identity_sha256": identity_sha256(ELIGIBLE),
        "model_eligible_game_ids": list(ELIGIBLE),
        "receipt_sha256": "a" * 64,
    }
    _write_json(source_receipt, source_payload)

    candidate = root / "baseline/tierlists/champion-elo-candidate-v1.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"baseline-candidate")
    current_manifest = root / "baseline/tierlists/production-manifest-v1.json"
    prospective = root / "baseline/tierlists/prospective-evaluation-v1.json"
    current_manifest.write_bytes(b"current-manifest")
    prospective.write_bytes(b"prospective")

    source = {
        "source_as_of": SOURCE_AS_OF,
        "source_game_count": len(IDS),
        "source_identity_sha256": identity_sha256(IDS),
        "source_receipt_sha256": source_payload["receipt_sha256"],
        "source_receipt_file_sha256": _sha(source_receipt),
        "model_eligible_game_count": len(ELIGIBLE),
        "model_eligible_identity_sha256": identity_sha256(ELIGIBLE),
        "accepted_game_ids": list(IDS),
        "model_eligible_game_ids": list(ELIGIBLE),
        "source_receipt": {
            **_record(source_receipt),
            "receipt_sha256": source_payload["receipt_sha256"],
        },
        "source_files": {
            "maps": _record(maps),
            "players": _record(players),
            "teams": _record(teams),
            "accepted_census": _record(census),
        },
    }
    bundle = {
        "schema_version": "scryglass:future-value-tier-baseline-rebuild:v1",
        "status": "research_only",
        "authority": {"research_only": True},
        "source": source,
        "candidate": {
            **_record(candidate, locator="baseline/tierlists/champion-elo-candidate-v1.json"),
            "raw_sha256": _sha(candidate),
            "artifact_sha256": "baseline-artifact",
        },
        "current_production_manifest": _record(
            current_manifest, locator="baseline/tierlists/production-manifest-v1.json"
        ),
        "current_prospective_evaluation": _record(
            prospective, locator="baseline/tierlists/prospective-evaluation-v1.json"
        ),
    }
    bundle_path = root / "tier-trust-inputs.json"
    _write_json(bundle_path, bundle)

    ledger_records: dict[str, dict[str, object]] = {}
    for variant in adapter.VARIANT_ORDER:
        ledger = tmp_path / f"{variant}-ledger.json"
        receipt = tmp_path / f"{variant}-receipt.json"
        _write_json(ledger, {"rows": []})
        _write_json(receipt, {"receipt_sha256": f"{variant}-receipt"})
        ledger_records[variant] = {
            "variant": variant,
            "ledger": _record(ledger),
            "receipt": _record(receipt),
            "game_count": len(ELIGIBLE),
            "game_identity_sha256": identity_sha256(ELIGIBLE),
            "provenance": {
                "variant": variant,
                "offsets_sha256": f"{adapter.VARIANT_ORDER.index(variant) + 1:064x}",
            },
        }
    shadow = {
        "schema_version": "scryglass:future-value-tier-shadow-fourway:v1",
        "status": "research_only",
        "authority": dict(adapter.AUTHORITY),
        "variants": ledger_records,
        "source": {
            key: source[key]
            for key in (
                "source_as_of",
                "source_game_count",
                "source_identity_sha256",
                "source_receipt_sha256",
                "source_receipt_file_sha256",
                "model_eligible_game_count",
                "model_eligible_identity_sha256",
            )
        },
    }
    shadow["manifest_sha256"] = hashlib.sha256(
        adapter.canonical_json_bytes(shadow)
    ).hexdigest()
    shadow_path = tmp_path / "fourway-tier-shadow-manifest.json"
    _write_json(shadow_path, shadow)
    return bundle, bundle_path, _sha(shadow_path), _sha(bundle_path), shadow_path


def _patch_verified_staging(monkeypatch: pytest.MonkeyPatch, player_sha: str) -> None:
    def fake_stage(output, players, meta, repository, *, require_assets):
        (output / "runtime").mkdir(parents=True)
        return {
            "player_source": {
                "locator": adapter.RUNTIME_PLAYER_FILE,
                "sha256": player_sha,
            }
        }

    monkeypatch.setattr(adapter, "_stage_runtime", fake_stage)
    monkeypatch.setattr(adapter, "verify_target_parity", lambda *args, **kwargs: None)


def test_builds_four_candidates_on_one_exact_eligible_universe_and_keeps_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_path, shadow_sha, bundle_sha, shadow_path = _fixture(tmp_path)
    baseline_path = bundle_path.parent / "baseline/tierlists/champion-elo-candidate-v1.json"
    baseline_before = baseline_path.read_bytes()
    player_sha = _sha(bundle_path.parent / "oe_player_games.parquet")
    _patch_verified_staging(monkeypatch, player_sha)
    monkeypatch.setattr(
        adapter,
        "load_tier_baseline_bundle",
        lambda path, expected_raw_sha256: bundle,
    )

    calls: list[dict[str, object]] = []

    def fake_offsets(ledger, receipt, *, source_receipt, variant):
        index = adapter.VARIANT_ORDER.index(variant) + 1
        provenance = {
            "variant": variant,
            "offsets_sha256": f"{index:064x}",
        }
        return {"g1": float(index)}, provenance

    monkeypatch.setattr(adapter, "load_tier_offset_ledger", fake_offsets)

    def fake_builder(root, **kwargs):
        calls.append(kwargs)
        return {
            "as_of": SOURCE_AS_OF,
            "expected_live_as_of": SOURCE_AS_OF,
            "source": {
                "locator": adapter.SOURCE_LOCATOR,
                "source_files": [adapter.SOURCE_LOCATOR],
                "source_latest_replayed": SOURCE_AS_OF,
                "raw_sha256": hashlib.sha256(
                    adapter.canonical_json_bytes(
                        [{"locator": adapter.SOURCE_LOCATOR, "raw_sha256": player_sha}]
                    )
                ).hexdigest(),
            },
        }

    monkeypatch.setattr(adapter, "build_pooled_candidate", fake_builder)
    monkeypatch.setattr(
        adapter,
        "validate_candidate",
        lambda candidate, **kwargs: {"variant": kwargs["variant"]},
    )

    result = adapter.build_fourway_tier_candidates(
        shadow_manifest_path=shadow_path,
        baseline_bundle_path=bundle_path,
        expected_shadow_manifest_sha256=shadow_sha,
        expected_baseline_bundle_sha256=bundle_sha,
        output_root=tmp_path / "output",
        expected_accepted_game_count=2,
        expected_accepted_identity_sha256=identity_sha256(IDS),
        expected_model_eligible_game_count=1,
        expected_model_eligible_identity_sha256=identity_sha256(ELIGIBLE),
        candidate_builder=fake_builder,
    )

    assert len(calls) == 4
    assert all(call["allowed_game_ids"] == ["g1"] for call in calls)
    assert all(set(call["pre_map_offset_override"]) == {"g1"} for call in calls)
    assert baseline_path.read_bytes() == baseline_before
    assert result["status"] == "research_only"
    assert result["authority"] == adapter.AUTHORITY
    assert result["source"]["model_eligible_game_ids"] == ["g1"]
    for variant in adapter.VARIANT_ORDER:
        assert (tmp_path / "output/candidates" / f"{variant}.json").is_file()
    assert (tmp_path / "output" / adapter.MANIFEST_FILE).is_file()


def test_rejects_offset_ledger_with_extra_game_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_path, shadow_sha, bundle_sha, shadow_path = _fixture(tmp_path)
    monkeypatch.setattr(
        adapter,
        "load_tier_baseline_bundle",
        lambda path, expected_raw_sha256: bundle,
    )
    monkeypatch.setattr(adapter, "verify_target_parity", lambda *args, **kwargs: None)

    def extra_offsets(ledger, receipt, *, source_receipt, variant):
        index = adapter.VARIANT_ORDER.index(variant) + 1
        return {
            "g1": 0.0,
            "g2": 1.0,
        }, {"variant": variant, "offsets_sha256": f"{index:064x}"}

    monkeypatch.setattr(adapter, "load_tier_offset_ledger", extra_offsets)
    with pytest.raises(adapter.FourwayTierCandidateError, match="census"):
        adapter.build_fourway_tier_candidates(
            shadow_manifest_path=shadow_path,
            baseline_bundle_path=bundle_path,
            expected_shadow_manifest_sha256=shadow_sha,
            expected_baseline_bundle_sha256=bundle_sha,
            output_root=tmp_path / "output",
            expected_accepted_game_count=2,
            expected_accepted_identity_sha256=identity_sha256(IDS),
            expected_model_eligible_game_count=1,
            expected_model_eligible_identity_sha256=identity_sha256(ELIGIBLE),
            candidate_builder=lambda *args, **kwargs: {},
        )


def test_rejects_shadow_source_mismatch_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_path, _shadow_sha, bundle_sha, shadow_path = _fixture(tmp_path)
    shadow = json.loads(shadow_path.read_text())
    shadow["source"]["source_receipt_sha256"] = "f" * 64
    shadow.pop("manifest_sha256")
    shadow["manifest_sha256"] = hashlib.sha256(
        adapter.canonical_json_bytes(shadow)
    ).hexdigest()
    shadow_path.write_bytes(adapter.canonical_json_bytes(shadow) + b"\n")
    monkeypatch.setattr(
        adapter,
        "load_tier_baseline_bundle",
        lambda path, expected_raw_sha256: bundle,
    )
    with pytest.raises(adapter.FourwayTierCandidateError, match="source binding"):
        adapter.build_fourway_tier_candidates(
            shadow_manifest_path=shadow_path,
            baseline_bundle_path=bundle_path,
            expected_shadow_manifest_sha256=_sha(shadow_path),
            expected_baseline_bundle_sha256=bundle_sha,
            output_root=tmp_path / "output",
            expected_accepted_game_count=2,
            expected_accepted_identity_sha256=identity_sha256(IDS),
            expected_model_eligible_game_count=1,
            expected_model_eligible_identity_sha256=identity_sha256(ELIGIBLE),
            candidate_builder=lambda *args, **kwargs: {},
        )


def test_rejects_raw_file_hash_when_candidate_binding_hash_is_required() -> None:
    player_sha = "a" * 64
    runtime = {
        "player_source": {
            "locator": adapter.RUNTIME_PLAYER_FILE,
            "sha256": player_sha,
        }
    }
    candidate_source = {
        "locator": adapter.SOURCE_LOCATOR,
        "source_files": [adapter.SOURCE_LOCATOR],
        "raw_sha256": player_sha,
    }
    with pytest.raises(adapter.FourwayTierCandidateError, match="source bytes"):
        adapter._verify_candidate_source(candidate_source, runtime, "current_only")
