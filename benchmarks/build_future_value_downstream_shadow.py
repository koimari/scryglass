"""Build research-only downstream shadows for the four rating variants.

The builder joins existing source-bound artifacts. It does not fit a model and
does not modify a public pack. The variant-specific game fields come from the
fold component evidence emitted by the four-model evaluation. Player and team
snapshots stay current-rating snapshots until a source-bound future-value
snapshot producer exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from benchmarks.future_value_downstream_diff import (
    VARIANT_NAMES,
    compare_downstream_variants,
    write_downstream_diff_report,
)
from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)


SCHEMA_VERSION = "scryglass:future-value-downstream-shadow:v1"
DEFAULT_RUN_ROOT = Path("/private/tmp/scryglass-four-variant-runs")
DEFAULT_PUBLIC_PACK = Path(
    "/Users/river/Library/Application Support/Scryglass Worker/public-packs/"
    "v2026.08.20.210112"
)
DEFAULT_SOURCE_ROOT = Path("/private/tmp/scryglass-four-variant-freeze-20260820T145129/source")
DEFAULT_SOURCE_RECEIPT = Path(
    "/private/tmp/scryglass-four-variant-freeze-20260820T145129/future-value-source-receipt.json"
)
DEFAULT_CURRENT_ROOT = DEFAULT_RUN_ROOT / "current-ratings"
DEFAULT_EVALUATION_ROOT = DEFAULT_RUN_ROOT / "evaluation-v2"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "downstream-shadows-v4"


class ShadowBuildError(RuntimeError):
    """The downstream shadow cannot be built from verified inputs."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ShadowBuildError("value is not canonical JSON") from error


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ShadowBuildError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowBuildError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise ShadowBuildError(f"{label} must be an object")
    return value


def _load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise ShadowBuildError(f"{label} is missing or unsafe: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # pandas/pyarrow use several exception types
        raise ShadowBuildError(f"{label} cannot be read: {path}") from error
    if frame.empty:
        raise ShadowBuildError(f"{label} is empty")
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.isoformat().replace("+00:00", "Z")
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ShadowBuildError("non-finite value in shadow input")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _authority() -> dict[str, bool]:
    return {
        "research_only": True,
        "public_player_rating": False,
        "public_team_rating": False,
        "public_probability": False,
        "promotion": False,
        "deployment": False,
        "odds": False,
        "expected_value": False,
        "recommendation": False,
        "betting": False,
    }


def _source_envelope(source: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": _authority(),
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "accepted_game_ids": list(source["accepted_game_ids"]),
        "rows": rows,
    }


def _validate_source(source: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    try:
        validate_future_value_source_receipt_payload(dict(source))
    except Exception as error:
        raise ShadowBuildError(f"source receipt failed validation: {error}") from error
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ShadowBuildError("source receipt path is not a regular file")
    result = dict(source)
    if result.get("status") != "accepted_source_bound_development_only":
        raise ShadowBuildError("source receipt is not development-only")
    if result.get("authority", {}).get("research_only") is not True:
        raise ShadowBuildError("source receipt does not bind research-only authority")
    return result


def _verify_model(model: Mapping[str, Any], variant: str, source: Mapping[str, Any]) -> dict[str, Any]:
    if model.get("schema_version") != "scryglass:future-value-four-variant-evaluation:v1":
        raise ShadowBuildError(f"{variant} evaluation schema is invalid")
    model_source = model.get("source")
    if not isinstance(model_source, Mapping):
        raise ShadowBuildError(f"{variant} evaluation source is missing")
    expected_fields = {
        "source_as_of": source.get("source_as_of"),
        "source_game_count": source.get("source_game_count"),
        "source_identity_sha256": source.get("source_identity_sha256"),
        "source_receipt_sha256": source.get("receipt_sha256"),
    }
    for field, expected in expected_fields.items():
        if model_source.get(field) != expected:
            raise ShadowBuildError(f"{variant} evaluation source {field} changed")
    variants = model.get("variants")
    if not isinstance(variants, Mapping) or variant not in variants:
        raise ShadowBuildError(f"{variant} evaluation payload is missing")
    payload = variants[variant]
    if not isinstance(payload, Mapping) or payload.get("status") != "development_evaluated":
        raise ShadowBuildError(f"{variant} evaluation is not complete")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or any(
        bool(value) for key, value in authority.items() if key != "research_only"
    ):
        raise ShadowBuildError(f"{variant} evaluation grants authority")
    ledger = payload.get("prediction_ledger")
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("rows"), list):
        raise ShadowBuildError(f"{variant} prediction ledger is missing")
    rows = ledger["rows"]
    if ledger.get("row_count") != len(rows):
        raise ShadowBuildError(f"{variant} prediction ledger row count changed")
    claimed = ledger.get("sha256")
    actual = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    if claimed != actual:
        raise ShadowBuildError(f"{variant} prediction ledger hash changed")
    return dict(payload)


def _component_rows(payload: Mapping[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for fold in payload.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        evidence = fold.get("component_evidence")
        if not isinstance(evidence, Mapping):
            continue
        for raw_row in evidence.get("rows", []):
            if not isinstance(raw_row, Mapping):
                continue
            game_id = str(raw_row.get("game_id") or "").strip()
            if not game_id:
                continue
            if game_id in output:
                raise ShadowBuildError(f"{variant} component evidence duplicates {game_id}")
            needed = ("current_rating_logit", "player_value_logit", "scaling_curve_logit", "full_model_logit")
            if any(key not in raw_row for key in needed):
                raise ShadowBuildError(f"{variant} component evidence is incomplete for {game_id}")
            selected: dict[str, Any] = {}
            for key in (*needed, "team_context_logit", "data_quality_logit"):
                calibrated_key = f"calibrated_{key}"
                selected[key] = raw_row.get(calibrated_key, raw_row.get(key))
            selected["support_status"] = raw_row.get("support_status")
            selected["calibration_slope"] = raw_row.get("calibration_slope", 1.0)
            selected["component_scale"] = (
                "strict_prior_calibrated"
                if "calibrated_full_model_logit" in raw_row
                else "raw_identity"
            )
            output[game_id] = selected
    if not output:
        raise ShadowBuildError(f"{variant} component evidence is empty")
    return output


def _real_draft_rows(
    public_games: Mapping[str, Any],
    component_rows: Mapping[str, Mapping[str, Any]],
    variant: str,
    current_base: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    missing_components = 0
    for game_id in sorted(set(public_games) & set(component_rows)):
        game = public_games[game_id]
        evidence = component_rows[game_id]
        if not isinstance(game, Mapping):
            continue
        edge = game.get("edge_components")
        if not isinstance(edge, Mapping):
            missing_components += 1
            continue
        required = ("base", "ally_synergy", "archetype_interactions", "enemy_counter", "same_role")
        if any(key not in edge or edge[key] is None for key in required):
            missing_components += 1
            continue
        current = current_base.get(game_id, {})
        current_team = current.get("base_team_logit")
        current_player = current.get("base_player_logit")
        if current_team is None or current_player is None:
            raise ShadowBuildError(f"current rating feature is missing for {game_id}")
        row: dict[str, Any] = {
            "game_uid": game_id,
            "league": game.get("league"),
            "competition_tier": game.get("competition_tier"),
            "base": edge["base"],
            "ally_synergy": edge["ally_synergy"],
            "archetype_interactions": edge["archetype_interactions"],
            "enemy_counter": edge["enemy_counter"],
            "same_role": edge["same_role"],
            "draft_edge": game.get("draft_edge", edge.get("total")),
            "target": game.get("y"),
            "current_rating_logit": float(current_team) + float(current_player),
            "future_player_form_logit": evidence["player_value_logit"]
            if variant in {"future_player_form", "both"}
            else None,
            "scaling_raw_logit": evidence["scaling_curve_logit"]
            if variant in {"scaling_curve", "both"}
            else None,
            # The current model emits one scaling component. It does not emit
            # a separately verified raw/shape/atom split.
            "scaling_shape_logit": None,
            "curve_atom_interaction_logit": None,
            "composite_logit": evidence["full_model_logit"],
        }
        rows.append(row)
    if not rows:
        raise ShadowBuildError(f"{variant} has no real Draft Score overlap")
    return rows, {"draft_rows": len(rows), "draft_rows_missing_components": missing_components}


def _real_tier_rows(public_games: Mapping[str, Any], game_ids: set[str]) -> list[dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for game_id in sorted(game_ids):
        game = public_games.get(game_id)
        if not isinstance(game, Mapping):
            continue
        pool = game.get("draft_pool")
        picks = pool.get("picked") if isinstance(pool, Mapping) else None
        patch = pool.get("patch") if isinstance(pool, Mapping) else None
        scope = game.get("league")
        if not isinstance(picks, list) or not isinstance(patch, str) or not isinstance(scope, str):
            continue
        for pick in picks:
            if not isinstance(pick, Mapping):
                continue
            champion = str(pick.get("champion") or "").strip()
            role = str(pick.get("role") or "").strip()
            rank = pick.get("tier_rank")
            if not champion or not role or isinstance(rank, bool) or not isinstance(rank, (int, float)):
                continue
            identity = "|".join((scope, patch, role, champion))
            by_identity.setdefault(
                identity,
                {
                    "champion": champion,
                    "role": role,
                    "scope": scope,
                    "patch": patch,
                    "rank": rank,
                },
            )
    return [by_identity[key] for key in sorted(by_identity)]


def _current_snapshot_rows(path: Path, *, team: bool) -> list[dict[str, Any]]:
    frame = _load_parquet(path, "team snapshot" if team else "player snapshot")
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        row: dict[str, Any] = {}
        identity = "team" if team else "player"
        if raw.get(identity) is None:
            continue
        row[identity] = raw[identity]
        for key in ("mu_total", "mu_effective", "mu_regional", "mu_meta", "sigma", "n_maps"):
            if key in raw:
                row[key] = raw[key]
        # These fields are present in every variant so a missing producer is
        # visible as null rather than silently treated as zero.
        row["future_value"] = None
        row["team_value"] = None
        row["scaling_curve"] = None
        rows.append(row)
    if not rows:
        raise ShadowBuildError("current snapshot has no rows")
    return rows


def _real_profile_rows(
    profile_games: Mapping[str, Any],
    game_ids: set[str],
    component_rows: Mapping[str, Mapping[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_id in sorted(game_ids):
        if game_id not in profile_games:
            continue
        evidence = component_rows[game_id]
        rows.append(
            {
                "game_uid": game_id,
                "future_value": evidence["player_value_logit"]
                if variant in {"future_player_form", "both"}
                else None,
                "team_value": None,
                "scaling_curve": evidence["scaling_curve_logit"]
                if variant in {"scaling_curve", "both"}
                else None,
            }
        )
    return rows


def _real_match_rows(
    profile_games: Mapping[str, Any],
    draft_games: Mapping[str, Any],
    game_ids: set[str],
    component_rows: Mapping[str, Mapping[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_id in sorted(game_ids):
        game = profile_games.get(game_id)
        draft = draft_games.get(game_id)
        if not isinstance(game, Mapping) or not isinstance(draft, Mapping):
            continue
        evidence = component_rows[game_id]
        blue = game.get("blue_win")
        if isinstance(blue, bool) or not isinstance(blue, (int, float)):
            continue
        edge = draft.get("edge_components")
        rows.append(
            {
                "game_uid": game_id,
                "draft_edge": draft.get("draft_edge", edge.get("total") if isinstance(edge, Mapping) else None),
                "blue_result": blue,
                "red_result": 1 - blue,
                "future_player_value": evidence["player_value_logit"]
                if variant in {"future_player_form", "both"}
                else None,
                "future_team_value": None,
            }
        )
    return rows


def _variant_root(
    root: Path,
    *,
    source: Mapping[str, Any],
    variant: str,
    player_rows: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    tier_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "future-value-source-receipt.json", source)
    _write_json(root / "features/player_ratings_snapshot.json", _source_envelope(source, player_rows))
    _write_json(root / "features/ratings_snapshot.json", _source_envelope(source, team_rows))
    _write_json(root / "rankings/tierlists.json", _source_envelope(source, tier_rows))
    _write_json(root / "features/draft_records.json", _source_envelope(source, draft_rows))
    _write_json(root / "features/profile_records.json", _source_envelope(source, profile_rows))
    _write_json(root / "features/match_records_shadow.json", _source_envelope(source, match_rows))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": _authority(),
        "pack_id": "future-value-downstream-shadow-v1",
        "release_id": "future-value-downstream-shadow-v1",
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "accepted_game_ids": list(source["accepted_game_ids"]),
        "team_rating_rows": len(team_rows),
        "player_rating_rows": len(player_rows),
        "total_files": 7,
        "total_bytes": 0,
    }
    _write_json(root / "manifest.json", manifest)


def _effect_metrics(
    roots: Mapping[str, Path],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the nullable game-level effects written by this builder."""

    def rows(path: Path) -> list[dict[str, Any]]:
        payload = _load_json(path, str(path))
        value = payload.get("rows")
        if not isinstance(value, list):
            raise ShadowBuildError(f"shadow rows are missing: {path}")
        return [row for row in value if isinstance(row, Mapping)]

    draft_rows = {name: rows(root / "features/draft_records.json") for name, root in roots.items()}
    profile_rows = {name: rows(root / "features/profile_records.json") for name, root in roots.items()}
    match_rows = {name: rows(root / "features/match_records_shadow.json") for name, root in roots.items()}

    def finite(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def field_summary(mapping: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, values in mapping.items():
            numeric = [number for row in values if (number := finite(row.get(field))) is not None]
            summary[name] = {
                "rows": len(values),
                "available_rows": len(numeric),
                "coverage": len(numeric) / len(values) if values else 0.0,
                "mean": sum(numeric) / len(numeric) if numeric else None,
                "mean_abs": sum(abs(number) for number in numeric) / len(numeric) if numeric else None,
                "max_abs": max((abs(number) for number in numeric), default=None),
            }
        return summary

    def changed_from_baseline(mapping: Mapping[str, list[dict[str, Any]]], field: str) -> dict[str, Any]:
        baseline = {str(row.get("game_uid")): finite(row.get(field)) for row in mapping["current_only"]}
        result: dict[str, Any] = {}
        for name in VARIANT_NAMES[1:]:
            changed: list[float] = []
            for row in mapping[name]:
                identity = str(row.get("game_uid"))
                left = baseline.get(identity)
                right = finite(row.get(field))
                if left is None or right is None:
                    continue
                delta = right - left
                if abs(delta) > 1e-12:
                    changed.append(delta)
            result[name] = {
                "changed_rows": len(changed),
                "mean": sum(changed) / len(changed) if changed else 0.0,
                "mean_abs": sum(abs(value) for value in changed) / len(changed) if changed else 0.0,
                "max_abs": max((abs(value) for value in changed), default=0.0),
            }
        return result

    return {
        "draft": {
            "future_player_form_logit": field_summary(draft_rows, "future_player_form_logit"),
            "scaling_raw_logit": field_summary(draft_rows, "scaling_raw_logit"),
            "composite_logit": field_summary(draft_rows, "composite_logit"),
            "composite_logit_delta_from_current_only": changed_from_baseline(draft_rows, "composite_logit"),
        },
        "profiles": {
            "future_value": field_summary(profile_rows, "future_value"),
            "scaling_curve": field_summary(profile_rows, "scaling_curve"),
        },
        "matches": {
            "future_player_value": field_summary(match_rows, "future_player_value"),
            "future_team_value": field_summary(match_rows, "future_team_value"),
        },
        "source_receipt_sha256": source["receipt_sha256"],
    }


def build_shadows(
    *,
    source_receipt_path: Path = DEFAULT_SOURCE_RECEIPT,
    current_root: Path = DEFAULT_CURRENT_ROOT,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    public_pack_root: Path = DEFAULT_PUBLIC_PACK,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_receipt_path = source_receipt_path.resolve()
    source = _validate_source(_load_json(source_receipt_path, "source receipt"), source_receipt_path)
    if output_root.exists():
        raise ShadowBuildError(f"output root already exists: {output_root}")
    source_snapshot_receipt = _load_json(
        current_root / "current-rating-ledger-receipt.json", "current rating receipt"
    )
    if source_snapshot_receipt.get("source_receipt_sha256") != source.get("receipt_sha256"):
        raise ShadowBuildError("current rating receipt source changed")
    draft_manifest = _load_json(public_pack_root / "features/draft_records.json", "public Draft Score artifact")
    if draft_manifest.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise ShadowBuildError("public Draft Score source identity changed")
    draft_games = draft_manifest.get("games")
    if not isinstance(draft_games, Mapping):
        raise ShadowBuildError("public Draft Score games are missing")
    profile_payload = _load_json(public_pack_root / "features/profile_records.json", "public profile artifact")
    profile_games = profile_payload.get("games")
    if not isinstance(profile_games, Mapping):
        raise ShadowBuildError("public profile games are missing")
    player_rows = _current_snapshot_rows(current_root / "player/player_ratings_snapshot.parquet", team=False)
    team_rows = _current_snapshot_rows(current_root / "team/ratings_snapshot.parquet", team=True)
    # Use the producer-owned current rating ledger for the shared current
    # component. The public Draft Score base is kept as the atom coverage
    # source, while its rating rows have a smaller intersection with the fold
    # ledgers.
    base_frame = _load_parquet(
        current_root / "current-rating-ledger.parquet", "current rating feature ledger"
    )
    base_by_game = {
        str(row["game_id"]): row
        for row in base_frame.to_dict("records")
        if row.get("game_id") is not None
    }
    component_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_ids_by_variant: dict[str, set[str]] = {}
    for variant in VARIANT_NAMES:
        model = _load_json(evaluation_root / variant / "model.json", f"{variant} evaluation")
        payload = _verify_model(model, variant, source)
        component_by_variant[variant] = _component_rows(payload, variant)
        ledger = payload["prediction_ledger"]
        prediction_ids_by_variant[variant] = {
            str(row["game_id"])
            for row in ledger["rows"]
            if isinstance(row, Mapping) and str(row.get("game_id") or "").strip()
        }
    common_game_ids = set(draft_games) & set(profile_games)
    common_game_ids &= set.intersection(*prediction_ids_by_variant.values())
    common_game_ids &= set.intersection(*(set(value) for value in component_by_variant.values()))
    # The public pack has 2,051 real Draft Score records. The frozen fold
    # ledgers contain 3,606 validation records. Their verified intersection
    # is 630 games for this source receipt.
    if len(common_game_ids) != 630:
        raise ShadowBuildError(f"unexpected common shadow coverage: {len(common_game_ids)}")
    tier_rows = _real_tier_rows(draft_games, common_game_ids)
    roots: dict[str, Path] = {}
    coverage: dict[str, Any] = {
        "accepted_game_count": source["source_game_count"],
        "model_eligible_game_count": source.get("model_eligible_game_count"),
        "public_draft_game_count": len(draft_games),
        "public_profile_game_count": len(profile_games),
        "four_way_common_game_count": len(common_game_ids),
        "tier_rows": len(tier_rows),
        "player_snapshot_rows": len(player_rows),
        "team_snapshot_rows": len(team_rows),
    }
    for variant in VARIANT_NAMES:
        root = output_root / variant
        components = component_by_variant[variant]
        draft_rows, draft_coverage = _real_draft_rows(
            draft_games, components, variant, base_by_game
        )
        profile_rows = _real_profile_rows(profile_games, common_game_ids, components, variant)
        match_rows = _real_match_rows(
            profile_games, draft_games, common_game_ids, components, variant
        )
        _variant_root(
            root,
            source=source,
            variant=variant,
            player_rows=player_rows,
            team_rows=team_rows,
            tier_rows=tier_rows,
            draft_rows=draft_rows,
            profile_rows=profile_rows,
            match_rows=match_rows,
        )
        roots[variant] = root
        coverage[variant] = {
            **draft_coverage,
            "profile_rows": len(profile_rows),
            "match_rows": len(match_rows),
            "future_player_snapshot_rows": 0,
            "future_team_snapshot_rows": 0,
            "scaling_snapshot_rows": 0,
        }
    report = compare_downstream_variants(
        roots,
        source_receipt=source_receipt_path,
    )
    report["shadow_build"] = {
        "schema_version": SCHEMA_VERSION,
        "coverage": coverage,
        "source_receipt_path": str(source_receipt_path),
        "current_rating_receipt_path": str(current_root / "current-rating-ledger-receipt.json"),
        "public_draft_artifact_path": str(public_pack_root / "features/draft_records.json"),
        "public_draft_artifact_sha256": _sha256(public_pack_root / "features/draft_records.json"),
        "model_evaluation_root": str(evaluation_root),
        "effects_available": {
            "current_rating": True,
            "future_player_form_game_shadow": True,
            "scaling_curve_game_shadow": True,
            "future_player_snapshot": False,
            "future_team_snapshot": False,
            "tierlist_recalculation": False,
        },
        "blockers": [
            "future_player_snapshot_producer_missing",
            "future_team_snapshot_producer_missing",
            "tierlist_recalculation_producer_missing",
            "draft_crossfit_component_receipts_missing",
            "shadow_coverage_limited_to_630_games_common_to_public_atoms_profiles_and_four_fold_ledgers",
            "model_calibration_and_source_promotion_gates_remain_blocked",
        ],
        "claim_ceiling": "source-bound research shadow only; no public rating, tier list, probability, or deployment authority",
    }
    report["shadow_build"]["effects"] = _effect_metrics(roots, source)
    report_path = output_root / "downstream-diff.json"
    write_downstream_diff_report(report_path, report)
    _write_json(output_root / "shadow-build.json", report["shadow_build"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--public-pack-root", type=Path, default=DEFAULT_PUBLIC_PACK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = build_shadows(
        source_receipt_path=args.source_receipt,
        current_root=args.current_root,
        evaluation_root=args.evaluation_root,
        public_pack_root=args.public_pack_root,
        output_root=args.output_root,
    )
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
