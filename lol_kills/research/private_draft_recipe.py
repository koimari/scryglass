"""Hash-bound R9E composite diagnostics for the private runner.

This module gives the local private scorer a separate binding for the R9E
research checkpoint. It does not change the active map-win model. R9E emits a
development composite because its probability includes strength controls.
It is not a Draft Score estimand.

The binding is intended to live in the owner-only private model directory.
The R9E checkpoint and its source receipts remain outside the public pack.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "scryglass:private-r9e-development-composite:v1"
R9E_CANDIDATE_ID = "R9E_d4_ss"
R9E_RECORDED_AUC = 0.70681
R9E_RECORDED_BRIER = 0.21708
R9E_RECORDED_LOG_LOSS = 0.62330

ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
R9E_ROLE_ORDER = ("top", "jng", "mid", "bot", "sup")
ROLE_MAP = dict(zip(ROLE_ORDER, R9E_ROLE_ORDER))


class PrivateDraftRecipeError(ValueError):
    """Raised when a private descriptive recipe cannot be trusted."""


@dataclass(frozen=True)
class PrivateDraftRecipeBinding:
    """Validated local binding for one R9E composite checkpoint."""

    binding_path: Path
    module_path: Path
    manifest_path: Path
    cache_dir: Path
    binding_sha256: str
    module_sha256: str
    manifest_sha256: str
    candidate_id: str
    status: str
    authority: str
    recorded_auc: float
    recorded_brier: float
    recorded_log_loss: float
    fit_through: str
    patch_token: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PrivateDraftRecipeError(f"cannot read recipe source: {path}") from error
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PrivateDraftRecipeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PrivateDraftRecipeError(f"{label} must contain an object: {path}")
    return value


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_path(value: Any, field: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise PrivateDraftRecipeError(f"{field} must be an absolute path")
    raw_path = Path(value)
    if raw_path.is_symlink():
        raise PrivateDraftRecipeError(f"{field} is a symlink: {raw_path}")
    path = raw_path.resolve()
    if directory:
        valid = path.is_dir()
    else:
        valid = path.is_file()
    if not valid:
        raise PrivateDraftRecipeError(f"{field} is missing or is a symlink: {path}")
    return path


def _number(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise PrivateDraftRecipeError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise PrivateDraftRecipeError(f"{field} must be finite")
    return parsed


def load_private_descriptive_r9e_binding(path: Path) -> PrivateDraftRecipeBinding:
    """Validate one owner-only R9E binding and all hash-addressed sources."""

    raw_binding_path = Path(path)
    if raw_binding_path.is_symlink():
        raise PrivateDraftRecipeError(f"R9E binding is a symlink: {raw_binding_path}")
    binding_path = raw_binding_path.resolve()
    if not binding_path.is_file():
        raise PrivateDraftRecipeError(f"R9E binding is missing: {binding_path}")
    payload = _read_json(binding_path, "R9E binding")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PrivateDraftRecipeError("R9E binding schema changed")
    if payload.get("candidate_id") != R9E_CANDIDATE_ID:
        raise PrivateDraftRecipeError("R9E binding candidate is not R9E_d4_ss")
    if payload.get("mode") != "development_composite":
        raise PrivateDraftRecipeError("R9E binding mode is not development_composite")
    if payload.get("status") != "development_only":
        raise PrivateDraftRecipeError("R9E descriptive binding must stay development_only")
    if payload.get("authority") != "unavailable":
        raise PrivateDraftRecipeError("R9E descriptive binding must keep authority unavailable")
    claim_ceiling = payload.get("claim_ceiling")
    if not isinstance(claim_ceiling, Mapping):
        raise PrivateDraftRecipeError("R9E binding claim ceiling is missing")
    expected_false = ("descriptive_draft_score", "public_probability", "recommendation", "betting")
    if claim_ceiling.get("composite_diagnostic") is not True or any(
        claim_ceiling.get(key) is not False for key in expected_false
    ):
        raise PrivateDraftRecipeError("R9E binding claim ceiling is not composite-only")
    if payload.get("artifact_sha256") != _artifact_sha256(payload):
        raise PrivateDraftRecipeError("R9E binding hash changed")

    module_info = payload.get("module")
    manifest_info = payload.get("manifest")
    if not isinstance(module_info, Mapping) or not isinstance(manifest_info, Mapping):
        raise PrivateDraftRecipeError("R9E binding source receipts are missing")
    module_path = _required_path(module_info.get("path"), "R9E module path")
    manifest_path = _required_path(manifest_info.get("path"), "R9E manifest path")
    cache_dir = _required_path(payload.get("cache_dir"), "R9E cache directory", directory=True)
    if manifest_path != cache_dir / "manifest.json":
        raise PrivateDraftRecipeError("R9E manifest must be the bound cache manifest")
    module_sha256 = str(module_info.get("sha256") or "")
    manifest_sha256 = str(manifest_info.get("sha256") or "")
    if not hashlib.sha256(module_path.read_bytes()).hexdigest() == module_sha256:
        raise PrivateDraftRecipeError("R9E module hash changed")
    if not hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha256:
        raise PrivateDraftRecipeError("R9E manifest hash changed")
    manifest = _read_json(manifest_path, "R9E manifest")
    if manifest.get("schema_version") != "scryglass:r9e-fast-path:v1":
        raise PrivateDraftRecipeError("R9E checkpoint schema changed")
    if manifest.get("candidate_id") != R9E_CANDIDATE_ID:
        raise PrivateDraftRecipeError("R9E checkpoint candidate is not R9E_d4_ss")
    if manifest.get("status") != "development_only":
        raise PrivateDraftRecipeError("R9E checkpoint must stay development_only")
    recorded = manifest.get("recorded_metrics")
    if not isinstance(recorded, Mapping):
        raise PrivateDraftRecipeError("R9E checkpoint metrics are missing")
    for field, expected in (
        ("auc", R9E_RECORDED_AUC),
        ("brier", R9E_RECORDED_BRIER),
        ("log_loss", R9E_RECORDED_LOG_LOSS),
    ):
        if _number(recorded.get(field), f"R9E recorded {field}") != expected:
            raise PrivateDraftRecipeError(f"R9E recorded {field} changed")
    source_policy = manifest.get("source_policy")
    if not isinstance(source_policy, Mapping) or source_policy.get("public_authority") != "unavailable":
        raise PrivateDraftRecipeError("R9E checkpoint public authority ceiling changed")
    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise PrivateDraftRecipeError("R9E training receipt is missing")
    patch_token = str(training.get("patch_token") or "")
    fit_through = str(training.get("fit_through") or "")
    if not patch_token or not fit_through:
        raise PrivateDraftRecipeError("R9E training boundary is incomplete")

    return PrivateDraftRecipeBinding(
        binding_path=binding_path,
        module_path=module_path,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        binding_sha256=_sha256_path(binding_path),
        module_sha256=module_sha256,
        manifest_sha256=manifest_sha256,
        candidate_id=R9E_CANDIDATE_ID,
        status="development_only",
        authority="unavailable",
        recorded_auc=R9E_RECORDED_AUC,
        recorded_brier=R9E_RECORDED_BRIER,
        recorded_log_loss=R9E_RECORDED_LOG_LOSS,
        fit_through=fit_through,
        patch_token=patch_token,
    )


def build_r9e_query(
    registration: Mapping[str, Any],
    draft: Mapping[str, Mapping[str, str]],
    *,
    date: str,
    patch: str,
    mu_diff: float,
    sigma_pair: float,
    player_elo: Mapping[str, float],
    player_ratings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Convert the private exact-roster input into the R9E query contract."""

    teams = registration.get("teams")
    if not isinstance(teams, Sequence) or isinstance(teams, (str, bytes)) or len(teams) != 2:
        raise PrivateDraftRecipeError("R9E query needs two registered teams")
    output: dict[str, Any] = {
        "date": str(date),
        "patch": str(patch),
        "blue_team": None,
        "red_team": None,
        "mu_diff": _number(mu_diff, "R9E mu_diff"),
        "sigma_pair": _number(sigma_pair, "R9E sigma_pair"),
        "player_elo": {
            "p": _number(player_elo.get("p"), "R9E player_elo.p"),
            "sigma": _number(player_elo.get("sigma"), "R9E player_elo.sigma"),
        },
        "bans": {"blue": [], "red": []},
        "player_ratings": dict(player_ratings or {}),
    }
    if not 0.0 <= output["player_elo"]["p"] <= 1.0:
        raise PrivateDraftRecipeError("R9E player_elo.p must be between zero and one")
    for team in teams:
        if not isinstance(team, Mapping) or team.get("side") not in ("blue", "red"):
            raise PrivateDraftRecipeError("R9E team side is invalid")
        side = str(team["side"])
        name = team.get("organization_name")
        players = team.get("players")
        if not isinstance(name, str) or not name:
            raise PrivateDraftRecipeError(f"R9E {side} team name is invalid")
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)) or len(players) != 5:
            raise PrivateDraftRecipeError(f"R9E {side} roster must contain five players")
        by_role: dict[str, Mapping[str, Any]] = {}
        for player in players:
            if not isinstance(player, Mapping) or player.get("role") not in ROLE_ORDER:
                raise PrivateDraftRecipeError(f"R9E {side} roster role is invalid")
            role = str(player["role"])
            if role in by_role:
                raise PrivateDraftRecipeError(f"R9E {side} roster repeats a role")
            by_role[role] = player
        if set(by_role) != set(ROLE_ORDER):
            raise PrivateDraftRecipeError(f"R9E {side} roster roles are incomplete")
        side_draft = draft.get(side)
        if not isinstance(side_draft, Mapping) or set(side_draft) != set(ROLE_ORDER):
            raise PrivateDraftRecipeError(f"R9E {side} draft roles are incomplete")
        query_side: dict[str, dict[str, str]] = {}
        for role in ROLE_ORDER:
            player_name = by_role[role].get("display_name")
            champion = side_draft.get(role)
            if not isinstance(player_name, str) or not player_name:
                raise PrivateDraftRecipeError(f"R9E {side} player name is invalid")
            if not isinstance(champion, str) or not champion:
                raise PrivateDraftRecipeError(f"R9E {side} champion is invalid")
            query_side[ROLE_MAP[role]] = {"player": player_name, "champion": champion}
        output[f"{side}_team"] = name
        output[side] = query_side
    if not output["blue_team"] or not output["red_team"]:
        raise PrivateDraftRecipeError("R9E query needs blue and red teams")
    return output


def _load_module(path: Path) -> Any:
    module_name = f"_scryglass_private_r9e_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PrivateDraftRecipeError(f"cannot load R9E module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        raise PrivateDraftRecipeError(f"R9E module failed to load: {path}") from error
    return module


def score_private_descriptive_r9e(
    binding_path: Path,
    query: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the bound R9E checkpoint as a descriptive private diagnostic."""

    binding = load_private_descriptive_r9e_binding(binding_path)
    module = _load_module(binding.module_path)
    loader = getattr(module, "load_checkpoint", None)
    scorer = getattr(module, "score_query", None)
    if not callable(loader) or not callable(scorer):
        raise PrivateDraftRecipeError("R9E module does not expose the checkpoint API")
    try:
        checkpoint = loader(binding.cache_dir)
        result = scorer(binding.cache_dir, dict(query))
    except Exception as error:  # noqa: BLE001
        raise PrivateDraftRecipeError("R9E checkpoint scoring failed") from error
    if not isinstance(result, Mapping):
        raise PrivateDraftRecipeError("R9E score result is invalid")
    if (
        result.get("candidate_id") != R9E_CANDIDATE_ID
        or result.get("status") != "development_only"
        or result.get("authority") != "unavailable"
    ):
        raise PrivateDraftRecipeError("R9E score result widened its authority")
    if getattr(checkpoint, "manifest", {}).get("candidate_id") != R9E_CANDIDATE_ID:
        raise PrivateDraftRecipeError("R9E checkpoint identity changed during scoring")
    blue_probability = _number(result.get("blue_probability"), "R9E blue probability")
    red_probability = _number(result.get("red_probability"), "R9E red probability")
    if not 0.0 <= blue_probability <= 1.0 or not 0.0 <= red_probability <= 1.0:
        raise PrivateDraftRecipeError("R9E probabilities are outside zero to one")
    if abs(blue_probability + red_probability - 1.0) > 1e-8:
        raise PrivateDraftRecipeError("R9E side probabilities do not sum to one")
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": R9E_CANDIDATE_ID,
        "mode": "development_composite",
        "estimand": "composite_map_probability",
        "status": "development_only",
        "authority": "unavailable",
        "claim_ceiling": {
            "descriptive_draft_score": False,
            "composite_diagnostic": True,
            "public_probability": False,
            "recommendation": False,
            "betting": False,
        },
        "binding": {
            "binding_sha256": binding.binding_sha256,
            "module_sha256": binding.module_sha256,
            "manifest_sha256": binding.manifest_sha256,
        },
        "recorded_metrics": {
            "auc": binding.recorded_auc,
            "brier": binding.recorded_brier,
            "log_loss": binding.recorded_log_loss,
        },
        "fit_through": binding.fit_through,
        "patch_token": binding.patch_token,
        "control_fields": ["mu_diff", "sigma_pair", "player_elo"],
        "development_composite": {
            "blue_probability": blue_probability,
            "red_probability": red_probability,
            "edge_pp": round(100.0 * (blue_probability - red_probability), 6),
        },
        "query": dict(result.get("query") or {}),
        "note": (
            "R9E includes team and player strength controls. It is a development "
            "composite diagnostic, not a Draft Score estimand."
        ),
    }


def score_registered_draft_with_private_r9e(
    binding_path: Path,
    registration: Mapping[str, Any],
    draft: Mapping[str, Mapping[str, str]],
    *,
    date: str,
    patch: str,
    mu_diff: float,
    sigma_pair: float,
    player_elo: Mapping[str, float],
    player_ratings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build and score one exact private roster with the bound recipe."""

    query = build_r9e_query(
        registration,
        draft,
        date=date,
        patch=patch,
        mu_diff=mu_diff,
        sigma_pair=sigma_pair,
        player_elo=player_elo,
        player_ratings=player_ratings,
    )
    return score_private_descriptive_r9e(binding_path, query)
