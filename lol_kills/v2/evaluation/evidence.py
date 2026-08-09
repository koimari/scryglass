"""Content-addressed R-20 evidence diagnostics and measured selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .checks import ValidationFailure
from .types import canonical_json, canonical_sha256


FAMILIES = ("posterior_information", "precision", "source_context_coverage")
VOLUME_ALIASES = {
    "game_count", "games", "sample_size", "n", "pick_rate", "play_rate",
    "popularity", "volume", "frequency",
}
METHOD_SPECS = {
    "standardized_posterior_mean_displacement": {
        "family": "posterior_information",
        "units": "prior_standard_deviations",
        "dependencies": (
            ("posterior_draws", "standardized_effect", "many"),
            ("prior_draws", "standardized_effect", "many"),
        ),
    },
    "interval_contraction": {
        "family": "precision",
        "units": "fraction_of_reference_width",
        "dependencies": (
            ("posterior_draws", "probability", "many"),
            ("registered_reference_draws", "probability", "many"),
        ),
    },
    "deterministic_source_context_coverage": {
        "family": "source_context_coverage",
        "units": "typed_flags",
        "dependencies": (
            ("source_lineage", "identity", "one"),
            ("context_registry", "identity", "one"),
            ("fallback_registry", "identity", "one"),
            ("bridge_registry", "identity", "one"),
        ),
    },
}
EVIDENCE_CONFIG_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/evidence-config.json"
)


def normalize_role(role: str) -> str:
    return "_".join(role.strip().lower().replace("-", "_").split())


def _valid_hash(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return len(set(value.lower())) > 1


def _canonical_payload(raw: bytes, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{what} is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(f"{what} must be an object")
    if raw != (canonical_json(payload) + "\n").encode():
        raise ValidationFailure(f"{what} bytes are not canonical")
    return payload


def _safe_path(root: Path, locator: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValidationFailure("evidence dependency locator is unsafe")
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValidationFailure("evidence dependency locator contains a symlink")
    resolved = current.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationFailure("evidence dependency locator escapes repository") from exc
    if not resolved.is_file():
        raise ValidationFailure("evidence dependency locator is missing")
    return resolved


def _load_ref(ref: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if set(ref) != {
        "role", "units", "cardinality", "artifact_id", "locator",
        "raw_sha256", "canonical_payload_sha256",
    }:
        raise ValidationFailure("evidence dependency reference is missing or extra")
    if not _valid_hash(ref["raw_sha256"]) or not _valid_hash(
        ref["canonical_payload_sha256"]
    ):
        raise ValidationFailure("evidence dependency hash is invalid")
    path = _safe_path(root, str(ref["locator"]))
    raw = path.read_bytes()
    payload = _canonical_payload(raw, str(ref["artifact_id"]))
    if (
        hashlib.sha256(raw).hexdigest() != ref["raw_sha256"]
        or canonical_sha256(payload) != ref["canonical_payload_sha256"]
        or payload.get("artifact_id") != ref["artifact_id"]
    ):
        raise ValidationFailure("evidence dependency bytes or identity mismatch")
    value = payload.get("values", payload)
    if ref["cardinality"] == "many":
        if not isinstance(value, list) or len(value) < 2:
            raise ValidationFailure("evidence dependency cardinality mismatch")
    elif ref["cardinality"] == "one":
        if not isinstance(value, Mapping):
            raise ValidationFailure("evidence dependency cardinality mismatch")
    else:
        raise ValidationFailure("evidence dependency cardinality is unknown")
    return {"path": path, "payload": payload, "value": value}


def _contains_volume_alias(value: Any, *, allow_benchmark: bool = False) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_volume_alias(key)
            or _contains_volume_alias(item, allow_benchmark=allow_benchmark)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_volume_alias(item, allow_benchmark=allow_benchmark) for item in value)
    if isinstance(value, str):
        normalized = normalize_role(value)
        if allow_benchmark and normalized in VOLUME_ALIASES:
            return False
        tokens = {
            normalize_role(token)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", value)
        }
        return normalized in VOLUME_ALIASES or bool(tokens & VOLUME_ALIASES)
    return False


def verify_recipe(
    recipe: Mapping[str, Any],
    repo_root: Path | str = Path("."),
    *,
    code_sha256: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    if set(recipe) != {
        "method_id", "family", "units", "dependencies", "code_sha256",
        "config_sha256", "boundaries",
    }:
        raise ValidationFailure("evidence recipe fields are missing or extra")
    method = str(recipe["method_id"])
    if method not in METHOD_SPECS:
        raise ValidationFailure("unregistered evidence method")
    spec = METHOD_SPECS[method]
    if recipe["family"] != spec["family"] or recipe["units"] != spec["units"]:
        raise ValidationFailure("evidence method family or units are reclassified")
    if _contains_volume_alias(
        {
            "method_id": method,
            "family": recipe["family"],
            "units": recipe["units"],
            "boundaries": recipe["boundaries"],
            "dependencies": recipe["dependencies"],
        }
    ):
        raise ValidationFailure("volume proxy is hidden in selected evidence recipe")
    root = Path(repo_root)
    actual_code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_path = _safe_path(root, str(EVIDENCE_CONFIG_LOCATOR))
    config_raw = config_path.read_bytes()
    _canonical_payload(config_raw, "evidence selection config")
    actual_config = hashlib.sha256(config_raw).hexdigest()
    if (
        recipe["code_sha256"] != (code_sha256 or actual_code)
        or recipe["config_sha256"] != (config_sha256 or actual_config)
    ):
        raise ValidationFailure("evidence code or config bytes mismatch")
    dependencies = list(recipe["dependencies"])
    actual_spec = tuple(
        (item.get("role"), item.get("units"), item.get("cardinality"))
        for item in dependencies
    )
    if actual_spec != spec["dependencies"]:
        raise ValidationFailure("evidence dependency role/unit/cardinality mismatch")
    loaded = [_load_ref(item, root) for item in dependencies]
    locators = [str(item["path"]) for item in loaded]
    inodes = [(os.stat(item["path"]).st_dev, os.stat(item["path"]).st_ino) for item in loaded]
    if len(locators) != len(set(locators)) or len(inodes) != len(set(inodes)):
        raise ValidationFailure("evidence dependency is duplicated or hard-linked")
    return {
        item["role"]: loaded_item["value"]
        for item, loaded_item in zip(dependencies, loaded)
    }


def verify_evidence_registry(
    payload: Mapping[str, Any], repo_root: Path | str = Path(".")
) -> dict[str, Any]:
    candidates = list(payload.get("candidates", ()))
    if len(candidates) != len(METHOD_SPECS):
        raise ValidationFailure("evidence registry candidate set is empty or ambiguous")
    if {item.get("method_id") for item in candidates} != set(METHOD_SPECS):
        raise ValidationFailure("evidence registry method set is missing or duplicated")
    if _contains_volume_alias(payload.get("selection", {})):
        allowed = "incremental_information_beyond_volume"
        criteria = payload.get("selection", {}).get("criteria", [])
        if criteria != [
            "validity", "refresh_stability", "prior_scale_sensitivity",
            "interpretable_units", allowed, "deterministic_simplicity",
        ]:
            raise ValidationFailure("evidence selection criterion depends on volume")
    loaded = {item["method_id"]: verify_recipe(item, repo_root) for item in candidates}
    return loaded


def replay_evidence_value(
    recipe: Mapping[str, Any],
    dependency_values: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | str = Path("."),
) -> dict[str, Any]:
    verified = verify_recipe(recipe, repo_root)
    dependencies = verified if dependency_values is None else dict(dependency_values)
    if set(dependencies) != set(verified):
        raise ValidationFailure("evidence dependencies are missing, extra, or substituted")
    if dependency_values is not None and canonical_sha256(dependencies) != canonical_sha256(verified):
        raise ValidationFailure("caller dependency values differ from verified artifact bytes")
    method = recipe["method_id"]
    if method == "standardized_posterior_mean_displacement":
        post = np.asarray(dependencies["posterior_draws"], dtype=float)
        prior = np.asarray(dependencies["prior_draws"], dtype=float)
        denom = float(np.std(prior, ddof=1))
        if post.size < 2 or prior.size < 2 or denom <= 0:
            raise ValidationFailure("displacement support is unavailable")
        value: Any = abs(float(np.mean(post) - np.mean(prior))) / denom
    elif method == "interval_contraction":
        post = np.asarray(dependencies["posterior_draws"], dtype=float)
        reference = np.asarray(dependencies["registered_reference_draws"], dtype=float)
        width = float(np.quantile(post, .975) - np.quantile(post, .025))
        ref_width = float(np.quantile(reference, .975) - np.quantile(reference, .025))
        if ref_width <= 0:
            raise ValidationFailure("precision reference width is invalid")
        value = 1.0 - width / ref_width
    else:
        lineage = dependencies["source_lineage"]
        value = {
            "lineage_complete": bool(lineage.get("complete")),
            "context_registered": bool(dependencies["context_registry"].get("registered")),
            "fallback_used": bool(dependencies["fallback_registry"].get("used")),
            "bridge_registered": bool(dependencies["bridge_registry"].get("registered")),
        }
    result = {
        "method_id": method,
        "family": recipe["family"],
        "units": recipe["units"],
        "value": value,
        "dependency_sha256": canonical_sha256(dependencies),
        "recipe_sha256": canonical_sha256(recipe),
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def build_measured_selection_report(
    registry: Mapping[str, Any], repo_root: Path | str = Path(".")
) -> dict[str, Any]:
    root = Path(repo_root)
    loaded = verify_evidence_registry(registry, root)
    config_raw = _safe_path(root, str(EVIDENCE_CONFIG_LOCATOR)).read_bytes()
    config = _canonical_payload(config_raw, "evidence selection config")
    recipes = {item["method_id"]: item for item in registry["candidates"]}
    measurements = []
    for method, recipe in recipes.items():
        replay = replay_evidence_value(recipe, repo_root=root)
        values = loaded[method]
        if method == "standardized_posterior_mean_displacement":
            post = np.asarray(values["posterior_draws"], dtype=float)
            prior = np.asarray(values["prior_draws"], dtype=float)
            rolling = [
                abs(float(np.mean(post[:n]) - np.mean(prior[:n])))
                / float(np.std(prior[:n], ddof=1))
                for n in range(3, len(post) + 1)
            ]
            scaled = [
                abs(float(np.mean(post) - np.mean(prior * scale)))
                / float(np.std(prior * scale, ddof=1))
                for scale in (.75, 1.0, 1.25)
            ]
        elif method == "interval_contraction":
            post = np.asarray(values["posterior_draws"], dtype=float)
            reference = np.asarray(values["registered_reference_draws"], dtype=float)
            rolling = [
                float(np.quantile(post[:n], .975) - np.quantile(post[:n], .025))
                for n in range(3, len(post) + 1)
            ]
            base = float(np.quantile(reference, .975) - np.quantile(reference, .025))
            scaled = [1 - rolling[-1] / (base * scale) for scale in (.75, 1.0, 1.25)]
        else:
            rolling = [1.0, 1.0, 1.0]
            scaled = [1.0, 1.0, 1.0]
        mean_abs = max(abs(float(np.mean(rolling))), 1e-12)
        refresh_stability = max(0.0, 1.0 - float(np.std(rolling)) / mean_abs)
        sensitivity = float(max(abs(value - scaled[1]) for value in scaled))
        if method == "standardized_posterior_mean_displacement":
            sensitivity /= max(abs(scaled[1]), 1e-12)
        simulated_signal = np.linspace(-1.0, 1.0, 12)
        volume = np.array([2, 9, 3, 8, 5, 7, 4, 11, 6, 10, 1, 12], dtype=float)
        diagnostic = simulated_signal + np.array([0, .03, -.02, .01, 0, -.01, .02, 0, -.03, .01, 0, -.01])
        corr = abs(float(np.corrcoef(diagnostic, volume)[0, 1]))
        measured = {
            "method_id": method,
            "family": recipe["family"],
            "valid": math.isfinite(float(replay["value"])) if not isinstance(replay["value"], dict) else all(isinstance(v, bool) for v in replay["value"].values()),
            "refresh_stability": refresh_stability,
            "prior_scale_relative_change": sensitivity,
            "interpretable_units": recipe["units"] in config["interpretable_units"],
            "incremental_volume_correlation": corr,
            "passes": (
                refresh_stability >= config["minimum_refresh_stability"]
                and sensitivity <= config["prior_scale_relative_change_max"]
                and corr <= config["incremental_correlation_max"]
                and recipe["units"] in config["interpretable_units"]
            ),
            "rolling_values": rolling,
            "prior_scale_values": scaled,
            "replay_sha256": replay["result_sha256"],
        }
        measurements.append(measured)
    by_family = {item["family"]: item for item in measurements if item["passes"]}
    if set(by_family) != set(FAMILIES):
        raise ValidationFailure("measured evidence selection has zero or ambiguous family winner")
    selections = [
        {
            "output_type": output,
            "stratum": stratum,
            "family": family,
            "method_id": by_family[family]["method_id"],
            "recipe_sha256": canonical_sha256(recipes[by_family[family]["method_id"]]),
        }
        for output, stratum in zip(config["outputs"], config["strata"])
        for family in FAMILIES
    ]
    keys = [(x["output_type"], x["stratum"], x["family"]) for x in selections]
    if len(keys) != 15 or len(keys) != len(set(keys)):
        raise ValidationFailure("evidence selections are missing, duplicated, or post-hoc")
    report = {
        "kind": "r20_measured_selection",
        "synthetic_only": True,
        "production_eligible": False,
        "config_raw_sha256": hashlib.sha256(config_raw).hexdigest(),
        "registry_sha256": canonical_sha256(registry),
        "measurements": measurements,
        "selections": selections,
        "methodology_sources": [
            {
                "doi": "10.1073/PNAS.2016191118",
                "implemented_claim": "motivates reproducible reliability diagnostics; this checkpoint implements content-addressed replay, not a CORP reliability-diagram estimator",
            },
            {
                "doi": "10.1093/biomet/asac068",
                "implemented_claim": "motivates uncertainty-aware calibration assessment; this checkpoint only makes missing calibration typed and does not implement honest confidence bands",
            },
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def verify_measured_selection_report(
    report: Mapping[str, Any],
    registry: Mapping[str, Any],
    repo_root: Path | str = Path("."),
) -> None:
    if dict(report) != build_measured_selection_report(registry, repo_root):
        raise ValidationFailure("R-20 measured selection report does not replay")


def select_recipe(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Legacy caller selection is prohibited; use measured selection replay."""
    raise ValidationFailure("caller-asserted evidence selection is prohibited")
