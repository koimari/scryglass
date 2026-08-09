"""Executed, boundary-hashed R-20 candidate primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect

from typing import Any, Mapping, Sequence

import numpy as np

from .checks import ValidationFailure
from .types import canonical_sha256


def _vector(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationFailure(f"{path} must be a numeric vector")
    for index, item in enumerate(value):
        if type(item) is bool or not isinstance(item, (int, float)):
            raise ValidationFailure(f"{path}[{index}] must be numeric")
    array = np.asarray(list(value), dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValidationFailure(f"{path} must be a finite nonempty vector")
    if np.any(array < 0) or np.any(array > 1):
        raise ValidationFailure(f"{path} must lie in [0,1]")
    return array


def _strict_int(value: Any, path: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise ValidationFailure(f"{path} must be a strict int")
    return value


def _strict_float(value: Any, path: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValidationFailure(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValidationFailure(f"{path} must be finite")
    return result


def _strict_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValidationFailure(f"{path} must be a strict bool")
    return value


def _minimum_draws(boundaries: Mapping[str, Any]) -> int:
    value = _strict_int(boundaries.get("minimum_draws"), "minimum_draws")
    if value < 128:
        raise ValidationFailure("minimum_draws must be >=128")
    return value


def _posterior_information(
    method_id: str,
    dependencies: Mapping[str, Any],
    boundaries: Mapping[str, Any],
) -> float:
    minimum = _minimum_draws(boundaries)
    posterior = _vector(dependencies["posterior_draws"], "posterior_draws")
    prior = _vector(dependencies["prior_draws"], "prior_draws")
    if posterior.size < minimum or prior.size < minimum:
        raise ValidationFailure("insufficient registered draws")
    scale = float(np.std(prior, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValidationFailure("prior scale is unidentified")
    if method_id == "posterior_mean_displacement_v1":
        return abs(float(np.mean(posterior) - np.mean(prior))) / scale
    if method_id == "posterior_median_displacement_v1":
        return abs(float(np.median(posterior) - np.median(prior))) / scale
    raise ValidationFailure("unknown posterior-information method")


def _central_width(values: np.ndarray, mass: float) -> float:
    tail = (1.0 - mass) / 2.0
    return float(np.quantile(values, 1.0 - tail) - np.quantile(values, tail))


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _precision(
    method_id: str,
    dependencies: Mapping[str, Any],
    boundaries: Mapping[str, Any],
) -> float:
    minimum = _minimum_draws(boundaries)
    mass = _strict_float(boundaries.get("central_mass"), "central_mass")
    if not 0 < mass < 1:
        raise ValidationFailure("central_mass must lie in (0,1)")
    posterior = _vector(dependencies["posterior_draws"], "posterior_draws")
    reference = _vector(
        dependencies["registered_reference_draws"],
        "registered_reference_draws",
    )
    if posterior.size < minimum or reference.size < minimum:
        raise ValidationFailure("insufficient registered draws")
    if method_id == "central_interval_contraction_v2":
        posterior_scale = _central_width(posterior, mass)
        reference_scale = _central_width(reference, mass)
    elif method_id == "robust_mad_contraction_v1":
        posterior_scale = _mad(posterior)
        reference_scale = _mad(reference)
    else:
        raise ValidationFailure("unknown precision method")
    if reference_scale <= 0:
        raise ValidationFailure("registered reference scale is zero")
    contraction = 1.0 - posterior_scale / reference_scale
    if contraction < -1e-12:
        raise ValidationFailure("posterior precision expanded")
    return float(max(0.0, contraction))


def _source_inputs(dependencies: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = dependencies["source_lineage"]
    context = dependencies["context_registry"]
    fallback = dependencies["fallback_registry"]
    bridge = dependencies["bridge_registry"]
    if not isinstance(source, dict) or set(source) != {"complete", "registered"}:
        raise ValidationFailure("source_lineage shape mismatch")
    if not isinstance(context, dict) or set(context) != {
        "registered", "registry_version", "path"
    }:
        raise ValidationFailure("context_registry shape mismatch")
    if not isinstance(fallback, dict) or set(fallback) != {"used", "profile"}:
        raise ValidationFailure("fallback_registry shape mismatch")
    if not isinstance(bridge, dict) or set(bridge) != {"registered", "bridge_id"}:
        raise ValidationFailure("bridge_registry shape mismatch")

    for value in (
        source["complete"],
        source["registered"],
        context["registered"],
        fallback["used"],
        bridge["registered"],
    ):
        _strict_bool(value, "source/context flag")
    if fallback["profile"] not in {"none", "fallback"}:
        raise ValidationFailure("fallback profile is not registered")
    if fallback["used"] != (fallback["profile"] == "fallback"):
        raise ValidationFailure("fallback used/profile is inconsistent")

    issues: list[str] = []
    if not source["registered"]:
        issues.append("source_unregistered")
    if not source["complete"]:
        issues.append("lineage_incomplete")
    if not context["registered"]:
        issues.append("context_missing")
    if not bridge["registered"]:
        issues.append("bridge_missing")
    if fallback["used"]:
        issues.append("fallback_used")
    return {
        "source_registered": source["registered"],
        "lineage_complete": source["complete"],
        "context_registered": context["registered"],
        "bridge_registered": bridge["registered"],
        "fallback_used": fallback["used"],
        "fallback_profile": fallback["profile"],
    }, issues


def _source_context(
    method_id: str,
    dependencies: Mapping[str, Any],
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    if _strict_bool(
        boundaries.get("fallback_forbids_high"), "fallback_forbids_high"
    ) is not True:
        raise ValidationFailure("fallback_forbids_high must be exactly true")
    flags, issues = _source_inputs(dependencies)
    if method_id == "source_context_strict_v2":
        status = "complete" if not issues else "unavailable"
        usable = not issues
    elif method_id == "source_context_typed_partial_v1":
        severe = "fallback_used" in issues or len(issues) > 1
        status = "complete" if not issues else ("unavailable" if severe else "limited")
        usable = not severe
    else:
        raise ValidationFailure("unknown source/context method")
    return {
        "status": status,
        "usable": usable,
        "high_eligible": not issues,
        "issues": issues,
        "flags": flags,
    }


def _method_complexity(func: Any) -> int:
    source = inspect.getsource(func)
    tree = ast.parse(source)
    return sum(1 for _ in ast.walk(tree))


METHOD_SPECS = {
    "posterior_mean_displacement_v1": {
        "family": "posterior_information",
        "units": "prior_standard_deviations",
        "dependencies": ("posterior_draws", "prior_draws"),
        "replay": _posterior_information,
    },
    "posterior_median_displacement_v1": {
        "family": "posterior_information",
        "units": "prior_standard_deviations",
        "dependencies": ("posterior_draws", "prior_draws"),
        "replay": _posterior_information,
    },
    "central_interval_contraction_v2": {
        "family": "precision",
        "units": "fraction_reference_central_interval",
        "dependencies": ("posterior_draws", "registered_reference_draws"),
        "replay": _precision,
    },
    "robust_mad_contraction_v1": {
        "family": "precision",
        "units": "fraction_reference_mad",
        "dependencies": ("posterior_draws", "registered_reference_draws"),
        "replay": _precision,
    },
    "source_context_strict_v2": {
        "family": "source_context_coverage",
        "units": "typed_source_context_status",
        "dependencies": (
            "source_lineage",
            "context_registry",
            "fallback_registry",
            "bridge_registry",
        ),
        "replay": _source_context,
    },
    "source_context_typed_partial_v1": {
        "family": "source_context_coverage",
        "units": "typed_source_context_status",
        "dependencies": (
            "source_lineage",
            "context_registry",
            "fallback_registry",
            "bridge_registry",
        ),
        "replay": _source_context,
    },
}

METHOD_COMPLEXITY = {
    method_id: _method_complexity(
        METHOD_SPECS[method_id]["replay"],
    )
    for method_id in METHOD_SPECS
}


def replay_foundation_method(
    *,
    method_id: str,
    dependencies: Mapping[str, Any],
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Low-level primitive. It is not an authorized candidate replay."""

    spec = METHOD_SPECS.get(method_id)
    if spec is None:
        raise ValidationFailure("unregistered R-20 method")
    if set(dependencies) != set(spec["dependencies"]):
        raise ValidationFailure("dependency set is missing, extra, or substituted")
    if not isinstance(boundaries, Mapping):
        raise ValidationFailure("boundaries must be a mapping")
    value = spec["replay"](method_id, dependencies, boundaries)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValidationFailure("candidate emitted nonfinite value")
    if not isinstance(value, (float, int, dict)):
        raise ValidationFailure("candidate replay emitted unexpected type")
    return {
        "method_id": method_id,
        "family": spec["family"],
        "units": spec["units"],
        "value": float(value) if isinstance(value, (float, int)) else value,
        "executed_boundary_sha256": canonical_sha256(dict(boundaries)),
        "authorized_candidate": False,
        "replay_ok": True,
        "method_complexity": METHOD_COMPLEXITY[method_id],
    }


def replay_registered_precision_batch(
    *,
    method_id: str,
    posterior_draws: np.ndarray,
    reference_draws: np.ndarray,
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Vectorized replay of the exact registered precision eligibility rule."""

    if method_id not in {
        "central_interval_contraction_v2",
        "robust_mad_contraction_v1",
    }:
        raise ValidationFailure("batch replay requires a registered precision method")
    minimum = _minimum_draws(boundaries)
    mass = _strict_float(boundaries.get("central_mass"), "central_mass")
    if not 0 < mass < 1:
        raise ValidationFailure("central_mass must lie in (0,1)")
    posterior = np.asarray(posterior_draws)
    reference = np.asarray(reference_draws)
    if (
        posterior.ndim != 2
        or reference.ndim != 2
        or posterior.shape != reference.shape
        or posterior.shape[1] < minimum
        or not np.issubdtype(posterior.dtype, np.floating)
        or not np.issubdtype(reference.dtype, np.floating)
        or not np.isfinite(posterior).all()
        or not np.isfinite(reference).all()
        or np.any(posterior < 0)
        or np.any(posterior > 1)
        or np.any(reference < 0)
        or np.any(reference > 1)
    ):
        raise ValidationFailure("batch precision draws fail exact support/schema")
    if method_id == "central_interval_contraction_v2":
        tail = (1.0 - mass) / 2.0
        posterior_scale = np.quantile(
            posterior,
            1.0 - tail,
            axis=1,
        ) - np.quantile(posterior, tail, axis=1)
        reference_scale = np.quantile(
            reference,
            1.0 - tail,
            axis=1,
        ) - np.quantile(reference, tail, axis=1)
    else:
        posterior_median = np.median(posterior, axis=1)
        reference_median = np.median(reference, axis=1)
        posterior_scale = np.median(
            np.abs(posterior - posterior_median[:, None]),
            axis=1,
        )
        reference_scale = np.median(
            np.abs(reference - reference_median[:, None]),
            axis=1,
        )
    if np.any(reference_scale <= 0):
        raise ValidationFailure("registered reference scale is zero")
    raw_contraction = 1.0 - posterior_scale / reference_scale
    accepted = raw_contraction >= -1e-12
    values = np.where(accepted, np.maximum(0.0, raw_contraction), np.nan)
    return {
        "method_id": method_id,
        "boundary_sha256": canonical_sha256(dict(boundaries)),
        "raw_contraction": raw_contraction,
        "accepted": accepted,
        "value": values,
    }


__all__ = [
    "METHOD_SPECS",
    "METHOD_COMPLEXITY",
    "replay_foundation_method",
    "replay_registered_precision_batch",
]
