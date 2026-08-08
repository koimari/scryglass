"""Loader-issued immutable authority for the synthetic R-20 foundation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import inspect
from types import MappingProxyType
from typing import Any, Mapping
import weakref

import numpy as np

from .checks import ValidationFailure
from .r20_foundation_algorithms import (
    METHOD_COMPLEXITY,
    METHOD_SPECS,
    replay_foundation_method,
)
from .r20_foundation_generator import (
    FAMILY_DEFAULT,
    FOUNDATION_DRAWS,
    FOUNDATION_FOLDS,
    FOUNDATION_MAPS_PER_SERIES,
    INITIAL_TRAIN_SERIES_PER_CELL,
    OUTPUT_STRATA,
    REGIMES,
    SOURCE_CONTEXT_PATTERNS,
    REGISTERED_VOLUME_FIELDS,
    TEST_SERIES_PER_CELL_PER_FOLD,
    build_prequential_plan,
    build_r20_benchmark,
    verify_prequential_plan,
)
from .r20_foundation_inference import (
    INFERENCE_ADAPTER_ID,
    INFERENCE_SEED,
    POSTERIOR_DRAWS,
    PRIOR_ALPHA,
    PRIOR_BETA,
    REFERENCE_MODES,
    infer_beta_binomial,
    monte_carlo_width_design,
)
from .types import CONTRACT_TREE_SHA256, canonical_json, canonical_sha256


AUTHORITY_ARTIFACT_ID = "scryglass:b2:r20-foundation-authority:v2"
BENCHMARK_ARTIFACT_ID = "scryglass:b2:r20-foundation-benchmark:v2"
CANDIDATE_REGISTRY_ARTIFACT_ID = (
    "scryglass:b2:r20-foundation-evidence-candidate-registry:v2"
)
FOUNDATION_CONFIG_ARTIFACT_ID = "scryglass:b2:r20-foundation-config:v2"
AUTHORITY_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-foundation-authority.json")
BENCHMARK_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-foundation-benchmark.json")
CANDIDATE_REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/r20-foundation-evidence-candidate-registry.json"
)
FOUNDATION_CONFIG_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/r20-foundation-config.json"
)

_DEPENDENCY_LOCATORS = {
    role: Path(f"data/lol/v2/evaluation/b2/r20-foundation/dependencies/{role}.json")
    for role in {
        role
        for spec in METHOD_SPECS.values()
        for role in spec["dependencies"]
    }
}

_IMPORTED_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_FAMILIES = {
    "posterior_information",
    "precision",
    "source_context_coverage",
}
_REQUIRED_FAMILIES_LIST = (
    "posterior_information",
    "precision",
    "source_context_coverage",
)
_SOURCE_FILES = {
    "loader": Path("lol_kills/v2/evaluation/r20_foundation.py"),
    "generator": Path("lol_kills/v2/evaluation/r20_foundation_generator.py"),
    "inference_adapter": Path(
        "lol_kills/v2/evaluation/r20_foundation_inference.py"
    ),
    "algorithms": Path("lol_kills/v2/evaluation/r20_foundation_algorithms.py"),
    "artifact_generator": Path(
        "lol_kills/v2/evaluation/generate_r20_foundation_artifacts.py"
    ),
    "types": Path("lol_kills/v2/evaluation/types.py"),
    "checks": Path("lol_kills/v2/evaluation/checks.py"),
}
_VOLUME_BASIS = {
    "basis_id": "r20-volume-basis-v2",
    "basis_fields": list(REGISTERED_VOLUME_FIELDS),
    "volume_input_key": "volume_inputs",
    "imputation": "median",
    "scale": "training_std",
    "training_center": "per_output",
    "missing_policy": "drop_rank_deficiency",
    "registered_terms": ["intercept", "centered_volume", "centered_volume_squared"],
    "terms": ["intercept", "centered_volume", "centered_volume_squared"],
}
_AUTHORITY_THREAT_MODEL = {
    "boundary": "in_process_public_surface_tamper_resistance",
    "arbitrary_same_process_python_introspection": "out_of_scope",
    "cryptographic_unforgeability": False,
    "production_authority": False,
    "claim_ceiling": (
        "loader-issued synthetic capability under cooperative-process execution"
    ),
}
_METHODOLOGY_EVIDENCE = {
    "primary_sources": [
        {
            "doi": "10.1198/TAST.2009.0030",
            "use": "report Monte Carlo uncertainty and justify replication count",
        },
        {
            "doi": "10.1002/bimj.202200095",
            "use": "neutral simulation comparison, sensitivity, and MCSE reporting",
        },
    ],
    "wolfram_oracle": {
        "expression": (
            "(v-1/2)^2-((c-1/2)^2+2(c-1/2)(v-c)+(v-c)^2)"
        ),
        "result": "0",
        "role": "independent symbolic check; local executable identity is authoritative",
    },
    "academic_writing_review": {
        "issue_count": 0,
        "role": "claim-language review only; not executable evidence",
    },
}

def _callable_fingerprint(fn: Any, role: str) -> tuple[str, str, str, str]:
    if not callable(fn):
        _fail(f"{role} must be callable")
    module = inspect.getmodule(fn)
    if module is None or not hasattr(module, "__file__") or module.__file__ is None:
        _fail(f"{role} callable module identity is unavailable")
    file_path = Path(module.__file__).resolve()
    code = getattr(fn, "__code__", None)
    if code is None:
        _fail(f"{role} callable code identity is unavailable")
    return (
        module.__name__,
        getattr(fn, "__name__", ""),
        hashlib.sha256(file_path.read_bytes()).hexdigest(),
        hashlib.sha256(code.co_code + repr(code.co_consts).encode()).hexdigest(),
    )


def _fail(message: str) -> None:
    raise ValidationFailure(message)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _strict_hash(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
        or len(set(value)) == 1
    ):
        _fail(f"{path} must be nonplaceholder lowercase sha256")
    return value


def _strict_id(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or value.strip() != value
        or not 3 <= len(value) <= 256
    ):
        _fail(f"{path} must be a stable ID")
    return value


def _strict_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be bool")
    return value


def _strict_int(value: object, path: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        _fail(f"{path} must be int")
    return value


def _strict_float(value: object, path: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        _fail(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{path} must be finite")
    return result


def _canonical_payload(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{name} contains duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode(),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(f"{name} contains {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{name} is invalid JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{name} must be an object")
    if raw != (canonical_json(payload) + "\n").encode():
        _fail(f"{name} bytes are noncanonical")
    return payload


def _safe_file(root: Path, locator: object) -> Path:
    if isinstance(locator, Path):
        text = locator.as_posix()
    elif isinstance(locator, str):
        text = locator
    else:
        _fail("locator must be a string")
    relative = Path(text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _fail("unsafe locator")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("symlink locator component rejected")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationFailure("locator escapes authority root") from exc
    if not resolved.is_file():
        _fail("locator is not a regular file")
    stat = resolved.stat()
    if stat.st_nlink != 1:
        _fail("hard-linked authority file rejected")
    return resolved


def _read_ref(
    root: Path,
    ref: Mapping[str, Any],
    *,
    expected_id: str | None = None,
) -> tuple[dict[str, Any], bytes, Path]:
    if set(ref) != {
        "artifact_id", "locator", "raw_sha256", "canonical_payload_sha256"
    }:
        _fail("artifact ref shape mismatch")
    artifact_id = _strict_id(ref["artifact_id"], "artifact_id")
    if expected_id is not None and artifact_id != expected_id:
        _fail("artifact ID mismatch")
    path = _safe_file(root, ref["locator"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _strict_hash(
        ref["raw_sha256"], "raw_sha256"
    ):
        _fail("artifact raw hash mismatch")
    payload = _canonical_payload(raw, artifact_id)
    if payload.get("artifact_id") != artifact_id:
        _fail("artifact payload ID mismatch")
    if canonical_sha256(payload) != _strict_hash(
        ref["canonical_payload_sha256"], "object_sha256"
    ):
        _fail("artifact object hash mismatch")
    return payload, raw, path


def _artifact_ref(locator: Path, payload: Mapping[str, Any], raw: bytes) -> dict[str, str]:
    return {
        "artifact_id": _strict_id(payload["artifact_id"], "artifact_id"),
        "locator": locator.as_posix(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_payload_sha256": canonical_sha256(payload),
    }


def _expected_dependencies(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    dependencies: dict[str, dict[str, Any]] = {}
    for method in METHOD_SPECS.values():
        for role in method["dependencies"]:
            if role in dependencies:
                continue
            payload: dict[str, Any] = {
                "artifact_id": f"scryglass:b2:r20-foundation-dependency:{role}:v2",
                "role": role,
                "values_by_row_id": {
                    row["row_id"]: row["candidate_inputs"][role] for row in rows
                },
                "synthetic_only": True,
                "production_eligible": False,
            }
            dependencies[role] = payload
    return dependencies


def _expected_candidate_registry(
    *,
    rows: list[dict[str, Any]],
    benchmark_ref: Mapping[str, Any],
    config_ref: Mapping[str, Any],
    config_raw_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    dependencies = _expected_dependencies(rows)
    dependency_refs: dict[str, dict[str, str]] = {}
    for role, payload in dependencies.items():
        locator = _DEPENDENCY_LOCATORS[role]
        ref = _artifact_ref(locator, payload, canonical_json(payload).encode() + b"\n")
        dependency_refs[role] = {
            **ref,
            "artifact_id": payload["artifact_id"],
            "role": role,
        }

    candidates: list[dict[str, Any]] = []
    for method_id, spec in METHOD_SPECS.items():
        if spec["family"] == "precision":
            boundaries = {"minimum_draws": POSTERIOR_DRAWS, "central_mass": 0.95}
        elif spec["family"] == "posterior_information":
            boundaries = {"minimum_draws": POSTERIOR_DRAWS}
        else:
            boundaries = {"fallback_forbids_high": True}
        candidates.append(
            {
                "method_id": method_id,
                "family": spec["family"],
                "units": spec["units"],
                "simplicity_rank": METHOD_COMPLEXITY[method_id],
                "input_schema": {
                    "roles": list(spec["dependencies"]),
                    "cardinality": {
                        role: "one_per_row" for role in spec["dependencies"]
                    },
                },
                "implementation": {
                    "module": "lol_kills.v2.evaluation.r20_foundation_algorithms",
                    "entrypoint": "replay_foundation_method",
                    "source_sha256": source_hashes["algorithms"],
                },
                "boundaries": boundaries,
                "dependencies": [dependency_refs[role] for role in spec["dependencies"]],
                "code_sha256": source_hashes["algorithms"],
                "config_sha256": config_raw_sha256,
                "boundary_sha256": canonical_sha256(boundaries),
            }
        )
    return {
        "artifact_id": CANDIDATE_REGISTRY_ARTIFACT_ID,
        "benchmark_ref": dict(benchmark_ref),
        "config_ref": dict(config_ref),
        "candidates": candidates,
        "synthetic_only": True,
        "production_eligible": False,
    }, dependencies, dependency_refs


def _expected_dependency_payloads(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    dependencies: dict[str, Any] = {}
    dependency_refs: dict[str, dict[str, Any]] = {}
    for role, payload in _expected_dependencies(rows).items():
        raw = (canonical_json(payload) + "\n").encode()
        dependency_refs[role] = _artifact_ref(
            _DEPENDENCY_LOCATORS[role],
            payload,
            raw,
        )
        dependencies[role] = payload
    return dependencies, dependency_refs


def _expected_config(
    source_hashes: Mapping[str, str],
    *,
    monte_carlo_call: Any,
    benchmark_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_id": FOUNDATION_CONFIG_ARTIFACT_ID,
        "simulation_seed": FAMILY_DEFAULT.simulation_seed,
        "inference_seed": INFERENCE_SEED,
        "chronological_folds": FOUNDATION_FOLDS,
        "initial_train_series_per_cell": INITIAL_TRAIN_SERIES_PER_CELL,
        "test_series_per_cell_per_fold": TEST_SERIES_PER_CELL_PER_FOLD,
        "maps_per_series": FOUNDATION_MAPS_PER_SERIES,
        "draw_count": FOUNDATION_DRAWS,
        "central_mass": 0.95,
        "prior": {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA},
        "reference_modes": list(REFERENCE_MODES),
        "monte_carlo_design": monte_carlo_call(
            registered_observation_cells=_registered_observation_cells(
                benchmark_rows,
            ),
        ),
        "required_families": [
            "posterior_information",
            "precision",
            "source_context_coverage",
        ],
        "output_strata": [list(item) for item in OUTPUT_STRATA],
        "regimes": list(REGIMES),
        "source_context_patterns": list(SOURCE_CONTEXT_PATTERNS),
        "volume_only_basis": _VOLUME_BASIS,
        "source_closure": dict(source_hashes),
        "runtime": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
        },
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "authority_threat_model": _AUTHORITY_THREAT_MODEL,
        "methodology_evidence": _METHODOLOGY_EVIDENCE,
        "synthetic_only": True,
        "production_eligible": False,
    }


def _expected_artifacts(
    source_hashes: Mapping[str, str],
    *,
    generator_call: Any,
    monte_carlo_call: Any,
) -> tuple[
    dict[str, Any],  # authority payload
    dict[str, Any],  # config payload
    bytes,           # config raw
    dict[str, Any],  # benchmark payload
    bytes,           # benchmark raw
    dict[str, Any],  # registry payload
    bytes,           # registry raw
    dict[str, Any],  # dependency payloads
    dict[str, Any],  # dependency refs
]:
    benchmark, benchmark_raw = _expected_benchmark(
        source_hashes,
        generator_call=generator_call,
    )
    benchmark_ref = _artifact_ref(BENCHMARK_LOCATOR, benchmark, benchmark_raw)
    config = _expected_config(
        source_hashes,
        monte_carlo_call=monte_carlo_call,
        benchmark_rows=benchmark["rows"],
    )
    config_raw = (canonical_json(config) + "\n").encode()
    config_ref = _artifact_ref(FOUNDATION_CONFIG_LOCATOR, config, config_raw)

    dependency_payloads, dependency_refs = _expected_dependency_payloads(
        benchmark["rows"],
    )
    config_raw_sha256 = hashlib.sha256(config_raw).hexdigest()
    candidate_registry, _, _ = _expected_candidate_registry(
        rows=benchmark["rows"],
        benchmark_ref=benchmark_ref,
        config_ref=config_ref,
        config_raw_sha256=config_raw_sha256,
        source_hashes=source_hashes,
    )
    candidate_registry_ref = _artifact_ref(
        CANDIDATE_REGISTRY_LOCATOR,
        candidate_registry,
        (canonical_json(candidate_registry) + "\n").encode(),
    )

    authority = _expected_authority(
        source_hashes=source_hashes,
        config_ref=config_ref,
        benchmark_ref=benchmark_ref,
        candidate_registry_ref=candidate_registry_ref,
    )
    return (
        authority,
        config,
        config_raw,
        benchmark,
        benchmark_raw,
        candidate_registry,
        candidate_registry_ref,
        dependency_payloads,
        dependency_refs,
    )


def _expected_benchmark(
    source_hashes: Mapping[str, str],
    *,
    generator_call: Any,
) -> tuple[dict[str, Any], bytes]:
    regenerated = generator_call()
    payload = {
        "artifact_id": BENCHMARK_ARTIFACT_ID,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "generator": {
            "artifact_id": FAMILY_DEFAULT.artifact_id,
            "module": FAMILY_DEFAULT.module,
            "entrypoint": FAMILY_DEFAULT.entrypoint,
            "source_sha256": source_hashes["generator"],
        },
        "inference_adapter": {
            "adapter_id": INFERENCE_ADAPTER_ID,
            "module": "lol_kills.v2.evaluation.r20_foundation_inference",
            "entrypoint": "infer_beta_binomial",
            "source_sha256": source_hashes["inference_adapter"],
        },
        "chronological_folds": regenerated["chronological_folds"],
        "seed": regenerated["seed"],
        "rows": regenerated["rows"],
        "prequential_plan": regenerated["prequential_plan"],
        "rows_row_id_sha256": regenerated["rows_row_id_sha256"],
        "prequential_plan_sha256": regenerated["prequential_plan_sha256"],
        "volume_readiness": regenerated.get("volume_readiness", []),
        "synthetic_only": True,
        "production_eligible": False,
    }
    return payload, (canonical_json(payload) + "\n").encode()


def _expected_authority(
    *,
    source_hashes: Mapping[str, str],
    config_ref: Mapping[str, Any],
    benchmark_ref: Mapping[str, Any],
    candidate_registry_ref: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_id": AUTHORITY_ARTIFACT_ID,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "benchmark_ref": dict(benchmark_ref),
        "config_ref": dict(config_ref),
        "candidate_registry_ref": dict(candidate_registry_ref),
        "source_closure": dict(source_hashes),
        "execution_closure": {
            "source_files": dict(source_hashes),
            "config_ref": dict(config_ref),
            "runtime": {
                "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "numpy_version": np.__version__,
                "bit_generator": "PCG64",
            },
        },
        "authority_threat_model": _AUTHORITY_THREAT_MODEL,
        "synthetic_only": True,
        "production_eligible": False,
    }


def _source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, locator in _SOURCE_FILES.items():
        result[role] = hashlib.sha256(_safe_file(root, locator).read_bytes()).hexdigest()
    return result


def _registered_observation_cells(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells = {
        (
            row["candidate_inputs"]["generator_regime"],
            row["candidate_inputs"]["observation"]["successes"],
            row["candidate_inputs"]["observation"]["trials"],
        )
        for row in rows
    }
    return [
        {"regime": regime, "successes": successes, "trials": trials}
        for regime, successes, trials in sorted(cells)
    ]


def _assert_exact_imported_root(root: Path) -> None:
    if root.resolve() != _IMPORTED_ROOT:
        _fail("detached source root cannot issue foundation authority")
    module_names = {
        "loader": "lol_kills.v2.evaluation.r20_foundation",
        "generator": "lol_kills.v2.evaluation.r20_foundation_generator",
        "inference_adapter": "lol_kills.v2.evaluation.r20_foundation_inference",
        "algorithms": "lol_kills.v2.evaluation.r20_foundation_algorithms",
        "artifact_generator": "lol_kills.v2.evaluation.generate_r20_foundation_artifacts",
        "types": "lol_kills.v2.evaluation.types",
        "checks": "lol_kills.v2.evaluation.checks",
    }
    imported_modules = {
        role: sys.modules.get(module_name)
        for role, module_name in module_names.items()
    }
    for role, module_name in module_names.items():
        if imported_modules[role] is None:
            imported_modules[role] = importlib.import_module(module_name)
    imported_files = {
        role: Path(module.__file__).resolve()
        for role, module in imported_modules.items()
    }
    for role, locator in _SOURCE_FILES.items():
        if imported_files[role] != (root / locator).resolve():
            _fail("attested source is not the executed imported source")


def _validate_config(
    payload: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    *,
    monte_carlo_call: Any,
    benchmark_rows: list[dict[str, Any]],
) -> None:
    if set(payload) != {
        "artifact_id", "simulation_seed", "inference_seed", "chronological_folds",
        "initial_train_series_per_cell", "test_series_per_cell_per_fold",
        "maps_per_series", "draw_count", "central_mass", "prior",
        "reference_modes",
        "monte_carlo_design", "required_families", "output_strata", "regimes",
        "source_context_patterns", "volume_only_basis", "source_closure",
        "runtime", "contract_tree_sha256", "synthetic_only", "production_eligible",
        "authority_threat_model",
        "methodology_evidence",
    }:
        _fail("config shape mismatch")
    expected_scalars = {
        "artifact_id": FOUNDATION_CONFIG_ARTIFACT_ID,
        "simulation_seed": FAMILY_DEFAULT.simulation_seed,
        "inference_seed": INFERENCE_SEED,
        "chronological_folds": FOUNDATION_FOLDS,
        "initial_train_series_per_cell": INITIAL_TRAIN_SERIES_PER_CELL,
        "test_series_per_cell_per_fold": TEST_SERIES_PER_CELL_PER_FOLD,
        "maps_per_series": FOUNDATION_MAPS_PER_SERIES,
        "draw_count": POSTERIOR_DRAWS,
        "central_mass": 0.95,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "synthetic_only": True,
        "production_eligible": False,
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected or type(payload[key]) is not type(expected):
            _fail(f"config {key} mismatch")
    if payload["prior"] != {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA}:
        _fail("config prior mismatch")
    if tuple(payload["reference_modes"]) != REFERENCE_MODES:
        _fail("config reference-mode universe mismatch")
    if set(payload["required_families"]) != _REQUIRED_FAMILIES:
        _fail("config family universe mismatch")
    if [tuple(item) for item in payload["output_strata"]] != list(OUTPUT_STRATA):
        _fail("config cell universe mismatch")
    if tuple(payload["regimes"]) != REGIMES:
        _fail("config regime universe mismatch")
    if tuple(payload["source_context_patterns"]) != SOURCE_CONTEXT_PATTERNS:
        _fail("config source/context universe mismatch")
    if payload["volume_only_basis"] != _VOLUME_BASIS:
        _fail("config volume basis mismatch")
    if payload["source_closure"] != source_hashes:
        _fail("config source closure mismatch")
    if tuple(payload["required_families"]) != _REQUIRED_FAMILIES_LIST:
        _fail("config family universe mismatch")
    expected_runtime = {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
    }
    if payload["runtime"] != expected_runtime:
        _fail("config runtime identity mismatch")
    if payload["authority_threat_model"] != _AUTHORITY_THREAT_MODEL:
        _fail("authority threat model mismatch")
    if payload["methodology_evidence"] != _METHODOLOGY_EVIDENCE:
        _fail("methodology evidence mismatch")
    expected_mc = monte_carlo_call(
        registered_observation_cells=_registered_observation_cells(
            benchmark_rows,
        ),
    )
    if payload["monte_carlo_design"] != expected_mc or not expected_mc["passes"]:
        _fail("Monte Carlo draw-count design mismatch")


def _validate_probability_draws(value: object, path: str) -> None:
    if not isinstance(value, list) or len(value) != POSTERIOR_DRAWS:
        _fail(f"{path} draw cardinality mismatch")
    for item in value:
        number = _strict_float(item, path)
        if not 0 <= number <= 1:
            _fail(f"{path} outside [0,1]")


def _validate_rows(
    rows: object,
    *,
    inference_call: Any,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 1600:
        _fail("benchmark must contain exactly 1600 rows")
    required = {
        "row_id", "case_id", "series_id", "output_type", "stratum_id",
        "cohort_id", "issued", "event", "resolved", "latent_truth",
        "fixture_label", "fixture_label_dgp", "volume_inputs", "candidate_inputs",
        "lineage",
    }
    prior_order: tuple[str, str] | None = None
    seen_rows: set[str] = set()
    seen_cases: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            _fail("benchmark row shape mismatch")
        row_id = _strict_id(row["row_id"], "row_id")
        case_id = _strict_id(row["case_id"], "case_id")
        if row_id in seen_rows or case_id in seen_cases:
            _fail("duplicate row/case ID")
        seen_rows.add(row_id)
        seen_cases.add(case_id)
        order = (row["issued"], row_id)
        if prior_order is not None and order <= prior_order:
            _fail("benchmark row serialization is noncanonical")
        prior_order = order
        if (row["output_type"], row["stratum_id"]) not in OUTPUT_STRATA:
            _fail("unregistered benchmark cell")
        if row["cohort_id"] not in {
            "initial_train", "test_fold_0", "test_fold_1", "test_fold_2",
        }:
            _fail("unregistered benchmark cohort")
        parsed_times: list[datetime] = []
        for time_key in ("issued", "event", "resolved"):
            value = row[time_key]
            if not isinstance(value, str):
                _fail(f"{time_key} must be a canonical timestamp")
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValidationFailure(f"{time_key} is not parseable") from exc
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                _fail(f"{time_key} must be UTC")
            if parsed.isoformat() != value:
                _fail(f"{time_key} must use canonical ISO serialization")
            parsed_times.append(parsed)
        if not parsed_times[0] < parsed_times[1] < parsed_times[2]:
            _fail("row chronology is invalid")
        truth = _strict_float(row["latent_truth"], "latent_truth")
        if not 0 <= truth <= 1:
            _fail("latent truth outside [0,1]")
        if type(row["fixture_label"]) is bool or row["fixture_label"] not in {0, 1}:
            _fail("fixture label must be binary")
        fixture_label_dgp = row["fixture_label_dgp"]
        if not isinstance(fixture_label_dgp, dict) or set(fixture_label_dgp) != {
            "dgp_id", "seed", "stratum_rule", "probability_rule",
            "target_kind", "probability_semantics", "proper_score_eligible",
            "probability", "uniform_draw",
            "support_stratum", "sampling_weight",
        }:
            _fail("fixture-label DGP schema mismatch")
        if (
            fixture_label_dgp["dgp_id"]
            != "r20-preregistered-balanced-fixture-classification-v3"
        ):
            _fail("fixture-label DGP identity mismatch")
        if (
            type(fixture_label_dgp["seed"]) is not int
            or fixture_label_dgp["seed"] != 20260731
        ):
            _fail("fixture-label DGP seed mismatch")
        probability = _strict_float(
            fixture_label_dgp["probability"],
            "fixture-label probability",
        )
        uniform_draw = _strict_float(
            fixture_label_dgp["uniform_draw"],
            "fixture-label uniform",
        )
        if probability not in {0.0, 1.0} or not 0 <= uniform_draw < 1:
            _fail("fixture-label DGP support-control values invalid")
        if probability == truth:
            _fail("latent truth cannot be conflated with support-class probability")
        if (
            type(fixture_label_dgp["support_stratum"]) is not int
            or not 0 <= fixture_label_dgp["support_stratum"] < 8
        ):
            _fail("fixture-label DGP support stratum invalid")
        if (
            _strict_float(fixture_label_dgp["sampling_weight"], "sampling weight")
            != 1.0
        ):
            _fail("fixture-label DGP sampling weight mismatch")
        if row["fixture_label"] != int(uniform_draw < probability):
            _fail("fixture label does not replay from registered DGP")
        volume_inputs = row["volume_inputs"]
        if not isinstance(volume_inputs, dict) or set(volume_inputs) != set(REGISTERED_VOLUME_FIELDS):
            _fail("volume input schema mismatch")
        for field_id in REGISTERED_VOLUME_FIELDS:
            value = _strict_float(volume_inputs[field_id], f"volume_inputs.{field_id}")
            if field_id in {"sample_size", "game_count"}:
                if not float(value).is_integer() or value < 1:
                    _fail(f"volume_inputs.{field_id} must be a positive count")
            elif not 0 <= value <= 1:
                _fail(f"volume_inputs.{field_id} must lie in [0,1]")
        inputs = row["candidate_inputs"]
        if not isinstance(inputs, dict) or set(inputs) != {
            "generator_regime", "observation", "inference", "posterior_draws",
            "prior_draws", "registered_reference_draws", "source_lineage",
            "context_registry", "fallback_registry", "bridge_registry",
            "source_context_pattern",
        }:
            _fail("candidate input shape mismatch")
        if inputs["generator_regime"] not in REGIMES:
            _fail("unregistered generator regime")
        expected_stratum_rule = "balanced_fixture_strata_0_3_vs_4_7"
        if (
            fixture_label_dgp["stratum_rule"] != expected_stratum_rule
            or fixture_label_dgp["probability_rule"]
            != "p0_strata_0_3_p1_strata_4_7"
        ):
            _fail("fixture-label DGP preregistration rule mismatch")
        if (
            fixture_label_dgp["target_kind"] != "balanced_fixture_classification"
            or fixture_label_dgp["probability_semantics"]
            != "fixture_class_probability"
            or fixture_label_dgp["proper_score_eligible"] is not False
        ):
            _fail("support target cannot authorize predictive proper scores")
        if inputs["source_context_pattern"] not in SOURCE_CONTEXT_PATTERNS:
            _fail("unregistered source/context pattern")
        for role in ("posterior_draws", "prior_draws", "registered_reference_draws"):
            _validate_probability_draws(inputs[role], role)
        observation = inputs["observation"]
        if not isinstance(observation, dict) or set(observation) != {
            "successes", "trials",
        }:
            _fail("observation schema mismatch")
        if (
            type(observation["successes"]) is not int
            or type(observation["trials"]) is not int
            or observation["trials"] not in {12, 24, 36, 48}
            or not 0 <= observation["successes"] <= observation["trials"]
        ):
            _fail("observation values are invalid")
        inference = inputs["inference"]
        if not isinstance(inference, dict) or set(inference) != {
            "adapter_id", "inference_seed", "draw_count", "prior",
            "reference_mode", "posterior_parameters", "inference_output_sha256",
        }:
            _fail("inference lineage shape mismatch")
        replay = inference_call(
            observation=observation,
            inference_seed=inference["inference_seed"],
            draw_count=inference["draw_count"],
            prior_alpha=inference["prior"]["alpha"],
            prior_beta=inference["prior"]["beta"],
            reference_mode=inference["reference_mode"],
        )
        if inference["adapter_id"] != INFERENCE_ADAPTER_ID:
            _fail("inference adapter identity mismatch")
        if inference["draw_count"] != POSTERIOR_DRAWS:
            _fail("inference draw count mismatch")
        if inference["posterior_parameters"] != replay["posterior_parameters"]:
            _fail("inference posterior parameters mismatch")
        if inference["inference_output_sha256"] != replay["inference_output_sha256"]:
            _fail("inference output identity mismatch")
        for role in ("posterior_draws", "prior_draws", "registered_reference_draws"):
            if inputs[role] != replay[role]:
                _fail("persisted inference draws fail exact replay")
        fallback = inputs["fallback_registry"]
        if not isinstance(fallback, dict) or set(fallback) != {"used", "profile"}:
            _fail("fallback shape mismatch")
        if type(fallback["used"]) is not bool or fallback["profile"] not in {
            "none", "fallback"
        }:
            _fail("fallback fields invalid")
        if fallback["used"] != (fallback["profile"] == "fallback"):
            _fail("fallback used/profile inconsistency")
        source = inputs["source_lineage"]
        context = inputs["context_registry"]
        bridge = inputs["bridge_registry"]
        if (
            not isinstance(source, dict)
            or set(source) != {"complete", "registered"}
            or any(type(value) is not bool for value in source.values())
        ):
            _fail("source lineage schema mismatch")
        if (
            not isinstance(context, dict)
            or set(context) != {"registered", "registry_version", "path"}
            or type(context["registered"]) is not bool
            or not isinstance(context["registry_version"], str)
            or not isinstance(context["path"], str)
        ):
            _fail("context registry schema mismatch")
        if (
            not isinstance(bridge, dict)
            or set(bridge) != {"registered", "bridge_id"}
            or type(bridge["registered"]) is not bool
            or not isinstance(bridge["bridge_id"], str)
        ):
            _fail("bridge registry schema mismatch")
        lineage = row["lineage"]
        if not isinstance(lineage, dict) or set(lineage) != {
            "generator", "generator_regime", "cohort_id", "source_id",
            "inference_output_sha256",
        }:
            _fail("row lineage schema mismatch")
        if (
            lineage["generator"] != FAMILY_DEFAULT.entrypoint
            or lineage["generator_regime"] != inputs["generator_regime"]
            or lineage["cohort_id"] != row["cohort_id"]
            or lineage["inference_output_sha256"] != inference["inference_output_sha256"]
        ):
            _fail("row lineage linkage mismatch")
        _strict_id(lineage["source_id"], "lineage.source_id")
    build_prequential_plan(rows)
    return rows


def require_predictive_target_authority(
    rows: object,
    *,
    target_authority: object = None,
) -> None:
    """Fail closed until a separate proper predictive-target authority exists."""

    if isinstance(rows, list) and any(
        isinstance(row, dict)
        and ("fixture_label" in row or "fixture_label_dgp" in row)
        for row in rows
    ):
        _fail("balanced fixture labels cannot serve as predictive examples")
    if target_authority is None:
        _fail("no proper predictive-target authority is registered in 2B1")
    _fail("2B1 cannot authorize predictive examples")


def _validate_dependency(
    root: Path,
    ref: Mapping[str, Any],
    role: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ref_bare = {
        key: ref[key]
        for key in ("artifact_id", "locator", "raw_sha256", "canonical_payload_sha256")
    }
    payload, _, _ = _read_ref(root, ref_bare)
    if set(payload) != {
        "artifact_id", "role", "values_by_row_id", "synthetic_only",
        "production_eligible",
    } or payload["role"] != role:
        _fail("dependency payload role/shape mismatch")
    if payload["synthetic_only"] is not True or payload["production_eligible"] is not False:
        _fail("dependency production boundary mismatch")
    values = payload["values_by_row_id"]
    if not isinstance(values, dict) or set(values) != {row["row_id"] for row in rows}:
        _fail("dependency row coverage mismatch")
    for row in rows:
        if values[row["row_id"]] != row["candidate_inputs"][role]:
            _fail("dependency bytes differ from inference benchmark")
    return payload


def _candidate_behaviour(
    candidates: Mapping[str, Mapping[str, Any]],
    dependencies: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
    *,
    replay_call: Any,
    method_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    by_family: dict[str, list[str]] = {}
    outputs: dict[str, list[Any]] = {method: [] for method in candidates}
    for method, candidate in candidates.items():
        by_family.setdefault(candidate["family"], []).append(method)
        roles = method_specs[method]["dependencies"]
        for row in rows:
            row_dependencies = {
                role: dependencies[role]["values_by_row_id"][row["row_id"]]
                for role in roles
            }
            try:
                result = replay_call(
                    method_id=method,
                    dependencies=row_dependencies,
                    boundaries=candidate["boundaries"],
                )
                outputs[method].append(result["value"])
            except ValidationFailure:
                outputs[method].append({"status": "rejected"})
    for family, methods in by_family.items():
        if len(methods) != 2:
            _fail("every family requires exactly two candidates")
        left, right = outputs[methods[0]], outputs[methods[1]]
        if left == right:
            _fail(f"{family} candidates are behaviorally identical")
        if family != "source_context_coverage":
            pairs = [
                (float(a), float(b))
                for a, b in zip(left, right)
                if isinstance(a, float) and isinstance(b, float)
            ]
            if len(pairs) < 20:
                _fail(f"{family} lacks common numerical support")
            a = np.asarray([pair[0] for pair in pairs])
            b = np.asarray([pair[1] for pair in pairs])
            if np.array_equal(np.argsort(a), np.argsort(b)):
                _fail(f"{family} candidates have identical ranks")
            variance = float(np.sum((a - np.mean(a)) ** 2))
            if variance <= 0:
                _fail(f"{family} first candidate is constant")
            slope = float(
                np.sum((a - np.mean(a)) * (b - np.mean(b))) / variance
            )
            intercept = float(np.mean(b) - slope * np.mean(a))
            residual = b - (intercept + slope * a)
            if float(np.max(np.abs(residual))) < 1e-10:
                _fail(f"{family} candidates are exact affine rescalings")


def _validate_replayed_primitive(
    primitive: Mapping[str, Any],
    *,
    method_id: str,
    candidate: Mapping[str, Any],
    method_complexity: Mapping[str, int],
) -> None:
    required = {
        "method_id",
        "family",
        "units",
        "value",
        "executed_boundary_sha256",
        "authorized_candidate",
        "replay_ok",
        "method_complexity",
    }
    if set(primitive) != required:
        _fail("primitive output schema mismatch")
    if primitive["method_id"] != method_id:
        _fail("primitive method_id mismatch")
    if primitive["family"] != candidate["family"]:
        _fail("primitive family mismatch")
    if primitive["units"] != candidate["units"]:
        _fail("primitive units mismatch")
    if primitive["authorized_candidate"] is not False:
        _fail("primitive must be unauthorized")
    if primitive["replay_ok"] is not True:
        _fail("primitive replay must be successful")
    if (
        type(primitive["method_complexity"]) is not int
        or primitive["method_complexity"] != method_complexity[method_id]
    ):
        _fail("primitive complexity mismatch")
    if primitive["executed_boundary_sha256"] != canonical_sha256(_thaw(candidate["boundaries"])):
        _fail("primitive boundary hash mismatch")
    family = candidate["family"]
    value = primitive["value"]
    if family in {"posterior_information", "precision"}:
        if type(value) is not float or not math.isfinite(value):
            _fail("numeric primitive value must be a finite strict float")
        if family == "posterior_information" and value < 0:
            _fail("posterior-information value must be nonnegative")
        if family == "precision" and not 0 <= value <= 1:
            _fail("precision value must lie in [0,1]")
        return
    if not isinstance(value, dict) or set(value) != {
        "status", "usable", "high_eligible", "issues", "flags",
    }:
        _fail("source/context primitive value schema mismatch")
    if value["status"] not in {"complete", "limited", "unavailable"}:
        _fail("source/context status is invalid")
    if type(value["usable"]) is not bool or type(value["high_eligible"]) is not bool:
        _fail("source/context eligibility flags must be strict bools")
    if not isinstance(value["issues"], list) or any(
        not isinstance(item, str) or not item for item in value["issues"]
    ):
        _fail("source/context issues must be a string list")
    flags = value["flags"]
    if not isinstance(flags, dict) or set(flags) != {
        "source_registered", "lineage_complete", "context_registered",
        "bridge_registered", "fallback_used", "fallback_profile",
    }:
        _fail("source/context flags schema mismatch")
    for key in (
        "source_registered", "lineage_complete", "context_registered",
        "bridge_registered", "fallback_used",
    ):
        if type(flags[key]) is not bool:
            _fail("source/context flags must be strict bools")
    if flags["fallback_profile"] not in {"none", "fallback"}:
        _fail("source/context fallback profile is invalid")


def _validate_candidates(
    root: Path,
    payload: Mapping[str, Any],
    rows: list[dict[str, Any]],
    config_ref: Mapping[str, Any],
    config_raw_sha256: str,
    source_hashes: Mapping[str, str],
    *,
    replay_call: Any,
    method_specs: Mapping[str, Mapping[str, Any]],
    method_complexity: Mapping[str, int],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if set(payload) != {
        "artifact_id", "benchmark_ref", "config_ref", "candidates",
        "synthetic_only", "production_eligible",
    } or payload["artifact_id"] != CANDIDATE_REGISTRY_ARTIFACT_ID:
        _fail("candidate registry shape/identity mismatch")
    if payload["config_ref"] != config_ref:
        _fail("candidate config ref mismatch")
    if payload["synthetic_only"] is not True or payload["production_eligible"] is not False:
        _fail("candidate registry production boundary mismatch")
    records = payload["candidates"]
    if not isinstance(records, list) or len(records) != len(method_specs):
        _fail("candidate method set incomplete")
    candidates: dict[str, Mapping[str, Any]] = {}
    dependencies: dict[str, Mapping[str, Any]] = {}
    refs_by_role: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "method_id", "family", "units", "simplicity_rank", "input_schema",
            "implementation", "boundaries", "dependencies", "code_sha256",
            "config_sha256", "boundary_sha256",
        }:
            _fail("candidate record shape mismatch")
        method = _strict_id(record["method_id"], "method_id")
        if method not in method_specs or method in candidates:
            _fail("candidate method is unknown/duplicated")
        spec = method_specs[method]
        if record["family"] != spec["family"] or record["units"] != spec["units"]:
            _fail("candidate family/units mismatch")
        if _strict_int(record["simplicity_rank"], "simplicity_rank") != method_complexity[
            method
        ]:
            _fail("candidate simplicity complexity mismatch")
        if record["input_schema"] != {
            "roles": list(spec["dependencies"]),
            "cardinality": {role: "one_per_row" for role in spec["dependencies"]},
        }:
            _fail("candidate input schema mismatch")
        expected_impl = {
            "module": "lol_kills.v2.evaluation.r20_foundation_algorithms",
            "entrypoint": "replay_foundation_method",
            "source_sha256": source_hashes["algorithms"],
        }
        if record["implementation"] != expected_impl:
            _fail("candidate executed implementation identity mismatch")
        if record["code_sha256"] != source_hashes["algorithms"]:
            _fail("candidate code hash mismatch")
        if record["config_sha256"] != config_raw_sha256:
            _fail("candidate config hash mismatch")
        boundaries = record["boundaries"]
        if record["boundary_sha256"] != canonical_sha256(boundaries):
            _fail("candidate boundary identity mismatch")
        if spec["family"] == "precision":
            if boundaries != {"minimum_draws": POSTERIOR_DRAWS, "central_mass": 0.95}:
                _fail("precision boundaries mismatch frozen config")
        elif spec["family"] == "posterior_information":
            if boundaries != {"minimum_draws": POSTERIOR_DRAWS}:
                _fail("posterior boundaries mismatch frozen config")
        elif boundaries != {"fallback_forbids_high": True}:
            _fail("source/context boundaries mismatch")
        refs = record["dependencies"]
        if not isinstance(refs, list) or [ref.get("role") for ref in refs] != list(
            spec["dependencies"]
        ):
            _fail("candidate dependency order/cardinality mismatch")
        for ref in refs:
            role = ref["role"]
            if set(ref) != {
                "artifact_id", "role", "locator", "raw_sha256",
                "canonical_payload_sha256",
            }:
                _fail("dependency ref shape mismatch")
            bare = {
                key: ref[key]
                for key in (
                    "artifact_id", "locator", "raw_sha256",
                    "canonical_payload_sha256",
                )
            }
            if role in refs_by_role and refs_by_role[role] != bare:
                _fail("same-role dependency authority substituted")
            if role not in refs_by_role:
                dependencies[role] = _validate_dependency(root, ref, role, rows)
                refs_by_role[role] = bare
        candidates[method] = record
    if set(candidates) != set(method_specs):
        _fail("exact method universe mismatch")
    _candidate_behaviour(
        candidates,
        dependencies,
        rows,
        replay_call=replay_call,
        method_specs=method_specs,
    )
    return candidates, dependencies


def _load_foundation_artifacts_impl(
    root: Path | str,
    *,
    authority_type: type,
    issue: Any,
    generator_call: Any,
    inference_call: Any,
    replay_call: Any,
    monte_carlo_call: Any,
    method_specs: Mapping[str, Mapping[str, Any]],
    method_complexity: Mapping[str, int],
) -> Any:
    root_path = Path(root).resolve()
    _assert_exact_imported_root(root_path)
    source_hashes = _source_hashes(root_path)
    (
        expected_authority,
        expected_config,
        expected_config_raw,
        expected_benchmark,
        expected_benchmark_raw,
        expected_registry,
        _expected_registry_ref,
        expected_dependency_payloads,
        expected_dependency_refs,
    ) = _expected_artifacts(
        source_hashes,
        generator_call=generator_call,
        monte_carlo_call=monte_carlo_call,
    )
    authority_path = _safe_file(root_path, AUTHORITY_LOCATOR)
    authority_raw = authority_path.read_bytes()
    authority = _canonical_payload(authority_raw, AUTHORITY_ARTIFACT_ID)
    if authority != expected_authority:
        _fail("authority payload does not match closure-derived expected")

    if set(authority) != {
        "artifact_id", "contract_tree_sha256", "benchmark_ref", "config_ref",
        "candidate_registry_ref", "source_closure", "execution_closure", "synthetic_only",
        "production_eligible", "authority_threat_model",
    } or authority["artifact_id"] != AUTHORITY_ARTIFACT_ID:
        _fail("authority shape/identity mismatch")
    if authority["contract_tree_sha256"] != CONTRACT_TREE_SHA256:
        _fail("authority C0 mismatch")
    if authority["synthetic_only"] is not True or authority["production_eligible"] is not False:
        _fail("authority cannot promote")
    if authority["authority_threat_model"] != _AUTHORITY_THREAT_MODEL:
        _fail("authority threat model mismatch")
    if authority["source_closure"] != source_hashes:
        _fail("authority source closure mismatch")
    benchmark, benchmark_raw, _ = _read_ref(
        root_path, authority["benchmark_ref"], expected_id=BENCHMARK_ARTIFACT_ID
    )
    config, config_raw, _ = _read_ref(
        root_path, authority["config_ref"], expected_id=FOUNDATION_CONFIG_ARTIFACT_ID
    )
    registry, _, _ = _read_ref(
        root_path,
        authority["candidate_registry_ref"],
        expected_id=CANDIDATE_REGISTRY_ARTIFACT_ID,
    )

    if config != expected_config:
        _fail("config payload does not match closure-derived expected")
    if canonical_sha256(config) != canonical_sha256(expected_config):
        _fail("config payload identity mismatch")
    if config_raw != expected_config_raw:
        _fail("config raw bytes mismatch")
    if benchmark != expected_benchmark:
        _fail("benchmark payload does not match closure-derived expected")
    if canonical_sha256(benchmark) != canonical_sha256(expected_benchmark):
        _fail("benchmark payload identity mismatch")
    if benchmark_raw != expected_benchmark_raw:
        _fail("benchmark raw bytes mismatch")
    if registry != expected_registry:
        _fail("candidate registry payload does not match closure-derived expected")
    if authority["benchmark_ref"] != expected_authority["benchmark_ref"]:
        _fail("benchmark ref mismatch")
    if authority["config_ref"] != expected_authority["config_ref"]:
        _fail("config ref mismatch")
    if authority["candidate_registry_ref"] != expected_authority["candidate_registry_ref"]:
        _fail("candidate-registry ref mismatch")
    _validate_config(
        config,
        source_hashes,
        monte_carlo_call=monte_carlo_call,
        benchmark_rows=benchmark["rows"],
    )
    if authority["execution_closure"] != expected_authority["execution_closure"]:
        _fail("authority config/runtime execution closure mismatch")
    if set(benchmark) != {
        "artifact_id", "contract_tree_sha256", "generator", "inference_adapter",
        "seed", "chronological_folds", "rows", "prequential_plan",
        "rows_row_id_sha256", "prequential_plan_sha256", "synthetic_only",
        "production_eligible",
        "volume_readiness",
    } or benchmark["artifact_id"] != BENCHMARK_ARTIFACT_ID:
        _fail("benchmark shape/identity mismatch")
    if benchmark["contract_tree_sha256"] != CONTRACT_TREE_SHA256:
        _fail("benchmark C0 mismatch")
    if benchmark["synthetic_only"] is not True or benchmark["production_eligible"] is not False:
        _fail("benchmark cannot promote")
    if benchmark["generator"] != {
        "artifact_id": FAMILY_DEFAULT.artifact_id,
        "module": FAMILY_DEFAULT.module,
        "entrypoint": FAMILY_DEFAULT.entrypoint,
        "source_sha256": source_hashes["generator"],
    }:
        _fail("benchmark generator closure mismatch")
    if benchmark["inference_adapter"] != {
        "adapter_id": INFERENCE_ADAPTER_ID,
        "module": "lol_kills.v2.evaluation.r20_foundation_inference",
        "entrypoint": "infer_beta_binomial",
        "source_sha256": source_hashes["inference_adapter"],
    }:
        _fail("benchmark inference closure mismatch")
    for role, expected_ref in sorted(expected_dependency_refs.items()):
        if role not in _DEPENDENCY_LOCATORS:
            _fail("dependency artifact locator missing")
        payload, _, _ = _read_ref(root_path, expected_ref, expected_id=None)
        if payload != expected_dependency_payloads[role]:
            _fail("dependency payload does not match closure-derived expected")
    rows = _validate_rows(benchmark["rows"], inference_call=inference_call)
    verify_prequential_plan(rows, benchmark["prequential_plan"])
    regenerated = generator_call()
    for key in (
        "seed", "chronological_folds", "rows", "prequential_plan",
        "rows_row_id_sha256", "prequential_plan_sha256",
    ):
        if benchmark[key] != regenerated[key]:
            _fail(f"fresh generator/inference mismatch: {key}")
    if registry["benchmark_ref"] != authority["benchmark_ref"]:
        _fail("candidate benchmark ref mismatch")
    candidates, dependencies = _validate_candidates(
        root_path,
        registry,
        rows,
        authority["config_ref"],
        hashlib.sha256(config_raw).hexdigest(),
        source_hashes,
        replay_call=replay_call,
        method_specs=method_specs,
        method_complexity=method_complexity,
    )
    for role, payload in expected_dependency_payloads.items():
        if role not in dependencies:
            _fail(f"dependency {role} was not reconstructed")
        if dependencies[role] != payload:
            _fail(f"dependency {role} reconstruction mismatch")
    state = _freeze(
        {
            "authority": authority,
            "benchmark": benchmark,
            "config": config,
            "candidate_registry": registry,
            "candidate_payloads": candidates,
            "dependency_payloads": dependencies,
            "identity": {
                "authority_raw_sha256": hashlib.sha256(authority_raw).hexdigest(),
                "authority_object_sha256": canonical_sha256(authority),
                "source_closure": source_hashes,
            },
        },
    )
    return issue(state)


def _replay_foundation_row_candidate_impl(
    *,
    authority: Any,
    row_id: str,
    method_id: str,
    lookup: Any,
    replay_call: Any,
    method_specs: Mapping[str, Mapping[str, Any]],
    method_complexity: Mapping[str, int],
) -> dict[str, Any]:
    state = lookup(authority)
    candidates = state["candidate_payloads"]
    if method_id not in candidates:
        _fail("candidate is not registered")
    candidate = candidates[method_id]
    row = next(
        (item for item in state["benchmark"]["rows"] if item["row_id"] == row_id),
        None,
    )
    if row is None:
        _fail("row is not registered")
    roles = method_specs[method_id]["dependencies"]
    dependencies = {
        role: state["dependency_payloads"][role]["values_by_row_id"][row_id]
        for role in roles
    }
    primitive = replay_call(
        method_id=method_id,
        dependencies=_thaw(dependencies),
        boundaries=_thaw(candidate["boundaries"]),
    )
    _validate_replayed_primitive(
        primitive,
        method_id=method_id,
        candidate=candidate,
        method_complexity=method_complexity,
    )
    return {
        **primitive,
        "authorized_candidate": True,
        "authority_sha256": state["identity"]["authority_raw_sha256"],
        "candidate_sha256": canonical_sha256(_thaw(candidate)),
        "boundary_sha256": candidate["boundary_sha256"],
        "executed_boundary_sha256": primitive["executed_boundary_sha256"],
        "row_id": row_id,
        "synthetic_only": True,
        "production_eligible": False,
    }


def _make_foundation_api() -> tuple[type, Any, Any]:
    """Create the only issuing/lookup/replay boundary.

    All mutable authorization state and executable bindings live in this closure.
    The module exports no registrar or sentinel that a caller can replace.
    """

    fail_fn = _fail
    fingerprint_fn = _callable_fingerprint
    generator_fn = build_r20_benchmark
    inference_fn = infer_beta_binomial
    replay_fn = replay_foundation_method
    monte_carlo_fn = monte_carlo_width_design
    load_impl_fn = _load_foundation_artifacts_impl
    replay_impl_fn = _replay_foundation_row_candidate_impl
    callable_fingerprints = {
        "generator": fingerprint_fn(generator_fn, "GENERATOR"),
        "inference": fingerprint_fn(inference_fn, "INFERENCE_ADAPTER"),
        "replay": fingerprint_fn(replay_fn, "METHOD_REPLAY"),
        "monte_carlo": fingerprint_fn(monte_carlo_fn, "MONTE_CARLO"),
        "load_impl": fingerprint_fn(load_impl_fn, "LOAD_IMPL"),
        "replay_impl": fingerprint_fn(replay_impl_fn, "REPLAY_IMPL"),
    }
    specs_object = METHOD_SPECS
    complexity_object = METHOD_COMPLEXITY
    specs_snapshot = {
        method_id: {
            "family": spec["family"],
            "units": spec["units"],
            "dependencies": tuple(spec["dependencies"]),
        }
        for method_id, spec in specs_object.items()
    }
    complexity_snapshot = dict(complexity_object)
    helper_objects = {
        method_id: spec["replay"] for method_id, spec in specs_object.items()
    }
    helper_fingerprints = {
        method_id: fingerprint_fn(helper, f"METHOD_HELPER:{method_id}")
        for method_id, helper in helper_objects.items()
    }
    storage: dict[int, tuple[weakref.ReferenceType[Any], Mapping[str, Any]]] = {}

    def constant_payload(value: Any) -> Any:
        if is_dataclass(value):
            return constant_payload(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): constant_payload(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [constant_payload(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        if value is None or type(value) in {str, int, float, bool}:
            return value
        raise TypeError

    def namespace_snapshot(fn: Any) -> dict[str, Any]:
        globals_map = fn.__globals__
        functions: dict[str, tuple[Any, tuple[str, str, str, str]]] = {}
        constants: dict[str, Any] = {}
        for name, value in globals_map.items():
            if inspect.isfunction(value):
                functions[name] = (
                    value,
                    fingerprint_fn(value, f"NAMESPACE:{name}"),
                )
            elif inspect.ismodule(value) or inspect.isclass(value):
                constants[name] = ("identity", value)
            elif not name.startswith("__"):
                try:
                    constants[name] = ("value", constant_payload(value))
                except TypeError:
                    continue
        return {"functions": functions, "constants": constants}

    namespace_authority = {
        "generator": namespace_snapshot(generator_fn),
        "inference": namespace_snapshot(inference_fn),
        "replay": namespace_snapshot(replay_fn),
        "monte_carlo": namespace_snapshot(monte_carlo_fn),
        "load_impl": namespace_snapshot(load_impl_fn),
        "replay_impl": namespace_snapshot(replay_impl_fn),
    }

    def assert_namespace(role: str, fn: Any) -> None:
        expected = namespace_authority[role]
        globals_map = fn.__globals__
        for name, (expected_object, expected_fingerprint) in expected["functions"].items():
            current = globals_map.get(name)
            if (
                current is not expected_object
                or fingerprint_fn(current, f"NAMESPACE:{name}")
                != expected_fingerprint
            ):
                fail_fn(f"{role} helper namespace changed: {name}")
        for name, expected_value in expected["constants"].items():
            if name not in globals_map:
                fail_fn(f"{role} executable dependency disappeared: {name}")
            current = globals_map[name]
            kind, frozen_value = expected_value
            if kind == "identity":
                if current is not frozen_value:
                    fail_fn(f"{role} executable dependency changed: {name}")
            else:
                try:
                    current_value = constant_payload(current)
                except TypeError:
                    fail_fn(f"{role} executable dependency type changed: {name}")
                if current_value != frozen_value:
                    fail_fn(f"{role} executable dependency changed: {name}")

    def assert_callable(role: str, fn: Any) -> None:
        if fingerprint_fn(fn, role.upper()) != callable_fingerprints[role]:
            fail_fn(f"{role} callable identity changed")
        assert_namespace(role, fn)

    def assert_method_authority() -> None:
        if replay_fn.__globals__.get("METHOD_SPECS") is not specs_object:
            fail_fn("method specification authority was rebound")
        if replay_fn.__globals__.get("METHOD_COMPLEXITY") is not complexity_object:
            fail_fn("method complexity authority was rebound")
        if set(specs_object) != set(specs_snapshot):
            fail_fn("method specification universe changed")
        for method_id, expected in specs_snapshot.items():
            actual = specs_object[method_id]
            if (
                actual.get("family") != expected["family"]
                or actual.get("units") != expected["units"]
                or tuple(actual.get("dependencies", ())) != expected["dependencies"]
                or actual.get("replay") is not helper_objects[method_id]
                or fingerprint_fn(
                    actual.get("replay"),
                    f"METHOD_HELPER:{method_id}",
                )
                != helper_fingerprints[method_id]
                or complexity_object.get(method_id) != complexity_snapshot[method_id]
            ):
                fail_fn("method helper/specification authority changed")

    def generator_call(**kwargs: Any) -> dict[str, Any]:
        assert_callable("generator", generator_fn)
        return generator_fn(**kwargs)

    def inference_call(**kwargs: Any) -> dict[str, Any]:
        assert_callable("inference", inference_fn)
        return inference_fn(**kwargs)

    def replay_call(**kwargs: Any) -> dict[str, Any]:
        assert_callable("replay", replay_fn)
        assert_method_authority()
        return replay_fn(**kwargs)

    def monte_carlo_call(**kwargs: Any) -> dict[str, Any]:
        assert_callable("monte_carlo", monte_carlo_fn)
        return monte_carlo_fn(**kwargs)

    def lookup(authority: Any) -> Mapping[str, Any]:
        if type(authority) is not VerifiedFoundationAuthority:
            fail_fn("candidate replay requires loader-issued authority")
        entry = storage.get(id(authority))
        if entry is None or entry[0]() is not authority:
            fail_fn("candidate replay requires loader-issued authority")
        return entry[1]

    class VerifiedFoundationAuthority:
        """Opaque, exact-type capability backed only by closure-private state."""

        __slots__ = ("__weakref__",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise TypeError("VerifiedFoundationAuthority is loader-issued only")

        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise TypeError("VerifiedFoundationAuthority cannot be subclassed")

        @property
        def authority_sha256(self) -> str:
            return lookup(self)["identity"]["authority_raw_sha256"]

        @property
        def authority(self) -> dict[str, Any]:
            return _thaw(lookup(self)["authority"])

        @property
        def benchmark(self) -> dict[str, Any]:
            return _thaw(lookup(self)["benchmark"])

        @property
        def config(self) -> dict[str, Any]:
            return _thaw(lookup(self)["config"])

        @property
        def candidate_registry(self) -> dict[str, Any]:
            return _thaw(lookup(self)["candidate_registry"])

        @property
        def candidate_payloads(self) -> dict[str, Any]:
            return {
                method_id: _thaw(candidate)
                for method_id, candidate in lookup(self)["candidate_payloads"].items()
            }

        @property
        def dependency_payloads(self) -> dict[str, Any]:
            return {
                role: _thaw(payload)
                for role, payload in lookup(self)["dependency_payloads"].items()
            }

    def issue(state: Mapping[str, Any]) -> VerifiedFoundationAuthority:
        capability = object.__new__(VerifiedFoundationAuthority)
        capability_id = id(capability)

        def expired(reference: weakref.ReferenceType[Any], *, key: int = capability_id) -> None:
            entry = storage.get(key)
            if entry is not None and entry[0] is reference:
                storage.pop(key, None)

        storage[capability_id] = (weakref.ref(capability, expired), state)
        return capability

    def load(
        root: Path | str = _IMPORTED_ROOT,
    ) -> VerifiedFoundationAuthority:
        assert_callable("load_impl", load_impl_fn)
        return load_impl_fn(
            root,
            authority_type=VerifiedFoundationAuthority,
            issue=issue,
            generator_call=generator_call,
            inference_call=inference_call,
            replay_call=replay_call,
            monte_carlo_call=monte_carlo_call,
            method_specs=specs_snapshot,
            method_complexity=complexity_snapshot,
        )

    def replay(
        *,
        authority: VerifiedFoundationAuthority,
        row_id: str,
        method_id: str,
    ) -> dict[str, Any]:
        assert_callable("replay_impl", replay_impl_fn)
        return replay_impl_fn(
            authority=authority,
            row_id=row_id,
            method_id=method_id,
            lookup=lookup,
            replay_call=replay_call,
            method_specs=specs_snapshot,
            method_complexity=complexity_snapshot,
        )

    return VerifiedFoundationAuthority, load, replay


(
    VerifiedFoundationAuthority,
    load_foundation_artifacts,
    replay_foundation_row_candidate,
) = _make_foundation_api()


def volume_basis_design(
    volume_values: list[float],
    *,
    center: float,
    scale: float = 1.0,
) -> np.ndarray:
    values = np.asarray(volume_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValidationFailure("volume values must be a finite vector")
    if type(center) is bool or type(scale) is bool:
        raise ValidationFailure("volume center/scale must be numeric")
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise ValidationFailure("volume center/scale must be finite and non-zero")
    centered = (values - float(center)) / float(scale)
    return np.column_stack([np.ones(values.size), centered, centered**2])


__all__ = [
    "AUTHORITY_ARTIFACT_ID",
    "AUTHORITY_LOCATOR",
    "BENCHMARK_ARTIFACT_ID",
    "BENCHMARK_LOCATOR",
    "CANDIDATE_REGISTRY_ARTIFACT_ID",
    "CANDIDATE_REGISTRY_LOCATOR",
    "FOUNDATION_CONFIG_ARTIFACT_ID",
    "FOUNDATION_CONFIG_LOCATOR",
    "VerifiedFoundationAuthority",
    "load_foundation_artifacts",
    "replay_foundation_row_candidate",
    "require_predictive_target_authority",
    "volume_basis_design",
]
