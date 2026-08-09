"""Regenerate canonical synthetic R-20 foundation artifacts from loader closure."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from . import r20_foundation as foundation


def _write(root: Path, locator: Path, payload: dict[str, Any]) -> bytes:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (foundation.canonical_json(payload) + "\n").encode()
    path.write_bytes(raw)
    return raw


def generate(root: Path) -> dict[str, dict[str, str]]:
    root = root.resolve()
    if root != foundation._IMPORTED_ROOT:
        raise ValueError("artifact generation must run from the imported source root")

    (
        authority,
        config,
        config_raw,
        benchmark,
        benchmark_raw,
        candidate_registry,
        candidate_registry_ref,
        dependency_payloads,
        dependency_refs,
    ) = foundation._expected_artifacts(
        foundation._source_hashes(root),
        generator_call=foundation.build_r20_benchmark,
        monte_carlo_call=foundation.monte_carlo_width_design,
    )

    authority_raw = _write(root, foundation.AUTHORITY_LOCATOR, authority)
    config_written = _write(root, foundation.FOUNDATION_CONFIG_LOCATOR, config)
    benchmark_written = _write(root, foundation.BENCHMARK_LOCATOR, benchmark)
    candidate_registry_raw = _write(
        root, foundation.CANDIDATE_REGISTRY_LOCATOR, candidate_registry
    )
    for role, payload in sorted(dependency_payloads.items()):
        payload_bytes = (foundation.canonical_json(payload) + "\n").encode()
        path = root / foundation._DEPENDENCY_LOCATORS[role]  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload_bytes)

    if config_written != config_raw:
        raise AssertionError("config bytes diverged from closure expectation")
    if benchmark_written != benchmark_raw:
        raise AssertionError("benchmark bytes diverged from closure expectation")
    if candidate_registry_raw != (
        foundation.canonical_json(candidate_registry) + "\n"
    ).encode():
        raise AssertionError("candidate registry bytes diverged from closure expectation")
    if candidate_registry_ref != foundation._artifact_ref(
        foundation.CANDIDATE_REGISTRY_LOCATOR,
        candidate_registry,
        candidate_registry_raw,
    ):
        raise AssertionError("candidate registry ref mismatch")

    return {
        "authority": foundation._artifact_ref(
            foundation.AUTHORITY_LOCATOR, authority, authority_raw
        ),
        "config": foundation._artifact_ref(
            foundation.FOUNDATION_CONFIG_LOCATOR, config, config_written
        ),
        "benchmark": foundation._artifact_ref(
            foundation.BENCHMARK_LOCATOR, benchmark, benchmark_written
        ),
        "candidate_registry": candidate_registry_ref,
        **{
            f"dependency:{role}": ref for role, ref in sorted(dependency_refs.items())
        },
    }


if __name__ == "__main__":
    print(json.dumps(generate(Path.cwd()), indent=2, sort_keys=True))
