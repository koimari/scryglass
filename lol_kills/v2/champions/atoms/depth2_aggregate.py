"""Build and validate the LCC depth-2 numeric atom index.

The index contains static champion descriptors.  It has no model or public
prediction authority.  Source bytes are pinned so a changed LCC corpus must
produce a new artifact and a new evaluation receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import HASH_RE, canonical_sha256

SCHEMA_VERSION = "scryglass:atom-corpus-aggregate:v2"
PUBLIC_PATCH = "26.16"
CLIENT_PATCH = "16.16"
TEMPO_CLASSES = (
    "burst",
    "sustained",
    "poke",
    "engage",
    "peel",
    "sustain",
    "utility",
    "setup",
)
FEATURE_KEYS = tuple(
    sorted(
        (
            "d2_ad_ratio",
            "d2_ap_ratio",
            "d2_burst",
            "d2_cc_duration",
            "d2_channels",
            "d2_dps",
            "d2_magic",
            "d2_mean_cd",
            "d2_mean_range",
            "d2_physical",
            "d2_recasts",
            "d2_resets",
            "d2_states",
            "d2_sustain",
            "d2_uptime",
            *(f"d2_tempo_{name}" for name in TEMPO_CLASSES),
        )
    )
)
ROW_KEYS = (
    "d2_burst",
    "d2_dps",
    "d2_mean_cd",
    "d2_mean_range",
    "d2_cc_duration",
    "d2_sustain",
    "d2_uptime",
    "d2_states",
    "d2_recasts",
    "d2_channels",
    "d2_resets",
    "d2_ap_ratio",
    "d2_ad_ratio",
    "d2_magic",
    "d2_physical",
    *(f"d2_tempo_{name}" for name in TEMPO_CLASSES),
)
DEFAULT_LCC_REPO = Path(
    os.environ.get(
        "SCRYGLASS_LCC_REPO",
        "/Users/river/Projects/league-combat-calculator",
    )
)
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "atom-corpus-aggregate-v2.json"
)
SOURCE_SCHEMA = "data/atoms/atoms.schema.v2.json"
SOURCE_GLOB = "data/atoms/v2/*.atoms.v2.json"
SOURCE_REPOSITORY = "https://github.com/koimari/league-combat-calculator"
_SLUG_RE = re.compile(r"^[a-z0-9]+$")


class Depth2AggregateError(ValueError):
    """Raised when the depth-2 corpus or its artifact fails validation."""


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _numbers(values: object) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _last_number(values: object) -> float:
    numbers = _numbers(values)
    return numbers[-1] if numbers else 0.0


def _max_number(values: object) -> float:
    numbers = _numbers(values)
    return max(numbers) if numbers else 0.0


def _atom_row(atoms: list[Mapping[str, Any]]) -> dict[str, float]:
    damage_values: list[float] = []
    cooldowns: list[float] = []
    ranges: list[float] = []
    cc_durations: list[float] = []
    sustain_durations: list[float] = []
    uptimes: list[float] = []
    tempo: Counter[str] = Counter()
    state_count = 0
    recasts = 0
    channels = 0
    resets = 0
    ap_ratios = 0.0
    ad_ratios = 0.0
    magic_damage = 0
    physical_damage = 0

    for atom in atoms:
        parameters = atom.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {}
        cycle = atom.get("cycle")
        if not isinstance(cycle, Mapping):
            cycle = {}
        family = str(atom.get("family") or "")

        damage = _max_number(parameters.get("damage"))
        if damage > 0:
            damage_values.append(damage)
        cooldown = cycle.get("cooldown_seconds")
        if not isinstance(cooldown, (int, float)):
            cooldown = _last_number(parameters.get("cooldown"))
        if isinstance(cooldown, (int, float)) and math.isfinite(float(cooldown)):
            cooldowns.append(float(cooldown))
        cast_range = _max_number(parameters.get("cast_range"))
        if cast_range > 0:
            ranges.append(cast_range)
        duration = _max_number(parameters.get("duration"))
        if family == "crowd-control-mobility" and duration > 0:
            cc_durations.append(duration)
        if family == "heal-shield" and duration > 0:
            sustain_durations.append(duration)
        uptime = cycle.get("uptime_fraction")
        if isinstance(uptime, (int, float)) and math.isfinite(float(uptime)):
            uptimes.append(float(uptime))
        tempo_class = cycle.get("tempo_class")
        if isinstance(tempo_class, str) and tempo_class:
            tempo[tempo_class] += 1

        states = atom.get("states")
        if not isinstance(states, list):
            states = []
        state_count += len(states)
        for state in states:
            if not isinstance(state, Mapping):
                continue
            if state.get("state") == "recast":
                recasts += 1
            elif state.get("state") == "channel":
                channels += 1
        reset_on = cycle.get("reset_on")
        resets += len(reset_on) if isinstance(reset_on, list) else 0

        ratios = parameters.get("ratios")
        if isinstance(ratios, Mapping):
            for key, value in ratios.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    if "ap" in str(key).casefold():
                        ap_ratios += float(value)
                    elif "ad" in str(key).casefold():
                        ad_ratios += float(value)
        damage_type = parameters.get("damage_type")
        if damage_type == "magic":
            magic_damage += 1
        elif damage_type == "physical":
            physical_damage += 1

    burst = sum(sorted(damage_values, reverse=True)[:3])
    mean_cooldown = sum(cooldowns) / len(cooldowns) if cooldowns else 0.0
    row: dict[str, float] = {
        "d2_ad_ratio": ad_ratios,
        "d2_ap_ratio": ap_ratios,
        "d2_burst": burst,
        "d2_cc_duration": float(sum(cc_durations)),
        "d2_channels": float(channels),
        "d2_dps": burst / mean_cooldown if mean_cooldown > 0 else 0.0,
        "d2_magic": float(magic_damage),
        "d2_mean_cd": mean_cooldown,
        "d2_mean_range": sum(ranges) / len(ranges) if ranges else 0.0,
        "d2_physical": float(physical_damage),
        "d2_recasts": float(recasts),
        "d2_resets": float(resets),
        "d2_states": float(state_count) / max(len(atoms), 1),
        "d2_sustain": float(sum(sustain_durations)),
        "d2_uptime": sum(uptimes) / len(uptimes) if uptimes else 0.0,
    }
    for tempo_class in TEMPO_CLASSES:
        row[f"d2_tempo_{tempo_class}"] = float(tempo.get(tempo_class, 0))
    return {key: row[key] for key in ROW_KEYS}


def build_depth2_payload(
    lcc_repo: Path = DEFAULT_LCC_REPO,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a development-only depth-2 index from one LCC checkout."""

    repo = lcc_repo.resolve()
    schema_path = repo / SOURCE_SCHEMA
    source_paths = sorted(repo.glob(SOURCE_GLOB))
    if not schema_path.is_file():
        raise Depth2AggregateError(f"LCC depth-2 schema is missing: {schema_path}")
    if not source_paths:
        raise Depth2AggregateError(f"LCC depth-2 corpus is missing under {repo}")

    champions: dict[str, dict[str, float]] = {}
    files = [schema_path, *source_paths]
    file_sha256 = {
        path.relative_to(repo).as_posix(): _sha256_path(path) for path in files
    }
    for path in source_paths:
        slug = path.name.removesuffix(".atoms.v2.json")
        if not _SLUG_RE.fullmatch(slug):
            raise Depth2AggregateError(f"invalid champion slug in {path.name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Depth2AggregateError(f"invalid LCC depth-2 file: {path}") from exc
        if not isinstance(payload, list) or not all(
            isinstance(atom, Mapping) for atom in payload
        ):
            raise Depth2AggregateError(f"{path} must contain a list of atom objects")
        champions[slug] = _atom_row(payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "public_patch": PUBLIC_PATCH,
        "client_patch": CLIENT_PATCH,
        "development_only": True,
        "feature_keys": list(FEATURE_KEYS),
        "tempo_classes": list(TEMPO_CLASSES),
        "provenance": {
            "source_repository": SOURCE_REPOSITORY,
            "lcc_commit": _git_head(repo),
            "source_schema": SOURCE_SCHEMA,
            "source_glob": SOURCE_GLOB,
            "file_sha256": dict(sorted(file_sha256.items())),
        },
        "champions": champions,
    }


def write_depth2_artifact(
    payload: Mapping[str, Any],
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    digest = canonical_sha256(unsigned)
    artifact = {**unsigned, "artifact_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=1, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return digest


def load_depth2_artifact(
    path: Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, dict[str, float]]:
    """Validate an index and return its champion rows."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Depth2AggregateError(f"cannot load depth-2 artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Depth2AggregateError("depth-2 artifact must be an object")
    submitted = payload.get("artifact_sha256")
    if not isinstance(submitted, str) or not HASH_RE.fullmatch(submitted):
        raise Depth2AggregateError("artifact_sha256 must be a lowercase sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    if canonical_sha256(unsigned) != submitted:
        raise Depth2AggregateError("depth-2 artifact sha256 changed")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Depth2AggregateError("depth-2 schema version changed")
    if payload.get("public_patch") != PUBLIC_PATCH:
        raise Depth2AggregateError("depth-2 public patch changed")
    if payload.get("client_patch") != CLIENT_PATCH:
        raise Depth2AggregateError("depth-2 client patch changed")
    if payload.get("development_only") is not True:
        raise Depth2AggregateError("depth-2 artifact exceeded development authority")
    if payload.get("feature_keys") != list(FEATURE_KEYS):
        raise Depth2AggregateError("depth-2 feature inventory changed")
    if payload.get("tempo_classes") != list(TEMPO_CLASSES):
        raise Depth2AggregateError("depth-2 tempo inventory changed")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Depth2AggregateError("depth-2 provenance is missing")
    if provenance.get("source_repository") != SOURCE_REPOSITORY:
        raise Depth2AggregateError("depth-2 source repository changed")
    commit = provenance.get("lcc_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Depth2AggregateError("depth-2 LCC commit is invalid")
    files = provenance.get("file_sha256")
    if (
        not isinstance(files, Mapping)
        or SOURCE_SCHEMA not in files
        or len(files) != 167
    ):
        raise Depth2AggregateError("depth-2 source hashes are missing")
    for key, value in files.items():
        if not isinstance(key, str) or not HASH_RE.fullmatch(str(value)):
            raise Depth2AggregateError("depth-2 source hash is invalid")

    champions = payload.get("champions")
    if not isinstance(champions, Mapping) or len(champions) != 166:
        raise Depth2AggregateError("depth-2 champion index is empty")
    result: dict[str, dict[str, float]] = {}
    for slug, row in champions.items():
        if not isinstance(slug, str) or not _SLUG_RE.fullmatch(slug):
            raise Depth2AggregateError("depth-2 champion slug is invalid")
        if not isinstance(row, Mapping) or set(row) != set(FEATURE_KEYS):
            raise Depth2AggregateError(f"depth-2 feature row changed for {slug}")
        validated_row: dict[str, float] = {}
        for key in FEATURE_KEYS:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Depth2AggregateError(f"champions.{slug}.{key} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise Depth2AggregateError(f"champions.{slug}.{key} must be finite")
            validated_row[key] = number
        result[slug] = validated_row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the LCC depth-2 atom index")
    parser.add_argument("--lcc-repo", type=Path, default=DEFAULT_LCC_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args()
    payload = build_depth2_payload(args.lcc_repo, generated_at=args.generated_at)
    digest = write_depth2_artifact(payload, args.out)
    print(f"wrote {args.out}")
    print(f"artifact_sha256 = {digest}")
    print(f"champions = {len(payload['champions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
