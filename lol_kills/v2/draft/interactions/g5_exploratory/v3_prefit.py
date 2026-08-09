"""Versioned, no-read G5 pre-fit review for the corrected Player artifact.

The original G5 review bundle is historical evidence and remains byte-pinned to
the v2 Player development runner.  This module creates a separate v3 identity
boundary after the display-scale repair.  It never opens feature rows, target
rows, or the final holdout, and it deliberately refuses to describe the bundle
as executable: the checked-in G1 draft-feature snapshot still records the v2
Player artifact as its upstream membership origin and must be replayed before
any Draft Score fit can be authorized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from . import contract as historical


ROOT = Path(__file__).resolve().parents[5]
SCHEMA = "scryglass:g5-private-exploratory-prefit-contract:v3"
REVIEW_SCHEMA = "scryglass:g5-private-exploratory-prefit-review:v3"
G2_RUNNER = "lol_kills/v2/ratings/player/private_development_runner.py"
G2_MODEL = "lol_kills/v2/ratings/player/model.py"
G2_ARTIFACT = "data/lol/v2/models/player/real-v1/private-development-artifact-v3.json"
G2_RUNNER_RAW_SHA256 = "800755c6c4b425bb74690cce8ee8aea38db3cddf45016c7bec35a10ed5bfc5c7"
G2_MODEL_RAW_SHA256 = "426bc9d5b5de9014779fd4d2803421e851040111957a41567de5edc5b55782fa"
G2_ARTIFACT_RAW_SHA256 = "11fb9a43c6c2bb50d9c6046eb8e0fbbed3755607518bc483b9ad82bb556568e7"
G2_ARTIFACT_CANONICAL_SHA256 = "510d2cde52a92f92f6aa373bbe5c497d2b9dc652d1f7edf15f9cae006ee0f7a0"
G1_FEATURES_V3_MANIFEST = "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-manifest.json"
G1_FEATURES_V3_MANIFEST_RAW_SHA256 = "d77f0e357a539ae172f826e514547eb1db8ec599b26cf90af5bcc941b73de2a5"
G1_FEATURES_V3_MANIFEST_CANONICAL_SHA256 = "7de49f219718af73e18d364b36e0846283994541ccdbc8cd568475cb6380b733"
G1_FEATURES_V3_ROWS = "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-rows.jsonl"
G1_FEATURES_V3_ROWS_RAW_SHA256 = "e742631e1c12fb1af7148468a0d595ff6cf23e816af4edb20af162a04a6a9680"
G1_FEATURES_V3_ROWS_CANONICAL_SHA256 = "52d59dd0c41a212f7eb07b6f6132841f3c152f28324308b376042f8e262c141d"
G1_FEATURES_V3_TRANSFORM_RAW_SHA256 = "b5c366ede303bd4011d2d03ebbb638236b7e7caa43b3eaa1d1b1cd59cd913def"
NAMESPACE = ROOT / "data/lol/v2/models/draft-interactions/g5-exploratory-v3"


class G5V3PreFitError(ValueError):
    """A v3 no-read identity or claim-ceiling invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G5V3PreFitError("noncanonical G5 v3 payload") from error


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _raw_sha256(path: Path) -> str:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise G5V3PreFitError("unsafe v3 dependency")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_bound(locator: str, expected: str) -> dict[str, str]:
    actual = _raw_sha256(ROOT / locator)
    if actual != expected:
        raise G5V3PreFitError(f"v3 bound dependency changed: {locator}")
    return {"locator": locator, "raw_sha256": actual}


def _verify_canonical_artifact() -> dict[str, Any]:
    path = ROOT / G2_ARTIFACT
    raw = path.read_bytes()
    if _raw_sha256(path) != G2_ARTIFACT_RAW_SHA256:
        raise G5V3PreFitError("v3 G2 artifact raw identity changed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G5V3PreFitError("v3 G2 artifact is not JSON") from error
    if raw != canonical_bytes(payload) + b"\n":
        raise G5V3PreFitError("v3 G2 artifact is not canonical")
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != G2_ARTIFACT_CANONICAL_SHA256 or sha256(unsigned) != G2_ARTIFACT_CANONICAL_SHA256:
        raise G5V3PreFitError("v3 G2 artifact canonical identity changed")
    if payload.get("schema_version") != "scryglass:player-real-v1-private-development:v3":
        raise G5V3PreFitError("v3 G2 artifact schema mismatch")
    if payload.get("output_checks", {}).get("display_scale", {}).get("scale") != 400.0 / math.log(10.0):
        raise G5V3PreFitError("v3 G2 artifact display scale mismatch")
    return {
        "locator": G2_ARTIFACT,
        "raw_sha256": G2_ARTIFACT_RAW_SHA256,
        "canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
        "schema_version": payload["schema_version"],
        "selected_candidate": payload.get("decision", {}).get("selected_candidate_id"),
        "display_scale": payload["output_checks"]["display_scale"],
    }


def _verify_v3_feature_manifest() -> dict[str, Any]:
    """Bind feature metadata and bytes without decoding any feature row."""

    path = ROOT / G1_FEATURES_V3_MANIFEST
    raw = path.read_bytes()
    if _raw_sha256(path) != G1_FEATURES_V3_MANIFEST_RAW_SHA256:
        raise G5V3PreFitError("v3 feature manifest raw identity changed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G5V3PreFitError("v3 feature manifest is not JSON") from error
    if raw != canonical_bytes(payload) + b"\n":
        raise G5V3PreFitError("v3 feature manifest is not canonical")
    unsigned = dict(payload)
    if unsigned.pop("manifest_sha256", None) != G1_FEATURES_V3_MANIFEST_CANONICAL_SHA256 or sha256(unsigned) != G1_FEATURES_V3_MANIFEST_CANONICAL_SHA256:
        raise G5V3PreFitError("v3 feature manifest canonical identity changed")
    if payload.get("schema_version") != "scryglass:g1-lpl-completed-draft-features:v2" or payload.get("rows_locator") != G1_FEATURES_V3_ROWS:
        raise G5V3PreFitError("v3 feature manifest schema or row locator mismatch")
    if payload.get("rows_raw_sha256") != G1_FEATURES_V3_ROWS_RAW_SHA256 or payload.get("rows_canonical_sha256") != G1_FEATURES_V3_ROWS_CANONICAL_SHA256:
        raise G5V3PreFitError("v3 feature row identity mismatch")
    if payload.get("accepted_membership_origin", {}).get("g2_artifact_canonical_sha256") != G2_ARTIFACT_CANONICAL_SHA256:
        raise G5V3PreFitError("v3 feature G2 origin mismatch")
    if payload.get("upstream_rebind", {}).get("status") != "OUTCOME_FREE_FEATURE_REPLAY_WITH_V3_IDENTITY":
        raise G5V3PreFitError("v3 feature replay status mismatch")
    return {
        "manifest_locator": G1_FEATURES_V3_MANIFEST,
        "manifest_raw_sha256": G1_FEATURES_V3_MANIFEST_RAW_SHA256,
        "manifest_canonical_sha256": G1_FEATURES_V3_MANIFEST_CANONICAL_SHA256,
        "rows_locator": G1_FEATURES_V3_ROWS,
        "rows_raw_sha256": G1_FEATURES_V3_ROWS_RAW_SHA256,
        "rows_canonical_sha256": G1_FEATURES_V3_ROWS_CANONICAL_SHA256,
        "transform_raw_sha256": G1_FEATURES_V3_TRANSFORM_RAW_SHA256,
        "feature_row_count": payload.get("coverage", {}).get("feature_row_count"),
        "pick_count": payload.get("coverage", {}).get("pick_count"),
    }


def _identity_bundle() -> dict[str, Any]:
    # These are identity-only references; neither G1 row file is decoded.
    g1 = dict(historical.G1)
    g1_features = _verify_v3_feature_manifest()
    g2 = {
        "runner_locator": G2_RUNNER,
        "runner_raw_sha256": G2_RUNNER_RAW_SHA256,
        "model_locator": G2_MODEL,
        "model_raw_sha256": G2_MODEL_RAW_SHA256,
        "artifact_locator": G2_ARTIFACT,
        "artifact_raw_sha256": G2_ARTIFACT_RAW_SHA256,
        "artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
        "accepted_candidate": "static_baseline",
        "display_scale": {"anchor": 1500.0, "scale": 400.0 / math.log(10.0)},
    }
    return {
        "G1": g1,
        "G1_features": g1_features,
        "G2": g2,
        "clusters": dict(historical.CLUSTERS),
        "compatibility": {
            "status": "V3_FEATURE_REPLAY_BOUND_INDEPENDENT_REVIEW_REQUIRED",
            "reason_code": "FRESH_G5_EXECUTION_REVIEW_AND_PERMIT_REQUIRED",
            "feature_snapshot_g2_artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
            "current_g2_artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256,
            "required_action": "independently_review_v3_feature_replay_and_issue_fresh_g5_permit_before_any_protected_load",
            "execution_authorized": False,
        },
    }


def verify_v3_dependencies() -> dict[str, Any]:
    """Verify current v3 metadata and executable bytes without opening rows."""

    bindings = {
        "G2_runner": _verify_bound(G2_RUNNER, G2_RUNNER_RAW_SHA256),
        "G2_model": _verify_bound(G2_MODEL, G2_MODEL_RAW_SHA256),
        "G2_artifact": _verify_canonical_artifact(),
    }
    return {"bindings": bindings, "protected_reads": 0, "final_holdout_reads": 0}


def build_v3_prefit_bundle() -> dict[str, Any]:
    """Build the v3 no-read review bundle; never opens source rows."""

    identity = _identity_bundle()
    dependencies = verify_v3_dependencies()
    # Reuse only the frozen mathematical declarations.  No historical verifier
    # is called here because it would require the old G2 byte pin.
    unsigned = {
        "schema_id": SCHEMA,
        "state": "PREFIT_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "execution_authorization": False,
        "input_identities": identity,
        "candidate_protocol": historical._candidate_protocol(),
        "bootstrap": historical._bootstrap(),
        "mathematics_availability_uncertainty": historical._uncertainty_and_availability(),
        "target_access": {
            "pre_freeze_target_or_outcome_row_reads": 0,
            "final_holdout_reads": 0,
            "allowed_prefit_data": "identity hashes and metadata only",
            "later_execution_requires": "new independent review plus exact v3 feature replay and sealed target adapter",
        },
        "claim_ceiling": {
            "prefit_contract": True,
            "execution_authorization": False,
            "prediction": False,
            "forecast": False,
            "publication": False,
            "promotion": False,
            "sota": False,
            "reliability": False,
            "final_holdout": False,
        },
        "dependencies": dependencies,
        "research_record": historical._research_record(),
    }
    return {**unsigned, "artifact_sha256": sha256(unsigned)}


def validate_v3_prefit_bundle(bundle: Mapping[str, Any]) -> str:
    expected = build_v3_prefit_bundle()
    if not isinstance(bundle, Mapping) or dict(bundle) != expected:
        raise G5V3PreFitError("v3 prefit bundle is not the exact re-derived payload")
    return expected["artifact_sha256"]


def write_v3_prefit_bundle(path: Path = NAMESPACE / "prefit-contract.json") -> str:
    """Write the new bundle safely, without overwriting historical G5 files."""

    payload = build_v3_prefit_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise G5V3PreFitError("v3 output path is unsafe")
    data = canonical_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()
