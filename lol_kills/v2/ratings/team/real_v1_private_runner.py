"""Private, exact-five Team Rating bridge for the real-v1 Player artifact.

This is intentionally a very small L5 handoff.  It can aggregate an
independently attested ordered LPL five at the Player artifact's exact as-of
boundary, but it cannot invent such a roster from historical match rows.
Consequently the checked-in real-v1 artifact is explicitly unavailable until
an authoritative current-roster receipt is supplied.  It never adds an
organisation residual, League Rating, policy effect, or lineup synergy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from lol_kills.v2.ratings.player.private_development_runner import (
    DISPLAY_ANCHOR,
    DISPLAY_SCALE,
    verify_private_development_artifact,
)


ROOT = Path(__file__).resolve().parents[4]
PLAYER_ARTIFACT_PATH = ROOT / "data/lol/v2/models/player/real-v1/private-development-artifact-v3.json"
PLAYER_ARTIFACT_SHA256 = "510d2cde52a92f92f6aa373bbe5c497d2b9dc652d1f7edf15f9cae006ee0f7a0"
TEAM_ARTIFACT_PATH = ROOT / "data/lol/v2/models/team/real-v1/private-development-artifact-v3.json"
SCHEMA_VERSION = "scryglass:team-real-v1-private-development:v2"
ROLES = ("top", "jungle", "mid", "bot", "support")

# This is an exact subset of what the accepted L4 artifact permits.  It is
# deliberately incapable of authorising any customer-facing interpretation.
CLAIM_CEILING = {
    "private_model_fit": False,
    "private_rank_selection": False,
    "current_exact_roster_aggregation": True,
    "prediction": False,
    "production": False,
    "publication": False,
    "promotion": False,
    "sota": False,
    "final_holdout": False,
}


class TeamRealV1Error(ValueError):
    """A private Team Rating input or artifact failed its closed boundary."""


class TeamRealV1Unavailable(TeamRealV1Error):
    """The requested Team Rating is not identified by the supplied receipts."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TeamRealV1Error("team artifact contains a non-canonical or non-finite value") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TeamRealV1Error(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class RatedPlayer:
    """One accepted L4 posterior tied to one declared roster role."""

    role: str
    player_id: str
    display_mean: float
    display_uncertainty: float


@dataclass(frozen=True)
class ExactCurrentLplRoster:
    """An externally attested, role-ordered current roster at the L4 as-of map."""

    roster_id: str
    organization_id: str
    league_id: str
    as_of_source_game_id: str
    identity_receipt_sha256: str
    players: tuple[RatedPlayer, ...]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, ratings: Mapping[str, Mapping[str, Any]], expected_as_of_source_game_id: str,
        expected_identity_receipt_sha256: str | None,
    ) -> "ExactCurrentLplRoster":
        required = {
            "organization_id", "league_id", "as_of_source_game_id", "identity_receipt_sha256",
            "official", "active", "fresh", "ambiguous", "substitute", "players",
        }
        if set(value) != required:
            raise TeamRealV1Unavailable("EXACT_ROSTER_SCHEMA_MISMATCH")
        if value.get("official") is not True or value.get("active") is not True or value.get("fresh") is not True:
            raise TeamRealV1Unavailable("EXACT_ROSTER_INACTIVE_OR_STALE")
        if value.get("ambiguous") is not False:
            raise TeamRealV1Unavailable("EXACT_ROSTER_AMBIGUOUS")
        if value.get("substitute") is not False:
            raise TeamRealV1Unavailable("EXACT_ROSTER_SUBSTITUTE")
        if value.get("league_id") != "LPL":
            raise TeamRealV1Unavailable("EXACT_ROSTER_LEAGUE_IDENTITY_MISMATCH")
        if value.get("as_of_source_game_id") != expected_as_of_source_game_id:
            raise TeamRealV1Unavailable("EXACT_ROSTER_STALE_OR_AS_OF_MISMATCH")
        if not isinstance(value.get("organization_id"), str) or not value["organization_id"]:
            raise TeamRealV1Unavailable("EXACT_ROSTER_ORGANIZATION_IDENTITY_MISMATCH")
        receipt = value.get("identity_receipt_sha256")
        if not isinstance(receipt, str) or len(receipt) != 64 or any(char not in "0123456789abcdef" for char in receipt):
            raise TeamRealV1Unavailable("EXACT_ROSTER_IDENTITY_RECEIPT_MISSING")
        if expected_identity_receipt_sha256 is None:
            raise TeamRealV1Unavailable("EXACT_ROSTER_EXTERNAL_RECEIPT_PIN_REQUIRED")
        if receipt != expected_identity_receipt_sha256:
            raise TeamRealV1Unavailable("EXACT_ROSTER_IDENTITY_RECEIPT_MISMATCH")
        raw_players = value.get("players")
        if not isinstance(raw_players, list) or len(raw_players) != 5:
            raise TeamRealV1Unavailable("EXACT_ROSTER_NOT_ORDERED_FIVE")
        parsed: list[RatedPlayer] = []
        for raw in raw_players:
            if not isinstance(raw, Mapping) or set(raw) != {"role", "player_id"}:
                raise TeamRealV1Unavailable("EXACT_ROSTER_PLAYER_IDENTITY_MISMATCH")
            role, player_id = raw.get("role"), raw.get("player_id")
            if not isinstance(role, str) or not isinstance(player_id, str):
                raise TeamRealV1Unavailable("EXACT_ROSTER_PLAYER_IDENTITY_MISMATCH")
            posterior = ratings.get(player_id)
            if posterior is None:
                raise TeamRealV1Unavailable("PLAYER_POSTERIOR_IDENTITY_MISMATCH")
            parsed.append(RatedPlayer(role, player_id, _finite(posterior.get("posterior_mean"), "player posterior mean"), _finite(posterior.get("posterior_uncertainty"), "player posterior uncertainty")))
        if tuple(player.role for player in parsed) != ROLES or len({player.player_id for player in parsed}) != 5:
            raise TeamRealV1Unavailable("EXACT_ROSTER_ROLE_OR_PLAYER_IDENTITY_MISMATCH")
        identity = {
            "league_id": "LPL", "as_of_source_game_id": expected_as_of_source_game_id,
            "roles": [{"role": player.role, "player_id": player.player_id} for player in parsed],
        }
        return cls(_sha256(identity), value["organization_id"], "LPL", expected_as_of_source_game_id, receipt, tuple(parsed))


def load_accepted_static_player_artifact(path: Path = PLAYER_ARTIFACT_PATH) -> dict[str, Any]:
    """Load only the independently pinned, validation-gated static baseline."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TeamRealV1Unavailable("ACCEPTED_PLAYER_ARTIFACT_UNAVAILABLE") from error
    artifact = verify_private_development_artifact(value, expected_artifact_sha256=PLAYER_ARTIFACT_SHA256)
    decision = artifact.get("decision")
    scope = artifact.get("private_scope")
    development = artifact.get("development_winner_posterior_ratings")
    if decision != {
        "development_winner_candidate_id": "static_baseline",
        "external_validation_gate_passed": True,
        "selected_candidate_id": "static_baseline",
    }:
        raise TeamRealV1Unavailable("ACCEPTED_STATIC_BASELINE_NOT_AVAILABLE")
    if not isinstance(scope, Mapping) or scope.get("authorizes") != ["private_model_fit", "private_rank_selection"]:
        raise TeamRealV1Unavailable("PLAYER_ARTIFACT_PRIVATE_SCOPE_MISMATCH")
    if not isinstance(development, Mapping) or development.get("candidate_id") != "static_baseline" or development.get("validation_gate_passed") is not True:
        raise TeamRealV1Unavailable("PLAYER_STATIC_POSTERIOR_NOT_AVAILABLE")
    if not isinstance(development.get("as_of_source_game_id"), str) or not isinstance(development.get("ratings"), list):
        raise TeamRealV1Unavailable("PLAYER_STATIC_POSTERIOR_IDENTITY_MISSING")
    return artifact


def _ratings_by_id(player_artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    development = player_artifact["development_winner_posterior_ratings"]
    ratings = development["ratings"]
    result: dict[str, Mapping[str, Any]] = {}
    for item in ratings:
        if not isinstance(item, Mapping) or not isinstance(item.get("player_id"), str) or item["player_id"] in result:
            raise TeamRealV1Unavailable("PLAYER_POSTERIOR_IDENTITY_MISMATCH")
        _finite(item.get("posterior_mean"), "player posterior mean")
        if _finite(item.get("posterior_uncertainty"), "player posterior uncertainty") < 0.0:
            raise TeamRealV1Unavailable("PLAYER_POSTERIOR_UNCERTAINTY_INVALID")
        result[item["player_id"]] = item
    return result


def aggregate_exact_current_lpl_five(
    roster: Mapping[str, Any], *, expected_identity_receipt_sha256: str | None = None,
    player_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the ordered-five ``x^T mu`` L4-scale aggregation.

    L4 has authorised a diagonal assumed-density posterior, not a fitted full
    joint covariance.  The 5x5 diagonal representation below is therefore a
    model-assumption representation: its off-diagonals are zero *by that
    stated approximation*, never a newly fitted independence or synergy fact.
    """

    artifact = load_accepted_static_player_artifact() if player_artifact is None else dict(player_artifact)
    # Re-verify even an injected mapping before it can influence aggregation.
    artifact = verify_private_development_artifact(artifact, expected_artifact_sha256=PLAYER_ARTIFACT_SHA256)
    development = artifact["development_winner_posterior_ratings"]
    exact = ExactCurrentLplRoster.from_mapping(
        roster, ratings=_ratings_by_id(artifact), expected_as_of_source_game_id=development["as_of_source_game_id"],
        expected_identity_receipt_sha256=expected_identity_receipt_sha256,
    )
    display_covariance = [
        [player.display_uncertainty ** 2 if i == j else 0.0 for j, _ in enumerate(exact.players)]
        for i, player in enumerate(exact.players)
    ]
    latent_covariance = [[entry / (DISPLAY_SCALE ** 2) for entry in row] for row in display_covariance]
    weights = [1.0 / 5.0] * 5
    player_latents = [(player.display_mean - DISPLAY_ANCHOR) / DISPLAY_SCALE for player in exact.players]
    team_latent_mean = sum(weight * latent for weight, latent in zip(weights, player_latents))
    team_latent_variance = sum(weights[i] * latent_covariance[i][j] * weights[j] for i in range(5) for j in range(5))
    team_mean = DISPLAY_ANCHOR + DISPLAY_SCALE * team_latent_mean
    team_variance = DISPLAY_SCALE ** 2 * team_latent_variance
    if team_variance < 0.0 or not math.isfinite(team_variance):
        raise TeamRealV1Error("TEAM_COVARIANCE_NOT_PSD")
    return {
        "status": "private_development_only",
        "roster_id": exact.roster_id,
        "organization_id": exact.organization_id,
        "league_id": "LPL",
        "as_of_source_game_id": exact.as_of_source_game_id,
        "player_ids": [player.player_id for player in exact.players],
        "roles": list(ROLES),
        "scale": {"display_anchor": DISPLAY_ANCHOR, "display_scale": DISPLAY_SCALE, "team_latent_definition": "mean_of_five_player_latents"},
        "team_posterior_latent_mean": team_latent_mean,
        "team_posterior_latent_variance": team_latent_variance,
        "team_posterior_display_mean": team_mean,
        "team_posterior_display_variance": team_variance,
        "team_posterior_display_uncertainty": math.sqrt(team_variance),
        "player_display_covariance": display_covariance,
        "player_latent_covariance": latent_covariance,
        "covariance_assumption": {
            "kind": "DIAGONAL_ASSUMED_DENSITY_REPRESENTATION",
            "joint_covariance_status": "unavailable",
            "off_diagonal": "zero_by_l4_diagonal_model_assumption_not_new_estimate",
            "full_joint_covariance": None,
        },
        "components": {
            "player_aggregation": {"status": "available", "kind": "exact_ordered_five_mean"},
            "lineup_synergy": {"status": "unavailable", "value": None, "blocker": "not identified by the accepted player-only artifact"},
            "policy": {"status": "unavailable", "value": None, "blocker": "not identified by the accepted player-only artifact"},
            "league_rating": {"status": "unavailable", "value": None, "blocker": "not identified for the regional LPL-only artifact"},
        },
        "claim_ceiling": dict(CLAIM_CEILING),
    }


def _baseline_comparison(player_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Carry the L4 evidence while refusing a new team prediction claim."""

    static = next(item for item in player_artifact["candidate_results"] if item["candidate"]["candidate_id"] == "static_baseline")
    return {
        "status": "unavailable",
        "blocker": "exact ordered-five roster receipts are absent for every development and validation target",
        "reason": "a team-level predictive comparison is not identified by the accepted player-only artifact",
        "accepted_player_only_static_baseline": {
            # The carried L4 metrics are development-only evidence, not a
            # newly identified Team Rating comparison.  Keep their original
            # metric fields for inspection while attaching the boundary at
            # the same level.
            "development": {"status": "development_only", **static["development"]},
            "validation": {"status": "development_only", **static["validation"]},
            "fold_prediction_sha256": static["fold_prediction_sha256"],
        },
    }


def build_private_team_real_v1_artifact(
    *, player_artifact_path: Path = PLAYER_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Build the deterministic private L5 handoff; default state is unavailable."""

    player = load_accepted_static_player_artifact(player_artifact_path)
    development = player["development_winner_posterior_ratings"]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": "UNAVAILABLE",
        "private_scope": {
            "authorizes": ["private_exact_roster_player_only_aggregation"],
            "blocked": ["forecast", "prediction", "production", "publication", "promotion", "sota", "final_holdout", "team_rank"],
        },
        "claim_ceiling": dict(CLAIM_CEILING),
        "accepted_player_artifact": {
            "artifact_sha256": PLAYER_ARTIFACT_SHA256,
            "schema_version": player["schema_version"],
            "adapter_input_pins": player["adapter_input_pins"],
            "candidate_id": "static_baseline",
            "as_of_source_game_id": development["as_of_source_game_id"],
            "ordered_origin_sha256": development["ordered_origin_sha256"],
        },
        "aggregation": {
            "status": "unavailable",
            "blocker": "no accepted current official LPL roster receipt is present in the Player Rating artifact",
            "required_receipt": "exact ordered top/jungle/mid/bot/support roster, active/fresh/official/non-substitute/non-ambiguous, pinned to the player artifact as-of source game",
        },
        "predictive_comparison": _baseline_comparison(player),
        "unavailable_components": {
            "lineup_synergy": "not identified", "policy": "not identified", "league_rating": "not identified for regional LPL-only aggregation",
        },
    }
    value["artifact_sha256"] = _sha256(value)
    return value


def verify_private_team_real_v1_artifact(
    artifact: Mapping[str, Any], *, expected_artifact_sha256: str,
) -> dict[str, Any]:
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != _sha256(unsigned):
        raise TeamRealV1Error("TEAM_ARTIFACT_DIGEST_MISMATCH")
    if claimed != expected_artifact_sha256:
        raise TeamRealV1Error("TEAM_ARTIFACT_EXTERNAL_PIN_MISMATCH")
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("result_state") != "UNAVAILABLE":
        raise TeamRealV1Error("TEAM_ARTIFACT_STATE_MISMATCH")
    if artifact.get("claim_ceiling") != CLAIM_CEILING:
        raise TeamRealV1Error("TEAM_ARTIFACT_CLAIM_CEILING_MISMATCH")
    if artifact.get("private_scope", {}).get("blocked") != ["forecast", "prediction", "production", "publication", "promotion", "sota", "final_holdout", "team_rank"]:
        raise TeamRealV1Error("TEAM_ARTIFACT_SCOPE_MISMATCH")
    if artifact.get("aggregation", {}).get("status") != "unavailable" or artifact.get("predictive_comparison", {}).get("status") != "unavailable":
        raise TeamRealV1Error("TEAM_ARTIFACT_IDENTIFICATION_BOUNDARY_MISMATCH")
    return dict(artifact)


def write_private_team_real_v1_artifact(path: Path = TEAM_ARTIFACT_PATH) -> str:
    artifact = build_private_team_real_v1_artifact()
    absolute = path.absolute()
    # The repository root and the platform temp directory are independently
    # known directory anchors.  Preserve a lexical relative suffix from either
    # so that a symlink introduced below the anchor cannot be erased by
    # ``resolve`` (while normal macOS /var -> /private/var plumbing remains
    # usable for a temporary test output).
    current = Path(absolute.anchor)
    parent_parts = absolute.parts[1:-1]
    for logical_anchor in (ROOT, Path(tempfile.gettempdir())):
        try:
            relative = absolute.relative_to(logical_anchor.absolute())
        except ValueError:
            continue
        current = logical_anchor.resolve()
        parent_parts = relative.parts[:-1]
        break
    # Walk every parent with lstat rather than resolving it: a symlinked
    # component must not redirect an otherwise safe-looking output path.
    for part in parent_parts:
        current = current / part
        try:
            parent_metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise TeamRealV1Error("TEAM_ARTIFACT_PARENT_MISSING") from error
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise TeamRealV1Error("TEAM_ARTIFACT_PARENT_UNSAFE")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TeamRealV1Error("TEAM_ARTIFACT_OUTPUT_UNSAFE")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(artifact) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact["artifact_sha256"]
