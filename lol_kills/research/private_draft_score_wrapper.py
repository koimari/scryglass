"""Private-only wrapper that adds an optional R9E composite diagnostic.

The existing owner-only scorer remains the source for the original
``draft_score`` and selected model output. This wrapper adds one separate
``draft_score_recipe`` field. R9E output stays in a separate
``development_composite`` field because its probability includes strength
controls. It never replaces component probabilities and it never changes
authority flags.

The private scorer lives under the Git-ignored experiments directory. This
tracked seam lets the local ``who-wins-this-game`` skill call it without
putting private model data in the public app or public pack.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep absolute-path CLI calls independent of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lol_kills.research.private_draft_recipe import (
    R9E_CANDIDATE_ID,
    SCHEMA_VERSION,
    PrivateDraftRecipeError,
    score_registered_draft_with_private_r9e,
)


PRIVATE_ROOT = REPO_ROOT / "data/lol/v2/experiments/private-oe-probability"
DEFAULT_SCORER = PRIVATE_ROOT / "score_private_draft.py"
DEFAULT_BINDING = PRIVATE_ROOT / "private-model-2026/active/draft/descriptive-r9e-binding.json"
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
AUTHORITY_KEYS = (
    "model_validation_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)


class PrivateDraftScoreWrapperError(ValueError):
    """Raised when the owner-only scorer or descriptive recipe is unavailable."""


def _authority_is_closed(value: Any) -> bool:
    return isinstance(value, Mapping) and all(value.get(key) is False for key in AUTHORITY_KEYS)


def _unavailable_recipe(reason: str, missing_inputs: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": R9E_CANDIDATE_ID,
        "mode": "development_composite",
        "estimand": "composite_map_probability",
        "status": "unavailable",
        "authority": "unavailable",
        "reason": reason,
        "missing_inputs": list(missing_inputs),
        "claim_ceiling": {
            "descriptive_draft_score": False,
            "composite_diagnostic": False,
            "public_probability": False,
            "recommendation": False,
            "betting": False,
        },
    }


def _run_owner_scorer(
    scorer: Path,
    scorer_args: Sequence[str],
) -> dict[str, Any]:
    if not scorer.is_absolute() or not scorer.is_file() or scorer.is_symlink():
        raise PrivateDraftScoreWrapperError(f"owner-only scorer is unavailable: {scorer}")
    process = subprocess.run(
        [sys.executable, str(scorer), *scorer_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or "owner-only scorer failed"
        raise PrivateDraftScoreWrapperError(message)
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise PrivateDraftScoreWrapperError("owner-only scorer returned invalid JSON") from error
    if not isinstance(value, dict):
        raise PrivateDraftScoreWrapperError("owner-only scorer returned an invalid object")
    if not _authority_is_closed(value.get("authority")):
        raise PrivateDraftScoreWrapperError("owner-only scorer widened its authority")
    return value


def _load_private_modules() -> tuple[Any, Any]:
    """Load the ignored scorer helpers without adding them to public imports."""

    private_root = str(PRIVATE_ROOT)
    if private_root not in sys.path:
        sys.path.insert(0, private_root)
    try:
        import prospective_runner  # type: ignore[import-not-found]
        import score_private_draft  # type: ignore[import-not-found]
    except ImportError as error:
        raise PrivateDraftScoreWrapperError("private scorer helpers are unavailable") from error
    return prospective_runner, score_private_draft


def _build_recipe(
    args: argparse.Namespace,
) -> dict[str, Any]:
    binding = Path(args.recipe_binding).expanduser()
    if not binding.is_file() or binding.is_symlink():
        return _unavailable_recipe("r9e_binding_unavailable", ("recipe_binding",))
    missing = [
        name
        for name, value in (
            ("as_of", args.as_of),
            ("patch", args.patch),
            ("r9e_player_elo_p", args.r9e_player_elo_p),
            ("r9e_player_elo_sigma", args.r9e_player_elo_sigma),
        )
        if value is None or value == ""
    ]
    if missing:
        return _unavailable_recipe("r9e_query_inputs_missing", missing)
    try:
        prospective_runner, private_scorer = _load_private_modules()
        bundle = prospective_runner.load_active_bundle(
            PRIVATE_ROOT / "private-model-2026/active"
        )
        blue_players = private_scorer.exact_five(args.blue_players, "blue players")
        red_players = private_scorer.exact_five(args.red_players, "red players")
        registration = {
            "event": {"league": args.league},
            "teams": [
                private_scorer.resolved_team("blue", args.blue_name, blue_players, bundle),
                private_scorer.resolved_team("red", args.red_name, red_players, bundle),
            ],
        }
        blue_picks = private_scorer.exact_five(args.blue, "blue draft")
        red_picks = private_scorer.exact_five(args.red, "red draft")
        draft = {
            "blue": dict(zip(ROLE_ORDER, blue_picks)),
            "red": dict(zip(ROLE_ORDER, red_picks)),
        }
        blue, red = registration["teams"]
        sigma_pair = math.hypot(float(blue["team_rating_sigma"]), float(red["team_rating_sigma"]))
        player_ratings = {
            player["display_name"]: float(player["rating_mu"])
            for team in registration["teams"]
            for player in team["players"]
        }
        return score_registered_draft_with_private_r9e(
            binding,
            registration,
            draft,
            date=str(args.as_of),
            patch=str(args.patch),
            mu_diff=float(blue["team_rating_mu"]) - float(red["team_rating_mu"]),
            sigma_pair=sigma_pair,
            player_elo={
                "p": float(args.r9e_player_elo_p),
                "sigma": float(args.r9e_player_elo_sigma),
            },
            player_ratings=player_ratings,
        )
    except (OSError, PrivateDraftRecipeError, ValueError, TypeError, KeyError) as error:
        return _unavailable_recipe(f"r9e_unavailable:{type(error).__name__}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    scorer_args = [
        "--league", args.league,
        "--blue-name", args.blue_name,
        "--red-name", args.red_name,
        "--blue-players", args.blue_players,
        "--red-players", args.red_players,
        "--blue", args.blue,
        "--red", args.red,
    ]
    output = _run_owner_scorer(Path(args.scorer), scorer_args)
    recipe = _build_recipe(args)
    if recipe.get("status") == "development_only" and recipe.get("mode") == "development_composite":
        output["development_composite"] = recipe
        output["draft_score_recipe"] = _unavailable_recipe(
            "r9e_checkpoint_is_composite_not_draft_score"
        )
    else:
        output["draft_score_recipe"] = recipe
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", default=str(DEFAULT_SCORER))
    parser.add_argument("--recipe-binding", default=str(DEFAULT_BINDING))
    parser.add_argument("--as-of", help="Exact UTC query time after the R9E fit boundary")
    parser.add_argument("--patch", help="Exact source patch token, such as 16.15")
    parser.add_argument("--r9e-player-elo-p", type=float)
    parser.add_argument("--r9e-player-elo-sigma", type=float)
    parser.add_argument("--league", required=True)
    parser.add_argument("--blue-name", required=True)
    parser.add_argument("--red-name", required=True)
    parser.add_argument("--blue-players", required=True)
    parser.add_argument("--red-players", required=True)
    parser.add_argument("--blue", required=True)
    parser.add_argument("--red", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
    except PrivateDraftScoreWrapperError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
