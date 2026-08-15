from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from lol_kills.bookmaker_quote_capture import (
    QuoteCaptureError,
    RegisteredQuoteUnavailable,
    load_registered_quote,
)
from lol_kills.draft_model import predict_total
from lol_kills.live_model import evaluate_live_state
from lol_kills.live_oe_prior import load_oe_pace_prior
from lol_kills.live_totals_candidate import (
    development_candidate_path,
    validate_development_candidate,
)
from lol_kills.live_totals_model import price_live_totals
from lol_kills.market_decision import (
    evaluate_two_way_market,
    unavailable_authority,
    validate_authority_receipt,
)
from lol_kills.v2.market.match_winner_future_protocol_v1 import (
    SETTLEMENT_RULE_ID as MATCH_WINNER_SETTLEMENT_RULE_ID,
)
from lol_kills.v2.market.event_probability_v1 import (
    DEFAULT_REGISTRY as MATCH_WINNER_PROBABILITY_REGISTRY,
    EventProbabilityError,
    load_registered_event_probability,
)
from lol_kills.pregame_roster_capture import (
    PregameRosterError,
    RegisteredPregameRosterUnavailable,
    load_registered_pregame_roster,
)
from lol_kills.private_decision_readiness import (
    inspect_private_decision_readiness,
)
from lol_kills.private_rating_authority import (
    RatingAuthorityError,
    RegisteredEventRatingUnavailable,
    load_registered_event_rating,
)


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_PATH = ROOT / "apps" / "scryglass" / "data" / "draft" / "context.json"
RUNTIME_PATH = ROOT / "apps" / "scryglass" / "data" / "draft" / "runtime.json"
DRAFT_MODEL_PATH = ROOT / "data" / "lol" / "draft_model.json"
LIVE_COEFS_PATH = ROOT / "data" / "lol" / "models" / "draft_live_coefs.json"
LIVE_TOTALS_PATH = development_candidate_path(ROOT)
TSX_PATH = ROOT / "apps" / "scryglass" / "node_modules" / ".bin" / "tsx"
PREGAME_BRIDGE_PATH = Path(__file__).with_name("pregame_bridge.ts")
APP_PATH = ROOT / "apps" / "scryglass"
PRIVATE_AUTHORITY_DIR = ROOT / "data" / "lol" / "private_market_authority"
QUOTE_REGISTRY_LOCATOR = "data/lol/private_market_quotes/registry.json"
QUOTE_REGISTRY_SHA_ENV = "SCRYGLASS_PRIVATE_QUOTE_REGISTRY_SHA256"
ROSTER_REGISTRY_LOCATOR = "data/lol/private_pregame_rosters/registry.json"
ROSTER_REGISTRY_SHA_ENV = "SCRYGLASS_PRIVATE_ROSTER_REGISTRY_SHA256"
RATING_REGISTRY_LOCATOR = "data/lol/private_rating_authority/registry.json"
RATING_REGISTRY_SHA_ENV = "SCRYGLASS_PRIVATE_RATING_REGISTRY_SHA256"
MATCH_WINNER_PROBABILITY_REGISTRY_SHA_ENV = (
    "SCRYGLASS_PRIVATE_MATCH_WINNER_PROBABILITY_REGISTRY_SHA256"
)

AUTHORITY_FILES = {
    "match_winner": PRIVATE_AUTHORITY_DIR / "match_winner.json",
    "total_kills": PRIVATE_AUTHORITY_DIR / "total_kills.json",
}
AUTHORITY_SHA_ENV = {
    "match_winner": "SCRYGLASS_PRIVATE_MATCH_WINNER_AUTHORITY_SHA256",
    "total_kills": "SCRYGLASS_PRIVATE_TOTAL_KILLS_AUTHORITY_SHA256",
}

ROLES = ("Top", "Jungle", "Mid", "Bottom", "Support")
LEAGUES = ("LCK", "LPL", "LEC", "LCS", "MSI", "EWC", "Worlds")


class ModelInputError(ValueError):
    pass


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelInputError(f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ModelInputError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ModelInputError(f"{label} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ModelInputError(f"{label} must be at most {maximum:g}")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int = 99) -> int:
    number = _number(value, label, minimum=minimum, maximum=maximum)
    if int(number) != number:
        raise ModelInputError(f"{label} must be a whole number")
    return int(number)


def _optional_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value in (None, ""):
        return None
    return _number(value, label, minimum=minimum, maximum=maximum)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _logit(probability: float) -> float:
    p = min(1 - 1e-9, max(1e-9, probability))
    return math.log(p / (1 - p))


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_binding_sha256(market_type: str) -> str:
    if market_type == "total_kills":
        paths = (LIVE_TOTALS_PATH,)
    elif market_type == "match_winner":
        paths = (
            CONTEXT_PATH,
            RUNTIME_PATH,
            LIVE_COEFS_PATH,
            PREGAME_BRIDGE_PATH,
            APP_PATH / "src" / "lib" / "draftScore.ts",
            APP_PATH / "src" / "lib" / "draftTerminalScore.ts",
            ROOT
            / "data"
            / "lol"
            / "v2"
            / "models"
            / "draft-terminal"
            / "terminal-model-neutral-development-v1.json",
            ROOT / "lol_kills" / "live_model.py",
            ROOT / "lol_kills" / "pregame_roster_capture.py",
            ROOT / "lol_kills" / "private_rating_authority.py",
            ROOT
            / "lol_kills"
            / "v2"
            / "ratings"
            / "semantic_rating_authority_v1.py",
            ROOT
            / "lol_kills"
            / "v2"
            / "draft"
            / "terminal"
            / "semantic_draft_authority_v1.py",
            ROOT
            / "lol_kills"
            / "v2"
            / "draft"
            / "terminal"
            / "promotion.py",
            Path(__file__),
        )
    else:
        raise ValueError(f"unsupported market type: {market_type}")
    manifest = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(paths, key=lambda item: str(item))
    }
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _private_market_authority(
    *, league: str, market_type: str, as_of: datetime
) -> dict[str, Any]:
    """Load a receipt only when its digest is independently pinned in the environment."""
    try:
        artifact_sha256 = _model_binding_sha256(market_type)
    except OSError:
        result = unavailable_authority("market_model_binding_unavailable")
        return {
            **result,
            "model_artifact_sha256": None,
            "registered_authority_sha256": os.environ.get(
                AUTHORITY_SHA_ENV[market_type]
            ),
        }
    path = AUTHORITY_FILES[market_type]
    expected_sha256 = os.environ.get(AUTHORITY_SHA_ENV[market_type])
    if not path.exists():
        result = unavailable_authority(
            "market_authority_receipt_missing",
            *(
                ("independent_market_authority_not_registered",)
                if not expected_sha256
                else ()
            ),
        )
    else:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = unavailable_authority("market_authority_receipt_invalid")
        else:
            result = validate_authority_receipt(
                receipt,
                expected_sha256=expected_sha256,
                league=league,
                market_type=market_type,
                model_artifact_sha256=artifact_sha256,
                as_of=as_of,
            )
    return {
        **result,
        "model_artifact_sha256": artifact_sha256,
        "registered_authority_sha256": expected_sha256,
    }


def _registered_market_quote(
    *,
    event_id: str,
    market_type: str,
    settlement_rule_id: str,
    as_of: datetime,
) -> dict[str, Any]:
    if not event_id:
        return {
            "status": "unavailable",
            "quote": None,
            "quote_sha256": None,
            "blockers": ["market_event_id_missing"],
        }
    try:
        registered = load_registered_quote(
            registry_locator=QUOTE_REGISTRY_LOCATOR,
            expected_registry_sha256=os.environ.get(QUOTE_REGISTRY_SHA_ENV),
            event_id=event_id,
            market_type=market_type,
            settlement_rule_id=settlement_rule_id,
            as_of=as_of,
            root=ROOT,
        )
    except RegisteredQuoteUnavailable as exc:
        return {
            "status": "unavailable",
            "quote": None,
            "quote_sha256": None,
            "blockers": [exc.code],
        }
    except QuoteCaptureError:
        return {
            "status": "unavailable",
            "quote": None,
            "quote_sha256": None,
            "blockers": ["registered_market_quote_invalid"],
        }
    return {**registered, "blockers": []}


def _registered_event_probability(
    *,
    event_id: str,
    league: str,
    market_type: str,
    selection: str,
    opposing_selection: str,
    authority: dict[str, Any],
    as_of: datetime,
) -> dict[str, Any]:
    if (
        authority.get("status") != "approved"
        or authority.get("betting_decision_authorized") is not True
    ):
        return {
            "status": "unavailable",
            "receipt": None,
            "receipt_sha256": None,
            "registry": None,
            "registry_sha256": None,
            "blockers": ["market_authority_unavailable_for_probability_load"],
        }
    expected_registry_sha256 = os.environ.get(
        MATCH_WINNER_PROBABILITY_REGISTRY_SHA_ENV
    )
    if not expected_registry_sha256:
        return {
            "status": "unavailable",
            "receipt": None,
            "receipt_sha256": None,
            "registry": None,
            "registry_sha256": None,
            "blockers": ["event_probability_registry_not_registered"],
        }
    if expected_registry_sha256 != authority.get(
        "event_probability_registry_sha256"
    ):
        return {
            "status": "unavailable",
            "receipt": None,
            "receipt_sha256": None,
            "registry": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": ["event_probability_registry_authority_binding_mismatch"],
        }
    try:
        loaded = load_registered_event_probability(
            registry_locator=MATCH_WINNER_PROBABILITY_REGISTRY.as_posix(),
            expected_registry_sha256=expected_registry_sha256,
            event_id=event_id,
            league=league,
            market_type=market_type,
            selection=selection,
            opposing_selection=opposing_selection,
            model_artifact_sha256=authority["model_artifact_sha256"],
            market_protocol_artifact_sha256=authority[
                "market_protocol_artifact_sha256"
            ],
            calibration_artifact_sha256=authority[
                "calibration_artifact_sha256"
            ],
            uncertainty_artifact_sha256=authority[
                "uncertainty_artifact_sha256"
            ],
            generation_code_sha256=authority[
                "event_probability_generation_code_sha256"
            ],
            as_of=as_of,
            root=ROOT,
        )
    except (EventProbabilityError, OSError, KeyError, ValueError):
        return {
            "status": "unavailable",
            "receipt": None,
            "receipt_sha256": None,
            "registry": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": ["registered_event_probability_invalid"],
        }
    return {**loaded, "blockers": []}


def _registered_pregame_roster(
    *,
    event_id: str,
    event_start: str,
    league: str,
    blue_team: str,
    red_team: str,
    as_of: datetime,
) -> dict[str, Any]:
    blockers = []
    if not event_id:
        blockers.append("event_id_missing")
    if not event_start:
        blockers.append("event_start_missing")
    expected_registry_sha256 = os.environ.get(ROSTER_REGISTRY_SHA_ENV)
    if not expected_registry_sha256:
        blockers.append("roster_registry_not_registered")
    if blockers:
        return {
            "status": "unavailable",
            "roster": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": sorted(set(blockers)),
        }
    try:
        registered = load_registered_pregame_roster(
            registry_locator=ROSTER_REGISTRY_LOCATOR,
            expected_registry_sha256=expected_registry_sha256,
            event_id=event_id,
            event_start=event_start,
            league=league,
            blue_organization_name=blue_team,
            red_organization_name=red_team,
            as_of=as_of,
            root=ROOT,
        )
    except RegisteredPregameRosterUnavailable as exc:
        return {
            "status": "unavailable",
            "roster": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": [exc.code],
        }
    except (OSError, PregameRosterError):
        return {
            "status": "unavailable",
            "roster": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": ["registered_pregame_roster_invalid"],
        }
    return {**registered, "blockers": []}


def _registered_event_rating(
    *,
    event_id: str,
    event_start: str,
    league: str,
    blue_team: str,
    red_team: str,
    roster_registration: dict[str, Any],
    as_of: datetime,
) -> dict[str, Any]:
    blockers = []
    if roster_registration.get("status") != "registered":
        blockers.append("pre_event_roster_not_registered")
    if not event_id:
        blockers.append("event_id_missing")
    if not event_start:
        blockers.append("event_start_missing")
    expected_registry_sha256 = os.environ.get(RATING_REGISTRY_SHA_ENV)
    if not expected_registry_sha256:
        blockers.append("rating_registry_not_registered")
    if blockers:
        return {
            "status": "unavailable",
            "player_rating_authorized": False,
            "team_rating_authorized": False,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "ratings": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": sorted(set(blockers)),
        }
    try:
        registered = load_registered_event_rating(
            registry_locator=RATING_REGISTRY_LOCATOR,
            expected_registry_sha256=expected_registry_sha256,
            registered_roster=roster_registration,
            event_id=event_id,
            event_start=event_start,
            league=league,
            blue_organization_name=blue_team,
            red_organization_name=red_team,
            as_of=as_of,
            root=ROOT,
        )
    except RegisteredEventRatingUnavailable as exc:
        return {
            "status": "unavailable",
            "player_rating_authorized": False,
            "team_rating_authorized": False,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "ratings": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": [exc.code],
        }
    except (OSError, RatingAuthorityError):
        return {
            "status": "unavailable",
            "player_rating_authorized": False,
            "team_rating_authorized": False,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "ratings": None,
            "receipt_sha256": None,
            "registry_sha256": expected_registry_sha256,
            "blockers": ["registered_event_rating_invalid"],
        }
    return registered


@lru_cache(maxsize=1)
def _context() -> dict[str, Any]:
    return json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _draft_model() -> dict[str, Any]:
    return json.loads(DRAFT_MODEL_PATH.read_text(encoding="utf-8"))["model"]


@lru_cache(maxsize=1)
def _live_coefficients() -> dict[str, Any]:
    return json.loads(LIVE_COEFS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _live_totals_artifact() -> dict[str, Any]:
    return validate_development_candidate(ROOT)


def _resolve_team(requested: str) -> dict[str, Any]:
    query = requested.strip().lower()
    if not query:
        raise ModelInputError("Both teams are required")
    teams = _context().get("teams") or []
    exact = [
        team
        for team in teams
        if str(team.get("team") or "").strip().lower() == query
    ]
    if exact:
        return exact[0]
    prefix = [
        team
        for team in teams
        if str(team.get("team") or "").strip().lower().startswith(query)
    ]
    if len(prefix) == 1:
        return prefix[0]
    raise ModelInputError(f"Team is not uniquely identified: {requested}")


def options() -> dict[str, Any]:
    context = _context()
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    champions = sorted((runtime.get("champ_game_counts") or {}).keys())
    teams = sorted(
        [
            {
                "name": str(team.get("team")),
                "league": str(team.get("league") or ""),
            }
            for team in context.get("teams") or []
            if team.get("team")
        ],
        key=lambda row: row["name"].lower(),
    )
    return {
        "teams": teams,
        "champions": champions,
        "leagues": list(LEAGUES),
        "roles": list(ROLES),
        "model_as_of": context.get("as_of"),
    }


@lru_cache(maxsize=1)
def private_readiness() -> dict[str, Any]:
    """Run the expensive full blocker audit once per local server process."""
    return inspect_private_decision_readiness(ROOT)


def _parse_picks(payload: dict[str, Any], side: str) -> list[str]:
    picks = payload.get(f"{side}_picks")
    if not isinstance(picks, list) or len(picks) != 5:
        raise ModelInputError(f"{side.title()} draft requires five champions")
    parsed = [str(champion or "").strip() for champion in picks]
    if any(not champion for champion in parsed):
        raise ModelInputError(f"{side.title()} draft has an empty role")
    return parsed


def _pregame_kills_prior(
    league: str,
    blue_team: str,
    red_team: str,
    champions: list[str],
) -> dict[str, Any]:
    oe = load_oe_pace_prior(league, blue_team, red_team)
    draft = predict_total(_draft_model(), champions, league=league)
    totals = np.asarray(oe.get("totals") or [], dtype=float)
    oe_sd = float(np.std(totals, ddof=1)) if len(totals) >= 2 else 8.0
    oe_mean = float(oe["target_mu"])
    draft_mean = float(draft["expected_total"])
    draft_sd = max(float(draft["sd"]), 1.0)
    coverage = float(draft["n_champs_applied"]) / 10.0
    weight_oe = 1.0 / max(oe_sd**2, 1.0)
    weight_draft = coverage / max(draft_sd**2, 1.0)
    denominator = weight_oe + weight_draft
    mean = (
        (weight_oe * oe_mean + weight_draft * draft_mean) / denominator
        if denominator > 0
        else oe_mean
    )
    estimate_variance = 1.0 / denominator if denominator > 0 else oe_sd**2
    predictive_sd = math.sqrt(estimate_variance + 0.5 * oe_sd**2)
    return {
        "mean": mean,
        "sd": predictive_sd,
        "oe_mean": oe_mean,
        "oe_sd": oe_sd,
        "oe_source": oe.get("source"),
        "oe_n": int(oe.get("n") or 0),
        "mean_length": float(oe["mean_length"]),
        "draft_mean": draft_mean,
        "draft_delta": float(draft["delta_vs_baseline"]),
        "draft_coverage": coverage,
        "unknown_champions": list(draft["unknown_champions"]),
    }


def _validated_live_totals(
    *,
    minute: float,
    current_kills: int,
    gold_difference: float | None,
    league: str,
    patch: str | None,
    as_of: datetime,
    blue_team: str,
    red_team: str,
    champions: list[str],
    lines: list[dict[str, Any]],
    event_id: str = "",
) -> dict[str, Any]:
    priced = price_live_totals(
        _live_totals_artifact(),
        league=league,
        blue_team=blue_team,
        red_team=red_team,
        champions=champions,
        minute=minute,
        current_kills=current_kills,
        gold_difference=gold_difference,
        patch=patch,
        as_of=as_of,
        lines=lines,
    )
    model_supported = priced["eligibility"]["status"] == "supported"
    authority = _private_market_authority(
        league=league,
        market_type="total_kills",
        as_of=as_of,
    )
    quote_registration = _registered_market_quote(
        event_id=event_id,
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        as_of=as_of,
    )
    registered_quote = quote_registration.get("quote")
    registered_quote_sha256 = quote_registration.get("quote_sha256")
    line_views = []
    for source, market in zip(priced["lines"], lines):
        under_probability = source["under_probability"]
        over_probability = source["over_probability"]
        under_interval = source.get("under_probability_interval")
        over_interval = source.get("over_probability_interval")
        line = float(source["line"])
        under_selection = f"under:{line:g}"
        over_selection = f"over:{line:g}"
        line_views.append(
            {
                "line": line,
                "under": _market_view(
                    under_probability,
                    market.get("under_odds"),
                    market.get("over_odds"),
                    probability_interval=under_interval,
                    quote=registered_quote,
                    expected_quote_sha256=registered_quote_sha256,
                    quote_registry_sha256=quote_registration.get("registry_sha256"),
                    authority=authority,
                    expected_authority_sha256=authority.get(
                        "registered_authority_sha256"
                    ),
                    as_of=as_of,
                    selection=under_selection,
                    opposing_selection=over_selection,
                    event_id=event_id,
                    market_type="total_kills",
                    settlement_rule_id="map-total-kills-v1",
                ),
                "over": _market_view(
                    over_probability,
                    market.get("over_odds"),
                    market.get("under_odds"),
                    probability_interval=over_interval,
                    quote=registered_quote,
                    expected_quote_sha256=registered_quote_sha256,
                    quote_registry_sha256=quote_registration.get("registry_sha256"),
                    authority=authority,
                    expected_authority_sha256=authority.get(
                        "registered_authority_sha256"
                    ),
                    as_of=as_of,
                    selection=over_selection,
                    opposing_selection=under_selection,
                    event_id=event_id,
                    market_type="total_kills",
                    settlement_rule_id="map-total-kills-v1",
                ),
            }
        )
    blockers = list(priced["eligibility"]["blockers"])
    uncertainty = priced.get("uncertainty") or {
        "status": "unavailable",
        "method": None,
        "confidence": None,
        "blockers": ["dependence_aware_uncertainty_unavailable"],
    }
    uncertainty_blockers = list(uncertainty.get("blockers") or [])
    authority_blockers = list(authority.get("blockers") or [])
    quote_blockers = list(quote_registration.get("blockers") or [])
    decision_available = any(
        view[side]["status"] == "authorized"
        for view in line_views
        for side in ("under", "over")
    )
    return {
        "checkpoint": int(minute) if float(minute).is_integer() else None,
        "projected_mean": priced["projected_mean"],
        "projected_sd": None,
        "effective_n": next(
            (
                source.get("uncertainty", {}).get("effective_series_n")
                for source in priced["lines"]
                if source.get("uncertainty", {}).get("effective_series_n")
                is not None
            ),
            None,
        ),
        "model_supported": model_supported,
        "classification_available": decision_available,
        "eligibility": priced["eligibility"],
        "uncertainty": uncertainty,
        "decision_authority": authority,
        "quote_registration": {
            key: value
            for key, value in quote_registration.items()
            if key != "quote"
        },
        "lines": line_views,
        "warnings": [
            *(
                []
                if model_supported
                else [
                    "Total-kills probability withheld: " + ", ".join(blockers),
                    "Only exact league/checkpoint windows with fresh same-patch evidence are eligible.",
                ]
            ),
            *(
                []
                if uncertainty.get("status") == "available"
                else [
                    "Dependence-aware total-kills interval withheld: "
                    + ", ".join(
                        uncertainty_blockers
                        or ["dependence_aware_uncertainty_unavailable"]
                    )
                ]
            ),
            *(
                [
                    "No total-kills betting decision is authorized: "
                    + ", ".join(authority_blockers),
                    "A dependence-valid probability interval and a fresh, provenance-bound two-sided quote are also required.",
                ]
                if authority.get("status") != "approved"
                else []
            ),
            *(
                [
                    "No independently registered total-kills quote is available: "
                    + ", ".join(quote_blockers)
                ]
                if quote_blockers
                else []
            ),
        ],
    }


def _market_view(
    probability: float | None,
    offered_value: Any,
    opposing_value: Any,
    *,
    probability_interval: tuple[float, float] | list[float] | None = None,
    quote: dict[str, Any] | None = None,
    expected_quote_sha256: str | None = None,
    quote_registry_sha256: str | None = None,
    probability_receipt: dict[str, Any] | None = None,
    expected_probability_sha256: str | None = None,
    probability_registry: dict[str, Any] | None = None,
    expected_probability_registry_sha256: str | None = None,
    authority: dict[str, Any] | None = None,
    expected_authority_sha256: str | None = None,
    as_of: datetime | None = None,
    selection: str = "selection",
    opposing_selection: str = "opposing",
    event_id: str = "",
    market_type: str = "unregistered",
    settlement_rule_id: str = "",
) -> dict[str, Any]:
    offered = (
        None
        if offered_value in (None, "")
        else _number(offered_value, "Offered odds", minimum=1.001, maximum=100)
    )
    opposing = (
        None
        if opposing_value in (None, "")
        else _number(opposing_value, "Opposing odds", minimum=1.001, maximum=100)
    )
    evaluated = evaluate_two_way_market(
        model_probability=probability,
        probability_interval=probability_interval,
        probability_receipt=probability_receipt,
        expected_probability_sha256=expected_probability_sha256,
        probability_registry=probability_registry,
        expected_probability_registry_sha256=(
            expected_probability_registry_sha256
        ),
        offered_odds=offered,
        opposing_odds=opposing,
        quote=quote,
        expected_quote_sha256=expected_quote_sha256,
        quote_registry_sha256=quote_registry_sha256,
        authority=authority or unavailable_authority(),
        expected_authority_sha256=expected_authority_sha256,
        as_of=as_of or datetime.now(timezone.utc),
        selection=selection,
        opposing_selection=opposing_selection,
        event_id=event_id,
        market_type=market_type,
        settlement_rule_id=settlement_rule_id,
    )
    market = evaluated["market"]
    diagnostic = evaluated["diagnostic"]
    expected_return = evaluated["expected_return"]
    conservative_return = evaluated["conservative_expected_return"]
    return {
        "status": evaluated["status"],
        "decision": evaluated["decision"],
        "blockers": evaluated["blockers"],
        "probability": _round(evaluated["authorized_probability"]),
        "diagnostic_probability": _round(diagnostic["model_probability"]),
        "probability_interval": diagnostic["probability_interval"],
        "claim_ceiling": diagnostic["claim_ceiling"],
        "fair_odds": _round(evaluated["fair_odds"], 3),
        "offered_odds": _round(market["offered_odds"], 3),
        "opposing_odds": _round(market["opposing_odds"], 3),
        "break_even_probability": _round(market["raw_break_even_probability"]),
        "no_vig_break_even_probability": _round(
            market["no_vig_break_even_probability"]
        ),
        "overround": _round(market["overround"]),
        "edge_pp": _round(evaluated["edge_pp"], 2),
        "expected_return_pct": (
            _round(100 * expected_return, 2) if expected_return is not None else None
        ),
        "conservative_expected_return_pct": (
            _round(100 * conservative_return, 2)
            if conservative_return is not None
            else None
        ),
    }


def _pregame_winner(
    *,
    league: str,
    blue_team: str,
    red_team: str,
    blue_picks: list[str],
    red_picks: list[str],
    event_id: str,
    event_start: str,
    draft_source_available_at: str,
) -> dict[str, Any]:
    if not TSX_PATH.exists():
        return {
            "available": False,
            "unavailable_reason": f"Local TypeScript runtime is unavailable: {TSX_PATH}",
        }
    request = {
        "league": league,
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_picks": blue_picks,
        "red_picks": red_picks,
        "event_id": event_id or None,
        "event_start": event_start or None,
        "draft_source_available_at": draft_source_available_at or None,
    }
    try:
        completed = subprocess.run(
            [str(TSX_PATH), str(PREGAME_BRIDGE_PATH)],
            cwd=APP_PATH,
            input=json.dumps(request),
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        result = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "unavailable_reason": f"Pregame winner model unavailable: {exc}",
        }
    if not isinstance(result, dict):
        return {
            "available": False,
            "unavailable_reason": "Pregame winner model returned an unsupported result",
        }
    return result


def _objective_rows(kind: str, count: int) -> list[dict[str, Any]]:
    return [{"type": kind, "completionCount": count}] if count else []


def _personal_live_win(
    *,
    minute: float,
    blue_team: dict[str, Any],
    red_team: dict[str, Any],
    blue_picks: list[str],
    red_picks: list[str],
    state: dict[str, int | float | None],
    registered_rating_difference: float | None,
) -> dict[str, Any]:
    if registered_rating_difference is None:
        return {
            "available": False,
            "p_blue": None,
            "p_red": None,
            "diagnostic_p_blue": None,
            "diagnostic_p_red": None,
            "fair_odds": None,
            "claim_ceiling": "unavailable",
            "component_authority": {
                "live_state": "development_only",
                "team_rating": "unavailable",
                "player_rating": "unavailable",
                "probability_interval": "unavailable",
            },
            "decision_blockers": [
                "independently_registered_event_rating_unavailable",
                "dependence_valid_probability_interval_missing",
            ],
            "warnings": [
                "Live winner diagnostic is withheld because an exact-roster event rating is not independently registered."
            ],
            "personal_soft_extension": None,
        }
    blue_id = "manual-blue"
    red_id = "manual-red"
    objectives_blue = [
        *_objective_rows("dragon", int(state["blue_dragons"])),
        *_objective_rows("herald", int(state["blue_heralds"])),
        *_objective_rows("tower", int(state["blue_towers"])),
    ]
    objectives_red = [
        *_objective_rows("dragon", int(state["red_dragons"])),
        *_objective_rows("herald", int(state["red_heralds"])),
        *_objective_rows("tower", int(state["red_towers"])),
    ]
    actions = [
        *[
            {
                "type": "pick",
                "drafter": {"id": blue_id},
                "draftable": {"name": champion},
            }
            for champion in blue_picks
        ],
        *[
            {
                "type": "pick",
                "drafter": {"id": red_id},
                "draftable": {"name": champion},
            }
            for champion in red_picks
        ],
    ]
    series_state = {
        "games": [
            {
                "started": True,
                "finished": False,
                "clock": {"currentSeconds": minute * 60},
                "teams": [
                    {
                        "id": blue_id,
                        "name": blue_team["team"],
                        "side": "blue",
                        "netWorth": state["blue_gold"],
                        "kills": state["blue_kills"],
                        "objectives": objectives_blue,
                        "players": [],
                    },
                    {
                        "id": red_id,
                        "name": red_team["team"],
                        "side": "red",
                        "netWorth": state["red_gold"],
                        "kills": state["red_kills"],
                        "objectives": objectives_red,
                        "players": [],
                    },
                ],
                "draftActions": actions,
            }
        ]
    }
    evaluation = evaluate_live_state(
        series_state, elo_diff=registered_rating_difference
    )
    base_probability = evaluation.p_blue
    if base_probability is None:
        return {
            **evaluation.as_dict(),
            "fair_odds": None,
            "personal_soft_extension": None,
        }

    coefficients = _live_coefficients()
    priors = coefficients.get("live_obj_priors") or {}
    kill_diff = int(state["blue_kills"]) - int(state["red_kills"])
    dragon_diff = int(state["blue_dragons"]) - int(state["red_dragons"])
    tower_diff = int(state["blue_towers"]) - int(state["red_towers"])
    grub_diff = int(state["blue_grubs"]) - int(state["red_grubs"])
    extra_dragons = dragon_diff - (1 if dragon_diff > 0 else -1 if dragon_diff < 0 else 0)
    extra_towers = tower_diff - (1 if tower_diff > 0 else -1 if tower_diff < 0 else 0)
    raw_adjustment = (
        float(priors.get("kill_diff") or 0) * kill_diff
        + float(priors.get("dragon_diff") or 0) * extra_dragons
        + float(priors.get("tower_diff") or 0) * extra_towers
        + float(priors.get("void_grub") or 0) * grub_diff
    )
    cap = float(coefficients.get("adv_cap") or 1.45)
    adjustment = cap * math.tanh(raw_adjustment / cap)
    probability = _sigmoid(_logit(base_probability) + adjustment)
    return {
        **evaluation.as_dict(),
        "p_blue": None,
        "p_red": None,
        "diagnostic_p_blue": _round(probability),
        "diagnostic_p_red": _round(1 - probability),
        "fair_odds": None,
        "claim_ceiling": "research_diagnostic_only",
        "component_authority": {
            "live_state": "development_only",
            "team_rating": "registered_component",
            "player_rating": "registered_component",
            "probability_interval": "unavailable",
        },
        "decision_blockers": [
            "live_state_model_independent_authority_unavailable",
            "dependence_valid_probability_interval_missing",
        ],
        "rating_input": {
            "status": "registered_component",
            "blue_minus_red": _round(registered_rating_difference, 3),
        },
        "personal_soft_extension": {
            "logit_adjustment": _round(adjustment),
            "kill_diff": kill_diff,
            "extra_dragon_diff": extra_dragons,
            "extra_tower_diff": extra_towers,
            "grub_diff": grub_diff,
            "note": (
                "Personal soft-prior extension for current counts. Barons and inhibitors "
                "are recorded but receive no invented coefficient."
            ),
        },
    }


def score_manual_state(payload: dict[str, Any]) -> dict[str, Any]:
    league = str(payload.get("league") or "").strip().upper()
    if league not in LEAGUES:
        raise ModelInputError(f"Unsupported league: {league or 'missing'}")
    blue_team = _resolve_team(str(payload.get("blue_team") or ""))
    red_team = _resolve_team(str(payload.get("red_team") or ""))
    if blue_team["team"] == red_team["team"]:
        raise ModelInputError("Blue and red teams must be different")
    blue_picks = _parse_picks(payload, "blue")
    red_picks = _parse_picks(payload, "red")
    minute = _number(payload.get("minute"), "Game minute", minimum=1, maximum=90)
    patch = str(payload.get("patch") or "").strip() or None
    as_of_value = payload.get("as_of")
    if as_of_value:
        try:
            as_of = datetime.fromisoformat(str(as_of_value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ModelInputError("As-of time must be ISO-8601") from exc
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
    else:
        as_of = datetime.now(timezone.utc)
    event_id = str(payload.get("event_id") or "").strip()
    event_start = str(payload.get("event_start") or "").strip()
    draft_source_available_at = str(
        payload.get("draft_source_available_at") or ""
    ).strip()

    state: dict[str, int | float | None] = {
        "blue_kills": _integer(payload.get("blue_kills"), "Blue kills"),
        "red_kills": _integer(payload.get("red_kills"), "Red kills"),
        "blue_gold": _optional_number(payload.get("blue_gold"), "Blue gold", minimum=1),
        "red_gold": _optional_number(payload.get("red_gold"), "Red gold", minimum=1),
        "blue_dragons": _integer(payload.get("blue_dragons", 0), "Blue dragons", maximum=7),
        "red_dragons": _integer(payload.get("red_dragons", 0), "Red dragons", maximum=7),
        "blue_towers": _integer(payload.get("blue_towers", 0), "Blue towers", maximum=11),
        "red_towers": _integer(payload.get("red_towers", 0), "Red towers", maximum=11),
        "blue_grubs": _integer(payload.get("blue_grubs", 0), "Blue grubs", maximum=6),
        "red_grubs": _integer(payload.get("red_grubs", 0), "Red grubs", maximum=6),
        "blue_heralds": _integer(payload.get("blue_heralds", 0), "Blue heralds", maximum=2),
        "red_heralds": _integer(payload.get("red_heralds", 0), "Red heralds", maximum=2),
        "blue_barons": _integer(payload.get("blue_barons", 0), "Blue barons", maximum=6),
        "red_barons": _integer(payload.get("red_barons", 0), "Red barons", maximum=6),
        "blue_inhibitors": _integer(payload.get("blue_inhibitors", 0), "Blue inhibitors", maximum=6),
        "red_inhibitors": _integer(payload.get("red_inhibitors", 0), "Red inhibitors", maximum=6),
    }
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ModelInputError("Add at least one total-kills line")
    if len(lines) > 20:
        raise ModelInputError("Use at most 20 total-kills lines")

    champions = blue_picks + red_picks
    pregame = _pregame_kills_prior(
        league,
        str(blue_team["team"]),
        str(red_team["team"]),
        champions,
    )
    blue_gold = state["blue_gold"]
    red_gold = state["red_gold"]
    gold_difference = (
        float(blue_gold) - float(red_gold)
        if blue_gold is not None and red_gold is not None
        else None
    )
    live_totals = _validated_live_totals(
        minute=minute,
        current_kills=int(state["blue_kills"]) + int(state["red_kills"]),
        gold_difference=gold_difference,
        league=league,
        patch=patch,
        as_of=as_of,
        blue_team=str(blue_team["team"]),
        red_team=str(red_team["team"]),
        champions=champions,
        lines=lines,
        event_id=event_id,
    )
    roster_registration = _registered_pregame_roster(
        event_id=event_id,
        event_start=event_start,
        league=league,
        blue_team=str(blue_team["team"]),
        red_team=str(red_team["team"]),
        as_of=as_of,
    )
    roster_registered = roster_registration.get("status") == "registered"
    rating_registration = _registered_event_rating(
        event_id=event_id,
        event_start=event_start,
        league=league,
        blue_team=str(blue_team["team"]),
        red_team=str(red_team["team"]),
        roster_registration=roster_registration,
        as_of=as_of,
    )
    rating_registered = (
        rating_registration.get("status") == "registered"
        and rating_registration.get("player_rating_authorized") is True
        and rating_registration.get("team_rating_authorized") is True
    )
    rating_strength = (
        (rating_registration.get("ratings") or {}).get("strength_difference")
        or {}
    )
    registered_rating_difference = (
        float(rating_strength["posterior_mean"])
        if rating_registered
        and isinstance(rating_strength.get("posterior_mean"), (int, float))
        and not isinstance(rating_strength.get("posterior_mean"), bool)
        and math.isfinite(float(rating_strength["posterior_mean"]))
        else None
    )
    if rating_registered and registered_rating_difference is None:
        rating_registration = {
            **rating_registration,
            "status": "unavailable",
            "player_rating_authorized": False,
            "team_rating_authorized": False,
            "ratings": None,
            "blockers": sorted(
                set(
                    list(rating_registration.get("blockers") or [])
                    + ["registered_rating_strength_difference_invalid"]
                )
            ),
        }
        rating_registered = False
    live_win = _personal_live_win(
        minute=minute,
        blue_team=blue_team,
        red_team=red_team,
        blue_picks=blue_picks,
        red_picks=red_picks,
        state=state,
        registered_rating_difference=registered_rating_difference,
    )
    pregame_win = _pregame_winner(
        league=league,
        blue_team=str(blue_team["team"]),
        red_team=str(red_team["team"]),
        blue_picks=blue_picks,
        red_picks=red_picks,
        event_id=event_id,
        event_start=event_start,
        draft_source_available_at=draft_source_available_at,
    )
    strength_expectation = dict(pregame_win.get("strength_expectation") or {})
    strength_blockers = [
        blocker
        for blocker in strength_expectation.get("blockers", [])
        if not (
            roster_registered
            and blocker == "pre_event_roster_authority_unavailable"
        )
        and not (
            rating_registered
            and blocker
            in {
                "independently_validated_team_rating_unavailable",
                "independently_validated_player_rating_unavailable",
            }
        )
    ]
    if not roster_registered and "pre_event_roster_authority_unavailable" not in strength_blockers:
        strength_blockers.append("pre_event_roster_authority_unavailable")
    strength_blockers.extend(rating_registration.get("blockers") or [])
    pregame_win = {
        **pregame_win,
        "strength_expectation": {
            **strength_expectation,
            "status": (
                "registered_rating_components_probability_unavailable"
                if rating_registered
                else "roster_registered_ratings_unavailable"
                if roster_registered
                else strength_expectation.get("status", "unavailable")
            ),
            "team_rating_authorized": rating_registered,
            "player_rating_authorized": rating_registered,
            "pre_event_roster_authorized": roster_registered,
            "match_probability_authorized": False,
            "strength_difference": (
                rating_strength if rating_registered else None
            ),
            "blockers": sorted(set(strength_blockers)),
        },
        "roster_registration": {
            key: value
            for key, value in roster_registration.items()
            if key != "roster"
        },
        "rating_registration": {
            key: value
            for key, value in rating_registration.items()
            if key != "ratings"
        },
    }
    live_available = live_win.get("p_blue") is not None
    pregame_available = (
        bool(pregame_win.get("available"))
        and pregame_win.get("p_blue") is not None
    )
    winner_mode = (
        "live"
        if live_available
        else "pregame"
        if pregame_available
        else "unavailable"
    )
    winner_blue_probability = (
        live_win.get("p_blue")
        if live_available
        else pregame_win.get("p_blue")
        if pregame_available
        else None
    )
    winner_red_probability = (
        live_win.get("p_red")
        if live_available
        else pregame_win.get("p_red")
        if pregame_available
        else None
    )
    blue_selection = f"winner:{blue_team['team']}"
    red_selection = f"winner:{red_team['team']}"
    winner_authority = _private_market_authority(
        league=league,
        market_type="match_winner",
        as_of=as_of,
    )
    winner_quote_registration = _registered_market_quote(
        event_id=event_id,
        market_type="match_winner",
        settlement_rule_id=MATCH_WINNER_SETTLEMENT_RULE_ID,
        as_of=as_of,
    )
    winner_quote = winner_quote_registration.get("quote")
    winner_quote_sha256 = winner_quote_registration.get("quote_sha256")
    blue_probability_registration = _registered_event_probability(
        event_id=event_id,
        league=league,
        market_type="match_winner",
        selection=blue_selection,
        opposing_selection=red_selection,
        authority=winner_authority,
        as_of=as_of,
    )
    red_probability_registration = _registered_event_probability(
        event_id=event_id,
        league=league,
        market_type="match_winner",
        selection=red_selection,
        opposing_selection=blue_selection,
        authority=winner_authority,
        as_of=as_of,
    )
    winner_blue_view = _market_view(
        winner_blue_probability,
        payload.get("blue_win_odds"),
        payload.get("red_win_odds"),
        probability_interval=None,
        probability_receipt=blue_probability_registration.get("receipt"),
        expected_probability_sha256=blue_probability_registration.get(
            "receipt_sha256"
        ),
        probability_registry=blue_probability_registration.get("registry"),
        expected_probability_registry_sha256=blue_probability_registration.get(
            "registry_sha256"
        ),
        quote=winner_quote,
        expected_quote_sha256=winner_quote_sha256,
        quote_registry_sha256=winner_quote_registration.get("registry_sha256"),
        authority=winner_authority,
        expected_authority_sha256=winner_authority.get(
            "registered_authority_sha256"
        ),
        as_of=as_of,
        selection=blue_selection,
        opposing_selection=red_selection,
        event_id=event_id,
        market_type="match_winner",
        settlement_rule_id=MATCH_WINNER_SETTLEMENT_RULE_ID,
    )
    winner_red_view = _market_view(
        winner_red_probability,
        payload.get("red_win_odds"),
        payload.get("blue_win_odds"),
        probability_interval=None,
        probability_receipt=red_probability_registration.get("receipt"),
        expected_probability_sha256=red_probability_registration.get(
            "receipt_sha256"
        ),
        probability_registry=red_probability_registration.get("registry"),
        expected_probability_registry_sha256=red_probability_registration.get(
            "registry_sha256"
        ),
        quote=winner_quote,
        expected_quote_sha256=winner_quote_sha256,
        quote_registry_sha256=winner_quote_registration.get("registry_sha256"),
        authority=winner_authority,
        expected_authority_sha256=winner_authority.get(
            "registered_authority_sha256"
        ),
        as_of=as_of,
        selection=red_selection,
        opposing_selection=blue_selection,
        event_id=event_id,
        market_type="match_winner",
        settlement_rule_id=MATCH_WINNER_SETTLEMENT_RULE_ID,
    )
    decision_blockers = sorted(
        set(winner_blue_view["blockers"] + winner_red_view["blockers"])
    )
    draft_diagnostic = pregame_win.get("draft_score") or {}
    component_authority = {
        "draft_score": {
            "status": draft_diagnostic.get("status", "unavailable"),
            "authorized": False,
            "model_kind": draft_diagnostic.get("model_kind"),
            "blockers": ["independent_l2_authority_unavailable"],
        },
        "team_rating": {
            "status": rating_registration.get("status", "unavailable"),
            "authorized": rating_registered,
            "receipt_sha256": rating_registration.get("receipt_sha256"),
            "registry_sha256": rating_registration.get("registry_sha256"),
            "blockers": (
                []
                if rating_registered
                else list(rating_registration.get("blockers") or [])
            ),
        },
        "player_rating": {
            "status": rating_registration.get("status", "unavailable"),
            "authorized": rating_registered,
            "receipt_sha256": rating_registration.get("receipt_sha256"),
            "registry_sha256": rating_registration.get("registry_sha256"),
            "blockers": (
                []
                if rating_registered
                else list(rating_registration.get("blockers") or [])
            ),
        },
        "rating_to_probability": {
            "status": "unavailable",
            "authorized": False,
            "blockers": (
                [
                    "rating_to_match_probability_calibration_unavailable",
                    "draft_rating_combination_authority_unavailable",
                ]
                if rating_registered
                else ["registered_rating_components_unavailable"]
            ),
        },
        "pre_event_roster": {
            "status": roster_registration.get("status", "unavailable"),
            "authorized": roster_registered,
            "receipt_sha256": roster_registration.get("receipt_sha256"),
            "registry_sha256": roster_registration.get("registry_sha256"),
            "blockers": list(roster_registration.get("blockers") or []),
        },
    }
    winner_reprice = {
        "mode": winner_mode,
        "source": (
            "Live research diagnostic"
            if winner_mode == "live"
            else "Draft diagnostic only; match probability unavailable"
            if draft_diagnostic
            else "Unavailable"
        ),
        "p_blue": winner_blue_view["probability"],
        "p_red": winner_red_view["probability"],
        "diagnostic_p_blue": _round(winner_blue_probability),
        "diagnostic_p_red": _round(winner_red_probability),
        "decision_authority": winner_authority,
        "quote_registration": {
            key: value
            for key, value in winner_quote_registration.items()
            if key != "quote"
        },
        "component_authority": component_authority,
        "blue": winner_blue_view,
        "red": winner_red_view,
        "warnings": [
            *(
                [
                    "Live update is unavailable; the terminal Draft Score remains a separate development diagnostic and is not a match-win probability."
                ]
                if winner_mode != "live" and draft_diagnostic
                else []
            ),
            *(
                [
                    "No independently registered winner quote is available: "
                    + ", ".join(winner_quote_registration.get("blockers") or [])
                ]
                if winner_quote_registration.get("blockers")
                else []
            ),
            *(
                [
                    "No independently registered exact pre-event roster is available: "
                    + ", ".join(roster_registration.get("blockers") or [])
                ]
                if roster_registration.get("blockers")
                else []
            ),
            *(
                [
                    "No independently registered event rating is available: "
                    + ", ".join(rating_registration.get("blockers") or [])
                ]
                if not rating_registered
                and rating_registration.get("blockers")
                else []
            ),
            *(
                [str(pregame_win.get("unavailable_reason"))]
                if winner_mode == "unavailable"
                and pregame_win.get("unavailable_reason")
                else []
            ),
            *(
                [
                    "No winner betting decision is authorized: "
                    + ", ".join(decision_blockers)
                ]
                if decision_blockers
                else []
            ),
            *(
                [
                    "Draft Score is the canonical terminal neutral development replay; independent L2 promotion is absent.",
                    (
                        "Registered player/team ratings remain component-only; rating-to-probability and Draft combination authority are absent."
                        if rating_registered
                        else "No team/player strength was blended: exact pre-event roster and independently validated rating authority are unavailable."
                    ),
                ]
                if draft_diagnostic
                else []
            ),
        ],
    }
    return {
        "status": "ok",
        "scope": "personal-local",
        "teams": {
            "blue": str(blue_team["team"]),
            "red": str(red_team["team"]),
        },
        "league": league,
        "event_id": event_id or None,
        "event_start": event_start or None,
        "minute": _round(minute, 2),
        "state": state,
        "draft": {"blue": blue_picks, "red": red_picks},
        "pregame_kills": {
            key: _round(value, 3) if isinstance(value, float) else value
            for key, value in pregame.items()
            if key not in {"unknown_champions"}
        }
        | {"unknown_champions": pregame["unknown_champions"]},
        "pregame_win": pregame_win,
        "live_win": live_win,
        "rating_registration": rating_registration,
        "winner_reprice": winner_reprice,
        "live_totals": {
            **live_totals,
            "projected_mean": _round(live_totals.get("projected_mean"), 2),
            "projected_sd": _round(live_totals.get("projected_sd"), 2),
            "effective_n": _round(live_totals.get("effective_n"), 1),
        },
        "coverage": {
            "win_model": (
                "development diagnostic: checkpoint model + personal soft objective priors"
                if live_win.get("diagnostic_p_blue") is not None
                else "development diagnostic withheld"
            ),
            "total_model": "research diagnostic: held-out exact-checkpoint model",
            "betting_decision": (
                "authorized"
                if winner_authority.get("status") == "approved"
                and live_totals["classification_available"]
                else "not authorized"
            ),
            "barons_used": False,
            "inhibitors_used": False,
            "manual_state": True,
        },
    }
