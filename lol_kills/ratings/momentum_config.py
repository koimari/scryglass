"""Registered configurations for the descriptive rating momentum state.

The active runtime uses zero momentum.  The seven-game, scale-80 setting is
the research candidate from the 2026-08-15 handoff.  It has no public,
production, probability, recommendation, odds, EV, or promotion authority.
"""

from __future__ import annotations

from typing import Any

MOMENTUM_SCHEMA_VERSION = "scryglass:rating-momentum-config:v1"
DEFAULT_MOMENTUM_WINDOW_GAMES = 0
DEFAULT_MOMENTUM_SCALE = 0.0
CANDIDATE_MOMENTUM_WINDOW_GAMES = 7
CANDIDATE_MOMENTUM_SCALE = 80.0


def momentum_configuration(*, window_games: int, scale: float, status: str) -> dict[str, Any]:
    """Return the canonical, serializable configuration record."""

    return {
        "schema_version": MOMENTUM_SCHEMA_VERSION,
        "window_games": int(window_games),
        "scale": float(scale),
        "scale_unit": "rating_points_per_residual_unit",
        "status": str(status),
        "authority": {
            "public": False,
            "production": False,
            "probability": False,
            "recommendation": False,
            "odds": False,
            "ev": False,
            "promotion": False,
        },
    }


def active_momentum_configuration() -> dict[str, Any]:
    """Return the zero-momentum configuration used by default."""

    return momentum_configuration(
        window_games=DEFAULT_MOMENTUM_WINDOW_GAMES,
        scale=DEFAULT_MOMENTUM_SCALE,
        status="active_default",
    )


def candidate_momentum_configuration() -> dict[str, Any]:
    """Return the research-only seven-map, scale-80 candidate."""

    return momentum_configuration(
        window_games=CANDIDATE_MOMENTUM_WINDOW_GAMES,
        scale=CANDIDATE_MOMENTUM_SCALE,
        status="research_candidate",
    )


def selected_momentum_configuration(*, window_games: int, scale: float) -> dict[str, Any]:
    """Describe a run selected by an explicit private configuration.

    A non-zero run is a research candidate.  This function never grants
    authority to a model or to a public output.
    """

    is_default = int(window_games) == DEFAULT_MOMENTUM_WINDOW_GAMES and float(scale) == DEFAULT_MOMENTUM_SCALE
    return momentum_configuration(
        window_games=window_games,
        scale=scale,
        status="active_default" if is_default else "research_candidate",
    )


def registered_momentum_bundle() -> dict[str, Any]:
    """Return active, candidate, and promotion records for manifests."""

    return {
        "schema_version": MOMENTUM_SCHEMA_VERSION,
        "active": active_momentum_configuration(),
        "candidate": candidate_momentum_configuration(),
        "promotion": {
            "status": "unavailable",
            "reason": "independent time-split, calibration, and receipt gates are pending",
            "public": False,
            "production": False,
            "probability": False,
            "recommendation": False,
            "odds": False,
            "ev": False,
        },
    }
