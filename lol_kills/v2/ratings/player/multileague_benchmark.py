"""Strong-baseline and roster-change benchmark for multi-league ratings.

The Player Rating development winner must be compared with a roster-agnostic
organization model, not merely another temporal variant of itself.  This
module locks that comparison without changing the frozen rating artifact:

* the Player candidate is selected by the already-frozen development artifact;
* an organization baseline family is selected on DEVELOPMENT only;
* both are replayed with identical series freeze and strict 48-hour embargo;
* VALIDATION is compared with series-cluster bootstrap intervals overall, by
  domestic league, and by pre-match roster-change stratum; and
* sealed-final outcomes remain unopened.

The benchmark is non-authorizing.  Passing it would be necessary, not
sufficient, for a production Player/Team Rating.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from lol_kills.v2.evaluation.bootstrap import series_cluster_bootstrap

from . import multileague_development as adapter
from . import multileague_runner as rating


SCHEMA_VERSION = "scryglass:multileague-rating-strong-baseline-benchmark:v1"
DEFAULT_RATING_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v1/private-development-artifact-v1.json"
)
MINIMUM_SERIES = 20
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260811
ROSTER_STRATA = (
    "BOTH_ROSTERS_STABLE",
    "ONE_OR_BOTH_ROSTERS_CHANGED",
    "NO_PRIOR_EXACT_LINEUP",
)

ORGANIZATION_CANDIDATES = (
    rating.Candidate("organization_static_no_reset", "STATIC", 0.0),
    rating.Candidate(
        "organization_random_walk_no_reset",
        "RANDOM_WALK",
        rating.RANDOM_WALK_VARIANCE_PER_DAY,
    ),
    rating.Candidate(
        "organization_mean_reversion_no_reset",
        "MEAN_REVERSION",
        rating.RANDOM_WALK_VARIANCE_PER_DAY,
        rating.MEAN_REVERSION_HALF_LIFE_DAYS,
    ),
)


class MultiLeagueBenchmarkError(ValueError):
    """The strong-baseline replay or artifact failed closed."""


@dataclass(frozen=True)
class _OrganizationPending:
    available_at: datetime
    series: adapter.DevelopmentSeries
    features: tuple[rating._FeatureVector, ...]


@dataclass
class _OrganizationReplay:
    candidate: rating.Candidate
    predictions: list[dict[str, Any]]
    state: rating._GaussianState
    applied_series: int
    applied_maps: int
    bridge_diagnostics: dict[str, int]
    roster_strata: dict[str, int]


def _organization_identity(team_id: str) -> str:
    return f"organization:{team_id}"


def _organization_key(team_id: str) -> str:
    # The bounded Gaussian engine names dynamic entities with its player-key
    # namespace.  The explicit organization prefix prevents identity overlap.
    return rating._player_key(_organization_identity(team_id))


def _series_organizations(series: adapter.DevelopmentSeries) -> tuple[str, ...]:
    values = {
        lineup.team_id
        for item in series.maps
        for lineup in (item.blue_lineup, item.red_lineup)
    }
    if len(values) != 2:
        raise MultiLeagueBenchmarkError("organization series must contain exactly two teams")
    return tuple(sorted(values))


def _lineup_identity(lineup: adapter.ObservedLineup) -> tuple[tuple[str, str], ...]:
    if tuple(slot.role for slot in lineup.players) != adapter.ROLE_ORDER:
        raise MultiLeagueBenchmarkError("lineup role order changed")
    return tuple((slot.role, slot.player_id) for slot in lineup.players)


def _roster_status(
    item: adapter.DevelopmentMap,
    known_lineups: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[str, str, str]:
    values = []
    for lineup in (item.blue_lineup, item.red_lineup):
        previous = known_lineups.get(lineup.team_id)
        current = _lineup_identity(lineup)
        values.append(
            "NO_PRIOR_EXACT_LINEUP"
            if previous is None
            else "STABLE"
            if previous == current
            else "CHANGED"
        )
    if "NO_PRIOR_EXACT_LINEUP" in values:
        stratum = "NO_PRIOR_EXACT_LINEUP"
    elif "CHANGED" in values:
        stratum = "ONE_OR_BOTH_ROSTERS_CHANGED"
    else:
        stratum = "BOTH_ROSTERS_STABLE"
    return values[0], values[1], stratum


def _organization_feature(
    state: rating._GaussianState,
    item: adapter.DevelopmentMap,
    known_home_leagues: Mapping[str, str],
) -> rating._FeatureVector:
    blue_home, red_home, status = rating._home_leagues(item, known_home_leagues)
    state.ensure_structural_keys(
        [league for league in (blue_home, red_home) if league is not None]
    )
    weights: dict[str, float] = defaultdict(float)
    weights[_organization_key(item.blue_lineup.team_id)] += 1.0
    weights[_organization_key(item.red_lineup.team_id)] -= 1.0
    if blue_home is not None:
        weights[rating._league_key(blue_home)] += 1.0
    if red_home is not None:
        weights[rating._league_key(red_home)] -= 1.0
    weights[rating.BLUE_SIDE_KEY] += 1.0
    return rating._FeatureVector(
        {key: value for key, value in weights.items() if abs(value) > 1e-15},
        blue_home,
        red_home,
        status,
    )


def _organization_replay(
    input_data: adapter.PrivateMultiLeagueRatingInput,
    candidate: rating.Candidate,
) -> _OrganizationReplay:
    state = rating._GaussianState(candidate)
    predictions: list[dict[str, Any]] = []
    pending: list[tuple[datetime, int, str, _OrganizationPending]] = []
    sequence = 0
    known_home_leagues: dict[str, str] = {}
    home_order: dict[str, tuple[datetime, int, str]] = {}
    known_lineups: dict[str, tuple[tuple[str, str], ...]] = {}
    lineup_order: dict[str, tuple[datetime, int, str]] = {}
    applied_series = 0
    applied_maps = 0
    bridge_diagnostics: dict[str, int] = defaultdict(int)
    roster_strata: dict[str, int] = defaultdict(int)

    def apply(value: _OrganizationPending) -> None:
        nonlocal applied_series, applied_maps
        organizations = [_organization_identity(team) for team in _series_organizations(value.series)]
        state.transition_players(organizations, value.available_at)
        for item, feature in zip(value.series.maps, value.features):
            state.update(feature.weights, item.blue_win)
            at = adapter.source_local_datetime(item.source_local_start)
            order = (at, item.game_number, item.game_id)
            for lineup in (item.blue_lineup, item.red_lineup):
                if lineup_order.get(lineup.team_id, (datetime.min, 0, "")) < order:
                    lineup_order[lineup.team_id] = order
                    known_lineups[lineup.team_id] = _lineup_identity(lineup)
                if (
                    value.series.league in adapter.DOMESTIC_LEAGUES
                    and home_order.get(lineup.team_id, (datetime.min, 0, "")) < order
                ):
                    home_order[lineup.team_id] = order
                    known_home_leagues[lineup.team_id] = value.series.league
        applied_series += 1
        applied_maps += len(value.series.maps)

    def flush(boundary: datetime) -> None:
        while pending and pending[0][0] < boundary:
            _available, _sequence, _identity, value = heapq.heappop(pending)
            apply(value)

    for series in input_data.development_series:
        start = adapter.source_local_datetime(series.source_local_start)
        flush(start)
        organizations = [_organization_identity(team) for team in _series_organizations(series)]
        state.transition_players(organizations, start)
        features: list[rating._FeatureVector] = []
        for item in series.maps:
            feature = _organization_feature(state, item, known_home_leagues)
            features.append(feature)
            probability, latent_mean, latent_variance = state.predict(feature.weights)
            blue_status, red_status, stratum = _roster_status(item, known_lineups)
            bridge_diagnostics[feature.bridge_status] += 1
            roster_strata[stratum] += 1
            predictions.append(
                {
                    "game_id": item.game_id,
                    "series_id": series.series_id,
                    "series_identity_kind": series.series_identity_kind,
                    "fold_id": series.fold_id,
                    "league": series.league,
                    "source_local_start": item.source_local_start,
                    "game_number": item.game_number,
                    "probability": probability,
                    "latent_mean": latent_mean,
                    "latent_variance": latent_variance,
                    "outcome": item.blue_win,
                    "league_bridge_status": feature.bridge_status,
                    "blue_home_league": feature.blue_home_league,
                    "red_home_league": feature.red_home_league,
                    "blue_roster_status": blue_status,
                    "red_roster_status": red_status,
                    "roster_change_stratum": stratum,
                }
            )
        available = adapter.source_local_datetime(series.source_local_end) + timedelta(
            hours=rating.AVAILABILITY_EMBARGO_HOURS
        )
        value = _OrganizationPending(available, series, tuple(features))
        heapq.heappush(pending, (available, sequence, series.series_id, value))
        sequence += 1
    flush(adapter.SEALED_FINAL_START.to_pydatetime())
    state.assert_psd()
    return _OrganizationReplay(
        candidate=candidate,
        predictions=predictions,
        state=state,
        applied_series=applied_series,
        applied_maps=applied_maps,
        bridge_diagnostics=dict(sorted(bridge_diagnostics.items())),
        roster_strata=dict(sorted(roster_strata.items())),
    )


def _series_macro(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    metrics = rating._metric_payload(rows)["series_macro"]
    if metrics["log_loss"] is None or metrics["brier"] is None:
        raise MultiLeagueBenchmarkError("development baseline selection has no rows")
    return float(metrics["log_loss"]), float(metrics["brier"])


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    league: str | None = None,
    roster_stratum: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["fold_id"] == fold_id
        and (league is None or row["league"] == league)
        and (
            roster_stratum is None
            or row.get("roster_change_stratum") == roster_stratum
        )
    ]


def _paired_player_minus_organization(
    player_rows: Sequence[Mapping[str, Any]],
    organization_rows: Sequence[Mapping[str, Any]],
    *,
    league: str | None = None,
    roster_stratum: str | None = None,
) -> dict[str, Any]:
    player = {
        str(row["game_id"]): row
        for row in _selected_rows(
            player_rows,
            fold_id="VALIDATION",
            league=league,
            roster_stratum=roster_stratum,
        )
    }
    organization = {
        str(row["game_id"]): row
        for row in _selected_rows(
            organization_rows,
            fold_id="VALIDATION",
            league=league,
            roster_stratum=roster_stratum,
        )
    }
    if set(player) != set(organization):
        raise MultiLeagueBenchmarkError("player and organization validation rows differ")
    ids = sorted(player)
    if not ids:
        return {
            "status": "UNAVAILABLE_NO_ROWS",
            "league": league,
            "roster_change_stratum": roster_stratum,
            "maps": 0,
            "series": 0,
            "log_loss_player_minus_organization": None,
            "brier_player_minus_organization": None,
        }
    clusters = [str(player[game_id]["series_id"]) for game_id in ids]
    series = len(set(clusters))
    results: dict[str, dict[str, Any]] = {}
    for offset, metric in enumerate(("log_loss", "brier")):
        deltas = [
            rating._loss(
                float(player[game_id]["probability"]), int(player[game_id]["outcome"]), metric
            )
            - rating._loss(
                float(organization[game_id]["probability"]),
                int(organization[game_id]["outcome"]),
                metric,
            )
            for game_id in ids
        ]
        result = series_cluster_bootstrap(
            deltas,
            clusters,
            [True] * len(ids),
            row_ids=ids,
            n_boot=BOOTSTRAP_SAMPLES,
            random_seed=BOOTSTRAP_SEED + offset,
            cluster_unit="source-or-derived-series-dependence-cluster",
        )
        sizes: dict[str, int] = defaultdict(int)
        for size in result.cluster_size_distribution.values():
            sizes[str(int(size))] += 1
        results[metric] = {
            "point": result.point,
            "lower_95": result.lower_95,
            "upper_95": result.upper_95,
            "cluster_count": result.cluster_count,
            "cluster_size_distribution": dict(
                sorted(sizes.items(), key=lambda item: int(item[0]))
            ),
        }
    enough = series >= MINIMUM_SERIES
    passed = enough and all(value["upper_95"] <= 0.0 for value in results.values())
    return {
        "status": (
            "PASS_NONPOSITIVE_UPPER_95"
            if passed
            else "FAIL_INSUFFICIENT_SERIES"
            if not enough
            else "FAIL_UPPER_95_ABOVE_ZERO"
        ),
        "league": league,
        "roster_change_stratum": roster_stratum,
        "maps": len(ids),
        "series": series,
        "minimum_required_series": MINIMUM_SERIES,
        "log_loss_player_minus_organization": results["log_loss"],
        "brier_player_minus_organization": results["brier"],
    }


def _attach_roster_strata(
    player_rows: Sequence[Mapping[str, Any]],
    organization_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row["game_id"]): row for row in organization_rows}
    if {str(row["game_id"]) for row in player_rows} != set(by_id):
        raise MultiLeagueBenchmarkError("player and organization replay populations differ")
    return [
        {
            **dict(row),
            "blue_roster_status": by_id[str(row["game_id"])]["blue_roster_status"],
            "red_roster_status": by_id[str(row["game_id"])]["red_roster_status"],
            "roster_change_stratum": by_id[str(row["game_id"])]["roster_change_stratum"],
        }
        for row in player_rows
    ]


def _constant_half_metrics(rows: Sequence[Mapping[str, Any]], fold_id: str) -> dict[str, Any]:
    values = [
        {**dict(row), "probability": 0.5}
        for row in rows
        if row["fold_id"] == fold_id
    ]
    return rating._metric_payload(values)


def _validate_rating_artifact(
    artifact: Mapping[str, Any], expected_sha256: str
) -> dict[str, Any]:
    expected = rating._require_sha256(expected_sha256, "expected_rating_artifact_sha256")
    try:
        value = rating.validate_multileague_development_artifact(artifact)
    except rating.MultiLeagueRunnerError as error:
        raise MultiLeagueBenchmarkError("rating artifact is invalid") from error
    if value["artifact_sha256"] != expected:
        raise MultiLeagueBenchmarkError("rating artifact does not match the independent pin")
    return value


def build_strong_baseline_benchmark(
    *,
    expected_maps_sha256: str,
    expected_players_sha256: str,
    rating_artifact: Mapping[str, Any],
    expected_rating_artifact_sha256: str,
    input_loader: Callable[..., adapter.PrivateMultiLeagueRatingInput] = adapter.load_multileague_development_input,
) -> dict[str, Any]:
    frozen_rating = _validate_rating_artifact(
        rating_artifact, expected_rating_artifact_sha256
    )
    input_data = input_loader(
        expected_maps_sha256=expected_maps_sha256,
        expected_players_sha256=expected_players_sha256,
    )
    rating._validate_input(
        input_data,
        expected_maps_sha256=expected_maps_sha256,
        expected_players_sha256=expected_players_sha256,
    )
    if (
        frozen_rating["input"]["maps_sha256"] != input_data.maps_sha256
        or frozen_rating["input"]["players_sha256"] != input_data.players_sha256
        or frozen_rating["input"]["cluster_partition_sha256"]
        != input_data.cluster_partition_sha256
    ):
        raise MultiLeagueBenchmarkError("benchmark input does not match the frozen rating input")

    player_id = frozen_rating["selection"]["development_winner_candidate_id"]
    player_candidate = next(
        (candidate for candidate in rating.CANDIDATES if candidate.candidate_id == player_id),
        None,
    )
    if player_candidate is None:
        raise MultiLeagueBenchmarkError("frozen rating selected an unknown Player candidate")
    player_replay = rating._replay(input_data, player_candidate)
    frozen_candidate = next(
        item
        for item in frozen_rating["candidate_results"]
        if item["candidate"]["candidate_id"] == player_id
    )
    for fold in ("DEVELOPMENT", "VALIDATION"):
        replayed = rating._evaluation_payload(player_replay.predictions, fold)
        if replayed["prediction_rows_sha256"] != frozen_candidate[fold.lower()][
            "prediction_rows_sha256"
        ]:
            raise MultiLeagueBenchmarkError("Player prediction replay does not match frozen artifact")

    organization_replays = [
        _organization_replay(input_data, candidate)
        for candidate in ORGANIZATION_CANDIDATES
    ]
    selected_organization = min(
        organization_replays,
        key=lambda replay: (
            *_series_macro(
                _selected_rows(replay.predictions, fold_id="DEVELOPMENT")
            ),
            replay.candidate.candidate_id,
        ),
    )
    player_rows = _attach_roster_strata(
        player_replay.predictions, selected_organization.predictions
    )
    overall = _paired_player_minus_organization(
        player_rows, selected_organization.predictions
    )
    by_league = [
        _paired_player_minus_organization(
            player_rows, selected_organization.predictions, league=league
        )
        for league in adapter.DOMESTIC_LEAGUES
    ]
    by_roster = [
        _paired_player_minus_organization(
            player_rows,
            selected_organization.predictions,
            roster_stratum=stratum,
        )
        for stratum in ROSTER_STRATA
    ]
    required = [overall, *by_league, *by_roster[:2]]
    gate_failures = []
    for item in required:
        if item["status"] != "PASS_NONPOSITIVE_UPPER_95":
            label = item.get("league") or item.get("roster_change_stratum") or "overall"
            gate_failures.append(f"{str(label).lower()}_player_not_superior_to_organization")
    gate_passed = not gate_failures

    organization_results = []
    for replay in organization_replays:
        organization_results.append(
            {
                "candidate": replay.candidate.payload(),
                "development": rating._evaluation_payload(
                    replay.predictions, "DEVELOPMENT"
                ),
                "validation": rating._evaluation_payload(
                    replay.predictions, "VALIDATION"
                ),
                "replay": {
                    "series_predictions_are_prior_frozen": True,
                    "strict_embargo_hours": rating.AVAILABILITY_EMBARGO_HOURS,
                    "applied_series_at_sealed_boundary": replay.applied_series,
                    "applied_maps_at_sealed_boundary": replay.applied_maps,
                    "league_bridge_diagnostics": replay.bridge_diagnostics,
                    "roster_strata_all_folds": replay.roster_strata,
                    "posterior_state_sha256": rating._state_digest(replay.state),
                    "latent_dimension": len(replay.state.keys),
                },
                "numerics": {
                    "finite": True,
                    "covariance": "FULL_GAUSSIAN_RANK_ONE_UPDATED",
                    **replay.state.assert_psd(),
                },
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": (
            "PLAYER_BEATS_STRONG_BASELINE_VALIDATION_GATE_PASSED"
            if gate_passed
            else "PLAYER_DOES_NOT_BEAT_STRONG_BASELINE"
        ),
        "private_scope": {
            "development_evaluation": True,
            "production_authorized": False,
            "sealed_final_opened": False,
            "probability_authorized": False,
            "betting_authorized": False,
        },
        "input": {
            "maps_sha256": input_data.maps_sha256,
            "players_sha256": input_data.players_sha256,
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
            "rating_artifact_sha256": frozen_rating["artifact_sha256"],
            "rating_artifact_result_state": frozen_rating["result_state"],
        },
        "selection": {
            "player_candidate_id_from_frozen_rating_artifact": player_id,
            "organization_selection_fold": "DEVELOPMENT",
            "organization_candidate_id": selected_organization.candidate.candidate_id,
            "validation_gate_passed": gate_passed,
            "validation_gate_failures": gate_failures,
            "sealed_final_opened": False,
        },
        "player_candidate": {
            "candidate": player_candidate.payload(),
            "development": rating._evaluation_payload(
                player_rows, "DEVELOPMENT"
            ),
            "validation": rating._evaluation_payload(player_rows, "VALIDATION"),
            "constant_half_comparator": {
                "development": _constant_half_metrics(player_rows, "DEVELOPMENT"),
                "validation": _constant_half_metrics(player_rows, "VALIDATION"),
            },
        },
        "organization_candidates": organization_results,
        "validation_player_minus_selected_organization": {
            "overall": overall,
            "by_domestic_league": by_league,
            "by_roster_change_stratum": by_roster,
        },
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "generator": {
            "sources": [
                {
                    "locator": "lol_kills/v2/ratings/player/multileague_benchmark.py",
                    "raw_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                {
                    "locator": "lol_kills/v2/ratings/player/multileague_runner.py",
                    "raw_sha256": hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
                },
                {
                    "locator": "lol_kills/v2/ratings/player/multileague_development.py",
                    "raw_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
                },
            ],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_series_per_required_stratum": MINIMUM_SERIES,
        },
    }
    artifact["artifact_sha256"] = rating._canonical_sha256(artifact)
    return artifact


def validate_strong_baseline_benchmark(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise MultiLeagueBenchmarkError("benchmark artifact must be an object")
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != rating._canonical_sha256(unsigned):
        raise MultiLeagueBenchmarkError("benchmark artifact digest mismatch")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise MultiLeagueBenchmarkError("benchmark artifact schema mismatch")
    if artifact.get("result_state") not in {
        "PLAYER_BEATS_STRONG_BASELINE_VALIDATION_GATE_PASSED",
        "PLAYER_DOES_NOT_BEAT_STRONG_BASELINE",
    }:
        raise MultiLeagueBenchmarkError("benchmark result state is invalid")
    scope = artifact.get("private_scope") or {}
    if (
        scope.get("production_authorized") is not False
        or scope.get("sealed_final_opened") is not False
        or scope.get("probability_authorized") is not False
        or scope.get("betting_authorized") is not False
    ):
        raise MultiLeagueBenchmarkError("benchmark claim boundary is invalid")
    if any(value is not None for value in (artifact.get("decision_outputs") or {}).values()):
        raise MultiLeagueBenchmarkError("benchmark emitted an actionable decision output")
    selection = artifact.get("selection") or {}
    if selection.get("sealed_final_opened") is not False:
        raise MultiLeagueBenchmarkError("benchmark opened sealed final")
    expected_sources = {
        "lol_kills/v2/ratings/player/multileague_benchmark.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "lol_kills/v2/ratings/player/multileague_runner.py": hashlib.sha256(
            Path(rating.__file__).read_bytes()
        ).hexdigest(),
        "lol_kills/v2/ratings/player/multileague_development.py": hashlib.sha256(
            Path(adapter.__file__).read_bytes()
        ).hexdigest(),
    }
    actual_sources = {
        item.get("locator"): item.get("raw_sha256")
        for item in (artifact.get("generator") or {}).get("sources", [])
        if isinstance(item, Mapping)
    }
    if actual_sources != expected_sources:
        raise MultiLeagueBenchmarkError("benchmark executable source binding is stale")
    return dict(artifact)


def write_strong_baseline_benchmark_no_clobber(
    artifact: Mapping[str, Any], path: Path
) -> str:
    value = validate_strong_baseline_benchmark(artifact)
    raw = rating._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MultiLeagueBenchmarkError("benchmark output exists; refusing to clobber") from error
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise MultiLeagueBenchmarkError("benchmark output is not a regular file")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the multi-league strong-baseline benchmark.")
    parser.add_argument("--expected-maps-sha256", required=True)
    parser.add_argument("--expected-players-sha256", required=True)
    parser.add_argument("--rating-artifact", type=Path, default=DEFAULT_RATING_ARTIFACT)
    parser.add_argument("--expected-rating-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.rating_artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiLeagueBenchmarkError("rating artifact cannot be loaded") from error
    artifact = build_strong_baseline_benchmark(
        expected_maps_sha256=args.expected_maps_sha256,
        expected_players_sha256=args.expected_players_sha256,
        rating_artifact=source,
        expected_rating_artifact_sha256=args.expected_rating_artifact_sha256,
    )
    raw_sha256 = write_strong_baseline_benchmark_no_clobber(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": artifact["artifact_sha256"],
                "result_state": artifact["result_state"],
                "selection": artifact["selection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

