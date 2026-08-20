from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import (
    TRANSPORT_LEAGUE_LABELS,
    _text,
    canonicalize_competition_frame,
    classify_competition,
    source_league,
    team_identity_key,
)


def _rowwise_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """The pre-optimization implementation used as a parity oracle."""

    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()

    out = frame.copy()
    if "league" in out.columns:
        if "league_source" not in out.columns:
            out["league_source"] = out["league"]
        labels = []
        for _, row in out.iterrows():
            source = source_league(row.get("league_source"))
            fallback = source_league(row.get("league"))
            value = (
                fallback
                if source in TRANSPORT_LEAGUE_LABELS
                and fallback not in TRANSPORT_LEAGUE_LABELS
                else source or fallback
            )
            labels.append(classify_competition(value, row.get("tournament")))
        out["league_source"] = [label.source for label in labels]
        out["league"] = [label.league for label in labels]
        out["competition_scope"] = [label.scope for label in labels]
        out["event_kind"] = [label.event_kind for label in labels]
        out["is_international"] = [label.is_international for label in labels]
        out["is_interregional"] = [label.is_interregional for label in labels]
        out["competition_tier"] = [label.tier for label in labels]

    team_columns = ("teamname", "blue_team", "red_team", "blue_teamname", "red_teamname")
    for column in team_columns:
        if column not in out.columns:
            continue
        out[column] = out[column].map(
            lambda value: normalize_team(_text(value)) if _text(value) else value
        )
        key_column = {
            "teamname": "team_key",
            "blue_team": "blue_team_key",
            "red_team": "red_team_key",
            "blue_teamname": "blue_team_key",
            "red_teamname": "red_team_key",
        }[column]
        if key_column not in out.columns or column in ("blue_teamname", "red_teamname"):
            out[key_column] = out[column].map(team_identity_key)

    return out


def _classification_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    work = frame.copy()
    if "league_source" not in work.columns:
        work["league_source"] = work["league"]
    source_values = work["league_source"].map(source_league)
    fallback_values = work["league"].map(source_league)
    source_array = source_values.to_numpy(dtype=object)
    fallback_array = fallback_values.to_numpy(dtype=object)
    use_fallback = (
        source_values.eq("").to_numpy()
        | (
            source_values.isin(TRANSPORT_LEAGUE_LABELS).to_numpy()
            & ~fallback_values.isin(TRANSPORT_LEAGUE_LABELS).to_numpy()
        )
    )
    values = source_array.copy()
    values[use_fallback] = fallback_array[use_fallback]
    tournament_values = (
        work["tournament"].map(source_league)
        if "tournament" in work.columns
        else pd.Series("", index=work.index, dtype=object)
    )
    return list(zip(values.tolist(), tournament_values.to_numpy(dtype=object).tolist()))


def _fixture(*, include_optional_columns: bool = True) -> pd.DataFrame:
    rows = [
        {
            "league": "LTA N",
            "league_source": "LTA N",
            "tournament": "LTA 2026 Split",
            "teamname": "KC",
            "blue_team": "DK",
            "red_team": "Gen.G",
        },
        {
            "league": "LTA S",
            "league_source": None,
            "tournament": None,
            "teamname": "FURIA",
            "blue_team": "G2",
            "red_team": "T1",
        },
        {
            "league": "LTA",
            "league_source": "LTA",
            "tournament": "Americas Cross Region",
            "teamname": np.nan,
            "blue_team": pd.NA,
            "red_team": None,
        },
        {
            "league": "LCK",
            "league_source": "ORACLE_ELIXIR_API",
            "tournament": "LCK 2026 Road to MSI",
            "teamname": "T1",
            "blue_team": "Gen.G",
            "red_team": "DK",
        },
        {
            "league": "ORACLE_ELIXIR_API",
            "league_source": "OE API",
            "tournament": "MSI",
            "teamname": "G2",
            "blue_team": "KC",
            "red_team": "FURIA",
        },
        {
            "league": "EM",
            "league_source": "EM",
            "tournament": "Eternal Masters",
            "teamname": "Movistar KOI",
            "blue_team": "Cloud9",
            "red_team": "FlyQuest",
        },
        {
            "league": None,
            "league_source": np.nan,
            "tournament": pd.NA,
            "teamname": pd.NA,
            "blue_team": "",
            "red_team": " ",
        },
        {
            "league": "WLDs",
            "league_source": "WLDs",
            "tournament": "World Championship",
            "teamname": "Team Liquid",
            "blue_team": "BLG",
            "red_team": "RNG",
        },
        {
            "league": "LCK",
            "league_source": "ORACLE_ELIXIR_API",
            "tournament": "LCK 2026 Road to MSI",
            "teamname": "T1",
            "blue_team": "Gen.G",
            "red_team": "DK",
        },
    ]
    frame = pd.DataFrame(rows)
    if not include_optional_columns:
        frame = frame.drop(columns=["league_source", "tournament"])
    return frame.astype(object)


def test_vectorized_frame_matches_rowwise_reference_with_aliases_and_missing_values() -> None:
    frame = _fixture()
    before = frame.copy(deep=True)

    actual = canonicalize_competition_frame(frame)
    expected = _rowwise_reference(frame)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    pd.testing.assert_frame_equal(frame, before, check_dtype=True)


def test_vectorized_frame_matches_reference_without_optional_columns() -> None:
    frame = _fixture(include_optional_columns=False)

    actual = canonicalize_competition_frame(frame)
    expected = _rowwise_reference(frame)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_vectorized_frame_preserves_duplicate_index_row_order() -> None:
    frame = _fixture()
    frame.index = [4, 4, 2, 2, 1, 1, 0, 0, 0]

    actual = canonicalize_competition_frame(frame)
    expected = _rowwise_reference(frame)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_classification_runs_once_per_unique_normalized_pair() -> None:
    frame = _fixture()
    expected_calls = len(set(_classification_keys(frame)))

    with patch(
        "lol_kills.etl.competition.classify_competition",
        wraps=classify_competition,
    ) as classify:
        canonicalize_competition_frame(frame)

    assert classify.call_count == expected_calls
