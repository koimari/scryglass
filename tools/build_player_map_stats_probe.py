"""Local driver: build features/player_map_stats.json from the real OE census.

Read-only probe used to size the artifact before it is wired into the release.
It mirrors the release path in ``lol_kills.export.public_pack``: the accepted
rating game population is derived with the same
``filter_public_team_rating_maps`` predicate the team ladder is fit on, and the
census trio is derived from the same canonical source game identities, so the
sizes and counts printed here describe what a release would actually publish.

Usage:
    python tools/build_player_map_stats_probe.py <parquet> <out.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lol_kills.etl.competition import canonicalize_competition_frame  # noqa: E402
from lol_kills.etl.source_keys import canonical_source_game_key  # noqa: E402
from lol_kills.export.pack_records import filter_public_team_rating_maps  # noqa: E402
from lol_kills.export.player_map_stats import (  # noqa: E402
    build_player_map_stats,
    player_map_stats_row_count,
)
from lol_kills.export.public_pack import source_identity_sha256  # noqa: E402
from lol_kills.ratings.player_elo import build_maps_frame_from_players  # noqa: E402

COLUMNS = (
    "gameid", "date", "league", "league_source", "result", "side", "position",
    "teamname", "playername", "champion", "kills", "deaths", "assists",
    "teamkills", "gamelength", "dpm", "damageshare", "totalgold", "cspm",
    "year", "oe_year", "competition_tier", "competition_scope", "event_kind",
)


def _canonical_ids(frame: pd.DataFrame) -> pd.Series:
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    source = frame["game_uid"] if "game_uid" in frame.columns else frame["gameid"]
    return pd.Series(
        [
            canonical_source_game_key(
                value, fallback.loc[index] if fallback is not None else None
            )
            for index, value in source.items()
        ],
        index=frame.index,
        dtype="string",
    )


def accepted_rating_game_ids(frame: pd.DataFrame) -> set[str]:
    """The rating population: the same predicate the team ladder is fit on."""

    maps = build_maps_frame_from_players(frame)
    if maps.empty:
        return set()
    if "competition_tier" in frame.columns:
        # ``build_maps_frame_from_players`` does not carry the tier through, and
        # the tier clause is one of the three exclusions the filter applies.
        tiers = (
            frame.assign(_gid=_canonical_ids(frame))
            .drop_duplicates("_gid")
            .set_index("_gid")["competition_tier"]
        )
        maps["competition_tier"] = maps["game_uid"].map(tiers).astype("object")
    accepted = filter_public_team_rating_maps(maps)
    return set(accepted["game_uid"].dropna().astype(str))


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    available = set(pq.ParquetFile(source).schema_arrow.names)
    columns = [column for column in COLUMNS if column in available]
    frame = pq.read_table(source, columns=columns).to_pandas()
    frame = canonicalize_competition_frame(frame)

    accepted = accepted_rating_game_ids(frame)
    all_ids = sorted({value for value in _canonical_ids(frame).dropna().astype(str) if value})
    as_of = pd.to_datetime(frame["date"], utc=True, errors="coerce").max()

    started = time.perf_counter()
    payload = build_player_map_stats(
        frame,
        accepted_game_ids=accepted,
        source_as_of=None if pd.isna(as_of) else as_of.isoformat().replace("+00:00", "Z"),
        source_game_count=len(all_ids),
        source_identity_sha256=source_identity_sha256(all_ids),
    )
    elapsed = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    size = destination.stat().st_size
    print(f"build seconds        : {elapsed:.1f}")
    print(f"source games         : {len(all_ids)}")
    print(f"accepted rating games: {len(accepted)}")
    print(f"players              : {len(payload['players'])}")
    print(f"teams                : {len(payload['teams'])}")
    print(f"emitted map rows     : {player_map_stats_row_count(payload)}")
    print(f"bytes                : {size} ({size / 1_000_000:.2f} MB)")
    print(f"source_as_of         : {payload['source_as_of']}")
    print(f"source_game_count    : {payload['source_game_count']}")
    print(f"source_identity      : {payload['source_identity_sha256']}")
    print(f"rating_game_count    : {payload['rating_game_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
