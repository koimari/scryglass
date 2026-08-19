"""Local driver: build features/player_map_stats.json from the real OE census.

Read-only probe used to size the artifact before it is wired into the release.
Usage:
    python tools/build_player_map_stats_probe.py <parquet> <out.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lol_kills.etl.competition import canonicalize_competition_frame  # noqa: E402
from lol_kills.export.player_map_stats import (  # noqa: E402
    build_player_map_stats,
    player_map_stats_row_count,
)

COLUMNS = (
    "gameid", "date", "league", "league_source", "result", "side", "position",
    "teamname", "playername", "champion", "kills", "deaths", "assists",
    "teamkills", "gamelength", "dpm", "damageshare", "totalgold", "cspm",
    "year", "oe_year", "competition_tier", "competition_scope", "event_kind",
)


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    available = set(pq.ParquetFile(source).schema_arrow.names)
    columns = [column for column in COLUMNS if column in available]
    frame = pq.read_table(source, columns=columns).to_pandas()
    frame = canonicalize_competition_frame(frame)
    started = time.perf_counter()
    payload = build_player_map_stats(frame)
    elapsed = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    size = destination.stat().st_size
    print(f"build seconds      : {elapsed:.1f}")
    print(f"players            : {len(payload['players'])}")
    print(f"teams              : {len(payload['teams'])}")
    print(f"emitted map rows   : {player_map_stats_row_count(payload)}")
    print(f"bytes              : {size} ({size / 1_000_000:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
