export type SeriesRow = Record<string, unknown>;

export type SeriesCard = {
  key: string;
  date: string;
  league: string;
  patch: string;
  tournament: string;
  teamA: string;
  teamB: string;
  winsA: number;
  winsB: number;
  bestOf: 1 | 3 | 5 | null;
  games: SeriesRow[];
  year: number;
  source: "oe" | "grid" | "mixed" | "unknown";
  completionSource: string | null;
};

export function formatCompletionSource(source: string | null | undefined): string | null {
  if (source === "events_game_end") return "GRID game-end event";
  if (source === "end_state_summary") return "GRID verified end-state summary";
  if (source === "mixed") return "GRID mixed completion evidence";
  return null;
}

export function formatGameDate(dateVal: unknown): string {
  const raw = String(dateVal ?? "").trim();
  const iso = raw.match(/(\d{4}-\d{2}-\d{2})/);
  if (iso) return iso[1];

  if (/^\d{10,13}$/.test(raw)) {
    const numeric = Number(raw);
    const date = new Date(raw.length === 13 ? numeric : numeric * 1000);
    if (Number.isFinite(date.getTime())) return date.toISOString().slice(0, 10);
  }

  const parsed = Date.parse(raw);
  if (Number.isFinite(parsed)) return new Date(parsed).toISOString().slice(0, 10);
  return raw.slice(0, 10);
}

function teamPairKey(a: string, b: string): string {
  return [a, b].sort((x, y) => x.localeCompare(y)).join("||");
}

function explicitSeriesKey(row: SeriesRow): string | null {
  const gridSeries = String(row.grid_series_id ?? "").trim();
  if (gridSeries) return `grid:${gridSeries}`;
  const uid = String(row.game_uid ?? row.oe_gameid ?? "").trim();
  const match = uid.match(/^(.*?)(?:_game_\d+)$/i);
  if (match?.[1]) return `oe:${match[1]}`;
  return null;
}

function contiguousGridIndices(games: SeriesRow[]): boolean {
  const indices = games.map((game) => Number(game.grid_game_index));
  if (indices.some((index) => !Number.isInteger(index) || index < 1)) return false;
  const unique = [...new Set(indices)].sort((a, b) => a - b);
  return unique.length === games.length && unique.every((index, offset) => index === offset + 1);
}

function inferBestOf(
  games: SeriesRow[],
  winsA: number,
  winsB: number,
  strictGridSeries: boolean,
): 1 | 3 | 5 | null {
  if (strictGridSeries && (games.length === 1 || !contiguousGridIndices(games))) return null;
  const winningScore = Math.max(winsA, winsB);
  if (winningScore >= 3) return 5;
  if (winningScore >= 2) return 3;
  if (games.length === 1) return 1;
  // A tied multi-map group is not enough evidence to claim a completed Bo3.
  return null;
}

/** Group OE/GRID map rows into Bo1/Bo3/Bo5 series using stable IDs first. */
export function groupMapsIntoSeries(rows: SeriesRow[]): SeriesCard[] {
  const buckets = new Map<string, SeriesRow[]>();
  for (const row of rows) {
    const blue = String(row.blue_teamname ?? "");
    const red = String(row.red_teamname ?? "");
    if (!blue || !red) continue;
    const day = formatGameDate(row.date);
    const league = String(row.league ?? "");
    const tournament = String(row.tournament ?? "");
    const seriesId = explicitSeriesKey(row);
    const key = seriesId
      ? `${seriesId}|${league}|${tournament}`
      : `${teamPairKey(blue, red)}|${day}|${league}|${tournament}`;
    const list = buckets.get(key) ?? [];
    list.push(row);
    buckets.set(key, list);
  }

  const series: SeriesCard[] = [];
  for (const [key, games] of buckets) {
    games.sort(
      (a, b) =>
        Number(a.grid_game_index ?? a.game ?? 0) - Number(b.grid_game_index ?? b.game ?? 0),
    );
    const first = games[0];
    const blue = String(first.blue_teamname);
    const red = String(first.red_teamname);
    // Stable A/B by alphabetical for scoreline display.
    const [teamA, teamB] = [blue, red].sort((x, y) => x.localeCompare(y));
    let winsA = 0;
    let winsB = 0;
    for (const game of games) {
      const blueName = String(game.blue_teamname);
      const blueWon =
        game.blue_result === 1 ||
        game.blue_result === true ||
        game.y_blue_win === 1 ||
        Number(game.y_blue_win) >= 0.5;
      const winner = blueWon ? blueName : String(game.red_teamname);
      if (winner === teamA) winsA += 1;
      else if (winner === teamB) winsB += 1;
    }
    const hasGrid = games.some(
      (game) => game.source_grid === true || Number(game.source_grid) === 1,
    );
    const hasOe = games.some(
      (game) => game.source_oe === true || Number(game.source_oe) === 1,
    );
    const completionSources = [
      ...new Set(
        games
          .map((game) => String(game.grid_completion_source ?? "").trim())
          .filter(Boolean),
      ),
    ];
    const strictGridSeries = games.some((game) => String(game.grid_series_id ?? "").trim());
    series.push({
      key,
      date: formatGameDate(first.date),
      league: String(first.league ?? ""),
      patch: String(first.patch ?? ""),
      tournament: String(first.tournament ?? ""),
      teamA,
      teamB,
      winsA,
      winsB,
      bestOf: inferBestOf(games, winsA, winsB, strictGridSeries),
      games,
      year:
        Number(first._year ?? first.year ?? 0) ||
        Number(formatGameDate(first.date).slice(0, 4)) ||
        0,
      source: hasGrid && hasOe ? "mixed" : hasGrid ? "grid" : hasOe ? "oe" : "unknown",
      completionSource:
        completionSources.length === 1
          ? completionSources[0]
          : completionSources.length > 1
            ? "mixed"
            : null,
    });
  }
  series.sort((a, b) => b.date.localeCompare(a.date) || a.key.localeCompare(b.key));
  return series;
}
