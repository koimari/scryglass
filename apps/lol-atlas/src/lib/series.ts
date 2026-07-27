export type SeriesRow = Record<string, unknown>;

export type CanonicalSeriesStatus =
  | "completed"
  | "incomplete"
  | "invalid"
  | "quarantined"
  | "unverified";

export type SeriesCard = {
  key: string;
  recordKind: "canonical_series" | "unverified_map_group";
  date: string;
  league: string;
  patch: string;
  tournament: string;
  teamA: string;
  teamB: string;
  winsA: number | null;
  winsB: number | null;
  knownOutcomeMaps: number;
  bestOf: 1 | 3 | 5 | null;
  games: SeriesRow[];
  year: number;
  source: "oe" | "grid" | "mixed" | "unknown";
  completionSource: string | null;
  status: CanonicalSeriesStatus | "conflicting";
};

const CANONICAL_SERIES_STATUSES = new Set<CanonicalSeriesStatus>([
  "completed",
  "incomplete",
  "invalid",
  "quarantined",
  "unverified",
]);

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

function canonicalSeriesKey(row: SeriesRow): string | null {
  const canonical = String(row.canonical_series_id ?? "").trim();
  return canonical || null;
}

export function canonicalSeriesStatus(value: unknown): CanonicalSeriesStatus | null {
  const status = String(value ?? "").trim().toLowerCase();
  return CANONICAL_SERIES_STATUSES.has(status as CanonicalSeriesStatus)
    ? (status as CanonicalSeriesStatus)
    : null;
}

function falseLike(value: unknown): boolean {
  return (
    value === false ||
    value === 0 ||
    value === "0" ||
    String(value ?? "").trim().toLowerCase() === "false"
  );
}

/**
 * Canonical rows fail closed when their status is absent, malformed, not
 * complete, or explicitly excluded from rating use. Legacy rows have no
 * canonical status contract and remain displayable only as unverified groups.
 */
export function isQuarantinedSeriesRow(row: SeriesRow): boolean {
  if (!canonicalSeriesKey(row)) return false;
  return (
    canonicalSeriesStatus(row.canonical_series_status) !== "completed" ||
    falseLike(row.series_rating_eligible)
  );
}

function parseBestOf(value: unknown): 1 | 3 | 5 | null {
  const text = String(value ?? "").trim();
  const match = text.match(/(?:bo|best\s*of)\s*([135])/i);
  const parsed = match ? Number(match[1]) : Number(text);
  return parsed === 1 || parsed === 3 || parsed === 5 ? parsed : null;
}

function verifiedBestOf(
  games: SeriesRow[],
  recordKind: SeriesCard["recordKind"],
): 1 | 3 | 5 | null {
  if (recordKind !== "canonical_series") return null;
  if (
    games.some(
      (game) =>
        game.series_format_registry_conflict === true ||
        Number(game.series_format_registry_conflict) === 1,
    )
  ) {
    return null;
  }
  const values = new Set(
    games
      .map((game) => parseBestOf(game.scheduled_best_of ?? game.series_format))
      .filter((value): value is 1 | 3 | 5 => value != null),
  );
  const hasCanonicalValidation = games.every((game) =>
    canonicalSeriesStatus(game.canonical_series_status) === "completed",
  );
  const hasRegistryValidation = games.every(
    (game) =>
      game.series_format_registry_verified === true ||
      Number(game.series_format_registry_verified) === 1,
  );
  return values.size === 1 && (hasCanonicalValidation || hasRegistryValidation)
    ? [...values][0]
    : null;
}

export function formatSeriesLabel(series: SeriesCard): string {
  if (series.recordKind === "unverified_map_group") return "Unverified map group";
  if (series.status === "incomplete") return "Incomplete series";
  if (series.status === "invalid") return "Invalid series";
  if (series.status === "quarantined") return "Quarantined series";
  if (series.status === "conflicting") return "Conflicting series status";
  if (series.status === "unverified") return "Series status unverified";
  if (series.bestOf) return `Bo${series.bestOf}`;
  return "Format unverified";
}

export function formatSeriesScore(series: SeriesCard): string {
  return series.winsA == null || series.winsB == null
    ? "outcome unavailable"
    : `${series.winsA}–${series.winsB}`;
}

function binaryResult(value: unknown): 0 | 1 | null {
  if (
    value === true ||
    value === 1 ||
    value === "1" ||
    (typeof value === "bigint" && String(value) === "1")
  ) return 1;
  if (
    value === false ||
    value === 0 ||
    value === "0" ||
    (typeof value === "bigint" && String(value) === "0")
  ) return 0;
  return null;
}

function mapWinnerSide(row: SeriesRow): "blue" | "red" | null {
  const candidates = new Set<"blue" | "red">();
  const blue = binaryResult(row.blue_result);
  const red = binaryResult(row.red_result);
  const yBlue = binaryResult(row.y_blue_win);
  if (blue === 1) candidates.add("blue");
  if (blue === 0) candidates.add("red");
  if (red === 1) candidates.add("red");
  if (red === 0) candidates.add("blue");
  if (yBlue === 1) candidates.add("blue");
  if (yBlue === 0) candidates.add("red");
  return candidates.size === 1 ? [...candidates][0] : null;
}

/** Group canonical series, or legacy rows as explicit team-pair/day map groups. */
export function groupMapsIntoSeries(rows: SeriesRow[]): SeriesCard[] {
  const buckets = new Map<string, SeriesRow[]>();
  for (const row of rows) {
    const blue = String(row.blue_teamname ?? "");
    const red = String(row.red_teamname ?? "");
    if (!blue || !red) continue;
    const day = formatGameDate(row.date);
    const league = String(row.league ?? "");
    const tournament = String(row.tournament ?? "");
    const seriesId = canonicalSeriesKey(row);
    const key = seriesId
      ? `canonical:${seriesId}|${league}|${tournament}`
      : `unverified:${teamPairKey(blue, red)}|${day}|${league}|${tournament}`;
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
    const recordKind: SeriesCard["recordKind"] = canonicalSeriesKey(first)
      ? "canonical_series"
      : "unverified_map_group";
    // Stable A/B by alphabetical for scoreline display.
    const [teamA, teamB] = [blue, red].sort((x, y) => x.localeCompare(y));
    let winsA = 0;
    let winsB = 0;
    let knownOutcomeMaps = 0;
    for (const game of games) {
      const blueName = String(game.blue_teamname);
      const winnerSide = mapWinnerSide(game);
      if (!winnerSide) continue;
      knownOutcomeMaps += 1;
      const winner =
        winnerSide === "blue" ? blueName : String(game.red_teamname);
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
    const statuses = [
      ...new Set(
        games
          .map((game) => canonicalSeriesStatus(game.canonical_series_status))
          .filter((status): status is CanonicalSeriesStatus => status != null),
      ),
    ];
    const canonicalStatusMissing =
      recordKind === "canonical_series" &&
      games.some((game) => canonicalSeriesStatus(game.canonical_series_status) == null);
    const status: SeriesCard["status"] =
      recordKind === "unverified_map_group" || canonicalStatusMissing
        ? "unverified"
        : statuses.length === 1
          ? statuses[0]
          : statuses.length > 1
            ? "conflicting"
            : "unverified";
    series.push({
      key,
      recordKind,
      date: formatGameDate(first.date),
      league: String(first.league ?? ""),
      patch: String(first.patch ?? ""),
      tournament: String(first.tournament ?? ""),
      teamA,
      teamB,
      winsA: knownOutcomeMaps === games.length ? winsA : null,
      winsB: knownOutcomeMaps === games.length ? winsB : null,
      knownOutcomeMaps,
      bestOf: verifiedBestOf(games, recordKind),
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
      status,
    });
  }
  series.sort((a, b) => b.date.localeCompare(a.date) || a.key.localeCompare(b.key));
  return series;
}
