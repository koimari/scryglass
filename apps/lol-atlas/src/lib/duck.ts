"use client";

import type { AsyncDuckDB } from "@duckdb/duckdb-wasm";
import { normalizePatchVersion, playerCs } from "./format";
import { parseBinaryResult } from "./binaryResult";

let dbPromise: Promise<AsyncDuckDB> | null = null;

async function getDb(): Promise<AsyncDuckDB> {
  if (!dbPromise) {
    dbPromise = (async () => {
      const duckdb = await import("@duckdb/duckdb-wasm");
      const bundles = duckdb.getJsDelivrBundles();
      const bundle = await duckdb.selectBundle(bundles);
      const workerUrl = URL.createObjectURL(
        new Blob([`importScripts("${bundle.mainWorker!}");`], {
          type: "text/javascript",
        }),
      );
      const worker = new Worker(workerUrl);
      const logger = new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING);
      const db = new duckdb.AsyncDuckDB(logger, worker);
      await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
      URL.revokeObjectURL(workerUrl);
      return db;
    })();
  }
  return dbPromise;
}

/** DuckDB-WASM needs an absolute http(s) URL, not a site-relative path. */
export function absolutePackUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  if (typeof window === "undefined") return pathOrUrl;
  const path = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return new URL(path, window.location.origin).href;
}

function esc(s: string): string {
  return s.replace(/'/g, "''");
}

export type QueryRow = Record<string, unknown>;
export type MapWinnerSide = "blue" | "red";

export function finiteNumberOrNull(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function packTimestampIso(value: unknown): string | null {
  if (value instanceof Date) {
    return Number.isFinite(value.getTime()) ? value.toISOString() : null;
  }
  // Apache Arrow's Timestamp.toJSON contract is Unix epoch milliseconds.
  // This is an explicit interchange contract, not unit inference by magnitude.
  if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)) {
    const parsed = new Date(value);
    return Number.isFinite(parsed.getTime()) ? parsed.toISOString() : null;
  }
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (!raw) return null;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : null;
}

function binaryResult(value: unknown): 0 | 1 | null {
  return parseBinaryResult(value);
}

/**
 * Resolve a completed map outcome without turning absent or contradictory
 * result fields into a winner.
 */
export function resolveMapWinnerSide(
  map: QueryRow,
  players: QueryRow[] = [],
): MapWinnerSide | null {
  const playerResults = {
    blue: new Set<0 | 1>(),
    red: new Set<0 | 1>(),
  };
  for (const player of players) {
    const result = binaryResult(player.result);
    const side = String(player.side ?? "").toLowerCase();
    if (result != null && (side === "blue" || side === "red")) {
      playerResults[side].add(result);
    }
  }
  const playerWinner =
    playerResults.blue.size === 1 &&
    playerResults.red.size === 1 &&
    [...playerResults.blue][0] !== [...playerResults.red][0]
      ? [...playerResults.blue][0] === 1
        ? "blue"
        : "red"
      : null;

  const blueResult = binaryResult(map.blue_result);
  const redResult = binaryResult(map.red_result);
  const yBlueWin = binaryResult(map.y_blue_win);
  const mapCandidates = new Set<MapWinnerSide>();
  if (blueResult === 1) mapCandidates.add("blue");
  if (blueResult === 0) mapCandidates.add("red");
  if (redResult === 1) mapCandidates.add("red");
  if (redResult === 0) mapCandidates.add("blue");
  if (yBlueWin === 1) mapCandidates.add("blue");
  if (yBlueWin === 0) mapCandidates.add("red");
  if (mapCandidates.size > 1) return null;

  const mapWinner = mapCandidates.size === 1 ? [...mapCandidates][0] : null;
  if (playerWinner && mapWinner && playerWinner !== mapWinner) return null;
  return mapWinner ?? playerWinner;
}

export function sumKnownNumbers(values: unknown[]): number | null {
  const parsed = values.map(finiteNumberOrNull);
  return parsed.every((value): value is number => value != null)
    ? parsed.reduce((total, value) => total + value, 0)
    : null;
}

export function normalizeMapQueryRow(row: QueryRow): QueryRow {
  return {
    ...row,
    grid_game_index: row.canonical_game_index ?? row.grid_game_index,
    grid_completion_source:
      row.canonical_series_completion_source ?? row.grid_completion_source,
  };
}

export {
  canonicalSeriesStatus,
  formatCompletionSource,
  formatGameDate,
  formatSeriesLabel,
  formatSeriesScore,
  groupMapsIntoSeries,
  isQuarantinedSeriesRow,
} from "./series";
export type { SeriesCard } from "./series";

export async function queryPackParquet(
  parquetUrl: string,
  sql: string,
): Promise<QueryRow[]> {
  const db = await getDb();
  const conn = await db.connect();
  try {
    const abs = absolutePackUrl(parquetUrl);
    const wrapped = sql.includes("$PARQUET")
      ? sql.replaceAll("$PARQUET", `'${abs.replace(/'/g, "''")}'`)
      : sql;
    const result = await conn.query(wrapped);
    return result.toArray().map((row) => row.toJSON() as QueryRow);
  } finally {
    await conn.close();
  }
}

/** Multi-file query: replace $MAPS and $PLAYERS (or any $KEY) with absolute parquet URLs. */
export async function queryPackSql(
  sql: string,
  urls: Record<string, string>,
): Promise<QueryRow[]> {
  const db = await getDb();
  const conn = await db.connect();
  try {
    let wrapped = sql;
    for (const [key, url] of Object.entries(urls)) {
      const abs = absolutePackUrl(url).replace(/'/g, "''");
      wrapped = wrapped.replaceAll(`$${key}`, `'${abs}'`);
    }
    const result = await conn.query(wrapped);
    return result.toArray().map((row) => row.toJSON() as QueryRow);
  } finally {
    await conn.close();
  }
}

function mapsUrl(baseUrl: string, year: number): string {
  return `${baseUrl.replace(/\/$/, "")}/maps/year=${year}/part.parquet`;
}

function playersUrl(baseUrl: string, year: number): string {
  return `${baseUrl.replace(/\/$/, "")}/player_games/year=${year}/part.parquet`;
}

function teamGamesUrl(baseUrl: string, year: number): string {
  return `${baseUrl.replace(/\/$/, "")}/team_games/year=${year}/part.parquet`;
}

const MAP_SELECT = `
  oe_gameid,
  game_uid,
  date,
  league,
  patch,
  split,
  playoffs,
  tournament,
  game,
  blue_teamname,
  red_teamname,
  total_kills,
  y_blue_win,
  blue_result,
  red_result,
  blue_teamkills,
  red_teamkills,
  blue_totalgold,
  red_totalgold,
  blue_dragons,
  red_dragons,
  blue_void_grubs,
  red_void_grubs,
  blue_towers,
  red_towers,
  blue_heralds,
  red_heralds,
  blue_barons,
  red_barons,
  blue_inhibitors,
  red_inhibitors,
  blue_golddiffat15,
  red_golddiffat15,
  source_oe,
  length_min,
  gamelength,
  blue_ban1, blue_ban2, blue_ban3, blue_ban4, blue_ban5,
  red_ban1, red_ban2, red_ban3, red_ban4, red_ban5,
  blue_pick1, blue_pick2, blue_pick3, blue_pick4, blue_pick5,
  red_pick1, red_pick2, red_pick3, red_pick4, red_pick5
`;

export type MapFilters = {
  league?: string;
  leagues?: string[];
  team?: string;
  teams?: string[];
  teamA?: string;
  teamB?: string;
  patch?: string;
  side?: "blue" | "red";
  limit?: number;
};

async function mapSelectForPack(parquetUrl: string): Promise<string> {
  // Older public packs contain source_oe but predate source_grid. Keep the
  // browser query compatible with both schemas while newer packs roll out.
  const optionalColumns: string[] = [];
  try {
    const columns = await queryPackParquet(
      parquetUrl,
      "DESCRIBE SELECT * FROM read_parquet($PARQUET)",
    );
    const available = new Set(columns.map((row) => String(row.column_name ?? "")));
    for (const column of [
      "source_grid",
      "map_detail_source",
      "grid_series_id",
      "grid_game_id",
      "grid_game_index",
      "grid_completion_source",
      "series_format",
      "series_format_source",
      "series_format_stage_id",
      "series_format_registry_snapshot_id",
      "series_format_registry_verified",
      "series_format_registry_conflict",
      "canonical_series_id",
      "scheduled_best_of",
      "canonical_game_index",
      "raw_source_game_index",
      "raw_source_game_uid",
      "canonical_series_status",
      "canonical_series_completion_source",
      "series_rating_eligible",
      "canonical_series_winner_team_key",
      "series_quarantine_reasons",
    ]) {
      if (available.has(column)) optionalColumns.push(column);
    }
  } catch {
    // The main map query will report the actual pack failure if its stable
    // columns are unavailable; optional provenance must not cause one.
  }
  return optionalColumns.length ? `${MAP_SELECT.trimEnd()}\n  ,${optionalColumns.join(",\n  ")}` : MAP_SELECT;
}

export function patchFilterClause(value: unknown): string {
  const normalized = normalizePatchVersion(value);
  if (!normalized) return "FALSE";
  const [major, minor] = normalized.split(".").map(Number);
  return `(
    TRY_CAST(split_part(TRIM(CAST(patch AS VARCHAR)), '.', 1) AS INTEGER) = ${major}
    AND TRY_CAST(split_part(TRIM(CAST(patch AS VARCHAR)), '.', 2) AS INTEGER) = ${minor}
  )`;
}

export function mapFilterClauses(opts: MapFilters): string[] {
  const clauses: string[] = ["1=1"];
  if (opts.league) clauses.push(`league = '${esc(opts.league)}'`);
  if (opts.leagues?.length) {
    const list = opts.leagues.map((L) => `'${esc(L)}'`).join(", ");
    clauses.push(`league IN (${list})`);
  }
  if (opts.patch) clauses.push(patchFilterClause(opts.patch));
  if (opts.team) {
    const t = esc(opts.team);
    clauses.push(`(blue_teamname ILIKE '%${t}%' OR red_teamname ILIKE '%${t}%')`);
  }
  if (opts.teams?.length) {
    const parts = opts.teams.map((t) => {
      const e = esc(t);
      return `(blue_teamname ILIKE '%${e}%' OR red_teamname ILIKE '%${e}%')`;
    });
    clauses.push(`(${parts.join(" OR ")})`);
  }
  if (opts.teamA && opts.teamB) {
    const a = esc(opts.teamA);
    const b = esc(opts.teamB);
    clauses.push(`(
      (blue_teamname ILIKE '%${a}%' AND red_teamname ILIKE '%${b}%')
      OR (blue_teamname ILIKE '%${b}%' AND red_teamname ILIKE '%${a}%')
    )`);
  }
  if (opts.side === "blue" && opts.teams?.length) {
    clauses.push(
      `(${opts.teams
        .map((team) => `blue_teamname ILIKE '%${esc(team)}%'`)
        .join(" OR ")})`,
    );
  }
  if (opts.side === "red" && opts.teams?.length) {
    clauses.push(
      `(${opts.teams
        .map((team) => `red_teamname ILIKE '%${esc(team)}%'`)
        .join(" OR ")})`,
    );
  }
  return clauses;
}

export async function queryMaps(
  baseUrl: string,
  year: number,
  opts: MapFilters = {},
): Promise<QueryRow[]> {
  const url = mapsUrl(baseUrl, year);
  const lim = opts.limit ?? 80;
  const clauses = mapFilterClauses(opts);
  const select = await mapSelectForPack(url);
  const sql = `
    SELECT ${select}
    FROM read_parquet($PARQUET)
    WHERE ${clauses.join(" AND ")}
    ORDER BY date DESC
    LIMIT ${lim}
  `;
  const rows = await queryPackParquet(url, sql);
  return rows.map(normalizeMapQueryRow);
}

export async function queryMapsYears(
  baseUrl: string,
  years: number[],
  opts: MapFilters = {},
): Promise<QueryRow[]> {
  const lim = opts.limit ?? 200;
  const out: QueryRow[] = [];
  for (const year of [...years].sort((a, b) => b - a)) {
    const rows = await queryMaps(baseUrl, year, { ...opts, limit: lim });
    for (const r of rows) out.push({ ...r, _year: year });
    if (out.length >= lim) break;
  }
  return out.slice(0, lim);
}

export type FavoriteHitRateResult =
  | {
      status: "ok";
      n: number;
      hits: number;
      rate: number;
    }
  | {
      status: "sample_empty";
      n: 0;
      hits: 0;
      rate: null;
    }
  | {
      status: "error";
      n: 0;
      hits: 0;
      rate: null;
      code: "query_failed" | "integrity_failed";
    };

export function favoriteHitRateFromRow(row: QueryRow | undefined): FavoriteHitRateResult {
  if (!row) {
    return { status: "sample_empty", n: 0, hits: 0, rate: null };
  }
  const duplicateHistory = finiteNumberOrNull(row.duplicate_history_games) ?? 0;
  const duplicateMaps = finiteNumberOrNull(row.duplicate_map_games) ?? 0;
  const invalidProbabilities = finiteNumberOrNull(row.invalid_probabilities) ?? 0;
  const invalidOutcomes = finiteNumberOrNull(row.invalid_outcomes) ?? 0;
  if (duplicateHistory || duplicateMaps || invalidProbabilities || invalidOutcomes) {
    return {
      status: "error",
      n: 0,
      hits: 0,
      rate: null,
      code: "integrity_failed",
    };
  }
  const n = finiteNumberOrNull(row.n) ?? 0;
  const hits = finiteNumberOrNull(row.hits) ?? 0;
  if (n <= 0) return { status: "sample_empty", n: 0, hits: 0, rate: null };
  if (
    !Number.isInteger(n) ||
    !Number.isInteger(hits) ||
    hits < 0 ||
    hits > n
  ) {
    return {
      status: "error",
      n: 0,
      hits: 0,
      rate: null,
      code: "integrity_failed",
    };
  }
  return { status: "ok", n, hits, rate: hits / n };
}

/** Threshold favorite hit rate; this is not a proper probability score. */
export async function queryFavoriteHitRate(
  baseUrl: string,
  year: number,
): Promise<FavoriteHitRateResult> {
  const histUrl = `${baseUrl.replace(/\/$/, "")}/features/ratings_history.parquet`;
  const mapsU = mapsUrl(baseUrl, year);
  try {
    const rows = await queryPackSql(
      `
      WITH history AS (
        SELECT
          game_uid,
          COUNT(*)::INT AS row_count,
          MIN(p_dual_elo) AS p_dual_elo
        FROM read_parquet($HIST)
        GROUP BY game_uid
      ),
      maps AS (
        SELECT
          game_uid,
          COUNT(*)::INT AS row_count,
          MIN(y_blue_win) AS y_blue_win
        FROM read_parquet($MAPS)
        GROUP BY game_uid
      ),
      joined AS (
        SELECT
          h.row_count AS history_rows,
          m.row_count AS map_rows,
          h.p_dual_elo,
          m.y_blue_win
        FROM history h
        JOIN maps m USING (game_uid)
      )
      SELECT
        SUM(CASE
          WHEN history_rows = 1
            AND map_rows = 1
            AND p_dual_elo BETWEEN 0 AND 1
            AND y_blue_win IN (0, 1)
          THEN 1 ELSE 0
        END)::INT AS n,
        SUM(CASE
          WHEN history_rows = 1 AND map_rows = 1
            AND p_dual_elo >= 0.5 AND y_blue_win = 1 THEN 1
          WHEN history_rows = 1 AND map_rows = 1
            AND p_dual_elo < 0.5 AND y_blue_win = 0 THEN 1
          ELSE 0
        END)::INT AS hits,
        SUM(CASE WHEN history_rows <> 1 THEN 1 ELSE 0 END)::INT
          AS duplicate_history_games,
        SUM(CASE WHEN map_rows <> 1 THEN 1 ELSE 0 END)::INT
          AS duplicate_map_games,
        SUM(CASE
          WHEN p_dual_elo IS NOT NULL AND NOT (p_dual_elo BETWEEN 0 AND 1)
          THEN 1 ELSE 0 END
        )::INT AS invalid_probabilities,
        SUM(CASE
          WHEN y_blue_win IS NOT NULL AND y_blue_win NOT IN (0, 1)
          THEN 1 ELSE 0 END
        )::INT AS invalid_outcomes
      FROM joined
      WHERE p_dual_elo IS NOT NULL AND y_blue_win IS NOT NULL
      `,
      { HIST: histUrl, MAPS: mapsU },
    );
    return favoriteHitRateFromRow(rows[0]);
  } catch {
    return {
      status: "error",
      n: 0,
      hits: 0,
      rate: null,
      code: "query_failed",
    };
  }
}

export async function queryMapByGameId(
  baseUrl: string,
  year: number,
  gameId: string,
): Promise<QueryRow | null> {
  const url = mapsUrl(baseUrl, year);
  const select = await mapSelectForPack(url);
  const rows = await queryPackParquet(
    url,
    `SELECT ${select}
     FROM read_parquet($PARQUET)
     WHERE oe_gameid = '${esc(gameId)}' OR game_uid = '${esc(gameId)}'
     LIMIT 1`,
  );
  return rows[0] ?? null;
}

export async function queryPlayersForGame(
  baseUrl: string,
  year: number,
  gameId: string,
): Promise<QueryRow[]> {
  const url = playersUrl(baseUrl, year);
  const sql = `
    SELECT
      gameid, side, position, playername, teamname, champion,
      kills, deaths, assists, totalgold, result,
      minionkills, monsterkills, cspm, gamelength,
      ban1, ban2, ban3, ban4, ban5,
      damageshare, earnedgoldshare, visionscore
    FROM read_parquet($PARQUET)
    WHERE gameid = '${esc(gameId)}'
  `;
  return queryPackParquet(url, sql);
}

/** Load map + players; try years until found. */
export async function loadMatchBundle(
  baseUrl: string,
  years: number[],
  gameId: string,
): Promise<{ year: number; map: QueryRow; players: QueryRow[] } | null> {
  for (const year of [...years].sort((a, b) => b - a)) {
    const map = await queryMapByGameId(baseUrl, year, gameId);
    if (!map) continue;
    const playerGameId = String(map.oe_gameid ?? map.game_uid ?? gameId);
    const players = await queryPlayersForGame(baseUrl, year, playerGameId);
    return { year, map, players };
  }
  return null;
}

export type MatchModelPrior = {
  pBlueWin: number | null;
  expectedFavorite: string | null;
  expectedKills: number | null;
  expectedKillsN: number;
  expectedKillsCutoff: string | null;
  killsLine: number | null;
  muDiff: number | null;
  playerMuDiff: number | null;
  sourceNote: string;
};

/** Dual Elo prior plus a strictly earlier-date, pack-year league kill benchmark. */
export async function loadMatchModelPrior(
  baseUrl: string,
  year: number,
  map: QueryRow,
): Promise<MatchModelPrior> {
  const gameId = String(map.oe_gameid ?? map.game_uid ?? "");
  const league = String(map.league ?? "");
  const blue = String(map.blue_teamname ?? "Blue");
  const red = String(map.red_teamname ?? "Red");
  const histUrl = `${baseUrl.replace(/\/$/, "")}/features/ratings_history.parquet`;
  const mapsU = mapsUrl(baseUrl, year);

  let pBlueWin: number | null = null;
  let muDiff: number | null = null;
  let playerMuDiff: number | null = null;
  try {
    const rows = await queryPackParquet(
      histUrl,
      `SELECT p_dual_elo, mu_diff, mu_blue, mu_red
       FROM read_parquet($PARQUET)
       WHERE game_uid = '${esc(gameId)}'
       LIMIT 1`,
    );
    if (rows[0]?.p_dual_elo != null) {
      const probability = Number(rows[0].p_dual_elo);
      pBlueWin =
        Number.isFinite(probability) && probability >= 0 && probability <= 1
          ? probability
          : null;
    }
    if (rows[0]?.mu_diff != null) muDiff = Number(rows[0].mu_diff);
  } catch {
    pBlueWin = null;
  }
  try {
    const playerHistUrl = `${baseUrl.replace(/\/$/, "")}/features/player_ratings_history.parquet`;
    const rows = await queryPackParquet(
      playerHistUrl,
      `SELECT player_mu_diff
       FROM read_parquet($PARQUET)
       WHERE game_uid = '${esc(gameId)}'
       LIMIT 1`,
    );
    if (rows[0]?.player_mu_diff != null) playerMuDiff = Number(rows[0].player_mu_diff);
  } catch {
    playerMuDiff = null;
  }

  let expectedKills: number | null = null;
  let expectedKillsN = 0;
  const targetDate = packTimestampIso(map.date) ?? "";
  if (targetDate) {
    try {
      const leagueClause = league ? `AND league = '${esc(league)}'` : "";
      const escapedDate = esc(targetDate);
      const rows = await queryPackParquet(
        mapsU,
        `SELECT
           avg(total_kills) AS mu_kills,
           count(total_kills)::INT AS n_kills
         FROM read_parquet($PARQUET)
         WHERE TRY_CAST(date AS TIMESTAMPTZ) < TRY_CAST('${escapedDate}' AS TIMESTAMPTZ)
           ${leagueClause}
           AND total_kills IS NOT NULL
           AND total_kills >= 0`,
      );
      if (rows[0]?.mu_kills != null) expectedKills = Number(rows[0].mu_kills);
      expectedKillsN = Number(rows[0]?.n_kills ?? 0);
      if (!Number.isFinite(expectedKills ?? NaN) || expectedKillsN <= 0) {
        expectedKills = null;
        expectedKillsN = 0;
      }
    } catch (error) {
      console.warn("[match-prior] earlier-date kill benchmark failed", {
        gameId,
        league,
        targetDate,
        error: error instanceof Error ? error.message : String(error),
      });
      expectedKills = null;
      expectedKillsN = 0;
    }
  }

  const killsLine =
    expectedKills != null && Number.isFinite(expectedKills)
      ? Math.floor(expectedKills) + 0.5
      : null;

  let expectedFavorite: string | null = null;
  if (pBlueWin != null) {
    if (pBlueWin >= 0.5) expectedFavorite = blue;
    else expectedFavorite = red;
  }

  return {
    pBlueWin,
    expectedFavorite,
    expectedKills,
    expectedKillsN,
    expectedKillsCutoff: targetDate || null,
    killsLine,
    muDiff,
    playerMuDiff,
    sourceNote:
      pBlueWin != null
        ? "Winner: pre-match rating-history probability. Kills: descriptive pack-year league mean using only maps on earlier dates; it is a naive benchmark, not a fitted kills forecast. Draft forecast is withheld."
        : "Rating-history probability unavailable for this game. Kills use the same strictly earlier-date descriptive benchmark when enough rows exist. Draft forecast is withheld.",
  };
}

export type ChampAgg = {
  champion: string;
  n: number;
  kills: number | null;
  killsN: number;
  deaths: number | null;
  deathsN: number;
  assists: number | null;
  assistsN: number;
  gold: number | null;
  goldN: number;
  dpm: number | null;
  dpmN: number;
  cs: number | null;
  csN: number;
  wr: number;
};

type ChampBucket = {
  n: number;
  k: number;
  kN: number;
  d: number;
  dN: number;
  a: number;
  aN: number;
  gold: number;
  goldN: number;
  dpm: number;
  dpmN: number;
  cs: number;
  csN: number;
  wins: number;
};

function emptyBucket(): ChampBucket {
  return {
    n: 0,
    k: 0,
    kN: 0,
    d: 0,
    dN: 0,
    a: 0,
    aN: 0,
    gold: 0,
    goldN: 0,
    dpm: 0,
    dpmN: 0,
    cs: 0,
    csN: 0,
    wins: 0,
  };
}

function ingestChampRow(buckets: Map<string, ChampBucket>, r: QueryRow) {
  const champ = String(r.champion);
  const b = buckets.get(champ) ?? emptyBucket();
  b.n += 1;
  const kills = finiteNumberOrNull(r.kills);
  const deaths = finiteNumberOrNull(r.deaths);
  const assists = finiteNumberOrNull(r.assists);
  const gold = finiteNumberOrNull(r.totalgold);
  const dpm = finiteNumberOrNull(r.dpm);
  if (kills != null) {
    b.k += kills;
    b.kN += 1;
  }
  if (deaths != null) {
    b.d += deaths;
    b.dN += 1;
  }
  if (assists != null) {
    b.a += assists;
    b.aN += 1;
  }
  if (gold != null) {
    b.gold += gold;
    b.goldN += 1;
  }
  if (dpm != null) {
    b.dpm += dpm;
    b.dpmN += 1;
  }
  const cs = playerCs(r);
  if (cs != null) {
    b.cs += cs;
    b.csN += 1;
  }
  if (Number(r.result) === 1) b.wins += 1;
  buckets.set(champ, b);
}

function bucketsToAgg(buckets: Map<string, ChampBucket>, limit: number): ChampAgg[] {
  return [...buckets.entries()]
    .map(([champion, b]) => ({
      champion,
      n: b.n,
      kills: b.kN ? b.k / b.kN : null,
      killsN: b.kN,
      deaths: b.dN ? b.d / b.dN : null,
      deathsN: b.dN,
      assists: b.aN ? b.a / b.aN : null,
      assistsN: b.aN,
      gold: b.goldN ? b.gold / b.goldN : null,
      goldN: b.goldN,
      dpm: b.dpmN ? b.dpm / b.dpmN : null,
      dpmN: b.dpmN,
      cs: b.csN ? b.cs / b.csN : null,
      csN: b.csN,
      wr: b.wins / b.n,
    }))
    .sort((a, b) => b.n - a.n || b.wr - a.wr)
    .slice(0, limit);
}

export function aggregateChampionRows(
  rows: QueryRow[],
  limit = Number.POSITIVE_INFINITY,
): ChampAgg[] {
  const buckets = new Map<string, ChampBucket>();
  for (const row of rows) {
    const champion = String(row.champion ?? "").trim();
    if (!champion) continue;
    ingestChampRow(buckets, { ...row, champion });
  }
  return bucketsToAgg(buckets, limit);
}

export async function queryPlayerChampStats(
  baseUrl: string,
  years: number[],
  playername: string,
  limit = 5,
): Promise<ChampAgg[]> {
  const byPlayer = await queryRosterChampStats(baseUrl, years, [playername], limit);
  return byPlayer[playername] ?? [];
}

/** Most-played role for a player across the selected pack years. */
export async function queryPlayerRole(
  baseUrl: string,
  years: number[],
  playername: string,
): Promise<string | null> {
  const counts = new Map<string, number>();
  for (const year of years) {
    try {
      const rows = await queryPackParquet(
        playersUrl(baseUrl, year),
        `SELECT position, COUNT(*) AS n
         FROM read_parquet($PARQUET)
         WHERE playername = '${esc(playername)}'
           AND position IS NOT NULL
           AND position <> ''
         GROUP BY position`,
      );
      for (const row of rows) {
        const position = String(row.position ?? "").trim();
        if (position) counts.set(position, (counts.get(position) ?? 0) + Number(row.n ?? 0));
      }
    } catch {
      // A missing year partition should not block the rest of a player profile.
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

/** One scan per year for the whole roster (avoids N× full parquet passes). */
export async function queryRosterChampStats(
  baseUrl: string,
  years: number[],
  playernames: string[],
  limit = 5,
): Promise<Record<string, ChampAgg[]>> {
  const names = [...new Set(playernames.map((p) => p.trim()).filter(Boolean))];
  const out: Record<string, Map<string, ChampBucket>> = {};
  for (const n of names) out[n] = new Map();
  if (!names.length) return {};

  const inList = names.map((n) => `'${esc(n)}'`).join(", ");
  for (const year of years) {
    const url = playersUrl(baseUrl, year);
    const rows = await queryPackParquet(
      url,
      `SELECT playername, champion, kills, deaths, assists, totalgold, dpm,
              minionkills, monsterkills, cspm, gamelength, result
       FROM read_parquet($PARQUET)
       WHERE playername IN (${inList})
         AND champion IS NOT NULL`,
    );
    for (const r of rows) {
      const pn = String(r.playername);
      const buckets = out[pn];
      if (!buckets) continue;
      ingestChampRow(buckets, r);
    }
  }

  const result: Record<string, ChampAgg[]> = {};
  for (const n of names) {
    result[n] = bucketsToAgg(out[n], limit);
  }
  return result;
}

/** @deprecated Prefer features/major_teams.json — kept for scripts. */
export async function listMajorTeams(
  baseUrl: string,
  years: number[],
): Promise<Set<string>> {
  const majors = new Set([
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "MSI",
    "EWC",
    "WORLD",
    "Worlds",
    "First Stand",
  ]);
  const out = new Set<string>();
  for (const year of years) {
    const url = mapsUrl(baseUrl, year);
    const rows = await queryPackParquet(
      url,
      `SELECT DISTINCT league, blue_teamname, red_teamname FROM read_parquet($PARQUET)`,
    );
    for (const r of rows) {
      const L = String(r.league ?? "");
      if (![...majors].some((m) => L.toUpperCase().includes(m.toUpperCase()))) continue;
      if (r.blue_teamname) out.add(String(r.blue_teamname));
      if (r.red_teamname) out.add(String(r.red_teamname));
    }
  }
  return out;
}

export async function listLeagues(baseUrl: string, year: number): Promise<string[]> {
  const url = mapsUrl(baseUrl, year);
  const rows = await queryPackParquet(
    url,
    `SELECT DISTINCT league FROM read_parquet($PARQUET) ORDER BY 1`,
  );
  return rows.map((r) => String(r.league)).filter(Boolean);
}

export async function listTeams(baseUrl: string, years: number[]): Promise<string[]> {
  const names = new Set<string>();
  for (const year of years) {
    const rows = await queryPackParquet(
      mapsUrl(baseUrl, year),
      `SELECT DISTINCT blue_teamname, red_teamname FROM read_parquet($PARQUET)`,
    );
    for (const row of rows) {
      if (row.blue_teamname) names.add(String(row.blue_teamname));
      if (row.red_teamname) names.add(String(row.red_teamname));
    }
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

/** @deprecated Prefer queryMaps — kept for any leftover callers */
export async function queryTeamGames(
  baseUrl: string,
  year: number,
  opts: { league?: string; team?: string; limit?: number },
): Promise<QueryRow[]> {
  const url = teamGamesUrl(baseUrl, year);
  const lim = opts.limit ?? 50;
  const clauses: string[] = ["1=1"];
  if (opts.league) clauses.push(`league = '${esc(opts.league)}'`);
  if (opts.team) clauses.push(`teamname ILIKE '%${esc(opts.team)}%'`);
  const sql = `
    SELECT date, league, patch, teamname, side, result, teamkills, dragons, void_grubs, golddiffat15, gamelength
    FROM read_parquet($PARQUET)
    WHERE ${clauses.join(" AND ")}
    ORDER BY date DESC
    LIMIT ${lim}
  `;
  return queryPackParquet(url, sql);
}

export {
  champIconUrl,
  formatClock,
  formatGold,
  playerCs,
  sortPlayersByRole,
} from "@/lib/format";
