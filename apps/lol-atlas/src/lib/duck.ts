"use client";

import type { AsyncDuckDB } from "@duckdb/duckdb-wasm";

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
export { formatCompletionSource, formatGameDate, groupMapsIntoSeries } from "./series";
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
      "grid_series_id",
      "grid_game_index",
      "grid_completion_source",
    ]) {
      if (available.has(column)) optionalColumns.push(column);
    }
  } catch {
    // The main map query will report the actual pack failure if its stable
    // columns are unavailable; optional provenance must not cause one.
  }
  return optionalColumns.length ? `${MAP_SELECT.trimEnd()}\n  ,${optionalColumns.join(",\n  ")}` : MAP_SELECT;
}

function mapFilterClauses(opts: MapFilters): string[] {
  const clauses: string[] = ["1=1"];
  if (opts.league) clauses.push(`league = '${esc(opts.league)}'`);
  if (opts.leagues?.length) {
    const list = opts.leagues.map((L) => `'${esc(L)}'`).join(", ");
    clauses.push(`league IN (${list})`);
  }
  if (opts.patch) clauses.push(`CAST(patch AS VARCHAR) ILIKE '${esc(opts.patch)}%'`);
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
  if (opts.side === "blue" && opts.teams?.[0]) {
    clauses.push(`blue_teamname ILIKE '%${esc(opts.teams[0])}%'`);
  }
  if (opts.side === "red" && opts.teams?.[0]) {
    clauses.push(`red_teamname ILIKE '%${esc(opts.teams[0])}%'`);
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
  return queryPackParquet(url, sql);
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

export type ModelAccuracySummary = {
  n: number;
  eloHits: number;
  eloRate: number | null;
  draftOverlapHits: number;
  draftOverlapN: number;
  draftOverlapRate: number | null;
};

/** Elo favorite accuracy from ratings_history for a year sample. */
export async function queryEloAccuracy(
  baseUrl: string,
  year: number,
): Promise<{ n: number; hits: number; rate: number | null }> {
  const histUrl = `${baseUrl.replace(/\/$/, "")}/features/ratings_history.parquet`;
  const mapsU = mapsUrl(baseUrl, year);
  try {
    const rows = await queryPackSql(
      `
      SELECT
        COUNT(*)::INT AS n,
        SUM(CASE
          WHEN h.p_dual_elo >= 0.5 AND m.y_blue_win >= 0.5 THEN 1
          WHEN h.p_dual_elo < 0.5 AND m.y_blue_win < 0.5 THEN 1
          ELSE 0
        END)::INT AS hits
      FROM read_parquet($HIST) h
      JOIN read_parquet($MAPS) m ON h.game_uid = m.game_uid
      WHERE h.p_dual_elo IS NOT NULL AND m.y_blue_win IS NOT NULL
      `,
      { HIST: histUrl, MAPS: mapsU },
    );
    const n = Number(rows[0]?.n ?? 0);
    const hits = Number(rows[0]?.hits ?? 0);
    return { n, hits, rate: n ? hits / n : null };
  } catch {
    return { n: 0, hits: 0, rate: null };
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
    const players = await queryPlayersForGame(baseUrl, year, String(map.oe_gameid));
    return { year, map, players };
  }
  return null;
}

export type MatchModelPrior = {
  pBlueWin: number | null;
  expectedFavorite: string | null;
  expectedKills: number | null;
  killsLine: number | null;
  muDiff: number | null;
  sourceNote: string;
};

/** Dual Elo win prior from ratings_history + league mean kills as kill prior. */
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
  try {
    const rows = await queryPackParquet(
      histUrl,
      `SELECT p_dual_elo, mu_diff, mu_blue, mu_red
       FROM read_parquet($PARQUET)
       WHERE game_uid = '${esc(gameId)}'
       LIMIT 1`,
    );
    if (rows[0]?.p_dual_elo != null) pBlueWin = Number(rows[0].p_dual_elo);
    if (rows[0]?.mu_diff != null) muDiff = Number(rows[0].mu_diff);
  } catch {
    pBlueWin = null;
  }

  let expectedKills: number | null = null;
  try {
    const clause = league ? `WHERE league = '${esc(league)}'` : "";
    const rows = await queryPackParquet(
      mapsU,
      `SELECT avg(total_kills) AS mu_kills
       FROM read_parquet($PARQUET)
       ${clause}`,
    );
    if (rows[0]?.mu_kills != null) expectedKills = Number(rows[0].mu_kills);
  } catch {
    expectedKills = null;
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
    killsLine,
    muDiff,
    sourceNote:
      pBlueWin != null
        ? "Winner: Dual Elo pre-match p. Kills: league mean total kills in pack year. Draft WR: league-calibrated Draft Score."
        : "Dual Elo history miss for this game_uid. Kills: league mean if available. Draft WR from picks when available.",
  };
}

export type ChampAgg = {
  champion: string;
  n: number;
  kills: number;
  deaths: number;
  assists: number;
  gold: number;
  dpm: number | null;
  cs: number | null;
  wr: number;
};

type ChampBucket = {
  n: number;
  k: number;
  d: number;
  a: number;
  gold: number;
  dpm: number;
  dpmN: number;
  cs: number;
  csN: number;
  wins: number;
};

function emptyBucket(): ChampBucket {
  return { n: 0, k: 0, d: 0, a: 0, gold: 0, dpm: 0, dpmN: 0, cs: 0, csN: 0, wins: 0 };
}

function ingestChampRow(buckets: Map<string, ChampBucket>, r: QueryRow) {
  const champ = String(r.champion);
  const b = buckets.get(champ) ?? emptyBucket();
  b.n += 1;
  b.k += Number(r.kills) || 0;
  b.d += Number(r.deaths) || 0;
  b.a += Number(r.assists) || 0;
  b.gold += Number(r.totalgold) || 0;
  if (r.dpm != null && Number.isFinite(Number(r.dpm))) {
    b.dpm += Number(r.dpm);
    b.dpmN += 1;
  }
  const cs =
    r.minionkills != null || r.monsterkills != null
      ? (Number(r.minionkills) || 0) + (Number(r.monsterkills) || 0)
      : null;
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
      kills: b.k / b.n,
      deaths: b.d / b.n,
      assists: b.a / b.n,
      gold: b.gold / b.n,
      dpm: b.dpmN ? b.dpm / b.dpmN : null,
      cs: b.csN ? b.cs / b.csN : null,
      wr: b.wins / b.n,
    }))
    .sort((a, b) => b.n - a.n || b.wr - a.wr)
    .slice(0, limit);
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
      `SELECT playername, champion, kills, deaths, assists, totalgold, dpm, minionkills, monsterkills, result
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
