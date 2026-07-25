"use client";

import * as duckdb from "@duckdb/duckdb-wasm";

let dbPromise: Promise<duckdb.AsyncDuckDB> | null = null;

async function getDb(): Promise<duckdb.AsyncDuckDB> {
  if (!dbPromise) {
    dbPromise = (async () => {
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
  game,
  blue_teamname,
  red_teamname,
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
  length_min,
  gamelength,
  blue_ban1, blue_ban2, blue_ban3, blue_ban4, blue_ban5,
  red_ban1, red_ban2, red_ban3, red_ban4, red_ban5,
  blue_pick1, blue_pick2, blue_pick3, blue_pick4, blue_pick5,
  red_pick1, red_pick2, red_pick3, red_pick4, red_pick5
`;

export type MapFilters = {
  league?: string;
  team?: string;
  teamA?: string;
  teamB?: string;
  limit?: number;
};

export async function queryMaps(
  baseUrl: string,
  year: number,
  opts: MapFilters = {},
): Promise<QueryRow[]> {
  const url = mapsUrl(baseUrl, year);
  const lim = opts.limit ?? 80;
  const clauses: string[] = ["1=1"];
  if (opts.league) clauses.push(`league = '${esc(opts.league)}'`);
  if (opts.team) {
    const t = esc(opts.team);
    clauses.push(`(blue_teamname ILIKE '%${t}%' OR red_teamname ILIKE '%${t}%')`);
  }
  if (opts.teamA && opts.teamB) {
    const a = esc(opts.teamA);
    const b = esc(opts.teamB);
    clauses.push(`(
      (blue_teamname ILIKE '%${a}%' AND red_teamname ILIKE '%${b}%')
      OR (blue_teamname ILIKE '%${b}%' AND red_teamname ILIKE '%${a}%')
    )`);
  }
  const sql = `
    SELECT ${MAP_SELECT}
    FROM read_parquet($PARQUET)
    WHERE ${clauses.join(" AND ")}
    ORDER BY date DESC
    LIMIT ${lim}
  `;
  return queryPackParquet(url, sql);
}

export async function queryMapByGameId(
  baseUrl: string,
  year: number,
  gameId: string,
): Promise<QueryRow | null> {
  const url = mapsUrl(baseUrl, year);
  const rows = await queryPackParquet(
    url,
    `SELECT ${MAP_SELECT}
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

export async function queryPlayerChampStats(
  baseUrl: string,
  years: number[],
  playername: string,
  limit = 5,
): Promise<ChampAgg[]> {
  const urls = years.map((y) => playersUrl(baseUrl, y));
  // Query each year and merge in JS (simpler than multi-parquet union in WASM)
  const buckets = new Map<
    string,
    { n: number; k: number; d: number; a: number; gold: number; dpm: number; dpmN: number; cs: number; csN: number; wins: number }
  >();
  for (const url of urls) {
    const rows = await queryPackParquet(
      url,
      `SELECT champion, kills, deaths, assists, totalgold, dpm, minionkills, monsterkills, result
       FROM read_parquet($PARQUET)
       WHERE playername = '${esc(playername)}'
         AND champion IS NOT NULL`,
    );
    for (const r of rows) {
      const champ = String(r.champion);
      const b = buckets.get(champ) ?? {
        n: 0,
        k: 0,
        d: 0,
        a: 0,
        gold: 0,
        dpm: 0,
        dpmN: 0,
        cs: 0,
        csN: 0,
        wins: 0,
      };
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
  }
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

/** Data Dragon champion square icon URL from OE champion name. */
export function champIconUrl(name: string | null | undefined): string | null {
  if (!name) return null;
  const key = String(name)
    .replace(/['.]/g, "")
    .replace(/\s+/g, "")
    .replace(/&/g, "");
  // Known OE / wiki mismatches
  const aliases: Record<string, string> = {
    Wukong: "MonkeyKing",
    Nunu: "Nunu",
    "Nunu&Willump": "Nunu",
    RenataGlasc: "Renata",
    BelVeth: "Belveth",
    Kaisa: "Kaisa",
    Khazix: "Khazix",
    Chogath: "Chogath",
    DrMundo: "DrMundo",
    JarvanIV: "JarvanIV",
    LeeSin: "LeeSin",
    MasterYi: "MasterYi",
    MissFortune: "MissFortune",
    TwistedFate: "TwistedFate",
    XinZhao: "XinZhao",
    AurelionSol: "AurelionSol",
  };
  const id = aliases[key] ?? key;
  return `https://ddragon.leagueoflegends.com/cdn/15.14.1/img/champion/${id}.png`;
}

export function formatGold(n: unknown): string {
  if (n == null || n === "") return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return String(Math.round(v));
}

export function formatClock(gamelengthSec: unknown, lengthMin?: unknown): string {
  if (lengthMin != null && Number.isFinite(Number(lengthMin))) {
    const m = Number(lengthMin);
    const mins = Math.floor(m);
    const secs = Math.round((m - mins) * 60);
    return `${mins}:${String(secs).padStart(2, "0")}`;
  }
  if (gamelengthSec == null) return "—";
  const s = Number(gamelengthSec);
  if (!Number.isFinite(s)) return "—";
  const mins = Math.floor(s / 60);
  const secs = Math.round(s % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

export function playerCs(row: QueryRow): number | null {
  const min = row.minionkills != null ? Number(row.minionkills) : null;
  const mon = row.monsterkills != null ? Number(row.monsterkills) : null;
  if (min != null && mon != null && Number.isFinite(min) && Number.isFinite(mon)) {
    return min + mon;
  }
  if (row.cspm != null && row.gamelength != null) {
    const cspm = Number(row.cspm);
    const gl = Number(row.gamelength);
    if (Number.isFinite(cspm) && Number.isFinite(gl) && gl > 0) {
      return Math.round((cspm * gl) / 60);
    }
  }
  return null;
}

const ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"] as const;

export function sortPlayersByRole(players: QueryRow[]): QueryRow[] {
  const rank = (p: string) => {
    const i = ROLE_ORDER.indexOf(p.toLowerCase() as (typeof ROLE_ORDER)[number]);
    return i === -1 ? 99 : i;
  };
  return [...players].sort((a, b) => rank(String(a.position ?? "")) - rank(String(b.position ?? "")));
}
