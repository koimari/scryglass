/** Pure display helpers — keep out of DuckDB-WASM import graph. */

import { normalizePatchOrBuild } from "./patch";

export function champIconUrl(name: string | null | undefined): string | null {
  if (!name) return null;
  const key = String(name)
    .replace(/['.]/g, "")
    .replace(/\s+/g, "")
    .replace(/&/g, "");
  const aliases: Record<string, string> = {
    Wukong: "MonkeyKing",
    Nunu: "Nunu",
    "Nunu&Willump": "Nunu",
    RenataGlasc: "Renata",
    BelVeth: "Belveth",
    KaiSa: "Kaisa",
    KhaZix: "Khazix",
    ChoGath: "Chogath",
    RekSai: "RekSai",
    KogMaw: "KogMaw",
    VelKoz: "Velkoz",
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
  return `https://ddragon.leagueoflegends.com/cdn/img/champion/tiles/${id}_0.jpg`;
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
    const totalSeconds = Math.round(Number(lengthMin) * 60);
    if (totalSeconds < 0) return "—";
    return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
  }
  if (gamelengthSec == null) return "—";
  const s = Number(gamelengthSec);
  if (!Number.isFinite(s) || s < 0) return "—";
  const totalSeconds = Math.round(s);
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

/**
 * Reduce a patch or full game build to its public major.minor patch identity.
 * No current-patch assumption is embedded here.
 */
export function normalizePatchVersion(value: unknown): string | null {
  return normalizePatchOrBuild(value);
}

export type FormatRow = Record<string, unknown>;

export function playerCs(row: FormatRow): number | null {
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

export function sortPlayersByRole(players: FormatRow[]): FormatRow[] {
  const rank = (p: string) => {
    const i = ROLE_ORDER.indexOf(p.toLowerCase() as (typeof ROLE_ORDER)[number]);
    return i === -1 ? 99 : i;
  };
  return [...players].sort(
    (a, b) => rank(String(a.position ?? "")) - rank(String(b.position ?? "")),
  );
}
