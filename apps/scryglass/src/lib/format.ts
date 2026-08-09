/** Pure display helpers — keep out of DuckDB-WASM import graph. */

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
