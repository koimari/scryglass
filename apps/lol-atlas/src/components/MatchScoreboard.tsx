"use client";

import {
  champIconUrl,
  formatClock,
  formatGold,
  playerCs,
  sortPlayersByRole,
  type QueryRow,
} from "@/lib/duck";

type Props = {
  map: QueryRow;
  players: QueryRow[];
  draftPctBlue?: number | null;
};

function bans(prefix: "blue" | "red", map: QueryRow): string[] {
  return [1, 2, 3, 4, 5]
    .map((i) => map[`${prefix}_ban${i}`])
    .filter((x) => x != null && String(x).length > 0)
    .map(String);
}

function ChampThumb({ name, size = 40 }: { name: string; size?: number }) {
  const src = champIconUrl(name);
  if (!src) {
    return (
      <span className="champ-fallback" style={{ width: size, height: size }}>
        {name.slice(0, 2)}
      </span>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={name}
      width={size}
      height={size}
      className="champ-thumb"
      loading="lazy"
    />
  );
}

function BanStrip({ champs, side }: { champs: string[]; side: "blue" | "red" }) {
  return (
    <div className={`ban-strip ban-${side}`}>
      {champs.map((c) => (
        <ChampThumb key={c} name={c} size={28} />
      ))}
    </div>
  );
}

function fmtObj(n: unknown): string {
  if (n == null || n === "") return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return String(n);
  return String(Math.round(v));
}

function ObjCell({
  label,
  blue,
  red,
  title,
}: {
  label: string;
  blue: unknown;
  red: unknown;
  title: string;
}) {
  return (
    <div className="obj-cell" aria-label={`${title}, blue ${fmtObj(blue)}, red ${fmtObj(red)}`}>
      <span className="obj-blue font-mono">{fmtObj(blue)}</span>
      <span className="obj-label">{label}</span>
      <span className="obj-red font-mono">{fmtObj(red)}</span>
    </div>
  );
}

function PlayerHalf({
  row,
  mirror,
}: {
  row: QueryRow;
  mirror: boolean;
}) {
  const cs = playerCs(row);
  const gold = formatGold(row.totalgold);
  const kda = `${row.kills ?? 0}/${row.deaths ?? 0}/${row.assists ?? 0}`;
  const name = String(row.playername ?? "—");
  const champ = String(row.champion ?? "");

  if (mirror) {
    return (
      <div className="sb-player sb-player-red">
        <div className="sb-metrics">
          <span className="font-mono">{gold}</span>
          <span className="font-mono muted">{cs != null ? cs : "—"}</span>
          <span className="font-mono">{kda}</span>
        </div>
        <div className="sb-identity">
          <span className="sb-ign">{name}</span>
          <ChampThumb name={champ} size={44} />
        </div>
      </div>
    );
  }

  return (
    <div className="sb-player sb-player-blue">
      <div className="sb-identity">
        <ChampThumb name={champ} size={44} />
        <span className="sb-ign">{name}</span>
      </div>
      <div className="sb-metrics">
        <span className="font-mono">{kda}</span>
        <span className="font-mono muted">{cs != null ? cs : "—"}</span>
        <span className="font-mono">{gold}</span>
      </div>
    </div>
  );
}

export function MatchScoreboard({ map, players, draftPctBlue }: Props) {
  const blueName = String(map.blue_teamname ?? "Blue");
  const redName = String(map.red_teamname ?? "Red");
  const blueWin = Number(map.blue_result) === 1;
  const redWin = Number(map.red_result) === 1;
  const clock = formatClock(map.gamelength, map.length_min);
  const blueGold = formatGold(map.blue_totalgold);
  const redGold = formatGold(map.red_totalgold);
  const blueKills = fmtObj(map.blue_teamkills);
  const redKills = fmtObj(map.red_teamkills);

  const bySide = {
    Blue: sortPlayersByRole(players.filter((p) => String(p.side) === "Blue")),
    Red: sortPlayersByRole(players.filter((p) => String(p.side) === "Red")),
  };

  const roles = ["top", "jng", "mid", "bot", "sup"] as const;
  const pairs = roles.map((role) => {
    const blue = bySide.Blue.find((p) => String(p.position).toLowerCase() === role);
    const red = bySide.Red.find((p) => String(p.position).toLowerCase() === role);
    return { role, blue, red };
  });

  const blueBans = bans("blue", map);
  const redBans = bans("red", map);

  return (
    <article className="scoreboard anim-fade-up" aria-label={`${blueName} vs ${redName}`}>
      <header className="sb-header">
        <div className={`sb-team sb-team-blue ${blueWin ? "is-winner" : ""}`}>
          <span className="sb-result">{blueWin ? "Victory" : "Defeat"}</span>
          <span className="sb-teamname">{blueName}</span>
          <div className="sb-team-stats">
            <span className="font-mono">{blueGold} gold</span>
          </div>
        </div>
        <div className="sb-center">
          <span className="sb-score font-mono">
            {blueKills} – {redKills}
          </span>
          <span className="sb-clock font-mono">{clock}</span>
          {draftPctBlue != null && Number.isFinite(draftPctBlue) && (
            <span className="sb-draft font-mono" title="League-calibrated Draft Score → WR">
              Draft {(100 * draftPctBlue).toFixed(0)}–
              {(100 * (1 - draftPctBlue)).toFixed(0)}%
            </span>
          )}
        </div>
        <div className={`sb-team sb-team-red ${redWin ? "is-winner" : ""}`}>
          <span className="sb-result">{redWin ? "Victory" : "Defeat"}</span>
          <span className="sb-teamname">{redName}</span>
          <div className="sb-team-stats">
            <span className="font-mono">{redGold} gold</span>
          </div>
        </div>
      </header>

      <div className="sb-col-legend" aria-hidden>
        <div className="sb-legend-half">
          <span>KDA</span>
          <span>CS</span>
          <span>Gold</span>
        </div>
        <div className="sb-role-gap" />
        <div className="sb-legend-half sb-legend-red">
          <span>Gold</span>
          <span>CS</span>
          <span>KDA</span>
        </div>
      </div>

      <div className="sb-grid">
        {pairs.map(({ role, blue, red }) => (
          <div className="sb-row" key={role}>
            {blue ? <PlayerHalf row={blue} mirror={false} /> : <div className="sb-player empty" />}
            <div className="sb-role-gap">{role}</div>
            {red ? <PlayerHalf row={red} mirror /> : <div className="sb-player empty" />}
          </div>
        ))}
      </div>

      <footer className="sb-footer">
        <BanStrip champs={blueBans} side="blue" />
        <div className="sb-objectives">
          <ObjCell label="TWR" title="Towers" blue={map.blue_towers} red={map.red_towers} />
          <ObjCell label="HLD" title="Herald" blue={map.blue_heralds} red={map.red_heralds} />
          <ObjCell label="GRB" title="Void grubs" blue={map.blue_void_grubs} red={map.red_void_grubs} />
          <ObjCell label="DRG" title="Dragons" blue={map.blue_dragons} red={map.red_dragons} />
          <ObjCell label="BAR" title="Barons" blue={map.blue_barons} red={map.red_barons} />
          <ObjCell label="INH" title="Inhibitors" blue={map.blue_inhibitors} red={map.red_inhibitors} />
        </div>
        <BanStrip champs={redBans} side="red" />
      </footer>

      <div className="sb-meta font-mono">
        <span>{String(map.date ?? "").slice(0, 16)}</span>
        <span>{String(map.league ?? "")}</span>
        <span>Patch {String(map.patch ?? "—")}</span>
        <span className="muted">{String(map.oe_gameid ?? map.game_uid ?? "")}</span>
      </div>
    </article>
  );
}
