import Link from "next/link";
import { evidenceFields, evidenceInfo, formatEvidenceCell } from "@/lib/evidence";
import {
  adjustedRating,
  formatWr,
  packUpdatedLabel,
  PLAYER_SIGMA_MIN,
  playerSlug,
  softMu,
  TEAM_SIGMA_MIN,
  teamSlug,
  type PackManifest,
  type PlayerChampionRecord,
  type PlayerRating,
  type PlayerRecord,
  type TeamRating,
  type TeamRecord,
} from "@/lib/pack";
import styles from "./RatingProfiles.module.css";

function roleLabel(role: string | null | undefined): string {
  if (!role) return "—";
  return ({ jng: "Jungle", jungle: "Jungle", adc: "Bot", bot: "Bot", sup: "Support", support: "Support", top: "Top", mid: "Mid" } as Record<string, string>)[role.toLowerCase()] ?? role;
}

export function TeamRatingProfile({
  team,
  roster,
  record,
  playerRecords,
  manifest,
}: {
  team: TeamRating;
  roster: PlayerRating[];
  record?: TeamRecord;
  playerRecords: Record<string, PlayerRecord>;
  manifest: PackManifest;
}) {
  const trust = evidenceInfo(evidenceFields(team as unknown as Record<string, unknown>), team.sigma, record?.games);
  const players = [...roster].sort(
    (a, b) => softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) - softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
  );

  return (
    <div className={styles.page}>
      <p className={styles.back}><Link className="row-link" href="/elo">← Team ratings</Link></p>
      <header className={styles.header}>
        <div>
          <p className={styles.scope}>Team · {record?.current_league ?? record?.primary ?? "current pack"}</p>
          <h1>{team.team}</h1>
          <p className={styles.summary}>{trust.layman}</p>
        </div>
        <dl className={styles.metrics}>
          <div><dt>Adjusted rating</dt><dd>{adjustedRating(team, TEAM_SIGMA_MIN).toFixed(1)}</dd></div>
          <div><dt>Raw rating</dt><dd>{team.mu_total.toFixed(1)}</dd></div>
          <div><dt>Confidence</dt><dd>{formatEvidenceCell(trust)}</dd></div>
          <div><dt>Games</dt><dd>{record?.games ?? team.n_maps ?? "—"}</dd></div>
          <div><dt>Record</dt><dd>{record ? `${record.wins}–${record.games - record.wins}` : "—"}</dd></div>
          <div><dt>Win rate</dt><dd>{formatWr(record?.wr)}</dd></div>
          <div><dt>Updated</dt><dd>{packUpdatedLabel(manifest)}</dd></div>
        </dl>
      </header>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><h2>Recent player affiliations</h2><span>{players.length} listed</span></div>
        {players.length ? (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead><tr><th>Player</th><th>Role</th><th>Adjusted</th><th>Games</th><th>Win rate</th></tr></thead>
              <tbody>
                {players.map((player) => {
                  const playerRecord = playerRecords[player.player];
                  return (
                    <tr key={player.player}>
                      <td><Link className="row-link" href={`/elo/player/${playerSlug(player.player)}`}>{player.player}</Link></td>
                      <td>{roleLabel(playerRecord?.primary_role)}</td>
                      <td>{softMu(player.mu_total, player.sigma, PLAYER_SIGMA_MIN).toFixed(1)}</td>
                      <td>{playerRecord?.games ?? player.n_maps}</td>
                      <td>{formatWr(playerRecord?.wr)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <p className={styles.empty}>No recent player affiliation is available for this team.</p>}
      </section>
    </div>
  );
}

export function PlayerRatingProfile({
  player,
  champions,
  record,
  team,
  manifest,
}: {
  player: PlayerRating;
  champions: PlayerChampionRecord[];
  record?: PlayerRecord;
  team?: TeamRating | null;
  manifest: PackManifest;
}) {
  const trust = evidenceInfo(evidenceFields(player as unknown as Record<string, unknown>), player.sigma, player.n_maps);
  const currentTeam = record?.current_team ?? player.last_team;
  const role = roleLabel(record?.primary_role);

  return (
    <div className={styles.page}>
      <p className={styles.back}>
        <Link className="row-link" href="/elo?tab=players">← Player ratings</Link>
        {currentTeam ? <> · <Link className="row-link" href={`/elo/team/${teamSlug(currentTeam)}`}>{currentTeam}</Link></> : null}
      </p>
      <header className={styles.header}>
        <div>
          <p className={styles.scope}>Player · {record?.current_league ?? record?.primary ?? "current pack"}</p>
          <h1>{player.player}</h1>
          <p className={styles.summary}>
            {team ? <><Link className="row-link" href={`/elo/team/${teamSlug(team.team)}`}>{team.team}</Link> · {role}. </> : role !== "—" ? `${role}. ` : null}
            {trust.layman}
          </p>
        </div>
        <dl className={styles.metrics}>
          <div><dt>Adjusted rating</dt><dd>{softMu(player.mu_total, player.sigma, PLAYER_SIGMA_MIN).toFixed(1)}</dd></div>
          <div><dt>Raw rating</dt><dd>{player.mu_total.toFixed(1)}</dd></div>
          <div><dt>Confidence</dt><dd>{formatEvidenceCell(trust)}</dd></div>
          <div><dt>Games</dt><dd>{record?.games ?? player.n_maps}</dd></div>
          <div><dt>Record</dt><dd>{record ? `${record.wins}–${record.games - record.wins}` : "—"}</dd></div>
          <div><dt>Win rate</dt><dd>{formatWr(record?.wr)}</dd></div>
          <div><dt>Blue side</dt><dd>{formatWr(record?.blue_wr)}</dd></div>
          <div><dt>Red side</dt><dd>{formatWr(record?.red_wr)}</dd></div>
          <div><dt>Updated</dt><dd>{packUpdatedLabel(manifest)}</dd></div>
        </dl>
      </header>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2>Champions</h2>
          <span>{champions.length} played</span>
        </div>
        {champions.length ? (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr><th>Champion</th><th>Games</th><th>Record</th><th>Win rate</th><th>Average K / D / A</th></tr>
              </thead>
              <tbody>
                {champions.map((champion) => (
                  <tr key={champion.champion}>
                    <td>{champion.champion}</td>
                    <td>{champion.games}</td>
                    <td>{champion.wins}–{champion.losses}</td>
                    <td>{formatWr(champion.wr)}</td>
                    <td>
                      {champion.kills?.toFixed(1) ?? "—"} / {champion.deaths?.toFixed(1) ?? "—"} / {champion.assists?.toFixed(1) ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className={styles.empty}>No champion records are available for this player.</p>}
      </section>
    </div>
  );
}
