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
  type ProfileGame,
  type ProfileGrade,
  type ProfileParticipant,
  type TeamRating,
  type TeamRecord,
} from "@/lib/pack";
import styles from "./RatingProfiles.module.css";

const ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"];

export type TeamRosterEntry = {
  player: string;
  role: string;
  rating: PlayerRating | null;
  ratingNote?: string;
};

function roleLabel(role: string | null | undefined): string {
  if (!role) return "—";
  return ({ jng: "Jungle", jungle: "Jungle", adc: "Bot", bot: "Bot", sup: "Support", support: "Support", top: "Top", mid: "Mid" } as Record<string, string>)[role.toLowerCase()] ?? role;
}

function tierLabel(value: string | null | undefined): string {
  if (!value) return "Current circuit";
  return value.replace(/^tier/i, "Tier ");
}

function shortDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 10);
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}

function rosterEvidenceLabel(value: string): string {
  const readable = value.replaceAll("_", " ");
  return readable.charAt(0).toUpperCase() + readable.slice(1);
}

function playerInGame(game: ProfileGame, player: string): ProfileParticipant | undefined {
  return game.players.find((participant) => participant.player.toLowerCase() === player.toLowerCase());
}

function teamWon(game: ProfileGame, team: string): boolean {
  return game.blue_team.toLowerCase() === team.toLowerCase() ? game.blue_win === 1 : game.blue_win === 0;
}

function scoreGrade(score: number): string {
  if (score >= 90) return "A+";
  if (score >= 75) return "A";
  if (score >= 55) return "B";
  if (score >= 35) return "C";
  if (score >= 15) return "D";
  return "F";
}

function gradeSummary(games: ProfileGame[], player: string): { grade: string; trend: string; maps: number } | null {
  const grades = games
    .map((game) => playerInGame(game, player)?.grade)
    .filter((grade): grade is Extract<ProfileGrade, { status: "available" }> => grade?.status === "available");
  if (!grades.length) return null;
  const score = grades.reduce((total, grade) => total + grade.score, 0) / grades.length;
  const self = grades.reduce((total, grade) => total + grade.components.self, 0) / grades.length;
  return {
    grade: scoreGrade(score),
    trend: self >= 0.35 ? "Above own standard" : self <= -0.35 ? "Below own standard" : "Near own standard",
    maps: grades.length,
  };
}

function ChampionPortrait({ name, imageUrl, size = "small" }: { name: string | null; imageUrl?: string | null; size?: "small" | "large" }) {
  return (
    <span className={`${styles.portrait} ${size === "large" ? styles.portraitLarge : ""}`} title={name ?? undefined}>
      {imageUrl ? (
        // The source is a small square CommunityDragon asset.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={imageUrl} alt={name ?? "Champion"} loading="lazy" />
      ) : <span aria-hidden>{name?.slice(0, 1) ?? "?"}</span>}
    </span>
  );
}

function RecentMatches({ games, championImages, player, team }: { games: ProfileGame[]; championImages: Record<string, string>; player?: string; team?: string }) {
  if (!games.length) return <p className={styles.empty}>Recent match details are waiting for the next accepted data refresh.</p>;
  return (
    <div className={styles.matchList}>
      {games.map((game) => {
        const participant = player ? playerInGame(game, player) : undefined;
        const focusTeam = participant
          ? participant.side === "Blue" ? game.blue_team : game.red_team
          : team ?? "";
        const won = focusTeam ? teamWon(game, focusTeam) : false;
        const opponent = game.blue_team.toLowerCase() === focusTeam.toLowerCase() ? game.red_team : game.blue_team;
        return (
          <Link className={styles.matchRow} href={`/matches/${encodeURIComponent(game.game_id)}`} key={game.game_id}>
            <span className={`${styles.resultMark} ${won ? styles.win : styles.loss}`}>{won ? "W" : "L"}</span>
            {participant ? <ChampionPortrait name={participant.champion} imageUrl={participant.champion ? championImages[participant.champion] : null} /> : null}
            <div className={styles.matchMain}>
              <strong>{opponent || `${game.blue_team} vs ${game.red_team}`}</strong>
              <span>{game.league} · {shortDate(game.date)}</span>
            </div>
            {participant ? (
              <div className={styles.matchScore}>
                <strong>{participant.grade?.status === "available" ? participant.grade.grade : "—"} · {participant.kills ?? "—"} / {participant.deaths ?? "—"} / {participant.assists ?? "—"}</strong>
                <span>{roleLabel(participant.role)}</span>
              </div>
            ) : <span className={styles.matchSide}>{game.blue_team === focusTeam ? "Blue" : "Red"}</span>}
          </Link>
        );
      })}
    </div>
  );
}

export function TeamRatingProfile({
  team,
  roster,
  record,
  playerRecords,
  standing,
  roleRanks,
  recentGames,
  championImages,
  manifest,
}: {
  team: TeamRating;
  roster: TeamRosterEntry[];
  record?: TeamRecord;
  playerRecords: Record<string, PlayerRecord>;
  standing: { tierRank: number; tierTotal: number };
  roleRanks: Record<string, { rank: number; total: number }>;
  recentGames: ProfileGame[];
  championImages: Record<string, string>;
  manifest: PackManifest;
}) {
  const trust = evidenceInfo(evidenceFields(team as unknown as Record<string, unknown>), team.sigma, record?.games);
  const players = [...roster].sort((a, b) => {
    return ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role);
  });
  const exactRoster = team.exact_roster?.players.length === 5 ? team.exact_roster : null;
  const rosterSource = exactRoster
    ? "Published exact roster"
    : recentGames.length
      ? "Latest observed lineup"
      : "Affiliation snapshot";
  const rosterEvidence = exactRoster
    ? `${rosterEvidenceLabel(exactRoster.evidence_state)} evidence · ${exactRoster.model_scope} model · effective ${shortDate(exactRoster.roster_effective_at)} · source receipt ${exactRoster.roster_receipt_sha256.slice(0, 12)}…`
    : recentGames.length
      ? `Recorded in the latest accepted map on ${shortDate(recentGames[0].date)}. A later lineup will appear after another accepted map arrives.`
      : "These player affiliations come from the ratings snapshot. An accepted played-map lineup is unavailable.";

  return (
    <div className={styles.page}>
      <p className={styles.back}><Link className="row-link" href="/elo">← Team ratings</Link></p>
      <header className={styles.hero}>
        <div className={styles.identity}>
          <p className={styles.scope}>{tierLabel(record?.current_tier)} · {record?.current_league ?? record?.primary ?? "current pack"}</p>
          <h1>{team.team}</h1>
          <p className={styles.summary}>Results across the current rating window. {trust.layman}</p>
        </div>
        <div className={styles.ratingBlock}>
          <span>Adjusted team rating</span>
          <strong>{adjustedRating(team, TEAM_SIGMA_MIN).toFixed(1)}</strong>
          {standing.tierRank > 0 ? <small>#{standing.tierRank} of {standing.tierTotal} in {tierLabel(record?.current_tier)}</small> : null}
        </div>
      </header>

      <dl className={styles.statBand}>
        <div><dt>Record</dt><dd>{record ? `${record.wins}–${record.games - record.wins}` : "—"}</dd></div>
        <div><dt>Win rate</dt><dd>{formatWr(record?.wr)}</dd></div>
        <div><dt>Maps</dt><dd>{record?.games ?? team.n_maps ?? "—"}</dd></div>
        <div><dt>Confidence</dt><dd>{formatEvidenceCell(trust)}</dd></div>
        <div><dt>Updated</dt><dd>{packUpdatedLabel(manifest)}</dd></div>
      </dl>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>{rosterSource}</p><h2>Players by role</h2></div><span>{players.length} listed</span></div>
        <p className={styles.empty}>{rosterEvidence}</p>
        {players.length ? (
          <div className={styles.rosterGrid}>
            {players.map((player) => {
              const playerRecord = playerRecords[player.player];
              const roleRank = roleRanks[player.player];
              const recent = gradeSummary(recentGames, player.player);
              const content = <>
                  <span className={styles.rosterRole}>{roleLabel(player.role ?? playerRecord?.primary_role)}</span>
                  <strong>{player.player}</strong>
                  <span className={styles.rosterRating}>{player.rating ? softMu(player.rating.mu_total, player.rating.sigma, PLAYER_SIGMA_MIN).toFixed(1) : "Pending"}</span>
                  <small>{roleRank?.rank > 0 ? `#${roleRank.rank} of ${roleRank.total} ${roleLabel(player.role).toLowerCase()}s` : player.ratingNote ?? "Role rank unavailable"}</small>
                  <small>{recent ? `Last ${recent.maps}: ${recent.grade} · ${recent.trend}` : "Recent grade unavailable"}</small>
                </>;
              return player.rating ? (
                <Link className={styles.rosterCard} href={`/elo/player/${playerSlug(player.player)}`} key={player.player}>{content}</Link>
              ) : (
                <article className={styles.rosterCard} key={player.player}>{content}</article>
              );
            })}
          </div>
        ) : <p className={styles.empty}>No roster source is available for this team.</p>}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>Latest results</p><h2>Recent maps</h2></div><Link className="row-link" href="/matches">All matches →</Link></div>
        <RecentMatches games={recentGames} championImages={championImages} team={team.team} />
      </section>
    </div>
  );
}

function GradeDetails({ grade }: { grade?: ProfileGrade }) {
  if (!grade || grade.status !== "available") {
    return <span className={styles.gradeUnavailable}>{grade?.reason ?? "Grade unavailable"}</span>;
  }
  return (
    <div className={styles.gradeDetails}>
      <strong>{grade.grade}</strong>
      <span>{grade.score.toFixed(1)}</span>
      <small>Usual form {grade.components.self.toFixed(2)} · Teammates {grade.components.team.toFixed(2)} · Opposing role {grade.components.opponent.toFixed(2)} · League peers {grade.components.league_role.toFixed(2)}</small>
    </div>
  );
}

export function MatchRatingProfile({ game, championImages }: { game: ProfileGame; championImages: Record<string, string> }) {
  const gradesAvailable = game.players.some((player) => player.grade?.status === "available");
  const sides = (["Blue", "Red"] as const).map((side) => ({
    side,
    team: side === "Blue" ? game.blue_team : game.red_team,
    won: side === "Blue" ? game.blue_win === 1 : game.blue_win === 0,
    players: game.players.filter((player) => player.side === side).sort((a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role)),
  }));
  return (
    <div className={styles.page}>
      <p className={styles.back}><Link className="row-link" href="/matches">← Matches</Link></p>
      <header className={styles.matchHero}>
        <p className={styles.scope}>{game.league} · {shortDate(game.date)}</p>
        <h1>{game.blue_team} vs {game.red_team}</h1>
        <p>{gradesAvailable
          ? "Player grades compare this map with the player’s prior form, their teammates, the same-role opponent, and the league-role baseline. Match result is excluded."
          : "The result, roster, roles, and champions are available. Player grades will appear after the completed stat line reaches the accepted source."}</p>
      </header>
      <div className={styles.matchTeams}>
        {sides.map((side) => (
          <section className={styles.matchTeam} key={side.side}>
            <div className={styles.sectionHeader}><div><p>{side.side} side</p><h2><Link className="row-link" href={`/elo/team/${teamSlug(side.team)}`}>{side.team}</Link></h2></div><span>{side.won ? "Win" : "Loss"}</span></div>
            {side.players.map((player) => (
              <article className={styles.matchPlayer} key={player.player}>
                <ChampionPortrait name={player.champion} imageUrl={player.champion ? championImages[player.champion] : null} size="large" />
                <div><span>{roleLabel(player.role)}</span><strong><Link className="row-link" href={`/elo/player/${playerSlug(player.player)}`}>{player.player}</Link></strong><small>{player.champion} · {player.kills ?? "—"} / {player.deaths ?? "—"} / {player.assists ?? "—"}</small></div>
                <GradeDetails grade={player.grade} />
              </article>
            ))}
          </section>
        ))}
      </div>
    </div>
  );
}

export function PlayerRatingProfile({
  player,
  champions,
  record,
  team,
  standing,
  recentGames,
  championImages,
  manifest,
}: {
  player: PlayerRating;
  champions: PlayerChampionRecord[];
  record?: PlayerRecord;
  team?: TeamRating | null;
  standing: { tierRank: number; tierTotal: number; roleRank: number; roleTotal: number };
  recentGames: ProfileGame[];
  championImages: Record<string, string>;
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
      <header className={styles.hero}>
        <div className={styles.identity}>
          <p className={styles.scope}>{tierLabel(record?.current_tier)} · {record?.current_league ?? record?.primary ?? "current pack"}</p>
          <h1>{player.player}</h1>
          <p className={styles.summary}>
            {team ? <><Link className="row-link" href={`/elo/team/${teamSlug(team.team)}`}>{team.team}</Link> · {role}. </> : role !== "—" ? `${role}. ` : null}
            This rating tracks the strength of team results with this player in the lineup. Recent map grades describe individual performance against four comparison baselines.
          </p>
        </div>
        <div className={styles.ratingBlock}>
          <span>Adjusted results rating</span>
          <strong>{softMu(player.mu_total, player.sigma, PLAYER_SIGMA_MIN).toFixed(1)}</strong>
          <div className={styles.rankLines}>
            {standing.tierRank > 0 ? <small>#{standing.tierRank} of {standing.tierTotal} in {tierLabel(record?.current_tier)}</small> : null}
            {standing.roleRank > 0 ? <small>#{standing.roleRank} of {standing.roleTotal} {role.toLowerCase()}s</small> : null}
          </div>
        </div>
      </header>

      <dl className={styles.statBand}>
        <div><dt>Record</dt><dd>{record ? `${record.wins}–${record.games - record.wins}` : "—"}</dd></div>
        <div><dt>Win rate</dt><dd>{formatWr(record?.wr)}</dd></div>
        <div><dt>Blue / Red</dt><dd>{formatWr(record?.blue_wr)} / {formatWr(record?.red_wr)}</dd></div>
        <div><dt>Confidence</dt><dd>{formatEvidenceCell(trust)}</dd></div>
        <div><dt>Updated</dt><dd>{packUpdatedLabel(manifest)}</dd></div>
      </dl>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>Latest results</p><h2>Recent maps</h2></div><Link className="row-link" href="/matches">All matches →</Link></div>
        <RecentMatches games={recentGames} championImages={championImages} player={player.player} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>Career in this window</p><h2>Champion pool</h2></div><span>{champions.length} played</span></div>
        {champions.length ? (
          <div className={styles.championGrid}>
            {champions.map((champion) => (
              <article className={styles.championCard} key={champion.champion} title={`${champion.champion}: ${champion.games} maps, ${formatWr(champion.wr)} win rate`}>
                <ChampionPortrait name={champion.champion} imageUrl={championImages[champion.champion] ?? champion.champion_image_url} size="large" />
                <strong>{champion.games}</strong>
                <span>{formatWr(champion.wr)}</span>
              </article>
            ))}
          </div>
        ) : <p className={styles.empty}>No champion records are available for this player.</p>}
      </section>
    </div>
  );
}
