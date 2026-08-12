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
  type PlayerPositionDeltas,
  type PlayerRankComparison,
  type PlayerRecord,
  type ProfileGame,
  type ProfileGrade,
  type ProfileParticipant,
  type ProfileTeamStats,
  type TeamRating,
  type TeamRecord,
} from "@/lib/pack";
import { playerPortrait, type PlayerVisualIdentity } from "@/lib/playerPortraits";
import { matchTeamHref } from "@/lib/matchFilters";
import { PlayerPortrait } from "./PlayerPortrait";
import { RecentGames } from "./RecentGames";
import { TeamMark } from "./TeamMark";
import styles from "./RatingProfiles.module.css";

const ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"];
const COMPACT_NUMBER_FORMATTER = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

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

function fullDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 10);
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "long", year: "numeric", timeZone: "UTC" });
}

function gameCount(value: number): string {
  return `${value} ${value === 1 ? "game" : "games"}`;
}

function durationLabel(seconds: number | null | undefined): string | null {
  if (!seconds || seconds < 1) return null;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function compactNumber(value: number): string {
  return COMPACT_NUMBER_FORMATTER.format(value);
}

function percentLabel(value: number): string {
  const normalized = Math.abs(value) <= 1 ? value * 100 : value;
  return `${normalized.toFixed(0)}%`;
}

function signedNumber(value: number): string {
  return `${value > 0 ? "+" : ""}${Math.round(value)}`;
}

function positionChange(comparison: PlayerRankComparison | undefined): {
  label: string;
  tone: "up" | "down" | "flat" | "new" | "pending";
  title: string;
} {
  if (!comparison) {
    return {
      label: "—",
      tone: "pending",
      title: "Available after the next historical ratings refresh.",
    };
  }
  if (comparison.rank === null || comparison.delta === null) {
    return {
      label: "New",
      tone: "new",
      title: `The player was not ranked on ${shortDate(comparison.as_of)}.`,
    };
  }
  if (comparison.delta > 0) {
    return {
      label: `+${comparison.delta}`,
      tone: "up",
      title: `Climbed ${comparison.delta} places from #${comparison.rank}.`,
    };
  }
  if (comparison.delta < 0) {
    return {
      label: `${comparison.delta}`,
      tone: "down",
      title: `Fell ${Math.abs(comparison.delta)} places from #${comparison.rank}.`,
    };
  }
  return {
    label: "Same",
    tone: "flat",
    title: `Held the same position since ${shortDate(comparison.as_of)}.`,
  };
}

function rosterEvidenceLabel(value: string): string {
  const readable = value.replaceAll("_", " ");
  return readable.charAt(0).toUpperCase() + readable.slice(1);
}

function playerInGame(game: ProfileGame, player: string): ProfileParticipant | undefined {
  return game.players.find((participant) => participant.player.toLowerCase() === player.toLowerCase());
}

function scoreGrade(score: number): string {
  if (score >= 90) return "A+";
  if (score >= 75) return "A";
  if (score >= 55) return "B";
  if (score >= 35) return "C";
  if (score >= 15) return "D";
  return "F";
}

function gradeSignal(grade: Extract<ProfileGrade, { status: "available" }>): string {
  const signals = [
    { value: grade.components.self, positive: "Above usual", negative: "Below usual" },
    { value: grade.components.team, positive: "Above teammates", negative: "Below teammates" },
    { value: grade.components.opponent, positive: "Ahead of role opponent", negative: "Behind role opponent" },
    { value: grade.components.league_role, positive: "Above role baseline", negative: "Below role baseline" },
  ]
    .filter((signal) => Math.abs(signal.value) >= 0.25)
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 2)
    .map((signal) => signal.value >= 0 ? signal.positive : signal.negative);
  return signals.join(" · ") || "Near all baselines";
}

function gameLanguage(value: string): string {
  return value
    .replace(/\bMaps\b/g, "Games")
    .replace(/\bMap\b/g, "Game")
    .replace(/\bmaps\b/g, "games")
    .replace(/\bmap\b/g, "game");
}

function gradeSummary(games: ProfileGame[], player: string): { grade: string; trend: string; games: number } | null {
  const grades = games
    .map((game) => playerInGame(game, player)?.grade)
    .filter((grade): grade is Extract<ProfileGrade, { status: "available" }> => grade?.status === "available");
  if (!grades.length) return null;
  const score = grades.reduce((total, grade) => total + grade.score, 0) / grades.length;
  const self = grades.reduce((total, grade) => total + grade.components.self, 0) / grades.length;
  return {
    grade: scoreGrade(score),
    trend: self >= 0.35 ? "Above own standard" : self <= -0.35 ? "Below own standard" : "Near own standard",
    games: grades.length,
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
      ? `Recorded in the latest accepted game on ${shortDate(recentGames[0].date)}. A later lineup will appear after another accepted game arrives.`
      : "These player affiliations come from the ratings snapshot. An accepted game lineup is unavailable.";

  return (
    <div className={styles.page}>
      <p className={styles.back}><Link className="row-link" href="/elo">← Team ratings</Link></p>
      <header className={styles.hero}>
        <div className={styles.heroIdentity}>
          <TeamMark team={team.team} size="large" />
          <div className={styles.identity}>
            <p className={styles.scope}>{tierLabel(record?.current_tier)} · {record?.current_league ?? record?.primary ?? "current pack"}</p>
            <h1>{team.team}</h1>
            <p className={styles.summary}>Results across the current rating window. {trust.layman}</p>
          </div>
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
        <div><dt>Games</dt><dd>{record?.games ?? team.n_maps ?? "—"}</dd></div>
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
                  <PlayerPortrait
                    player={player.player}
                    team={team.team}
                    portrait={playerPortrait(player.player, team.team)}
                    variant="roster"
                  />
                  <div className={styles.rosterBody}>
                    <span className={styles.rosterRole}>{roleLabel(player.role ?? playerRecord?.primary_role)}</span>
                    <strong>{player.player}</strong>
                    <span className={styles.rosterRating}>{player.rating ? softMu(player.rating.mu_total, player.rating.sigma, PLAYER_SIGMA_MIN).toFixed(1) : "Pending"}</span>
                    <small>{roleRank?.rank > 0 ? `#${roleRank.rank} of ${roleRank.total} ${roleLabel(player.role).toLowerCase()}s` : player.ratingNote ?? "Role rank unavailable"}</small>
                    <small>{recent ? `Last ${recent.games}: ${recent.grade} · ${recent.trend}` : "Recent grade unavailable"}</small>
                  </div>
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
        <div className={styles.sectionHeader}><div><p>Latest results</p><h2>Recent games</h2></div><Link className="row-link" href={matchTeamHref(team.team)}>All matches →</Link></div>
        <RecentGames games={recentGames} championImages={championImages} team={team.team} />
      </section>
    </div>
  );
}

function GradeDetails({ grade }: { grade?: ProfileGrade }) {
  if (!grade || grade.status !== "available") {
    return <span className={styles.gradeUnavailable}>{gameLanguage(grade?.reason ?? "Grade unavailable")}</span>;
  }
  return (
    <div className={styles.gradeDetails}>
      <strong>{grade.grade}</strong>
      <span>{grade.score.toFixed(0)}</span>
      <small>{gradeSignal(grade)}</small>
    </div>
  );
}

function TeamGameStats({ stats }: { stats?: ProfileTeamStats }) {
  if (!stats) return null;
  const values = [
    ["Kills", stats.kills],
    ["Gold", stats.gold == null ? null : compactNumber(stats.gold)],
    ["Towers", stats.towers],
    ["Dragons", stats.dragons],
    ["Barons", stats.barons],
    ["Grubs", stats.void_grubs],
    ["Heralds", stats.heralds],
    ["Atakhans", stats.atakhans],
    ["Inhibitors", stats.inhibitors],
  ].filter((entry) => entry[1] !== null && entry[1] !== undefined);
  if (!values.length) return null;
  return <dl className={styles.teamGameStats}>{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}

function PlayerGameStats({ player }: { player: ProfileParticipant }) {
  const participation = player.team_kills && player.kills != null && player.assists != null
    ? (player.kills + player.assists) / player.team_kills
    : null;
  const values = [
    ["KDA", `${player.kills ?? "—"} / ${player.deaths ?? "—"} / ${player.assists ?? "—"}`],
    ["KP", participation == null ? null : percentLabel(participation)],
    ["CS", player.cs == null ? null : `${Math.round(player.cs)}${player.cs_per_minute == null ? "" : ` · ${player.cs_per_minute.toFixed(1)}/m`}`],
    ["DPM", player.damage_per_minute == null ? null : Math.round(player.damage_per_minute).toString()],
    ["Gold", player.gold == null ? null : compactNumber(player.gold)],
    ["Damage", player.damage_share == null ? null : percentLabel(player.damage_share)],
    ["Vision", player.vision_score == null ? null : `${Math.round(player.vision_score)}${player.wards_placed == null ? "" : ` · ${Math.round(player.wards_placed)}w`}`],
    ["GD@10", player.gold_diff_at_10 == null ? null : signedNumber(player.gold_diff_at_10)],
  ].filter((entry) => entry[1] !== null);
  return <dl className={styles.playerGameStats}>{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}

export function MatchRatingProfile({ game, championImages }: { game: ProfileGame; championImages: Record<string, string> }) {
  const gradesAvailable = game.players.some((player) => player.grade?.status === "available");
  const blueWon = game.blue_win === 1;
  const winner = blueWon ? game.blue_team : game.red_team;
  const loser = blueWon ? game.red_team : game.blue_team;
  const duration = durationLabel(game.duration_seconds);
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
        <p className={styles.scope}>{game.league} · {fullDate(game.date)}{duration ? ` · ${duration}` : ""}</p>
        <h1>{winner} defeated {loser}</h1>
        <div className={styles.matchResult}>
          <span><TeamMark team={game.blue_team} size="medium" /><strong>{game.blue_team}</strong><small>Blue · {blueWon ? "Winner" : "Defeat"}</small></span>
          <b>vs</b>
          <span><TeamMark team={game.red_team} size="medium" /><strong>{game.red_team}</strong><small>Red · {!blueWon ? "Winner" : "Defeat"}</small></span>
        </div>
        {gradesAvailable ? (
          <aside className={styles.matchGradeGuide} aria-label="Player grade guide">
            <strong>Game grade</strong>
            <span>A standout · B strong · C typical · D below standard · F poor</span>
            <small>Each score compares the player with their usual form, teammates, role opponent, and league-role history. <Link className="row-link" href="/methodology#game-grades">How grades work</Link></small>
          </aside>
        ) : <p>Player grades will appear when Oracle&apos;s Elixir supplies a complete stat line.</p>}
      </header>
      <div className={styles.matchTeams}>
        {sides.map((side) => (
          <section className={styles.matchTeam} key={side.side}>
            <div className={styles.sectionHeader}><div><p>{side.side} side</p><h2><Link className="row-link" href={`/elo/team/${teamSlug(side.team)}`}>{side.team}</Link></h2></div><span>{side.won ? "Winner" : "Defeat"}</span></div>
            <TeamGameStats stats={game.team_stats?.[side.side]} />
            {side.players.map((player) => (
              <article className={styles.matchPlayer} key={`${side.side}-${player.role}-${player.player}`}>
                <ChampionPortrait name={player.champion} imageUrl={player.champion ? championImages[player.champion] : null} size="large" />
                <div className={styles.matchPlayerIdentity}><span>{roleLabel(player.role)}</span><strong><Link className="row-link" href={`/elo/player/${playerSlug(player.player)}`}>{player.player}</Link></strong><small>{player.champion}</small></div>
                <PlayerGameStats player={player} />
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
  portrait,
  champions,
  record,
  team,
  standing,
  positionDeltas,
  recentGames,
  championImages,
  manifest,
}: {
  player: PlayerRating;
  portrait?: PlayerVisualIdentity | null;
  champions: PlayerChampionRecord[];
  record?: PlayerRecord;
  team?: TeamRating | null;
  standing: { tierRank: number; tierTotal: number; roleRank: number; roleTotal: number };
  positionDeltas?: PlayerPositionDeltas;
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
        <div className={styles.heroIdentity}>
          <PlayerPortrait player={player.player} team={currentTeam} portrait={portrait} />
          <div className={styles.identity}>
            <p className={styles.scope}>{tierLabel(record?.current_tier)} · {record?.current_league ?? record?.primary ?? "current pack"}</p>
            <h1>{player.player}</h1>
            <p className={styles.summary}>
              {team ? <><Link className="row-link" href={`/elo/team/${teamSlug(team.team)}`}>{team.team}</Link> · {role}. </> : role !== "—" ? `${role}. ` : null}
              This rating tracks the strength of team results with this player in the lineup. Recent game grades describe individual performance against four comparison baselines.
            </p>
          </div>
        </div>
        <div className={`${styles.ratingBlock} ${styles.playerStanding}`}>
          <div className={styles.ladderPlace}>
            <span>{tierLabel(record?.current_tier)} ladder</span>
            <strong>{standing.tierRank > 0 ? `#${standing.tierRank}` : "Unranked"}</strong>
            {standing.tierRank > 0 ? <small>of {standing.tierTotal} active players</small> : null}
            {standing.roleRank > 0 ? <small>#{standing.roleRank} of {standing.roleTotal} {role.toLowerCase()}s</small> : null}
          </div>
          <div className={styles.ratingReference}>
            <span>Adjusted rating</span>
            <strong>{softMu(player.mu_total, player.sigma, PLAYER_SIGMA_MIN).toFixed(1)}</strong>
          </div>
          <div className={styles.positionMovement}>
            <div className={styles.positionMovementHeader}>
              <span>Position change</span>
              <small>+ means climbed</small>
            </div>
            <div className={styles.positionMovementGrid}>
              {(["1m", "3m", "12m"] as const).map((period) => {
                const change = positionChange(positionDeltas?.[period]);
                return (
                  <div key={period} title={change.title}>
                    <span>{period}</span>
                    <strong className={styles[change.tone]}>{change.label}</strong>
                  </div>
                );
              })}
            </div>
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

      <p className={styles.confidenceNote}>
        <strong>Confidence is a data-quality label.</strong> Precision, stability, freshness, sample support, and active status are checks around the rating. They are not five equal parts of the score. The adjusted rating separately becomes more cautious when uncertainty is high.
      </p>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>Latest results</p><h2>Recent games</h2></div><Link className="row-link" href={currentTeam ? matchTeamHref(currentTeam) : "/matches?section=results"}>All matches →</Link></div>
        <aside className={styles.gradeGuide} aria-label="Game grade guide">
          <strong>Full game, not KDA</strong>
          <span>A+/A standout</span>
          <span>B strong</span>
          <span>C typical</span>
          <span>D below standard</span>
          <span>F poor</span>
          <small>Kill participation, survival, damage, gold, farm, and vision are compared with 4 baselines. Champion choice is not a baseline. <Link className="row-link" href="/methodology#game-grades">How grades work</Link></small>
        </aside>
        <RecentGames games={recentGames} championImages={championImages} player={player.player} />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}><div><p>Career in this window</p><h2>Champion pool</h2></div><span>{champions.length} played</span></div>
        {champions.length ? (
          <div className={styles.championGrid} data-native-scroll tabIndex={0} aria-label={`${player.player} champion pool`}>
            {champions.map((champion) => (
              <article className={styles.championCard} key={champion.champion} title={`${champion.champion}: ${gameCount(champion.games)}, ${formatWr(champion.wr)} win rate`}>
                <ChampionPortrait name={champion.champion} imageUrl={championImages[champion.champion] ?? champion.champion_image_url} size="large" />
                <div className={styles.championCardCopy}>
                  <strong>{champion.champion}</strong>
                  <span>{gameCount(champion.games)}</span>
                  <span>{formatWr(champion.wr)} WR</span>
                </div>
              </article>
            ))}
          </div>
        ) : <p className={styles.empty}>No champion records are available for this player.</p>}
      </section>
    </div>
  );
}
