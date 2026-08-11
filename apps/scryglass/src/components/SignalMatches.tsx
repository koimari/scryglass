"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { teamSlug, type ProfileGame } from "@/lib/pack";
import { TeamMark } from "./TeamMark";
import styles from "./SignalMatches.module.css";

type View = "gallery" | "timeline" | "tournaments";

const DAY_FORMATTER = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" });

function dayLabel(value: string): string {
  return DAY_FORMATTER.format(new Date(value));
}

function timeLabel(value: string): string {
  return TIME_FORMATTER.format(new Date(value));
}

function gradesAvailable(game: ProfileGame): number {
  return game.players.filter((player) => player.grade?.status === "available").length;
}

function gameChampions(game: ProfileGame, limit: number): Array<{ champion: string; image: string | null }> {
  return game.players
    .filter((player) => player.champion)
    .slice(0, limit)
    .map((player) => ({ champion: player.champion ?? "", image: null }));
}

function ChampionLine({ game, images, limit = 10 }: { game: ProfileGame; images: Record<string, string>; limit?: number }) {
  const champions = gameChampions(game, limit);
  return (
    <span className={styles.championLine} aria-label={champions.map((item) => item.champion).join(", ")}>
      {champions.map((item) => images[item.champion] ? (
        // CommunityDragon supplies these portraits through the published pack.
        // eslint-disable-next-line @next/next/no-img-element
        <img key={`${item.champion}-${images[item.champion]}`} src={images[item.champion]} alt={item.champion} loading="lazy" />
      ) : null)}
    </span>
  );
}

function MatchCard({ game, images, featured = false }: { game: ProfileGame; images: Record<string, string>; featured?: boolean }) {
  const blueWon = game.blue_win === 1;
  const grades = gradesAvailable(game);
  return (
    <article className={`${styles.matchCard} ${featured ? styles.matchCardFeatured : ""}`}>
      <header><span>{game.league}</span><time dateTime={game.date}>{dayLabel(game.date)} · {timeLabel(game.date)} UTC</time></header>
      <div className={styles.cardTeams}>
        <Link href={`/elo/team/${teamSlug(game.blue_team)}`} className={blueWon ? styles.winner : ""}><span className={styles.teamIdentity}><TeamMark team={game.blue_team} size={featured ? "medium" : "small"} /><strong>{game.blue_team}</strong></span><b>{blueWon ? "W" : "L"}</b></Link>
        <Link href={`/elo/team/${teamSlug(game.red_team)}`} className={!blueWon ? styles.winner : ""}><span className={styles.teamIdentity}><TeamMark team={game.red_team} size={featured ? "medium" : "small"} /><strong>{game.red_team}</strong></span><b>{!blueWon ? "W" : "L"}</b></Link>
      </div>
      <ChampionLine game={game} images={images} limit={featured ? 10 : 5} />
      <footer><span>{grades ? `${grades}/10 grades` : "Stats pending"}</span><Link href={`/matches/${encodeURIComponent(game.game_id)}`}>Open map →</Link></footer>
    </article>
  );
}

export function SignalMatches({ games, championImages }: { games: ProfileGame[]; championImages: Record<string, string> }) {
  const [view, setView] = useState<View>("gallery");
  const [league, setLeague] = useState("");
  const [expanded, setExpanded] = useState(false);
  const leagues = useMemo(() => [...new Set(games.map((game) => game.league))].sort(), [games]);
  const filtered = useMemo(() => league ? games.filter((game) => game.league === league) : games, [games, league]);
  const byDay = useMemo(() => {
    const groups = new Map<string, ProfileGame[]>();
    for (const game of filtered) {
      const key = game.date.slice(0, 10);
      groups.set(key, [...(groups.get(key) ?? []), game]);
    }
    return [...groups.entries()];
  }, [filtered]);
  const byLeague = useMemo(() => {
    const groups = new Map<string, ProfileGame[]>();
    for (const game of filtered) groups.set(game.league, [...(groups.get(game.league) ?? []), game]);
    return [...groups.entries()].sort((a, b) => Date.parse(b[1][0]?.date ?? "") - Date.parse(a[1][0]?.date ?? ""));
  }, [filtered]);

  return (
    <div className={styles.root}>
      <section className={styles.controls}>
        <div className={styles.views} aria-label="Match view">
          {(["gallery", "timeline", "tournaments"] as const).map((value) => <button key={value} type="button" className={view === value ? styles.active : ""} aria-pressed={view === value} onClick={() => { setView(value); setExpanded(false); }}>{value === "tournaments" ? "Tournaments" : value.charAt(0).toUpperCase() + value.slice(1)}</button>)}
        </div>
        <label><span>Tournament</span><select value={league} onChange={(event) => { setLeague(event.target.value); setExpanded(false); }}><option value="">All tournaments</option>{leagues.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
      </section>

      {filtered.length ? (
        <>
          <section className={styles.summaryLine} aria-label="Match summary">
            <p><span>Accepted maps</span><strong>{filtered.length}</strong></p>
            <p><span>Latest result</span><strong>{dayLabel(filtered[0].date)}</strong></p>
            <p><span>Tournament</span><strong>{filtered[0].league}</strong></p>
          </section>

          {view === "gallery" ? (
            <>
              <section className={styles.gallery} aria-label="Latest match gallery">
                <MatchCard game={filtered[0]} images={championImages} featured />
                {filtered.slice(1, expanded ? filtered.length : 31).map((game) => <MatchCard key={game.game_id} game={game} images={championImages} />)}
              </section>
              {filtered.length > 31 ? <button className={styles.more} type="button" onClick={() => setExpanded((current) => !current)}>{expanded ? "Show latest 31 maps" : `Show all ${filtered.length} maps`}</button> : null}
            </>
          ) : null}

          {view === "timeline" ? (
            <div className={styles.timeline}>
              {byDay.map(([day, dayGames]) => (
                <section key={day} className={styles.day}>
                  <header><time dateTime={day}>{dayLabel(`${day}T12:00:00Z`)}</time><span>{dayGames.length} maps</span></header>
                  <div>{dayGames.map((game) => <MatchCard key={game.game_id} game={game} images={championImages} />)}</div>
                </section>
              ))}
            </div>
          ) : null}

          {view === "tournaments" ? (
            <div className={styles.tournaments}>
              {byLeague.map(([name, leagueGames]) => (
                <section key={name}>
                  <header><div><span>Tournament</span><h2>{name}</h2></div><p>{leagueGames.length} maps · through {dayLabel(leagueGames[0].date)}</p></header>
                  <div>{leagueGames.slice(0, 8).map((game) => <MatchCard key={game.game_id} game={game} images={championImages} />)}</div>
                </section>
              ))}
            </div>
          ) : null}
        </>
      ) : <p className={styles.empty}>No accepted maps match this tournament.</p>}
    </div>
  );
}
