import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  packSourceUpdatedLabel,
  recentProfileGames,
  teamSlug,
  type ProfileGame,
  type ProfileRecords,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./MatchesPage.module.css";

export const revalidate = 21_600;

export const metadata: Metadata = {
  title: "Matches — Scryglass",
  description: "Latest completed professional League of Legends maps in Scryglass.",
};

const DAY_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

const TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "UTC",
});

function dayKey(value: string): string {
  return value.slice(0, 10);
}

function dayLabel(value: string): string {
  const date = new Date(`${value}T12:00:00Z`);
  return DAY_FORMATTER.format(date);
}

function timeLabel(value: string): string {
  return TIME_FORMATTER.format(new Date(value));
}

function teamResult(game: ProfileGame, side: "blue" | "red"): "W" | "L" {
  const won = side === "blue" ? game.blue_win === 1 : game.blue_win === 0;
  return won ? "W" : "L";
}

export default async function MatchesPage() {
  const manifest = await readPackManifest();
  let profiles: ProfileRecords;
  try {
    profiles = await readPackJson(manifest, "features/profile_records.json");
  } catch {
    notFound();
  }
  const games = recentProfileGames(profiles, 100);
  const groups = new Map<string, ProfileGame[]>();
  for (const game of games) {
    const key = dayKey(game.date);
    groups.set(key, [...(groups.get(key) ?? []), game]);
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Completed professional maps</p>
          <h1>Matches</h1>
          <p>Latest accepted results. Open a map to compare every player with their usual form, teammates, opposing role, and league peers.</p>
        </div>
        <div className={styles.freshness}>
          <span>Source through</span>
          <strong>{packSourceUpdatedLabel(manifest)}</strong>
          <small>{games.length} latest maps</small>
        </div>
      </header>

      {games.length ? (
        <div className={styles.ledger}>
          {[...groups.entries()].map(([day, dayGames]) => (
            <section className={styles.day} key={day}>
              <h2>{dayLabel(day)}</h2>
              <div className={styles.dayGames}>
                {dayGames.map((game) => (
                  <article className={styles.match} key={game.game_id}>
                    <div className={styles.meta}>
                      <strong>{game.league}</strong>
                      <time dateTime={game.date}>{timeLabel(game.date)} UTC</time>
                    </div>
                    <div className={styles.teams}>
                      <Link href={`/elo/team/${teamSlug(game.blue_team)}`} className={game.blue_win === 1 ? styles.winner : undefined}>
                        <span>{game.blue_team}</span><b>{teamResult(game, "blue")}</b>
                      </Link>
                      <Link href={`/elo/team/${teamSlug(game.red_team)}`} className={game.blue_win === 0 ? styles.winner : undefined}>
                        <span>{game.red_team}</span><b>{teamResult(game, "red")}</b>
                      </Link>
                    </div>
                    <Link className={styles.open} href={`/matches/${encodeURIComponent(game.game_id)}`}>
                      {game.players.some((player) => player.grade?.status === "available") ? "Player grades" : "Map details"} <span aria-hidden>→</span>
                    </Link>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <p className={styles.empty}>Completed maps are waiting for the next accepted refresh.</p>
      )}
    </div>
  );
}
