import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { SignalMatches } from "@/components/SignalMatches";
import {
  packSourceUpdatedLabel,
  recentProfileGames,
  type MatchIndex,
  type MatchSummary,
  type PublicSchedule,
  type ProfileRecords,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./MatchesPage.module.css";

// The active release is a runtime dependency. A build must stay independent
// from a retired or temporarily unavailable data object.
export const dynamic = "force-dynamic";
export const revalidate = 21_600;

export const metadata: Metadata = {
  title: "Matches — Scryglass",
  description: "Upcoming professional series and completed League of Legends games in Scryglass.",
};

export default async function MatchesPage() {
  const manifest = await readPackManifest();
  let profiles: ProfileRecords;
  try {
    profiles = await readPackJson(manifest, "features/profile_records.json");
  } catch {
    notFound();
  }
  let games: MatchSummary[];
  try {
    const archive = await readPackJson<MatchIndex>(manifest, "features/match_index.json");
    games = archive.games;
  } catch {
    games = recentProfileGames(profiles, 100).map((game) => ({
      game_id: game.game_id,
      date: game.date,
      league: game.league,
      competition_tier: game.competition_tier,
      blue_team: game.blue_team,
      red_team: game.red_team,
      blue_win: game.blue_win,
      champions: game.players.flatMap((player) => player.champion ? [player.champion] : []),
      grades_available: game.players.filter((player) => player.grade?.status === "available").length,
    }));
  }
  let schedule: PublicSchedule | null = null;
  try {
    schedule = await readPackJson<PublicSchedule>(manifest, "features/schedule.json");
  } catch {
    schedule = null;
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Matches</h1>
          <p>Upcoming professional series, plus accepted Oracle&apos;s Elixir games since January 2025. Open a result for its roster, champions, KDA, and player grades.</p>
        </div>
        <div className={styles.freshness}>
          <span>Source through</span>
          <strong>{packSourceUpdatedLabel(manifest)}</strong>
        </div>
      </header>

      <Suspense fallback={<p>Loading results…</p>}>
        <SignalMatches games={games} championImages={profiles.champion_images} schedule={schedule} />
      </Suspense>
    </div>
  );
}
