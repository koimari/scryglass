import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SignalMatches } from "@/components/SignalMatches";
import {
  packSourceUpdatedLabel,
  recentProfileGames,
  type ProfileRecords,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./MatchesPage.module.css";

export const revalidate = 21_600;

export const metadata: Metadata = {
  title: "Matches — Scryglass",
  description: "Latest completed professional League of Legends maps in Scryglass.",
};

export default async function MatchesPage() {
  const manifest = await readPackManifest();
  let profiles: ProfileRecords;
  try {
    profiles = await readPackJson(manifest, "features/profile_records.json");
  } catch {
    notFound();
  }
  const games = recentProfileGames(profiles, 100);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Matches</h1>
          <p>Latest accepted results. Open a map to compare every player with their usual form, teammates, opposing role, and league peers.</p>
        </div>
        <div className={styles.freshness}>
          <span>Source through</span>
          <strong>{packSourceUpdatedLabel(manifest)}</strong>
        </div>
      </header>

      <SignalMatches games={games} championImages={profiles.champion_images} />
    </div>
  );
}
