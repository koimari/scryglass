import { notFound } from "next/navigation";
import { MatchRatingProfile } from "@/components/RatingProfiles";
import type { MatchIndex, MatchRecords, ProfileGame, ProfileRecords } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ game: string }> };

export default async function MatchPage({ params }: Props) {
  const { game: raw } = await params;
  const gameId = decodeURIComponent(raw);
  const manifest = await readPackManifest();
  let profiles: ProfileRecords;
  try {
    profiles = await readPackJson(manifest, "features/profile_records.json");
  } catch {
    notFound();
  }
  let game: ProfileGame | undefined = profiles.games[gameId];
  if (!game) {
    try {
      const index = await readPackJson<MatchIndex>(manifest, "features/match_index.json");
      const summary = index.games.find((candidate) => candidate.game_id === gameId);
      const year = summary ? new Date(summary.date).getUTCFullYear() : 0;
      if (year === 2025 || year === 2026) {
        const archive = await readPackJson<MatchRecords>(manifest, `features/match_records_${year}.json`);
        game = archive.games[gameId];
      }
    } catch {
      game = undefined;
    }
  }
  if (!game) notFound();
  return <MatchRatingProfile game={game} championImages={profiles.champion_images} />;
}
