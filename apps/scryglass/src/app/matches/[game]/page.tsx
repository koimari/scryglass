import { notFound } from "next/navigation";
import { MatchRatingProfile } from "@/components/RatingProfiles";
import type { ProfileRecords } from "@/lib/pack";
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
  const game = profiles.games[gameId];
  if (!game) notFound();
  return <MatchRatingProfile game={game} championImages={profiles.champion_images} />;
}
