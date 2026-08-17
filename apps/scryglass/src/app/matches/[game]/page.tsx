import { notFound } from "next/navigation";
import { MatchRatingProfile } from "@/components/RatingProfiles";
import { hasPromotedDraftAuthority, hasPublishedDraftAuthority, type MatchIndex, type MatchRecords, type ProfileGame, type ProfileRecords } from "@/lib/pack";
import { getMatch, queryApiAvailable, validatePromotedDraftResultsPayload, type PromotedDraftResultsPayload, type PublicDraftScoreResult } from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ game: string }> };

async function promotedDraftResult(
  manifest: Awaited<ReturnType<typeof readPackManifest>>,
  gameId: string,
): Promise<PublicDraftScoreResult | null> {
  if (!hasPromotedDraftAuthority(manifest)) return null;
  const payload = await readPackJson<PromotedDraftResultsPayload>(
    manifest,
    "features/promoted_draft_results.json",
  );
  validatePromotedDraftResultsPayload(payload, manifest);
  return payload.results[gameId] ?? null;
}

export default async function MatchPage({ params }: Props) {
  const { game: raw } = await params;
  const gameId = decodeURIComponent(raw);
  const manifest = await readPackManifest();
  if (queryApiAvailable(manifest)) {
    const result = await getMatch(manifest, gameId);
    if (!result.row) notFound();
    const draftScore = await promotedDraftResult(manifest, gameId);
    return (
      <MatchRatingProfile
        game={hasPublishedDraftAuthority(manifest)
          ? result.row.payload
          : { ...result.row.payload, draft_pool: undefined, draft_contribution: undefined }}
        championImages={result.champion_images}
        releaseId={manifest.pack_id}
        draftScore={draftScore}
      />
    );
  }
  const profiles = await readPackJson<ProfileRecords>(manifest, "features/profile_records.json");
  let game: ProfileGame | undefined = profiles.games[gameId];
  if (!game) {
    const index = await readPackJson<MatchIndex>(manifest, "features/match_index.json");
    const summary = index.games.find((candidate) => candidate.game_id === gameId);
    if (!summary) notFound();
    const year = new Date(summary.date).getUTCFullYear();
    if (year !== 2025 && year !== 2026) notFound();
    const archive = await readPackJson<MatchRecords>(manifest, `features/match_records_${year}.json`);
    game = archive.games[gameId];
  }
  if (!game) notFound();
  const publishedGame = hasPublishedDraftAuthority(manifest)
    ? game
    : { ...game, draft_pool: undefined, draft_contribution: undefined };
  return (
    <MatchRatingProfile
      game={publishedGame}
      championImages={profiles.champion_images}
      releaseId={manifest.pack_id}
      draftScore={await promotedDraftResult(manifest, gameId)}
    />
  );
}
