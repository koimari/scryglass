import { notFound } from "next/navigation";
import { TeamEloDetail } from "@/components/TeamEloDetail";
import type {
  PlayerRating,
  PlayerRatingsMeta,
  PlayerRecord,
  TeamRating,
  TeamRatingsMeta,
  TeamRecord,
} from "@/lib/pack";
import {
  currentMembershipContext,
  playerOutcomeOrderingVerified,
  verifiedPlayerAffiliation,
  verifiedTeamAffiliation,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ team: string }> };

export default async function TeamEloPage({ params }: Props) {
  const { team: raw } = await params;
  const teamName = decodeURIComponent(raw);
  const man = await readPackManifest();
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const players = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  let teamRecords: Record<string, TeamRecord> = {};
  let playerRecords: Record<string, PlayerRecord> = {};
  let teamRatingsMeta: TeamRatingsMeta | null = null;
  let playerRatingsMeta: PlayerRatingsMeta | null = null;
  try {
    teamRecords = await readPackJson(man, "features/team_records.json");
  } catch {
    teamRecords = {};
  }
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }
  try {
    teamRatingsMeta = await readPackJson<TeamRatingsMeta>(man, "features/ratings_meta.json");
  } catch {
    teamRatingsMeta = null;
  }
  try {
    playerRatingsMeta = await readPackJson<PlayerRatingsMeta>(
      man,
      "features/player_ratings_meta.json",
    );
  } catch {
    playerRatingsMeta = null;
  }
  const team = teams.find((t) => t.team.toLowerCase() === teamName.toLowerCase());
  if (!team) notFound();

  const membershipContext = currentMembershipContext(man);
  const playerOrderingVerified = playerOutcomeOrderingVerified(
    playerRatingsMeta,
    players,
  );
  const teamAffiliation = verifiedTeamAffiliation(
    teamRecords[team.team],
    membershipContext,
  );
  const roster = players.filter(
    (player) => {
      const affiliation = verifiedPlayerAffiliation(
        playerRecords[player.player],
        membershipContext,
      );
      return (
        affiliation?.team.toLowerCase() === team.team.toLowerCase() &&
        affiliation.league === teamAffiliation?.league &&
        affiliation.tournament === teamAffiliation?.tournament
      );
    },
  );
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <TeamEloDetail
      team={team}
      roster={roster}
      record={teamRecords[team.team]}
      baseUrl={baseUrl}
      years={man.filters.years}
      manifest={man}
      membershipContext={membershipContext}
      teamAffiliation={teamAffiliation}
      teamRatingsMeta={teamRatingsMeta}
      playerRatingsMeta={playerRatingsMeta}
      playerOrderingVerified={playerOrderingVerified}
    />
  );
}
