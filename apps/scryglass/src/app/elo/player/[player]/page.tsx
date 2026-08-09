import { notFound } from "next/navigation";
import { PlayerEloDetail } from "@/components/PlayerEloDetail";
import type {
  PlayerRating,
  PlayerRecord,
  TeamRating,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ player: string }> };

export default async function PlayerEloPage({ params }: Props) {
  const { player: raw } = await params;
  const name = decodeURIComponent(raw);
  const man = await readPackManifest();
  const players = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  let playerRecords: Record<string, PlayerRecord> = {};
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }

  const player = players.find((p) => p.player.toLowerCase() === name.toLowerCase());
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const currentTeam = rec?.current_team ?? player.last_team;
  const team = currentTeam
    ? teams.find((t) => t.team.toLowerCase() === currentTeam.toLowerCase())
    : null;

  // Peers: same primary league when available, else same last_team pool / top sample
  const primary = rec?.primary;
  let peers = players.filter((p) => (p.n_maps ?? 0) >= 20);
  if (primary) {
    const narrowed = peers.filter((p) => playerRecords[p.player]?.primary === primary);
    if (narrowed.length >= 10) peers = narrowed;
  } else if (currentTeam) {
    peers = peers.filter((p) => (playerRecords[p.player]?.current_team ?? p.last_team) === currentTeam);
  }
  const intlPeers = players.filter(
    (p) => (p.n_maps ?? 0) >= 20 && Boolean(playerRecords[p.player]?.intl),
  );

  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <PlayerEloDetail
      player={player}
      record={rec}
      team={team}
      peers={peers}
      intlPeers={intlPeers}
      baseUrl={baseUrl}
      years={man.filters.years}
      manifest={man}
    />
  );
}
