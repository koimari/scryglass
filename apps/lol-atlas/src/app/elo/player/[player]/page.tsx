import { promises as fs } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import { PlayerEloDetail } from "@/components/PlayerEloDetail";
import type {
  PackManifest,
  PlayerRating,
  PlayerRecord,
  TeamRating,
} from "@/lib/pack";

type Props = { params: Promise<{ player: string }> };

async function loadJson<T>(filePath: string): Promise<T> {
  return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
}

export default async function PlayerEloPage({ params }: Props) {
  const { player: raw } = await params;
  const name = decodeURIComponent(raw);
  const man = await loadJson<PackManifest>(
    path.join(process.cwd(), "public", "packs", "manifest.json"),
  );
  const base = path.join(process.cwd(), "public", "packs", man.pack_id);
  const players = await loadJson<PlayerRating[]>(
    path.join(base, "features", "player_ratings_snapshot.json"),
  );
  const teams = await loadJson<TeamRating[]>(
    path.join(base, "features", "ratings_snapshot.json"),
  );
  let playerRecords: Record<string, PlayerRecord> = {};
  try {
    playerRecords = await loadJson(path.join(base, "features", "player_records.json"));
  } catch {
    playerRecords = {};
  }

  const player = players.find((p) => p.player.toLowerCase() === name.toLowerCase());
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const team = player.last_team
    ? teams.find((t) => t.team.toLowerCase() === player.last_team!.toLowerCase())
    : null;

  // Peers: same primary league when available, else same last_team pool / top sample
  const primary = rec?.primary;
  let peers = players.filter((p) => (p.n_maps ?? 0) >= 20);
  if (primary) {
    const narrowed = peers.filter((p) => playerRecords[p.player]?.primary === primary);
    if (narrowed.length >= 10) peers = narrowed;
  } else if (player.last_team) {
    peers = peers.filter((p) => p.last_team === player.last_team);
  }

  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <PlayerEloDetail
      player={player}
      record={rec}
      team={team}
      peers={peers}
      baseUrl={baseUrl}
      years={man.filters.years}
      manifest={man}
    />
  );
}
