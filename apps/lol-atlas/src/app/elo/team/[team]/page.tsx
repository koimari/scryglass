import { promises as fs } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import { TeamEloDetail } from "@/components/TeamEloDetail";
import type { PackManifest, PlayerRating, TeamRating, TeamRecord } from "@/lib/pack";

type Props = { params: Promise<{ team: string }> };

async function loadJson<T>(filePath: string): Promise<T> {
  return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
}

export default async function TeamEloPage({ params }: Props) {
  const { team: raw } = await params;
  const teamName = decodeURIComponent(raw);
  const man = await loadJson<PackManifest>(
    path.join(process.cwd(), "public", "packs", "manifest.json"),
  );
  const base = path.join(process.cwd(), "public", "packs", man.pack_id);
  const teams = await loadJson<TeamRating[]>(
    path.join(base, "features", "ratings_snapshot.json"),
  );
  const players = await loadJson<PlayerRating[]>(
    path.join(base, "features", "player_ratings_snapshot.json"),
  );
  let teamRecords: Record<string, TeamRecord> = {};
  try {
    teamRecords = await loadJson(path.join(base, "features", "team_records.json"));
  } catch {
    teamRecords = {};
  }
  const team = teams.find((t) => t.team.toLowerCase() === teamName.toLowerCase());
  if (!team) notFound();

  const roster = players.filter(
    (p) => (p.last_team || "").toLowerCase() === team.team.toLowerCase(),
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
    />
  );
}
