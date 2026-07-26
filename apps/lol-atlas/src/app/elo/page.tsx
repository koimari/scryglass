import { Suspense } from "react";
import { EloLadders } from "@/components/EloLadders";
import type {
  PlayerRating,
  PlayerRecord,
  TeamRating,
  TeamRecord,
} from "@/lib/pack";
import { packUpdatedLabel, packUrl, softMu } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

// Ratings are refreshed independently of the app deployment and are served
// from the current Blob pack at request time.
export const dynamic = "force-dynamic";

function thinPlayers(players: PlayerRating[]): PlayerRating[] {
  return players
    .filter((p) => (p.n_maps ?? 0) >= 5)
    .map((p) => ({
      player: p.player,
      mu_total: p.mu_total,
      mu_regional: p.mu_regional,
      mu_meta: p.mu_meta,
      sigma: p.sigma,
      n_maps: p.n_maps,
      last_team: p.last_team,
    }))
    .sort(
      (a, b) =>
        softMu(b.mu_total, b.sigma, 28) - softMu(a.mu_total, a.sigma, 28),
    );
}

export default async function EloPage() {
  const man = await readPackManifest();
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const playersRaw = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = thinPlayers(playersRaw);

  let teamRecords: Record<string, TeamRecord> = {};
  let playerRecords: Record<string, PlayerRecord> = {};
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

  const leagueSet = new Set<string>();
  for (const rec of Object.values(teamRecords)) {
    for (const lg of rec.leagues || []) leagueSet.add(lg);
  }
  const availableLeagues = [...leagueSet].sort();

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Ratings · Dual Elo</p>
        <h1 className="font-display mt-2 text-3xl">Dual Elo ladders</h1>
        <p className="lede">
          Open a team for the roster, or a player for their board. Adjusted rating is the default
          sort — it accounts for how much evidence supports the number.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Last updated</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Pack</strong> {man.pack_id}
          </span>
          <span>
            <strong>Orgs</strong> {teams.length}
          </span>
          <span>
            <a className="row-link" href={packUrl(man, "features/ratings_snapshot.json")}>
              Snapshot JSON
            </a>
          </span>
        </div>
      </header>
      <Suspense fallback={<div className="skeleton-block" aria-hidden />}>
        <EloLadders
          teams={teams}
          players={players}
          teamRecords={teamRecords}
          playerRecords={playerRecords}
          availableLeagues={availableLeagues}
        />
      </Suspense>
    </div>
  );
}
