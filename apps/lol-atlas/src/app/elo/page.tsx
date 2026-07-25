import { promises as fs } from "fs";
import path from "path";
import { EloLadders } from "@/components/EloLadders";
import type { EloCalibration, PackManifest, PlayerRating, TeamRating } from "@/lib/pack";
import { packUrl } from "@/lib/pack";

async function loadJson<T>(filePath: string): Promise<T> {
  return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
}

export default async function EloPage() {
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
  let calibration: EloCalibration | null = null;
  try {
    calibration = await loadJson<EloCalibration>(
      path.join(base, "models", "elo_wr_calibration.json"),
    );
  } catch {
    calibration = null;
  }
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Ratings · Dual Elo</p>
        <h1 className="font-display mt-2 text-3xl">Dual Elo ladders</h1>
        <p className="lede">
          Click a team for roster ratings and top champions. Adj. rating is the default sort so
          high-uncertainty regional spikes don&apos;t outrank settled major orgs.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Pack</strong> {man.pack_id}
          </span>
          <span>
            <strong>Teams</strong>{" "}
            <a className="row-link" href={packUrl(man, "features/ratings_snapshot.json")}>
              JSON
            </a>
          </span>
        </div>
      </header>
      <EloLadders
        teams={teams}
        players={players}
        calibration={calibration}
        baseUrl={baseUrl}
        years={man.filters.years}
      />
    </div>
  );
}
