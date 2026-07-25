import { promises as fs } from "fs";
import path from "path";
import { HeadToHead } from "@/components/HeadToHead";
import type { PackManifest } from "@/lib/pack";

export default async function HeadToHeadPage() {
  const man = JSON.parse(
    await fs.readFile(path.join(process.cwd(), "public", "packs", "manifest.json"), "utf8"),
  ) as PackManifest;
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Warehouse · Meetings</p>
        <h1 className="font-display mt-2 text-3xl">Head-to-head</h1>
        <p className="lede">
          Find meetings between two teams, then open a ticker-style post-game board (KDA, gold, CS,
          bans, objectives).
        </p>
      </header>
      <HeadToHead baseUrl={baseUrl} years={man.filters.years} />
    </div>
  );
}
