import { Suspense } from "react";
import { BrowseMatches } from "@/components/BrowseMaps";
import { packUpdatedLabel } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

export default async function BrowsePage() {
  const man = await readPackManifest();
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Matches · Series</p>
        <h1 className="font-display mt-2 text-3xl">Matches</h1>
        <p className="lede">
          Find a completed series, open a board. Bo3 and Bo5 keep their games together. Recent GRID
          rows fill the freshness gap until Oracle&apos;s Elixir catches up; Dual Elo favorite hit rate
          for the year sits at the top.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Last updated</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Pack</strong> {man.pack_id}
          </span>
        </div>
      </header>
      <Suspense fallback={<div className="skeleton-block" />}>
        <BrowseMatches baseUrl={baseUrl} years={man.filters.years} />
      </Suspense>
    </div>
  );
}
