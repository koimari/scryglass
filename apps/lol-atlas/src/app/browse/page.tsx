import { Suspense } from "react";
import { BrowseMatches } from "@/components/BrowseMaps";
import { packDataThroughLabel, packUpdatedLabel } from "@/lib/pack";
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
          Find a completed map and open its board. Canonical series IDs and verified scheduled
          formats are used when the pack provides them; legacy grouping is labeled unverified.
          Canonical inclusion and optional GRID detail enrichment follow the source roles declared
          by this immutable pack. The favorite
          hit rate above is a threshold diagnostic, not a probability-quality score.
        </p>
        <p className="lede">
          Counts on this page describe the public map ledger returned by the explorer query. They
          are not claimed to equal a rating model&apos;s training population unless the pack publishes
          that reconciliation.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Pack published</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Data through</strong> {packDataThroughLabel(man)}
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
